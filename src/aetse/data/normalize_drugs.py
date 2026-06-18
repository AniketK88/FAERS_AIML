"""Batch drug name normalization and seriousness derivation.

This module handles:
- Deriving faers_cases.serious from OUTC files (DE/LT/HO/DS/CA/RI = serious)
- Loading RxNorm lookup and running fuzzy matching on all drug names
- Batch updating faers_drugs.drugname_norm + rxnorm_rxcui in DuckDB

Usage:
    python -m aetse.data.normalize_drugs

All file I/O uses Polars (not Pandas) per project rules.
DuckDB queries use parameterized values — no f-string SQL with user data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

from aetse.config.settings import settings
from aetse.data.ingest_rxnorm import run_rxnorm_ingestion, RXNORM_LOOKUP_PATH
from aetse.pipeline.agents.rxnorm_matcher import RxNormMatcher
from aetse.utils.db import get_duckdb_connection
from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Seriousness derivation from OUTC table
# ---------------------------------------------------------------------------

# Outcome codes that indicate a serious case
SERIOUS_OUTCOME_CODES: set[str] = {"DE", "LT", "HO", "DS", "CA", "RI"}


def _read_outc_file(path: Path) -> pl.DataFrame:
    """Read a pipe-delimited OUTC file with Polars.

    Args:
        path: Path to OUTC*.txt file.

    Returns:
        Polars DataFrame with columns: caseid, outc_cod
    """
    logger.info(f"Reading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    try:
        df = pl.read_csv(
            path,
            separator="$",
            has_header=True,
            infer_schema_length=10000,
            encoding="utf8",
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
    except Exception:
        logger.warning(f"UTF-8 failed for {path.name}, retrying with utf8-lossy")
        df = pl.read_csv(
            path,
            separator="$",
            has_header=True,
            infer_schema_length=10000,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

    # Normalize column names
    df = df.rename({col: col.lower().strip() for col in df.columns})
    logger.info(f"  → {df.height:,} rows, columns: {df.columns}")
    return df


def derive_seriousness() -> int:
    """Derive the serious flag from OUTC files and update faers_cases.

    Logic:
    - Load OUTC files from all quarters
    - For each caseid, if ANY outc_cod is in {DE, LT, HO, DS, CA, RI} → serious = True
    - All other cases → serious = False
    - UPDATE faers_cases SET serious = derived_value

    Returns:
        Number of cases updated.
    """
    logger.info("=" * 60)
    logger.info("Deriving seriousness from OUTC files")
    logger.info("=" * 60)

    raw_dir = settings.data.faers_raw_dir
    outc_frames: list[pl.DataFrame] = []

    # Discover OUTC files across all quarter directories
    for quarter_dir in sorted(raw_dir.iterdir()):
        if not quarter_dir.is_dir() or not quarter_dir.name.startswith("faers_ascii_"):
            continue

        ascii_dir = quarter_dir / "ascii"
        if not ascii_dir.exists():
            ascii_dir = quarter_dir

        for candidate in ascii_dir.iterdir():
            if candidate.is_file() and candidate.name.upper().startswith("OUTC"):
                if candidate.suffix.lower() == ".txt":
                    df = _read_outc_file(candidate)
                    outc_frames.append(df)
                    break

    if not outc_frames:
        logger.warning("No OUTC files found — serious column will remain NULL")
        return 0

    # Concatenate all quarters
    outc = pl.concat(outc_frames, how="diagonal_relaxed")
    logger.info(f"Combined OUTC: {outc.height:,} rows")

    # Ensure caseid and outc_cod are string
    outc = outc.with_columns([
        pl.col("caseid").cast(pl.Utf8),
        pl.col("outc_cod").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
    ])

    # Determine which caseids are serious:
    # A case is serious if ANY of its outcomes is in SERIOUS_OUTCOME_CODES
    serious_cases = (
        outc.filter(pl.col("outc_cod").is_in(list(SERIOUS_OUTCOME_CODES)))
        .select("caseid")
        .unique()
        .with_columns(pl.lit(True).alias("is_serious"))
    )

    # Get all caseids from OUTC (including non-serious)
    all_outc_cases = outc.select("caseid").unique()

    # Join: cases in OUTC but NOT in serious_cases are non-serious
    cases_with_seriousness = all_outc_cases.join(
        serious_cases, on="caseid", how="left"
    ).with_columns(
        pl.col("is_serious").fill_null(False)
    )

    logger.info(
        f"  Serious cases: {cases_with_seriousness.filter(pl.col('is_serious')).height:,}"
    )
    logger.info(
        f"  Non-serious cases: {cases_with_seriousness.filter(~pl.col('is_serious')).height:,}"
    )

    # Update DuckDB: set serious = True for serious cases
    with get_duckdb_connection() as conn:
        # First set all cases that appear in OUTC to False (non-serious baseline)
        conn.register("outc_seriousness", cases_with_seriousness.to_arrow())

        # Update in one batch using a staged approach
        conn.execute("""
            UPDATE faers_cases
            SET serious = sub.is_serious
            FROM (SELECT caseid, is_serious FROM outc_seriousness) sub
            WHERE faers_cases.caseid = sub.caseid
        """)

        # Verify
        result = conn.execute("""
            SELECT
                serious,
                COUNT(*) as n
            FROM faers_cases
            GROUP BY serious
            ORDER BY serious
        """).fetchall()

    logger.info("Seriousness distribution after update:")
    total_updated = 0
    for row in result:
        label = "NULL" if row[0] is None else ("Serious" if row[0] else "Non-serious")
        logger.info(f"  {label}: {row[1]:,}")
        if row[0] is not None:
            total_updated += row[1]

    logger.info(f"  → {total_updated:,} cases updated with seriousness flag")
    return total_updated


# ---------------------------------------------------------------------------
# Drug name normalization
# ---------------------------------------------------------------------------

def normalize_drug_names() -> dict[str, int]:
    """Normalize all drug names in faers_drugs.

    Strategy:
    1. Try RxTerms download + fuzzy matching (RxNormMatcher)
    2. If download fails (no network), fall back to curated keyword lookup
    3. Batch UPDATE faers_drugs with results

    Returns:
        Dict with stats: {total, matched, unmatched, match_rate_pct}
    """
    logger.info("=" * 60)
    logger.info("Drug Name Normalization")
    logger.info("=" * 60)

    # Step 1: Get unique drug names from faers_drugs
    with get_duckdb_connection(read_only=True) as conn:
        unique_drugs_df = conn.execute("""
            SELECT DISTINCT drugname_raw
            FROM faers_drugs
            WHERE drugname_raw IS NOT NULL
            ORDER BY drugname_raw
        """).pl()

    unique_names = unique_drugs_df["drugname_raw"].to_list()
    logger.info(f"Unique drug names to normalize: {len(unique_names):,}")

    # Step 2: Try RxTerms, fall back to curated lookup
    match_results: list[dict] | None = None

    try:
        lookup_path = run_rxnorm_ingestion()
        matcher = RxNormMatcher(lookup_path, threshold=80.0)
        match_results = matcher.batch_normalize(unique_names)
        logger.info("Used RxTerms fuzzy matching")
    except Exception as e:
        logger.warning(f"RxTerms not available ({e}), using curated keyword lookup")

        from aetse.data.curated_drug_lookup import (
            build_curated_lookup,
            batch_normalize_drugs,
        )
        build_curated_lookup()
        match_results = batch_normalize_drugs(unique_names)
        logger.info("Used curated keyword-based matching")

    # Step 3: Build mapping and update DuckDB
    matched_count = sum(1 for r in match_results if r["drugname_norm"] is not None)
    unmatched_count = len(match_results) - matched_count

    logger.info("Updating faers_drugs in DuckDB...")

    mapping_records = [
        {
            "drugname_raw": r["drugname_raw"],
            "drugname_norm": r["drugname_norm"],
            "rxnorm_rxcui": r["rxnorm_rxcui"],
        }
        for r in match_results
        if r["drugname_norm"] is not None
    ]

    if mapping_records:
        mapping_df = pl.DataFrame(mapping_records)

        with get_duckdb_connection() as conn:
            conn.register("drug_mappings", mapping_df.to_arrow())

            conn.execute("""
                UPDATE faers_drugs
                SET
                    drugname_norm = dm.drugname_norm,
                    rxnorm_rxcui = dm.rxnorm_rxcui
                FROM (SELECT drugname_raw, drugname_norm, rxnorm_rxcui FROM drug_mappings) dm
                WHERE faers_drugs.drugname_raw = dm.drugname_raw
            """)

            result = conn.execute("""
                SELECT
                    COUNT(*) as total_drugs,
                    COUNT(drugname_norm) as normalized,
                    ROUND(COUNT(drugname_norm) * 100.0 / COUNT(*), 1) as pct_normalized
                FROM faers_drugs
            """).fetchone()

        logger.info(f"  Total drug records: {result[0]:,}")
        logger.info(f"  Normalized: {result[1]:,}")
        logger.info(f"  Normalization rate: {result[2]}%")
    else:
        logger.warning("No drug names matched — faers_drugs.drugname_norm unchanged")

    # Log top mappings
    _log_top_mappings(match_results)
    _log_unmatched(match_results)

    stats = {
        "total": len(unique_names),
        "matched": matched_count,
        "unmatched": unmatched_count,
        "match_rate_pct": round(matched_count * 100 / max(len(unique_names), 1), 1),
    }
    logger.info(f"Normalization stats: {stats}")
    return stats


def _log_top_mappings(results: list[dict]) -> None:
    """Log the top drug name → normalized name mappings."""
    matched = [r for r in results if r["drugname_norm"] is not None]
    matched.sort(key=lambda x: x["drugname_raw"])

    logger.info("Top 15 drug name mappings:")
    for r in matched[:15]:
        logger.info(
            f"  {r['drugname_raw']:<45} → {r['drugname_norm']:<25} "
            f"(score={r['match_score']}, rxcui={r['rxnorm_rxcui']})"
        )


def _log_unmatched(results: list[dict]) -> None:
    """Log drug names that could not be matched."""
    unmatched = [r for r in results if r["drugname_norm"] is None]

    if unmatched:
        logger.info(f"Unmatched drug names ({len(unmatched)}):")
        for r in unmatched[:10]:
            logger.info(f"  ✗ {r['drugname_raw']:<45} (best score={r['match_score']})")
        if len(unmatched) > 10:
            logger.info(f"  ... and {len(unmatched) - 10} more")


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------

def run_verification_queries() -> None:
    """Run Day 2 verification queries and log results."""
    with get_duckdb_connection(read_only=True) as conn:
        # 1. Seriousness distribution
        logger.info("=" * 60)
        logger.info("Day 2 Verification")
        logger.info("=" * 60)

        logger.info("\n1. Seriousness distribution:")
        rows = conn.execute("""
            SELECT serious, COUNT(*) as n
            FROM faers_cases
            GROUP BY serious
            ORDER BY serious
        """).fetchall()
        for row in rows:
            label = "NULL" if row[0] is None else ("Serious" if row[0] else "Non-serious")
            logger.info(f"   {label}: {row[1]:,}")

        # 2. Normalization coverage
        logger.info("\n2. Normalization coverage:")
        result = conn.execute("""
            SELECT
                COUNT(*) as total_drugs,
                COUNT(drugname_norm) as normalized,
                ROUND(COUNT(drugname_norm) * 100.0 / COUNT(*), 1) as pct_normalized
            FROM faers_drugs
        """).fetchone()
        logger.info(f"   Total drug records: {result[0]:,}")
        logger.info(f"   Normalized: {result[1]:,}")
        logger.info(f"   Rate: {result[2]}%")

        # 3. Salt variant collapse
        logger.info("\n3. Salt variant collapse verification:")
        rows = conn.execute("""
            SELECT drugname_raw, drugname_norm, COUNT(*) as n
            FROM faers_drugs
            WHERE drugname_norm IN ('atorvastatin', 'metoprolol', 'metformin')
            GROUP BY 1, 2
            ORDER BY 3 DESC
            LIMIT 20
        """).fetchall()
        for row in rows:
            logger.info(f"   {row[0]:<45} → {row[1]:<20} ({row[2]:,} records)")

        # 4. Top normalized drugs
        logger.info("\n4. Top 15 normalized drugs:")
        rows = conn.execute("""
            SELECT drugname_norm, COUNT(*) as n
            FROM faers_drugs
            WHERE drugname_norm IS NOT NULL
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 15
        """).fetchall()
        for row in rows:
            logger.info(f"   {row[0]:<30} {row[1]:>10,}")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_day2_pipeline() -> None:
    """Run the complete Day 2 pipeline.

    Steps:
    1. Derive seriousness from OUTC files → update faers_cases.serious
    2. Download/load RxNorm RxTerms
    3. Normalize drug names → update faers_drugs.drugname_norm + rxnorm_rxcui
    4. Run verification queries
    """
    start_time = time.time()

    # Step 1: Seriousness derivation (independent, runs first)
    derive_seriousness()

    # Step 2+3: RxNorm normalization
    normalize_drug_names()

    # Step 4: Verification
    run_verification_queries()

    elapsed = time.time() - start_time
    logger.info(f"Day 2 pipeline completed in {elapsed:.1f}s")
    logger.info("✅ Day 2 — RxNorm Normalization + Seriousness Derivation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_day2_pipeline()
