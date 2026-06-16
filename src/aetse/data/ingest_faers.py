"""FAERS quarterly ASCII data ingestion and DuckDB normalization.

This module handles:
- Loading FAERS ASCII text files (DEMO, DRUG, REAC, OUTC)
- Deduplication by (caseid, caseversion) — keeping latest version
- Age normalization to years using age_cod mapping
- Schema creation and data insertion into DuckDB
- Filtering to target drug classes (NSAIDs, cardiovascular)
- Checkpoint/resume support for interrupted ingestion

Usage:
    python -m aetse.data.ingest_faers

All file I/O uses Polars (not Pandas) per project rules.
DuckDB queries are always parameterized — no f-string SQL with user data.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from aetse.config.settings import settings
from aetse.utils.db import get_duckdb_connection, init_schema, get_table_counts, run_validation_query
from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target drugs for filtering (case-insensitive match against prod_ai)
TARGET_DRUGS: list[str] = [
    # NSAIDs
    "ibuprofen",
    "naproxen",
    "aspirin",
    "diclofenac",
    "celecoxib",
    "rofecoxib",
    # Cardiovascular
    "amlodipine",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "metoprolol",
]

# Age code → multiplier to convert to years
AGE_COD_TO_YEARS: dict[str, float] = {
    "YR": 1.0,        # Years
    "DEC": 10.0,       # Decades → multiply by 10
    "MON": 1.0 / 12,   # Months → divide by 12
    "WK": 1.0 / 52,    # Weeks → divide by 52
    "DY": 1.0 / 365,   # Days → divide by 365
    "HR": 1.0 / 8760,  # Hours → divide by 8760
}

# FAERS file types we need
FILE_TYPES: list[str] = ["DEMO", "DRUG", "REAC", "OUTC"]

# Checkpoint path
CHECKPOINT_PATH: Path = settings.project_root / "data" / "processed" / "faers_checkpoint.json"


# ---------------------------------------------------------------------------
# Step A — File Discovery
# ---------------------------------------------------------------------------

def discover_faers_files(raw_dir: Path) -> dict[str, dict[str, Path]]:
    """Scan data/raw/faers/ for extracted quarter directories.

    Looks for directories matching the pattern faers_ascii_YYYYQN/ascii/
    and detects which DEMO/DRUG/REAC/OUTC files exist per quarter.

    Args:
        raw_dir: Path to data/raw/faers/

    Returns:
        Dict mapping quarter label (e.g. "2024Q4") to dict of
        file_type -> Path, e.g. {"DEMO": Path("...DEMO24Q4.txt"), ...}
    """
    quarters: dict[str, dict[str, Path]] = {}

    if not raw_dir.exists():
        logger.warning(f"FAERS raw directory does not exist: {raw_dir}")
        return quarters

    # Look for directories matching faers_ascii_*
    for quarter_dir in sorted(raw_dir.iterdir()):
        if not quarter_dir.is_dir():
            continue
        if not quarter_dir.name.startswith("faers_ascii_"):
            continue

        # Extract quarter label from directory name (e.g. faers_ascii_2024Q4 → 2024Q4)
        quarter_label = quarter_dir.name.replace("faers_ascii_", "")

        # ASCII files are typically in an /ascii/ subdirectory
        ascii_dir = quarter_dir / "ascii"
        if not ascii_dir.exists():
            # Some extractions put files directly in the quarter directory
            ascii_dir = quarter_dir

        files: dict[str, Path] = {}
        for file_type in FILE_TYPES:
            # Try common naming patterns:
            # DEMO24Q4.txt, DEMO24Q4.TXT, demo24q4.txt, etc.
            for candidate in ascii_dir.iterdir():
                if candidate.is_file() and candidate.name.upper().startswith(file_type):
                    if candidate.suffix.lower() == ".txt":
                        files[file_type] = candidate
                        break

        if files:
            quarters[quarter_label] = files
            logger.info(
                f"Discovered quarter {quarter_label}: "
                f"{', '.join(f'{k}={v.name}' for k, v in sorted(files.items()))}"
            )
        else:
            logger.warning(f"No FAERS files found in {ascii_dir}")

    if not quarters:
        logger.error(
            f"No FAERS quarter directories found in {raw_dir}. "
            f"Expected directories like: faers_ascii_2024Q4/ascii/"
        )

    return quarters


def _read_faers_file(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Read a pipe-delimited FAERS ASCII file with Polars.

    Handles encoding fallback: tries UTF-8 first, then latin-1.

    Args:
        path: Path to the .txt file
        columns: Optional list of columns to select (case-insensitive matching)

    Returns:
        Polars DataFrame with the file contents
    """
    logger.info(f"Reading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    # FAERS files use '$' as separator with a header row
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
        logger.warning(f"UTF-8 failed for {path.name}, retrying with latin-1")
        df = pl.read_csv(
            path,
            separator="$",
            has_header=True,
            infer_schema_length=10000,
            encoding="utf8-lossy",
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

    # Normalize column names to lowercase for consistent access
    df = df.rename({col: col.lower().strip() for col in df.columns})

    logger.info(f"  → {df.height:,} rows, {df.width} columns: {df.columns}")

    if columns:
        # Select only requested columns (case-insensitive)
        available = {c.lower() for c in df.columns}
        selected = [c for c in columns if c.lower() in available]
        missing = [c for c in columns if c.lower() not in available]
        if missing:
            logger.warning(f"  Columns not found in {path.name}: {missing}")
        df = df.select(selected)

    return df


# ---------------------------------------------------------------------------
# Step B — DEMO Table Loading
# ---------------------------------------------------------------------------

def load_demo_tables(quarters: dict[str, dict[str, Path]]) -> pl.DataFrame:
    """Load and process DEMO files from all quarters.

    Processing steps:
    1. Read all DEMO files and concatenate
    2. Deduplicate: group by caseid, keep row with MAX(caseversion)
    3. Normalize age to years using age_cod mapping
    4. Parse receivedate YYYYMMDD → Date type
    5. Map serious field ('1' → True, '2' → False)

    Args:
        quarters: Dict from discover_faers_files()

    Returns:
        Deduplicated, normalized Polars DataFrame ready for DuckDB insertion
    """
    demo_frames: list[pl.DataFrame] = []

    for quarter, files in quarters.items():
        if "DEMO" not in files:
            logger.warning(f"Quarter {quarter}: no DEMO file found, skipping")
            continue

        df = _read_faers_file(files["DEMO"])
        demo_frames.append(df)

    if not demo_frames:
        logger.error("No DEMO files found across any quarter")
        return pl.DataFrame()

    # Concatenate all quarters
    demo = pl.concat(demo_frames, how="diagonal_relaxed")
    logger.info(f"Combined DEMO: {demo.height:,} rows before deduplication")

    # --- Deduplication: keep MAX(caseversion) per caseid ---
    # Cast caseid and caseversion to correct types
    demo = demo.with_columns([
        pl.col("caseid").cast(pl.Utf8).alias("caseid"),
        pl.col("caseversion").cast(pl.Int64, strict=False).alias("caseversion"),
    ])

    # Drop rows with null caseid
    demo = demo.filter(pl.col("caseid").is_not_null() & (pl.col("caseid") != ""))

    # Sort by caseversion descending and take first per caseid
    demo = (
        demo.sort("caseversion", descending=True)
        .unique(subset=["caseid"], keep="first")
    )
    logger.info(f"After deduplication: {demo.height:,} rows (unique caseids)")

    # --- Age normalization ---
    # Ensure 'age' and 'age_cod' columns exist
    if "age" in demo.columns and "age_cod" in demo.columns:
        demo = demo.with_columns(
            pl.col("age").cast(pl.Float64, strict=False).alias("age_numeric")
        )

        # Build age_years: age_numeric * multiplier based on age_cod
        demo = demo.with_columns(
            pl.when(pl.col("age_cod").str.to_uppercase() == "YR")
            .then(pl.col("age_numeric") * 1.0)
            .when(pl.col("age_cod").str.to_uppercase() == "DEC")
            .then(pl.col("age_numeric") * 10.0)
            .when(pl.col("age_cod").str.to_uppercase() == "MON")
            .then(pl.col("age_numeric") / 12.0)
            .when(pl.col("age_cod").str.to_uppercase() == "WK")
            .then(pl.col("age_numeric") / 52.0)
            .when(pl.col("age_cod").str.to_uppercase() == "DY")
            .then(pl.col("age_numeric") / 365.0)
            .when(pl.col("age_cod").str.to_uppercase() == "HR")
            .then(pl.col("age_numeric") / 8760.0)
            .otherwise(pl.lit(None))
            .alias("age_years")
        )

        # Filter out unreasonable ages (negative or > 150 years)
        demo = demo.with_columns(
            pl.when((pl.col("age_years") >= 0) & (pl.col("age_years") <= 150))
            .then(pl.col("age_years"))
            .otherwise(pl.lit(None))
            .alias("age_years")
        )
    else:
        logger.warning("Missing 'age' or 'age_cod' column — age_years will be NULL")
        demo = demo.with_columns(pl.lit(None).cast(pl.Float64).alias("age_years"))

    # --- Parse receivedate (YYYYMMDD string → Date) ---
    if "event_dt" in demo.columns:
        # Some FAERS files use event_dt instead of receivedate
        date_col = "event_dt"
    elif "receivedate" in demo.columns:
        date_col = "receivedate"
    else:
        date_col = None
        logger.warning("No date column found (receivedate/event_dt)")

    if date_col:
        demo = demo.with_columns(
            pl.col(date_col)
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("report_date")
        )
    else:
        demo = demo.with_columns(pl.lit(None).cast(pl.Date).alias("report_date"))

    # --- Map serious field: '1' → True, '2' → False ---
    if "serious" in demo.columns:
        demo = demo.with_columns(
            pl.col("serious")
            .cast(pl.Utf8)
            .map_elements(
                lambda v: True if str(v).strip() == "1" else (False if str(v).strip() == "2" else None),
                return_dtype=pl.Boolean,
            )
            .alias("serious_bool")
        )
    else:
        demo = demo.with_columns(pl.lit(None).cast(pl.Boolean).alias("serious_bool"))

    # --- Country ---
    country_col = "reporter_country" if "reporter_country" in demo.columns else (
        "occr_country" if "occr_country" in demo.columns else None
    )

    # --- Sex ---
    if "sex" in demo.columns:
        demo = demo.with_columns(
            pl.col("sex").cast(pl.Utf8).str.to_uppercase().str.strip_chars().alias("sex_clean")
        )
    else:
        demo = demo.with_columns(pl.lit(None).cast(pl.Utf8).alias("sex_clean"))

    # --- Build final output DataFrame ---
    result = demo.select([
        pl.col("caseid"),
        pl.col("caseversion"),
        pl.col("age_years"),
        pl.col("sex_clean").alias("sex"),
        pl.col("report_date"),
        pl.col(country_col).cast(pl.Utf8).alias("country") if country_col else pl.lit(None).cast(pl.Utf8).alias("country"),
        pl.col("serious_bool").alias("serious"),
    ])

    logger.info(
        f"DEMO processed: {result.height:,} cases, "
        f"age non-null: {result['age_years'].drop_nulls().height:,}, "
        f"date non-null: {result['report_date'].drop_nulls().height:,}"
    )

    return result


def insert_cases(cases_df: pl.DataFrame) -> int:
    """Insert deduplicated cases into faers_cases DuckDB table.

    Args:
        cases_df: Polars DataFrame from load_demo_tables()

    Returns:
        Number of rows inserted
    """
    if cases_df.is_empty():
        logger.warning("No cases to insert")
        return 0

    with get_duckdb_connection() as conn:
        # Register Polars DataFrame as a DuckDB view and INSERT
        conn.register("cases_staging", cases_df.to_arrow())

        # Use INSERT OR REPLACE to handle potential duplicates from re-runs
        conn.execute("""
            INSERT OR REPLACE INTO faers_cases
                (caseid, caseversion, age_years, sex, report_date, country, serious)
            SELECT caseid, caseversion, age_years, sex, report_date, country, serious
            FROM cases_staging
        """)

        count = conn.execute("SELECT COUNT(*) FROM faers_cases").fetchone()[0]

    logger.info(f"faers_cases: {count:,} rows after insertion")
    return count


# ---------------------------------------------------------------------------
# Step C — DRUG Table Loading
# ---------------------------------------------------------------------------

def load_drug_tables(quarters: dict[str, dict[str, Path]]) -> pl.DataFrame:
    """Load and process DRUG files from all quarters.

    Processing steps:
    1. Read all DRUG files and concatenate
    2. Use prod_ai (active ingredient) preferentially over drugname
    3. Filter to target drugs (case-insensitive match)
    4. Keep only caseids that exist in faers_cases (inner join logic)

    Args:
        quarters: Dict from discover_faers_files()

    Returns:
        Polars DataFrame ready for DuckDB insertion
    """
    drug_frames: list[pl.DataFrame] = []

    for quarter, files in quarters.items():
        if "DRUG" not in files:
            logger.warning(f"Quarter {quarter}: no DRUG file found, skipping")
            continue

        df = _read_faers_file(files["DRUG"])
        drug_frames.append(df)

    if not drug_frames:
        logger.error("No DRUG files found across any quarter")
        return pl.DataFrame()

    drugs = pl.concat(drug_frames, how="diagonal_relaxed")
    logger.info(f"Combined DRUG: {drugs.height:,} rows before filtering")

    # Cast caseid and drug_seq
    drugs = drugs.with_columns([
        pl.col("caseid").cast(pl.Utf8).alias("caseid"),
        pl.col("drug_seq").cast(pl.Int64, strict=False).alias("drug_seq"),
    ])

    # --- Use prod_ai preferentially over drugname ---
    # prod_ai is the active ingredient (cleaner); drugname is free-text
    if "prod_ai" in drugs.columns and "drugname" in drugs.columns:
        drugs = drugs.with_columns(
            pl.when(pl.col("prod_ai").is_not_null() & (pl.col("prod_ai") != ""))
            .then(pl.col("prod_ai"))
            .otherwise(pl.col("drugname"))
            .cast(pl.Utf8)
            .alias("drugname_raw")
        )
    elif "prod_ai" in drugs.columns:
        drugs = drugs.with_columns(pl.col("prod_ai").cast(pl.Utf8).alias("drugname_raw"))
    elif "drugname" in drugs.columns:
        drugs = drugs.with_columns(pl.col("drugname").cast(pl.Utf8).alias("drugname_raw"))
    else:
        logger.error("Neither 'prod_ai' nor 'drugname' column found in DRUG files")
        return pl.DataFrame()

    # --- Role code ---
    if "role_cod" in drugs.columns:
        drugs = drugs.with_columns(pl.col("role_cod").cast(pl.Utf8).alias("role_cod"))
    else:
        drugs = drugs.with_columns(pl.lit(None).cast(pl.Utf8).alias("role_cod"))

    # --- Filter to target drugs (case-insensitive ILIKE match) ---
    # Build a filter expression that matches any target drug
    drug_filter = pl.lit(False)
    for drug_name in TARGET_DRUGS:
        drug_filter = drug_filter | pl.col("drugname_raw").str.to_lowercase().str.contains(
            drug_name.lower(), literal=True
        )

    drugs_filtered = drugs.filter(drug_filter)
    logger.info(
        f"After target drug filter: {drugs_filtered.height:,} rows "
        f"(from {drugs.height:,} total)"
    )

    # --- Keep only caseids present in faers_cases ---
    with get_duckdb_connection(read_only=True) as conn:
        valid_caseids = conn.execute(
            "SELECT caseid FROM faers_cases"
        ).pl()

    drugs_filtered = drugs_filtered.join(
        valid_caseids,
        on="caseid",
        how="inner",
    )
    logger.info(f"After caseid join with faers_cases: {drugs_filtered.height:,} rows")

    # Build final output
    result = drugs_filtered.select([
        pl.col("caseid"),
        pl.col("drug_seq"),
        pl.col("drugname_raw"),
        pl.lit(None).cast(pl.Utf8).alias("drugname_norm"),    # Day 2 — RxNorm
        pl.lit(None).cast(pl.Utf8).alias("rxnorm_rxcui"),     # Day 2 — RxNorm
        pl.col("role_cod"),
    ])

    # Deduplicate on (caseid, drug_seq) — keep first
    result = result.unique(subset=["caseid", "drug_seq"], keep="first")

    return result


def insert_drugs(drugs_df: pl.DataFrame) -> int:
    """Insert drug records into faers_drugs DuckDB table.

    Args:
        drugs_df: Polars DataFrame from load_drug_tables()

    Returns:
        Number of rows inserted
    """
    if drugs_df.is_empty():
        logger.warning("No drug records to insert")
        return 0

    with get_duckdb_connection() as conn:
        conn.register("drugs_staging", drugs_df.to_arrow())

        conn.execute("""
            INSERT OR REPLACE INTO faers_drugs
                (caseid, drug_seq, drugname_raw, drugname_norm, rxnorm_rxcui, role_cod)
            SELECT caseid, drug_seq, drugname_raw, drugname_norm, rxnorm_rxcui, role_cod
            FROM drugs_staging
        """)

        count = conn.execute("SELECT COUNT(*) FROM faers_drugs").fetchone()[0]

    logger.info(f"faers_drugs: {count:,} rows after insertion")
    return count


# ---------------------------------------------------------------------------
# Step D — REAC Table Loading
# ---------------------------------------------------------------------------

def load_reac_tables(quarters: dict[str, dict[str, Path]]) -> pl.DataFrame:
    """Load and process REAC files from all quarters.

    Processing steps:
    1. Read all REAC files and concatenate
    2. Keep only caseids that exist in faers_cases
    3. Generate reac_seq per caseid for primary key

    Args:
        quarters: Dict from discover_faers_files()

    Returns:
        Polars DataFrame ready for DuckDB insertion
    """
    reac_frames: list[pl.DataFrame] = []

    for quarter, files in quarters.items():
        if "REAC" not in files:
            logger.warning(f"Quarter {quarter}: no REAC file found, skipping")
            continue

        df = _read_faers_file(files["REAC"])
        reac_frames.append(df)

    if not reac_frames:
        logger.error("No REAC files found across any quarter")
        return pl.DataFrame()

    reacs = pl.concat(reac_frames, how="diagonal_relaxed")
    logger.info(f"Combined REAC: {reacs.height:,} rows before filtering")

    # Cast caseid
    reacs = reacs.with_columns(
        pl.col("caseid").cast(pl.Utf8).alias("caseid")
    )

    # --- Keep only caseids present in faers_cases ---
    with get_duckdb_connection(read_only=True) as conn:
        valid_caseids = conn.execute("SELECT caseid FROM faers_cases").pl()

    reacs = reacs.join(valid_caseids, on="caseid", how="inner")
    logger.info(f"After caseid join with faers_cases: {reacs.height:,} rows")

    # --- Extract PT (MedDRA Preferred Term) ---
    if "pt" in reacs.columns:
        reacs = reacs.with_columns(
            pl.col("pt").cast(pl.Utf8).str.strip_chars().alias("pt_name")
        )
    else:
        logger.error("Column 'pt' not found in REAC files")
        return pl.DataFrame()

    # --- Generate reac_seq per caseid ---
    # The REAC table may not have a unique sequence number,
    # so we generate one per caseid
    reacs = reacs.with_columns(
        pl.col("caseid")
        .cum_count()
        .over("caseid")
        .alias("reac_seq")
    )

    # Deduplicate on (caseid, pt_name) to avoid duplicate reactions per case
    reacs = reacs.unique(subset=["caseid", "pt_name"], keep="first")

    # Re-generate reac_seq after dedup
    reacs = reacs.with_columns(
        pl.col("caseid")
        .cum_count()
        .over("caseid")
        .alias("reac_seq")
    )

    result = reacs.select([
        pl.col("caseid"),
        pl.col("reac_seq"),
        pl.col("pt_name"),
    ])

    return result


def insert_reactions(reacs_df: pl.DataFrame) -> int:
    """Insert reaction records into faers_reactions DuckDB table.

    Args:
        reacs_df: Polars DataFrame from load_reac_tables()

    Returns:
        Number of rows inserted
    """
    if reacs_df.is_empty():
        logger.warning("No reaction records to insert")
        return 0

    with get_duckdb_connection() as conn:
        conn.register("reacs_staging", reacs_df.to_arrow())

        conn.execute("""
            INSERT OR REPLACE INTO faers_reactions
                (caseid, reac_seq, pt_name)
            SELECT caseid, reac_seq, pt_name
            FROM reacs_staging
        """)

        count = conn.execute("SELECT COUNT(*) FROM faers_reactions").fetchone()[0]

    logger.info(f"faers_reactions: {count:,} rows after insertion")
    return count


# ---------------------------------------------------------------------------
# Step E — Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict | None:
    """Load the checkpoint file if it exists.

    Returns:
        Checkpoint dict or None if no checkpoint exists.
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint: step={checkpoint.get('step')}")
        return checkpoint
    return None


def save_checkpoint(step: str, counts: dict[str, int]) -> None:
    """Save a checkpoint after completing a pipeline step.

    Args:
        step: Name of the completed step (e.g. "demo", "drug", "reac", "complete")
        counts: Dict with current table counts
    """
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "caseid_count": counts.get("faers_cases", 0),
        "drug_count": counts.get("faers_drugs", 0),
        "reac_count": counts.get("faers_reactions", 0),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Checkpoint saved: step={step}, counts={counts}")


# ---------------------------------------------------------------------------
# Step F — Final Validation
# ---------------------------------------------------------------------------

def log_validation_results() -> dict[str, int]:
    """Run and log the final validation query.

    Returns:
        Dict with validation counts
    """
    results = run_validation_query()

    logger.info("=" * 60)
    logger.info("FAERS Ingestion — Final Validation")
    logger.info("=" * 60)
    logger.info(f"  Total cases:              {results['total_cases']:>10,}")
    logger.info(f"  Cases with drugs:         {results['cases_with_drugs']:>10,}")
    logger.info(f"  Cases with reactions:     {results['cases_with_reactions']:>10,}")
    logger.info(f"  Total drug records:       {results['total_drug_records']:>10,}")
    logger.info(f"  Total reaction records:   {results['total_reaction_records']:>10,}")
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

STEP_ORDER = ["demo", "drug", "reac", "complete"]


def step_is_done(checkpoint: dict | None, step: str) -> bool:
    """Check if a step has already been completed based on checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict or None
        step: Step name to check

    Returns:
        True if the step (or a later step) is already complete
    """
    if checkpoint is None:
        return False
    completed_step = checkpoint.get("step", "")
    if completed_step not in STEP_ORDER:
        return False
    return STEP_ORDER.index(completed_step) >= STEP_ORDER.index(step)


def run_ingestion() -> None:
    """Execute the full FAERS ingestion pipeline.

    Steps:
    A. Discover FAERS files
    B. Load DEMO tables → faers_cases
    C. Load DRUG tables → faers_drugs
    D. Load REAC tables → faers_reactions
    E. Checkpoint after each step
    F. Run final validation query
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("AET-SE — FAERS ASCII Ingestion Pipeline")
    logger.info("=" * 60)

    # Initialize DuckDB schema
    logger.info("Initializing DuckDB schema...")
    init_schema()

    # Check for existing checkpoint
    checkpoint = load_checkpoint()

    # Step A — File Discovery
    raw_dir = settings.data.faers_raw_dir
    logger.info(f"Scanning for FAERS files in: {raw_dir}")
    quarters = discover_faers_files(raw_dir)

    if not quarters:
        logger.error(
            "No FAERS data files found. Please download and extract FAERS "
            "quarterly ASCII files to data/raw/faers/faers_ascii_YYYYQN/ascii/"
        )
        logger.error(
            "Download from: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"
        )
        sys.exit(1)

    logger.info(f"Found {len(quarters)} quarter(s): {', '.join(quarters.keys())}")

    # Step B — DEMO Table Loading
    if step_is_done(checkpoint, "demo"):
        logger.info("Step B (DEMO) — skipping, already completed (checkpoint)")
    else:
        logger.info("Step B — Loading DEMO tables...")
        cases_df = load_demo_tables(quarters)
        if cases_df.is_empty():
            logger.error("DEMO loading produced empty DataFrame — aborting")
            sys.exit(1)
        case_count = insert_cases(cases_df)
        save_checkpoint("demo", get_table_counts())

        # Free memory
        del cases_df

    # Step C — DRUG Table Loading
    if step_is_done(checkpoint, "drug"):
        logger.info("Step C (DRUG) — skipping, already completed (checkpoint)")
    else:
        logger.info("Step C — Loading DRUG tables...")
        drugs_df = load_drug_tables(quarters)
        if drugs_df.is_empty():
            logger.warning("DRUG loading produced empty DataFrame — check target drug list")
        else:
            drug_count = insert_drugs(drugs_df)
        save_checkpoint("drug", get_table_counts())

        # Free memory
        del drugs_df

    # Step D — REAC Table Loading
    if step_is_done(checkpoint, "reac"):
        logger.info("Step D (REAC) — skipping, already completed (checkpoint)")
    else:
        logger.info("Step D — Loading REAC tables...")
        reacs_df = load_reac_tables(quarters)
        if reacs_df.is_empty():
            logger.warning("REAC loading produced empty DataFrame")
        else:
            reac_count = insert_reactions(reacs_df)
        save_checkpoint("reac", get_table_counts())

        # Free memory
        del reacs_df

    # Step E/F — Final validation
    results = log_validation_results()
    save_checkpoint("complete", get_table_counts())

    elapsed = time.time() - start_time
    logger.info(f"Ingestion completed in {elapsed:.1f}s")

    # Sanity check: exit code 1 if total_cases < 1000
    if results["total_cases"] < 1000:
        logger.error(
            f"SANITY CHECK FAILED: Only {results['total_cases']} cases loaded. "
            f"Expected at least 1,000. Check your FAERS data files."
        )
        sys.exit(1)

    logger.info("✅ Day 1 — FAERS Ingestion complete. All tables populated.")


# ---------------------------------------------------------------------------
# Entry point: python -m aetse.data.ingest_faers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
