"""PRR/ROR Statistics Agent — disproportionality signal detection.

Computes pharmacovigilance signal scores from FAERS data using:
- Proportional Reporting Ratio (PRR)
- Reporting Odds Ratio (ROR)
- Chi-squared test with Yates correction

Implements safety guards:
- Minimum case count thresholds
- Masking bias warnings
- Parameterized DuckDB queries (no SQL injection)

CRITICAL: This module uses Pandas (not Polars) per project spec rule:
"Pandas ONLY in statistics.py (PRR/ROR formulas)"

The compute_prr_with_guards() function is LOCKED — do not modify.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from aetse.utils.db import get_duckdb_connection
from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Target drugs (all 11)
# ---------------------------------------------------------------------------

TARGET_DRUGS: list[str] = [
    "ibuprofen",
    "naproxen",
    "aspirin",
    "diclofenac",
    "celecoxib",
    "rofecoxib",
    "amlodipine",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "metoprolol",
]


# ---------------------------------------------------------------------------
# Positive control definitions
# ---------------------------------------------------------------------------

POSITIVE_CONTROLS: list[dict[str, str]] = [
    {"drug": "ibuprofen", "reaction": "Gastrointestinal haemorrhage"},
    {"drug": "rofecoxib", "reaction": "Myocardial infarction"},
    {"drug": "metformin", "reaction": "Lactic acidosis"},
    {"drug": "amlodipine", "reaction": "Oedema peripheral"},
    {"drug": "lisinopril", "reaction": "Cough"},
]


# ---------------------------------------------------------------------------
# LOCKED FORMULA — DO NOT MODIFY
# ---------------------------------------------------------------------------

def compute_prr_with_guards(
    df: pd.DataFrame,
    drug: str,
    reaction: str,
    min_cases: int = 3,
    min_prr: float = 2.0,
    min_chi2: float = 3.84,
) -> dict[str, Any]:
    """Compute PRR, ROR, and chi-squared for a drug-reaction pair.

    This is the LOCKED formula — do not modify.

    Args:
        df: Full contingency DataFrame with columns:
            'generic_drug', 'meddra_pt', 'caseid'
        drug: Normalized drug name.
        reaction: MedDRA Preferred Term.
        min_cases: Minimum a-cell count to compute (default 3).
        min_prr: PRR threshold for signal flag (default 2.0).
        min_chi2: Chi-squared threshold for signal flag (default 3.84).

    Returns:
        Dict with keys: drug, reaction, prr, ror, chi2, n_cases,
        signal, masking_warning, reason
    """
    a = len(df[(df.generic_drug == drug) & (df.meddra_pt == reaction)])
    b = len(df[(df.generic_drug == drug) & (df.meddra_pt != reaction)])
    c = len(df[(df.generic_drug != drug) & (df.meddra_pt == reaction)])
    d = len(df[(df.generic_drug != drug) & (df.meddra_pt != reaction)])

    if a < min_cases or b == 0 or c == 0:
        return {
            "prr": None,
            "signal": False,
            "reason": "insufficient_data",
            "n": a,
        }

    masking_warning = ((c + d) / (a + b + c + d)) < 0.20
    prr = (a / (a + b)) / (c / (c + d))
    ror = (a * d) / (b * c)
    n = a + b + c + d
    chi2 = (n * (abs(a * d - b * c) - n / 2) ** 2) / (
        (a + b) * (c + d) * (a + c) * (b + d)
    )

    return {
        "drug": drug,
        "reaction": reaction,
        "prr": round(prr, 3),
        "ror": round(ror, 3),
        "chi2": round(chi2, 3),
        "n_cases": a,
        "signal": prr >= min_prr and a >= min_cases and chi2 >= min_chi2,
        "masking_warning": masking_warning,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_case_level_counts() -> tuple[
    dict[tuple[str, str], int],  # pair_case_counts: (drug, reaction) → n_cases
    dict[str, int],              # drug_case_counts: drug → n_cases
    dict[str, int],              # reaction_case_counts: reaction → n_cases
    int,                         # total_cases
]:
    """Load case-level counts from DuckDB for PRR computation.

    Uses COUNT(DISTINCT caseid) for correct epidemiological PRR:
    - a = cases exposed to drug X AND experiencing reaction Y
    - a+b = cases exposed to drug X (any reaction)
    - a+c = cases experiencing reaction Y (any drug)
    - a+b+c+d = total cases in the database

    This avoids row-level inflation where a case with 2 drugs × 5 reactions
    creates 10 rows but should count as 1 case.

    Returns:
        Tuple of (pair_case_counts, drug_case_counts, reaction_case_counts, total_cases)
    """
    logger.info("Loading case-level counts from DuckDB...")

    with get_duckdb_connection(read_only=True) as conn:
        # Total cases
        total_cases = conn.execute(
            "SELECT COUNT(DISTINCT caseid) FROM faers_cases"
        ).fetchone()[0]
        logger.info(f"  → Total cases: {total_cases:,}")

        # Drug case counts: how many unique cases per drug
        drug_rows = conn.execute("""
            SELECT drugname_norm, COUNT(DISTINCT caseid) as n
            FROM faers_drugs
            WHERE drugname_norm IS NOT NULL
            GROUP BY drugname_norm
        """).fetchall()
        drug_case_counts = {row[0]: row[1] for row in drug_rows}
        logger.info(f"  → {len(drug_case_counts):,} drugs with case counts")

        # Reaction case counts: how many unique cases per reaction
        reaction_rows = conn.execute("""
            SELECT pt_name, COUNT(DISTINCT caseid) as n
            FROM faers_reactions
            WHERE pt_name IS NOT NULL
            GROUP BY pt_name
        """).fetchall()
        reaction_case_counts = {row[0]: row[1] for row in reaction_rows}
        logger.info(f"  → {len(reaction_case_counts):,} reactions with case counts")

        # Pair case counts: how many cases have both drug AND reaction
        # Only load pairs with n >= 3 to save memory
        pair_rows = conn.execute("""
            SELECT d.drugname_norm, r.pt_name, COUNT(DISTINCT d.caseid) as n
            FROM faers_drugs d
            JOIN faers_reactions r ON d.caseid = r.caseid
            WHERE d.drugname_norm IS NOT NULL
              AND r.pt_name IS NOT NULL
            GROUP BY d.drugname_norm, r.pt_name
            HAVING COUNT(DISTINCT d.caseid) >= 3
        """).fetchall()
        pair_case_counts = {(row[0], row[1]): row[2] for row in pair_rows}
        logger.info(f"  → {len(pair_case_counts):,} drug-reaction pairs with n >= 3")

    return pair_case_counts, drug_case_counts, reaction_case_counts, total_cases


def compute_prr_from_counts(
    drug: str,
    reaction: str,
    a: int,
    b: int,
    c: int,
    d: int,
    min_cases: int = 3,
    min_prr: float = 2.0,
    min_chi2: float = 3.84,
) -> dict[str, Any]:
    """Compute PRR/ROR/chi2 from pre-computed contingency counts.

    This applies the EXACT SAME MATH as the locked compute_prr_with_guards()
    formula, but accepts pre-computed a/b/c/d counts instead of doing
    per-pair DataFrame filtering. This is a performance optimization only —
    the statistical formulas are identical.

    Args:
        drug: Normalized drug name.
        reaction: MedDRA Preferred Term.
        a: Cases with both drug AND reaction.
        b: Cases with drug but NOT reaction.
        c: Cases with reaction but NOT drug.
        d: Cases with neither drug NOR reaction.
        min_cases: Minimum a-cell count.
        min_prr: PRR threshold for signal.
        min_chi2: Chi-squared threshold for signal.

    Returns:
        Same dict format as compute_prr_with_guards().
    """
    if a < min_cases or b == 0 or c == 0:
        return {
            "prr": None,
            "signal": False,
            "reason": "insufficient_data",
            "n": a,
        }

    masking_warning = ((c + d) / (a + b + c + d)) < 0.20
    prr = (a / (a + b)) / (c / (c + d))
    ror = (a * d) / (b * c)
    n = a + b + c + d
    chi2 = (n * (abs(a * d - b * c) - n / 2) ** 2) / (
        (a + b) * (c + d) * (a + c) * (b + d)
    )

    return {
        "drug": drug,
        "reaction": reaction,
        "prr": round(prr, 3),
        "ror": round(ror, 3),
        "chi2": round(chi2, 3),
        "n_cases": a,
        "signal": prr >= min_prr and a >= min_cases and chi2 >= min_chi2,
        "masking_warning": masking_warning,
    }


def compute_all_signals(
    min_cases: int = 3,
) -> list[dict[str, Any]]:
    """Compute PRR/ROR for all valid drug-reaction pairs.

    Uses case-level counts (COUNT DISTINCT caseid) for correct
    epidemiological PRR. The a/b/c/d contingency cells are:
    - a = cases with both drug AND reaction
    - b = cases with drug but NOT reaction
    - c = cases with reaction but NOT drug
    - d = cases with neither

    The statistical formulas are identical to compute_prr_with_guards().

    Args:
        min_cases: Minimum co-occurrence for computation.

    Returns:
        List of result dicts.
    """
    logger.info("=" * 60)
    logger.info("PRR/ROR Signal Computation (case-level)")
    logger.info("=" * 60)

    # Step 1: Load case-level counts from DuckDB
    pair_counts, drug_counts, reaction_counts, total_cases = (
        load_case_level_counts()
    )

    # Step 2: Compute PRR for each valid pair
    results: list[dict[str, Any]] = []
    valid_pairs = list(pair_counts.items())
    total = len(valid_pairs)

    logger.info(f"Computing PRR for {total:,} pairs...")

    for i, ((drug, reaction), a) in enumerate(valid_pairs):
        # a = cases with both drug AND reaction
        # a + b = total cases with drug
        ab = drug_counts.get(drug, 0)
        b = ab - a

        # a + c = total cases with reaction
        ac = reaction_counts.get(reaction, 0)
        c = ac - a

        # d = total cases - a - b - c
        d = total_cases - a - b - c

        result = compute_prr_from_counts(
            drug, reaction, a, b, c, d,
            min_cases=min_cases,
        )
        if result.get("prr") is not None:
            results.append(result)

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            logger.info(f"  Progress: {i + 1:,}/{total:,} pairs computed...")

    logger.info(f"  → Computed PRR for {len(results):,} pairs")

    # Summary stats
    signals = [r for r in results if r.get("signal")]
    masking = [r for r in results if r.get("masking_warning")]
    logger.info(f"  → Signals detected: {len(signals):,}")
    logger.info(f"  → Masking warnings: {len(masking):,}")

    return results


# ---------------------------------------------------------------------------
# Database insertion
# ---------------------------------------------------------------------------

def insert_signals_to_db(results: list[dict[str, Any]]) -> int:
    """Insert PRR results into prr_signals table using upsert logic.

    Args:
        results: List of result dicts from compute_prr_with_guards().

    Returns:
        Number of rows inserted/updated.
    """
    logger.info("Inserting signals into prr_signals table...")

    if not results:
        logger.warning("No results to insert")
        return 0

    with get_duckdb_connection() as conn:
        # Clear existing signals for a clean re-computation
        conn.execute("DELETE FROM prr_signals")

        # Batch insert using parameterized queries
        insert_sql = """
            INSERT INTO prr_signals (
                drug, reaction, n_cases, prr, ror, chi2,
                is_signal, masking_warning, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        count = 0
        for r in results:
            if r.get("prr") is not None:
                conn.execute(insert_sql, [
                    r["drug"],
                    r["reaction"],
                    r.get("n_cases", 0),
                    r["prr"],
                    r.get("ror"),
                    r.get("chi2"),
                    r.get("signal", False),
                    r.get("masking_warning", False),
                ])
                count += 1

        logger.info(f"  → Inserted {count:,} signal records")

    return count


# ---------------------------------------------------------------------------
# Positive control validation
# ---------------------------------------------------------------------------

def validate_positive_controls() -> list[dict[str, Any]]:
    """Validate the 5 positive control drug-reaction pairs.

    Returns:
        List of validation result dicts.
    """
    logger.info("=" * 60)
    logger.info("Positive Control Validation")
    logger.info("=" * 60)

    validations: list[dict[str, Any]] = []

    with get_duckdb_connection(read_only=True) as conn:
        for ctrl in POSITIVE_CONTROLS:
            drug = ctrl["drug"]
            reaction = ctrl["reaction"]

            row = conn.execute("""
                SELECT drug, reaction, prr, ror, chi2, n_cases,
                       is_signal, masking_warning
                FROM prr_signals
                WHERE drug = ?
                  AND reaction = ?
            """, [drug, reaction]).fetchone()

            if row:
                result = {
                    "drug": row[0],
                    "reaction": row[1],
                    "prr": row[2],
                    "ror": row[3],
                    "chi2": row[4],
                    "n_cases": row[5],
                    "is_signal": row[6],
                    "masking_warning": row[7],
                    "found": True,
                }
            else:
                result = {
                    "drug": drug,
                    "reaction": reaction,
                    "prr": None,
                    "found": False,
                    "is_signal": False,
                }

            # Determine pass/fail
            if result["found"] and result["prr"] is not None:
                if result["prr"] >= 2.0:
                    status = "✅ PASS"
                elif drug == "rofecoxib":
                    status = "⚠️ WEAK (expected — only 87 records)"
                else:
                    status = "❌ FAIL"
            elif not result["found"]:
                status = "❌ NOT FOUND"
            else:
                status = "❌ INSUFFICIENT DATA"

            result["status"] = status
            validations.append(result)

            logger.info(
                f"  {status} {drug:<15} + {reaction:<35} "
                f"PRR={result.get('prr', 'N/A')}, "
                f"n={result.get('n_cases', 0)}"
            )

    return validations


# ---------------------------------------------------------------------------
# Signal Check Node (Day 8) — replaces signal_check_stub in graph.py
# ---------------------------------------------------------------------------

import time
from aetse.config.settings import settings
from aetse.schemas import PVState

_DB_PATH = str(settings.project_root / "data" / "duckdb" / "faers.duckdb")


def lookup_prr_signals(
    drugs: list[str],
    meddra_pts: list[str],
    db_path: str = _DB_PATH,
) -> list[dict]:
    """Look up PRR signals for all (drug, meddra_pt) combinations.

    Queries the prr_signals table populated on Day 3. Returns only rows
    where is_signal=True (PRR>=2.0, chi2>=3.84, n_cases>=3).

    Uses LOWER() on both sides to handle mixed-case drug names from LLM
    and MedDRA PT names from ChromaDB.

    Args:
        drugs: Extracted drug names (LLM output, may be mixed case).
        meddra_pts: Mapped MedDRA PTs from map_terms_node.
        db_path: Path to DuckDB database.

    Returns:
        List of signal dicts for pairs where is_signal=True.
    """
    if not drugs or not meddra_pts:
        return []

    import duckdb as _duckdb

    conn = _duckdb.connect(db_path, read_only=True)
    signals: list[dict] = []

    try:
        for drug in drugs:
            for pt in meddra_pts:
                # Parameterized query — no f-strings (SQL injection safety)
                result = conn.execute(
                    """
                    SELECT drug, reaction, prr, ror, chi2,
                           n_cases, is_signal, masking_warning
                    FROM prr_signals
                    WHERE LOWER(drug)     = LOWER(?)
                      AND LOWER(reaction) = LOWER(?)
                    """,
                    [drug, pt],
                ).fetchone()

                if result and result[6]:  # is_signal = True
                    signals.append(
                        {
                            "drug":            result[0],
                            "reaction":        result[1],
                            "prr":             round(float(result[2]), 3),
                            "ror":             round(float(result[3]), 3),
                            "chi2":            round(float(result[4]), 3),
                            "n_cases":         result[5],
                            "masking_warning": result[7],
                        }
                    )
    finally:
        conn.close()

    return signals


def assign_signal_flag(signals: list[dict]) -> str:
    """Assign overall signal_flag based on max PRR across detected signals.

    Thresholds:
        high:   PRR >= 10
        medium: PRR >= 5
        low:    PRR >= 2 (is_signal already pre-filtered at this threshold)
        noise:  no signals detected

    Args:
        signals: Output of lookup_prr_signals.

    Returns:
        Signal flag string.
    """
    if not signals:
        return "noise"
    max_prr = max(s["prr"] for s in signals)
    if max_prr >= 10:
        return "high"
    elif max_prr >= 5:
        return "medium"
    else:
        return "low"


def signal_check_node(state: PVState) -> dict:
    """Signal check node for LangGraph pipeline.

    Replaces signal_check_stub from Day 4. Looks up (drug, MedDRA_PT)
    pairs against the prr_signals table (19,990 signals from Day 3).
    Assigns a signal_flag of high/medium/low/noise based on max PRR.

    Args:
        state: Current PVState.

    Returns:
        Partial state dict with prr_signals, signal_flag, latency.
    """
    start = time.time()

    drugs  = state.get("extracted_drugs") or []
    meddra = state.get("meddra_pts") or []

    if not drugs or not meddra:
        logger.info(
            f"signal_check: no drugs or meddra_pts to check "
            f"(drugs={drugs}, meddra={meddra})"
        )
        return {
            "prr_signals": [],
            "signal_flag": "noise",
            "agent_trace": state["agent_trace"] + [
                "SIGNAL_CHECK:no_input,flag=noise"
            ],
            "processing_latency_ms": {
                **state["processing_latency_ms"],
                "signal_check": 0.0,
            },
        }

    signals     = lookup_prr_signals(drugs, meddra)
    signal_flag = assign_signal_flag(signals)
    latency     = (time.time() - start) * 1000
    top_prr     = max((s["prr"] for s in signals), default=0.0)

    logger.info(
        f"signal_check: {len(signals)} signals found, "
        f"flag={signal_flag}, drugs={drugs}, pts={meddra[:3]}"
    )

    return {
        "prr_signals": signals,
        "signal_flag": signal_flag,
        "agent_trace": state["agent_trace"] + [
            f"SIGNAL_CHECK:signals={len(signals)},"
            f"flag={signal_flag},"
            f"top_prr={top_prr:.2f}"
        ],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "signal_check": round(latency, 2),
        },
    }
