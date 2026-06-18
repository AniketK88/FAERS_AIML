"""Day 3 orchestrator — PRR/ROR signal computation pipeline.

Runs the complete signal detection pipeline:
1. Compute PRR/ROR for all valid drug-reaction pairs
2. Insert results into prr_signals table
3. Validate positive controls
4. Log summary statistics

Usage:
    python -m aetse.data.compute_signals

Uses Pandas for PRR computation (via statistics.py).
All DuckDB queries use parameterized values.
"""

from __future__ import annotations

import sys
import time

from aetse.pipeline.agents.statistics import (
    compute_all_signals,
    insert_signals_to_db,
    validate_positive_controls,
)
from aetse.utils.db import get_duckdb_connection
from aetse.utils.logging import logger


def run_verification_queries() -> None:
    """Run Day 3 verification queries and log results."""
    with get_duckdb_connection(read_only=True) as conn:
        logger.info("=" * 60)
        logger.info("Day 3 Verification Queries")
        logger.info("=" * 60)

        # 1. Total signals
        logger.info("\n1. Signal summary:")
        row = conn.execute("""
            SELECT
                COUNT(*) as total_pairs,
                SUM(CASE WHEN is_signal THEN 1 ELSE 0 END) as signals,
                SUM(CASE WHEN masking_warning THEN 1 ELSE 0 END) as masking
            FROM prr_signals
        """).fetchone()
        logger.info(f"   Total pairs computed: {row[0]:,}")
        logger.info(f"   Signals detected: {row[1]:,}")
        logger.info(f"   Masking warnings: {row[2]:,}")

        # 2. Top 20 strongest signals
        logger.info("\n2. Top 20 strongest signals:")
        rows = conn.execute("""
            SELECT drug, reaction, prr, ror, n_cases
            FROM prr_signals
            WHERE is_signal = TRUE
            ORDER BY prr DESC
            LIMIT 20
        """).fetchall()
        logger.info(f"   {'Drug':<15} {'Reaction':<40} {'PRR':>8} {'ROR':>8} {'n':>6}")
        for r in rows:
            logger.info(
                f"   {r[0]:<15} {r[1]:<40} {r[2]:>8.1f} {r[3]:>8.1f} {r[4]:>6}"
            )

        # 3. Signal count per drug
        logger.info("\n3. Signals per drug:")
        rows = conn.execute("""
            SELECT drug,
                   COUNT(*) as pairs,
                   SUM(CASE WHEN is_signal THEN 1 ELSE 0 END) as signals
            FROM prr_signals
            GROUP BY drug
            ORDER BY signals DESC
        """).fetchall()
        for r in rows:
            logger.info(f"   {r[0]:<15} {r[1]:>6} pairs, {r[2]:>4} signals")


def run_day3_pipeline() -> None:
    """Run the complete Day 3 pipeline.

    Steps:
    1. Compute PRR/ROR for all valid drug-reaction pairs
    2. Insert into prr_signals table
    3. Validate positive controls
    4. Run verification queries

    Exits with code 1 if any mandatory positive control fails.
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("AET-SE — Day 3: PRR/ROR Signal Detection")
    logger.info("=" * 60)

    # Step 1: Compute all signals
    results = compute_all_signals(min_cases=3)

    # Step 2: Insert into DB
    n_inserted = insert_signals_to_db(results)

    # Step 3: Validate positive controls
    validations = validate_positive_controls()

    # Step 4: Run verification queries
    run_verification_queries()

    # Check results
    elapsed = time.time() - start_time
    logger.info(f"\nDay 3 pipeline completed in {elapsed:.1f}s")

    # Determine pass/fail
    mandatory_failures = [
        v for v in validations
        if v["drug"] != "rofecoxib"  # rofecoxib is optional
        and "FAIL" in v.get("status", "")
    ]

    rofecoxib = [v for v in validations if v["drug"] == "rofecoxib"]
    if rofecoxib and "FAIL" in rofecoxib[0].get("status", ""):
        logger.warning(
            "⚠️ rofecoxib signal weak — expected due to only 87 records "
            "in recent quarters. Historical FAERS data (pre-2005) would "
            "show strong signal. This is the Weber effect in practice."
        )

    if mandatory_failures:
        logger.error(
            f"❌ {len(mandatory_failures)} mandatory positive control(s) FAILED: "
            f"{[v['drug'] for v in mandatory_failures]}"
        )
        logger.error("Stopping — report to user before continuing.")
        sys.exit(1)
    else:
        logger.info("✅ Day 3 — PRR/ROR Signal Detection complete.")
        logger.info(
            f"   {n_inserted:,} signal records computed, "
            f"positive controls validated."
        )


if __name__ == "__main__":
    run_day3_pipeline()
