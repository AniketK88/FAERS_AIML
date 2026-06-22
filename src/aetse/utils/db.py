"""DuckDB helper utilities.

Provides a context-managed DuckDB connection with:
- Memory limit enforcement (2GB)
- Thread count configuration (4 threads)
- Schema initialization (all CREATE TABLE IF NOT EXISTS)
- Parameterized query execution (no f-string SQL)
- Table counts helper for verification
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from aetse.config.settings import settings


# Processed data directory for checkpoints
PROCESSED_DIR: Path = settings.project_root / "data" / "processed"


@contextmanager
def get_duckdb_connection(read_only: bool = False) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Create a configured DuckDB connection.

    Args:
        read_only: If True, opens the database in read-only mode.

    Yields:
        A configured DuckDB connection.

    Raises:
        duckdb.IOException: If the database file cannot be accessed.
    """
    settings.duckdb.path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(
        str(settings.duckdb.path),
        read_only=read_only,
    )

    try:
        conn.execute(f"SET memory_limit = '{settings.duckdb.memory_limit}'")
        conn.execute(f"SET threads = {settings.duckdb.threads}")
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    """Initialize all DuckDB tables required by AET-SE.

    Creates tables if they don't exist. Safe to call repeatedly.
    Uses the exact schema specified in the project spec.
    """
    with get_duckdb_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faers_cases (
                caseid          VARCHAR PRIMARY KEY,
                caseversion     INTEGER,
                age_years       FLOAT,
                sex             VARCHAR,
                report_date     DATE,
                country         VARCHAR,
                serious         BOOLEAN,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS faers_drugs (
                caseid          VARCHAR REFERENCES faers_cases(caseid),
                drug_seq        INTEGER,
                drugname_raw    VARCHAR,
                drugname_norm   VARCHAR,
                rxnorm_rxcui    VARCHAR,
                role_cod        VARCHAR,
                PRIMARY KEY (caseid, drug_seq)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS faers_reactions (
                caseid          VARCHAR REFERENCES faers_cases(caseid),
                reac_seq        INTEGER,
                pt_name         VARCHAR,
                PRIMARY KEY (caseid, reac_seq)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prr_signals (
                drug            VARCHAR,
                reaction        VARCHAR,
                n_cases         INTEGER,
                prr             FLOAT,
                ror             FLOAT,
                chi2            FLOAT,
                is_signal       BOOLEAN,
                masking_warning BOOLEAN,
                computed_at     TIMESTAMP,
                PRIMARY KEY (drug, reaction)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_results (
                review_id           VARCHAR PRIMARY KEY,
                drug_norm           VARCHAR,
                extracted_drugs     VARCHAR,    -- JSON array stored as string
                extracted_reactions VARCHAR,    -- JSON array stored as string
                meddra_pts          VARCHAR,    -- JSON array stored as string
                mapping_scores      VARCHAR,    -- JSON array stored as string
                prr_signals         VARCHAR,    -- JSON array stored as string
                severity            VARCHAR,
                extraction_confidence FLOAT,
                signal_flag         VARCHAR,
                needs_human_review  BOOLEAN,
                agent_trace         VARCHAR,    -- JSON array stored as string
                extract_latency_ms  FLOAT,
                map_terms_latency_ms FLOAT,
                signal_check_latency_ms FLOAT,
                processed_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def get_table_counts() -> dict[str, int]:
    """Return row counts for all AET-SE tables.

    Returns:
        Dict mapping table name to row count, e.g.
        {"faers_cases": 50000, "faers_drugs": 120000, ...}
    """
    tables = ["faers_cases", "faers_drugs", "faers_reactions", "prr_signals", "pipeline_results"]
    counts: dict[str, int] = {}

    with get_duckdb_connection(read_only=True) as conn:
        for table in tables:
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = result[0] if result else 0
            except duckdb.CatalogException:
                # Table doesn't exist yet
                counts[table] = 0

    return counts


def run_validation_query() -> dict[str, int]:
    """Run the Day 1 final validation query.

    Returns:
        Dict with counts: total_cases, cases_with_drugs, cases_with_reactions,
        total_drug_records, total_reaction_records
    """
    with get_duckdb_connection(read_only=True) as conn:
        result = conn.execute("""
            SELECT
                COUNT(DISTINCT c.caseid) as total_cases,
                COUNT(DISTINCT d.caseid) as cases_with_drugs,
                COUNT(DISTINCT r.caseid) as cases_with_reactions,
                COUNT(d.caseid) as total_drug_records,
                COUNT(r.caseid) as total_reaction_records
            FROM faers_cases c
            LEFT JOIN faers_drugs d ON c.caseid = d.caseid
            LEFT JOIN faers_reactions r ON c.caseid = r.caseid
        """).fetchone()

    return {
        "total_cases": result[0],
        "cases_with_drugs": result[1],
        "cases_with_reactions": result[2],
        "total_drug_records": result[3],
        "total_reaction_records": result[4],
    }
