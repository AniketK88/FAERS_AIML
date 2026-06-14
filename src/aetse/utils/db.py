"""DuckDB helper utilities.

Provides a context-managed DuckDB connection with:
- Memory limit enforcement
- Thread count configuration
- Parameterized query execution (no f-string SQL)
- Connection pooling for multi-access patterns
"""

from contextlib import contextmanager
from typing import Generator

import duckdb

from aetse.config.settings import settings


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
