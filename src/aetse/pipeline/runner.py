"""Pipeline runner — entry point for batch and interactive processing.

This module handles:
- Loading drug reviews from the processed Parquet file
- Running each review through the compiled LangGraph
- Storing results in DuckDB and agent traces in SQLite
- Progress tracking with tqdm
- CLI argument parsing (--sample N for quick tests)
"""
