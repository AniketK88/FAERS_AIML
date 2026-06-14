"""FAERS quarterly CSV data ingestion and DuckDB normalization.

This module handles:
- Loading FAERS ASCII text files (DEMO, DRUG, REAC, OUTC)
- Deduplication by (caseid, caseversion) — keeping latest version
- Schema creation and data insertion into DuckDB
- Filtering to target drug classes (NSAIDs, cardiovascular)
"""
