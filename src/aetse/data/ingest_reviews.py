"""Kaggle Drug Reviews ingestion and filtering.

This module handles:
- Loading the Kaggle drug reviews TSV dataset
- Keyword-based filtering for adverse event mentions
- Computing word count and has_ae_mention flags
- Outputting filtered reviews as Parquet for downstream processing
"""
