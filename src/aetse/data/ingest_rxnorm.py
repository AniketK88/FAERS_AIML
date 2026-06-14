"""RxNorm RxTerms ingestion for drug name normalization.

This module handles:
- Loading the RxTerms text file (tab-separated)
- Building a generic drug name lookup (brand → generic)
- Providing a fuzzy matcher using rapidfuzz for approximate matching
- Exporting the lookup table as Parquet
"""
