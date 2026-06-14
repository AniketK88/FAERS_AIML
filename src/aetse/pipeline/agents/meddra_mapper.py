"""MedDRA Mapping Agent — vector similarity-based term mapping.

Maps extracted adverse reaction strings to MedDRA Preferred Terms using:
- ChromaDB persistent vector store
- BAAI/bge-small-en-v1.5 sentence embeddings
- Top-K retrieval with cosine similarity scoring

The MedDRA PT vocabulary is sourced from FAERS REAC table data
(no paid MedDRA license required).
"""
