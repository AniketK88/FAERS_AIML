"""Sentence embedding service using bge-small-en-v1.5 on CPU.

Forces CPU device so the GPU stays reserved for Ollama LLM inference.
Embeddings are normalized (unit vectors) enabling cosine similarity
via dot product — required for ChromaDB cosine space.

Usage:
    from aetse.pipeline.agents.embeddings import EmbeddingService
    svc = EmbeddingService()
    vec = svc.embed_single("stomach bleeding")
    vecs = svc.embed(["pain", "nausea", "headache"])
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from aetse.utils.logging import logger

MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingService:
    """CPU-bound embedding service for MedDRA term similarity."""

    def __init__(self) -> None:
        # Force CPU — GPU reserved for Ollama
        logger.info(f"Loading embedding model: {MODEL_NAME} (CPU)")
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        logger.info("Embedding model loaded")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            2D float32 array of shape (len(texts), embedding_dim),
            normalized to unit vectors.
        """
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # required for cosine similarity
        )

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: String to embed.

        Returns:
            1D float32 array of shape (embedding_dim,).
        """
        return self.embed([text])[0]
