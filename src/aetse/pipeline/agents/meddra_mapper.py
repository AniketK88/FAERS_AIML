"""MedDRA Mapping Agent — ChromaDB + bge-small-en-v1.5.

Builds a persistent ChromaDB vector index of MedDRA Preferred Terms
extracted from the faers_reactions table (free license workaround —
FAERS uses real MedDRA PTs as coded values).

Provides:
- MedDRAMapper: builds index, maps reaction strings to nearest PT
- map_terms_node: LangGraph node replacing map_terms_stub

ChromaDB index persists to data/chroma_db/ — build once, reused
on all subsequent runs (idempotent build_index).

Usage:
    from aetse.pipeline.agents.meddra_mapper import map_terms_node
    # Wire into graph.py: builder.add_node("map_terms", map_terms_node)
"""

from __future__ import annotations

import time
from typing import Any

import chromadb
import duckdb

from aetse.config.settings import settings
from aetse.pipeline.agents.embeddings import EmbeddingService
from aetse.schemas import PVState
from aetse.utils.logging import logger

CHROMA_PATH = str(settings.project_root / "data" / "chroma_db")
COLLECTION_NAME = "meddra_pts"
SIMILARITY_THRESHOLD = 0.85
DB_PATH = str(settings.project_root / "data" / "duckdb" / "faers.duckdb")


class MedDRAMapper:
    """Maps free-text reaction strings to MedDRA Preferred Terms via ChromaDB."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection: Any = None

    def build_index(self, db_path: str = DB_PATH) -> int:
        """Build ChromaDB index from FAERS reaction terms.

        Idempotent — if collection already has documents, skips rebuild.

        Args:
            db_path: Path to DuckDB database with faers_reactions table.

        Returns:
            Number of PTs in the collection.
        """
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() > 0:
            logger.info(
                f"ChromaDB already has {self.collection.count():,} PTs — skipping rebuild"
            )
            return self.collection.count()

        # Load distinct PT names from FAERS reactions table
        conn = duckdb.connect(db_path, read_only=True)
        pts = conn.execute(
            "SELECT DISTINCT pt_name FROM faers_reactions WHERE pt_name IS NOT NULL ORDER BY pt_name"
        ).fetchall()
        conn.close()

        pt_names = [row[0] for row in pts]
        logger.info(f"Embedding {len(pt_names):,} MedDRA PTs into ChromaDB...")

        batch_size = 100
        for i in range(0, len(pt_names), batch_size):
            batch = pt_names[i : i + batch_size]
            embeddings = self.embedding_service.embed(batch)
            self.collection.add(
                ids=[f"pt_{i + j}" for j in range(len(batch))],
                embeddings=embeddings.tolist(),
                documents=batch,
            )
            if (i // batch_size) % 10 == 0:
                logger.info(
                    f"  Embedded {min(i + batch_size, len(pt_names)):,}/{len(pt_names):,}"
                )

        logger.info(f"ChromaDB index built: {self.collection.count():,} PTs")
        return self.collection.count()

    def _ensure_collection(self) -> None:
        """Load collection if not already loaded."""
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    def map_reaction(self, reaction_text: str) -> tuple[str | None, float]:
        """Map a single reaction string to its nearest MedDRA PT.

        Args:
            reaction_text: Free-text reaction string from LLM extraction.

        Returns:
            Tuple of (pt_name, similarity_score).
            pt_name is None if best score < SIMILARITY_THRESHOLD.
        """
        self._ensure_collection()

        if not reaction_text or not reaction_text.strip():
            return None, 0.0

        query_embedding = self.embedding_service.embed_single(reaction_text)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            include=["documents", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return None, 0.0

        best_pt = results["documents"][0][0]
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Similarity = 1 - (distance / 2)
        best_distance = results["distances"][0][0]
        similarity = 1.0 - (best_distance / 2.0)

        if similarity >= SIMILARITY_THRESHOLD:
            return best_pt, round(similarity, 3)
        return None, round(similarity, 3)

    def map_reactions(
        self, reaction_list: list[str]
    ) -> tuple[list[str], list[float]]:
        """Map a list of reaction strings to MedDRA PTs.

        Args:
            reaction_list: Free-text reactions from LLM.

        Returns:
            Tuple of (pts, scores). If a reaction doesn't meet the
            threshold, falls back to the original reaction text.
        """
        pts: list[str] = []
        scores: list[float] = []
        for reaction in reaction_list:
            pt, score = self.map_reaction(reaction)
            pts.append(pt if pt is not None else reaction)
            scores.append(score)
        return pts, scores


# ---------------------------------------------------------------------------
# Module-level singleton — built once per process
# ---------------------------------------------------------------------------

_mapper: MedDRAMapper | None = None


def get_mapper() -> MedDRAMapper:
    """Return (or build) the module-level MedDRAMapper singleton."""
    global _mapper
    if _mapper is None:
        _mapper = MedDRAMapper()
        _mapper.build_index()
    return _mapper


# ---------------------------------------------------------------------------
# LangGraph node — replaces map_terms_stub
# ---------------------------------------------------------------------------

def map_terms_node(state: PVState) -> dict:
    """MedDRA mapping node for LangGraph pipeline.

    Replaces map_terms_stub from Day 4. Maps extracted reaction strings
    to MedDRA Preferred Terms using ChromaDB cosine similarity search
    with bge-small-en-v1.5 embeddings.

    Args:
        state: Current PVState.

    Returns:
        Partial state dict with meddra_pts, mapping_scores, latency.
    """
    start = time.time()
    mapper = get_mapper()
    reactions = state.get("extracted_reactions") or []

    if not reactions:
        latency = (time.time() - start) * 1000
        return {
            "meddra_pts": [],
            "mapping_scores": [],
            "agent_trace": state["agent_trace"] + ["MAPPING:no_reactions_to_map"],
            "processing_latency_ms": {
                **state["processing_latency_ms"],
                "map_terms": round(latency, 2),
            },
        }

    pts, scores = mapper.map_reactions(reactions)
    latency = (time.time() - start) * 1000

    top_pt = pts[0] if pts else "unknown"
    top_score = scores[0] if scores else 0.0

    logger.info(
        f"MedDRA mapping: {len(reactions)} reactions → "
        f"top='{top_pt}' (score={top_score})"
    )

    return {
        "meddra_pts": pts,
        "mapping_scores": scores,
        "agent_trace": state["agent_trace"] + [
            f"MAPPING:reactions={len(reactions)},"
            f"top_pt={top_pt},"
            f"top_score={top_score}"
        ],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "map_terms": round(latency, 2),
        },
    }
