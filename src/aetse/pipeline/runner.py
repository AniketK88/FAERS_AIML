"""Pipeline runner — entry point for batch and interactive processing.

This module handles:
- Creating initial PVState from inputs
- Running a single report through the compiled LangGraph
- Logging results and agent traces
- CLI entry point for quick tests

Usage:
    python -m aetse.pipeline.runner
"""

from __future__ import annotations

from aetse.schemas import PVState
from aetse.pipeline.graph import build_graph
from aetse.utils.logging import logger


def run_pipeline(
    report_id: str,
    text: str,
    source: str = "review",
) -> PVState:
    """Run a single report through the LangGraph pipeline.

    Args:
        report_id: Unique identifier for the report/review.
        text: Raw text to process (FAERS narrative or drug review).
        source: Data source — "faers" or "review".

    Returns:
        Final PVState after all graph nodes have executed.
    """
    graph = build_graph()

    initial_state: PVState = {
        "report_id": report_id,
        "raw_text": text,
        "source": source,
        "extracted_drugs": None,
        "extracted_reactions": None,
        "severity": None,
        "extraction_confidence": 0.0,
        "extraction_retries": 0,
        "meddra_pts": None,
        "mapping_scores": None,
        "prr_signals": None,
        "needs_human_review": False,
        "signal_flag": None,
        "agent_trace": [],
        "processing_latency_ms": {},
    }

    config = {"configurable": {"thread_id": report_id}}
    result = graph.invoke(initial_state, config=config)
    return result


def batch_run(limit: int = 100) -> int:
    """
    Run pipeline on reviews not yet in pipeline_results.
    Returns count of newly processed reviews.
    Only call this from a separate terminal — not from dashboard.
    """
    import json
    import duckdb
    from aetse.config.settings import settings
    
    db_path = str(settings.project_root / "data" / "duckdb" / "faers.duckdb")
    conn = duckdb.connect(db_path)
    
    # Get reviews not yet processed
    reviews = conn.execute("""
        SELECT r.review_id, r.review_text, r.drug_norm
        FROM drug_reviews r
        LEFT JOIN pipeline_results p ON r.review_id = p.review_id
        WHERE r.has_ae_mention = TRUE
          AND p.review_id IS NULL
        ORDER BY r.review_id
        LIMIT ?
    """, [limit]).fetchall()
    
    conn.close()
    
    if not reviews:
        logger.info("No unprocessed reviews found")
        return 0
    
    count = 0
    
    for review_id, text, drug_norm in reviews:
        try:
            result = run_pipeline(review_id, text, source="review")
            latency = result.get("processing_latency_ms", {})
            
            with duckdb.connect(db_path) as insert_conn:
                insert_conn.execute("""
                    INSERT OR REPLACE INTO pipeline_results VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                    )
                """, [
                    review_id,
                    drug_norm,
                    json.dumps(result.get("extracted_drugs") or []),
                    json.dumps(result.get("extracted_reactions") or []),
                    json.dumps(result.get("meddra_pts") or []),
                    json.dumps(result.get("mapping_scores") or []),
                    json.dumps(result.get("prr_signals") or []),
                    result.get("severity"),
                    result.get("extraction_confidence"),
                    result.get("signal_flag"),
                    result.get("needs_human_review", False),
                    json.dumps(result.get("agent_trace") or []),
                    latency.get("extract"),
                    latency.get("map_terms"),
                    latency.get("signal_check"),
                ])
            count += 1
            logger.info(f"Processed {count}/{len(reviews)}: {review_id}")
        except Exception as e:
            logger.error(f"Failed {review_id}: {e}")
            continue
    
    return count


def main() -> None:
    """Run a quick smoke test with a sample report."""
    logger.info("=" * 60)
    logger.info("AET-SE Pipeline Runner — Smoke Test")
    logger.info("=" * 60)

    result = run_pipeline(
        report_id="smoke-test-001",
        text="Patient took ibuprofen 400mg twice daily and developed "
        "severe stomach bleeding after 3 weeks.",
        source="review",
    )

    logger.info("\nPipeline result:")
    logger.info(f"  report_id: {result['report_id']}")
    logger.info(f"  extracted_drugs: {result['extracted_drugs']}")
    logger.info(f"  extracted_reactions: {result['extracted_reactions']}")
    logger.info(f"  severity: {result['severity']}")
    logger.info(f"  extraction_confidence: {result['extraction_confidence']}")
    logger.info(f"  meddra_pts: {result['meddra_pts']}")
    logger.info(f"  signal_flag: {result['signal_flag']}")
    logger.info(f"  needs_human_review: {result['needs_human_review']}")

    logger.info("\nAgent trace:")
    for step in result["agent_trace"]:
        logger.info(f"  → {step}")

    logger.info(f"\nLatency: {result['processing_latency_ms']}")
    logger.info("✅ Smoke test complete.")


if __name__ == "__main__":
    main()
