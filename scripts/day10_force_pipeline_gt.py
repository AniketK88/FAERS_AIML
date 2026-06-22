import pandas as pd
from aetse.pipeline.graph import build_graph
from aetse.utils.db import get_duckdb_connection
import json
import traceback

def force_run_gt():
    print("Reading ground truth IDs...")
    df = pd.read_csv('data/ground_truth/labeling_template.csv')
    gt_ids = df['review_id'].tolist()
    
    with get_duckdb_connection(read_only=True) as conn:
        try:
            processed = conn.execute(f"SELECT review_id FROM pipeline_results WHERE review_id IN (SELECT unnest(?))", [gt_ids]).df()['review_id'].tolist()
        except Exception:
            processed = []
            
        to_process = [rid for rid in gt_ids if rid not in processed]
        
        if not to_process:
            print("All ground truth reviews already processed by pipeline!")
            return
            
        print(f"Need to process {len(to_process)} reviews for evaluation.")
        
        # Get review data
        reviews = conn.execute(f"""
            SELECT review_id, review_text, drug_norm
            FROM drug_reviews
            WHERE review_id IN (SELECT unnest(?))
        """, [to_process]).df()
        
    graph = build_graph()
    
    for i, row in reviews.iterrows():
        print(f"Processing {i+1}/{len(to_process)}: {row['review_id']} - {row['drug_norm']}")
        try:
            state = {
                "review_id": row["review_id"],
                "review_text": row["review_text"],
                "drug_norm": row["drug_norm"],
                "drugs": [], "reactions": [], "severity": "unknown",
                "extracted_terms": [], "mapped_terms": [],
                "signals": [], "signal_flag": None,
                "needs_human_review": False, "agent_trace": [],
                "extraction_confidence": 0.0,
                "extract_latency_ms": 0, "map_terms_latency_ms": 0, "signal_check_latency_ms": 0,
                "extraction_retries": 0
            }
            final_state = graph.invoke(state)
            
            # Save
            with get_duckdb_connection(read_only=False) as conn_write:
                conn_write.execute("""
                    INSERT INTO pipeline_results (
                        review_id, drugs, reactions, severity,
                        extraction_confidence, mapped_terms,
                        prr_signals, signal_flag, needs_human_review,
                        agent_trace, extract_latency_ms, map_terms_latency_ms,
                        signal_check_latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (review_id) DO UPDATE SET
                        drugs=EXCLUDED.drugs, reactions=EXCLUDED.reactions, severity=EXCLUDED.severity,
                        extraction_confidence=EXCLUDED.extraction_confidence, mapped_terms=EXCLUDED.mapped_terms,
                        prr_signals=EXCLUDED.prr_signals, signal_flag=EXCLUDED.signal_flag, 
                        needs_human_review=EXCLUDED.needs_human_review, agent_trace=EXCLUDED.agent_trace,
                        extract_latency_ms=EXCLUDED.extract_latency_ms, map_terms_latency_ms=EXCLUDED.map_terms_latency_ms,
                        signal_check_latency_ms=EXCLUDED.signal_check_latency_ms
                """, [
                    final_state['review_id'],
                    json.dumps(final_state['drugs']),
                    json.dumps(final_state['reactions']),
                    final_state['severity'],
                    final_state['extraction_confidence'],
                    json.dumps([m.model_dump() for m in final_state['mapped_terms']]),
                    json.dumps([s.model_dump() for s in final_state['signals']]),
                    final_state['signal_flag'],
                    final_state['needs_human_review'],
                    json.dumps(final_state['agent_trace']),
                    final_state['extract_latency_ms'],
                    final_state['map_terms_latency_ms'],
                    final_state['signal_check_latency_ms']
                ])
            print("✅ Saved")
        except Exception as e:
            print(f"❌ Failed: {e}")
            traceback.print_exc()
            
if __name__ == '__main__':
    force_run_gt()
