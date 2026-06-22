"""Day 7 Step 4 — Re-run same 10 reviews, compare Day 6 vs Day 7 routing."""
import duckdb, time, resource
from aetse.pipeline.runner import run_pipeline

conn = duckdb.connect("data/duckdb/faers.duckdb", read_only=True)
reviews = conn.execute("""
    SELECT review_id, review_text 
    FROM drug_reviews 
    WHERE has_ae_mention = TRUE 
    ORDER BY review_id
    LIMIT 10
""").fetchall()
conn.close()

print(f"Re-running {len(reviews)} reviews (same set as Day 6 baseline)...\n")
total_start = time.time()

map_terms_count = 0
flag_human_count = 0

for i, (review_id, text) in enumerate(reviews):
    start = time.time()
    result = run_pipeline(review_id, text, source="review")
    elapsed = time.time() - start
    routed = "flag_human" if result["needs_human_review"] else "map_terms"
    if routed == "map_terms":
        map_terms_count += 1
    else:
        flag_human_count += 1

    meddra = result.get("meddra_pts") or []
    scores = result.get("mapping_scores") or []
    print(f"{i+1:2}. {review_id}: conf={result['extraction_confidence']}  "
          f"routed={routed}  {elapsed:.1f}s")
    print(f"    drugs={result['extracted_drugs']}")
    print(f"    meddra={meddra[:3]}  scores={[round(s,3) for s in scores[:3]]}")

total = time.time() - total_start
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\n{'='*55}")
print(f"BEFORE (Day 6):  5/10 map_terms,  5/10 flag_human  [conf capped at 0.75]")
print(f"AFTER  (Day 7): {map_terms_count}/10 map_terms, {flag_human_count}/10 flag_human  [Signal 3 active, max conf=1.0]")
print(f"Total time: {total:.1f}s  Peak RAM: {peak_mb:.0f} MB")
