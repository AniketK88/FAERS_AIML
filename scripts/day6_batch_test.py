"""Day 6 — 10 review batch test with output written to file."""
import duckdb, time, resource, json
from aetse.pipeline.runner import run_pipeline

OUT_FILE = "data/cache/day6_batch_results.json"

conn = duckdb.connect("data/duckdb/faers.duckdb", read_only=True)
reviews = conn.execute("""
    SELECT review_id, review_text 
    FROM drug_reviews 
    WHERE has_ae_mention = TRUE 
    ORDER BY review_id
    LIMIT 10
""").fetchall()
conn.close()

print(f"Processing {len(reviews)} reviews with real LLM...")
total_start = time.time()
results = []

for i, (review_id, text) in enumerate(reviews):
    start = time.time()
    result = run_pipeline(review_id, text, source="review")
    elapsed = time.time() - start
    row = {
        "n": i + 1,
        "review_id": review_id,
        "conf": result["extraction_confidence"],
        "severity": result["severity"],
        "drugs": result["extracted_drugs"],
        "reactions": result["extracted_reactions"][:3],
        "needs_human_review": result["needs_human_review"],
        "signal_flag": result["signal_flag"],
        "time_s": round(elapsed, 1),
        "trace": result["agent_trace"],
    }
    results.append(row)
    print(f"  {i+1}/10 done: {review_id}  conf={row['conf']}  {elapsed:.1f}s")

total = time.time() - total_start
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

summary = {
    "total_time_s": round(total, 1),
    "avg_time_s": round(total / len(reviews), 1),
    "peak_ram_mb": round(peak_mb),
    "cache_hits": sum(1 for r in results if any("cached=True" in t for t in r["trace"])),
    "conf_above_075": sum(1 for r in results if r["conf"] >= 0.75),
    "human_flagged": sum(1 for r in results if r["needs_human_review"]),
    "results": results,
}

with open(OUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone in {total:.1f}s. Results written to {OUT_FILE}")
print(f"Cache hits: {summary['cache_hits']}/10")
print(f"Conf >=0.75: {summary['conf_above_075']}/10")
print(f"Human flagged: {summary['human_flagged']}/10")
print(f"Peak RAM: {peak_mb:.0f} MB")
