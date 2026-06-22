"""Day 8 Step 3 — Full 10-review run with all 4 real nodes."""
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

print(f"Running {len(reviews)} reviews through all 4 real nodes...\n")
total_start = time.time()
results = []

for i, (review_id, text) in enumerate(reviews):
    start = time.time()
    result = run_pipeline(review_id, text, source="review")
    elapsed = time.time() - start
    results.append(result)
    signals = result.get("prr_signals") or []
    meddra  = result.get("meddra_pts") or []
    flag    = result.get("signal_flag") or "n/a"
    print(f"{i+1:2}. {review_id}  flag={flag:<7}  "
          f"signals={len(signals)}  conf={result['extraction_confidence']}  {elapsed:.1f}s")
    print(f"    drugs={result['extracted_drugs']}")
    print(f"    meddra={meddra[:2]}")
    if signals:
        for s in signals[:2]:
            print(f"    SIGNAL: {s['drug']} -> {s['reaction']}  PRR={s['prr']}")

total = time.time() - total_start
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

with_signals = sum(1 for r in results if (r.get("signal_flag") or "noise") != "noise")
print(f"\n{'='*60}")
print(f"Total time:      {total:.1f}s ({total/len(reviews):.1f}s/review)")
print(f"Peak RAM:        {peak_mb:.0f} MB")
print(f"Signals found:   {with_signals}/{len(reviews)} reviews")
print(f"Breakdown:  "
      f"high={sum(1 for r in results if (r.get('signal_flag') or '') == 'high')}  "
      f"medium={sum(1 for r in results if (r.get('signal_flag') or '') == 'medium')}  "
      f"low={sum(1 for r in results if (r.get('signal_flag') or '') == 'low')}  "
      f"noise/n-a={sum(1 for r in results if (r.get('signal_flag') or 'noise') in ('noise', None))}")

print("\nLatency breakdown from last review (ms):")
for node, ms in results[-1]["processing_latency_ms"].items():
    print(f"  {node:15}: {ms:.1f}ms")
