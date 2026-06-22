"""Day 8 Step 2 — Positive control integration test.

ibuprofen + gastrointestinal haemorrhage must surface as a detected signal.
"""
from aetse.pipeline.runner import run_pipeline

text = (
    "I have been taking ibuprofen for my back pain for 3 weeks. "
    "Last week I developed severe stomach bleeding and had to be "
    "hospitalized. My doctor said it was gastrointestinal haemorrhage "
    "caused by the ibuprofen."
)

result = run_pipeline("positive-control-01", text, source="review")

print("=== POSITIVE CONTROL VERIFICATION ===")
print(f"drugs:       {result['extracted_drugs']}")
print(f"reactions:   {result['extracted_reactions']}")
print(f"meddra_pts:  {result['meddra_pts']}")
print(f"signal_flag: {result['signal_flag']}")
print(f"confidence:  {result['extraction_confidence']}")
print(f"human_review:{result['needs_human_review']}")
print(f"trace:       {result['agent_trace']}")
print()

signals = result.get("prr_signals") or []
if signals:
    print(f"SIGNALS DETECTED: {len(signals)}")
    for sig in signals:
        print(f"  drug={sig['drug']}  reaction={sig['reaction']}")
        print(f"  PRR={sig['prr']}  ROR={sig['ror']}  chi2={sig['chi2']}  n={sig['n_cases']}")
        print(f"  masking_warning={sig['masking_warning']}")
else:
    print("NO SIGNALS DETECTED")

print()
print("Latency breakdown (ms):")
for node, ms in result["processing_latency_ms"].items():
    print(f"  {node:15}: {ms:.1f}ms")
