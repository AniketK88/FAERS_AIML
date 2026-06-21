"""Day 7 verification script — steps 1 & 2."""
import resource

# Step 1: Direct mapping test
print("=" * 55)
print("Step 1: Direct MedDRA mapping test")
print("=" * 55)
from aetse.pipeline.agents.meddra_mapper import MedDRAMapper
mapper = MedDRAMapper()
mapper.build_index()  # idempotent — skips if already built

test_reactions = [
    "stomach bleeding",
    "gastrointestinal haemorrhage",
    "heart attack",
    "cough",
    "swelling in legs",
]
for reaction in test_reactions:
    pt, score = mapper.map_reaction(reaction)
    status = "OK" if pt else "NO_MATCH"
    print(f"  {status}: {reaction:<35} -> {pt or 'NO MATCH'} (score={score})")

# Step 2: CuratedDrugLookup interface
print()
print("=" * 55)
print("Step 2: CuratedDrugLookup.best_match() verification")
print("=" * 55)
from aetse.data.curated_drug_lookup import CuratedDrugLookup
matcher = CuratedDrugLookup()
for drug in ["ibuprofen", "Metformin", "advil", "lipitor", "penicillin", "amoxicillin"]:
    score = matcher.best_match(drug)
    print(f"  best_match('{drug}') = {score}")

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\nPeak RAM: {peak_mb:.0f} MB")
print("DONE")
