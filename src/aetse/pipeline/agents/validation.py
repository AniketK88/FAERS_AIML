"""Validation Gate — confidence scoring and routing logic.

Computes a composite confidence score (0.0–1.0) based on:
1. JSON parse success (40% weight)
2. Required field population (35% weight)
3. RxNorm drug name match ratio (25% weight)

Routes to:
- map_terms: confidence >= 0.75
- extract (retry): confidence < 0.75, retries < 2
- flag_human: retries exhausted
"""
