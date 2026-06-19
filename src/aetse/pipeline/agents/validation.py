"""Validation gate and extraction confidence scoring."""

from __future__ import annotations

from typing import Optional


def _rxnorm_score(rxnorm_matcher, drug_name: str) -> float:
    """Return an RxNorm score normalized to 0.0-1.0.

    Day 2's RxNormMatcher returns ``(score, generic, rxcui)`` on a 0-100
    scale, while Day 6's instructions allow a simple float-returning matcher.
    This helper accepts both shapes.
    """
    match = rxnorm_matcher.best_match(drug_name)
    if isinstance(match, tuple):
        score = float(match[0])
    else:
        score = float(match)
    return score / 100 if score > 1 else score


def compute_extraction_confidence(
    extracted: Optional[dict],
    rxnorm_matcher=None,
) -> float:
    """Compute extraction confidence from parse success, fields, and RxNorm.

    Passing ``rxnorm_matcher=None`` intentionally skips Signal 3 for Day 6,
    capping complete JSON outputs at 0.75 until Day 7 wires the real matcher.
    """
    if extracted is None:
        return 0.0

    score = 0.40
    fields = sum(
        [
            bool(extracted.get("drugs")),
            bool(extracted.get("reactions")),
            extracted.get("severity") in ["serious", "non-serious", "unknown"],
        ]
    )
    score += (fields / 3) * 0.35

    drugs = extracted.get("drugs") or []
    if rxnorm_matcher is not None and drugs:
        matched = sum(
            1
            for drug in drugs
            if _rxnorm_score(rxnorm_matcher, str(drug)) > 0.80
        )
        score += (matched / max(len(drugs), 1)) * 0.25

    return round(score, 3)
