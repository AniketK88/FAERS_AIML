"""Extraction confidence scoring.

Computes a 0.0–1.0 confidence score for LLM extraction results
based on three signals:
    Signal 1 (0.40): JSON parsed successfully
    Signal 2 (0.35): Field completeness (drugs, reactions, severity)
    Signal 3 (0.25): Drug name validation via RxNorm (optional)

If rxnorm_matcher is None, Signal 3 is skipped and max confidence
is capped at 0.75 (Signals 1+2 only). This is the Day 6 behavior.
Day 7 will wire the real rxnorm_matcher.
"""

from __future__ import annotations

from typing import Any, Optional

from aetse.utils.logging import logger


def compute_extraction_confidence(
    extracted: Optional[dict[str, Any]],
    rxnorm_matcher: Any = None,
) -> float:
    """Compute extraction confidence from parsed LLM output.

    Args:
        extracted: Parsed extraction dict with drugs, reactions, severity.
            None if parsing failed entirely.
        rxnorm_matcher: Optional RxNorm matcher for drug name validation.
            If None, Signal 3 is skipped (max conf = 0.75).

    Returns:
        Confidence score in [0.0, 1.0], rounded to 3 decimals.
    """
    if extracted is None:
        return 0.0

    # Signal 1: JSON parsed successfully → base score
    score = 0.40

    # Signal 2: Field completeness (drugs, reactions, severity present)
    fields = sum([
        bool(extracted.get("drugs")),
        bool(extracted.get("reactions")),
        extracted.get("severity") in ("serious", "non-serious", "unknown"),
    ])
    score += (fields / 3) * 0.35

    # Signal 3: Drug name validation via RxNorm (optional)
    if rxnorm_matcher is not None and extracted.get("drugs"):
        try:
            matched = sum(
                1
                for d in extracted["drugs"]
                if rxnorm_matcher.best_match(d) > 0.80
            )
            score += (matched / max(len(extracted["drugs"]), 1)) * 0.25
        except Exception as e:
            logger.warning(f"RxNorm validation failed: {e}")

    return round(score, 3)
