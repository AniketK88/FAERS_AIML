"""RxNorm fuzzy matching engine for drug name normalization.

Provides RxNormMatcher: loads the RxNorm lookup table and uses
rapidfuzz.fuzz.token_sort_ratio to find the best generic drug name
match for any raw drug name string.

Key behaviours:
- Lowercase + strip salt suffixes before matching
- In-memory cache for repeated drug names
- Configurable threshold (default 80)

Usage:
    from aetse.pipeline.agents.rxnorm_matcher import RxNormMatcher
    matcher = RxNormMatcher("data/processed/rxnorm_lookup.parquet")
    score, generic, rxcui = matcher.best_match("ATORVASTATIN CALCIUM")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import polars as pl
from rapidfuzz import fuzz, process

from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Salt suffixes to strip before matching
# ---------------------------------------------------------------------------

SALT_SUFFIXES: list[str] = [
    "hydrochloride",
    "hcl",
    "sodium",
    "potassium",
    "calcium",
    "sulfate",
    "phosphate",
    "succinate",
    "tartrate",
    "maleate",
    "acetate",
    "citrate",
    "fumarate",
    "mesylate",
    "besylate",
    "dihydrate",
    "monohydrate",
    "trihydrate",
    "magnesium",
    "propanediol",
    "dl-lysine",
    "diethylamine",
]

# Pre-compile regex: match any salt suffix at word boundaries
_SALT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in SALT_SUFFIXES) + r")\b",
    re.IGNORECASE,
)


def _strip_salts(name: str) -> str:
    """Remove common salt suffixes from a drug name.

    Args:
        name: Raw drug name string.

    Returns:
        Drug name with salt suffixes removed and whitespace normalized.
    """
    cleaned = _SALT_PATTERN.sub("", name)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# RxNormMatcher
# ---------------------------------------------------------------------------

class RxNormMatcher:
    """Fuzzy drug name matcher using RxNorm RxTerms lookup.

    Loads a Parquet lookup table and provides fast fuzzy matching
    via rapidfuzz.fuzz.token_sort_ratio.

    Attributes:
        threshold: Minimum score (0–100) to accept a match.
        lookup_names: List of uppercase display names for matching.
        lookup_dict: {display_name_upper: (rxcui, generic_name)}
        cache: In-memory cache for previously matched names.
    """

    def __init__(
        self,
        lookup_path: str | Path,
        threshold: float = 80.0,
    ) -> None:
        """Initialize the matcher by loading the lookup table.

        Args:
            lookup_path: Path to the rxnorm_lookup.parquet file.
            threshold: Minimum fuzzy match score (0–100) to accept.
        """
        self.threshold = threshold
        self._cache: dict[str, tuple[float, str, str]] = {}

        lookup_path = Path(lookup_path)
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"RxNorm lookup not found: {lookup_path}\n"
                f"Run: python -m aetse.data.ingest_rxnorm"
            )

        df = pl.read_parquet(lookup_path)
        logger.info(f"RxNormMatcher loaded {df.height:,} lookup entries from {lookup_path}")

        # Build lookup dict: {DISPLAY_NAME_UPPER: (rxcui, generic_name)}
        self.lookup_dict: dict[str, tuple[str, str]] = {}
        self.lookup_names: list[str] = []

        for row in df.iter_rows(named=True):
            display = str(row.get("display_name", "") or "").strip().upper()
            rxcui = str(row.get("rxcui", "") or "")
            generic = str(row.get("generic_name", "") or "").strip().lower()

            if display:
                self.lookup_dict[display] = (rxcui, generic)
                self.lookup_names.append(display)

        logger.info(f"  → {len(self.lookup_names):,} matchable drug names loaded")

    def best_match(self, drug_name: str) -> tuple[float, str, str]:
        """Find the best RxNorm match for a drug name.

        Strategy:
        1. Strip salt suffixes, attempt fuzzy match
        2. If score < threshold, try again with original name
        3. Return whichever has the higher score

        Args:
            drug_name: Raw drug name string.

        Returns:
            Tuple of (score, generic_name, rxcui).
            Score is 0–100 (float). If no match found above threshold,
            returns (best_score, "", "").
        """
        key = drug_name.strip().lower()

        # Check cache
        if key in self._cache:
            return self._cache[key]

        name_upper = drug_name.strip().upper()

        # Exact match first
        if name_upper in self.lookup_dict:
            rxcui, generic = self.lookup_dict[name_upper]
            result = (100.0, generic, rxcui)
            self._cache[key] = result
            return result

        # Strategy 1: Strip salts and match
        stripped = _strip_salts(name_upper)
        best_score_1 = 0.0
        best_match_1: str = ""

        if stripped and stripped != name_upper:
            match_result = process.extractOne(
                stripped,
                self.lookup_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=0,
            )
            if match_result:
                best_match_1, best_score_1, _ = match_result

        # Strategy 2: Match with original name
        match_result_2 = process.extractOne(
            name_upper,
            self.lookup_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=0,
        )
        best_score_2 = 0.0
        best_match_2: str = ""
        if match_result_2:
            best_match_2, best_score_2, _ = match_result_2

        # Take whichever has the higher score
        if best_score_1 >= best_score_2 and best_score_1 > 0:
            best_display = best_match_1
            best_score = best_score_1
        elif best_score_2 > 0:
            best_display = best_match_2
            best_score = best_score_2
        else:
            result = (0.0, "", "")
            self._cache[key] = result
            return result

        if best_score >= self.threshold and best_display in self.lookup_dict:
            rxcui, generic = self.lookup_dict[best_display]
            result = (best_score, generic, rxcui)
        else:
            result = (best_score, "", "")

        self._cache[key] = result
        return result

    def normalize(self, drug_name: str) -> Optional[str]:
        """Return the normalized generic name if match score >= threshold.

        Args:
            drug_name: Raw drug name string.

        Returns:
            Generic drug name (lowercase) or None if no good match.
        """
        score, generic, _ = self.best_match(drug_name)
        if score >= self.threshold and generic:
            return generic
        return None

    def batch_normalize(
        self,
        drug_names: list[str],
    ) -> list[dict[str, str | float | None]]:
        """Normalize a batch of drug names efficiently.

        Uses the in-memory cache to avoid re-processing duplicates.

        Args:
            drug_names: List of raw drug name strings.

        Returns:
            List of dicts with keys:
            {drugname_raw, drugname_norm, rxnorm_rxcui, match_score}
        """
        results: list[dict[str, str | float | None]] = []
        total = len(drug_names)

        for i, name in enumerate(drug_names):
            if (i + 1) % 200 == 0 or (i + 1) == total:
                logger.info(f"  Matching {i + 1:,}/{total:,} drug names...")

            score, generic, rxcui = self.best_match(name)

            results.append({
                "drugname_raw": name,
                "drugname_norm": generic if score >= self.threshold else None,
                "rxnorm_rxcui": rxcui if score >= self.threshold else None,
                "match_score": round(score, 1),
            })

        # Stats
        matched = sum(1 for r in results if r["drugname_norm"] is not None)
        logger.info(
            f"  Batch complete: {matched:,}/{total:,} matched "
            f"({matched * 100 / max(total, 1):.1f}% match rate)"
        )

        return results
