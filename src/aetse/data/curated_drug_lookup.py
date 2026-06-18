"""Curated drug name normalization lookup for AET-SE target drugs.

Since we are working with only 11 target drug active ingredients,
and the RxTerms download may not be available (no network),
this module provides a hand-curated lookup table that maps
all known FAERS drug name variants to their canonical generic name.

This is used as a fallback/replacement for RxTerms fuzzy matching.
It is actually MORE reliable than bulk RxTerms matching for our
specific 11-drug scope, because it handles:
- Salt suffixes (besylate, calcium, hydrochloride, etc.)
- Brand names (Lipitor, Norvasc, Toprol XL, etc.)
- Combination products (aspirin\\caffeine, lisinopril\\hctz, etc.)
- Free-text chaos (dosage strings, typos, etc.)
"""

from __future__ import annotations

import polars as pl
from pathlib import Path

from aetse.config.settings import settings
from aetse.utils.logging import logger


RXNORM_LOOKUP_PATH: Path = settings.project_root / "data" / "processed" / "rxnorm_lookup.parquet"

# ---------------------------------------------------------------------------
# Canonical target drug definitions
# Each entry: generic_name → (rxcui, list of known raw name prefixes/keywords)
# RxCUIs from RxNorm for reference accuracy
# ---------------------------------------------------------------------------

TARGET_DRUG_DEFINITIONS: dict[str, dict] = {
    "aspirin": {
        "rxcui": "1191",
        "keywords": [
            "aspirin", "acetylsalicylic", "asa ",
            "st. joseph aspirin", "bayer", "ecotrin",
        ],
    },
    "ibuprofen": {
        "rxcui": "5640",
        "keywords": [
            "ibuprofen", "advil", "motrin", "burana", "nurofen",
        ],
    },
    "naproxen": {
        "rxcui": "7258",
        "keywords": [
            "naproxen", "aleve", "naprosyn", "anaprox",
            "naproxeno", "nalgesin", "miranax",
        ],
    },
    "diclofenac": {
        "rxcui": "3355",
        "keywords": [
            "diclofenac", "voltaren", "voltadol", "voltrex",
            "reactin", "naboal",
        ],
    },
    "celecoxib": {
        "rxcui": "140587",
        "keywords": [
            "celecoxib", "celebrex", "meticel",
        ],
    },
    "rofecoxib": {
        "rxcui": "54552",
        "keywords": [
            "rofecoxib", "vioxx",
        ],
    },
    "amlodipine": {
        "rxcui": "17767",
        "keywords": [
            "amlodipine", "norvasc", "norvask", "nordex",
            "s-amlodipine",
        ],
    },
    "lisinopril": {
        "rxcui": "29046",
        "keywords": [
            "lisinopril", "prinivil", "zestril", "prinil",
            "novatec", "prinzide",
        ],
    },
    "metformin": {
        "rxcui": "6809",
        "keywords": [
            "metformin", "glucophage", "riomet", "fortamet",
            "metforminum", "metforming",
        ],
    },
    "atorvastatin": {
        "rxcui": "83367",
        "keywords": [
            "atorvastatin", "lipitor", "zarator", "ridlip",
            "novostat", "suvast", "stator", "teva- atorvastatin",
            "mint atorvastatin",
        ],
    },
    "metoprolol": {
        "rxcui": "6918",
        "keywords": [
            "metoprolol", "toprol", "lopressor", "beloc",
            "sandoz-metoprolol", "pms-metoprolol",
        ],
    },
}


def build_curated_lookup() -> pl.DataFrame:
    """Build the curated lookup table and save as Parquet.

    Returns:
        Polars DataFrame with columns: [display_name, rxcui, generic_name]
    """
    records: list[dict[str, str]] = []

    for generic_name, info in TARGET_DRUG_DEFINITIONS.items():
        rxcui = info["rxcui"]
        for keyword in info["keywords"]:
            records.append({
                "display_name": keyword.upper().strip(),
                "rxcui": rxcui,
                "generic_name": generic_name,
            })

    df = pl.DataFrame(records)

    # Save as parquet
    RXNORM_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RXNORM_LOOKUP_PATH)

    logger.info(f"Built curated lookup: {df.height} entries → {RXNORM_LOOKUP_PATH}")
    return df


def normalize_drug_name(raw_name: str) -> tuple[str | None, str | None, float]:
    """Normalize a single drug name using keyword matching.

    Strategy:
    1. Lowercase the input
    2. Check if any target drug keyword appears in the name
    3. If multiple match, prefer the longest keyword match (most specific)

    Args:
        raw_name: Raw drug name from FAERS.

    Returns:
        Tuple of (generic_name, rxcui, confidence_score).
        Score is 100.0 for exact keyword match, 90.0 for substring match.
        Returns (None, None, 0.0) if no match.
    """
    if not raw_name:
        return (None, None, 0.0)

    name_lower = raw_name.strip().lower()

    best_match: tuple[str | None, str | None, float] = (None, None, 0.0)
    best_keyword_len = 0

    for generic_name, info in TARGET_DRUG_DEFINITIONS.items():
        rxcui = info["rxcui"]

        for keyword in info["keywords"]:
            keyword_lower = keyword.lower().strip()

            if keyword_lower in name_lower:
                # Prefer longer keyword matches (more specific)
                if len(keyword_lower) > best_keyword_len:
                    # Exact match (entire name is just the keyword) → score 100
                    # Substring match → score 90
                    score = 100.0 if name_lower == keyword_lower else 90.0
                    best_match = (generic_name, rxcui, score)
                    best_keyword_len = len(keyword_lower)

    return best_match


def batch_normalize_drugs(drug_names: list[str]) -> list[dict]:
    """Normalize a batch of drug names.

    Args:
        drug_names: List of raw drug name strings.

    Returns:
        List of dicts: {drugname_raw, drugname_norm, rxnorm_rxcui, match_score}
    """
    results: list[dict] = []
    cache: dict[str, tuple[str | None, str | None, float]] = {}

    for i, name in enumerate(drug_names):
        if (i + 1) % 200 == 0 or (i + 1) == len(drug_names):
            logger.info(f"  Matching {i + 1:,}/{len(drug_names):,}...")

        key = name.strip().lower()
        if key in cache:
            generic, rxcui, score = cache[key]
        else:
            generic, rxcui, score = normalize_drug_name(name)
            cache[key] = (generic, rxcui, score)

        results.append({
            "drugname_raw": name,
            "drugname_norm": generic,
            "rxnorm_rxcui": rxcui,
            "match_score": score,
        })

    matched = sum(1 for r in results if r["drugname_norm"] is not None)
    logger.info(
        f"  Batch complete: {matched:,}/{len(drug_names):,} matched "
        f"({matched * 100 / max(len(drug_names), 1):.1f}%)"
    )

    return results
