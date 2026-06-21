"""Disk cache for LLM extraction results.

Uses xxhash for fast, deterministic, collision-resistant cache keys.
Cache files are stored as JSON in data/cache/llm_extractions/.
The cache key is based on the raw review text, so identical reviews
always produce the same cache key (important for idempotency).

Usage:
    from aetse.pipeline.agents.cache import cache_get, cache_set

    cached = cache_get(text)
    if cached is not None:
        return cached  # Cache HIT

    result = expensive_llm_call(text)
    cache_set(text, result)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import xxhash

from aetse.config.settings import settings
from aetse.utils.logging import logger

CACHE_DIR: Path = settings.project_root / "data" / "cache" / "llm_extractions"


def get_cache_key(text: str) -> str:
    """Generate xxhash cache key from input text.

    Args:
        text: Raw review text.

    Returns:
        16-char hex digest (xxh64).
    """
    return xxhash.xxh64(text.encode()).hexdigest()


def cache_get(text: str) -> Optional[dict[str, Any]]:
    """Look up a cached extraction result.

    Args:
        text: Raw review text.

    Returns:
        Cached result dict, or None if not found.
    """
    key = get_cache_key(text)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        logger.debug(f"Cache HIT: {key[:8]}...")
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            return None
    return None


def cache_set(text: str, result: dict[str, Any]) -> None:
    """Store an extraction result in the cache.

    Args:
        text: Raw review text (used to derive cache key).
        result: Extraction result dict to cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = get_cache_key(text)
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(json.dumps(result))
        logger.debug(f"Cache SET: {key[:8]}...")
    except OSError as e:
        logger.warning(f"Cache write failed for {key}: {e}")
