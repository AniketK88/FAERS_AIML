"""Disk cache for LLM extraction results.

Extraction calls are expensive and slow, so cache entries are keyed by a
deterministic xxhash of the input review text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import xxhash

from aetse.config.settings import settings
from aetse.utils.logging import logger


CACHE_DIR = Path(settings.cache.llm_cache_dir)


def get_cache_key(text: str) -> str:
    """Return a deterministic xxhash key for input text."""
    return xxhash.xxh64(text.encode()).hexdigest()


def cache_get(text: str) -> Optional[dict]:
    """Return cached extraction output for text, if present."""
    key = get_cache_key(text)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        logger.debug(f"Cache HIT: {key[:8]}...")
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


def cache_set(text: str, result: dict) -> None:
    """Persist extraction output for text."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = get_cache_key(text)
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    logger.debug(f"Cache SET: {key[:8]}...")
