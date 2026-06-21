"""LLM Extraction Agent — Ollama + Llama 3.1.

Day 7 update: CuratedDrugLookup wired as rxnorm_matcher.
Signal 3 (RxNorm validation, weight=0.25) is now active.
Max confidence rises from 0.75 → 1.0 for verified target drugs.

Extracts drugs, reactions, and severity from patient review text
using Llama 3.1 8B via Ollama. Features:
- Disk cache (xxhash keyed) to avoid re-processing
- JSON parsing with 3-stage fallback (direct, regex JSON block, regex fields)
- Tenacity retry for Ollama connection failures
- Confidence scoring via validation.py

Usage:
    from aetse.pipeline.agents.extraction import extract_node
    result = extract_node(state)  # PVState → partial state dict
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from aetse.data.curated_drug_lookup import CuratedDrugLookup
from aetse.pipeline.agents.cache import cache_get, cache_set
from aetse.pipeline.agents.validation import compute_extraction_confidence
from aetse.schemas import PVState
from aetse.utils.logging import logger

# Module-level rxnorm_matcher singleton — Signal 3 now active (Day 7)
# Validates extracted drug names against 11 target drugs.
_rxnorm_matcher: CuratedDrugLookup = CuratedDrugLookup()


# ---------------------------------------------------------------------------
# Prompt template (LOCKED — do not modify)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a pharmacovigilance expert analyzing \
patient drug reviews for adverse events.

Extract the following from the patient review below:
- drugs: list of drug names mentioned (generic names preferred)
- reactions: list of adverse reactions or side effects described
- severity: overall severity assessment

Return ONLY valid JSON with this exact structure, no markdown, \
no explanation, no preamble:
{{"drugs": ["drug1", "drug2"], "reactions": ["reaction1", "reaction2"], \
"severity": "serious|non-serious|unknown"}}

Severity rules:
- "serious": hospitalization, disability, life-threatening, death
- "non-serious": mild/moderate symptoms that resolved
- "unknown": cannot determine from text

Patient review:
{review_text}"""


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------

def parse_llm_output(raw_output: str) -> Optional[dict[str, Any]]:
    """Parse LLM output into structured extraction dict.

    Three-stage fallback:
    1. Direct JSON parse (clean output)
    2. Extract JSON block from output (handles markdown wrapping)
    3. Regex field extraction (last resort)

    Args:
        raw_output: Raw string from LLM.

    Returns:
        Dict with keys: drugs, reactions, severity. None if all fail.
    """
    if not raw_output or not raw_output.strip():
        return None

    # Attempt 1: direct JSON parse
    try:
        cleaned = raw_output.strip()
        # Remove markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        result = json.loads(cleaned)
        if isinstance(result, dict):
            logger.debug("Parse method: json_direct")
            result["_parse_method"] = "json"
            return result
    except json.JSONDecodeError:
        pass

    # Attempt 2: find JSON block in output
    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                logger.debug("Parse method: json_block")
                result["_parse_method"] = "json"
                return result
    except json.JSONDecodeError:
        pass

    # Attempt 3: regex field extraction
    try:
        drugs_match = re.findall(
            r'"drugs"\s*:\s*\[([^\]]*)\]', raw_output
        )
        reactions_match = re.findall(
            r'"reactions"\s*:\s*\[([^\]]*)\]', raw_output
        )
        severity_match = re.search(
            r'"severity"\s*:\s*"(serious|non-serious|unknown)"',
            raw_output,
        )

        if drugs_match or reactions_match:
            drugs = (
                [d.strip().strip('"').strip("'")
                 for d in drugs_match[0].split(",")
                 if d.strip().strip('"').strip("'")]
                if drugs_match
                else []
            )
            reactions = (
                [r.strip().strip('"').strip("'")
                 for r in reactions_match[0].split(",")
                 if r.strip().strip('"').strip("'")]
                if reactions_match
                else []
            )

            logger.debug("Parse method: regex_fallback")
            return {
                "drugs": drugs,
                "reactions": reactions,
                "severity": severity_match.group(1) if severity_match else "unknown",
                "_parse_method": "regex_fallback",
            }
    except Exception:
        pass

    logger.warning("All parse methods failed")
    return None


# ---------------------------------------------------------------------------
# Ollama call with tenacity retry
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def call_ollama(
    text: str,
    model: str = "llama3.1:8b-instruct-q4_K_M",
) -> str:
    """Call Ollama with the extraction prompt.

    Args:
        text: Patient review text.
        model: Ollama model name.

    Returns:
        Raw LLM output string.

    Raises:
        Exception: After 3 retries if Ollama is unavailable.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(review_text=text),
            }
        ],
        options={
            "temperature": 0,     # deterministic for extraction
            "num_predict": 256,   # cap output tokens — we only need JSON
        },
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Extract node (replaces extract_stub in graph.py)
# ---------------------------------------------------------------------------

def extract_node(state: PVState) -> dict:
    """LLM extraction node for LangGraph pipeline.

    Replaces extract_stub from Day 4. Calls Ollama to extract drugs,
    reactions, and severity from raw review text.

    CONTINUITY NOTE: extraction_retries is incremented FIRST, before
    any LLM call. This matches the Day 4 stub convention where
    validate reads retries=1 after the first extract call. The
    route_by_confidence function counts EXTRACTION_RETRY strings
    in agent_trace (not the integer), so both mechanisms stay in sync.

    Args:
        state: Current PVState.

    Returns:
        Partial state dict with extraction results.
    """
    # Increment retries FIRST — same convention as extract_stub (Day 4).
    # validate reads this after extract returns, so first call shows
    # retries=1 in the validation trace. route_by_confidence counts
    # EXTRACTION_RETRY strings in agent_trace, not this integer.
    new_retries = state["extraction_retries"] + 1
    start = time.time()

    text = state["raw_text"]
    report_id = state["report_id"]

    # Check disk cache first
    cached = cache_get(text)
    if cached:
        logger.info(f"Cache hit for {report_id}")
        latency = (time.time() - start) * 1000
        # Recompute confidence with current rxnorm_matcher (Day 7 — Signal 3
        # active). Day 6 cache files stored conf capped at 0.75 because Signal 3
        # was inactive. Recomputing ensures stale cached scores are upgraded.
        cached_extracted = {
            "drugs": cached.get("drugs", []),
            "reactions": cached.get("reactions", []),
            "severity": cached.get("severity", "unknown"),
        }
        conf = compute_extraction_confidence(
            cached_extracted, rxnorm_matcher=_rxnorm_matcher
        )
        # Use EXTRACTION_RETRY prefix on retries so route_by_confidence
        # counts them correctly and terminates the loop after 2 retries.
        is_retry = new_retries > 1
        trace_key = "EXTRACTION_RETRY" if is_retry else "EXTRACTION"
        drugs_str = ",".join(cached.get("drugs", [])[:3]) or "none"
        trace_entry = (
            f"{trace_key}:{new_retries},"
            f"cached=True,conf={conf},"
            f"drugs={drugs_str}"
        )
        return {
            "extracted_drugs": cached.get("drugs", []),
            "extracted_reactions": cached.get("reactions", []),
            "severity": cached.get("severity", "unknown"),
            "extraction_confidence": conf,
            "extraction_retries": new_retries,
            "agent_trace": state["agent_trace"] + [trace_entry],
            "processing_latency_ms": {
                **state["processing_latency_ms"],
                "extract": round(latency, 2),
            },
        }

    # Call Ollama
    extracted = None
    raw_output = ""
    try:
        raw_output = call_ollama(text)
        extracted = parse_llm_output(raw_output)
    except Exception as e:
        logger.error(f"Ollama call failed for {report_id}: {e}")

    if extracted is None:
        logger.warning(
            f"Extraction failed for {report_id}. "
            f"Raw output: {raw_output[:200]}"
        )

    # Compute confidence with real rxnorm_matcher (Day 7 — Signal 3 active).
    # Max confidence now 1.0 for verified target drugs.
    confidence = compute_extraction_confidence(
        extracted, rxnorm_matcher=_rxnorm_matcher
    )

    # Cache result (include confidence in cached object)
    if extracted is not None:
        cache_obj = {
            "drugs": extracted.get("drugs", []),
            "reactions": extracted.get("reactions", []),
            "severity": extracted.get("severity", "unknown"),
            "_confidence": confidence,
            "_parse_method": extracted.get("_parse_method", "unknown"),
        }
        cache_set(text, cache_obj)

    latency = (time.time() - start) * 1000

    # Build trace entry
    is_retry = new_retries > 1
    trace_key = "EXTRACTION_RETRY" if is_retry else "EXTRACTION"
    drugs_str = (
        ",".join(extracted.get("drugs", [])[:3])
        if extracted
        else "none"
    )
    trace_entry = (
        f"{trace_key}:{new_retries},"
        f"conf={confidence},"
        f"drugs={drugs_str}"
    )

    return {
        "extracted_drugs": (
            extracted.get("drugs", []) if extracted else []
        ),
        "extracted_reactions": (
            extracted.get("reactions", []) if extracted else []
        ),
        "severity": (
            extracted.get("severity", "unknown")
            if extracted
            else "unknown"
        ),
        "extraction_confidence": confidence,
        "extraction_retries": new_retries,
        "agent_trace": state["agent_trace"] + [trace_entry],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "extract": round(latency, 2),
        },
    }
