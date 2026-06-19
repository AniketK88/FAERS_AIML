"""Extraction agent for LLM-based adverse event entity extraction."""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from aetse.pipeline.agents.cache import cache_get, cache_set
from aetse.pipeline.agents.validation import compute_extraction_confidence
from aetse.schemas import PVState
from aetse.utils.logging import logger


EXTRACTION_PROMPT = """You are a pharmacovigilance expert analyzing 
patient drug reviews for adverse events.

Extract the following from the patient review below:
- drugs: list of drug names mentioned (generic names preferred)
- reactions: list of adverse reactions or side effects described
- severity: overall severity assessment

Return ONLY valid JSON with this exact structure, no markdown, 
no explanation, no preamble:
{{"drugs": ["drug1", "drug2"], "reactions": ["reaction1", "reaction2"], 
"severity": "serious|non-serious|unknown"}}

Severity rules:
- "serious": hospitalization, disability, life-threatening, death
- "non-serious": mild/moderate symptoms that resolved
- "unknown": cannot determine from text

Patient review:
{review_text}"""


def _parse_list_field(raw_items: str) -> list[str]:
    """Parse a comma-separated JSON-ish list body into clean strings."""
    return [
        item.strip().strip('"').strip("'")
        for item in raw_items.split(",")
        if item.strip().strip('"').strip("'")
    ]


def parse_llm_output(raw_output: str) -> Optional[dict]:
    """
    Attempt 1: direct JSON parse.
    Attempt 2: extract JSON block from output.
    Attempt 3: regex fallback for key fields.
    """
    try:
        cleaned = raw_output.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    try:
        drugs = re.findall(r'"drugs"\s*:\s*\[([^\]]*)\]', raw_output)
        reactions = re.findall(r'"reactions"\s*:\s*\[([^\]]*)\]', raw_output)
        severity_match = re.search(
            r'"severity"\s*:\s*"(serious|non-serious|unknown)"',
            raw_output,
        )
        if drugs or reactions:
            return {
                "drugs": _parse_list_field(drugs[0]) if drugs else [],
                "reactions": _parse_list_field(reactions[0]) if reactions else [],
                "severity": severity_match.group(1) if severity_match else "unknown",
            }
    except Exception:
        pass

    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def call_ollama(text: str, model: str = "llama3.1:8b-instruct-q4_K_M") -> str:
    """Call Ollama and return the raw model response content."""
    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(review_text=text),
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 256,
        },
    )
    return response["message"]["content"]


def extract_node(state: PVState) -> dict:
    """Extract drugs, reactions, and severity from a pipeline state."""
    # Increment retries FIRST: validate reads this after extract returns, and
    # this preserves the Day 4 trace/counter convention from extract_stub.
    new_retries = state["extraction_retries"] + 1
    start = time.time()

    text = state["raw_text"]
    report_id = state["report_id"]

    cached = cache_get(text)
    if cached:
        logger.info(f"Cache hit for {report_id}")
        latency = (time.time() - start) * 1000
        trace_key = "EXTRACTION_RETRY" if new_retries > 1 else "EXTRACTION"
        trace_entry = (
            f"{trace_key}:{new_retries},"
            f"cached=True,"
            f"conf={cached.get('_confidence', 0)}"
        )
        return {
            "extracted_drugs": cached.get("drugs", []),
            "extracted_reactions": cached.get("reactions", []),
            "severity": cached.get("severity", "unknown"),
            "extraction_confidence": cached.get("_confidence", 0.0),
            "extraction_retries": new_retries,
            "agent_trace": state["agent_trace"] + [trace_entry],
            "processing_latency_ms": {
                **state["processing_latency_ms"],
                "extract": round(latency, 2),
            },
        }

    try:
        raw_output = call_ollama(text)
        extracted = parse_llm_output(raw_output)
    except Exception as exc:
        logger.error(f"Ollama call failed for {report_id}: {exc}")
        extracted = None

    confidence = compute_extraction_confidence(extracted, rxnorm_matcher=None)

    if extracted:
        cache_set(text, {**extracted, "_confidence": confidence})

    latency = (time.time() - start) * 1000

    is_retry = new_retries > 1
    trace_key = "EXTRACTION_RETRY" if is_retry else "EXTRACTION"
    drugs_str = ",".join(extracted.get("drugs", [])[:3]) if extracted else "none"
    trace_entry = (
        f"{trace_key}:{new_retries},"
        f"conf={confidence},"
        f"drugs={drugs_str}"
    )

    return {
        "extracted_drugs": extracted.get("drugs", []) if extracted else [],
        "extracted_reactions": extracted.get("reactions", []) if extracted else [],
        "severity": extracted.get("severity", "unknown") if extracted else "unknown",
        "extraction_confidence": confidence,
        "extraction_retries": new_retries,
        "agent_trace": state["agent_trace"] + [trace_entry],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "extract": round(latency, 2),
        },
    }
