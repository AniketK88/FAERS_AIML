"""LangGraph state machine definition and graph construction.

This module defines:
- The StateGraph with 5 stub nodes: extract, validate, map_terms,
  signal_check, flag_human
- Conditional edges for retry routing and human-review flagging
- The compiled graph object for use by the pipeline runner

Graph topology:
    START → extract → validate →[confidence >= 0.75]→ map_terms → signal_check → END
                         ↓                    ↑
                  [confidence < 0.75]──→ extract (retry, max 2)
                         ↓
                  [retries exhausted]──→ flag_human → END

CRITICAL: All nodes are STUBS today (Day 4).
Real agent logic will be added in Days 5-7.
"""

from __future__ import annotations

import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from aetse.pipeline.agents.extraction import extract_node
from aetse.schemas import PVState
from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Routing function (LOCKED — do not modify)
# ---------------------------------------------------------------------------

def route_by_confidence(state: PVState) -> str:
    """Route based on extraction confidence and retry count.

    Logic:
    - confidence >= 0.75 → proceed to map_terms
    - confidence < 0.75 and retries < 2 → retry extraction
    - confidence < 0.75 and retries >= 2 → flag for human review

    This function is LOCKED. Do not modify.
    """
    retry_count = len(
        [t for t in state["agent_trace"] if "EXTRACTION_RETRY" in t]
    )
    if state["extraction_confidence"] >= 0.75:
        return "map_terms"
    elif retry_count < 2:
        return "extract"
    else:
        return "flag_human"


# ---------------------------------------------------------------------------
# Stub nodes — Day 4 scaffolds only
# ---------------------------------------------------------------------------

def extract_stub(state: PVState) -> dict:
    """STUB: Extraction agent node.

    In production (Day 5+): calls Llama 3.1 via Ollama to extract
    drugs, reactions, and severity from raw text.
    Today: returns hardcoded values.
    """
    start = time.time()
    logger.info(f"[STUB] extract called for report_id={state['report_id']}")
    latency = (time.time() - start) * 1000

    # Track retries: if we've been here before, increment and mark
    current_retries = state["extraction_retries"]
    trace_entry = f"EXTRACTION:stub=True,conf=0.80,drug=ibuprofen"
    if current_retries > 0:
        trace_entry = f"EXTRACTION_RETRY:{current_retries},stub=True,conf=0.80"

    return {
        "extracted_drugs": ["ibuprofen"],
        "extracted_reactions": ["headache"],
        "severity": "non-serious",
        "extraction_confidence": 0.80,
        "extraction_retries": current_retries + 1,
        "agent_trace": state["agent_trace"] + [trace_entry],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "extract": round(latency, 2),
        },
    }


def validate_stub(state: PVState) -> dict:
    """STUB: Validation node.

    In production (Day 5+): validates extraction quality,
    checks for hallucinations, computes confidence score.
    Today: passes through, logs current confidence.

    NOTE: validate does NOT change extraction_confidence.
    route_by_confidence reads the confidence set by extract node.
    """
    logger.info(
        f"[STUB] validate called, "
        f"confidence={state['extraction_confidence']}"
    )
    return {
        "agent_trace": state["agent_trace"]
        + [
            f"VALIDATION:conf={state['extraction_confidence']},"
            f"retries={state['extraction_retries']}"
        ]
    }


def map_terms_stub(state: PVState) -> dict:
    """STUB: MedDRA term mapping node.

    In production (Day 6): maps extracted reaction strings to
    MedDRA Preferred Terms using embedding similarity.
    Today: returns hardcoded values.
    """
    logger.info("[STUB] map_terms called")
    return {
        "meddra_pts": ["Headache"],
        "mapping_scores": [0.92],
        "agent_trace": state["agent_trace"]
        + ["MAPPING:stub=True,pt=Headache,score=0.92"],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "map_terms": 1.0,
        },
    }


def signal_check_stub(state: PVState) -> dict:
    """STUB: Signal detection node.

    In production (Day 6): looks up PRR signals from prr_signals
    table for each drug-reaction pair.
    Today: returns hardcoded values.
    """
    logger.info("[STUB] signal_check called")
    return {
        "prr_signals": [],
        "signal_flag": "noise",
        "agent_trace": state["agent_trace"]
        + ["SIGNAL_CHECK:stub=True,flag=noise"],
        "processing_latency_ms": {
            **state["processing_latency_ms"],
            "signal_check": 1.0,
        },
    }


def flag_human_stub(state: PVState) -> dict:
    """STUB: Human review flagging node.

    In production (Day 7): writes the case to a human-review queue
    with full context for manual triage.
    Today: sets needs_human_review=True and logs.
    """
    logger.info(
        f"[STUB] flag_human called for report_id={state['report_id']}"
    )
    return {
        "needs_human_review": True,
        "agent_trace": state["agent_trace"]
        + ["HUMAN_FLAG:reason=low_confidence"],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph state machine.

    Returns:
        Compiled LangGraph graph with MemorySaver checkpointer.
    """
    builder = StateGraph(PVState)

    # Add nodes
    builder.add_node("extract", extract_node)
    builder.add_node("validate", validate_stub)
    builder.add_node("map_terms", map_terms_stub)
    builder.add_node("signal_check", signal_check_stub)
    builder.add_node("flag_human", flag_human_stub)

    # Set entry point
    builder.set_entry_point("extract")

    # Wire edges
    builder.add_edge("extract", "validate")

    # Conditional routing from validate
    builder.add_conditional_edges(
        "validate",
        route_by_confidence,
        {
            "map_terms": "map_terms",
            "extract": "extract",
            "flag_human": "flag_human",
        },
    )

    builder.add_edge("map_terms", "signal_check")
    builder.add_edge("signal_check", END)
    builder.add_edge("flag_human", END)

    # Compile with in-memory checkpointer
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
