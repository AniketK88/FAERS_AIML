"""
AET-SE Data Schemas — Pydantic v2 models for all data structures.

These models enforce type safety across the entire pipeline.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# LangGraph State (TypedDict — required by LangGraph)
# =============================================================================

from typing import TypedDict


class PVState(TypedDict):
    """LangGraph state for the pharmacovigilance pipeline.

    Every node reads/writes to this shared state. Fields are typed with
    Optional where an upstream node may not have populated them yet.
    """

    # --- Input ---
    report_id: str
    raw_text: str
    source: Literal["faers", "review"]

    # --- Extraction Agent outputs ---
    extracted_drugs: Optional[list[str]]
    extracted_reactions: Optional[list[str]]
    severity: Optional[Literal["serious", "non-serious", "unknown"]]
    extraction_confidence: float          # 0.0–1.0 composite score
    extraction_retries: int               # bounded retry counter

    # --- MedDRA Mapping outputs ---
    meddra_pts: Optional[list[str]]       # mapped Preferred Terms
    mapping_scores: Optional[list[float]]  # cosine similarity per PT

    # --- Statistics Agent outputs ---
    prr_signals: Optional[list[dict]]     # {drug, reaction, prr, ror, chi2, n, signal}

    # --- Routing flags ---
    needs_human_review: bool
    signal_flag: Optional[Literal["high", "medium", "low", "noise"]]

    # --- Audit ---
    agent_trace: list[str]                # e.g., ["EXTRACTION:drug=ibuprofen,conf=0.82"]
    processing_latency_ms: dict[str, float]  # latency per agent hop


# =============================================================================
# Drug Review Models (Kaggle data source)
# =============================================================================

class DrugReview(BaseModel):
    """A single drug review from the Kaggle dataset.

    Args:
        review_id: Unique identifier for the review.
        drug_name: Name of the drug being reviewed.
        condition: Medical condition being treated.
        review_text: Full text of the patient review.
        rating: Patient rating (1–10 scale).
        has_ae_mention: Whether keyword filtering flagged potential AE mention.
        word_count: Number of words in review_text.
    """

    review_id: str
    drug_name: str
    condition: str
    review_text: str
    rating: int = Field(ge=1, le=10)
    has_ae_mention: bool = False
    word_count: int = Field(ge=0)


class GroundTruthLabel(BaseModel):
    """Ground truth annotation for evaluation.

    Args:
        review_id: Links to DrugReview.review_id.
        drugs_gt: Ground truth drug mentions.
        reactions_gt: Ground truth adverse reaction mentions.
        severity_gt: Ground truth severity classification.
        labeled_by: Source of the label (manual or scispacy_baseline).
        label_confidence: Confidence in the label (1.0 for manual).
    """

    review_id: str
    drugs_gt: list[str]
    reactions_gt: list[str]
    severity_gt: Literal["serious", "non-serious", "unknown"]
    labeled_by: str = "manual"
    label_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# =============================================================================
# Extraction Output Model
# =============================================================================

class ExtractionResult(BaseModel):
    """Structured output from the LLM Extraction Agent.

    Args:
        drugs: List of drug names extracted from the text.
        reactions: List of adverse reactions extracted.
        severity: Assessed severity level.
        raw_llm_output: The raw LLM response (for debugging).
        parse_method: How the output was parsed (json or regex_fallback).
    """

    drugs: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    severity: Literal["serious", "non-serious", "unknown"] = "unknown"
    raw_llm_output: str = ""
    parse_method: Literal["json", "regex_fallback", "failed"] = "failed"


# =============================================================================
# Signal Detection Models
# =============================================================================

class PRRSignal(BaseModel):
    """A single PRR/ROR signal computation result.

    Args:
        drug: Generic drug name.
        reaction: MedDRA Preferred Term.
        prr: Proportional Reporting Ratio.
        ror: Reporting Odds Ratio.
        chi2: Chi-squared statistic.
        n_cases: Number of co-occurrence cases.
        signal: Whether this meets signal thresholds.
        masking_warning: Whether masking bias may affect the signal.
        reason: Why signal is False (if applicable).
    """

    drug: str
    reaction: str
    prr: Optional[float] = None
    ror: Optional[float] = None
    chi2: Optional[float] = None
    n_cases: int = 0
    signal: bool = False
    masking_warning: bool = False
    reason: Optional[str] = None


# =============================================================================
# MedDRA Mapping Result
# =============================================================================

class MedDRAMapping(BaseModel):
    """Result of mapping a reaction string to MedDRA Preferred Term.

    Args:
        original_term: The raw reaction string from extraction.
        mapped_pt: The matched MedDRA Preferred Term.
        similarity_score: Cosine similarity score (0.0–1.0).
        alternatives: Top-K alternative mappings considered.
    """

    original_term: str
    mapped_pt: Optional[str] = None
    similarity_score: float = 0.0
    alternatives: list[dict[str, float]] = Field(default_factory=list)


# =============================================================================
# Agent Trace Entry
# =============================================================================

class AgentTraceEntry(BaseModel):
    """A single step in the agent execution trace.

    Args:
        node_name: LangGraph node that generated this entry.
        timestamp: When the step executed.
        message: Human-readable summary of what happened.
        latency_ms: Time taken by this step in milliseconds.
        metadata: Any additional key-value data.
    """

    node_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str
    latency_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)


# =============================================================================
# Evaluation Metrics
# =============================================================================

class ExtractionMetrics(BaseModel):
    """Precision / Recall / F1 for extraction evaluation.

    Args:
        precision: Fraction of extracted items that are correct.
        recall: Fraction of ground truth items that were extracted.
        f1: Harmonic mean of precision and recall.
        support: Number of samples evaluated.
        entity_type: What was evaluated (drugs, reactions, severity).
    """

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    support: int = 0
    entity_type: str = "drugs"
