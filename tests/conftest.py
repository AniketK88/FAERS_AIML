"""AET-SE test configuration — shared fixtures for pytest."""

import pytest


@pytest.fixture
def sample_review_text() -> str:
    """A sample drug review with known adverse events for testing.

    Returns:
        A drug review mentioning ibuprofen with GI side effects.
    """
    return (
        "I've been taking ibuprofen 400mg for my arthritis pain for about 3 months. "
        "While it helps with the joint pain, I started experiencing severe stomach "
        "cramps and noticed blood in my stool. My doctor said it could be "
        "gastrointestinal bleeding from the NSAID. I had to switch to acetaminophen."
    )


@pytest.fixture
def sample_pv_state() -> dict:
    """A minimal PVState dict for testing pipeline nodes.

    Returns:
        A dict matching the PVState TypedDict shape with test values.
    """
    return {
        "report_id": "TEST-001",
        "raw_text": "Taking ibuprofen caused stomach bleeding.",
        "source": "review",
        "extracted_drugs": None,
        "extracted_reactions": None,
        "severity": None,
        "extraction_confidence": 0.0,
        "extraction_retries": 0,
        "meddra_pts": None,
        "mapping_scores": None,
        "prr_signals": None,
        "needs_human_review": False,
        "signal_flag": None,
        "agent_trace": [],
        "processing_latency_ms": {},
    }


@pytest.fixture
def sample_extraction_output() -> dict:
    """A sample LLM extraction output for testing.

    Returns:
        A dict with extracted drugs, reactions, severity.
    """
    return {
        "drugs": ["ibuprofen"],
        "reactions": ["gastrointestinal bleeding", "stomach cramps"],
        "severity": "serious",
    }
