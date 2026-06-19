"""Unit tests for LangGraph pipeline routing.

Tests verify that:
1. Happy path: high confidence → extract → validate → map_terms → signal_check
2. Retry routing: low confidence → extract retries up to 2 times
3. Human flag: after 2 retries with low confidence → flag_human
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aetse.pipeline.runner import run_pipeline
from aetse.pipeline.graph import build_graph, extract_stub, route_by_confidence
from aetse.schemas import PVState


class TestHappyPath:
    """Test the normal flow: confidence >= 0.75."""

    def test_happy_path_routing(self):
        """confidence=0.80 → should route extract→validate→map_terms→signal_check."""
        result = run_pipeline(
            "test-001", "Patient took ibuprofen and had headache"
        )

        trace = " ".join(result["agent_trace"])
        assert "EXTRACTION:" in trace
        assert "VALIDATION:" in trace
        assert "MAPPING:" in trace
        assert "SIGNAL_CHECK:" in trace
        assert "HUMAN_FLAG:" not in trace
        assert result["needs_human_review"] is False
        assert result["signal_flag"] == "noise"

    def test_happy_path_has_all_fields(self):
        """Real LLM wired — assert structure not hardcoded stub values"""
        result = run_pipeline(
            "test-001",
            "Patient took ibuprofen and had stomach pain",
        )

        # Structure assertions — not value assertions
        assert isinstance(result["extracted_drugs"], list)
        assert isinstance(result["extracted_reactions"], list)
        assert result["severity"] in ["serious", "non-serious", "unknown"]
        assert isinstance(result["extraction_confidence"], float)
        assert 0.0 <= result["extraction_confidence"] <= 1.0
        assert isinstance(result["agent_trace"], list)
        assert len(result["agent_trace"]) >= 3
        assert result["needs_human_review"] == False
        assert isinstance(result["processing_latency_ms"], dict)
        assert "extract" in result["processing_latency_ms"]


class TestRetryRouting:
    """Test retry routing when confidence < 0.75."""

    def _make_low_confidence_stub(self):
        """Create a stub that always returns low confidence."""

        def low_conf_extract_stub(state: PVState) -> dict:
            import time

            start = time.time()
            current_retries = state["extraction_retries"]
            latency = (time.time() - start) * 1000

            trace_entry = (
                f"EXTRACTION_RETRY:{current_retries},stub=True,conf=0.40"
                if current_retries > 0
                else "EXTRACTION:stub=True,conf=0.40"
            )

            return {
                "extracted_drugs": ["unknown"],
                "extracted_reactions": ["unknown"],
                "severity": "unknown",
                "extraction_confidence": 0.40,
                "extraction_retries": current_retries + 1,
                "agent_trace": state["agent_trace"] + [trace_entry],
                "processing_latency_ms": {
                    **state["processing_latency_ms"],
                    "extract": round(latency, 2),
                },
            }

        return low_conf_extract_stub

    def test_retry_then_flag_human(self):
        """confidence=0.40 → should retry extract, then flag_human after 2 retries."""
        low_stub = self._make_low_confidence_stub()

        with patch(
            "aetse.pipeline.graph.extract_stub", low_stub
        ):
            # Need to rebuild graph with patched stub
            from aetse.pipeline import graph as graph_module

            original = graph_module.extract_stub
            graph_module.extract_stub = low_stub
            try:
                result = run_pipeline(
                    "test-retry-001",
                    "Garbled text no useful info",
                )
            finally:
                graph_module.extract_stub = original

        trace = " ".join(result["agent_trace"])
        assert "EXTRACTION_RETRY:" in trace
        assert "HUMAN_FLAG:" in trace
        assert "MAPPING:" not in trace
        assert result["needs_human_review"] is True

    def test_retry_count_capped_at_two(self):
        """Should not retry more than 2 times (3 total extract calls max)."""
        low_stub = self._make_low_confidence_stub()

        from aetse.pipeline import graph as graph_module

        original = graph_module.extract_stub
        graph_module.extract_stub = low_stub
        try:
            result = run_pipeline(
                "test-retry-002",
                "More garbled text",
            )
        finally:
            graph_module.extract_stub = original

        # Count extraction calls in trace
        extraction_calls = [
            t
            for t in result["agent_trace"]
            if t.startswith("EXTRACTION")
        ]
        # First call + at most 2 retries = max 3
        assert len(extraction_calls) <= 3
        assert result["needs_human_review"] is True


class TestHumanFlagRouting:
    """Test direct human flag routing."""

    def test_human_flag_sets_review_true(self):
        """After retries exhausted → needs_human_review=True."""
        low_stub = TestRetryRouting()._make_low_confidence_stub()

        from aetse.pipeline import graph as graph_module

        original = graph_module.extract_stub
        graph_module.extract_stub = low_stub
        try:
            result = run_pipeline(
                "test-human-001",
                "No useful text at all",
            )
        finally:
            graph_module.extract_stub = original

        assert result["needs_human_review"] is True
        assert "HUMAN_FLAG:reason=low_confidence" in result["agent_trace"]
        assert result["signal_flag"] is None  # Never reached signal_check


class TestRouteByConfidence:
    """Test the routing function directly."""

    def _make_state(
        self, confidence: float, retries: int, trace: list[str]
    ) -> PVState:
        return {
            "report_id": "test",
            "raw_text": "",
            "source": "review",
            "extracted_drugs": None,
            "extracted_reactions": None,
            "severity": None,
            "extraction_confidence": confidence,
            "extraction_retries": retries,
            "meddra_pts": None,
            "mapping_scores": None,
            "prr_signals": None,
            "needs_human_review": False,
            "signal_flag": None,
            "agent_trace": trace,
            "processing_latency_ms": {},
        }

    def test_high_confidence_routes_to_map_terms(self):
        state = self._make_state(0.85, 0, [])
        assert route_by_confidence(state) == "map_terms"

    def test_exactly_075_routes_to_map_terms(self):
        state = self._make_state(0.75, 0, [])
        assert route_by_confidence(state) == "map_terms"

    def test_low_confidence_no_retries_routes_to_extract(self):
        state = self._make_state(0.40, 0, [])
        assert route_by_confidence(state) == "extract"

    def test_low_confidence_one_retry_routes_to_extract(self):
        state = self._make_state(0.40, 1, ["EXTRACTION_RETRY:1"])
        assert route_by_confidence(state) == "extract"

    def test_low_confidence_two_retries_routes_to_flag_human(self):
        state = self._make_state(
            0.40, 2, ["EXTRACTION_RETRY:1", "EXTRACTION_RETRY:2"]
        )
        assert route_by_confidence(state) == "flag_human"
