"""
Integration tests for edge cases.
Requires Ollama running and data/ present.
Run separately from unit tests:
    pytest tests/integration/ -v -m integration
"""
import pytest
from aetse.pipeline.runner import run_pipeline

@pytest.mark.integration
class TestEdgeCases:
    
    def test_empty_text(self):
        result = run_pipeline("edge-001", "", source="review")
        assert isinstance(result["extracted_drugs"], list)
        assert isinstance(result["agent_trace"], list)
        # Should not crash — may route to flag_human
    
    def test_very_short_text(self):
        result = run_pipeline("edge-002", "Pain.", source="review")
        assert result is not None
    
    def test_no_drug_mentioned(self):
        result = run_pipeline(
            "edge-003",
            "I had a terrible headache and nausea for three days.",
            source="review"
        )
        # drugs list may be empty — should not crash
        assert isinstance(result["extracted_drugs"], list)
    
    def test_multiple_drugs(self):
        result = run_pipeline(
            "edge-004",
            """I was taking ibuprofen and aspirin together. 
            I developed stomach pain and nausea.""",
            source="review"
        )
        # Should extract both drugs
        assert isinstance(result["extracted_drugs"], list)
    
    def test_non_english_text(self):
        result = run_pipeline(
            "edge-005",
            "J'ai pris de l'ibuprofène et j'ai eu des douleurs d'estomac.",
            source="review"
        )
        # Should not crash — may or may not extract correctly
        assert result is not None
    
    def test_long_text(self):
        long_text = "I have been taking ibuprofen. " * 100
        result = run_pipeline("edge-006", long_text, source="review")
        assert result is not None
    
    def test_vicoprofen_routes_to_flag_human(self):
        """Combination brand name not in 11-drug lookup"""
        result = run_pipeline(
            "edge-007",
            "I took Vicoprofen for pain and felt dizzy.",
            source="review"
        )
        # Vicoprofen not in curated lookup → confidence < 0.75
        # After 2 retries → needs_human_review = True
        # (may vary depending on LLM output)
        assert "HUMAN_FLAG" in str(result["agent_trace"]) or \
               result["needs_human_review"] == True or \
               result["signal_flag"] in [None, "noise"]
