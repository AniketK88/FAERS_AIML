"""Unit tests for Pydantic schemas and data models."""

from aetse.schemas import (
    DrugReview,
    ExtractionResult,
    GroundTruthLabel,
    MedDRAMapping,
    PRRSignal,
)


class TestDrugReview:
    """Tests for the DrugReview model."""

    def test_valid_review(self) -> None:
        """Test creating a valid DrugReview."""
        review = DrugReview(
            review_id="R001",
            drug_name="ibuprofen",
            condition="arthritis",
            review_text="This drug helped with my pain.",
            rating=7,
            has_ae_mention=False,
            word_count=6,
        )
        assert review.review_id == "R001"
        assert review.rating == 7

    def test_rating_bounds(self) -> None:
        """Test that rating enforces 1-10 bounds."""
        import pytest

        with pytest.raises(Exception):
            DrugReview(
                review_id="R002",
                drug_name="test",
                condition="test",
                review_text="test",
                rating=11,
                word_count=1,
            )


class TestExtractionResult:
    """Tests for the ExtractionResult model."""

    def test_default_values(self) -> None:
        """Test default values for ExtractionResult."""
        result = ExtractionResult()
        assert result.drugs == []
        assert result.reactions == []
        assert result.severity == "unknown"
        assert result.parse_method == "failed"


class TestPRRSignal:
    """Tests for the PRRSignal model."""

    def test_no_signal(self) -> None:
        """Test creating a non-signal result."""
        signal = PRRSignal(
            drug="ibuprofen",
            reaction="headache",
            reason="insufficient_data",
        )
        assert signal.signal is False
        assert signal.prr is None
