"""Tests for result types."""

from pydantic import BaseModel

from structured_extractor.results.types import ExtractionResult, FieldResult


class SampleData(BaseModel):
    """Sample extracted data for testing."""

    name: str
    value: int


class TestFieldResult:
    """Tests for FieldResult class."""

    def test_basic_field_result(self) -> None:
        """Test basic field result creation."""
        result = FieldResult(
            name="invoice_number",
            value="INV-001",
        )

        assert result.name == "invoice_number"
        assert result.value == "INV-001"
        assert result.confidence is None
        assert result.source_text is None

    def test_field_result_with_confidence(self) -> None:
        """Test field result with confidence score."""
        result = FieldResult(
            name="total",
            value=150.00,
            confidence=0.95,
            source_text="Total: $150.00",
        )

        assert result.confidence == 0.95
        assert result.source_text == "Total: $150.00"


class TestExtractionResult:
    """Tests for ExtractionResult class."""

    def test_successful_extraction(self) -> None:
        """Test successful extraction result."""
        data = SampleData(name="Test", value=42)
        result = ExtractionResult(
            data=data,
            success=True,
            model_used="gpt-4.1",
            tokens_used=100,
        )

        assert result.data.name == "Test"
        assert result.data.value == 42
        assert result.success is True
        assert result.model_used == "gpt-4.1"
        assert result.tokens_used == 100
        assert result.error is None

    def test_failed_extraction(self) -> None:
        """Test failed extraction result."""
        data = SampleData(name="", value=0)
        result = ExtractionResult(
            data=data,
            success=False,
            error="Failed to parse document",
        )

        assert result.success is False
        assert result.error == "Failed to parse document"

    def test_extraction_with_confidence(self) -> None:
        """Test extraction result with confidence scores."""
        data = SampleData(name="Test", value=42)
        result = ExtractionResult(
            data=data,
            success=True,
            confidence=0.92,
            field_confidences={"name": 0.95, "value": 0.89},
        )

        assert result.confidence == 0.92
        assert result.field_confidences is not None
        assert result.field_confidences["name"] == 0.95
        assert result.field_confidences["value"] == 0.89

    def test_cached_result(self) -> None:
        """Test cached extraction result."""
        data = SampleData(name="Cached", value=1)
        result = ExtractionResult(
            data=data,
            success=True,
            cached=True,
        )

        assert result.cached is True

    def test_result_with_cost(self) -> None:
        """Test extraction result with cost tracking."""
        data = SampleData(name="Test", value=42)
        result = ExtractionResult(
            data=data,
            success=True,
            tokens_used=500,
            cost_usd=0.0025,
        )

        assert result.tokens_used == 500
        assert result.cost_usd == 0.0025
