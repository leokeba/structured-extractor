"""Tests for confidence scoring functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from structured_extractor import DocumentExtractor, ExtractionConfig
from structured_extractor.results.confidence import (
    ConfidenceAssessment,
    FieldConfidence,
    build_confidence_schema,
    compute_overall_confidence,
    identify_low_confidence_fields,
)


class Invoice(BaseModel):
    """Test invoice model."""

    invoice_number: str = Field(description="The invoice ID")
    total_amount: float = Field(description="Total amount due")
    vendor_name: str = Field(description="Name of the vendor")


class Person(BaseModel):
    """Test person model."""

    name: str
    age: int


class TestConfidenceUtilities:
    """Tests for confidence utility functions."""

    def test_compute_overall_confidence_returns_minimum(self) -> None:
        """Test that overall confidence is the minimum of all field confidences."""
        field_confidences = {
            "name": 0.95,
            "age": 0.75,
            "email": 0.88,
        }
        result = compute_overall_confidence(field_confidences)
        assert result == 0.75

    def test_compute_overall_confidence_empty_returns_zero(self) -> None:
        """Test that empty confidences return 0."""
        result = compute_overall_confidence({})
        assert result == 0.0

    def test_compute_overall_confidence_single_field(self) -> None:
        """Test with single field."""
        result = compute_overall_confidence({"name": 0.9})
        assert result == 0.9

    def test_identify_low_confidence_fields_default_threshold(self) -> None:
        """Test identifying low confidence fields with default threshold."""
        field_confidences = {
            "name": 0.95,
            "age": 0.65,
            "email": 0.75,
        }
        low_conf = identify_low_confidence_fields(field_confidences)
        assert "age" in low_conf
        assert "email" in low_conf
        assert "name" not in low_conf

    def test_identify_low_confidence_fields_custom_threshold(self) -> None:
        """Test with custom threshold."""
        field_confidences = {
            "name": 0.95,
            "age": 0.65,
            "email": 0.75,
        }
        low_conf = identify_low_confidence_fields(field_confidences, threshold=0.7)
        assert "age" in low_conf
        assert "email" not in low_conf
        assert "name" not in low_conf

    def test_identify_low_confidence_fields_all_high(self) -> None:
        """Test when all fields are above threshold."""
        field_confidences = {
            "name": 0.95,
            "age": 0.92,
        }
        low_conf = identify_low_confidence_fields(field_confidences)
        assert low_conf == []

    def test_identify_low_confidence_fields_all_low(self) -> None:
        """Test when all fields are below threshold."""
        field_confidences = {
            "name": 0.5,
            "age": 0.3,
        }
        low_conf = identify_low_confidence_fields(field_confidences)
        assert set(low_conf) == {"name", "age"}


class TestConfidenceSchemaBuilder:
    """Tests for build_confidence_schema function."""

    def test_build_confidence_schema_creates_correct_model(self) -> None:
        """Test that confidence schema has correct structure."""
        confidence_schema = build_confidence_schema(Invoice)

        # Check model name
        assert confidence_schema.__name__ == "InvoiceWithConfidence"

        # Check fields exist
        assert "extracted_data" in confidence_schema.model_fields
        assert "confidence_assessment" in confidence_schema.model_fields

    def test_build_confidence_schema_extracted_data_type(self) -> None:
        """Test that extracted_data has correct type."""
        confidence_schema = build_confidence_schema(Invoice)

        # The annotation should reference Invoice
        extracted_data_field = confidence_schema.model_fields["extracted_data"]
        assert extracted_data_field.annotation == Invoice

    def test_build_confidence_schema_confidence_assessment_type(self) -> None:
        """Test that confidence_assessment has correct type."""
        confidence_schema = build_confidence_schema(Invoice)

        confidence_field = confidence_schema.model_fields["confidence_assessment"]
        assert confidence_field.annotation == ConfidenceAssessment

    def test_build_confidence_schema_can_be_instantiated(self) -> None:
        """Test that the generated schema can be instantiated."""
        confidence_schema = build_confidence_schema(Person)

        instance = confidence_schema(
            extracted_data=Person(name="John", age=30),
            confidence_assessment=ConfidenceAssessment(
                overall_confidence=0.9,
                field_confidences=[
                    FieldConfidence(field_name="name", confidence=0.95),
                    FieldConfidence(field_name="age", confidence=0.85),
                ],
            ),
        )

        # Access via model_dump since the model is dynamically created
        data = instance.model_dump()
        assert data["extracted_data"]["name"] == "John"
        assert data["confidence_assessment"]["overall_confidence"] == 0.9


class TestFieldConfidence:
    """Tests for FieldConfidence model."""

    def test_field_confidence_basic(self) -> None:
        """Test basic field confidence creation."""
        fc = FieldConfidence(field_name="invoice_number", confidence=0.95)
        assert fc.field_name == "invoice_number"
        assert fc.confidence == 0.95
        assert fc.reasoning is None

    def test_field_confidence_with_reasoning(self) -> None:
        """Test field confidence with reasoning."""
        fc = FieldConfidence(
            field_name="total",
            confidence=0.7,
            reasoning="Value was partially obscured",
        )
        assert fc.reasoning == "Value was partially obscured"

    def test_field_confidence_bounds(self) -> None:
        """Test that confidence is bounded between 0 and 1."""
        # Valid bounds
        FieldConfidence(field_name="test", confidence=0.0)
        FieldConfidence(field_name="test", confidence=1.0)

        # Invalid bounds should raise
        with pytest.raises(ValueError):
            FieldConfidence(field_name="test", confidence=-0.1)

        with pytest.raises(ValueError):
            FieldConfidence(field_name="test", confidence=1.1)


class TestConfidenceAssessment:
    """Tests for ConfidenceAssessment model."""

    def test_confidence_assessment_basic(self) -> None:
        """Test basic confidence assessment creation."""
        assessment = ConfidenceAssessment(
            overall_confidence=0.85,
            field_confidences=[
                FieldConfidence(field_name="name", confidence=0.9),
                FieldConfidence(field_name="age", confidence=0.8),
            ],
        )
        assert assessment.overall_confidence == 0.85
        assert len(assessment.field_confidences) == 2
        assert assessment.assessment_notes is None

    def test_confidence_assessment_with_notes(self) -> None:
        """Test confidence assessment with notes."""
        assessment = ConfidenceAssessment(
            overall_confidence=0.6,
            field_confidences=[],
            assessment_notes="Document quality was poor",
        )
        assert assessment.assessment_notes == "Document quality was poor"


class TestExtractWithConfidence:
    """Tests for DocumentExtractor.extract_with_confidence method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_client: MagicMock) -> DocumentExtractor:
        """Create an extractor with mocked client."""
        with patch("structured_extractor.core.extractor.OpenAIClient", return_value=mock_client):
            return DocumentExtractor(api_key="test-key")

    def test_extract_with_confidence_returns_confidences(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that extract_with_confidence returns confidence scores."""
        # Build the confidence schema to get its type
        confidence_schema = build_confidence_schema(Person)

        # Create a mock parsed response
        mock_parsed = confidence_schema(
            extracted_data=Person(name="John Doe", age=30),
            confidence_assessment=ConfidenceAssessment(
                overall_confidence=0.92,
                field_confidences=[
                    FieldConfidence(field_name="name", confidence=0.95),
                    FieldConfidence(field_name="age", confidence=0.89),
                ],
            ),
        )

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.content = json.dumps({
            "extracted_data": {"name": "John Doe", "age": 30},
            "confidence_assessment": {
                "overall_confidence": 0.92,
                "field_confidences": [
                    {"field_name": "name", "confidence": 0.95},
                    {"field_name": "age", "confidence": 0.89},
                ],
            },
        })
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 200
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        result = extractor.extract_with_confidence("John Doe is 30 years old.", schema=Person)

        assert result.success is True
        assert result.confidence == 0.92
        assert result.field_confidences is not None
        assert result.field_confidences["name"] == 0.95
        assert result.field_confidences["age"] == 0.89
        assert result.data.name == "John Doe"
        assert result.data.age == 30

    def test_extract_with_confidence_identifies_low_confidence_fields(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that low confidence fields are identified."""
        confidence_schema = build_confidence_schema(Invoice)

        mock_parsed = confidence_schema(
            extracted_data=Invoice(
                invoice_number="INV-001",
                total_amount=100.0,
                vendor_name="Unknown",
            ),
            confidence_assessment=ConfidenceAssessment(
                overall_confidence=0.6,
                field_confidences=[
                    FieldConfidence(field_name="invoice_number", confidence=0.95),
                    FieldConfidence(field_name="total_amount", confidence=0.5),
                    FieldConfidence(field_name="vendor_name", confidence=0.3),
                ],
            ),
        )

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.content = "{}"
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 200
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        result = extractor.extract_with_confidence(
            "Invoice #INV-001, Total: $100",
            schema=Invoice,
        )

        assert result.success is True
        assert result.low_confidence_fields is not None
        assert "total_amount" in result.low_confidence_fields
        assert "vendor_name" in result.low_confidence_fields
        assert "invoice_number" not in result.low_confidence_fields

    def test_extract_with_confidence_custom_threshold(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test with custom confidence threshold."""
        confidence_schema = build_confidence_schema(Person)

        mock_parsed = confidence_schema(
            extracted_data=Person(name="Jane", age=25),
            confidence_assessment=ConfidenceAssessment(
                overall_confidence=0.75,
                field_confidences=[
                    FieldConfidence(field_name="name", confidence=0.85),
                    FieldConfidence(field_name="age", confidence=0.65),
                ],
            ),
        )

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.content = "{}"
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 150
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        # With default threshold (0.8), both would be flagged
        config = ExtractionConfig(confidence_threshold=0.9)
        result = extractor.extract_with_confidence(
            "Jane is 25",
            schema=Person,
            config=config,
        )

        assert result.success is True
        assert result.low_confidence_fields is not None
        # Both should be flagged with 0.9 threshold
        assert "name" in result.low_confidence_fields
        assert "age" in result.low_confidence_fields

    def test_extract_with_confidence_handles_error(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that extraction errors are handled gracefully."""
        mock_client.generate.side_effect = Exception("API Error")

        result = extractor.extract_with_confidence("Test doc", schema=Person)

        assert result.success is False
        assert "API Error" in str(result.error)


class TestExtractionResultWithConfidence:
    """Tests for ExtractionResult with confidence fields."""

    def test_extraction_result_low_confidence_fields(self) -> None:
        """Test ExtractionResult with low_confidence_fields."""
        from structured_extractor.results.types import ExtractionResult

        data = Person(name="Test", age=42)
        result = ExtractionResult(
            data=data,
            success=True,
            confidence=0.7,
            field_confidences={"name": 0.9, "age": 0.5},
            low_confidence_fields=["age"],
        )

        assert result.low_confidence_fields == ["age"]
        assert result.field_confidences is not None
        assert result.field_confidences["age"] == 0.5
