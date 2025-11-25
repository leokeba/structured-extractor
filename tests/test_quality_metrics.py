"""Tests for extraction quality metrics."""

import pytest
from pydantic import BaseModel, Field

from structured_extractor import ExtractionQualityMetrics, compute_quality_metrics


# Test schemas
class SimpleSchema(BaseModel):
    """Simple test schema."""

    name: str = Field(description="Person name")
    age: int = Field(description="Person age")


class SchemaWithOptional(BaseModel):
    """Schema with optional fields."""

    name: str = Field(description="Required name")
    email: str | None = Field(default=None, description="Optional email")
    phone: str | None = Field(default=None, description="Optional phone")


class LargeSchema(BaseModel):
    """Schema with many fields."""

    field1: str = Field(description="Field 1")
    field2: str = Field(description="Field 2")
    field3: str | None = Field(default=None, description="Field 3")
    field4: str | None = Field(default=None, description="Field 4")
    field5: str | None = Field(default=None, description="Field 5")


# ============================================================================
# ExtractionQualityMetrics Tests
# ============================================================================


class TestExtractionQualityMetrics:
    """Tests for ExtractionQualityMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating quality metrics."""
        metrics = ExtractionQualityMetrics(
            fields_extracted=5,
            fields_total=6,
            completeness_ratio=0.833,
            average_confidence=0.85,
            min_confidence=0.6,
            high_confidence_count=4,
            low_confidence_count=1,
            required_fields_filled=3,
            required_fields_total=3,
            required_completeness=1.0,
            needs_review=False,
            quality_warnings=[],
        )

        assert metrics.fields_extracted == 5
        assert metrics.fields_total == 6
        assert metrics.completeness_ratio == 0.833
        assert metrics.average_confidence == 0.85
        assert metrics.needs_review is False

    def test_metrics_with_warnings(self):
        """Test metrics with warnings."""
        metrics = ExtractionQualityMetrics(
            fields_extracted=2,
            fields_total=6,
            completeness_ratio=0.333,
            needs_review=True,
            quality_warnings=["Low completeness", "Missing required fields"],
        )

        assert metrics.needs_review is True
        assert len(metrics.quality_warnings) == 2


class TestComputeQualityMetrics:
    """Tests for compute_quality_metrics function."""

    def test_compute_full_extraction(self):
        """Test metrics for complete extraction."""
        data = SimpleSchema(name="John Doe", age=30)

        metrics = compute_quality_metrics(data, SimpleSchema)

        assert metrics.fields_extracted == 2
        assert metrics.fields_total == 2
        assert metrics.completeness_ratio == 1.0
        assert metrics.required_completeness == 1.0
        assert metrics.needs_review is False
        assert len(metrics.quality_warnings) == 0

    def test_compute_partial_extraction(self):
        """Test metrics for partial extraction with missing optional fields."""
        data = SchemaWithOptional(name="John Doe", email=None, phone=None)

        metrics = compute_quality_metrics(data, SchemaWithOptional)

        assert metrics.fields_extracted == 1  # Only name has value
        assert metrics.fields_total == 3
        assert metrics.completeness_ratio == pytest.approx(0.333, rel=0.01)
        assert metrics.required_fields_filled == 1
        assert metrics.required_fields_total == 1
        assert metrics.required_completeness == 1.0  # All required fields filled

    def test_compute_with_confidence_scores(self):
        """Test metrics with confidence scores provided."""
        data = SimpleSchema(name="John Doe", age=30)
        confidences = {"name": 0.95, "age": 0.75}

        metrics = compute_quality_metrics(
            data, SimpleSchema, field_confidences=confidences
        )

        assert metrics.average_confidence == pytest.approx(0.85, rel=0.01)
        assert metrics.min_confidence == 0.75
        assert metrics.high_confidence_count == 1  # name >= 0.8
        assert metrics.low_confidence_count == 0  # none < 0.5

    def test_compute_with_low_confidence(self):
        """Test metrics flags low confidence correctly."""
        data = SimpleSchema(name="John Doe", age=30)
        confidences = {"name": 0.4, "age": 0.3}  # Both below threshold

        metrics = compute_quality_metrics(
            data, SimpleSchema, field_confidences=confidences
        )

        assert metrics.min_confidence == 0.3
        assert metrics.low_confidence_count == 2
        assert metrics.needs_review is True
        assert any("low confidence" in w.lower() for w in metrics.quality_warnings)

    def test_compute_needs_review_missing_required(self):
        """Test that missing required fields trigger review."""
        # Create with model_construct to bypass validation
        data = SimpleSchema.model_construct(name=None, age=None)

        metrics = compute_quality_metrics(data, SimpleSchema)

        assert metrics.required_completeness < 1.0
        assert metrics.needs_review is True
        assert any("required" in w.lower() for w in metrics.quality_warnings)

    def test_compute_custom_thresholds(self):
        """Test with custom confidence thresholds."""
        data = SimpleSchema(name="John Doe", age=30)
        confidences = {"name": 0.7, "age": 0.6}

        # With default threshold (0.8), both are low
        metrics = compute_quality_metrics(
            data,
            SimpleSchema,
            field_confidences=confidences,
            confidence_threshold=0.8,
        )
        assert metrics.high_confidence_count == 0

        # With lower threshold (0.6), one is high
        metrics = compute_quality_metrics(
            data,
            SimpleSchema,
            field_confidences=confidences,
            confidence_threshold=0.6,
        )
        assert metrics.high_confidence_count == 2

    def test_compute_empty_confidences(self):
        """Test with empty confidence dict."""
        data = SimpleSchema(name="John Doe", age=30)

        metrics = compute_quality_metrics(
            data, SimpleSchema, field_confidences={}
        )

        assert metrics.average_confidence is None
        assert metrics.min_confidence is None

    def test_compute_low_completeness_warning(self):
        """Test that very low completeness triggers warning."""
        # Schema with 5 fields, only 1 filled
        data = LargeSchema(
            field1="value",
            field2="",  # Empty string counts as not extracted
            field3=None,
            field4=None,
            field5=None,
        )

        metrics = compute_quality_metrics(data, LargeSchema)

        assert metrics.completeness_ratio < 0.5
        assert metrics.needs_review is True
        assert any("half" in w.lower() for w in metrics.quality_warnings)
