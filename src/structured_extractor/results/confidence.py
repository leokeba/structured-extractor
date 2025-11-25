"""Confidence scoring utilities for extraction results."""

from typing import Any

from pydantic import BaseModel, Field, create_model


class FieldConfidence(BaseModel):
    """Confidence assessment for a single extracted field."""

    field_name: str = Field(description="Name of the field being assessed")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (no confidence) to 1.0 (certain)",
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation for the confidence level",
    )


class ConfidenceAssessment(BaseModel):
    """Overall confidence assessment for an extraction."""

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the extraction (0.0 to 1.0)",
    )
    field_confidences: list[FieldConfidence] = Field(
        description="Confidence scores for each extracted field",
    )
    assessment_notes: str | None = Field(
        default=None,
        description="General notes about the extraction quality",
    )


def build_confidence_schema(
    base_schema: type[BaseModel],
    include_reasoning: bool = True,
) -> type[BaseModel]:
    """Build a schema that wraps the base schema with confidence scores.

    This creates a new Pydantic model that includes both the extracted data
    and confidence assessments for each field.

    Args:
        base_schema: The original Pydantic model for extraction.
        include_reasoning: Whether to include reasoning for confidence scores.

    Returns:
        A new Pydantic model that includes data and confidence assessment.
    """
    # Create the combined schema dynamically
    schema_name = f"{base_schema.__name__}WithConfidence"

    # Build field annotations for the confidence model
    field_annotations: dict[str, Any] = {
        "extracted_data": (
            base_schema,
            Field(description="The extracted structured data"),
        ),
        "confidence_assessment": (
            ConfidenceAssessment,
            Field(description="Confidence scores for the extraction"),
        ),
    }

    # Create the dynamic model
    confidence_model: type[BaseModel] = create_model(
        schema_name,
        **field_annotations,
    )

    return confidence_model


def compute_overall_confidence(field_confidences: dict[str, float]) -> float:
    """Compute overall confidence from field-level confidences.

    Uses the minimum confidence among all fields, since extraction is only
    as reliable as its least confident field.

    Args:
        field_confidences: Dict mapping field names to confidence scores.

    Returns:
        Overall confidence score (0.0 to 1.0).
    """
    if not field_confidences:
        return 0.0

    # Use minimum confidence - extraction is only as good as its weakest field
    return min(field_confidences.values())


def identify_low_confidence_fields(
    field_confidences: dict[str, float],
    threshold: float = 0.8,
) -> list[str]:
    """Identify fields with confidence below the threshold.

    Args:
        field_confidences: Dict mapping field names to confidence scores.
        threshold: Minimum confidence threshold (default 0.8).

    Returns:
        List of field names with confidence below threshold.
    """
    return [
        field_name
        for field_name, confidence in field_confidences.items()
        if confidence < threshold
    ]
