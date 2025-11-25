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


class ExtractionQualityMetrics(BaseModel):
    """Quality metrics for an extraction result.

    Provides quantitative measures of extraction quality to help
    assess reliability and identify potential issues.
    """

    # Completeness metrics
    fields_extracted: int = Field(
        description="Number of fields that were successfully extracted"
    )
    fields_total: int = Field(
        description="Total number of fields in the schema"
    )
    completeness_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of extracted fields to total fields (0.0 to 1.0)",
    )

    # Confidence metrics
    average_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Average confidence across all extracted fields",
    )
    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence among extracted fields",
    )
    high_confidence_count: int = Field(
        default=0,
        description="Number of fields with confidence >= 0.8",
    )
    low_confidence_count: int = Field(
        default=0,
        description="Number of fields with confidence < 0.5",
    )

    # Schema adherence
    required_fields_filled: int = Field(
        default=0,
        description="Number of required fields that have values",
    )
    required_fields_total: int = Field(
        default=0,
        description="Total number of required fields in schema",
    )
    required_completeness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of filled required fields (1.0 = all required filled)",
    )

    # Quality indicators
    needs_review: bool = Field(
        default=False,
        description="True if extraction may need human review",
    )
    quality_warnings: list[str] = Field(
        default_factory=list,
        description="List of quality warnings or issues",
    )


def compute_quality_metrics(
    data: BaseModel,
    schema: type[BaseModel],
    field_confidences: dict[str, float] | None = None,
    confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.5,
) -> ExtractionQualityMetrics:
    """Compute quality metrics for an extraction result.

    Args:
        data: The extracted Pydantic model instance.
        schema: The schema class used for extraction.
        field_confidences: Optional dict of field name to confidence score.
        confidence_threshold: Threshold for high confidence (default 0.8).
        low_confidence_threshold: Threshold for low confidence (default 0.5).

    Returns:
        ExtractionQualityMetrics with computed values.
    """
    warnings: list[str] = []

    # Count fields
    total_fields = len(schema.model_fields)
    extracted_count = 0
    required_total = 0
    required_filled = 0

    for field_name, field_info in schema.model_fields.items():
        is_required = field_info.is_required()
        if is_required:
            required_total += 1

        # Check if field has a meaningful value
        value = getattr(data, field_name, None)
        has_value = value is not None and value != "" and value != []

        if has_value:
            extracted_count += 1
            if is_required:
                required_filled += 1
        elif is_required:
            warnings.append(f"Required field '{field_name}' is missing or empty")

    completeness = extracted_count / total_fields if total_fields > 0 else 0.0
    required_completeness = (
        required_filled / required_total if required_total > 0 else 1.0
    )

    # Confidence metrics
    avg_conf: float | None = None
    min_conf: float | None = None
    high_conf_count = 0
    low_conf_count = 0

    if field_confidences:
        conf_values = list(field_confidences.values())
        if conf_values:
            avg_conf = sum(conf_values) / len(conf_values)
            min_conf = min(conf_values)

            for conf in conf_values:
                if conf >= confidence_threshold:
                    high_conf_count += 1
                elif conf < low_confidence_threshold:
                    low_conf_count += 1

            if min_conf < low_confidence_threshold:
                warnings.append(
                    f"Some fields have very low confidence (min: {min_conf:.2f})"
                )

    # Determine if review is needed
    needs_review = (
        required_completeness < 1.0
        or (min_conf is not None and min_conf < low_confidence_threshold)
        or low_conf_count > 0
        or completeness < 0.5
    )

    if completeness < 0.5:
        warnings.append(
            f"Less than half of fields extracted ({completeness:.0%})"
        )

    return ExtractionQualityMetrics(
        fields_extracted=extracted_count,
        fields_total=total_fields,
        completeness_ratio=completeness,
        average_confidence=avg_conf,
        min_confidence=min_conf,
        high_confidence_count=high_conf_count,
        low_confidence_count=low_conf_count,
        required_fields_filled=required_filled,
        required_fields_total=required_total,
        required_completeness=required_completeness,
        needs_review=needs_review,
        quality_warnings=warnings,
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
