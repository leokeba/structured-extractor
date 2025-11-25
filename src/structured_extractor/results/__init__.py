"""Result types for extraction outputs."""

from structured_extractor.results.confidence import (
    ConfidenceAssessment,
    FieldConfidence,
    build_confidence_schema,
    compute_overall_confidence,
    identify_low_confidence_fields,
)
from structured_extractor.results.types import ExtractionResult, FieldResult

__all__ = [
    "ExtractionResult",
    "FieldResult",
    "ConfidenceAssessment",
    "FieldConfidence",
    "build_confidence_schema",
    "compute_overall_confidence",
    "identify_low_confidence_fields",
]
