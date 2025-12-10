"""Result types for extraction outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


class FieldResult(BaseModel):
    """Result for a single extracted field."""

    name: str = Field(description="Field name")
    value: Any = Field(description="Extracted value")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for this field (0-1)",
    )
    source_text: str | None = Field(
        default=None,
        description="Original text snippet this was extracted from",
    )


class ExtractionResult(BaseModel, Generic[T]):
    """Result of a document extraction operation.

    Generic type T is the Pydantic model used for extraction.
    """

    # Core result
    data: T | None = Field(
        default=None,
        description="The extracted structured data (None when extraction fails)",
    )
    success: bool = Field(default=True, description="Whether extraction succeeded")

    # Quality metrics
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score (0-1)",
    )
    field_confidences: dict[str, float] | None = Field(
        default=None,
        description="Per-field confidence scores",
    )
    field_results: list[FieldResult] | None = Field(
        default=None,
        description="Detailed results for each field",
    )
    low_confidence_fields: list[str] | None = Field(
        default=None,
        description="List of field names with confidence below threshold",
    )
    quality_metrics: Any = Field(
        default=None,
        description="Detailed quality metrics for the extraction (ExtractionQualityMetrics)",
    )

    # Metadata
    model_used: str | None = Field(
        default=None,
        description="LLM model used for extraction",
    )
    cached: bool = Field(
        default=False,
        description="Whether result was served from cache",
    )
    tokens_used: int | None = Field(
        default=None,
        description="Total tokens used for extraction",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Estimated cost in USD",
    )

    # Error handling
    error: str | None = Field(
        default=None,
        description="Error message if extraction failed",
    )
    raw_response: str | None = Field(
        default=None,
        description="Raw LLM response for debugging",
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _validate_data_presence(self) -> ExtractionResult[T]:
        """Ensure data is present when extraction succeeds."""
        if self.success and self.data is None:
            raise ValueError("data must be provided when success is True")
        return self
