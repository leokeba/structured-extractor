"""Configuration classes for extraction."""

from typing import Any

from pydantic import BaseModel, Field


class FieldConfig(BaseModel):
    """Configuration for a specific field extraction."""

    extraction_hint: str | None = Field(
        default=None,
        description="Additional hint for extracting this field",
    )
    required: bool = Field(
        default=True,
        description="Whether this field is required",
    )
    examples: list[Any] | None = Field(
        default=None,
        description="Example values for few-shot learning",
    )
    fallback_value: Any | None = Field(
        default=None,
        description="Default value if extraction fails",
    )


class ExtractionConfig(BaseModel):
    """Configuration for the extraction process."""

    # Confidence settings
    include_confidence: bool = Field(
        default=False,
        description="Whether to include confidence scores",
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for valid extraction",
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts",
    )
    retry_on_validation_error: bool = Field(
        default=True,
        description="Whether to retry on Pydantic validation errors",
    )

    # LLM settings
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for extraction (lower = more deterministic)",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens for LLM response",
    )

    # Prompt settings
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt override",
    )
    include_field_descriptions: bool = Field(
        default=True,
        description="Include field descriptions in the prompt",
    )
