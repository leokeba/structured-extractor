"""Document templates for reusable extraction configurations."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentTemplate(BaseModel):
    """A reusable template for document extraction.

    Templates combine a schema with extraction configuration,
    making it easy to reuse extraction settings across documents.
    """

    name: str = Field(description="Template name for identification")
    schema_class: type[BaseModel] = Field(description="Pydantic model for extraction")
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for this template",
    )
    field_configs: dict[str, Any] | None = Field(
        default=None,
        description="Field-specific extraction configurations",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of what this template extracts",
    )
    examples: list[dict[str, Any]] | None = Field(
        default=None,
        description="Few-shot examples for improved extraction",
    )

    model_config = {"arbitrary_types_allowed": True}
