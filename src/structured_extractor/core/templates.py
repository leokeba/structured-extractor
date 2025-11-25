"""Document templates for reusable extraction configurations.

This module provides:
- DocumentTemplate: Base class for extraction templates with serialization
- TemplateRegistry: Registry for managing and discovering templates

For built-in schemas and templates, see:
- structured_extractor.schemas: Pre-built Pydantic models for common documents
- structured_extractor.templates: Pre-configured templates with optimized prompts
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TemplateValidationError(Exception):
    """Raised when template validation fails."""

    pass


class DocumentTemplate(BaseModel):
    """A reusable template for document extraction.

    Templates combine a schema with extraction configuration,
    making it easy to reuse extraction settings across documents.

    Supports:
    - Serialization to/from JSON and YAML
    - Template inheritance via parent_template
    - Validation of template configuration
    - Field-specific extraction hints

    Example:
        ```python
        from pydantic import BaseModel, Field
        from structured_extractor import DocumentTemplate

        class Invoice(BaseModel):
            invoice_number: str
            total: float

        template = DocumentTemplate(
            name="invoice",
            schema_class=Invoice,
            description="Extract invoice data",
            field_hints={"total": "The final amount including tax"},
        )

        # Save to file
        template.to_json("templates/invoice.json")

        # Load from file
        loaded = DocumentTemplate.from_json("templates/invoice.json", Invoice)
        ```
    """

    name: str = Field(description="Template name for identification")
    schema_class: type[BaseModel] = Field(description="Pydantic model for extraction")
    description: str | None = Field(
        default=None,
        description="Human-readable description of what this template extracts",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for this template",
    )
    field_configs: dict[str, Any] | None = Field(
        default=None,
        description="Field-specific extraction configurations (deprecated, use field_hints)",
    )
    field_hints: dict[str, str] | None = Field(
        default=None,
        description="Hints for extracting specific fields",
    )
    examples: list[dict[str, Any]] | None = Field(
        default=None,
        description="Few-shot examples for improved extraction",
    )
    version: str = Field(
        default="1.0",
        description="Template version for compatibility tracking",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tags for categorizing and searching templates",
    )
    parent_template: str | None = Field(
        default=None,
        description="Name of parent template to inherit from",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate template name is non-empty and valid."""
        if not v or not v.strip():
            raise ValueError("Template name cannot be empty")
        # Allow alphanumeric, underscores, hyphens
        cleaned = v.strip().lower().replace(" ", "_")
        return cleaned

    def validate_template(self) -> list[str]:
        """Validate the template configuration.

        Returns:
            List of validation warnings (empty if valid).

        Raises:
            TemplateValidationError: If template has critical issues.
        """
        warnings: list[str] = []

        # Check schema has fields
        if not self.schema_class.model_fields:
            raise TemplateValidationError(
                f"Template '{self.name}' schema has no fields defined"
            )

        # Check field_hints reference valid fields
        if self.field_hints:
            schema_fields = set(self.schema_class.model_fields.keys())
            for field_name in self.field_hints:
                if field_name not in schema_fields:
                    warnings.append(
                        f"Field hint '{field_name}' does not match any schema field"
                    )

        # Check examples have valid structure
        if self.examples:
            for i, example in enumerate(self.examples):
                if "input" not in example and "input_text" not in example:
                    warnings.append(f"Example {i} missing 'input' or 'input_text' field")
                if "output" not in example:
                    warnings.append(f"Example {i} missing 'output' field")

        return warnings

    def to_dict(self, exclude_schema: bool = False) -> dict[str, Any]:
        """Convert template to dictionary for serialization.

        Args:
            exclude_schema: If True, only include schema name, not the class.

        Returns:
            Dictionary representation of the template.
        """
        data: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }

        if exclude_schema:
            data["schema_name"] = self.schema_class.__name__
        else:
            # Include schema as JSON schema for portability
            data["schema"] = self.schema_class.model_json_schema()
            data["schema_name"] = self.schema_class.__name__

        if self.description:
            data["description"] = self.description
        if self.system_prompt:
            data["system_prompt"] = self.system_prompt
        if self.field_hints:
            data["field_hints"] = self.field_hints
        if self.field_configs:
            data["field_configs"] = self.field_configs
        if self.examples:
            data["examples"] = self.examples
        if self.tags:
            data["tags"] = self.tags
        if self.parent_template:
            data["parent_template"] = self.parent_template

        return data

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize template to JSON.

        Args:
            path: Optional file path to write to.
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        data = self.to_dict(exclude_schema=True)
        json_str = json.dumps(data, indent=indent)

        if path:
            Path(path).write_text(json_str)

        return json_str

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize template to YAML.

        Args:
            path: Optional file path to write to.

        Returns:
            YAML string representation.
        """
        data = self.to_dict(exclude_schema=True)
        yaml_str: str = yaml.dump(data, default_flow_style=False, sort_keys=False)

        if path:
            Path(path).write_text(yaml_str)

        return yaml_str

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        schema_class: type[BaseModel],
    ) -> DocumentTemplate:
        """Create template from dictionary.

        Args:
            data: Dictionary with template configuration.
            schema_class: The Pydantic model to use for extraction.

        Returns:
            DocumentTemplate instance.
        """
        return cls(
            name=data.get("name", "unnamed"),
            schema_class=schema_class,
            description=data.get("description"),
            system_prompt=data.get("system_prompt"),
            field_hints=data.get("field_hints"),
            field_configs=data.get("field_configs"),
            examples=data.get("examples"),
            version=data.get("version", "1.0"),
            tags=data.get("tags"),
            parent_template=data.get("parent_template"),
        )

    @classmethod
    def from_json(
        cls,
        source: str | Path,
        schema_class: type[BaseModel],
    ) -> DocumentTemplate:
        """Load template from JSON file or string.

        Args:
            source: JSON string or path to JSON file.
            schema_class: The Pydantic model to use for extraction.

        Returns:
            DocumentTemplate instance.
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and Path(source).exists()
        ):
            content = Path(source).read_text()
        else:
            content = source

        data = json.loads(content)
        return cls.from_dict(data, schema_class)

    @classmethod
    def from_yaml(
        cls,
        source: str | Path,
        schema_class: type[BaseModel],
    ) -> DocumentTemplate:
        """Load template from YAML file or string.

        Args:
            source: YAML string or path to YAML file.
            schema_class: The Pydantic model to use for extraction.

        Returns:
            DocumentTemplate instance.
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and Path(source).exists()
        ):
            content = Path(source).read_text()
        else:
            content = source

        data = yaml.safe_load(content)
        return cls.from_dict(data, schema_class)

    def merge_with_parent(self, parent: DocumentTemplate) -> DocumentTemplate:
        """Merge this template with a parent template.

        Child template values override parent values. Field hints and
        examples are merged (child takes precedence for conflicts).

        Args:
            parent: Parent template to inherit from.

        Returns:
            New template with merged configuration.
        """
        # Merge field hints (child overrides parent)
        merged_hints = {**(parent.field_hints or {}), **(self.field_hints or {})}

        # Merge examples (child examples come after parent)
        merged_examples = (parent.examples or []) + (self.examples or [])

        # Merge tags
        merged_tags = list(set((parent.tags or []) + (self.tags or [])))

        return DocumentTemplate(
            name=self.name,
            schema_class=self.schema_class,
            description=self.description or parent.description,
            system_prompt=self.system_prompt or parent.system_prompt,
            field_hints=merged_hints if merged_hints else None,
            field_configs=self.field_configs or parent.field_configs,
            examples=merged_examples if merged_examples else None,
            version=self.version,
            tags=merged_tags if merged_tags else None,
            parent_template=parent.name,
        )


class TemplateRegistry:
    """Registry for managing document templates.

    Provides template discovery, registration, and inheritance resolution.

    Example:
        ```python
        registry = TemplateRegistry()

        # Register templates
        registry.register(invoice_template)
        registry.register(receipt_template)

        # Get template
        template = registry.get("invoice")

        # List all templates
        for name in registry.list_templates():
            print(name)

        # Search by tags
        financial_templates = registry.search_by_tags(["financial"])
        ```
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._templates: dict[str, DocumentTemplate] = {}

    def register(
        self,
        template: DocumentTemplate,
        overwrite: bool = False,
    ) -> None:
        """Register a template.

        Args:
            template: Template to register.
            overwrite: Whether to overwrite existing template.

        Raises:
            ValueError: If template with same name exists and overwrite=False.
        """
        if template.name in self._templates and not overwrite:
            raise ValueError(
                f"Template '{template.name}' already registered. "
                "Use overwrite=True to replace."
            )

        # Validate template
        template.validate_template()

        # Resolve parent if specified
        if template.parent_template:
            parent = self._templates.get(template.parent_template)
            if parent:
                template = template.merge_with_parent(parent)

        self._templates[template.name] = template

    def get(self, name: str) -> DocumentTemplate | None:
        """Get template by name.

        Args:
            name: Template name.

        Returns:
            Template or None if not found.
        """
        return self._templates.get(name)

    def get_or_raise(self, name: str) -> DocumentTemplate:
        """Get template by name or raise error.

        Args:
            name: Template name.

        Returns:
            Template.

        Raises:
            KeyError: If template not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found in registry")
        return self._templates[name]

    def unregister(self, name: str) -> bool:
        """Remove template from registry.

        Args:
            name: Template name.

        Returns:
            True if template was removed, False if not found.
        """
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def list_templates(self) -> list[str]:
        """List all registered template names.

        Returns:
            List of template names.
        """
        return list(self._templates.keys())

    def search_by_tags(self, tags: list[str]) -> list[DocumentTemplate]:
        """Find templates with matching tags.

        Args:
            tags: Tags to search for (any match).

        Returns:
            List of matching templates.
        """
        matching = []
        tag_set = set(tags)

        for template in self._templates.values():
            if template.tags and tag_set.intersection(template.tags):
                matching.append(template)

        return matching

    def search_by_description(self, query: str) -> list[DocumentTemplate]:
        """Find templates with description containing query.

        Args:
            query: Search string.

        Returns:
            List of matching templates.
        """
        query_lower = query.lower()
        return [
            t
            for t in self._templates.values()
            if t.description and query_lower in t.description.lower()
        ]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Export all templates as dictionary.

        Returns:
            Dict mapping template names to their configurations.
        """
        return {
            name: template.to_dict(exclude_schema=True)
            for name, template in self._templates.items()
        }

    def __len__(self) -> int:
        """Return number of registered templates."""
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        """Check if template is registered."""
        return name in self._templates

    def __iter__(self) -> Iterator[str]:
        """Iterate over template names."""
        return iter(self._templates)
