"""Dynamic prompt builder for extraction."""

import types
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class PromptBuilder:
    """Builds extraction prompts from Pydantic schemas."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert data extraction assistant. "
        "Your task is to extract structured information from documents accurately.\n\n"
        "Rules:\n"
        "1. Extract only information that is explicitly present in the document\n"
        "2. Use null/None for fields where information is not found\n"
        "3. Follow the exact format specified in the schema\n"
        "4. Be precise with numbers, dates, and proper nouns\n"
        "5. Do not infer or make up information that is not in the document"
    )

    def __init__(
        self,
        include_field_descriptions: bool = True,
        include_examples: bool = True,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            include_field_descriptions: Whether to include field descriptions in prompts
            include_examples: Whether to include examples if provided
        """
        self.include_field_descriptions = include_field_descriptions
        self.include_examples = include_examples

    def build_system_prompt(
        self,
        custom_prompt: str | None = None,
    ) -> str:
        """Build the system prompt.

        Args:
            custom_prompt: Optional custom system prompt to use instead of default

        Returns:
            The system prompt string
        """
        return custom_prompt or self.DEFAULT_SYSTEM_PROMPT

    def build_extraction_prompt(
        self,
        document: str,
        schema: type[BaseModel],
        field_hints: dict[str, str] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build the user prompt for extraction.

        Args:
            document: The document text to extract from
            schema: The Pydantic model defining the extraction schema
            field_hints: Optional hints for specific fields
            examples: Optional few-shot examples

        Returns:
            The formatted extraction prompt
        """
        parts: list[str] = []

        # Schema description
        schema_desc = self._describe_schema(schema, field_hints)
        parts.append(f"## Extraction Schema\n\n{schema_desc}")

        # Examples if provided
        if self.include_examples and examples:
            examples_text = self._format_examples(examples)
            parts.append(f"## Examples\n\n{examples_text}")

        # Document to extract from
        parts.append(f"## Document\n\n{document}")

        # Extraction instruction
        parts.append(
            "## Task\n\n"
            "Extract the structured data from the document above according to the schema. "
            "Return only the extracted data in the specified format."
        )

        return "\n\n".join(parts)

    def _describe_schema(
        self,
        schema: type[BaseModel],
        field_hints: dict[str, str] | None = None,
        indent: int = 0,
        described_models: set[str] | None = None,
    ) -> str:
        """Generate a human-readable description of the schema.

        Args:
            schema: The Pydantic model to describe
            field_hints: Optional hints for specific fields
            indent: Indentation level for nested schemas
            described_models: Set of already described model names to avoid recursion

        Returns:
            A formatted schema description
        """
        field_hints = field_hints or {}
        described_models = described_models or set()
        lines: list[str] = []
        indent_str = "  " * indent

        # Schema name and docstring
        schema_name = schema.__name__
        lines.append(f"{indent_str}**{schema_name}**")

        if schema.__doc__ and indent == 0:
            lines.append(f"\n{indent_str}{schema.__doc__.strip()}")

        lines.append(f"\n{indent_str}Fields:")

        # Track nested models to describe later
        nested_models: list[type[BaseModel]] = []

        # Describe each field
        for field_name, field_info in schema.model_fields.items():
            field_desc = self._describe_field(
                field_name, field_info, field_hints.get(field_name), indent + 1
            )
            lines.append(field_desc)

            # Collect nested models
            nested = self._get_nested_models(field_info.annotation)
            for model in nested:
                if model.__name__ not in described_models:
                    nested_models.append(model)
                    described_models.add(model.__name__)

        # Describe nested models
        if nested_models and indent == 0:
            lines.append(f"\n{indent_str}### Nested Types")
            described_models.add(schema_name)
            for nested_model in nested_models:
                lines.append("")
                nested_desc = self._describe_schema(
                    nested_model, field_hints, indent=0, described_models=described_models
                )
                lines.append(nested_desc)

        return "\n".join(lines)

    def _get_nested_models(self, annotation: Any) -> list[type[BaseModel]]:
        """Extract nested Pydantic models from a type annotation.

        Args:
            annotation: The type annotation to inspect

        Returns:
            List of nested BaseModel subclasses
        """
        models: list[type[BaseModel]] = []

        if annotation is None:
            return models

        # Direct BaseModel subclass
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            models.append(annotation)
            return models

        # Check generic types (List, Optional, Union, etc.)
        origin = get_origin(annotation)
        if origin is not None:
            for arg in get_args(annotation):
                models.extend(self._get_nested_models(arg))

        return models

    def _describe_field(
        self,
        name: str,
        field_info: FieldInfo,
        hint: str | None = None,
        indent: int = 0,
    ) -> str:
        """Describe a single field with full type and constraint information.

        Args:
            name: Field name
            field_info: Pydantic FieldInfo object
            hint: Optional extraction hint
            indent: Indentation level

        Returns:
            Formatted field description
        """
        indent_str = "  " * indent

        # Get type annotation
        type_str = self._format_type(field_info.annotation)

        # Check if required
        is_required = field_info.is_required()
        required_str = "required" if is_required else "optional"

        # Build description
        parts = [f"{indent_str}- **{name}** ({type_str}, {required_str})"]

        # Add field description if available
        if self.include_field_descriptions and field_info.description:
            parts.append(f": {field_info.description}")

        # Add extraction hint if provided
        if hint:
            parts.append(f" [Hint: {hint}]")

        # Add default if present and not PydanticUndefined
        if field_info.default is not None and not _is_pydantic_undefined(field_info.default):
            parts.append(f" [Default: {field_info.default}]")

        # Add field constraints from metadata
        constraints = self._get_field_constraints(field_info)
        if constraints:
            parts.append(f" {constraints}")

        # Add enum values if applicable
        enum_values = self._get_enum_values(field_info.annotation)
        if enum_values:
            parts.append(f" [Values: {enum_values}]")

        return "".join(parts)

    def _get_field_constraints(self, field_info: FieldInfo) -> str:
        """Extract field constraints from Pydantic metadata.

        Args:
            field_info: Pydantic FieldInfo object

        Returns:
            Formatted constraints string or empty string
        """
        constraints: list[str] = []

        # Check for common constraints in metadata
        for meta in field_info.metadata:
            meta_type = type(meta).__name__

            if meta_type == "Gt":
                constraints.append(f">{getattr(meta, 'gt', '?')}")
            elif meta_type == "Ge":
                constraints.append(f">={getattr(meta, 'ge', '?')}")
            elif meta_type == "Lt":
                constraints.append(f"<{getattr(meta, 'lt', '?')}")
            elif meta_type == "Le":
                constraints.append(f"<={getattr(meta, 'le', '?')}")
            elif meta_type == "MinLen":
                constraints.append(f"min_length={getattr(meta, 'min_length', '?')}")
            elif meta_type == "MaxLen":
                constraints.append(f"max_length={getattr(meta, 'max_length', '?')}")
            elif hasattr(meta, "pattern"):
                constraints.append(f"pattern={meta.pattern}")

        if constraints:
            return f"[Constraints: {', '.join(constraints)}]"
        return ""

    def _get_enum_values(self, annotation: Any) -> str:
        """Get enum values if the annotation is an Enum type.

        Args:
            annotation: The type annotation

        Returns:
            Formatted enum values or empty string
        """
        if annotation is None:
            return ""

        # Handle direct Enum
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            values = [str(e.value) for e in annotation]
            return ", ".join(values)

        # Handle Optional[Enum] or Union with Enum
        origin = get_origin(annotation)
        if origin is Union or origin is types.UnionType:
            for arg in get_args(annotation):
                if isinstance(arg, type) and issubclass(arg, Enum):
                    values = [str(e.value) for e in arg]
                    return ", ".join(values)

        return ""

    def _format_type(self, annotation: Any) -> str:
        """Format a type annotation as a readable string.

        Args:
            annotation: The type annotation

        Returns:
            Human-readable type string
        """
        if annotation is None:
            return "any"

        origin = get_origin(annotation)

        # Handle Optional, List, Dict, Union, etc.
        if origin is not None:
            args = get_args(annotation)

            # Handle Union types (including Optional via | None)
            if origin is Union or origin is types.UnionType:
                # Filter out NoneType for Optional display
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1 and type(None) in args:
                    return f"{self._format_type(non_none_args[0])} | null"
                return " | ".join(self._format_type(a) for a in args)

            # Handle List
            if origin is list:
                inner = self._format_type(args[0]) if args else "any"
                return f"list[{inner}]"

            # Handle Dict
            if origin is dict:
                key_type = self._format_type(args[0]) if args else "any"
                val_type = self._format_type(args[1]) if len(args) > 1 else "any"
                return f"dict[{key_type}, {val_type}]"

            # Handle tuple
            if origin is tuple:
                if args:
                    inner_types = ", ".join(self._format_type(a) for a in args)
                    return f"tuple[{inner_types}]"
                return "tuple"

            # Handle set
            if origin is set:
                inner = self._format_type(args[0]) if args else "any"
                return f"set[{inner}]"

        # Handle Enum types
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return f"enum({annotation.__name__})"

        # Handle Pydantic models (nested schemas)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.__name__

        # Handle Literal types
        if str(origin) == "typing.Literal":
            values = ", ".join(repr(v) for v in get_args(annotation))
            return f"literal[{values}]"

        # Handle basic types
        if hasattr(annotation, "__name__"):
            return annotation.__name__

        return str(annotation)

    def _format_examples(self, examples: list[dict[str, Any]]) -> str:
        """Format few-shot examples.

        Args:
            examples: List of example extractions

        Returns:
            Formatted examples string
        """
        formatted: list[str] = []

        for i, example in enumerate(examples, 1):
            input_text = example.get("input", "")
            output_data = example.get("output", {})
            formatted.append(f"Example {i}:\nInput: {input_text}\nOutput: {output_data}")

        return "\n\n".join(formatted)


def _is_pydantic_undefined(value: Any) -> bool:
    """Check if a value is Pydantic's undefined sentinel.

    Args:
        value: The value to check

    Returns:
        True if the value is PydanticUndefined
    """
    return type(value).__name__ == "PydanticUndefinedType"
