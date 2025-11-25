"""Dynamic prompt builder for extraction."""

from typing import Any, get_args, get_origin

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
    ) -> str:
        """Generate a human-readable description of the schema.

        Args:
            schema: The Pydantic model to describe
            field_hints: Optional hints for specific fields

        Returns:
            A formatted schema description
        """
        field_hints = field_hints or {}
        lines: list[str] = []

        # Schema name and docstring
        schema_name = schema.__name__
        lines.append(f"**{schema_name}**")

        if schema.__doc__:
            lines.append(f"\n{schema.__doc__.strip()}")

        lines.append("\nFields:")

        # Describe each field
        for field_name, field_info in schema.model_fields.items():
            field_desc = self._describe_field(field_name, field_info, field_hints.get(field_name))
            lines.append(field_desc)

        return "\n".join(lines)

    def _describe_field(
        self,
        name: str,
        field_info: FieldInfo,
        hint: str | None = None,
    ) -> str:
        """Describe a single field.

        Args:
            name: Field name
            field_info: Pydantic FieldInfo object
            hint: Optional extraction hint

        Returns:
            Formatted field description
        """
        # Get type annotation
        type_str = self._format_type(field_info.annotation)

        # Check if required
        is_required = field_info.is_required()
        required_str = "required" if is_required else "optional"

        # Build description
        parts = [f"- **{name}** ({type_str}, {required_str})"]

        # Add field description if available
        if self.include_field_descriptions and field_info.description:
            parts.append(f": {field_info.description}")

        # Add extraction hint if provided
        if hint:
            parts.append(f" [Hint: {hint}]")

        # Add default if present
        if field_info.default is not None:
            parts.append(f" [Default: {field_info.default}]")

        return "".join(parts)

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

        # Handle Optional, List, Dict, etc.
        if origin is not None:
            args = get_args(annotation)

            # Handle Union types (including Optional)
            if origin.__name__ == "UnionType" or str(origin) == "typing.Union":
                # Filter out NoneType for Optional
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

        # Handle Pydantic models (nested schemas)
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.__name__

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
