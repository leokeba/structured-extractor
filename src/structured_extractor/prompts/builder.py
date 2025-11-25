"""Dynamic prompt builder for extraction."""

import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass
class ExtractionExample:
    """A few-shot example for extraction."""

    input_text: str
    output: dict[str, Any] | BaseModel
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        output_data = (
            self.output.model_dump() if isinstance(self.output, BaseModel) else self.output
        )
        return {
            "input": self.input_text,
            "output": output_data,
            "explanation": self.explanation,
        }


@dataclass
class PromptStrategy:
    """Configuration for prompt generation strategy."""

    # Core settings
    include_field_descriptions: bool = True
    include_examples: bool = True
    include_constraints: bool = True

    # Advanced settings
    include_reasoning: bool = False  # Ask model to explain extraction reasoning
    strict_mode: bool = False  # Emphasize strict adherence to schema
    verbose_types: bool = False  # Include more type detail in schema description

    # Schema presentation
    schema_format: str = "markdown"  # "markdown", "json_schema", "natural"

    # Field hints mode
    hint_style: str = "inline"  # "inline", "section", "examples"


@dataclass
class PromptTemplate:
    """A reusable prompt template for specific extraction scenarios."""

    name: str
    description: str
    system_prompt: str
    extraction_instruction: str
    strategy: PromptStrategy = field(default_factory=PromptStrategy)
    default_field_hints: dict[str, str] = field(default_factory=dict)
    examples: list[ExtractionExample] = field(default_factory=list)


class PromptTemplates:
    """Built-in prompt templates for common extraction scenarios."""

    @staticmethod
    def default() -> PromptTemplate:
        """Default balanced extraction template."""
        return PromptTemplate(
            name="default",
            description="Balanced extraction for general documents",
            system_prompt=(
                "You are an expert data extraction assistant. "
                "Your task is to extract structured information from documents accurately.\n\n"
                "Rules:\n"
                "1. Extract only information that is explicitly present in the document\n"
                "2. Use null/None for fields where information is not found\n"
                "3. Follow the exact format specified in the schema\n"
                "4. Be precise with numbers, dates, and proper nouns\n"
                "5. Do not infer or make up information that is not in the document"
            ),
            extraction_instruction=(
                "Extract the structured data from the document above according to the schema. "
                "Return only the extracted data in the specified format."
            ),
        )

    @staticmethod
    def strict() -> PromptTemplate:
        """Strict extraction with emphasis on accuracy over completeness."""
        return PromptTemplate(
            name="strict",
            description="Strict extraction prioritizing accuracy over completeness",
            system_prompt=(
                "You are a meticulous data extraction specialist. "
                "Your primary goal is ACCURACY - only extract information you are "
                "certain about.\n\n"
                "Critical Rules:\n"
                "1. ONLY extract information that is EXPLICITLY and CLEARLY stated\n"
                "2. If there is ANY ambiguity, use null/None\n"
                "3. Never make assumptions or inferences\n"
                "4. Better to leave a field empty than provide uncertain data\n"
                "5. Match the exact format required by the schema"
            ),
            extraction_instruction=(
                "Carefully extract ONLY the information that is explicitly stated in the document. "
                "If you are uncertain about any field, use null. "
                "Accuracy is more important than completeness."
            ),
            strategy=PromptStrategy(strict_mode=True),
        )

    @staticmethod
    def lenient() -> PromptTemplate:
        """Lenient extraction allowing reasonable inferences."""
        return PromptTemplate(
            name="lenient",
            description="Lenient extraction allowing reasonable inferences",
            system_prompt=(
                "You are a helpful data extraction assistant. "
                "Your goal is to extract as much relevant information as possible.\n\n"
                "Guidelines:\n"
                "1. Extract information that is explicitly present in the document\n"
                "2. Make reasonable inferences when information is implied\n"
                "3. Use context to fill in likely values when appropriate\n"
                "4. Follow the schema format precisely\n"
                "5. Only use null when information truly cannot be determined"
            ),
            extraction_instruction=(
                "Extract the structured data from the document. "
                "You may make reasonable inferences based on context. "
                "Try to fill in as many fields as possible."
            ),
        )

    @staticmethod
    def invoice() -> PromptTemplate:
        """Specialized template for invoice extraction."""
        return PromptTemplate(
            name="invoice",
            description="Specialized extraction for invoices and financial documents",
            system_prompt=(
                "You are an expert financial document processor specializing in invoices. "
                "Your task is to extract structured data from invoices with high precision.\n\n"
                "Rules:\n"
                "1. Pay special attention to monetary values, ensuring correct amounts "
                "and currencies\n"
                "2. Extract dates in ISO format (YYYY-MM-DD) when possible\n"
                "3. Distinguish between invoice number, PO number, and reference numbers\n"
                "4. Extract all line items with quantities, unit prices, and totals\n"
                "5. Identify vendor/supplier and customer information separately\n"
                "6. Note any tax amounts, discounts, or additional charges"
            ),
            extraction_instruction=(
                "Extract all invoice information from the document above. "
                "Pay careful attention to numerical values, dates, and party information. "
                "Ensure all monetary values include their currency if specified."
            ),
            default_field_hints={
                "invoice_number": "Look for 'Invoice #', 'Invoice No.', or similar",
                "date": "The invoice date, not due date or order date",
                "total": "The final total amount including tax",
                "subtotal": "The amount before tax",
            },
        )

    @staticmethod
    def resume() -> PromptTemplate:
        """Specialized template for resume/CV extraction."""
        return PromptTemplate(
            name="resume",
            description="Specialized extraction for resumes and CVs",
            system_prompt=(
                "You are an expert HR document processor specializing in resume analysis. "
                "Your task is to extract structured career information accurately.\n\n"
                "Rules:\n"
                "1. Extract work experience with accurate dates and company names\n"
                "2. List all education with degrees, institutions, and dates\n"
                "3. Identify technical and soft skills separately\n"
                "4. Extract certifications with issuing organizations\n"
                "5. Note any quantifiable achievements with metrics\n"
                "6. Preserve the chronological order of experiences"
            ),
            extraction_instruction=(
                "Extract all professional information from this resume. "
                "Pay attention to employment dates, job titles, and educational qualifications. "
                "List skills and certifications as mentioned."
            ),
            default_field_hints={
                "name": "The candidate's full name",
                "current_role": "Most recent or current job title",
                "years_experience": "Total years of professional experience",
            },
        )

    @staticmethod
    def contract() -> PromptTemplate:
        """Specialized template for contract extraction."""
        return PromptTemplate(
            name="contract",
            description="Specialized extraction for legal contracts",
            system_prompt=(
                "You are an expert legal document analyst specializing in contracts. "
                "Your task is to extract key contractual terms and clauses accurately.\n\n"
                "Rules:\n"
                "1. Identify all parties to the contract with their full legal names\n"
                "2. Extract effective date, term duration, and termination clauses\n"
                "3. Note all monetary values, payment terms, and penalties\n"
                "4. Identify governing law and jurisdiction\n"
                "5. Extract key obligations of each party\n"
                "6. Note any special conditions, warranties, or limitations"
            ),
            extraction_instruction=(
                "Extract all key terms and clauses from this contract. "
                "Ensure party names, dates, and monetary values are accurate. "
                "Note important obligations and conditions."
            ),
            default_field_hints={
                "effective_date": "When the contract becomes effective",
                "parties": "All legal entities party to this agreement",
                "term": "Duration or end date of the agreement",
            },
        )

    @classmethod
    def get_all_templates(cls) -> dict[str, PromptTemplate]:
        """Get all available templates."""
        return {
            "default": cls.default(),
            "strict": cls.strict(),
            "lenient": cls.lenient(),
            "invoice": cls.invoice(),
            "resume": cls.resume(),
            "contract": cls.contract(),
        }


class PromptBuilder:
    """Builds extraction prompts from Pydantic schemas."""

    DEFAULT_SYSTEM_PROMPT = PromptTemplates.default().system_prompt

    def __init__(
        self,
        include_field_descriptions: bool = True,
        include_examples: bool = True,
        strategy: PromptStrategy | None = None,
        template: PromptTemplate | None = None,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            include_field_descriptions: Whether to include field descriptions in prompts
            include_examples: Whether to include examples if provided
            strategy: Optional prompt strategy configuration
            template: Optional prompt template to use
        """
        self.template = template
        self.strategy = strategy or (template.strategy if template else PromptStrategy())
        self.include_field_descriptions = (
            self.strategy.include_field_descriptions
            if strategy
            else include_field_descriptions
        )
        self.include_examples = (
            self.strategy.include_examples if strategy else include_examples
        )

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
        if custom_prompt:
            return custom_prompt
        if self.template:
            return self.template.system_prompt
        return self.DEFAULT_SYSTEM_PROMPT

    def build_extraction_prompt(
        self,
        document: str,
        schema: type[BaseModel],
        field_hints: dict[str, str] | None = None,
        examples: Sequence[dict[str, Any] | ExtractionExample] | None = None,
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

        # Merge field hints with template defaults
        merged_hints = {}
        if self.template and self.template.default_field_hints:
            merged_hints.update(self.template.default_field_hints)
        if field_hints:
            merged_hints.update(field_hints)

        # Schema description
        schema_desc = self._describe_schema(schema, merged_hints or None)
        parts.append(f"## Extraction Schema\n\n{schema_desc}")

        # Reasoning instruction if enabled
        if self.strategy.include_reasoning:
            parts.append(
                "## Reasoning\n\n"
                "Before extracting, briefly analyze the document to identify "
                "where each piece of information can be found."
            )

        # Examples - merge template examples with provided examples
        all_examples = []
        if self.template and self.template.examples:
            all_examples.extend(self.template.examples)
        if examples:
            all_examples.extend(examples)

        if self.include_examples and all_examples:
            examples_text = self._format_examples(all_examples)
            parts.append(f"## Examples\n\n{examples_text}")

        # Document to extract from
        parts.append(f"## Document\n\n{document}")

        # Extraction instruction
        instruction = self._get_extraction_instruction()
        parts.append(f"## Task\n\n{instruction}")

        return "\n\n".join(parts)

    def _get_extraction_instruction(self) -> str:
        """Get the extraction instruction based on template and strategy."""
        if self.template:
            return self.template.extraction_instruction

        if self.strategy.strict_mode:
            return (
                "Extract ONLY information that is explicitly and clearly stated. "
                "If uncertain about any field, use null. "
                "Accuracy is paramount."
            )

        return (
            "Extract the structured data from the document above according to the schema. "
            "Return only the extracted data in the specified format."
        )

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

    def _format_examples(
        self, examples: Sequence[dict[str, Any] | ExtractionExample]
    ) -> str:
        """Format few-shot examples.

        Args:
            examples: List of example extractions

        Returns:
            Formatted examples string
        """
        formatted: list[str] = []

        for i, example in enumerate(examples, 1):
            ex_dict = (
                example.to_dict() if isinstance(example, ExtractionExample) else example
            )

            input_text = ex_dict.get("input", "")
            output_data = ex_dict.get("output", {})
            explanation = ex_dict.get("explanation")

            parts = [f"**Example {i}:**"]
            parts.append(f"\nInput:\n```\n{input_text}\n```")
            parts.append(f"\nOutput:\n```json\n{output_data}\n```")

            if explanation:
                parts.append(f"\nExplanation: {explanation}")

            formatted.append("".join(parts))

        return "\n\n".join(formatted)


def _is_pydantic_undefined(value: Any) -> bool:
    """Check if a value is Pydantic's undefined sentinel.

    Args:
        value: The value to check

    Returns:
        True if the value is PydanticUndefined
    """
    return type(value).__name__ == "PydanticUndefinedType"
