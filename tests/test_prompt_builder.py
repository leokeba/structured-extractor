"""Tests for the prompt builder."""

from pydantic import BaseModel, Field

from structured_extractor.prompts.builder import PromptBuilder


class SimpleModel(BaseModel):
    """A simple test model."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")


class NestedAddress(BaseModel):
    """Address model."""

    street: str
    city: str
    country: str = Field(default="USA")


class PersonWithAddress(BaseModel):
    """Person with nested address."""

    name: str
    email: str | None = Field(default=None, description="Email address")
    address: NestedAddress


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_build_system_prompt_default(self) -> None:
        """Test default system prompt generation."""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()

        assert "data extraction" in prompt.lower()
        assert "precise" in prompt.lower() or "accurately" in prompt.lower()

    def test_build_system_prompt_custom(self) -> None:
        """Test custom system prompt."""
        builder = PromptBuilder()
        custom = "You are a specialized invoice extractor."
        prompt = builder.build_system_prompt(custom_prompt=custom)

        assert prompt == custom

    def test_describe_schema_simple(self) -> None:
        """Test schema description for simple model."""
        builder = PromptBuilder()
        description = builder._describe_schema(SimpleModel)

        assert "SimpleModel" in description
        assert "name" in description
        assert "age" in description
        assert "person's name" in description.lower()
        assert "required" in description.lower()

    def test_describe_schema_with_hints(self) -> None:
        """Test schema description with field hints."""
        builder = PromptBuilder()
        hints = {"name": "Look for the full legal name"}
        description = builder._describe_schema(SimpleModel, field_hints=hints)

        assert "full legal name" in description.lower()

    def test_describe_schema_nested(self) -> None:
        """Test schema description for nested models."""
        builder = PromptBuilder()
        description = builder._describe_schema(PersonWithAddress)

        assert "PersonWithAddress" in description
        assert "address" in description
        assert "NestedAddress" in description

    def test_describe_schema_optional_field(self) -> None:
        """Test schema description includes optional fields."""
        builder = PromptBuilder()
        description = builder._describe_schema(PersonWithAddress)

        assert "email" in description
        assert "optional" in description.lower()

    def test_build_extraction_prompt(self) -> None:
        """Test full extraction prompt generation."""
        builder = PromptBuilder()
        document = "John Smith is 30 years old."

        prompt = builder.build_extraction_prompt(
            document=document,
            schema=SimpleModel,
        )

        assert "## Extraction Schema" in prompt
        assert "## Document" in prompt
        assert "## Task" in prompt
        assert document in prompt
        assert "SimpleModel" in prompt

    def test_build_extraction_prompt_with_examples(self) -> None:
        """Test extraction prompt with few-shot examples."""
        builder = PromptBuilder()
        document = "Jane Doe is 25 years old."
        examples = [
            {
                "input": "Bob is 40.",
                "output": {"name": "Bob", "age": 40},
            }
        ]

        prompt = builder.build_extraction_prompt(
            document=document,
            schema=SimpleModel,
            examples=examples,
        )

        assert "## Examples" in prompt
        assert "Bob" in prompt

    def test_format_type_basic(self) -> None:
        """Test type formatting for basic types."""
        builder = PromptBuilder()

        assert builder._format_type(str) == "str"
        assert builder._format_type(int) == "int"
        assert builder._format_type(float) == "float"
        assert builder._format_type(bool) == "bool"

    def test_format_type_list(self) -> None:
        """Test type formatting for list types."""
        builder = PromptBuilder()

        assert builder._format_type(list[str]) == "list[str]"
        assert builder._format_type(list[int]) == "list[int]"

    def test_format_type_nested_model(self) -> None:
        """Test type formatting for Pydantic models."""
        builder = PromptBuilder()

        assert builder._format_type(NestedAddress) == "NestedAddress"
