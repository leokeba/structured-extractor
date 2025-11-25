"""Tests for advanced schema support (Phase 2)."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from structured_extractor.prompts.builder import PromptBuilder


# Test models for advanced schema features
class Priority(str, Enum):
    """Priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    """Task status."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Tag(BaseModel):
    """A simple tag."""

    name: str
    color: str | None = None


class Address(BaseModel):
    """Postal address."""

    street: str
    city: str
    state: str | None = None
    postal_code: str
    country: str = Field(default="USA")


class Person(BaseModel):
    """Person with address."""

    name: str
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str | None = Field(default=None, description="Email address")
    address: Address


class Task(BaseModel):
    """Task with enum fields."""

    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    priority: Priority
    status: Status = Field(default=Status.TODO)
    tags: list[Tag] = Field(default_factory=list)


class Invoice(BaseModel):
    """Invoice with nested lists."""

    invoice_number: str
    line_items: list[dict[str, float]]
    total: float = Field(ge=0)


class UnionModel(BaseModel):
    """Model with union types."""

    value: int | str
    optional_value: float | None = None


class LiteralModel(BaseModel):
    """Model with literal types."""

    size: Literal["small", "medium", "large"]
    color: Literal["red", "green", "blue"] | None = None


class DeepNested(BaseModel):
    """Deeply nested model."""

    person: Person
    tasks: list[Task]


class TestNestedModelSupport:
    """Tests for nested Pydantic model support."""

    def test_nested_model_description(self) -> None:
        """Test that nested models are properly described."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Person)

        # Main model fields
        assert "Person" in desc
        assert "name" in desc
        assert "age" in desc
        assert "address" in desc

        # Should include nested type section
        assert "Nested Types" in desc
        assert "Address" in desc

    def test_deeply_nested_models(self) -> None:
        """Test deeply nested model descriptions."""
        builder = PromptBuilder()
        desc = builder._describe_schema(DeepNested)

        # All models should be described
        assert "DeepNested" in desc
        assert "Person" in desc
        assert "Address" in desc
        assert "Task" in desc
        assert "Tag" in desc

    def test_nested_model_type_formatting(self) -> None:
        """Test type formatting for nested models."""
        builder = PromptBuilder()

        assert builder._format_type(Address) == "Address"
        assert builder._format_type(Person) == "Person"


class TestListArraySupport:
    """Tests for list/array field extraction."""

    def test_list_of_primitives(self) -> None:
        """Test list[str], list[int] formatting."""
        builder = PromptBuilder()

        assert builder._format_type(list[str]) == "list[str]"
        assert builder._format_type(list[int]) == "list[int]"
        assert builder._format_type(list[float]) == "list[float]"

    def test_list_of_models(self) -> None:
        """Test list[Model] formatting."""
        builder = PromptBuilder()

        assert builder._format_type(list[Tag]) == "list[Tag]"
        assert builder._format_type(list[Task]) == "list[Task]"

    def test_list_field_in_schema(self) -> None:
        """Test that list fields are properly described."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Task)

        assert "tags" in desc
        assert "list[Tag]" in desc

    def test_nested_list_extraction(self) -> None:
        """Test models with list of nested models."""
        builder = PromptBuilder()
        desc = builder._describe_schema(DeepNested)

        assert "tasks" in desc
        assert "list[Task]" in desc


class TestOptionalFieldsAndDefaults:
    """Tests for Optional[T], | None, and default values."""

    def test_optional_type_formatting(self) -> None:
        """Test Optional type formatting with | null."""
        builder = PromptBuilder()

        assert builder._format_type(str | None) == "str | null"
        assert builder._format_type(int | None) == "int | null"
        assert builder._format_type(Address | None) == "Address | null"

    def test_optional_field_marked(self) -> None:
        """Test that optional fields are marked as optional."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Person)

        # email is optional
        assert "email" in desc
        assert "optional" in desc

    def test_default_values_shown(self) -> None:
        """Test that default values are displayed."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Address)

        assert "country" in desc
        assert "Default: USA" in desc

    def test_required_vs_optional_distinction(self) -> None:
        """Test clear distinction between required and optional."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Task)

        # title is required
        assert "title" in desc
        # description is optional
        lines = desc.split("\n")
        title_line = next(line for line in lines if "title" in line)
        desc_line = next(line for line in lines if "description" in line)

        assert "required" in title_line
        assert "optional" in desc_line


class TestUnionTypeSupport:
    """Tests for Union[A, B] and A | B type annotations."""

    def test_union_type_formatting(self) -> None:
        """Test union type formatting."""
        builder = PromptBuilder()

        assert builder._format_type(int | str) == "int | str"

    def test_union_with_none(self) -> None:
        """Test union with None (Optional)."""
        builder = PromptBuilder()

        # Should show as optional with null
        result = builder._format_type(float | None)
        assert "float" in result
        assert "null" in result

    def test_union_field_in_schema(self) -> None:
        """Test union fields in schema description."""
        builder = PromptBuilder()
        desc = builder._describe_schema(UnionModel)

        assert "value" in desc
        assert "int" in desc
        assert "str" in desc


class TestEnumSupport:
    """Tests for Enum type extraction."""

    def test_enum_type_formatting(self) -> None:
        """Test enum type formatting."""
        builder = PromptBuilder()

        result = builder._format_type(Priority)
        assert "enum" in result
        assert "Priority" in result

    def test_enum_values_shown(self) -> None:
        """Test that enum values are displayed."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Task)

        # Should show priority enum values
        assert "priority" in desc
        assert "Values:" in desc
        # Check for actual enum values
        assert "low" in desc or "medium" in desc or "high" in desc

    def test_get_enum_values(self) -> None:
        """Test enum value extraction."""
        builder = PromptBuilder()

        values = builder._get_enum_values(Priority)
        assert "low" in values
        assert "medium" in values
        assert "high" in values
        assert "critical" in values

    def test_optional_enum(self) -> None:
        """Test Optional[Enum] handling."""
        builder = PromptBuilder()

        # Should still extract enum values for Optional[Enum]
        values = builder._get_enum_values(Priority | None)
        assert "low" in values


class TestLiteralSupport:
    """Tests for Literal type support."""

    def test_literal_type_formatting(self) -> None:
        """Test literal type formatting."""
        builder = PromptBuilder()

        result = builder._format_type(Literal["a", "b", "c"])
        assert "literal" in result
        assert "'a'" in result
        assert "'b'" in result
        assert "'c'" in result

    def test_literal_in_schema(self) -> None:
        """Test literal fields in schema description."""
        builder = PromptBuilder()
        desc = builder._describe_schema(LiteralModel)

        assert "size" in desc
        assert "small" in desc or "medium" in desc or "large" in desc


class TestFieldConstraints:
    """Tests for Pydantic Field validators and constraints."""

    def test_ge_le_constraints(self) -> None:
        """Test ge (>=) and le (<=) constraints."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Person)

        # age has ge=0, le=150
        assert "age" in desc
        # Should show constraints
        assert "Constraints:" in desc or ">=" in desc or "<=" in desc

    def test_min_max_length_constraints(self) -> None:
        """Test min_length and max_length constraints."""
        builder = PromptBuilder()
        desc = builder._describe_schema(Task)

        # title has min_length=1, max_length=200
        assert "title" in desc

    def test_get_field_constraints(self) -> None:
        """Test constraint extraction from field info."""
        builder = PromptBuilder()

        # Get field info from Task.title
        title_info = Task.model_fields["title"]
        constraints = builder._get_field_constraints(title_info)

        # Should have length constraints
        assert "min_length" in constraints or "max_length" in constraints


class TestDictSupport:
    """Tests for dict type support."""

    def test_dict_type_formatting(self) -> None:
        """Test dict type formatting."""
        builder = PromptBuilder()

        assert builder._format_type(dict[str, int]) == "dict[str, int]"
        assert builder._format_type(dict[str, float]) == "dict[str, float]"

    def test_list_of_dicts(self) -> None:
        """Test list[dict] formatting."""
        builder = PromptBuilder()

        result = builder._format_type(list[dict[str, float]])
        assert "list" in result
        assert "dict" in result


class TestTupleAndSetSupport:
    """Tests for tuple and set type support."""

    def test_tuple_type_formatting(self) -> None:
        """Test tuple type formatting."""
        builder = PromptBuilder()

        result = builder._format_type(tuple[str, int])
        assert "tuple" in result
        assert "str" in result
        assert "int" in result

    def test_set_type_formatting(self) -> None:
        """Test set type formatting."""
        builder = PromptBuilder()

        result = builder._format_type(set[str])
        assert "set" in result
        assert "str" in result


class TestCompletePromptGeneration:
    """Tests for complete prompt generation with advanced schemas."""

    def test_complete_prompt_with_nested_model(self) -> None:
        """Test full prompt generation with nested models."""
        builder = PromptBuilder()
        prompt = builder.build_extraction_prompt(
            document="John Doe, 30 years old, lives at 123 Main St, Boston, MA",
            schema=Person,
        )

        assert "## Extraction Schema" in prompt
        assert "## Document" in prompt
        assert "## Task" in prompt
        assert "Person" in prompt
        assert "Address" in prompt

    def test_complete_prompt_with_enums(self) -> None:
        """Test full prompt generation with enum fields."""
        builder = PromptBuilder()
        prompt = builder.build_extraction_prompt(
            document="High priority task: Fix the bug",
            schema=Task,
        )

        assert "priority" in prompt
        assert "status" in prompt
        # Enum values should be mentioned
        assert "low" in prompt or "medium" in prompt or "high" in prompt

    def test_complete_prompt_with_lists(self) -> None:
        """Test full prompt generation with list fields."""
        builder = PromptBuilder()
        prompt = builder.build_extraction_prompt(
            document="Task with tags: python, automation",
            schema=Task,
        )

        assert "tags" in prompt
        assert "list[Tag]" in prompt
