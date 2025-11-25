"""Tests for Phase 3: Prompt Engineering features."""

from enum import Enum

from pydantic import BaseModel, Field

from structured_extractor.prompts.builder import (
    ExtractionExample,
    PromptBuilder,
    PromptStrategy,
    PromptTemplate,
    PromptTemplates,
)


# Test models
class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class Address(BaseModel):
    """Address information."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(default="USA", description="Country code")


class Person(BaseModel):
    """A person's information."""

    name: str = Field(description="Full name")
    age: int | None = Field(default=None, description="Age in years")
    email: str = Field(description="Email address")
    status: Status = Field(default=Status.ACTIVE, description="Account status")
    address: Address | None = Field(default=None, description="Home address")


class TestExtractionExample:
    """Tests for ExtractionExample dataclass."""

    def test_basic_example(self):
        """Test creating a basic extraction example."""
        example = ExtractionExample(
            input_text="John Doe, 30 years old",
            output={"name": "John Doe", "age": 30},
        )
        assert example.input_text == "John Doe, 30 years old"
        assert example.output == {"name": "John Doe", "age": 30}
        assert example.explanation is None

    def test_example_with_explanation(self):
        """Test example with explanation."""
        example = ExtractionExample(
            input_text="Contact: jane@example.com",
            output={"email": "jane@example.com"},
            explanation="Email extracted from contact line",
        )
        assert example.explanation == "Email extracted from contact line"

    def test_example_to_dict(self):
        """Test converting example to dictionary."""
        example = ExtractionExample(
            input_text="Test input",
            output={"field": "value"},
            explanation="Test explanation",
        )
        result = example.to_dict()
        assert result["input"] == "Test input"
        assert result["output"] == {"field": "value"}
        assert result["explanation"] == "Test explanation"

    def test_example_with_pydantic_model(self):
        """Test example with Pydantic model as output."""
        address = Address(street="123 Main St", city="Boston")
        example = ExtractionExample(
            input_text="123 Main St, Boston",
            output=address,
        )
        result = example.to_dict()
        assert result["output"] == {"street": "123 Main St", "city": "Boston", "country": "USA"}


class TestPromptStrategy:
    """Tests for PromptStrategy configuration."""

    def test_default_strategy(self):
        """Test default strategy values."""
        strategy = PromptStrategy()
        assert strategy.include_field_descriptions is True
        assert strategy.include_examples is True
        assert strategy.include_constraints is True
        assert strategy.include_reasoning is False
        assert strategy.strict_mode is False
        assert strategy.verbose_types is False
        assert strategy.schema_format == "markdown"
        assert strategy.hint_style == "inline"

    def test_custom_strategy(self):
        """Test custom strategy configuration."""
        strategy = PromptStrategy(
            include_reasoning=True,
            strict_mode=True,
            verbose_types=True,
            schema_format="json_schema",
            hint_style="section",
        )
        assert strategy.include_reasoning is True
        assert strategy.strict_mode is True
        assert strategy.verbose_types is True
        assert strategy.schema_format == "json_schema"
        assert strategy.hint_style == "section"


class TestPromptTemplate:
    """Tests for PromptTemplate configuration."""

    def test_basic_template(self):
        """Test creating a basic template."""
        template = PromptTemplate(
            name="test",
            description="Test template",
            system_prompt="You are a test assistant",
            extraction_instruction="Extract the data",
        )
        assert template.name == "test"
        assert template.description == "Test template"
        assert template.system_prompt == "You are a test assistant"
        assert template.extraction_instruction == "Extract the data"

    def test_template_with_defaults(self):
        """Test template default values."""
        template = PromptTemplate(
            name="test",
            description="Test",
            system_prompt="Test",
            extraction_instruction="Test",
        )
        assert isinstance(template.strategy, PromptStrategy)
        assert template.default_field_hints == {}
        assert template.examples == []

    def test_template_with_field_hints(self):
        """Test template with default field hints."""
        template = PromptTemplate(
            name="test",
            description="Test",
            system_prompt="Test",
            extraction_instruction="Test",
            default_field_hints={"name": "Full legal name"},
        )
        assert template.default_field_hints == {"name": "Full legal name"}


class TestPromptTemplates:
    """Tests for built-in prompt templates."""

    def test_default_template(self):
        """Test the default template."""
        template = PromptTemplates.default()
        assert template.name == "default"
        assert "extraction assistant" in template.system_prompt.lower()
        assert "extract" in template.extraction_instruction.lower()

    def test_strict_template(self):
        """Test the strict template."""
        template = PromptTemplates.strict()
        assert template.name == "strict"
        assert "accuracy" in template.system_prompt.lower()
        assert template.strategy.strict_mode is True

    def test_lenient_template(self):
        """Test the lenient template."""
        template = PromptTemplates.lenient()
        assert template.name == "lenient"
        assert "inferences" in template.system_prompt.lower()

    def test_invoice_template(self):
        """Test the invoice template."""
        template = PromptTemplates.invoice()
        assert template.name == "invoice"
        assert "invoice" in template.system_prompt.lower()
        assert "invoice_number" in template.default_field_hints
        assert "date" in template.default_field_hints
        assert "total" in template.default_field_hints

    def test_resume_template(self):
        """Test the resume template."""
        template = PromptTemplates.resume()
        assert template.name == "resume"
        assert "resume" in template.system_prompt.lower()
        assert "name" in template.default_field_hints

    def test_contract_template(self):
        """Test the contract template."""
        template = PromptTemplates.contract()
        assert template.name == "contract"
        assert "contract" in template.system_prompt.lower()
        assert "effective_date" in template.default_field_hints

    def test_get_all_templates(self):
        """Test getting all templates."""
        templates = PromptTemplates.get_all_templates()
        assert len(templates) == 6
        assert "default" in templates
        assert "strict" in templates
        assert "lenient" in templates
        assert "invoice" in templates
        assert "resume" in templates
        assert "contract" in templates


class TestPromptBuilderWithTemplate:
    """Tests for PromptBuilder with templates."""

    def test_builder_with_template(self):
        """Test builder initialized with a template."""
        template = PromptTemplates.invoice()
        builder = PromptBuilder(template=template)

        assert builder.template == template
        system_prompt = builder.build_system_prompt()
        assert "invoice" in system_prompt.lower()

    def test_system_prompt_from_template(self):
        """Test that system prompt comes from template."""
        template = PromptTemplates.strict()
        builder = PromptBuilder(template=template)

        system_prompt = builder.build_system_prompt()
        assert "accuracy" in system_prompt.lower()

    def test_custom_prompt_overrides_template(self):
        """Test that custom prompt overrides template."""
        template = PromptTemplates.invoice()
        builder = PromptBuilder(template=template)

        custom = "Custom system prompt"
        system_prompt = builder.build_system_prompt(custom_prompt=custom)
        assert system_prompt == custom


class TestPromptBuilderWithStrategy:
    """Tests for PromptBuilder with strategy."""

    def test_builder_with_strategy(self):
        """Test builder with custom strategy."""
        strategy = PromptStrategy(strict_mode=True, include_reasoning=True)
        builder = PromptBuilder(strategy=strategy)

        assert builder.strategy == strategy

    def test_strict_mode_instruction(self):
        """Test strict mode affects extraction instruction."""
        strategy = PromptStrategy(strict_mode=True)
        builder = PromptBuilder(strategy=strategy)

        prompt = builder.build_extraction_prompt(
            document="Test document",
            schema=Person,
        )
        assert "accuracy" in prompt.lower() or "uncertain" in prompt.lower()

    def test_reasoning_instruction(self):
        """Test reasoning mode adds reasoning section."""
        strategy = PromptStrategy(include_reasoning=True)
        builder = PromptBuilder(strategy=strategy)

        prompt = builder.build_extraction_prompt(
            document="Test document",
            schema=Person,
        )
        assert "## Reasoning" in prompt
        assert "analyze the document" in prompt.lower()


class TestFieldHintsMerging:
    """Tests for field hints merging with template defaults."""

    def test_template_hints_used(self):
        """Test that template default hints are used."""
        # Create a template with hints matching the Person schema
        template = PromptTemplate(
            name="test",
            description="Test",
            system_prompt="Test",
            extraction_instruction="Test",
            default_field_hints={"name": "Full legal name from template"},
        )
        builder = PromptBuilder(template=template)

        prompt = builder.build_extraction_prompt(
            document="John Doe",
            schema=Person,
        )
        # Template hints should be applied
        assert "Hint:" in prompt
        assert "Full legal name from template" in prompt

    def test_custom_hints_override_template(self):
        """Test that custom hints override template defaults."""
        template = PromptTemplate(
            name="test",
            description="Test",
            system_prompt="Test",
            extraction_instruction="Test",
            default_field_hints={"name": "Template hint"},
        )
        builder = PromptBuilder(template=template)

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            field_hints={"name": "Custom hint"},
        )
        assert "Custom hint" in prompt
        # Template hint should be overridden
        assert "Template hint" not in prompt

    def test_hints_merge(self):
        """Test that hints merge properly."""
        template = PromptTemplate(
            name="test",
            description="Test",
            system_prompt="Test",
            extraction_instruction="Test",
            default_field_hints={"name": "Name hint"},
        )
        builder = PromptBuilder(template=template)

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            field_hints={"email": "Email hint"},
        )
        # Both hints should be present
        assert "Name hint" in prompt
        assert "Email hint" in prompt


class TestExamplesFormatting:
    """Tests for few-shot examples formatting."""

    def test_basic_examples(self):
        """Test basic examples formatting."""
        builder = PromptBuilder()
        examples = [
            {"input": "John, 30", "output": {"name": "John", "age": 30}},
        ]

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            examples=examples,
        )
        assert "## Examples" in prompt
        assert "**Example 1:**" in prompt
        assert "John, 30" in prompt

    def test_extraction_example_objects(self):
        """Test ExtractionExample objects in prompt."""
        builder = PromptBuilder()
        examples = [
            ExtractionExample(
                input_text="Jane Doe, jane@email.com",
                output={"name": "Jane Doe", "email": "jane@email.com"},
                explanation="Name and email extracted from text",
            ),
        ]

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            examples=examples,
        )
        assert "## Examples" in prompt
        assert "Jane Doe, jane@email.com" in prompt
        assert "Explanation:" in prompt
        assert "Name and email extracted" in prompt

    def test_template_examples_merged(self):
        """Test that template examples are merged with provided examples."""
        template_example = ExtractionExample(
            input_text="Template example",
            output={"name": "Template"},
        )
        template = PromptTemplate(
            name="test",
            description="Test",
            system_prompt="Test",
            extraction_instruction="Test",
            examples=[template_example],
        )
        builder = PromptBuilder(template=template)

        custom_example = ExtractionExample(
            input_text="Custom example",
            output={"name": "Custom"},
        )

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            examples=[custom_example],
        )
        # Both examples should be present
        assert "Template example" in prompt
        assert "Custom example" in prompt

    def test_examples_disabled(self):
        """Test examples can be disabled."""
        builder = PromptBuilder(include_examples=False)
        examples = [
            {"input": "Test", "output": {"name": "Test"}},
        ]

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            examples=examples,
        )
        assert "## Examples" not in prompt


class TestExtractionInstructionCustomization:
    """Tests for extraction instruction customization."""

    def test_default_instruction(self):
        """Test default extraction instruction."""
        builder = PromptBuilder()
        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
        )
        assert "## Task" in prompt
        assert "extract" in prompt.lower()

    def test_template_instruction(self):
        """Test template-specific instruction."""
        template = PromptTemplates.invoice()
        builder = PromptBuilder(template=template)

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
        )
        # Invoice template has specific instruction
        assert "invoice" in prompt.lower()

    def test_strict_mode_instruction(self):
        """Test strict mode instruction."""
        strategy = PromptStrategy(strict_mode=True)
        builder = PromptBuilder(strategy=strategy)

        instruction = builder._get_extraction_instruction()
        assert "accuracy" in instruction.lower() or "uncertain" in instruction.lower()


class TestCompletePromptGeneration:
    """Tests for complete prompt generation with all features."""

    def test_complete_prompt_structure(self):
        """Test complete prompt has all expected sections."""
        strategy = PromptStrategy(include_reasoning=True)
        builder = PromptBuilder(strategy=strategy)

        examples = [
            ExtractionExample(
                input_text="Sample input",
                output={"name": "Sample", "email": "sample@email.com"},
            ),
        ]

        prompt = builder.build_extraction_prompt(
            document="John Doe, john@example.com, 25 years old",
            schema=Person,
            field_hints={"name": "Full legal name"},
            examples=examples,
        )

        # Check all sections are present
        assert "## Extraction Schema" in prompt
        assert "## Reasoning" in prompt
        assert "## Examples" in prompt
        assert "## Document" in prompt
        assert "## Task" in prompt

    def test_complete_prompt_with_invoice_template(self):
        """Test complete prompt with invoice template."""
        template = PromptTemplates.invoice()
        builder = PromptBuilder(template=template)

        prompt = builder.build_extraction_prompt(
            document="Invoice #12345\nTotal: $500.00",
            schema=Person,  # Even with Person schema, hints apply
        )

        assert "## Extraction Schema" in prompt
        assert "## Document" in prompt
        assert "## Task" in prompt

    def test_all_templates_produce_valid_prompts(self):
        """Test all built-in templates produce valid prompts."""
        templates = PromptTemplates.get_all_templates()

        for _name, template in templates.items():
            builder = PromptBuilder(template=template)

            # Should not raise
            system_prompt = builder.build_system_prompt()
            assert len(system_prompt) > 0

            prompt = builder.build_extraction_prompt(
                document="Test document",
                schema=Person,
            )
            assert len(prompt) > 0
            assert "## Extraction Schema" in prompt
            assert "## Document" in prompt
            assert "## Task" in prompt


class TestBackwardsCompatibility:
    """Tests to ensure backwards compatibility with Phase 1 and 2 code."""

    def test_builder_default_init(self):
        """Test builder works with default initialization."""
        builder = PromptBuilder()

        system_prompt = builder.build_system_prompt()
        assert len(system_prompt) > 0

    def test_builder_simple_usage(self):
        """Test builder works with simple usage pattern."""
        builder = PromptBuilder(
            include_field_descriptions=True,
            include_examples=True,
        )

        prompt = builder.build_extraction_prompt(
            document="Test document",
            schema=Person,
        )
        assert len(prompt) > 0

    def test_field_hints_still_work(self):
        """Test field hints still work as before."""
        builder = PromptBuilder()

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            field_hints={"name": "Full legal name"},
        )
        assert "Full legal name" in prompt

    def test_dict_examples_still_work(self):
        """Test dict-style examples still work."""
        builder = PromptBuilder()
        examples = [
            {"input": "Test input", "output": {"name": "Test"}},
        ]

        prompt = builder.build_extraction_prompt(
            document="Test",
            schema=Person,
            examples=examples,
        )
        assert "Test input" in prompt
