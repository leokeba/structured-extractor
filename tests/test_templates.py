"""Tests for document templates module."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from structured_extractor import (
    DocumentTemplate,
    TemplateRegistry,
    TemplateValidationError,
)


# Test schemas
class SimpleSchema(BaseModel):
    """Simple test schema."""

    name: str = Field(description="Person name")
    age: int = Field(description="Person age")


class InvoiceSchema(BaseModel):
    """Invoice schema for testing."""

    invoice_number: str = Field(description="Invoice ID")
    vendor: str = Field(description="Vendor name")
    total: float = Field(description="Total amount")
    line_items: list[str] = Field(default_factory=list, description="Line items")


class EmptySchema(BaseModel):
    """Schema with no fields for testing validation."""

    pass


# ============================================================================
# DocumentTemplate Tests
# ============================================================================


class TestDocumentTemplateCreation:
    """Tests for DocumentTemplate creation and basic operations."""

    def test_create_simple_template(self):
        """Test creating a basic template."""
        template = DocumentTemplate(
            name="simple",
            schema_class=SimpleSchema,
            description="A simple test template",
        )

        assert template.name == "simple"
        assert template.schema_class is SimpleSchema
        assert template.description == "A simple test template"
        assert template.version == "1.0"

    def test_create_template_with_all_options(self):
        """Test creating template with all options specified."""
        template = DocumentTemplate(
            name="full_template",
            schema_class=InvoiceSchema,
            description="Full invoice template",
            system_prompt="Extract invoice data carefully.",
            field_hints={"total": "Include tax in total"},
            examples=[
                {"input": "Invoice 123", "output": {"invoice_number": "123"}}
            ],
            version="2.0",
            tags=["financial", "invoice"],
            parent_template="base_financial",
        )

        assert template.name == "full_template"
        assert template.system_prompt == "Extract invoice data carefully."
        assert template.field_hints == {"total": "Include tax in total"}
        assert template.examples is not None
        assert len(template.examples) == 1
        assert template.version == "2.0"
        assert template.tags is not None
        assert "financial" in template.tags
        assert template.parent_template == "base_financial"

    def test_name_validation_strips_whitespace(self):
        """Test that template names are cleaned."""
        template = DocumentTemplate(
            name="  My Template  ",
            schema_class=SimpleSchema,
        )
        assert template.name == "my_template"

    def test_name_validation_converts_spaces_to_underscore(self):
        """Test that spaces in names become underscores."""
        template = DocumentTemplate(
            name="My Invoice Template",
            schema_class=SimpleSchema,
        )
        assert template.name == "my_invoice_template"

    def test_name_validation_rejects_empty(self):
        """Test that empty names are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DocumentTemplate(name="", schema_class=SimpleSchema)

    def test_name_validation_rejects_whitespace_only(self):
        """Test that whitespace-only names are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DocumentTemplate(name="   ", schema_class=SimpleSchema)


class TestDocumentTemplateValidation:
    """Tests for template validation."""

    def test_validate_empty_schema_raises_error(self):
        """Test that schemas with no fields cause validation error."""
        template = DocumentTemplate(
            name="empty",
            schema_class=EmptySchema,
        )

        with pytest.raises(TemplateValidationError, match="no fields"):
            template.validate_template()

    def test_validate_invalid_field_hints_warns(self):
        """Test that field hints for non-existent fields produce warnings."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            field_hints={"nonexistent_field": "Some hint"},
        )

        warnings = template.validate_template()
        assert len(warnings) == 1
        assert "nonexistent_field" in warnings[0]

    def test_validate_valid_field_hints_no_warnings(self):
        """Test that valid field hints produce no warnings."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            field_hints={"name": "Full name", "age": "Age in years"},
        )

        warnings = template.validate_template()
        assert len(warnings) == 0

    def test_validate_examples_missing_input(self):
        """Test that examples without input produce warnings."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            examples=[{"output": {"name": "John", "age": 30}}],
        )

        warnings = template.validate_template()
        assert any("input" in w for w in warnings)

    def test_validate_examples_missing_output(self):
        """Test that examples without output produce warnings."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            examples=[{"input": "Some text"}],
        )

        warnings = template.validate_template()
        assert any("output" in w for w in warnings)

    def test_validate_examples_accepts_input_text(self):
        """Test that 'input_text' is valid alternative to 'input'."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            examples=[{"input_text": "Some text", "output": {"name": "John"}}],
        )

        warnings = template.validate_template()
        # Should not warn about missing input
        assert not any("input" in w for w in warnings)


class TestDocumentTemplateSerialization:
    """Tests for template serialization to JSON and YAML."""

    def test_to_dict_includes_required_fields(self):
        """Test that to_dict includes all required fields."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            description="Test template",
        )

        data = template.to_dict()

        assert data["name"] == "test"
        assert data["version"] == "1.0"
        assert "schema" in data
        assert "schema_name" in data
        assert data["description"] == "Test template"

    def test_to_dict_exclude_schema(self):
        """Test that to_dict can exclude full schema."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
        )

        data = template.to_dict(exclude_schema=True)

        assert "schema" not in data
        assert data["schema_name"] == "SimpleSchema"

    def test_to_dict_excludes_none_values(self):
        """Test that None values are not included in dict."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
        )

        data = template.to_dict()

        assert "system_prompt" not in data
        assert "field_hints" not in data
        assert "examples" not in data

    def test_to_json_returns_valid_json(self):
        """Test that to_json returns valid JSON string."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            description="Test template",
        )

        json_str = template.to_json()
        parsed = json.loads(json_str)

        assert parsed["name"] == "test"
        assert parsed["description"] == "Test template"

    def test_to_json_writes_to_file(self):
        """Test that to_json can write to file."""
        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "template.json"
            template.to_json(path)

            assert path.exists()
            content = json.loads(path.read_text())
            assert content["name"] == "test"

    def test_from_json_string(self):
        """Test loading template from JSON string."""
        json_str = json.dumps({
            "name": "loaded",
            "description": "Loaded template",
            "version": "2.0",
            "field_hints": {"name": "Full name"},
        })

        template = DocumentTemplate.from_json(json_str, SimpleSchema)

        assert template.name == "loaded"
        assert template.description == "Loaded template"
        assert template.version == "2.0"
        assert template.field_hints == {"name": "Full name"}
        assert template.schema_class is SimpleSchema

    def test_from_json_file(self):
        """Test loading template from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "template.json"
            path.write_text(json.dumps({
                "name": "from_file",
                "description": "From file",
            }))

            template = DocumentTemplate.from_json(path, SimpleSchema)

            assert template.name == "from_file"
            assert template.description == "From file"

    def test_json_roundtrip(self):
        """Test that JSON serialization roundtrips correctly."""
        original = DocumentTemplate(
            name="roundtrip",
            schema_class=InvoiceSchema,
            description="Roundtrip test",
            field_hints={"total": "Include tax"},
            tags=["test"],
            version="1.5",
        )

        json_str = original.to_json()
        loaded = DocumentTemplate.from_json(json_str, InvoiceSchema)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.field_hints == original.field_hints
        assert loaded.tags == original.tags
        assert loaded.version == original.version


class TestDocumentTemplateYAML:
    """Tests for YAML serialization."""

    def test_to_yaml_returns_valid_yaml(self):
        """Test that to_yaml returns valid YAML string."""
        import yaml

        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            description="Test template",
        )

        yaml_str = template.to_yaml()
        parsed = yaml.safe_load(yaml_str)

        assert parsed["name"] == "test"
        assert parsed["description"] == "Test template"

    def test_to_yaml_writes_to_file(self):
        """Test that to_yaml can write to file."""
        import yaml

        template = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "template.yaml"
            template.to_yaml(path)

            assert path.exists()
            content = yaml.safe_load(path.read_text())
            assert content["name"] == "test"

    def test_from_yaml_string(self):
        """Test loading template from YAML string."""
        import yaml

        yaml_str = yaml.dump({
            "name": "loaded",
            "description": "Loaded template",
            "version": "2.0",
        })

        template = DocumentTemplate.from_yaml(yaml_str, SimpleSchema)

        assert template.name == "loaded"
        assert template.description == "Loaded template"
        assert template.version == "2.0"

    def test_yaml_roundtrip(self):
        """Test that YAML serialization roundtrips correctly."""
        original = DocumentTemplate(
            name="yaml_roundtrip",
            schema_class=InvoiceSchema,
            description="YAML roundtrip test",
            field_hints={"total": "Include tax"},
            tags=["test", "yaml"],
        )

        yaml_str = original.to_yaml()
        loaded = DocumentTemplate.from_yaml(yaml_str, InvoiceSchema)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.field_hints == original.field_hints
        assert loaded.tags == original.tags


class TestDocumentTemplateInheritance:
    """Tests for template inheritance."""

    def test_merge_with_parent_basic(self):
        """Test basic parent/child merging."""
        parent = DocumentTemplate(
            name="parent",
            schema_class=SimpleSchema,
            description="Parent description",
            system_prompt="Parent system prompt",
        )

        child = DocumentTemplate(
            name="child",
            schema_class=SimpleSchema,
            parent_template="parent",
        )

        merged = child.merge_with_parent(parent)

        assert merged.name == "child"
        assert merged.description == "Parent description"
        assert merged.system_prompt == "Parent system prompt"
        assert merged.parent_template == "parent"

    def test_merge_child_overrides_parent(self):
        """Test that child values override parent values."""
        parent = DocumentTemplate(
            name="parent",
            schema_class=SimpleSchema,
            description="Parent description",
            system_prompt="Parent prompt",
        )

        child = DocumentTemplate(
            name="child",
            schema_class=SimpleSchema,
            description="Child description",
            system_prompt="Child prompt",
        )

        merged = child.merge_with_parent(parent)

        assert merged.description == "Child description"
        assert merged.system_prompt == "Child prompt"

    def test_merge_field_hints_combined(self):
        """Test that field hints are merged with child taking precedence."""
        parent = DocumentTemplate(
            name="parent",
            schema_class=SimpleSchema,
            field_hints={"name": "Parent hint for name", "age": "Parent hint for age"},
        )

        child = DocumentTemplate(
            name="child",
            schema_class=SimpleSchema,
            field_hints={"name": "Child hint for name"},  # Override name only
        )

        merged = child.merge_with_parent(parent)

        assert merged.field_hints is not None
        assert merged.field_hints["name"] == "Child hint for name"
        assert merged.field_hints["age"] == "Parent hint for age"

    def test_merge_examples_concatenated(self):
        """Test that examples from parent and child are combined."""
        parent = DocumentTemplate(
            name="parent",
            schema_class=SimpleSchema,
            examples=[{"input": "Parent example", "output": {"name": "P"}}],
        )

        child = DocumentTemplate(
            name="child",
            schema_class=SimpleSchema,
            examples=[{"input": "Child example", "output": {"name": "C"}}],
        )

        merged = child.merge_with_parent(parent)

        assert merged.examples is not None
        assert len(merged.examples) == 2
        assert merged.examples[0]["input"] == "Parent example"
        assert merged.examples[1]["input"] == "Child example"

    def test_merge_tags_combined_and_deduplicated(self):
        """Test that tags are merged and deduplicated."""
        parent = DocumentTemplate(
            name="parent",
            schema_class=SimpleSchema,
            tags=["common", "parent"],
        )

        child = DocumentTemplate(
            name="child",
            schema_class=SimpleSchema,
            tags=["common", "child"],
        )

        merged = child.merge_with_parent(parent)

        assert merged.tags is not None
        assert len(merged.tags) == 3
        assert set(merged.tags) == {"common", "parent", "child"}


# ============================================================================
# TemplateRegistry Tests
# ============================================================================


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return TemplateRegistry()

    @pytest.fixture
    def sample_template(self):
        """Create a sample template for testing."""
        return DocumentTemplate(
            name="sample",
            schema_class=SimpleSchema,
            description="A sample template",
            tags=["test"],
        )

    def test_register_and_get(self, registry, sample_template):
        """Test basic registration and retrieval."""
        registry.register(sample_template)

        retrieved = registry.get("sample")
        assert retrieved is not None
        assert retrieved.name == "sample"

    def test_get_nonexistent_returns_none(self, registry):
        """Test that getting non-existent template returns None."""
        result = registry.get("nonexistent")
        assert result is None

    def test_get_or_raise_nonexistent(self, registry):
        """Test that get_or_raise raises KeyError for missing template."""
        with pytest.raises(KeyError, match="not found"):
            registry.get_or_raise("nonexistent")

    def test_register_duplicate_raises_error(self, registry, sample_template):
        """Test that registering duplicate name raises error."""
        registry.register(sample_template)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(sample_template)

    def test_register_duplicate_with_overwrite(self, registry):
        """Test that overwrite=True allows replacing templates."""
        template1 = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            description="First version",
        )
        template2 = DocumentTemplate(
            name="test",
            schema_class=SimpleSchema,
            description="Second version",
        )

        registry.register(template1)
        registry.register(template2, overwrite=True)

        retrieved = registry.get("test")
        assert retrieved.description == "Second version"

    def test_unregister_existing(self, registry, sample_template):
        """Test unregistering an existing template."""
        registry.register(sample_template)

        result = registry.unregister("sample")

        assert result is True
        assert registry.get("sample") is None

    def test_unregister_nonexistent(self, registry):
        """Test unregistering a non-existent template."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_list_templates(self, registry):
        """Test listing all template names."""
        t1 = DocumentTemplate(name="alpha", schema_class=SimpleSchema)
        t2 = DocumentTemplate(name="beta", schema_class=SimpleSchema)
        t3 = DocumentTemplate(name="gamma", schema_class=SimpleSchema)

        registry.register(t1)
        registry.register(t2)
        registry.register(t3)

        names = registry.list_templates()

        assert set(names) == {"alpha", "beta", "gamma"}

    def test_search_by_tags(self, registry):
        """Test searching templates by tags."""
        t1 = DocumentTemplate(
            name="invoice", schema_class=SimpleSchema, tags=["financial", "invoice"]
        )
        t2 = DocumentTemplate(
            name="receipt", schema_class=SimpleSchema, tags=["financial", "receipt"]
        )
        t3 = DocumentTemplate(
            name="resume", schema_class=SimpleSchema, tags=["hr", "document"]
        )

        registry.register(t1)
        registry.register(t2)
        registry.register(t3)

        financial = registry.search_by_tags(["financial"])
        assert len(financial) == 2
        assert all("financial" in t.tags for t in financial)

        hr = registry.search_by_tags(["hr"])
        assert len(hr) == 1
        assert hr[0].name == "resume"

    def test_search_by_tags_any_match(self, registry):
        """Test that tag search returns templates with ANY matching tag."""
        t1 = DocumentTemplate(
            name="a", schema_class=SimpleSchema, tags=["tag1"]
        )
        t2 = DocumentTemplate(
            name="b", schema_class=SimpleSchema, tags=["tag2"]
        )
        t3 = DocumentTemplate(
            name="c", schema_class=SimpleSchema, tags=["tag3"]
        )

        registry.register(t1)
        registry.register(t2)
        registry.register(t3)

        results = registry.search_by_tags(["tag1", "tag2"])
        assert len(results) == 2

    def test_search_by_description(self, registry):
        """Test searching templates by description content."""
        t1 = DocumentTemplate(
            name="a", schema_class=SimpleSchema, description="Extract invoice data"
        )
        t2 = DocumentTemplate(
            name="b", schema_class=SimpleSchema, description="Extract receipt items"
        )
        t3 = DocumentTemplate(
            name="c", schema_class=SimpleSchema, description="Parse resume information"
        )

        registry.register(t1)
        registry.register(t2)
        registry.register(t3)

        extract_results = registry.search_by_description("extract")
        assert len(extract_results) == 2

        resume_results = registry.search_by_description("resume")
        assert len(resume_results) == 1

    def test_search_by_description_case_insensitive(self, registry):
        """Test that description search is case-insensitive."""
        t = DocumentTemplate(
            name="test", schema_class=SimpleSchema, description="INVOICE Data"
        )
        registry.register(t)

        results = registry.search_by_description("invoice")
        assert len(results) == 1

    def test_len(self, registry, sample_template):
        """Test __len__ returns correct count."""
        assert len(registry) == 0

        registry.register(sample_template)
        assert len(registry) == 1

    def test_contains(self, registry, sample_template):
        """Test __contains__ (in operator)."""
        assert "sample" not in registry

        registry.register(sample_template)
        assert "sample" in registry

    def test_iter(self, registry):
        """Test __iter__ iterates over template names."""
        t1 = DocumentTemplate(name="a", schema_class=SimpleSchema)
        t2 = DocumentTemplate(name="b", schema_class=SimpleSchema)

        registry.register(t1)
        registry.register(t2)

        names = list(registry)
        assert set(names) == {"a", "b"}

    def test_to_dict_exports_all_templates(self, registry):
        """Test that to_dict exports all templates."""
        t1 = DocumentTemplate(
            name="a", schema_class=SimpleSchema, description="Template A"
        )
        t2 = DocumentTemplate(
            name="b", schema_class=SimpleSchema, description="Template B"
        )

        registry.register(t1)
        registry.register(t2)

        exported = registry.to_dict()

        assert "a" in exported
        assert "b" in exported
        assert exported["a"]["description"] == "Template A"
        assert exported["b"]["description"] == "Template B"

    def test_register_validates_template(self, registry):
        """Test that registration validates the template."""
        invalid_template = DocumentTemplate(
            name="invalid",
            schema_class=EmptySchema,  # No fields
        )

        with pytest.raises(TemplateValidationError):
            registry.register(invalid_template)

    def test_register_resolves_parent_inheritance(self, registry):
        """Test that parent templates are resolved during registration."""
        parent = DocumentTemplate(
            name="parent",
            schema_class=SimpleSchema,
            description="Parent template",
            field_hints={"name": "Parent hint"},
        )

        child = DocumentTemplate(
            name="child",
            schema_class=SimpleSchema,
            parent_template="parent",
            field_hints={"age": "Child hint"},
        )

        registry.register(parent)
        registry.register(child)

        retrieved = registry.get("child")

        # Should have merged field hints
        assert retrieved.field_hints["name"] == "Parent hint"
        assert retrieved.field_hints["age"] == "Child hint"
        # Should inherit description
        assert retrieved.description == "Parent template"

    def test_register_unresolved_parent_stores_as_is(self, registry):
        """Test that templates with unresolved parents are stored as-is."""
        child = DocumentTemplate(
            name="orphan",
            schema_class=SimpleSchema,
            description="Child description",
            parent_template="nonexistent_parent",
        )

        # Should not raise - parent just isn't resolved
        registry.register(child)

        retrieved = registry.get("orphan")
        assert retrieved.parent_template == "nonexistent_parent"
        assert retrieved.description == "Child description"
