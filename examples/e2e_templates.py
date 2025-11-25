"""End-to-end test: Document templates and template registry.

This script demonstrates and tests:
1. Creating and using DocumentTemplates
2. Template serialization (JSON/YAML)
3. Template inheritance
4. TemplateRegistry for managing templates
5. Template validation
6. Using templates with extraction

Prerequisites:
- OPENAI_API_KEY environment variable set

Usage:
    python examples/e2e_templates.py
"""

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import (
    DocumentExtractor,
    DocumentTemplate,
    ExtractionResult,
    TemplateRegistry,
)

# Load environment variables
load_dotenv()


# ============================================================================
# Test Result Tracking
# ============================================================================


@dataclass
class TemplateTestResult:
    """Result of a template test."""

    test_name: str
    success: bool
    latency_ms: float
    error: str | None = None
    details: dict[str, Any] | None = None


@dataclass
class TemplateTestSuite:
    """Results from template test suite."""

    suite_name: str
    results: list[TemplateTestResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.success)

    def print_summary(self) -> None:
        """Print summary of template tests."""
        print(f"\n{'=' * 60}")
        print(f"📊 {self.suite_name} - Results Summary")
        print("=" * 60)
        print(f"Total Tests:  {self.total_tests}")
        print(f"Passed:       {self.passed_tests} ✅")
        print(f"Failed:       {self.total_tests - self.passed_tests} ❌")
        print(f"Success Rate: {self.passed_tests / self.total_tests:.1%}")
        print("=" * 60)


# ============================================================================
# Schema Definitions
# ============================================================================


class Invoice(BaseModel):
    """Invoice data for template testing."""

    invoice_number: str = Field(description="Unique invoice identifier")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Vendor/supplier name")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")


class Receipt(BaseModel):
    """Receipt data."""

    store_name: str = Field(description="Store/restaurant name")
    date: str = Field(description="Transaction date")
    items: list[str] = Field(default_factory=list, description="Items purchased")
    total: float = Field(description="Total amount")
    payment_method: str | None = Field(default=None, description="Payment method")


class ContactCard(BaseModel):
    """Contact card data."""

    name: str = Field(description="Person's name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    company: str | None = Field(default=None, description="Company name")


class ShippingLabel(BaseModel):
    """Shipping label data."""

    sender_name: str = Field(description="Sender's name")
    sender_address: str = Field(description="Sender's address")
    recipient_name: str = Field(description="Recipient's name")
    recipient_address: str = Field(description="Recipient's address")
    tracking_number: str | None = Field(default=None, description="Tracking number")
    weight: str | None = Field(default=None, description="Package weight")


# ============================================================================
# Test Documents
# ============================================================================

SAMPLE_INVOICE = """
INVOICE #INV-2024-001

Date: November 25, 2024
From: ABC Supplies Co.
      123 Business St, Austin TX

To: Customer Corp

Items:
- Office Supplies: $250.00
- Printer Paper: $75.00

Total Due: $325.00 USD
"""

SAMPLE_RECEIPT = """
COFFEE SHOP RECEIPT
-------------------
Store: Morning Brew Cafe
Date: Nov 25, 2024

Items:
- Latte (Large): $5.50
- Croissant: $3.25
- Muffin: $2.75

Total: $11.50
Paid: Credit Card
"""

SAMPLE_CONTACT = """
Business Card:
Jane Smith
Senior Developer
TechCorp Inc.
jane.smith@techcorp.com
(555) 987-6543
"""


# ============================================================================
# Test Functions
# ============================================================================


def test_template_creation(suite: TemplateTestSuite) -> DocumentTemplate | None:
    """Test basic template creation."""
    print("\n" + "-" * 50)
    print("Test: Template Creation")
    print("-" * 50)

    start_time = time.time()
    template = None

    try:
        template = DocumentTemplate(
            name="invoice_template",
            schema_class=Invoice,
            description="Template for extracting invoice data",
            system_prompt=(
                "You are an expert invoice processor. "
                "Extract invoice data with precision."
            ),
            field_hints={
                "invoice_number": "Look for patterns like 'INV-XXXX' or '#XXXX'",
                "total_amount": "The final total after all taxes and fees",
            },
            tags=["financial", "invoice"],
            version="1.0",
        )

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="template_creation",
                success=True,
                latency_ms=latency_ms,
                details={
                    "name": template.name,
                    "schema": template.schema_class.__name__,
                    "tags": template.tags,
                },
            )
        )

        print(f"✅ Template created: {template.name}")
        print(f"   Schema: {template.schema_class.__name__}")
        print(f"   Tags: {template.tags}")
        print(f"   Field hints: {len(template.field_hints or {})} defined")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="template_creation",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")

    return template


def test_template_validation(suite: TemplateTestSuite) -> None:
    """Test template validation."""
    print("\n" + "-" * 50)
    print("Test: Template Validation")
    print("-" * 50)

    start_time = time.time()

    try:
        # Valid template
        valid_template = DocumentTemplate(
            name="valid_template",
            schema_class=Invoice,
            description="Valid template",
            field_hints={"invoice_number": "Valid hint"},
        )
        warnings = valid_template.validate_template()

        # Template with warnings (hint for non-existent field)
        warning_template = DocumentTemplate(
            name="warning_template",
            schema_class=Invoice,
            description="Template with warnings",
            field_hints={"nonexistent_field": "This field doesn't exist"},
        )
        warnings_list = warning_template.validate_template()

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="template_validation",
                success=True,
                latency_ms=latency_ms,
                details={
                    "valid_warnings": len(warnings),
                    "warning_template_warnings": len(warnings_list),
                },
            )
        )

        print("✅ Validation tests passed")
        print(f"   Valid template warnings: {len(warnings)}")
        print(f"   Warning template warnings: {len(warnings_list)}")
        if warnings_list:
            for w in warnings_list:
                print(f"     ⚠️ {w}")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="template_validation",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_template_serialization_json(suite: TemplateTestSuite) -> None:
    """Test template JSON serialization and deserialization."""
    print("\n" + "-" * 50)
    print("Test: Template JSON Serialization")
    print("-" * 50)

    start_time = time.time()

    try:
        # Create template
        original = DocumentTemplate(
            name="json_test_template",
            schema_class=Invoice,
            description="Template for JSON serialization test",
            field_hints={"invoice_number": "Invoice ID pattern"},
            tags=["test", "json"],
            version="2.0",
        )

        # Serialize to JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        json_str = original.to_json(json_path)

        # Deserialize from JSON
        loaded = DocumentTemplate.from_json(json_path, Invoice)

        # Verify
        assert loaded.name == original.name
        assert loaded.description == original.description
        assert loaded.field_hints == original.field_hints
        assert loaded.version == original.version

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="json_serialization",
                success=True,
                latency_ms=latency_ms,
                details={"json_length": len(json_str)},
            )
        )

        print("✅ JSON serialization successful")
        print(f"   JSON size: {len(json_str)} bytes")
        print(f"   Saved to: {json_path}")

        # Cleanup
        Path(json_path).unlink()

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="json_serialization",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_template_serialization_yaml(suite: TemplateTestSuite) -> None:
    """Test template YAML serialization and deserialization."""
    print("\n" + "-" * 50)
    print("Test: Template YAML Serialization")
    print("-" * 50)

    start_time = time.time()

    try:
        # Create template
        original = DocumentTemplate(
            name="yaml_test_template",
            schema_class=Receipt,
            description="Template for YAML serialization test",
            field_hints={"store_name": "Name of the establishment"},
            tags=["test", "yaml"],
        )

        # Serialize to YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        yaml_str = original.to_yaml(yaml_path)

        # Deserialize from YAML
        loaded = DocumentTemplate.from_yaml(yaml_path, Receipt)

        # Verify
        assert loaded.name == original.name
        assert loaded.description == original.description

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="yaml_serialization",
                success=True,
                latency_ms=latency_ms,
                details={"yaml_length": len(yaml_str)},
            )
        )

        print("✅ YAML serialization successful")
        print(f"   YAML size: {len(yaml_str)} bytes")
        print("   YAML content preview:")
        for line in yaml_str.split("\n")[:5]:
            print(f"     {line}")

        # Cleanup
        Path(yaml_path).unlink()

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="yaml_serialization",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_template_registry(suite: TemplateTestSuite) -> TemplateRegistry | None:
    """Test template registry operations."""
    print("\n" + "-" * 50)
    print("Test: Template Registry")
    print("-" * 50)

    start_time = time.time()
    registry = None

    try:
        registry = TemplateRegistry()

        # Create and register templates
        invoice_template = DocumentTemplate(
            name="invoice",
            schema_class=Invoice,
            description="Invoice extraction template",
            tags=["financial", "invoice"],
        )

        receipt_template = DocumentTemplate(
            name="receipt",
            schema_class=Receipt,
            description="Receipt extraction template",
            tags=["financial", "receipt"],
        )

        contact_template = DocumentTemplate(
            name="contact",
            schema_class=ContactCard,
            description="Contact card extraction template",
            tags=["contact", "personal"],
        )

        # Register templates
        registry.register(invoice_template)
        registry.register(receipt_template)
        registry.register(contact_template)

        # Test operations
        assert len(registry) == 3
        assert "invoice" in registry
        assert registry.get("invoice") is not None
        assert registry.get("nonexistent") is None

        # Test search by tags
        financial = registry.search_by_tags(["financial"])
        assert len(financial) == 2

        # Test search by description
        contact_results = registry.search_by_description("contact")
        assert len(contact_results) == 1

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="template_registry",
                success=True,
                latency_ms=latency_ms,
                details={
                    "registered_templates": len(registry),
                    "financial_templates": len(financial),
                },
            )
        )

        print("✅ Registry operations successful")
        print(f"   Registered templates: {len(registry)}")
        print(f"   Templates: {registry.list_templates()}")
        print(f"   Financial templates: {len(financial)}")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="template_registry",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")

    return registry


def test_template_inheritance(suite: TemplateTestSuite) -> None:
    """Test template inheritance via parent templates."""
    print("\n" + "-" * 50)
    print("Test: Template Inheritance")
    print("-" * 50)

    start_time = time.time()

    try:
        # Create parent template
        base_financial = DocumentTemplate(
            name="base_financial",
            schema_class=Invoice,
            description="Base financial document template",
            system_prompt="You are an expert financial document processor.",
            field_hints={
                "date": "Date in ISO format",
                "currency": "Three-letter currency code",
            },
            tags=["financial", "base"],
        )

        # Create child template that inherits
        specialized_invoice = DocumentTemplate(
            name="specialized_invoice",
            schema_class=Invoice,
            description="Specialized invoice template",
            field_hints={
                "invoice_number": "Look for INV- prefix",
            },
            tags=["invoice"],
            parent_template="base_financial",
        )

        # Merge with parent
        merged = specialized_invoice.merge_with_parent(base_financial)

        # Verify inheritance
        assert merged.system_prompt == base_financial.system_prompt
        assert "date" in (merged.field_hints or {})  # From parent
        assert "invoice_number" in (merged.field_hints or {})  # From child
        assert "financial" in (merged.tags or [])  # From parent
        assert "invoice" in (merged.tags or [])  # From child

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="template_inheritance",
                success=True,
                latency_ms=latency_ms,
                details={
                    "merged_hints": len(merged.field_hints or {}),
                    "merged_tags": len(merged.tags or []),
                },
            )
        )

        print("✅ Inheritance test passed")
        print(f"   Parent hints: {len(base_financial.field_hints or {})}")
        print(f"   Child hints: {len(specialized_invoice.field_hints or {})}")
        print(f"   Merged hints: {len(merged.field_hints or {})}")
        print(f"   Merged tags: {merged.tags}")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="template_inheritance",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_template_with_examples(suite: TemplateTestSuite) -> None:
    """Test template with few-shot examples."""
    print("\n" + "-" * 50)
    print("Test: Template with Examples")
    print("-" * 50)

    start_time = time.time()

    try:
        # Create template with examples
        template = DocumentTemplate(
            name="invoice_with_examples",
            schema_class=Invoice,
            description="Invoice template with few-shot examples",
            examples=[
                {
                    "input": "Invoice #ABC-123 from Vendor Inc. Total: $500.00",
                    "output": {
                        "invoice_number": "ABC-123",
                        "vendor_name": "Vendor Inc.",
                        "total_amount": 500.00,
                        "currency": "USD",
                    },
                },
                {
                    "input": "Bill #999 - Company XYZ - Amount Due: €250",
                    "output": {
                        "invoice_number": "999",
                        "vendor_name": "Company XYZ",
                        "total_amount": 250.00,
                        "currency": "EUR",
                    },
                },
            ],
        )

        # Validate
        warnings = template.validate_template()

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="template_with_examples",
                success=True,
                latency_ms=latency_ms,
                details={"example_count": len(template.examples or [])},
            )
        )

        print("✅ Template with examples created")
        print(f"   Examples: {len(template.examples or [])}")
        print(f"   Validation warnings: {len(warnings)}")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="template_with_examples",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_template_extraction(
    extractor: DocumentExtractor, suite: TemplateTestSuite
) -> None:
    """Test using template for extraction."""
    print("\n" + "-" * 50)
    print("Test: Template-based Extraction")
    print("-" * 50)

    start_time = time.time()

    try:
        # Create template
        template = DocumentTemplate(
            name="invoice_extraction",
            schema_class=Invoice,
            description="Invoice extraction template",
            system_prompt=(
                "You are an expert invoice processor. Extract invoice data precisely. "
                "Pay special attention to numerical values and dates."
            ),
            field_hints={
                "invoice_number": "Look for patterns like 'INV-XXXX', '#XXXX', or 'Invoice #'",
                "total_amount": "The final total amount due, after all taxes",
                "vendor_name": "The company that issued the invoice",
            },
        )

        # Extract using template
        result: ExtractionResult[Invoice] = extractor.extract(
            document=SAMPLE_INVOICE,
            template=template,
        )

        latency_ms = (time.time() - start_time) * 1000

        if result.success:
            suite.results.append(
                TemplateTestResult(
                    test_name="template_extraction",
                    success=True,
                    latency_ms=latency_ms,
                    details={
                        "invoice_number": result.data.invoice_number,
                        "vendor": result.data.vendor_name,
                        "total": result.data.total_amount,
                        "tokens": result.tokens_used,
                    },
                )
            )

            print("✅ Template extraction successful")
            print(f"   Invoice #: {result.data.invoice_number}")
            print(f"   Vendor: {result.data.vendor_name}")
            print(f"   Total: ${result.data.total_amount}")
            print(f"   Latency: {latency_ms:.0f}ms")
        else:
            suite.results.append(
                TemplateTestResult(
                    test_name="template_extraction",
                    success=False,
                    latency_ms=latency_ms,
                    error=result.error,
                )
            )
            print(f"❌ Extraction failed: {result.error}")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="template_extraction",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_multiple_templates(
    extractor: DocumentExtractor, suite: TemplateTestSuite
) -> None:
    """Test using multiple different templates."""
    print("\n" + "-" * 50)
    print("Test: Multiple Template Types")
    print("-" * 50)

    start_time = time.time()

    try:
        templates_and_docs = [
            (
                DocumentTemplate(
                    name="invoice_template",
                    schema_class=Invoice,
                    description="Invoice extraction",
                ),
                SAMPLE_INVOICE,
                "Invoice",
            ),
            (
                DocumentTemplate(
                    name="receipt_template",
                    schema_class=Receipt,
                    description="Receipt extraction",
                ),
                SAMPLE_RECEIPT,
                "Receipt",
            ),
            (
                DocumentTemplate(
                    name="contact_template",
                    schema_class=ContactCard,
                    description="Contact extraction",
                ),
                SAMPLE_CONTACT,
                "Contact",
            ),
        ]

        results_summary = []

        for template, doc, doc_type in templates_and_docs:
            result: ExtractionResult[Any] = extractor.extract(document=doc, template=template)
            results_summary.append({
                "type": doc_type,
                "success": result.success,
                "tokens": result.tokens_used,
            })
            print(f"   {doc_type}: {'✅' if result.success else '❌'}")

        latency_ms = (time.time() - start_time) * 1000

        all_success = all(r["success"] for r in results_summary)

        suite.results.append(
            TemplateTestResult(
                test_name="multiple_templates",
                success=all_success,
                latency_ms=latency_ms,
                details={"results": results_summary},
            )
        )

        if all_success:
            print("\n✅ All template extractions successful")
        else:
            print("\n⚠️ Some extractions failed")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="multiple_templates",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


def test_registry_with_extraction(
    extractor: DocumentExtractor, suite: TemplateTestSuite
) -> None:
    """Test using registry to manage templates for extraction."""
    print("\n" + "-" * 50)
    print("Test: Registry-based Extraction Workflow")
    print("-" * 50)

    start_time = time.time()

    try:
        # Create and populate registry
        registry = TemplateRegistry()

        registry.register(
            DocumentTemplate(
                name="invoice",
                schema_class=Invoice,
                description="Invoice extraction",
                tags=["financial"],
            )
        )

        registry.register(
            DocumentTemplate(
                name="receipt",
                schema_class=Receipt,
                description="Receipt extraction",
                tags=["financial"],
            )
        )

        # Use registry to get template and extract
        invoice_template = registry.get_or_raise("invoice")
        result: ExtractionResult[Invoice] = extractor.extract(
            document=SAMPLE_INVOICE, template=invoice_template
        )

        # Search and use
        financial_templates = registry.search_by_tags(["financial"])

        latency_ms = (time.time() - start_time) * 1000

        suite.results.append(
            TemplateTestResult(
                test_name="registry_extraction",
                success=result.success,
                latency_ms=latency_ms,
                details={
                    "registry_size": len(registry),
                    "financial_templates": len(financial_templates),
                    "extraction_success": result.success,
                },
            )
        )

        print("✅ Registry workflow successful")
        print(f"   Registry size: {len(registry)}")
        print(f"   Financial templates: {len(financial_templates)}")
        print(f"   Extraction: {'✅' if result.success else '❌'}")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        suite.results.append(
            TemplateTestResult(
                test_name="registry_extraction",
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        )
        print(f"❌ Failed: {e}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all E2E template tests."""
    print("\n📋 Structured Extractor - E2E Template Tests")
    print("=" * 60)

    # Check for API key (needed for extraction tests)
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize extractor for extraction tests
    extractor = DocumentExtractor(
        model="gpt-4.1",
        cache_dir="cache",
    )

    # Initialize results tracker
    suite = TemplateTestSuite(suite_name="Template Tests")

    # Run all tests
    print("\n📋 Running template tests...")

    # Non-extraction tests (no API calls)
    test_template_creation(suite)
    test_template_validation(suite)
    test_template_serialization_json(suite)
    test_template_serialization_yaml(suite)
    test_template_registry(suite)
    test_template_inheritance(suite)
    test_template_with_examples(suite)

    # Extraction tests (API calls)
    test_template_extraction(extractor, suite)
    test_multiple_templates(extractor, suite)
    test_registry_with_extraction(extractor, suite)

    # Print summary
    suite.print_summary()

    # Print failures
    failures = [r for r in suite.results if not r.success]
    if failures:
        print("\n❌ Failed Tests:")
        for f in failures:
            print(f"   - {f.test_name}: {f.error}")


if __name__ == "__main__":
    main()
