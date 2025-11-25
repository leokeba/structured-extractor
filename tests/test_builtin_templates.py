"""Tests for built-in templates and schemas."""


from structured_extractor import (
    BuiltinTemplates,
    ContractParty,
    ContractSchema,
    DocumentTemplate,
    Education,
    InvoiceLineItem,
    InvoiceSchema,
    ReceiptSchema,
    ResumeSchema,
    TemplateRegistry,
    WorkExperience,
)

# ============================================================================
# Built-in Schema Tests
# ============================================================================


class TestInvoiceSchema:
    """Tests for InvoiceSchema."""

    def test_minimal_invoice(self):
        """Test creating invoice with minimal required fields."""
        invoice = InvoiceSchema(
            invoice_number="INV-001",
            date="2024-01-15",
            vendor_name="ACME Corp",
            total_amount=100.0,
        )

        assert invoice.invoice_number == "INV-001"
        assert invoice.currency == "USD"  # default
        assert invoice.line_items == []  # default

    def test_full_invoice(self):
        """Test creating invoice with all fields."""
        invoice = InvoiceSchema(
            invoice_number="INV-001",
            date="2024-01-15",
            due_date="2024-02-15",
            vendor_name="ACME Corp",
            vendor_address="123 Main St",
            customer_name="TechStart Inc",
            customer_address="456 Oak Ave",
            line_items=[
                InvoiceLineItem(
                    description="Consulting",
                    quantity=10,
                    unit_price=100.0,
                    total=1000.0,
                )
            ],
            subtotal=1000.0,
            tax_amount=100.0,
            tax_rate=10.0,
            total_amount=1100.0,
            currency="EUR",
            payment_terms="Net 30",
            notes="Thank you for your business",
        )

        assert invoice.total_amount == 1100.0
        assert len(invoice.line_items) == 1
        assert invoice.currency == "EUR"

    def test_line_item(self):
        """Test InvoiceLineItem schema."""
        item = InvoiceLineItem(
            description="Software License",
            quantity=5,
            unit_price=200.0,
            total=1000.0,
        )

        assert item.description == "Software License"
        assert item.quantity == 5


class TestReceiptSchema:
    """Tests for ReceiptSchema."""

    def test_minimal_receipt(self):
        """Test creating receipt with minimal fields."""
        receipt = ReceiptSchema(
            merchant_name="Coffee Shop",
            date="2024-01-15",
            total=5.50,
        )

        assert receipt.merchant_name == "Coffee Shop"
        assert receipt.items == []

    def test_full_receipt(self):
        """Test creating receipt with all fields."""
        receipt = ReceiptSchema(
            merchant_name="Grocery Store",
            merchant_address="123 Main St",
            date="2024-01-15",
            time="14:30",
            items=["Milk", "Bread", "Eggs"],
            subtotal=10.0,
            tax=0.80,
            total=10.80,
            payment_method="Credit Card",
            card_last_four="1234",
        )

        assert len(receipt.items) == 3
        assert receipt.card_last_four == "1234"


class TestResumeSchema:
    """Tests for ResumeSchema."""

    def test_minimal_resume(self):
        """Test creating resume with minimal fields."""
        resume = ResumeSchema(name="John Doe")

        assert resume.name == "John Doe"
        assert resume.skills == []
        assert resume.work_experience == []

    def test_full_resume(self):
        """Test creating resume with work experience and education."""
        resume = ResumeSchema(
            name="John Doe",
            email="john@example.com",
            phone="555-1234",
            location="New York, NY",
            summary="Experienced software engineer",
            work_experience=[
                WorkExperience(
                    company="Tech Corp",
                    title="Senior Engineer",
                    start_date="2020-01",
                    end_date="Present",
                    achievements=["Led team of 5", "Increased performance 50%"],
                )
            ],
            education=[
                Education(
                    institution="MIT",
                    degree="B.S. Computer Science",
                    graduation_date="2015",
                )
            ],
            skills=["Python", "JavaScript", "SQL"],
            certifications=["AWS Certified"],
            languages=["English", "Spanish"],
        )

        assert len(resume.work_experience) == 1
        assert resume.work_experience[0].company == "Tech Corp"
        assert len(resume.education) == 1

    def test_work_experience(self):
        """Test WorkExperience schema."""
        exp = WorkExperience(
            company="Startup Inc",
            title="Developer",
            start_date="2018-06",
            description="Full-stack development",
        )

        assert exp.company == "Startup Inc"
        assert exp.achievements == []

    def test_education(self):
        """Test Education schema."""
        edu = Education(
            institution="Stanford University",
            degree="Master's in Computer Science",
            field_of_study="AI/ML",
            gpa=3.9,
        )

        assert edu.institution == "Stanford University"
        assert edu.gpa == 3.9


class TestContractSchema:
    """Tests for ContractSchema."""

    def test_minimal_contract(self):
        """Test creating contract with minimal fields."""
        contract = ContractSchema()

        assert contract.parties == []
        assert contract.key_obligations == []

    def test_full_contract(self):
        """Test creating contract with all fields."""
        contract = ContractSchema(
            title="Service Agreement",
            contract_type="Service Contract",
            parties=[
                ContractParty(name="Company A", role="Provider"),
                ContractParty(name="Company B", role="Client"),
            ],
            effective_date="2024-01-01",
            termination_date="2025-01-01",
            term_duration="1 year",
            total_value=50000.0,
            currency="USD",
            payment_terms="Monthly",
            governing_law="New York",
            key_obligations=["Deliver services", "Pay on time"],
            termination_clauses=["30-day notice required"],
        )

        assert len(contract.parties) == 2
        assert contract.total_value == 50000.0

    def test_contract_party(self):
        """Test ContractParty schema."""
        party = ContractParty(
            name="ACME Corporation",
            role="Buyer",
            address="123 Business Ave",
        )

        assert party.name == "ACME Corporation"


# ============================================================================
# BuiltinTemplates Tests
# ============================================================================


class TestBuiltinTemplates:
    """Tests for BuiltinTemplates factory class."""

    def test_invoice_template(self):
        """Test invoice template creation."""
        template = BuiltinTemplates.invoice()

        assert template.name == "builtin_invoice"
        assert template.schema_class is InvoiceSchema
        assert "financial" in template.tags
        assert "invoice" in template.tags
        assert template.system_prompt is not None
        assert template.field_hints is not None
        assert "invoice_number" in template.field_hints

    def test_receipt_template(self):
        """Test receipt template creation."""
        template = BuiltinTemplates.receipt()

        assert template.name == "builtin_receipt"
        assert template.schema_class is ReceiptSchema
        assert "receipt" in template.tags
        assert template.field_hints is not None

    def test_resume_template(self):
        """Test resume template creation."""
        template = BuiltinTemplates.resume()

        assert template.name == "builtin_resume"
        assert template.schema_class is ResumeSchema
        assert "hr" in template.tags
        assert "resume" in template.tags

    def test_contract_template(self):
        """Test contract template creation."""
        template = BuiltinTemplates.contract()

        assert template.name == "builtin_contract"
        assert template.schema_class is ContractSchema
        assert "legal" in template.tags

    def test_get_all_templates(self):
        """Test getting all built-in templates."""
        templates = BuiltinTemplates.get_all()

        assert "invoice" in templates
        assert "receipt" in templates
        assert "resume" in templates
        assert "contract" in templates
        assert len(templates) == 4

        # Verify each is a DocumentTemplate
        for template in templates.values():
            assert isinstance(template, DocumentTemplate)

    def test_create_registry(self):
        """Test creating pre-populated registry."""
        registry = BuiltinTemplates.create_registry()

        assert isinstance(registry, TemplateRegistry)
        assert len(registry) == 4
        assert "builtin_invoice" in registry
        assert "builtin_receipt" in registry
        assert "builtin_resume" in registry
        assert "builtin_contract" in registry

    def test_templates_validate(self):
        """Test that all built-in templates pass validation."""
        for name, template in BuiltinTemplates.get_all().items():
            # Should not raise
            warnings = template.validate_template()
            # No critical warnings expected
            assert not any("error" in w.lower() for w in warnings), f"{name}: {warnings}"

    def test_templates_have_examples(self):
        """Test that invoice template has examples."""
        template = BuiltinTemplates.invoice()
        assert template.examples is not None
        assert len(template.examples) > 0

    def test_template_system_prompts_not_empty(self):
        """Test that all templates have non-empty system prompts."""
        for name, template in BuiltinTemplates.get_all().items():
            assert template.system_prompt, f"{name} missing system prompt"
            assert len(template.system_prompt) > 50, f"{name} system prompt too short"


class TestBuiltinTemplatesIntegration:
    """Integration tests for built-in templates with registry."""

    def test_search_financial_templates(self):
        """Test searching for financial templates."""
        registry = BuiltinTemplates.create_registry()

        financial = registry.search_by_tags(["financial"])

        assert len(financial) == 2  # invoice and receipt
        names = [t.name for t in financial]
        assert "builtin_invoice" in names
        assert "builtin_receipt" in names

    def test_search_by_description(self):
        """Test searching templates by description."""
        registry = BuiltinTemplates.create_registry()

        results = registry.search_by_description("invoice")

        assert len(results) >= 1
        assert any(t.name == "builtin_invoice" for t in results)

    def test_can_extend_builtin_template(self):
        """Test that built-in templates can be extended via inheritance."""
        from pydantic import Field

        class ExtendedInvoice(InvoiceSchema):
            """Extended invoice with custom field."""

            purchase_order: str | None = Field(
                default=None, description="PO number"
            )

        parent = BuiltinTemplates.invoice()
        child = DocumentTemplate(
            name="custom_invoice",
            schema_class=ExtendedInvoice,
            field_hints={"purchase_order": "Look for PO# or Purchase Order"},
            parent_template=parent.name,
        )

        merged = child.merge_with_parent(parent)

        # Should have parent's hints plus child's new one
        assert "invoice_number" in merged.field_hints  # from parent
        assert "purchase_order" in merged.field_hints  # from child
