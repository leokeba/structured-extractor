"""Built-in document templates factory.

Provides pre-configured templates for common document types
with optimized prompts and field hints.
"""

from structured_extractor.core.templates import DocumentTemplate, TemplateRegistry
from structured_extractor.schemas.contract import ContractSchema
from structured_extractor.schemas.invoice import InvoiceSchema
from structured_extractor.schemas.receipt import ReceiptSchema
from structured_extractor.schemas.resume import ResumeSchema


class BuiltinTemplates:
    """Factory class for built-in document templates.

    Provides pre-configured templates for common document types
    with optimized prompts and field hints.

    Example:
        ```python
        from structured_extractor import BuiltinTemplates, DocumentExtractor

        # Get a built-in template
        template = BuiltinTemplates.invoice()

        # Use with extractor
        extractor = DocumentExtractor()
        result = extractor.extract(document_text, template=template)
        ```
    """

    @staticmethod
    def invoice() -> DocumentTemplate:
        """Create a template for invoice extraction.

        Returns:
            DocumentTemplate configured for invoice documents.
        """
        return DocumentTemplate(
            name="builtin_invoice",
            schema_class=InvoiceSchema,
            description="Extract structured data from invoices and bills",
            system_prompt=(
                "You are an expert financial document processor specializing in invoices. "
                "Extract structured data from invoices with high precision.\n\n"
                "Guidelines:\n"
                "1. Pay special attention to monetary values and currencies\n"
                "2. Extract dates in ISO format (YYYY-MM-DD)\n"
                "3. Distinguish between invoice number, PO number, and reference numbers\n"
                "4. Extract all line items with quantities and prices\n"
                "5. Identify vendor and customer information separately\n"
                "6. Note tax amounts, discounts, and additional charges"
            ),
            field_hints={
                "invoice_number": "Look for 'Invoice #', 'Invoice No.', 'Inv #', or similar",
                "date": "The invoice date, not due date or order date",
                "total_amount": "The final total including all taxes and fees",
                "subtotal": "Amount before tax is applied",
                "vendor_name": "The company issuing the invoice",
                "customer_name": "The company or person being billed",
            },
            examples=[
                {
                    "input": "INVOICE #12345\nDate: 2024-01-15\nACME Corp\nTotal: $1,500.00",
                    "output": {
                        "invoice_number": "12345",
                        "date": "2024-01-15",
                        "vendor_name": "ACME Corp",
                        "total_amount": 1500.0,
                    },
                }
            ],
            tags=["financial", "invoice", "billing"],
        )

    @staticmethod
    def receipt() -> DocumentTemplate:
        """Create a template for receipt extraction.

        Returns:
            DocumentTemplate configured for receipt documents.
        """
        return DocumentTemplate(
            name="builtin_receipt",
            schema_class=ReceiptSchema,
            description="Extract structured data from receipts and purchase records",
            system_prompt=(
                "You are an expert at extracting information from receipts. "
                "Extract all relevant purchase information accurately.\n\n"
                "Guidelines:\n"
                "1. Extract the merchant name and address if available\n"
                "2. List all purchased items\n"
                "3. Note subtotal, tax, and total amounts\n"
                "4. Identify payment method if shown\n"
                "5. Extract date and time of transaction"
            ),
            field_hints={
                "merchant_name": "Store or business name at the top of receipt",
                "total": "Final amount paid including tax",
                "date": "Transaction date, often near the top or bottom",
                "items": "List of items purchased with prices",
            },
            tags=["financial", "receipt", "purchase"],
        )

    @staticmethod
    def resume() -> DocumentTemplate:
        """Create a template for resume/CV extraction.

        Returns:
            DocumentTemplate configured for resume documents.
        """
        return DocumentTemplate(
            name="builtin_resume",
            schema_class=ResumeSchema,
            description="Extract structured data from resumes and CVs",
            system_prompt=(
                "You are an expert HR document processor specializing in resume analysis. "
                "Extract professional and educational information accurately.\n\n"
                "Guidelines:\n"
                "1. Extract work experience with dates and company names\n"
                "2. List education with degrees and institutions\n"
                "3. Identify technical and soft skills\n"
                "4. Note certifications with issuing organizations\n"
                "5. Preserve chronological order of experiences\n"
                "6. Extract contact information if available"
            ),
            field_hints={
                "name": "The candidate's full name, usually at the top",
                "work_experience": "Job history with companies, titles, and dates",
                "skills": "Technical skills, tools, languages, and soft skills",
                "education": "Schools, degrees, and graduation dates",
            },
            tags=["hr", "resume", "cv", "recruitment"],
        )

    @staticmethod
    def contract() -> DocumentTemplate:
        """Create a template for contract extraction.

        Returns:
            DocumentTemplate configured for legal contracts.
        """
        return DocumentTemplate(
            name="builtin_contract",
            schema_class=ContractSchema,
            description="Extract key terms and clauses from legal contracts",
            system_prompt=(
                "You are an expert legal document analyst specializing in contracts. "
                "Extract key contractual terms and clauses accurately.\n\n"
                "Guidelines:\n"
                "1. Identify all parties with their full legal names\n"
                "2. Extract effective date, term duration, and termination clauses\n"
                "3. Note all monetary values, payment terms, and penalties\n"
                "4. Identify governing law and jurisdiction\n"
                "5. Extract key obligations of each party\n"
                "6. Note special conditions, warranties, or limitations"
            ),
            field_hints={
                "effective_date": "When the contract becomes legally effective",
                "parties": "All legal entities that are party to this agreement",
                "term_duration": "How long the agreement lasts",
                "governing_law": "Which jurisdiction's laws govern the contract",
            },
            tags=["legal", "contract", "agreement"],
        )

    @classmethod
    def get_all(cls) -> dict[str, DocumentTemplate]:
        """Get all built-in templates.

        Returns:
            Dictionary mapping template names to DocumentTemplate instances.
        """
        return {
            "invoice": cls.invoice(),
            "receipt": cls.receipt(),
            "resume": cls.resume(),
            "contract": cls.contract(),
        }

    @classmethod
    def create_registry(cls) -> TemplateRegistry:
        """Create a registry pre-populated with all built-in templates.

        Returns:
            TemplateRegistry with all built-in templates registered.

        Example:
            ```python
            registry = BuiltinTemplates.create_registry()
            invoice_template = registry.get("builtin_invoice")
            ```
        """
        registry = TemplateRegistry()
        for template in cls.get_all().values():
            registry.register(template)
        return registry
