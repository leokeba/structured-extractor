"""Invoice extraction schema.

Provides a standard schema for extracting structured data from invoices
and billing documents.
"""

from pydantic import BaseModel, Field


class InvoiceLineItem(BaseModel):
    """A line item in an invoice."""

    description: str = Field(description="Description of the item or service")
    quantity: float = Field(default=1.0, description="Quantity of items")
    unit_price: float = Field(description="Price per unit")
    total: float = Field(description="Total for this line item")


class InvoiceSchema(BaseModel):
    """Standard invoice extraction schema.

    Use this schema to extract structured data from invoices,
    bills, and similar financial documents.

    Example:
        ```python
        from structured_extractor import DocumentExtractor, InvoiceSchema

        extractor = DocumentExtractor()
        result = extractor.extract(invoice_text, schema=InvoiceSchema)
        print(result.data.total_amount)
        ```
    """

    invoice_number: str = Field(description="The unique invoice identifier")
    date: str = Field(description="Invoice date in ISO format (YYYY-MM-DD)")
    due_date: str | None = Field(default=None, description="Payment due date")
    vendor_name: str = Field(description="Name of the vendor/supplier")
    vendor_address: str | None = Field(default=None, description="Vendor's address")
    customer_name: str | None = Field(default=None, description="Customer/bill-to name")
    customer_address: str | None = Field(default=None, description="Customer's address")
    line_items: list[InvoiceLineItem] = Field(
        default_factory=list, description="Individual line items"
    )
    subtotal: float | None = Field(default=None, description="Subtotal before tax")
    tax_amount: float | None = Field(default=None, description="Tax amount")
    tax_rate: float | None = Field(default=None, description="Tax rate as percentage")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code (e.g., USD, EUR)")
    payment_terms: str | None = Field(default=None, description="Payment terms")
    notes: str | None = Field(default=None, description="Additional notes")
