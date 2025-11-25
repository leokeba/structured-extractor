"""Example: Basic usage of structured-extractor."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import DocumentExtractor, ExtractionConfig

# Load environment variables
load_dotenv()


# Define extraction schemas
class Invoice(BaseModel):
    """Invoice data extraction template."""

    invoice_number: str = Field(description="The unique invoice identifier")
    date: str = Field(description="Invoice date in ISO format (YYYY-MM-DD)")
    vendor_name: str = Field(description="Name of the vendor/supplier")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")


class LineItem(BaseModel):
    """Individual line item."""

    description: str = Field(description="Item description")
    quantity: int = Field(default=1, description="Quantity")
    unit_price: float = Field(description="Price per unit")
    total: float = Field(description="Line total")


class DetailedInvoice(BaseModel):
    """Detailed invoice with line items."""

    invoice_number: str
    date: str
    vendor_name: str
    vendor_address: str | None = Field(default=None)
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float


def example_basic_extraction() -> None:
    """Demonstrate basic extraction from a document."""
    print("=" * 60)
    print("Example 1: Basic Invoice Extraction")
    print("=" * 60)

    # Initialize extractor
    extractor = DocumentExtractor(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",
        cache_dir="cache",
    )

    # Sample document
    document = """
    INVOICE #INV-2024-0042

    Date: November 25, 2024
    From: Acme Corporation
          123 Business Street
          San Francisco, CA 94105

    Bill To: TechStart Inc.

    Items:
    - Consulting Services: $5,000.00
    - Software License: $2,500.00

    Subtotal: $7,500.00
    Tax (10%): $750.00
    Total Due: $8,250.00

    Payment Terms: Net 30
    """

    # Extract structured data
    result = extractor.extract(document, schema=Invoice)

    if result.success:
        print(f"Invoice Number: {result.data.invoice_number}")
        print(f"Date: {result.data.date}")
        print(f"Vendor: {result.data.vendor_name}")
        print(f"Total: {result.data.currency} {result.data.total_amount}")
        print(f"\nCached: {result.cached}")
        print(f"Tokens used: {result.tokens_used}")
    else:
        print(f"Extraction failed: {result.error}")


def example_nested_extraction() -> None:
    """Demonstrate extraction with nested structures."""
    print("\n" + "=" * 60)
    print("Example 2: Detailed Invoice with Line Items")
    print("=" * 60)

    extractor = DocumentExtractor(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1",
        cache_dir="cache",
    )

    document = """
    INVOICE #INV-2024-0099

    Date: 2024-11-25
    Vendor: Cloud Services Inc.
    Address: 456 Tech Park, Austin, TX 78701

    Line Items:
    1. Web Hosting (12 months) - Qty: 1 - $120.00/mo - Total: $1,440.00
    2. SSL Certificate - Qty: 2 - $50.00/ea - Total: $100.00
    3. Domain Registration - Qty: 5 - $15.00/ea - Total: $75.00

    Subtotal: $1,615.00
    Tax (8.25%): $133.24
    Grand Total: $1,748.24
    """

    result = extractor.extract(document, schema=DetailedInvoice)

    if result.success:
        print(f"Invoice: {result.data.invoice_number}")
        print(f"Vendor: {result.data.vendor_name}")
        print("\nLine Items:")
        for item in result.data.line_items:
            print(f"  - {item.description}: ${item.total:.2f}")
        print(f"\nSubtotal: ${result.data.subtotal:.2f}")
        print(f"Tax: ${result.data.tax:.2f}")
        print(f"Total: ${result.data.total:.2f}")
    else:
        print(f"Extraction failed: {result.error}")


def example_custom_config() -> None:
    """Demonstrate extraction with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    extractor = DocumentExtractor(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-mini",  # Use smaller model
        cache_dir="cache",
    )

    config = ExtractionConfig(
        temperature=0.0,  # Deterministic
        max_retries=2,
        system_prompt="You are an expert financial document parser. Be precise with numbers.",
    )

    document = """
    Quick Invoice
    INV-QUICK-001
    ABC Supplies - Total $499.99
    Date: Nov 25, 2024
    """

    result = extractor.extract(document, schema=Invoice, config=config)

    if result.success:
        print(f"Extracted: {result.data.model_dump_json(indent=2)}")
        print(f"Model used: {result.model_used}")
    else:
        print(f"Failed: {result.error}")


if __name__ == "__main__":
    example_basic_extraction()
    example_nested_extraction()
    example_custom_config()
