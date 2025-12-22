"""End-to-end test: Image-based extraction from forms and documents.

This script demonstrates extracting structured data from images of:
1. Invoices
2. Forms
3. Receipts
4. Business cards

Prerequisites:
- OPENAI_API_KEY environment variable set
- GPT-4.1 model access (for vision capabilities)

Usage:
    python examples/e2e_image_extraction.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import DocumentExtractor, ExtractionError

# Load environment variables
load_dotenv()

# Define the assets directory
ASSETS_DIR = Path(__file__).parent / "assets"


# ============================================================================
# Schema Definitions
# ============================================================================


class InvoiceData(BaseModel):
    """Structured invoice data extracted from an image."""

    invoice_number: str | None = Field(default=None, description="The unique invoice identifier")
    invoice_date: str | None = Field(
        default=None, description="Invoice date (YYYY-MM-DD format if possible)"
    )
    due_date: str | None = Field(default=None, description="Payment due date")
    vendor_name: str = Field(description="Name of the vendor/seller")
    vendor_address: str | None = Field(default=None, description="Full address of the vendor")
    customer_name: str | None = Field(default=None, description="Name of the customer/buyer")
    subtotal: float | None = Field(default=None, description="Subtotal before tax")
    tax_amount: float | None = Field(default=None, description="Tax amount")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code (e.g., USD)")


class LineItem(BaseModel):
    """Individual line item from an invoice."""

    description: str = Field(description="Item description")
    quantity: float | None = Field(default=None, description="Quantity")
    unit_price: float | None = Field(default=None, description="Price per unit")
    total: float = Field(description="Line total")


class DetailedInvoice(BaseModel):
    """Detailed invoice with line items."""

    invoice_number: str | None = Field(default=None, description="Invoice number")
    invoice_date: str | None = Field(default=None, description="Invoice date")
    vendor_name: str = Field(description="Vendor/company name")
    line_items: list[LineItem] = Field(default_factory=list, description="List of line items")
    subtotal: float | None = Field(default=None, description="Subtotal before tax")
    tax: float | None = Field(default=None, description="Tax amount")
    total: float = Field(description="Total amount")


class ReceiptData(BaseModel):
    """Receipt data from a store or restaurant."""

    store_name: str = Field(description="Name of the store/restaurant")
    store_address: str | None = Field(default=None, description="Store address")
    date: str | None = Field(default=None, description="Transaction date")
    time: str | None = Field(default=None, description="Transaction time")
    items: list[LineItem] = Field(default_factory=list, description="List of purchased items")
    subtotal: float | None = Field(default=None, description="Subtotal before tax")
    tax: float | None = Field(default=None, description="Tax amount")
    tip: float | None = Field(default=None, description="Tip amount if applicable")
    total: float = Field(description="Total amount")
    payment_method: str | None = Field(default=None, description="Payment method used")


class BusinessCard(BaseModel):
    """Business card information."""

    name: str = Field(description="Person's full name")
    title: str | None = Field(default=None, description="Job title")
    company: str | None = Field(default=None, description="Company name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    mobile: str | None = Field(default=None, description="Mobile/cell phone")
    website: str | None = Field(default=None, description="Website URL")
    address: str | None = Field(default=None, description="Full address")


class W2Form(BaseModel):
    """W-2 tax form data."""

    employee_name: str = Field(description="Employee's name")
    employee_ssn_last4: str | None = Field(
        default=None, description="Last 4 digits of SSN (for verification)"
    )
    employer_name: str = Field(description="Employer's name")
    employer_ein: str | None = Field(default=None, description="Employer ID number")
    wages: float = Field(description="Box 1: Wages, tips, other compensation")
    federal_tax_withheld: float = Field(description="Box 2: Federal income tax withheld")
    social_security_wages: float | None = Field(default=None, description="Box 3")
    social_security_tax: float | None = Field(default=None, description="Box 4")
    medicare_wages: float | None = Field(default=None, description="Box 5")
    medicare_tax: float | None = Field(default=None, description="Box 6")
    tax_year: int = Field(description="Tax year (e.g., 2024)")


# ============================================================================
# Test Functions
# ============================================================================


def test_invoice_extraction(extractor: DocumentExtractor, image_path: Path) -> None:
    """Test extracting data from an invoice image."""
    print("\n" + "=" * 60)
    print("Test: Invoice Extraction")
    print(f"Image: {image_path}")
    print("=" * 60)

    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Please add a sample invoice image to test.")
        return

    # Note: Could use PromptBuilder(template=PromptTemplates.invoice())
    # for a more specialized invoice extraction

    try:
        result = extractor.extract_from_image(
            image=image_path,
            schema=InvoiceData,
            field_hints={
                "invoice_number": "Look for 'Invoice #', 'INV-', or similar patterns",
                "total_amount": "The final total including all taxes and fees",
            },
        )

        print("\n✅ Extraction successful!")
        print("\n📄 Extracted Data:")
        print(f"   Invoice #: {result.data.invoice_number}")
        print(f"   Date: {result.data.invoice_date}")
        print(f"   Vendor: {result.data.vendor_name}")
        print(f"   Customer: {result.data.customer_name}")
        print(f"   Subtotal: {result.data.currency} {result.data.subtotal}")
        print(f"   Tax: {result.data.currency} {result.data.tax_amount}")
        print(f"   Total: {result.data.currency} {result.data.total_amount}")
        print("\n📊 Metadata:")
        print(f"   Model: {result.model_used}")
        print(f"   Tokens: {result.tokens_used}")
        print(f"   Cost: ${result.cost_usd:.4f}" if result.cost_usd else "   Cost: N/A")
        print(f"   Cached: {result.cached}")
    except ExtractionError as e:
        print(f"\n❌ Extraction failed: {e}")


def test_receipt_extraction(extractor: DocumentExtractor, image_path: Path) -> None:
    """Test extracting data from a receipt image."""
    print("\n" + "=" * 60)
    print("Test: Receipt Extraction")
    print(f"Image: {image_path}")
    print("=" * 60)

    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        return

    try:
        result = extractor.extract_from_image(
            image=image_path,
            schema=ReceiptData,
            additional_context="This is a receipt from a store or restaurant.",
        )

        print("\n✅ Extraction successful!")
        print("\n🧾 Extracted Data:")
        print(f"   Store: {result.data.store_name}")
        print(f"   Date: {result.data.date} {result.data.time or ''}")
        print(f"   Items: {len(result.data.items)}")
        for item in result.data.items[:5]:  # Show first 5 items
            print(f"     - {item.description}: ${item.total:.2f}")
        if len(result.data.items) > 5:
            print(f"     ... and {len(result.data.items) - 5} more items")
        print(f"   Subtotal: ${result.data.subtotal:.2f}" if result.data.subtotal else "")
        print(f"   Tax: ${result.data.tax:.2f}" if result.data.tax else "")
        print(f"   Total: ${result.data.total:.2f}")
    except ExtractionError as e:
        print(f"\n❌ Extraction failed: {e}")


def test_business_card_extraction(extractor: DocumentExtractor, image_path: Path) -> None:
    """Test extracting data from a business card image."""
    print("\n" + "=" * 60)
    print("Test: Business Card Extraction")
    print(f"Image: {image_path}")
    print("=" * 60)

    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        return

    try:
        result = extractor.extract_from_image(
            image=image_path,
            schema=BusinessCard,
            additional_context="Extract contact information from this business card.",
        )

        print("\n✅ Extraction successful!")
        print("\n👤 Contact Info:")
        print(f"   Name: {result.data.name}")
        print(f"   Title: {result.data.title}")
        print(f"   Company: {result.data.company}")
        print(f"   Email: {result.data.email}")
        print(f"   Phone: {result.data.phone}")
        print(f"   Mobile: {result.data.mobile}")
        print(f"   Website: {result.data.website}")
    except ExtractionError as e:
        print(f"\n❌ Extraction failed: {e}")


def test_url_image_extraction(extractor: DocumentExtractor) -> None:
    """Test extracting from an image URL."""
    print("\n" + "=" * 60)
    print("Test: URL Image Extraction")
    print("=" * 60)

    # Use a sample invoice image from Wikipedia (public domain)
    # This is a W-4 tax form image from the IRS (public domain)
    sample_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/"
        "Form_W-4%2C_2020.pdf/page1-463px-Form_W-4%2C_2020.pdf.jpg"
    )

    print(f"Image URL: {sample_url}")

    class BasicFormData(BaseModel):
        """Basic form data extraction."""

        form_name: str | None = Field(default=None, description="Name/title of the form")
        form_number: str | None = Field(default=None, description="Form number (e.g., W-4)")
        year: int | None = Field(default=None, description="Tax year if applicable")
        issuing_agency: str | None = Field(default=None, description="Agency that issues the form")

    try:
        result = extractor.extract_from_image(
            image=sample_url,
            schema=BasicFormData,
            additional_context="This is a tax form. Extract basic identifying information.",
        )

        print("\n✅ Extraction successful!")
        print(f"   Form Name: {result.data.form_name}")
        print(f"   Form Number: {result.data.form_number}")
        print(f"   Year: {result.data.year}")
        print(f"   Agency: {result.data.issuing_agency}")
        print(f"   Tokens: {result.tokens_used}")
    except ExtractionError as e:
        print(f"\n❌ Extraction failed: {e}")


def test_multi_image_extraction(extractor: DocumentExtractor, image_paths: list[Path]) -> None:
    """Test extracting from multiple images at once."""
    print("\n" + "=" * 60)
    print("Test: Multi-Image Extraction")
    print("=" * 60)

    existing_images = [p for p in image_paths if p.exists()]

    if len(existing_images) < 2:
        print("⚠️  Need at least 2 images for multi-image test")
        return

    print(f"Processing {len(existing_images)} images...")

    try:
        result = extractor.extract_from_image(
            image=existing_images,
            schema=DetailedInvoice,
            additional_context=(
                "These images may be different pages of the same invoice. "
                "Combine the information from all pages."
            ),
        )

        print("\n✅ Extraction successful!")
        print(f"   Combined from {len(existing_images)} images")
        print(f"   Line items found: {len(result.data.line_items)}")
        print(f"   Total: ${result.data.total:.2f}")
    except ExtractionError as e:
        print(f"\n❌ Extraction failed: {e}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all E2E image extraction tests."""
    print("\n🚀 Structured Extractor - E2E Image Extraction Tests")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key to run these tests.")
        return

    # Initialize extractor with GPT-4.1 (vision-capable)
    extractor = DocumentExtractor(
        model="gpt-4.1",
        cache_dir="cache",
    )

    # Define test image paths
    invoice_image = ASSETS_DIR / "sample_invoice.png"
    receipt_image = ASSETS_DIR / "sample_receipt.png"
    business_card_image = ASSETS_DIR / "sample_business_card.png"

    # Run tests
    print("\n📋 Running extraction tests...")
    print(f"   Assets directory: {ASSETS_DIR}")

    # Test 1: Invoice from local file
    test_invoice_extraction(extractor, invoice_image)

    # Test 2: Receipt
    test_receipt_extraction(extractor, receipt_image)

    # Test 3: Business card
    test_business_card_extraction(extractor, business_card_image)

    # Test 4: URL-based image (always works, no local file needed)
    test_url_image_extraction(extractor)

    # Test 5: Multi-image (if multiple images exist)
    test_multi_image_extraction(
        extractor,
        [invoice_image, receipt_image],
    )

    print("\n" + "=" * 60)
    print("✅ E2E Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
