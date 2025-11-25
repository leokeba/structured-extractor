"""Generate sample test images for E2E testing."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_sample_invoice(output_path: Path) -> None:
    """Create a sample invoice image for testing."""
    # Create a white background image
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a built-in font, fall back to default
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Company header
    draw.text((50, 50), "ACME Corporation", fill="black", font=font_large)
    draw.text((50, 80), "123 Business Street", fill="gray", font=font_small)
    draw.text((50, 95), "San Francisco, CA 94105", fill="gray", font=font_small)

    # Invoice title
    draw.text((600, 50), "INVOICE", fill="navy", font=font_large)

    # Invoice details
    draw.text((500, 100), "Invoice #: INV-2024-0042", fill="black", font=font_medium)
    draw.text((500, 125), "Date: November 25, 2024", fill="black", font=font_medium)
    draw.text((500, 150), "Due Date: December 25, 2024", fill="black", font=font_medium)

    # Bill To section
    draw.text((50, 180), "Bill To:", fill="gray", font=font_small)
    draw.text((50, 200), "TechStart Inc.", fill="black", font=font_medium)
    draw.text((50, 220), "456 Innovation Ave", fill="black", font=font_small)
    draw.text((50, 235), "Austin, TX 78701", fill="black", font=font_small)

    # Line separator
    draw.line([(50, 280), (750, 280)], fill="gray", width=1)

    # Table header
    draw.text((50, 300), "Description", fill="gray", font=font_medium)
    draw.text((400, 300), "Qty", fill="gray", font=font_medium)
    draw.text((500, 300), "Unit Price", fill="gray", font=font_medium)
    draw.text((650, 300), "Total", fill="gray", font=font_medium)

    # Line items
    items = [
        ("Consulting Services", "10 hrs", "$150.00", "$1,500.00"),
        ("Software License (Annual)", "1", "$2,500.00", "$2,500.00"),
        ("Cloud Hosting (Monthly)", "12", "$99.00", "$1,188.00"),
        ("Technical Support", "1", "$500.00", "$500.00"),
    ]

    y = 340
    for desc, qty, unit, total in items:
        draw.text((50, y), desc, fill="black", font=font_small)
        draw.text((400, y), qty, fill="black", font=font_small)
        draw.text((500, y), unit, fill="black", font=font_small)
        draw.text((650, y), total, fill="black", font=font_small)
        y += 30

    # Line separator
    draw.line([(400, y + 10), (750, y + 10)], fill="gray", width=1)

    # Totals
    y += 30
    draw.text((500, y), "Subtotal:", fill="black", font=font_medium)
    draw.text((650, y), "$5,688.00", fill="black", font=font_medium)

    y += 25
    draw.text((500, y), "Tax (8.5%):", fill="black", font=font_medium)
    draw.text((650, y), "$483.48", fill="black", font=font_medium)

    y += 30
    draw.text((500, y), "TOTAL:", fill="navy", font=font_large)
    draw.text((650, y), "$6,171.48", fill="navy", font=font_large)

    # Footer
    draw.text((50, 900), "Payment Terms: Net 30", fill="gray", font=font_small)
    draw.text((50, 920), "Thank you for your business!", fill="gray", font=font_small)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"✅ Created: {output_path}")


def create_sample_receipt(output_path: Path) -> None:
    """Create a sample receipt image for testing."""
    img = Image.new("RGB", (400, 600), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Store header
    draw.text((100, 30), "QUICK MART", fill="black", font=font_large)
    draw.text((100, 55), "789 Main Street", fill="gray", font=font_small)
    draw.text((100, 70), "Boston, MA 02101", fill="gray", font=font_small)

    # Date/Time
    draw.text((50, 110), "Date: 11/25/2024", fill="black", font=font_small)
    draw.text((250, 110), "Time: 2:34 PM", fill="black", font=font_small)

    # Line
    draw.line([(30, 140), (370, 140)], fill="gray", width=1)

    # Items
    items = [
        ("Coffee (Large)", "$4.50"),
        ("Bagel w/ Cream Cheese", "$3.75"),
        ("Orange Juice", "$2.99"),
        ("Granola Bar", "$1.50"),
    ]

    y = 160
    for item, price in items:
        draw.text((50, y), item, fill="black", font=font_small)
        draw.text((300, y), price, fill="black", font=font_small)
        y += 25

    # Line
    draw.line([(30, y + 10), (370, y + 10)], fill="gray", width=1)

    # Totals
    y += 30
    draw.text((200, y), "Subtotal:", fill="black", font=font_medium)
    draw.text((300, y), "$12.74", fill="black", font=font_medium)

    y += 25
    draw.text((200, y), "Tax (6.25%):", fill="black", font=font_medium)
    draw.text((300, y), "$0.80", fill="black", font=font_medium)

    y += 30
    draw.text((200, y), "TOTAL:", fill="black", font=font_large)
    draw.text((300, y), "$13.54", fill="black", font=font_large)

    y += 40
    draw.text((200, y), "VISA ****1234", fill="gray", font=font_small)

    # Footer
    draw.text((100, 500), "Thank you!", fill="black", font=font_medium)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"✅ Created: {output_path}")


def create_sample_business_card(output_path: Path) -> None:
    """Create a sample business card image for testing."""
    img = Image.new("RGB", (500, 300), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    # Name
    draw.text((30, 40), "John Smith", fill="navy", font=font_large)
    draw.text((30, 70), "Senior Software Engineer", fill="gray", font=font_medium)

    # Company
    draw.text((30, 110), "TechCorp Inc.", fill="black", font=font_medium)

    # Contact info
    draw.text((30, 160), "📧 john.smith@techcorp.com", fill="black", font=font_small)
    draw.text((30, 185), "📱 (555) 123-4567", fill="black", font=font_small)
    draw.text((30, 210), "🌐 www.techcorp.com", fill="black", font=font_small)

    # Address
    draw.text((280, 200), "100 Tech Park Drive", fill="gray", font=font_small)
    draw.text((280, 220), "Silicon Valley, CA 94000", fill="gray", font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"✅ Created: {output_path}")


def main() -> None:
    """Generate all sample test images."""
    assets_dir = Path(__file__).parent / "assets"

    print("🎨 Generating sample test images...")
    print()

    create_sample_invoice(assets_dir / "sample_invoice.png")
    create_sample_receipt(assets_dir / "sample_receipt.png")
    create_sample_business_card(assets_dir / "sample_business_card.png")

    print()
    print("✅ All sample images created!")
    print(f"   Location: {assets_dir}")


if __name__ == "__main__":
    main()
