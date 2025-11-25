"""Generate test images for E2E image extraction testing.

This script generates a variety of test images with ground truth data:
- Invoices (simple, detailed, international)
- Receipts (grocery, restaurant)
- Business cards (modern, minimal)
- Forms (application form)
- Shipping labels
- ID cards
- Data tables

Usage:
    uv run python examples/generate_test_images.py
"""

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Font Helpers
# ============================================================================


def get_fonts() -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """Get fonts with fallback to default."""
    fonts: dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    try:
        fonts["title"] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        fonts["large"] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        fonts["medium"] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        fonts["small"] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        fonts["tiny"] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except Exception:
        default = ImageFont.load_default()
        fonts = dict.fromkeys(["title", "large", "medium", "small", "tiny"], default)
    return fonts


# ============================================================================
# Invoice Generators
# ============================================================================


def create_invoice_simple(output_path: Path) -> dict[str, Any]:
    """Create a simple invoice image."""
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Header
    draw.text((50, 30), "INVOICE", fill="navy", font=fonts["title"])
    draw.text((600, 30), "#INV-2024-0001", fill="black", font=fonts["large"])
    draw.text((600, 60), "Date: 2024-11-25", fill="black", font=fonts["medium"])

    # Vendor
    draw.text((50, 100), "From: ABC Company LLC", fill="black", font=fonts["medium"])
    draw.text((50, 120), "123 Business Ave, New York, NY 10001", fill="gray", font=fonts["small"])

    # Customer
    draw.text((50, 160), "Bill To: Customer Corp", fill="black", font=fonts["medium"])

    # Items
    draw.line([(50, 200), (750, 200)], fill="gray", width=1)
    draw.text((50, 210), "Description", fill="gray", font=fonts["small"])
    draw.text((500, 210), "Amount", fill="gray", font=fonts["small"])

    items = [
        ("Professional Services", "$2,500.00"),
        ("Software License", "$1,200.00"),
    ]

    y = 240
    for desc, amt in items:
        draw.text((50, y), desc, fill="black", font=fonts["small"])
        draw.text((500, y), amt, fill="black", font=fonts["small"])
        y += 25

    # Total
    draw.line([(400, y + 10), (750, y + 10)], fill="gray", width=1)
    y += 30
    draw.text((400, y), "Total:", fill="black", font=fonts["large"])
    draw.text((500, y), "$3,700.00", fill="navy", font=fonts["large"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "invoice_number": "INV-2024-0001",
        "date": "2024-11-25",
        "vendor_name": "ABC Company LLC",
        "customer_name": "Customer Corp",
        "total_amount": 3700.00,
        "currency": "USD",
    }


def create_invoice_detailed(output_path: Path) -> dict[str, Any]:
    """Create a detailed invoice with line items."""
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Company logo area (just text for now)
    draw.rectangle([(50, 30), (200, 80)], outline="navy", width=2)
    draw.text((70, 45), "TECH SOLUTIONS", fill="navy", font=fonts["medium"])

    # Invoice title
    draw.text((550, 30), "INVOICE", fill="black", font=fonts["title"])
    draw.text((550, 70), "Invoice #: TS-2024-0789", fill="black", font=fonts["medium"])
    draw.text((550, 95), "Date: November 25, 2024", fill="black", font=fonts["small"])
    draw.text((550, 115), "Due: December 25, 2024", fill="black", font=fonts["small"])

    # Vendor details
    draw.text((50, 120), "Tech Solutions Inc.", fill="black", font=fonts["medium"])
    draw.text((50, 140), "456 Innovation Drive", fill="gray", font=fonts["small"])
    draw.text((50, 155), "San Francisco, CA 94102", fill="gray", font=fonts["small"])
    draw.text((50, 170), "Tax ID: 12-3456789", fill="gray", font=fonts["small"])

    # Bill To
    draw.text((50, 210), "BILL TO:", fill="gray", font=fonts["small"])
    draw.text((50, 230), "Enterprise Corp", fill="black", font=fonts["medium"])
    draw.text((50, 250), "789 Corporate Blvd", fill="black", font=fonts["small"])
    draw.text((50, 265), "Austin, TX 78701", fill="black", font=fonts["small"])

    # Table header
    draw.rectangle([(50, 310), (750, 340)], fill="lightgray")
    draw.text((60, 315), "Item", fill="black", font=fonts["small"])
    draw.text((300, 315), "Description", fill="black", font=fonts["small"])
    draw.text((500, 315), "Qty", fill="black", font=fonts["small"])
    draw.text((560, 315), "Rate", fill="black", font=fonts["small"])
    draw.text((680, 315), "Amount", fill="black", font=fonts["small"])

    # Line items
    items = [
        ("001", "Web Development", "40", "$125.00", "$5,000.00"),
        ("002", "UI/UX Design", "20", "$100.00", "$2,000.00"),
        ("003", "Cloud Hosting (Annual)", "1", "$1,200.00", "$1,200.00"),
        ("004", "Technical Support", "10", "$75.00", "$750.00"),
        ("005", "Training Sessions", "5", "$150.00", "$750.00"),
    ]

    y = 350
    for item_num, desc, qty, rate, amount in items:
        draw.text((60, y), item_num, fill="black", font=fonts["small"])
        draw.text((300, y), desc, fill="black", font=fonts["small"])
        draw.text((510, y), qty, fill="black", font=fonts["small"])
        draw.text((560, y), rate, fill="black", font=fonts["small"])
        draw.text((680, y), amount, fill="black", font=fonts["small"])
        y += 30
        draw.line([(50, y - 5), (750, y - 5)], fill="lightgray", width=1)

    # Totals
    y += 20
    draw.text((550, y), "Subtotal:", fill="black", font=fonts["medium"])
    draw.text((680, y), "$9,700.00", fill="black", font=fonts["medium"])

    y += 25
    draw.text((550, y), "Tax (8.25%):", fill="black", font=fonts["medium"])
    draw.text((680, y), "$800.25", fill="black", font=fonts["medium"])

    y += 30
    draw.rectangle([(540, y - 5), (750, y + 25)], fill="navy")
    draw.text((550, y), "TOTAL DUE:", fill="white", font=fonts["medium"])
    draw.text((680, y), "$10,500.25", fill="white", font=fonts["medium"])

    # Footer
    draw.text((50, 850), "Payment Terms: Net 30", fill="gray", font=fonts["small"])
    draw.text(
        (50, 870),
        "Bank: First National Bank | Account: 1234567890",
        fill="gray",
        font=fonts["tiny"],
    )
    draw.text((50, 890), "Thank you for your business!", fill="black", font=fonts["small"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "invoice_number": "TS-2024-0789",
        "date": "2024-11-25",
        "vendor_name": "Tech Solutions Inc.",
        "customer_name": "Enterprise Corp",
        "subtotal": 9700.00,
        "tax_amount": 800.25,
        "total_amount": 10500.25,
        "line_items_count": 5,
        "currency": "USD",
    }


def create_invoice_international(output_path: Path) -> dict[str, Any]:
    """Create an international invoice with EUR currency."""
    img = Image.new("RGB", (800, 700), color="#f8f8f8")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Header with EU styling
    draw.rectangle([(0, 0), (800, 80)], fill="#003399")
    draw.text((50, 25), "FACTURE / INVOICE", fill="white", font=fonts["title"])
    draw.text((600, 40), "N° FA-2024-0456", fill="white", font=fonts["medium"])

    # Company
    draw.text((50, 100), "EuroTech GmbH", fill="black", font=fonts["large"])
    draw.text((50, 130), "Hauptstraße 42", fill="gray", font=fonts["small"])
    draw.text((50, 145), "10115 Berlin, Germany", fill="gray", font=fonts["small"])
    draw.text((50, 160), "VAT: DE123456789", fill="gray", font=fonts["small"])

    # Invoice info
    draw.text((500, 100), "Invoice Date: 25.11.2024", fill="black", font=fonts["small"])
    draw.text((500, 120), "Due Date: 25.12.2024", fill="black", font=fonts["small"])

    # Client
    draw.text((50, 200), "Client:", fill="gray", font=fonts["small"])
    draw.text((50, 220), "Société Française SARL", fill="black", font=fonts["medium"])
    draw.text((50, 245), "75001 Paris, France", fill="black", font=fonts["small"])

    # Items table
    draw.line([(50, 290), (750, 290)], fill="gray", width=1)
    draw.text((50, 300), "Description", fill="gray", font=fonts["small"])
    draw.text((550, 300), "Montant / Amount", fill="gray", font=fonts["small"])

    items = [
        ("Consulting Services / Beratungsleistungen", "€ 4.500,00"),
        ("Software Development / Softwareentwicklung", "€ 8.750,00"),
        ("Annual Maintenance / Jahreswartung", "€ 1.200,00"),
    ]

    y = 330
    for desc, amt in items:
        draw.text((50, y), desc, fill="black", font=fonts["small"])
        draw.text((550, y), amt, fill="black", font=fonts["small"])
        y += 35

    # Totals
    draw.line([(400, y + 10), (750, y + 10)], fill="gray", width=1)
    y += 30
    draw.text((450, y), "Subtotal / Zwischensumme:", fill="black", font=fonts["small"])
    draw.text((650, y), "€ 14.450,00", fill="black", font=fonts["small"])

    y += 25
    draw.text((450, y), "VAT / MwSt (19%):", fill="black", font=fonts["small"])
    draw.text((650, y), "€ 2.745,50", fill="black", font=fonts["small"])

    y += 30
    draw.text((450, y), "TOTAL / GESAMT:", fill="navy", font=fonts["large"])
    draw.text((650, y), "€ 17.195,50", fill="navy", font=fonts["large"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "invoice_number": "FA-2024-0456",
        "date": "2024-11-25",
        "vendor_name": "EuroTech GmbH",
        "customer_name": "Société Française SARL",
        "subtotal": 14450.00,
        "tax_amount": 2745.50,
        "total_amount": 17195.50,
        "currency": "EUR",
    }


# ============================================================================
# Receipt Generators
# ============================================================================


def create_receipt_grocery(output_path: Path) -> dict[str, Any]:
    """Create a grocery store receipt."""
    img = Image.new("RGB", (400, 700), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Header
    draw.text((120, 20), "FRESH MART", fill="green", font=fonts["large"])
    draw.text((100, 50), "Your Local Grocery Store", fill="gray", font=fonts["small"])
    draw.text((80, 70), "1234 Market Street, Chicago, IL", fill="gray", font=fonts["tiny"])
    draw.text((140, 85), "Tel: 312-555-0199", fill="gray", font=fonts["tiny"])

    draw.line([(30, 110), (370, 110)], fill="gray", width=1)

    # Transaction info
    draw.text((30, 120), "Date: 11/25/2024  Time: 14:23", fill="black", font=fonts["small"])
    draw.text((30, 140), "Cashier: MIKE  Register: 03", fill="black", font=fonts["small"])
    draw.text((30, 160), "Trans#: 789456123", fill="black", font=fonts["small"])

    draw.line([(30, 185), (370, 185)], fill="gray", width=1)

    # Items
    items = [
        ("Organic Milk 1gal", "5.99"),
        ("Whole Wheat Bread", "3.49"),
        ("Bananas 2.5lb", "1.87"),
        ("Chicken Breast 1.2lb", "8.99"),
        ("Cheddar Cheese 8oz", "4.29"),
        ("Eggs Large Dozen", "3.99"),
        ("Orange Juice 64oz", "4.49"),
        ("Spinach Organic", "3.99"),
    ]

    y = 200
    subtotal = 0.0
    for item, price in items:
        draw.text((30, y), item, fill="black", font=fonts["small"])
        draw.text((310, y), f"${price}", fill="black", font=fonts["small"])
        subtotal += float(price)
        y += 22

    draw.line([(30, y + 5), (370, y + 5)], fill="gray", width=1)

    # Totals
    y += 20
    tax = subtotal * 0.0625
    total = subtotal + tax

    draw.text((200, y), "SUBTOTAL:", fill="black", font=fonts["small"])
    draw.text((310, y), f"${subtotal:.2f}", fill="black", font=fonts["small"])

    y += 20
    draw.text((200, y), "TAX 6.25%:", fill="black", font=fonts["small"])
    draw.text((310, y), f"${tax:.2f}", fill="black", font=fonts["small"])

    y += 25
    draw.text((200, y), "TOTAL:", fill="black", font=fonts["large"])
    draw.text((310, y), f"${total:.2f}", fill="black", font=fonts["large"])

    y += 35
    draw.text((200, y), "VISA ***1234", fill="gray", font=fonts["small"])

    # Footer
    draw.text((100, 620), "Thank you for shopping!", fill="black", font=fonts["small"])
    draw.text((80, 640), "Visit us at freshmart.com", fill="gray", font=fonts["tiny"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "store_name": "FRESH MART",
        "date": "2024-11-25",
        "time": "14:23",
        "items_count": len(items),
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
        "payment_method": "VISA",
    }


def create_receipt_restaurant(output_path: Path) -> dict[str, Any]:
    """Create a restaurant receipt."""
    img = Image.new("RGB", (400, 650), color="#fffef5")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Restaurant header
    draw.text((100, 25), "THE GOLDEN FORK", fill="darkred", font=fonts["large"])
    draw.text((110, 55), "Fine Italian Dining", fill="gray", font=fonts["small"])
    draw.text((90, 75), "555 Restaurant Row, NYC", fill="gray", font=fonts["tiny"])

    draw.line([(30, 100), (370, 100)], fill="lightgray", width=1)

    # Check info
    draw.text((30, 110), "Server: MARIA", fill="black", font=fonts["small"])
    draw.text((250, 110), "Table: 12", fill="black", font=fonts["small"])
    draw.text((30, 130), "Date: 11/25/2024", fill="black", font=fonts["small"])
    draw.text((250, 130), "Check: 4521", fill="black", font=fonts["small"])
    draw.text((30, 150), "Guests: 4", fill="black", font=fonts["small"])

    draw.line([(30, 175), (370, 175)], fill="lightgray", width=1)

    # Items
    items = [
        ("Bruschetta", "12.00"),
        ("Caesar Salad", "14.00"),
        ("Spaghetti Carbonara", "22.00"),
        ("Margherita Pizza", "18.00"),
        ("Osso Buco", "34.00"),
        ("Tiramisu x2", "18.00"),
        ("House Red Wine (btl)", "45.00"),
        ("Espresso x4", "16.00"),
    ]

    y = 190
    subtotal = 0.0
    for item, price in items:
        draw.text((30, y), item, fill="black", font=fonts["small"])
        draw.text((310, y), f"${price}", fill="black", font=fonts["small"])
        subtotal += float(price)
        y += 22

    draw.line([(30, y + 5), (370, y + 5)], fill="lightgray", width=1)

    # Totals
    y += 20
    tax = subtotal * 0.08875
    total_before_tip = subtotal + tax

    draw.text((180, y), "Subtotal:", fill="black", font=fonts["small"])
    draw.text((310, y), f"${subtotal:.2f}", fill="black", font=fonts["small"])

    y += 20
    draw.text((180, y), "Tax (8.875%):", fill="black", font=fonts["small"])
    draw.text((310, y), f"${tax:.2f}", fill="black", font=fonts["small"])

    y += 25
    draw.text((180, y), "Total:", fill="black", font=fonts["medium"])
    draw.text((310, y), f"${total_before_tip:.2f}", fill="black", font=fonts["medium"])

    # Tip suggestions
    y += 40
    draw.text((30, y), "Suggested Tips:", fill="gray", font=fonts["small"])
    y += 18
    for pct, label in [(18, "18%"), (20, "20%"), (22, "22%")]:
        tip = subtotal * (pct / 100)
        draw.text((30, y), f"{label}: ${tip:.2f}", fill="gray", font=fonts["tiny"])
        y += 15

    # Footer
    draw.text((90, 580), "Grazie! Thank you!", fill="darkred", font=fonts["medium"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "store_name": "THE GOLDEN FORK",
        "date": "2024-11-25",
        "server": "MARIA",
        "table": "12",
        "guests": 4,
        "items_count": len(items),
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total_before_tip, 2),
    }


# ============================================================================
# Business Card Generators
# ============================================================================


def create_business_card_modern(output_path: Path) -> dict[str, Any]:
    """Create a modern business card."""
    img = Image.new("RGB", (550, 320), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Left accent bar
    draw.rectangle([(0, 0), (15, 320)], fill="#2563eb")

    # Name and title
    draw.text((40, 50), "Sarah Johnson", fill="#1e293b", font=fonts["title"])
    draw.text((40, 90), "Chief Technology Officer", fill="#64748b", font=fonts["medium"])

    # Company
    draw.text((40, 140), "INNOVATE TECH", fill="#2563eb", font=fonts["large"])

    # Contact details
    draw.text((40, 190), "sarah.johnson@innovatetech.com", fill="#334155", font=fonts["small"])
    draw.text((40, 215), "+1 (415) 555-0192", fill="#334155", font=fonts["small"])
    draw.text((40, 240), "www.innovatetech.com", fill="#334155", font=fonts["small"])

    # Address
    draw.text((300, 240), "100 Tech Campus Dr", fill="#94a3b8", font=fonts["tiny"])
    draw.text((300, 255), "San Francisco, CA 94105", fill="#94a3b8", font=fonts["tiny"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "name": "Sarah Johnson",
        "title": "Chief Technology Officer",
        "company": "INNOVATE TECH",
        "email": "sarah.johnson@innovatetech.com",
        "phone": "+1 (415) 555-0192",
        "website": "www.innovatetech.com",
    }


def create_business_card_minimal(output_path: Path) -> dict[str, Any]:
    """Create a minimal business card."""
    img = Image.new("RGB", (550, 320), color="#1a1a1a")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Name
    draw.text((50, 80), "DAVID CHEN", fill="white", font=fonts["title"])

    # Role
    draw.text((50, 130), "Software Architect", fill="#888888", font=fonts["medium"])

    # Contact
    draw.text((50, 200), "d.chen@nexusdev.io", fill="#cccccc", font=fonts["small"])
    draw.text((50, 225), "415.555.0234", fill="#cccccc", font=fonts["small"])
    draw.text((50, 250), "linkedin.com/in/davidchen", fill="#0a66c2", font=fonts["small"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "name": "DAVID CHEN",
        "title": "Software Architect",
        "email": "d.chen@nexusdev.io",
        "phone": "415.555.0234",
    }


# ============================================================================
# Form Generators
# ============================================================================


def create_form_application(output_path: Path) -> dict[str, Any]:
    """Create an application form image."""
    img = Image.new("RGB", (800, 1000), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Header
    draw.rectangle([(0, 0), (800, 70)], fill="#1e40af")
    draw.text((50, 20), "EMPLOYMENT APPLICATION FORM", fill="white", font=fonts["large"])

    # Personal Information Section
    y = 100
    draw.text((50, y), "PERSONAL INFORMATION", fill="#1e40af", font=fonts["medium"])
    y += 30
    draw.line([(50, y), (750, y)], fill="#e5e7eb", width=1)

    # Form fields (simulating filled form)
    fields = [
        ("Full Name:", "Michael Robert Thompson"),
        ("Date of Birth:", "March 15, 1990"),
        ("Email:", "m.thompson@email.com"),
        ("Phone:", "(555) 867-5309"),
        ("Address:", "742 Evergreen Terrace, Springfield, IL 62701"),
    ]

    y += 20
    for label, value in fields:
        draw.text((50, y), label, fill="gray", font=fonts["small"])
        draw.text((180, y), value, fill="black", font=fonts["small"])
        draw.line([(180, y + 18), (750, y + 18)], fill="#e5e7eb", width=1)
        y += 35

    # Employment Section
    y += 20
    draw.text((50, y), "EMPLOYMENT HISTORY", fill="#1e40af", font=fonts["medium"])
    y += 30
    draw.line([(50, y), (750, y)], fill="#e5e7eb", width=1)

    employment = [
        ("Current Employer:", "Tech Solutions Inc."),
        ("Position:", "Senior Software Developer"),
        ("Start Date:", "January 2020"),
        ("Salary:", "$95,000/year"),
    ]

    y += 20
    for label, value in employment:
        draw.text((50, y), label, fill="gray", font=fonts["small"])
        draw.text((180, y), value, fill="black", font=fonts["small"])
        y += 30

    # Education Section
    y += 20
    draw.text((50, y), "EDUCATION", fill="#1e40af", font=fonts["medium"])
    y += 30
    draw.line([(50, y), (750, y)], fill="#e5e7eb", width=1)

    education = [
        ("Degree:", "Bachelor of Science in Computer Science"),
        ("Institution:", "State University"),
        ("Graduation Year:", "2012"),
        ("GPA:", "3.7"),
    ]

    y += 20
    for label, value in education:
        draw.text((50, y), label, fill="gray", font=fonts["small"])
        draw.text((180, y), value, fill="black", font=fonts["small"])
        y += 30

    # Signature
    y += 40
    draw.text((50, y), "Applicant Signature:", fill="gray", font=fonts["small"])
    draw.text((200, y), "Michael Thompson", fill="navy", font=fonts["medium"])
    draw.line([(200, y + 20), (400, y + 20)], fill="black", width=1)

    draw.text((500, y), "Date:", fill="gray", font=fonts["small"])
    draw.text((550, y), "11/25/2024", fill="black", font=fonts["small"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "applicant_name": "Michael Robert Thompson",
        "date_of_birth": "1990-03-15",
        "email": "m.thompson@email.com",
        "phone": "(555) 867-5309",
        "current_employer": "Tech Solutions Inc.",
        "position": "Senior Software Developer",
        "degree": "Bachelor of Science in Computer Science",
        "institution": "State University",
    }


# ============================================================================
# Shipping Label Generator
# ============================================================================


def create_shipping_label(output_path: Path) -> dict[str, Any]:
    """Create a shipping label image."""
    img = Image.new("RGB", (600, 400), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Border
    draw.rectangle([(5, 5), (595, 395)], outline="black", width=2)

    # Header
    draw.rectangle([(10, 10), (590, 50)], fill="#ff6600")
    draw.text((20, 15), "PRIORITY MAIL", fill="white", font=fonts["large"])
    draw.text((400, 20), "2-DAY", fill="white", font=fonts["medium"])

    # From section
    draw.text((20, 60), "FROM:", fill="gray", font=fonts["small"])
    draw.text((20, 80), "Amazon Fulfillment Center", fill="black", font=fonts["medium"])
    draw.text((20, 105), "1000 Warehouse Way", fill="black", font=fonts["small"])
    draw.text((20, 125), "Seattle, WA 98101", fill="black", font=fonts["small"])

    # Divider
    draw.line([(10, 160), (590, 160)], fill="black", width=2)

    # To section (larger)
    draw.text((20, 170), "SHIP TO:", fill="gray", font=fonts["small"])
    draw.text((20, 195), "Jennifer Martinez", fill="black", font=fonts["title"])
    draw.text((20, 235), "456 Oak Avenue, Apt 12B", fill="black", font=fonts["medium"])
    draw.text((20, 265), "Los Angeles, CA 90001", fill="black", font=fonts["large"])

    # Barcode area (simulated)
    draw.rectangle([(350, 280), (580, 330)], fill="white", outline="black")
    for i in range(20):
        x = 360 + i * 10
        height = random.randint(30, 45)
        draw.line([(x, 330 - height), (x, 325)], fill="black", width=random.randint(1, 3))

    # Tracking number
    draw.text((350, 340), "1Z999AA10123456784", fill="black", font=fonts["small"])

    # Weight
    draw.text((20, 340), "Weight: 2.5 lbs", fill="black", font=fonts["small"])
    draw.text((20, 360), "Ship Date: 11/25/2024", fill="gray", font=fonts["tiny"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "sender_name": "Amazon Fulfillment Center",
        "sender_address": "1000 Warehouse Way, Seattle, WA 98101",
        "recipient_name": "Jennifer Martinez",
        "recipient_address": "456 Oak Avenue, Apt 12B, Los Angeles, CA 90001",
        "tracking_number": "1Z999AA10123456784",
        "weight": "2.5 lbs",
        "ship_date": "2024-11-25",
        "service": "PRIORITY MAIL 2-DAY",
    }


# ============================================================================
# ID Card Generator
# ============================================================================


def create_id_card(output_path: Path) -> dict[str, Any]:
    """Create an ID card image."""
    img = Image.new("RGB", (500, 320), color="#f0f4f8")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Header bar
    draw.rectangle([(0, 0), (500, 50)], fill="#1e3a8a")
    draw.text((20, 12), "EMPLOYEE IDENTIFICATION", fill="white", font=fonts["medium"])

    # Photo placeholder
    draw.rectangle([(20, 70), (150, 220)], fill="#d1d5db", outline="#9ca3af")
    draw.text((50, 135), "PHOTO", fill="#6b7280", font=fonts["small"])

    # Employee info
    draw.text((170, 75), "ANDERSON, EMILY J.", fill="#1e3a8a", font=fonts["large"])
    draw.text((170, 110), "Software Engineer III", fill="#374151", font=fonts["medium"])

    draw.text((170, 150), "Employee ID:", fill="gray", font=fonts["small"])
    draw.text((270, 150), "EMP-2024-0892", fill="black", font=fonts["small"])

    draw.text((170, 175), "Department:", fill="gray", font=fonts["small"])
    draw.text((270, 175), "Engineering", fill="black", font=fonts["small"])

    draw.text((170, 200), "Start Date:", fill="gray", font=fonts["small"])
    draw.text((270, 200), "06/15/2022", fill="black", font=fonts["small"])

    # Bottom bar
    draw.rectangle([(0, 250), (500, 320)], fill="#1e3a8a")
    draw.text((20, 260), "ACME CORPORATION", fill="white", font=fonts["medium"])
    draw.text((20, 285), "Valid through: 12/31/2025", fill="#93c5fd", font=fonts["small"])

    # Barcode area
    for i in range(25):
        x = 350 + i * 5
        height = random.randint(25, 40)
        draw.line([(x, 315 - height), (x, 310)], fill="white", width=random.randint(1, 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "name": "ANDERSON, EMILY J.",
        "title": "Software Engineer III",
        "employee_id": "EMP-2024-0892",
        "department": "Engineering",
        "start_date": "2022-06-15",
        "company": "ACME CORPORATION",
        "valid_through": "2025-12-31",
    }


# ============================================================================
# Table/Data Image Generator
# ============================================================================


def create_data_table(output_path: Path) -> dict[str, Any]:
    """Create a data table image."""
    img = Image.new("RGB", (700, 500), color="white")
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # Title
    draw.text((50, 20), "Q3 2024 Sales Report", fill="#1e293b", font=fonts["large"])
    draw.text((50, 50), "Regional Performance Summary", fill="#64748b", font=fonts["small"])

    # Table header
    headers = ["Region", "Q1 Sales", "Q2 Sales", "Q3 Sales", "YTD Total", "Growth"]
    col_widths = [100, 90, 90, 90, 100, 80]
    x_positions = [50]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    y = 100
    draw.rectangle([(45, y - 5), (655, y + 25)], fill="#3b82f6")
    for i, header in enumerate(headers):
        draw.text((x_positions[i], y), header, fill="white", font=fonts["small"])

    # Data rows
    data = [
        ("Northeast", "$245,000", "$267,000", "$298,000", "$810,000", "+12.5%"),
        ("Southeast", "$189,000", "$201,000", "$215,000", "$605,000", "+8.3%"),
        ("Midwest", "$312,000", "$298,000", "$325,000", "$935,000", "+5.1%"),
        ("Southwest", "$178,000", "$195,000", "$210,000", "$583,000", "+15.2%"),
        ("West Coast", "$425,000", "$456,000", "$489,000", "$1,370,000", "+9.8%"),
    ]

    y += 35
    for row_idx, row in enumerate(data):
        bg_color = "#f8fafc" if row_idx % 2 == 0 else "white"
        draw.rectangle([(45, y - 5), (655, y + 20)], fill=bg_color)
        for i, cell in enumerate(row):
            color = "#16a34a" if cell.startswith("+") else "#1e293b"
            draw.text((x_positions[i], y), cell, fill=color, font=fonts["small"])
        y += 30

    # Footer totals
    y += 10
    draw.rectangle([(45, y - 5), (655, y + 25)], fill="#1e293b")
    totals = ("TOTAL", "$1,349,000", "$1,417,000", "$1,537,000", "$4,303,000", "+10.2%")
    for i, cell in enumerate(totals):
        draw.text((x_positions[i], y), cell, fill="white", font=fonts["small"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "report_title": "Q3 2024 Sales Report",
        "regions_count": 5,
        "total_q3_sales": 1537000.00,
        "total_ytd": 4303000.00,
        "overall_growth": "+10.2%",
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Generate all test images and save ground truth."""
    assets_dir = Path(__file__).parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("🎨 Generating test images...")
    print("=" * 50)

    ground_truth: dict[str, dict[str, Any]] = {}

    # Generate all images
    generators = [
        ("invoice_simple.png", create_invoice_simple),
        ("invoice_detailed.png", create_invoice_detailed),
        ("invoice_international.png", create_invoice_international),
        ("receipt_grocery.png", create_receipt_grocery),
        ("receipt_restaurant.png", create_receipt_restaurant),
        ("business_card_modern.png", create_business_card_modern),
        ("business_card_minimal.png", create_business_card_minimal),
        ("form_application.png", create_form_application),
        ("shipping_label.png", create_shipping_label),
        ("id_card.png", create_id_card),
        ("data_table.png", create_data_table),
    ]

    for filename, generator in generators:
        output_path = assets_dir / filename
        try:
            gt = generator(output_path)
            ground_truth[filename] = gt
            print(f"✅ Created: {filename}")
        except Exception as e:
            print(f"❌ Failed: {filename} - {e}")

    # Save ground truth
    gt_path = assets_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\n📄 Ground truth saved to: {gt_path}")

    print("\n" + "=" * 50)
    print(f"✅ Generated {len(generators)} test images")
    print(f"   Location: {assets_dir}")


if __name__ == "__main__":
    main()
