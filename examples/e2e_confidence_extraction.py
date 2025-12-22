"""End-to-end test: Confidence scoring for extractions.

This script demonstrates and tests:
1. Basic confidence extraction
2. High vs low confidence scenarios
3. Per-field confidence scores
4. Confidence thresholds
5. Identifying unreliable extractions

Prerequisites:
- OPENAI_API_KEY environment variable set

Usage:
    python examples/e2e_confidence_extraction.py
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import DocumentExtractor, ExtractionConfig

# Load environment variables
load_dotenv()


# ============================================================================
# Performance Tracking
# ============================================================================


@dataclass
class ConfidenceTestResult:
    """Result of a confidence test."""

    test_name: str
    success: bool
    overall_confidence: float | None
    field_confidences: dict[str, float] | None
    low_confidence_fields: list[str] | None
    latency_ms: float
    tokens_used: int | None
    cost_usd: float | None
    error: str | None = None
    extracted_data: dict[str, Any] | None = None


@dataclass
class ConfidenceTestSuite:
    """Results from confidence test suite."""

    suite_name: str
    results: list[ConfidenceTestResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def avg_confidence(self) -> float:
        confs = [r.overall_confidence for r in self.results if r.overall_confidence is not None]
        return sum(confs) / len(confs) if confs else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.results)

    def print_summary(self) -> None:
        """Print summary of confidence tests."""
        print(f"\n{'=' * 60}")
        print(f"📊 {self.suite_name} - Results Summary")
        print("=" * 60)
        print(f"Total Tests:       {self.total_tests}")
        print(f"Passed:            {self.passed_tests}")
        print(f"Avg Confidence:    {self.avg_confidence:.2f}")
        print(f"Total Cost:        ${self.total_cost_usd:.4f}")
        print("=" * 60)


# ============================================================================
# Schema Definitions
# ============================================================================


class PersonInfo(BaseModel):
    """Basic person information for confidence testing."""

    name: str = Field(description="Person's full name")
    age: int | None = Field(default=None, description="Person's age in years")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    occupation: str | None = Field(default=None, description="Current job/occupation")


class ProductInfo(BaseModel):
    """Product information."""

    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    sku: str | None = Field(default=None, description="Stock keeping unit")
    category: str | None = Field(default=None, description="Product category")
    in_stock: bool | None = Field(default=None, description="Whether in stock")


class OrderDetails(BaseModel):
    """Order details with multiple fields."""

    order_id: str = Field(description="Unique order identifier")
    customer_name: str = Field(description="Customer name")
    order_date: str = Field(description="Date of order")
    total_amount: float = Field(description="Total order amount")
    shipping_address: str | None = Field(default=None, description="Shipping address")
    payment_method: str | None = Field(default=None, description="Payment method used")
    status: str | None = Field(default=None, description="Order status")


class MedicalRecord(BaseModel):
    """Medical record with sensitive/precise fields."""

    patient_name: str = Field(description="Patient's full name")
    date_of_birth: str = Field(description="Date of birth")
    blood_type: str | None = Field(default=None, description="Blood type (A, B, AB, O with +/-)")
    allergies: list[str] = Field(default_factory=list, description="Known allergies")
    medications: list[str] = Field(default_factory=list, description="Current medications")
    diagnosis: str | None = Field(default=None, description="Primary diagnosis")


# ============================================================================
# Test Documents
# ============================================================================

# High confidence document - clear and explicit information
HIGH_CONFIDENCE_DOC = """
Customer Profile
================

Name: John Michael Smith
Age: 35 years old
Email: john.smith@example.com
Phone: (555) 123-4567
Occupation: Senior Software Engineer at Tech Corp Inc.

Last Updated: November 25, 2024
"""

# Medium confidence document - some information implied or partial
MEDIUM_CONFIDENCE_DOC = """
From: J. Smith <jsmith@email.com>
Subject: Meeting tomorrow

Hi,

I'm John, I work in software development. I'll be around 35-40 minutes late
to the meeting tomorrow. You can reach me on my cell if needed.

Best,
John
"""

# Low confidence document - vague and incomplete
LOW_CONFIDENCE_DOC = """
Note:
Someone named John called about the project.
Said he's in tech or something.
Left a number but I couldn't read it clearly.
"""

# Complete order document
COMPLETE_ORDER_DOC = """
ORDER CONFIRMATION
==================

Order ID: ORD-2024-78523
Date: November 25, 2024

Customer: Sarah Johnson
Email: sarah.j@email.com

Shipping Address:
123 Main Street, Apt 4B
New York, NY 10001

Items Ordered:
- Widget Pro (x2) - $49.99 each
- Gadget Plus (x1) - $79.99

Subtotal: $179.97
Shipping: $12.99
Tax: $15.74
-----------------
Total: $208.70

Payment Method: Visa ending in 4242
Status: Processing

Thank you for your order!
"""

# Partial order document
PARTIAL_ORDER_DOC = """
Quick order note:
Order #12345 from John
About $150 worth of stuff
Need to confirm shipping details
"""

# Medical record - complete
MEDICAL_RECORD_COMPLETE = """
PATIENT MEDICAL RECORD
======================

Patient Name: Emily Rose Thompson
Date of Birth: March 15, 1985

Blood Type: O+

Known Allergies:
- Penicillin (severe)
- Peanuts (moderate)
- Latex (mild)

Current Medications:
- Lisinopril 10mg daily
- Vitamin D 1000IU daily

Primary Diagnosis: Hypertension (well-controlled)

Last Visit: November 20, 2024
Physician: Dr. Robert Chen, MD
"""

# Medical record - incomplete/unclear
MEDICAL_RECORD_UNCLEAR = """
Patient notes from phone call:

Caller said her name was Emily or Emma (not sure).
Born sometime in the 1980s.
Has some allergies - mentioned nuts I think?
Taking blood pressure medication.
"""


# ============================================================================
# Test Functions
# ============================================================================


def run_confidence_test(
    name: str,
    extractor: DocumentExtractor,
    document: str,
    schema: type[BaseModel],
    config: ExtractionConfig | None = None,
) -> ConfidenceTestResult:
    """Run a confidence extraction test."""
    start_time = time.time()

    try:
        result = extractor.extract_with_confidence(
            document,
            schema=schema,
            config=config,
        )
        latency_ms = (time.time() - start_time) * 1000

        return ConfidenceTestResult(
            test_name=name,
            success=True,
            overall_confidence=result.confidence,
            field_confidences=result.field_confidences,
            low_confidence_fields=result.low_confidence_fields,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            error=None,
            extracted_data=result.data.model_dump(),
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return ConfidenceTestResult(
            test_name=name,
            success=False,
            overall_confidence=None,
            field_confidences=None,
            low_confidence_fields=None,
            latency_ms=latency_ms,
            tokens_used=None,
            cost_usd=None,
            error=str(e),
        )


def test_high_confidence_extraction(
    extractor: DocumentExtractor, suite: ConfidenceTestSuite
) -> None:
    """Test confidence on clear, explicit document."""
    print("\n" + "-" * 50)
    print("Test: High Confidence Document")
    print("-" * 50)

    result = run_confidence_test(
        "high_confidence",
        extractor,
        HIGH_CONFIDENCE_DOC,
        PersonInfo,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")
        print(f"   Latency: {result.latency_ms:.0f}ms")

        if result.field_confidences:
            print("\n   Field Confidences:")
            for field_name, conf in result.field_confidences.items():
                indicator = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
                print(f"     {indicator} {field_name}: {conf:.2f}")

        if result.low_confidence_fields:
            print(f"\n   ⚠️ Low confidence fields: {result.low_confidence_fields}")
    else:
        print(f"❌ Failed: {result.error}")


def test_medium_confidence_extraction(
    extractor: DocumentExtractor, suite: ConfidenceTestSuite
) -> None:
    """Test confidence on partially clear document."""
    print("\n" + "-" * 50)
    print("Test: Medium Confidence Document")
    print("-" * 50)

    result = run_confidence_test(
        "medium_confidence",
        extractor,
        MEDIUM_CONFIDENCE_DOC,
        PersonInfo,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")

        if result.field_confidences:
            print("\n   Field Confidences:")
            for field_name, conf in result.field_confidences.items():
                indicator = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
                print(f"     {indicator} {field_name}: {conf:.2f}")

        if result.low_confidence_fields:
            print(f"\n   ⚠️ Low confidence fields: {result.low_confidence_fields}")
    else:
        print(f"❌ Failed: {result.error}")


def test_low_confidence_extraction(
    extractor: DocumentExtractor, suite: ConfidenceTestSuite
) -> None:
    """Test confidence on vague document."""
    print("\n" + "-" * 50)
    print("Test: Low Confidence Document")
    print("-" * 50)

    result = run_confidence_test(
        "low_confidence",
        extractor,
        LOW_CONFIDENCE_DOC,
        PersonInfo,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")

        if result.field_confidences:
            print("\n   Field Confidences:")
            for field_name, conf in result.field_confidences.items():
                indicator = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
                print(f"     {indicator} {field_name}: {conf:.2f}")

        if result.low_confidence_fields:
            print(f"\n   ⚠️ Low confidence fields: {result.low_confidence_fields}")

        # Verify low confidence was detected
        if result.overall_confidence and result.overall_confidence < 0.5:
            print("\n   ✓ Correctly identified as low confidence document")
    else:
        print(f"❌ Failed: {result.error}")


def test_complete_order_confidence(
    extractor: DocumentExtractor, suite: ConfidenceTestSuite
) -> None:
    """Test confidence on complete order document."""
    print("\n" + "-" * 50)
    print("Test: Complete Order Document")
    print("-" * 50)

    result = run_confidence_test(
        "complete_order",
        extractor,
        COMPLETE_ORDER_DOC,
        OrderDetails,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")
        order_id = result.extracted_data["order_id"] if result.extracted_data else "N/A"
        total = result.extracted_data["total_amount"] if result.extracted_data else "N/A"
        print(f"   Order ID: {order_id}")
        print(f"   Total: ${total}")

        if result.field_confidences:
            high_conf = sum(1 for c in result.field_confidences.values() if c >= 0.8)
            print(f"\n   High confidence fields: {high_conf}/{len(result.field_confidences)}")
    else:
        print(f"❌ Failed: {result.error}")


def test_partial_order_confidence(extractor: DocumentExtractor, suite: ConfidenceTestSuite) -> None:
    """Test confidence on partial order document."""
    print("\n" + "-" * 50)
    print("Test: Partial Order Document")
    print("-" * 50)

    result = run_confidence_test(
        "partial_order",
        extractor,
        PARTIAL_ORDER_DOC,
        OrderDetails,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")

        if result.low_confidence_fields:
            print(f"   ⚠️ Missing/uncertain fields: {result.low_confidence_fields}")

        if result.field_confidences:
            print("\n   Field Confidences:")
            for field_name, conf in sorted(
                result.field_confidences.items(), key=lambda x: x[1], reverse=True
            ):
                indicator = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
                print(f"     {indicator} {field_name}: {conf:.2f}")
    else:
        print(f"❌ Failed: {result.error}")


def test_medical_record_confidence(
    extractor: DocumentExtractor, suite: ConfidenceTestSuite
) -> None:
    """Test confidence on medical records (high precision required)."""
    print("\n" + "-" * 50)
    print("Test: Medical Record - Complete")
    print("-" * 50)

    result = run_confidence_test(
        "medical_complete",
        extractor,
        MEDICAL_RECORD_COMPLETE,
        MedicalRecord,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")

        if result.extracted_data:
            print(f"   Patient: {result.extracted_data['patient_name']}")
            print(f"   Blood Type: {result.extracted_data['blood_type']}")
            allergies_count = len(result.extracted_data.get("allergies", []))
            print(f"   Allergies: {allergies_count} recorded")

        if result.field_confidences:
            critical_fields = ["blood_type", "allergies", "medications"]
            print("\n   Critical Field Confidences:")
            for field_name in critical_fields:
                if field_name in result.field_confidences:
                    conf = result.field_confidences[field_name]
                    indicator = "🟢" if conf >= 0.9 else "🟡" if conf >= 0.7 else "🔴"
                    print(f"     {indicator} {field_name}: {conf:.2f}")
    else:
        print(f"❌ Failed: {result.error}")


def test_medical_record_unclear(extractor: DocumentExtractor, suite: ConfidenceTestSuite) -> None:
    """Test confidence on unclear medical record."""
    print("\n" + "-" * 50)
    print("Test: Medical Record - Unclear")
    print("-" * 50)

    result = run_confidence_test(
        "medical_unclear",
        extractor,
        MEDICAL_RECORD_UNCLEAR,
        MedicalRecord,
    )
    suite.results.append(result)

    if result.success:
        print(f"✅ Passed - Overall Confidence: {result.overall_confidence:.2f}")

        if result.overall_confidence and result.overall_confidence < 0.7:
            print("   ✓ Correctly flagged as uncertain medical data")

        if result.low_confidence_fields:
            print("\n   ⚠️ Uncertain fields requiring verification:")
            for field_name in result.low_confidence_fields:
                print(f"     - {field_name}")
    else:
        print(f"❌ Failed: {result.error}")


def test_confidence_threshold_filtering(
    extractor: DocumentExtractor, suite: ConfidenceTestSuite
) -> None:
    """Test confidence threshold for filtering unreliable extractions."""
    print("\n" + "-" * 50)
    print("Test: Confidence Threshold Filtering")
    print("-" * 50)

    # Test with different thresholds
    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:
        config = ExtractionConfig(confidence_threshold=threshold)

        result = run_confidence_test(
            f"threshold_{threshold}",
            extractor,
            MEDIUM_CONFIDENCE_DOC,
            PersonInfo,
            config=config,
        )
        suite.results.append(result)

        if result.success:
            low_conf_count = len(result.low_confidence_fields or [])
            print(f"   Threshold {threshold}: {low_conf_count} fields below threshold")
        else:
            print(f"   Threshold {threshold}: Failed - {result.error}")


def test_confidence_comparison(extractor: DocumentExtractor, suite: ConfidenceTestSuite) -> None:
    """Compare confidence scores across different document qualities."""
    print("\n" + "-" * 50)
    print("Test: Confidence Score Comparison")
    print("-" * 50)

    documents = [
        ("High Quality", HIGH_CONFIDENCE_DOC),
        ("Medium Quality", MEDIUM_CONFIDENCE_DOC),
        ("Low Quality", LOW_CONFIDENCE_DOC),
    ]

    print("\n   Document Quality vs Confidence Score:")
    print("   " + "-" * 40)

    for quality_name, doc in documents:
        result = run_confidence_test(
            f"comparison_{quality_name.lower().replace(' ', '_')}",
            extractor,
            doc,
            PersonInfo,
        )
        suite.results.append(result)

        if result.success and result.overall_confidence is not None:
            bar_length = int(result.overall_confidence * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {quality_name:15} [{bar}] {result.overall_confidence:.2f}")
        else:
            print(f"   {quality_name:15} [Error: {result.error}]")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all E2E confidence extraction tests."""
    print("\n🎯 Structured Extractor - E2E Confidence Scoring Tests")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize extractor
    extractor = DocumentExtractor(
        model="gpt-4.1",
        cache_dir="cache",
    )

    # Initialize results tracker
    suite = ConfidenceTestSuite(suite_name="Confidence Scoring Tests")

    # Run all tests
    print("\n📋 Running confidence extraction tests...")

    test_high_confidence_extraction(extractor, suite)
    test_medium_confidence_extraction(extractor, suite)
    test_low_confidence_extraction(extractor, suite)
    test_complete_order_confidence(extractor, suite)
    test_partial_order_confidence(extractor, suite)
    test_medical_record_confidence(extractor, suite)
    test_medical_record_unclear(extractor, suite)
    test_confidence_threshold_filtering(extractor, suite)
    test_confidence_comparison(extractor, suite)

    # Print summary
    suite.print_summary()

    # Print confidence insights
    print("\n📈 Confidence Insights:")
    print("-" * 40)

    high_conf = [r for r in suite.results if (r.overall_confidence or 0) >= 0.8]
    medium_conf = [r for r in suite.results if 0.5 <= (r.overall_confidence or 0) < 0.8]
    low_conf = [r for r in suite.results if (r.overall_confidence or 0) < 0.5]

    print(f"   High confidence extractions (≥0.8):   {len(high_conf)}")
    print(f"   Medium confidence (0.5-0.8):         {len(medium_conf)}")
    print(f"   Low confidence (<0.5):               {len(low_conf)}")

    # Print any failures
    failures = [r for r in suite.results if not r.success]
    if failures:
        print("\n❌ Failed Tests:")
        for f in failures:
            print(f"   - {f.test_name}: {f.error}")


if __name__ == "__main__":
    main()
