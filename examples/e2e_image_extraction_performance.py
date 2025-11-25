"""E2E Image Extraction Performance Tests.

Comprehensive testing suite for image extraction capabilities with
performance metrics, accuracy validation, and benchmarking.

Prerequisites:
- Generate test images: uv run python examples/generate_test_images.py
- Set OPENAI_API_KEY in .env

Usage:
    # Run all tests
    uv run python examples/e2e_image_extraction_performance.py

    # Run specific test
    uv run python examples/e2e_image_extraction_performance.py --test invoices

    # Quick mode (fewer images)
    uv run python examples/e2e_image_extraction_performance.py --quick

    # Export results (saved to examples/results/)
    uv run python examples/e2e_image_extraction_performance.py --export results.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ============================================================================
# Schema Definitions
# ============================================================================


class Currency(str, Enum):
    """Currency options."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    OTHER = "OTHER"


class InvoiceSchema(BaseModel):
    """Schema for invoice extraction."""

    invoice_number: str = Field(description="The invoice ID or number")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    vendor_name: str = Field(description="Name of the company issuing the invoice")
    total_amount: float = Field(description="Total amount due")
    currency: Currency = Field(default=Currency.USD, description="Currency code")
    customer_name: str | None = Field(default=None, description="Customer/bill to name")
    tax_amount: float | None = Field(default=None, description="Tax amount if shown")


class ReceiptSchema(BaseModel):
    """Schema for receipt extraction."""

    store_name: str = Field(description="Name of the store or restaurant")
    date: str = Field(description="Transaction date in YYYY-MM-DD format")
    total: float = Field(description="Total amount")
    items_count: int = Field(description="Number of line items")
    payment_method: str | None = Field(default=None, description="Payment method used")
    tax: float | None = Field(default=None, description="Tax amount")


class BusinessCardSchema(BaseModel):
    """Schema for business card extraction."""

    name: str = Field(description="Full name of the person")
    title: str | None = Field(default=None, description="Job title")
    company: str | None = Field(default=None, description="Company name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    website: str | None = Field(default=None, description="Website URL")


class ShippingLabelSchema(BaseModel):
    """Schema for shipping label extraction."""

    recipient_name: str = Field(description="Name of the recipient")
    recipient_address: str = Field(description="Full delivery address")
    sender_name: str | None = Field(default=None, description="Sender name")
    tracking_number: str | None = Field(default=None, description="Tracking number")
    weight: str | None = Field(default=None, description="Package weight")
    service: str | None = Field(default=None, description="Shipping service type")


class IDCardSchema(BaseModel):
    """Schema for ID card extraction."""

    name: str = Field(description="Full name on the card")
    employee_id: str = Field(description="Employee ID number")
    department: str | None = Field(default=None, description="Department name")
    title: str | None = Field(default=None, description="Job title")
    company: str | None = Field(default=None, description="Company name")


class ApplicationFormSchema(BaseModel):
    """Schema for application form extraction."""

    applicant_name: str = Field(description="Full name of the applicant")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    current_employer: str | None = Field(default=None, description="Current employer name")
    position: str | None = Field(default=None, description="Current or applied position")
    degree: str | None = Field(default=None, description="Educational degree")


class DataTableSchema(BaseModel):
    """Schema for data table extraction."""

    report_title: str = Field(description="Title of the report")
    regions_count: int = Field(description="Number of regions/rows in the table")
    overall_growth: str = Field(description="Overall growth percentage")


# ============================================================================
# Performance Metrics
# ============================================================================


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction."""

    image_name: str
    schema_type: str
    latency_ms: float
    success: bool
    error_message: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    accuracy_score: float | None = None
    field_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TestSuiteResults:
    """Aggregate results for a test suite."""

    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_accuracy: float
    total_tokens_input: int
    total_tokens_output: int
    metrics: list[ExtractionMetrics] = field(default_factory=list)


# ============================================================================
# Accuracy Calculation
# ============================================================================


def calculate_field_accuracy(extracted: Any, expected: Any) -> float:
    """Calculate accuracy for a single field."""
    if expected is None:
        return 1.0 if extracted is None else 0.5  # Partial credit for extracting extra

    if extracted is None:
        return 0.0

    # String comparison (case-insensitive, normalized)
    if isinstance(expected, str):
        extracted_str = str(extracted).lower().strip()
        expected_str = expected.lower().strip()

        # Exact match
        if extracted_str == expected_str:
            return 1.0

        # Partial match (one contains the other)
        if expected_str in extracted_str or extracted_str in expected_str:
            return 0.8

        # Token overlap
        expected_tokens = set(expected_str.split())
        extracted_tokens = set(extracted_str.split())
        if expected_tokens and extracted_tokens:
            overlap = len(expected_tokens & extracted_tokens)
            union = len(expected_tokens | extracted_tokens)
            return overlap / union

        return 0.0

    # Numeric comparison (with tolerance)
    if isinstance(expected, int | float):
        try:
            extracted_num = float(extracted)
            expected_num = float(expected)
            if expected_num == 0:
                return 1.0 if extracted_num == 0 else 0.0
            relative_error = abs(extracted_num - expected_num) / abs(expected_num)
            if relative_error < 0.01:  # 1% tolerance
                return 1.0
            elif relative_error < 0.05:  # 5% tolerance
                return 0.9
            elif relative_error < 0.10:  # 10% tolerance
                return 0.7
            else:
                return max(0, 1 - relative_error)
        except (ValueError, TypeError):
            return 0.0

    # Enum comparison
    if isinstance(expected, Enum):
        if isinstance(extracted, Enum):
            return 1.0 if extracted.value == expected.value else 0.0
        return 1.0 if str(extracted) == expected.value else 0.0

    return 0.0


def calculate_extraction_accuracy(
    extracted: BaseModel,
    ground_truth: dict[str, Any],
    key_fields: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Calculate overall accuracy and per-field scores."""
    field_scores: dict[str, float] = {}
    extracted_dict = extracted.model_dump()

    fields_to_check = key_fields or list(ground_truth.keys())
    weights = dict.fromkeys(fields_to_check, 1.0)

    # Higher weight for key fields
    key_field_names = {"name", "total", "total_amount", "invoice_number", "date"}
    for f in fields_to_check:
        if f in key_field_names:
            weights[f] = 2.0

    total_weight = 0.0
    weighted_score = 0.0

    for field_name in fields_to_check:
        if field_name not in extracted_dict:
            continue

        expected_value = ground_truth.get(field_name)
        extracted_value = extracted_dict.get(field_name)

        score = calculate_field_accuracy(extracted_value, expected_value)
        field_scores[field_name] = score

        weight = weights.get(field_name, 1.0)
        weighted_score += score * weight
        total_weight += weight

    overall_accuracy = weighted_score / total_weight if total_weight > 0 else 0.0
    return overall_accuracy, field_scores


# ============================================================================
# Test Runner
# ============================================================================


class ImageExtractionTester:
    """Test runner for image extraction performance."""

    def __init__(self, assets_dir: Path, model: str = "gpt-4.1"):
        self.assets_dir = assets_dir
        self.model = model
        self.ground_truth: dict[str, dict[str, Any]] = {}
        self._load_ground_truth()

        # Import here to allow script to run even without API key for --help
        from structured_extractor import DocumentExtractor

        self.extractor = DocumentExtractor(model=model)

    def _load_ground_truth(self) -> None:
        """Load ground truth data."""
        gt_path = self.assets_dir / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                self.ground_truth = json.load(f)
            print(f"📄 Loaded ground truth for {len(self.ground_truth)} images")
        else:
            print("⚠️  No ground_truth.json found - accuracy metrics unavailable")

    def extract_image(
        self,
        image_path: Path,
        schema: type[BaseModel],
    ) -> tuple[BaseModel | None, float, str | None]:
        """Extract data from image and return result with timing."""
        start_time = time.perf_counter()
        try:
            result = self.extractor.extract_from_image(
                image=str(image_path),
                schema=schema,
            )
            latency = (time.perf_counter() - start_time) * 1000
            return result.data, latency, None
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return None, latency, str(e)

    def run_test_suite(
        self,
        suite_name: str,
        image_schema_pairs: list[tuple[str, type[BaseModel], list[str] | None]],
    ) -> TestSuiteResults:
        """Run a test suite and collect metrics."""
        print(f"\n{'=' * 60}")
        print(f"🧪 Test Suite: {suite_name}")
        print("=" * 60)

        metrics: list[ExtractionMetrics] = []
        passed = 0
        failed = 0

        for image_name, schema, key_fields in image_schema_pairs:
            image_path = self.assets_dir / image_name

            if not image_path.exists():
                print(f"  ⚠️  Skipping {image_name} - file not found")
                continue

            print(f"\n  📷 Testing: {image_name}")
            print(f"     Schema: {schema.__name__}")

            data, latency, error = self.extract_image(image_path, schema)

            if error:
                print(f"     ❌ Failed: {error}")
                metrics.append(
                    ExtractionMetrics(
                        image_name=image_name,
                        schema_type=schema.__name__,
                        latency_ms=latency,
                        success=False,
                        error_message=error,
                    )
                )
                failed += 1
                continue

            # Calculate accuracy
            accuracy = 0.0
            field_scores: dict[str, float] = {}
            gt = self.ground_truth.get(image_name)

            if gt and data:
                accuracy, field_scores = calculate_extraction_accuracy(
                    data, gt, key_fields
                )

            print(f"     ⏱️  Latency: {latency:.0f}ms")
            print(f"     📊 Accuracy: {accuracy * 100:.1f}%")

            # Show extracted data
            if data:
                print(f"     📋 Extracted: {data.model_dump()}")

            # Show field-level scores
            if field_scores:
                low_scores = {k: v for k, v in field_scores.items() if v < 0.9}
                if low_scores:
                    print(f"     ⚠️  Low scores: {low_scores}")

            metrics.append(
                ExtractionMetrics(
                    image_name=image_name,
                    schema_type=schema.__name__,
                    latency_ms=latency,
                    success=True,
                    accuracy_score=accuracy,
                    field_scores=field_scores,
                )
            )
            passed += 1

        # Calculate aggregate stats
        successful_metrics = [m for m in metrics if m.success]
        latencies = [m.latency_ms for m in successful_metrics]
        accuracies = [m.accuracy_score or 0 for m in successful_metrics]

        results = TestSuiteResults(
            suite_name=suite_name,
            total_tests=len(metrics),
            passed_tests=passed,
            failed_tests=failed,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            avg_accuracy=sum(accuracies) / len(accuracies) if accuracies else 0,
            total_tokens_input=0,  # Would need to track from API response
            total_tokens_output=0,
            metrics=metrics,
        )

        # Print summary
        print(f"\n  {'─' * 50}")
        print(f"  📈 Suite Summary: {suite_name}")
        print(f"     Total: {results.total_tests} | ✅ {passed} | ❌ {failed}")
        print(f"     Avg Latency: {results.avg_latency_ms:.0f}ms")
        print(f"     Avg Accuracy: {results.avg_accuracy * 100:.1f}%")

        return results


# ============================================================================
# Test Suites
# ============================================================================


def test_invoices(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test invoice extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "invoice_simple.png",
            InvoiceSchema,
            ["invoice_number", "date", "vendor_name", "total_amount"],
        ),
        (
            "invoice_detailed.png",
            InvoiceSchema,
            ["invoice_number", "date", "vendor_name", "total_amount", "tax_amount"],
        ),
        (
            "invoice_international.png",
            InvoiceSchema,
            ["invoice_number", "date", "vendor_name", "total_amount", "currency"],
        ),
    ]
    return tester.run_test_suite("Invoice Extraction", tests)


def test_receipts(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test receipt extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "receipt_grocery.png",
            ReceiptSchema,
            ["store_name", "date", "total", "items_count"],
        ),
        (
            "receipt_restaurant.png",
            ReceiptSchema,
            ["store_name", "date", "total", "items_count"],
        ),
    ]
    return tester.run_test_suite("Receipt Extraction", tests)


def test_business_cards(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test business card extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "business_card_modern.png",
            BusinessCardSchema,
            ["name", "title", "company", "email", "phone"],
        ),
        (
            "business_card_minimal.png",
            BusinessCardSchema,
            ["name", "title", "email", "phone"],
        ),
    ]
    return tester.run_test_suite("Business Card Extraction", tests)


def test_forms(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test form extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "form_application.png",
            ApplicationFormSchema,
            ["applicant_name", "email", "phone", "current_employer"],
        ),
    ]
    return tester.run_test_suite("Form Extraction", tests)


def test_shipping_labels(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test shipping label extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "shipping_label.png",
            ShippingLabelSchema,
            ["recipient_name", "recipient_address", "tracking_number"],
        ),
    ]
    return tester.run_test_suite("Shipping Label Extraction", tests)


def test_id_cards(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test ID card extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "id_card.png",
            IDCardSchema,
            ["name", "employee_id", "department", "company"],
        ),
    ]
    return tester.run_test_suite("ID Card Extraction", tests)


def test_data_tables(tester: ImageExtractionTester) -> TestSuiteResults:
    """Test data table extraction."""
    tests: list[tuple[str, type[BaseModel], list[str] | None]] = [
        (
            "data_table.png",
            DataTableSchema,
            ["report_title", "regions_count", "overall_growth"],
        ),
    ]
    return tester.run_test_suite("Data Table Extraction", tests)


# ============================================================================
# Report Generation
# ============================================================================


def generate_report(results: list[TestSuiteResults]) -> dict[str, Any]:
    """Generate comprehensive report."""
    total_tests = sum(r.total_tests for r in results)
    total_passed = sum(r.passed_tests for r in results)
    total_failed = sum(r.failed_tests for r in results)

    all_latencies: list[float] = []
    all_accuracies: list[float] = []
    for r in results:
        for m in r.metrics:
            if m.success:
                all_latencies.append(m.latency_ms)
                if m.accuracy_score is not None:
                    all_accuracies.append(m.accuracy_score)

    suites_data: list[dict[str, Any]] = []

    for r in results:
        suite_data: dict[str, Any] = {
            "name": r.suite_name,
            "total_tests": r.total_tests,
            "passed": r.passed_tests,
            "failed": r.failed_tests,
            "avg_latency_ms": r.avg_latency_ms,
            "avg_accuracy": r.avg_accuracy,
            "tests": [
                {
                    "image": m.image_name,
                    "schema": m.schema_type,
                    "success": m.success,
                    "latency_ms": m.latency_ms,
                    "accuracy": m.accuracy_score,
                    "error": m.error_message,
                }
                for m in r.metrics
            ],
        }
        suites_data.append(suite_data)

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "avg_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
            "min_latency_ms": min(all_latencies) if all_latencies else 0,
            "max_latency_ms": max(all_latencies) if all_latencies else 0,
            "avg_accuracy": sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0,
        },
        "suites": suites_data,
    }

    return report


def print_final_report(results: list[TestSuiteResults]) -> None:
    """Print final summary report."""
    total_tests = sum(r.total_tests for r in results)
    total_passed = sum(r.passed_tests for r in results)
    total_failed = sum(r.failed_tests for r in results)

    all_latencies = []
    all_accuracies = []
    for r in results:
        for m in r.metrics:
            if m.success:
                all_latencies.append(m.latency_ms)
                if m.accuracy_score is not None:
                    all_accuracies.append(m.accuracy_score)

    print("\n")
    print("=" * 60)
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 60)

    print("\n📈 Overall Statistics:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"   Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")

    print("\n⏱️  Latency Statistics:")
    if all_latencies:
        print(f"   Average: {sum(all_latencies)/len(all_latencies):.0f}ms")
        print(f"   Min: {min(all_latencies):.0f}ms")
        print(f"   Max: {max(all_latencies):.0f}ms")
        # Percentiles
        sorted_latencies = sorted(all_latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
        print(f"   P50: {p50:.0f}ms")
        print(f"   P90: {p90:.0f}ms")
    else:
        print("   No successful extractions")

    print("\n🎯 Accuracy Statistics:")
    if all_accuracies:
        print(f"   Average: {sum(all_accuracies)/len(all_accuracies)*100:.1f}%")
        print(f"   Min: {min(all_accuracies)*100:.1f}%")
        print(f"   Max: {max(all_accuracies)*100:.1f}%")
    else:
        print("   No accuracy data available")

    print("\n📋 Per-Suite Results:")
    print(f"   {'Suite':<25} {'Pass Rate':<12} {'Avg Latency':<12} {'Accuracy':<10}")
    print(f"   {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    for r in results:
        pass_rate = f"{r.passed_tests}/{r.total_tests}"
        latency = f"{r.avg_latency_ms:.0f}ms"
        accuracy = f"{r.avg_accuracy*100:.1f}%"
        print(f"   {r.suite_name:<25} {pass_rate:<12} {latency:<12} {accuracy:<10}")

    print("\n" + "=" * 60)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run image extraction performance tests."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="E2E Image Extraction Performance Tests"
    )
    parser.add_argument(
        "--test",
        choices=[
            "invoices",
            "receipts",
            "business_cards",
            "forms",
            "shipping",
            "id_cards",
            "tables",
        ],
        help="Run specific test suite only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests (fewer images)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="Model to use for extraction",
    )

    args = parser.parse_args()

    # Setup paths
    assets_dir = Path(__file__).parent / "assets"
    results_dir = Path(__file__).parent / "results"

    if not assets_dir.exists():
        print("❌ Assets directory not found!")
        print("   Please run: uv run python examples/generate_test_images.py")
        sys.exit(1)

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Image Extraction Performance Tests")
    print(f"   Model: {args.model}")
    print(f"   Assets: {assets_dir}")

    # Initialize tester
    tester = ImageExtractionTester(assets_dir, model=args.model)

    # Run tests
    results: list[TestSuiteResults] = []

    test_functions = {
        "invoices": test_invoices,
        "receipts": test_receipts,
        "business_cards": test_business_cards,
        "forms": test_forms,
        "shipping": test_shipping_labels,
        "id_cards": test_id_cards,
        "tables": test_data_tables,
    }

    if args.test:
        # Run specific test
        if args.test in test_functions:
            results.append(test_functions[args.test](tester))
    elif args.quick:
        # Quick mode - just invoices and receipts
        results.append(test_invoices(tester))
        results.append(test_receipts(tester))
    else:
        # Run all tests
        for test_func in test_functions.values():
            results.append(test_func(tester))

    # Print final report
    print_final_report(results)

    # Export if requested
    if args.export:
        report = generate_report(results)
        # If export path is just a filename, put it in results dir
        export_path = Path(args.export)
        if not export_path.is_absolute() and export_path.parent == Path("."):
            export_path = results_dir / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Results exported to: {export_path}")


if __name__ == "__main__":
    main()
