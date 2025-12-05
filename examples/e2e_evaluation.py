#!/usr/bin/env python
"""Example: Performance Evaluation of Document Extraction.

This example demonstrates how to use the evaluation module to measure
extraction quality against ground truth data.

Features demonstrated:
- Setting up an ExtractionEvaluator
- Evaluating extractions with different comparators
- Computing precision, recall, F1, and accuracy metrics
- Generating evaluation reports

Usage:
    # With mocked extraction (no API calls)
    python examples/e2e_evaluation.py

    # With real extraction (requires OPENAI_API_KEY)
    python examples/e2e_evaluation.py --live
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ============================================================================
# Test Schemas
# ============================================================================


class Invoice(BaseModel):
    """Invoice extraction schema."""

    invoice_number: str = Field(description="Unique invoice identifier")
    date: str = Field(description="Invoice date (YYYY-MM-DD)")
    vendor_name: str = Field(description="Name of the vendor")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")


class Person(BaseModel):
    """Person information schema."""

    name: str = Field(description="Full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")


# ============================================================================
# Test Data
# ============================================================================

TEST_CASES = [
    {
        "id": "invoice_001",
        "document": """
        INVOICE #INV-2024-001

        Date: 2024-11-25
        From: Acme Corporation

        Total Due: $1,250.00 USD
        """,
        "ground_truth": {
            "invoice_number": "INV-2024-001",
            "date": "2024-11-25",
            "vendor_name": "Acme Corporation",
            "total_amount": 1250.00,
            "currency": "USD",
        },
        "schema": Invoice,
    },
    {
        "id": "invoice_002",
        "document": """
        Bill #2024-999
        November 30, 2024

        XYZ Industries Ltd.

        Amount: €3,500.50
        """,
        "ground_truth": {
            "invoice_number": "2024-999",
            "date": "2024-11-30",
            "vendor_name": "XYZ Industries Ltd.",
            "total_amount": 3500.50,
            "currency": "EUR",
        },
        "schema": Invoice,
    },
    {
        "id": "person_001",
        "document": """
        Contact Information:
        John Smith
        Email: john.smith@email.com
        Phone: +1 (555) 123-4567
        """,
        "ground_truth": {
            "name": "John Smith",
            "email": "john.smith@email.com",
            "phone": "+1 (555) 123-4567",
        },
        "schema": Person,
    },
]


# ============================================================================
# Mock Extraction Results (for demo without API)
# ============================================================================

MOCK_EXTRACTIONS = {
    "invoice_001": {
        "invoice_number": "INV-2024-001",
        "date": "2024-11-25",
        "vendor_name": "Acme Corporation",
        "total_amount": 1250.00,
        "currency": "USD",
    },
    "invoice_002": {
        "invoice_number": "2024-999",
        "date": "2024-11-30",
        "vendor_name": "XYZ Industries",  # Slightly different (missing "Ltd.")
        "total_amount": 3500.00,  # Slightly off
        "currency": "EUR",
    },
    "person_001": {
        "name": "John Smith",
        "email": "john.smith@email.com",
        "phone": "(555) 123-4567",  # Different format
    },
}


def run_evaluation_demo(live_mode: bool = False) -> None:
    """Run the evaluation demonstration."""
    from structured_extractor import DocumentExtractor
    from structured_extractor.evaluation import (
        EvaluationReporter,
        ExtractionEvaluator,
    )

    print("=" * 70)
    print("  📊 Performance Evaluation Demo")
    print("=" * 70)
    print()

    if live_mode:
        print("🔴 LIVE MODE: Using real API calls")
        extractor = DocumentExtractor(model="gpt-4.1")
    else:
        print("🟢 DEMO MODE: Using mocked extractions")
        extractor = MagicMock(spec=DocumentExtractor)

    # Create evaluator with default thresholds
    evaluator = ExtractionEvaluator(
        extractor,
        default_string_threshold=0.85,  # 85% similarity for strings
        default_numeric_tolerance=0.02,  # 2% tolerance for numbers
    )

    print("\nDefault string threshold: 85%")
    print("Default numeric tolerance: 2%")
    print()

    # Run evaluations
    print("-" * 70)
    print("  Running Evaluations")
    print("-" * 70)

    for test_case in TEST_CASES:
        test_id = test_case["id"]
        schema = test_case["schema"]
        ground_truth = test_case["ground_truth"]

        print(f"\n📄 Test: {test_id} ({schema.__name__})")

        if live_mode:
            # Use real extraction
            result = evaluator.evaluate(
                document=test_case["document"],
                schema=schema,
                ground_truth=ground_truth,
                test_id=test_id,
            )
        else:
            # Use mocked extraction
            mock_data = MOCK_EXTRACTIONS[test_id]
            result = evaluator.evaluate_without_extraction(
                extracted_data=mock_data,
                ground_truth=ground_truth,
                schema=schema,
                test_id=test_id,
            )

        # Print per-evaluation results
        print(f"   Precision: {result.precision:.0%} | Recall: {result.recall:.0%} | "
              f"F1: {result.f1_score:.0%} | Score: {result.mean_score:.0%}")

        # Show field-level details
        for match in result.field_matches:
            status = "✓" if match.matched else "✗"
            print(f"     {status} {match.field_name}: {match.score:.2f} "
                  f"({match.comparator_type})")

    # Compute aggregate metrics
    print()
    print("-" * 70)
    print("  Aggregate Metrics")
    print("-" * 70)

    metrics = evaluator.compute_aggregate_metrics()

    print("\n📈 Summary:")
    print(f"   Total evaluations: {metrics.total_evaluations}")
    print(f"   Total fields: {metrics.total_fields}")
    print()
    print("📊 Binary Metrics (Micro-averaged):")
    print(f"   Precision: {metrics.micro_precision:.2%}")
    print(f"   Recall:    {metrics.micro_recall:.2%}")
    print(f"   F1 Score:  {metrics.micro_f1:.2%}")
    print(f"   Accuracy:  {metrics.micro_accuracy:.2%}")
    print()
    print("📊 Binary Metrics (Macro-averaged):")
    print(f"   Precision: {metrics.macro_precision:.2%}")
    print(f"   Recall:    {metrics.macro_recall:.2%}")
    print(f"   F1 Score:  {metrics.macro_f1:.2%}")
    print(f"   Accuracy:  {metrics.macro_accuracy:.2%}")
    print()
    print("🎯 Score Metrics:")
    print(f"   Mean Score:   {metrics.mean_score:.2%}")
    print(f"   Median Score: {metrics.median_score:.2%}")
    print(f"   Min Score:    {metrics.min_score:.2%}")
    print(f"   Max Score:    {metrics.max_score:.2%}")

    # Per-field breakdown
    print()
    print("-" * 70)
    print("  Per-Field Metrics")
    print("-" * 70)
    print()
    print(f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Score':>10}")
    print("-" * 60)
    for field_name, fm in sorted(metrics.field_metrics.items()):
        print(f"{field_name:<20} {fm.precision:>10.0%} {fm.recall:>10.0%} "
              f"{fm.f1_score:>10.0%} {fm.mean_score:>10.0%}")

    # Generate reports
    print()
    print("-" * 70)
    print("  Generating Reports")
    print("-" * 70)

    reporter = EvaluationReporter(
        evaluator.evaluations,
        metrics,
        title="Extraction Evaluation Report",
    )

    # Save reports
    output_dir = Path("examples/results")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"

    reporter.save(json_path)
    reporter.save(md_path)

    print("\n✅ Reports saved:")
    print(f"   - {json_path}")
    print(f"   - {md_path}")

    # Print summary
    print()
    reporter.print_summary()


def demo_custom_comparators() -> None:
    """Demonstrate using custom comparators."""
    from structured_extractor.evaluation import (
        ContainsComparator,
        DateComparator,
        ExactComparator,
        ListComparator,
        NumericComparator,
        StringComparator,
    )

    print()
    print("=" * 70)
    print("  🔧 Custom Comparators Demo")
    print("=" * 70)
    print()

    # ExactComparator
    print("ExactComparator:")
    comp = ExactComparator()
    result = comp.compare("INV-001", "INV-001")
    print(f"  'INV-001' vs 'INV-001': matched={result.matched}, score={result.score}")
    result = comp.compare("INV-001", "INV-002")
    print(f"  'INV-001' vs 'INV-002': matched={result.matched}, score={result.score}")
    print()

    # NumericComparator
    print("NumericComparator (5% tolerance):")
    comp = NumericComparator(threshold=0.05, mode="relative")
    result = comp.compare(100.0, 103.0)
    print(f"  100.0 vs 103.0: matched={result.matched}, score={result.score:.3f}")
    result = comp.compare(100.0, 110.0)
    print(f"  100.0 vs 110.0: matched={result.matched}, score={result.score:.3f}")
    print()

    # StringComparator
    print("StringComparator (80% threshold):")
    comp = StringComparator(threshold=0.8)
    result = comp.compare("Acme Corporation", "Acme Corp")
    print(f"  'Acme Corporation' vs 'Acme Corp': matched={result.matched}, "
          f"score={result.score:.3f}")
    result = comp.compare("Acme Corporation", "XYZ Industries")
    print(f"  'Acme Corporation' vs 'XYZ Industries': matched={result.matched}, "
          f"score={result.score:.3f}")
    print()

    # ContainsComparator
    print("ContainsComparator:")
    comp = ContainsComparator()
    result = comp.compare("John", "John Smith")
    print(f"  'John' vs 'John Smith': matched={result.matched}, score={result.score:.3f}")
    print()

    # DateComparator
    print("DateComparator:")
    comp = DateComparator()
    result = comp.compare("2024-11-25", "November 25, 2024")
    print(f"  '2024-11-25' vs 'November 25, 2024': matched={result.matched}, "
          f"score={result.score:.3f}")
    result = comp.compare("2024-11-25", "2024-11-30")
    print(f"  '2024-11-25' vs '2024-11-30': matched={result.matched}, "
          f"score={result.score:.3f}")
    print()

    # ListComparator
    print("ListComparator (order-insensitive):")
    comp = ListComparator(order_sensitive=False)
    result = comp.compare(["a", "b", "c"], ["c", "b", "a"])
    print(f"  ['a','b','c'] vs ['c','b','a']: matched={result.matched}, "
          f"score={result.score:.3f}")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance Evaluation Demo for structured-extractor"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live API calls instead of mocked extractions",
    )
    parser.add_argument(
        "--comparators",
        action="store_true",
        help="Show custom comparators demo",
    )
    args = parser.parse_args()

    if args.comparators:
        demo_custom_comparators()
    else:
        run_evaluation_demo(live_mode=args.live)


if __name__ == "__main__":
    main()
