"""Performance metrics collection and analysis for structured extraction.

This module provides:
1. Latency measurement (per-extraction and aggregate)
2. Throughput calculation (extractions per minute)
3. Token usage tracking
4. Cost analysis
5. Accuracy assessment against ground truth
6. Performance report generation

Usage:
    python examples/e2e_performance_metrics.py
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import DocumentExtractor, ExtractionResult


# Type definition for test cases
class TestCase(TypedDict):
    """Type definition for test cases with ground truth."""

    id: str
    schema: type[BaseModel]
    document: str
    ground_truth: dict[str, Any]

# Load environment variables
load_dotenv()


# ============================================================================
# Metrics Data Classes
# ============================================================================


@dataclass
class ExtractionMetrics:
    """Metrics for a single extraction."""

    test_id: str
    schema_name: str
    document_length: int
    success: bool
    latency_ms: float
    tokens_input: int | None
    tokens_output: int | None
    tokens_total: int | None
    cost_usd: float | None
    cached: bool
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing extraction to ground truth."""

    test_id: str
    total_fields: int
    correct_fields: int
    partially_correct_fields: int
    incorrect_fields: int
    missing_fields: int
    field_accuracy: dict[str, bool] = field(default_factory=dict)

    @property
    def accuracy_score(self) -> float:
        """Calculate accuracy score (0-1)."""
        if self.total_fields == 0:
            return 0.0
        return (self.correct_fields + 0.5 * self.partially_correct_fields) / self.total_fields

    @property
    def precision(self) -> float:
        """Calculate precision."""
        extracted = self.correct_fields + self.partially_correct_fields + self.incorrect_fields
        if extracted == 0:
            return 0.0
        return self.correct_fields / extracted


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple extractions."""

    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    cached_extractions: int = 0

    # Latency metrics
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_std_ms: float = 0.0

    # Token metrics
    total_tokens: int = 0
    avg_tokens_per_extraction: float = 0.0

    # Cost metrics
    total_cost_usd: float = 0.0
    avg_cost_per_extraction_usd: float = 0.0

    # Throughput
    extractions_per_minute: float = 0.0
    total_duration_seconds: float = 0.0

    # Accuracy (if ground truth available)
    avg_accuracy: float | None = None
    avg_precision: float | None = None


class MetricsCollector:
    """Collects and analyzes extraction metrics."""

    def __init__(self) -> None:
        self.extraction_metrics: list[ExtractionMetrics] = []
        self.accuracy_metrics: list[AccuracyMetrics] = []
        self._start_time: float | None = None

    def start_session(self) -> None:
        """Start a metrics collection session."""
        self._start_time = time.time()

    def record_extraction(
        self,
        test_id: str,
        schema_name: str,
        document_length: int,
        success: bool,
        latency_ms: float,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        cached: bool = False,
        error: str | None = None,
    ) -> ExtractionMetrics:
        """Record metrics for a single extraction."""
        metrics = ExtractionMetrics(
            test_id=test_id,
            schema_name=schema_name,
            document_length=document_length,
            success=success,
            latency_ms=latency_ms,
            tokens_input=None,  # Can be split if available
            tokens_output=None,
            tokens_total=tokens_used,
            cost_usd=cost_usd,
            cached=cached,
            error=error,
        )
        self.extraction_metrics.append(metrics)
        return metrics

    def record_accuracy(
        self,
        test_id: str,
        extracted_data: dict[str, Any],
        ground_truth: dict[str, Any],
        partial_match_fields: set[str] | None = None,
    ) -> AccuracyMetrics:
        """Record accuracy metrics by comparing extraction to ground truth."""
        partial_match_fields = partial_match_fields or set()

        total_fields = len(ground_truth)
        correct = 0
        partial = 0
        incorrect = 0
        missing = 0
        field_accuracy: dict[str, bool] = {}

        for field_name, expected_value in ground_truth.items():
            if field_name not in extracted_data:
                missing += 1
                field_accuracy[field_name] = False
            elif extracted_data[field_name] == expected_value:
                correct += 1
                field_accuracy[field_name] = True
            elif field_name in partial_match_fields and self._partial_match(
                extracted_data[field_name], expected_value
            ):
                partial += 1
                field_accuracy[field_name] = True  # Count partial as correct for field
            else:
                incorrect += 1
                field_accuracy[field_name] = False

        metrics = AccuracyMetrics(
            test_id=test_id,
            total_fields=total_fields,
            correct_fields=correct,
            partially_correct_fields=partial,
            incorrect_fields=incorrect,
            missing_fields=missing,
            field_accuracy=field_accuracy,
        )
        self.accuracy_metrics.append(metrics)
        return metrics

    def _partial_match(self, extracted: Any, expected: Any) -> bool:
        """Check for partial match (e.g., substring, close numeric values)."""
        if isinstance(expected, str) and isinstance(extracted, str):
            return expected.lower() in extracted.lower() or extracted.lower() in expected.lower()
        if isinstance(expected, (int, float)) and isinstance(extracted, (int, float)):
            # Allow 5% tolerance for numeric values
            if expected == 0:
                return extracted == 0
            return abs(extracted - expected) / abs(expected) <= 0.05
        return False

    def compute_aggregate_metrics(self) -> AggregateMetrics:
        """Compute aggregate metrics across all recorded extractions."""
        if not self.extraction_metrics:
            return AggregateMetrics()

        latencies = [m.latency_ms for m in self.extraction_metrics if not m.cached]
        tokens = [m.tokens_total for m in self.extraction_metrics if m.tokens_total]
        costs = [m.cost_usd for m in self.extraction_metrics if m.cost_usd]

        # Calculate total duration
        total_duration = time.time() - self._start_time if self._start_time else 0.0

        aggregate = AggregateMetrics(
            total_extractions=len(self.extraction_metrics),
            successful_extractions=sum(1 for m in self.extraction_metrics if m.success),
            failed_extractions=sum(1 for m in self.extraction_metrics if not m.success),
            cached_extractions=sum(1 for m in self.extraction_metrics if m.cached),
        )

        # Latency stats (excluding cached)
        if latencies:
            aggregate.latency_min_ms = min(latencies)
            aggregate.latency_max_ms = max(latencies)
            aggregate.latency_mean_ms = statistics.mean(latencies)
            aggregate.latency_median_ms = statistics.median(latencies)
            if len(latencies) > 1:
                aggregate.latency_std_ms = statistics.stdev(latencies)
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            aggregate.latency_p95_ms = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
            aggregate.latency_p99_ms = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

        # Token stats
        if tokens:
            aggregate.total_tokens = sum(tokens)
            aggregate.avg_tokens_per_extraction = statistics.mean(tokens)

        # Cost stats
        if costs:
            aggregate.total_cost_usd = sum(costs)
            aggregate.avg_cost_per_extraction_usd = statistics.mean(costs)

        # Throughput
        aggregate.total_duration_seconds = total_duration
        if total_duration > 0:
            aggregate.extractions_per_minute = (
                aggregate.total_extractions / total_duration * 60
            )

        # Accuracy stats
        if self.accuracy_metrics:
            aggregate.avg_accuracy = statistics.mean(
                m.accuracy_score for m in self.accuracy_metrics
            )
            aggregate.avg_precision = statistics.mean(
                m.precision for m in self.accuracy_metrics
            )

        return aggregate

    def generate_report(self, output_path: str | Path | None = None) -> str:
        """Generate a detailed performance report."""
        aggregate = self.compute_aggregate_metrics()

        report_lines = [
            "=" * 70,
            "📊 PERFORMANCE METRICS REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Overview",
            f"Total Extractions:     {aggregate.total_extractions}",
            f"Successful:            {aggregate.successful_extractions} "
            f"({aggregate.successful_extractions / aggregate.total_extractions * 100:.1f}%)"
            if aggregate.total_extractions > 0 else "",
            f"Failed:                {aggregate.failed_extractions}",
            f"Cached:                {aggregate.cached_extractions}",
            f"Total Duration:        {aggregate.total_duration_seconds:.1f}s",
            "",
            "## Latency (excluding cached)",
            f"Min:                   {aggregate.latency_min_ms:.0f} ms",
            f"Max:                   {aggregate.latency_max_ms:.0f} ms",
            f"Mean:                  {aggregate.latency_mean_ms:.0f} ms",
            f"Median:                {aggregate.latency_median_ms:.0f} ms",
            f"Std Dev:               {aggregate.latency_std_ms:.0f} ms",
            f"P95:                   {aggregate.latency_p95_ms:.0f} ms",
            f"P99:                   {aggregate.latency_p99_ms:.0f} ms",
            "",
            "## Throughput",
            f"Extractions/min:       {aggregate.extractions_per_minute:.2f}",
            "",
            "## Token Usage",
            f"Total Tokens:          {aggregate.total_tokens:,}",
            f"Avg Tokens/Extraction: {aggregate.avg_tokens_per_extraction:.0f}",
            "",
            "## Cost",
            f"Total Cost:            ${aggregate.total_cost_usd:.4f}",
            f"Avg Cost/Extraction:   ${aggregate.avg_cost_per_extraction_usd:.6f}",
        ]

        if aggregate.avg_accuracy is not None:
            report_lines.extend([
                "",
                "## Accuracy",
                f"Avg Accuracy:          {aggregate.avg_accuracy:.1%}",
                f"Avg Precision:         {aggregate.avg_precision:.1%}"
                if aggregate.avg_precision else "",
            ])

        # Add per-schema breakdown
        schemas = {m.schema_name for m in self.extraction_metrics}
        if len(schemas) > 1:
            report_lines.extend(["", "## Per-Schema Breakdown"])
            for schema in schemas:
                schema_metrics = [m for m in self.extraction_metrics if m.schema_name == schema]
                schema_success = sum(1 for m in schema_metrics if m.success)
                schema_latencies = [m.latency_ms for m in schema_metrics if not m.cached]
                avg_latency = statistics.mean(schema_latencies) if schema_latencies else 0
                report_lines.append(
                    f"  {schema}: {schema_success}/{len(schema_metrics)} success, "
                    f"{avg_latency:.0f}ms avg latency"
                )

        report_lines.extend(["", "=" * 70])

        report = "\n".join(report_lines)

        if output_path:
            Path(output_path).write_text(report)

        return report

    def export_json(self, output_path: str | Path) -> None:
        """Export all metrics to JSON."""
        data = {
            "generated": datetime.now().isoformat(),
            "aggregate": self.compute_aggregate_metrics().__dict__,
            "extractions": [m.__dict__ for m in self.extraction_metrics],
            "accuracy": [
                {**m.__dict__, "accuracy_score": m.accuracy_score, "precision": m.precision}
                for m in self.accuracy_metrics
            ],
        }
        Path(output_path).write_text(json.dumps(data, indent=2, default=str))


# ============================================================================
# Test Schemas and Ground Truth
# ============================================================================


class Invoice(BaseModel):
    """Invoice for performance testing."""

    invoice_number: str = Field(description="Invoice identifier")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Vendor name")
    total_amount: float = Field(description="Total amount")


class Person(BaseModel):
    """Person info for testing."""

    name: str = Field(description="Full name")
    age: int | None = Field(default=None, description="Age")
    email: str | None = Field(default=None, description="Email")
    occupation: str | None = Field(default=None, description="Job")


class Product(BaseModel):
    """Product info for testing."""

    name: str = Field(description="Product name")
    price: float = Field(description="Price")
    category: str | None = Field(default=None, description="Category")


# Test data with ground truth
TEST_CASES: list[TestCase] = [
    {
        "id": "invoice_1",
        "schema": Invoice,
        "document": """
        INVOICE #INV-001
        Date: 2024-11-25
        From: ABC Corp
        Total: $1,250.00
        """,
        "ground_truth": {
            "invoice_number": "INV-001",
            "date": "2024-11-25",
            "vendor_name": "ABC Corp",
            "total_amount": 1250.00,
        },
    },
    {
        "id": "invoice_2",
        "schema": Invoice,
        "document": """
        Bill #2024-999
        November 25, 2024
        XYZ Industries
        Amount Due: $3,500.50
        """,
        "ground_truth": {
            "invoice_number": "2024-999",
            "date": "2024-11-25",
            "vendor_name": "XYZ Industries",
            "total_amount": 3500.50,
        },
    },
    {
        "id": "person_1",
        "schema": Person,
        "document": """
        John Smith
        Age: 35
        Email: john@email.com
        Software Engineer
        """,
        "ground_truth": {
            "name": "John Smith",
            "age": 35,
            "email": "john@email.com",
            "occupation": "Software Engineer",
        },
    },
    {
        "id": "person_2",
        "schema": Person,
        "document": """
        Jane Doe, 28 years old
        Contact: jane.doe@company.com
        Works as Data Scientist
        """,
        "ground_truth": {
            "name": "Jane Doe",
            "age": 28,
            "email": "jane.doe@company.com",
            "occupation": "Data Scientist",
        },
    },
    {
        "id": "product_1",
        "schema": Product,
        "document": """
        Product: Wireless Headphones
        Price: $149.99
        Category: Electronics
        """,
        "ground_truth": {
            "name": "Wireless Headphones",
            "price": 149.99,
            "category": "Electronics",
        },
    },
]


# ============================================================================
# Performance Tests
# ============================================================================


def run_performance_tests(extractor: DocumentExtractor, collector: MetricsCollector) -> None:
    """Run performance tests with ground truth validation."""
    print("\n" + "=" * 60)
    print("🏃 Running Performance Tests")
    print("=" * 60)

    collector.start_session()

    for test_case in TEST_CASES:
        test_id = test_case["id"]
        schema = test_case["schema"]
        document = test_case["document"]
        ground_truth = test_case["ground_truth"]

        print(f"\n   Running: {test_id}...", end=" ")

        start_time = time.time()
        result = extractor.extract(document, schema=schema)
        latency_ms = (time.time() - start_time) * 1000

        # Record extraction metrics
        collector.record_extraction(
            test_id=test_id,
            schema_name=schema.__name__,
            document_length=len(document),
            success=result.success,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            cached=result.cached,
            error=result.error,
        )

        # Record accuracy metrics if successful
        if result.success:
            extracted_data = result.data.model_dump()
            accuracy = collector.record_accuracy(
                test_id=test_id,
                extracted_data=extracted_data,
                ground_truth=ground_truth,
                partial_match_fields={"date", "occupation"},  # Allow flexible matching
            )
            print(f"✅ Accuracy: {accuracy.accuracy_score:.0%}")
        else:
            print(f"❌ {result.error}")


def run_latency_benchmark(
    extractor: DocumentExtractor, collector: MetricsCollector, iterations: int = 5
) -> None:
    """Run latency benchmark with multiple iterations."""
    print("\n" + "=" * 60)
    print(f"⏱️ Latency Benchmark ({iterations} iterations)")
    print("=" * 60)

    document = """
    INVOICE #BENCH-001
    Date: 2024-11-25
    Vendor: Benchmark Corp
    Total: $999.99
    """

    # Clear cache for fresh benchmark
    for i in range(iterations):
        # Add iteration marker to avoid cache hits
        doc_with_marker = document + f"\n<!-- iteration {i} -->"

        start_time = time.time()
        result = extractor.extract(doc_with_marker, schema=Invoice)
        latency_ms = (time.time() - start_time) * 1000

        collector.record_extraction(
            test_id=f"latency_bench_{i}",
            schema_name="Invoice",
            document_length=len(doc_with_marker),
            success=result.success,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            cached=result.cached,
        )

        print(f"   Iteration {i + 1}: {latency_ms:.0f}ms {'(cached)' if result.cached else ''}")


def run_throughput_test(
    extractor: DocumentExtractor, collector: MetricsCollector, num_extractions: int = 10
) -> None:
    """Test extraction throughput."""
    print("\n" + "=" * 60)
    print(f"🚀 Throughput Test ({num_extractions} extractions)")
    print("=" * 60)

    documents = [
        f"""
        Invoice #{i + 1}
        Date: 2024-11-{25 - i}
        Vendor: Company {i + 1}
        Total: ${(i + 1) * 100}.00
        """
        for i in range(num_extractions)
    ]

    start_time = time.time()

    for i, doc in enumerate(documents):
        result = extractor.extract(doc, schema=Invoice)

        collector.record_extraction(
            test_id=f"throughput_{i}",
            schema_name="Invoice",
            document_length=len(doc),
            success=result.success,
            latency_ms=0,  # Will use aggregate timing
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            cached=result.cached,
        )

        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"   Completed {i + 1}/{num_extractions}")

    total_time = time.time() - start_time
    throughput = num_extractions / total_time * 60

    print(f"\n   Total time: {total_time:.1f}s")
    print(f"   Throughput: {throughput:.1f} extractions/minute")


def run_schema_complexity_test(
    extractor: DocumentExtractor, collector: MetricsCollector
) -> None:
    """Test performance across different schema complexities."""
    print("\n" + "=" * 60)
    print("📐 Schema Complexity Test")
    print("=" * 60)

    # Simple schema
    class SimpleSchema(BaseModel):
        name: str = Field(description="Name")
        value: float = Field(description="Value")

    # Medium schema
    class MediumSchema(BaseModel):
        name: str = Field(description="Name")
        description: str = Field(description="Description")
        price: float = Field(description="Price")
        quantity: int = Field(description="Quantity")
        category: str | None = Field(default=None, description="Category")

    # Complex schema
    class LineItem(BaseModel):
        name: str = Field(description="Item name")
        price: float = Field(description="Price")

    class ComplexSchema(BaseModel):
        title: str = Field(description="Title")
        description: str = Field(description="Description")
        items: list[LineItem] = Field(description="Items")
        total: float = Field(description="Total")
        metadata: dict[str, str] = Field(default_factory=dict, description="Metadata")

    schemas: list[tuple[str, type[BaseModel], str]] = [
        ("simple", SimpleSchema, "Name: Test\nValue: 100"),
        (
            "medium",
            MediumSchema,
            "Name: Product\nDescription: A great product\nPrice: $50\n"
            "Quantity: 5\nCategory: Electronics",
        ),
        ("complex", ComplexSchema, """
            Title: Order
            Description: Customer order

            Items:
            - Item A: $25
            - Item B: $35
            - Item C: $40

            Total: $100
            Tags: urgent, priority
        """),
    ]

    for complexity, schema, doc in schemas:
        start_time = time.time()
        result: ExtractionResult[Any] = extractor.extract(doc, schema=schema)
        latency_ms = (time.time() - start_time) * 1000

        collector.record_extraction(
            test_id=f"complexity_{complexity}",
            schema_name=schema.__name__,
            document_length=len(doc),
            success=result.success,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            cached=result.cached,
        )

        print(f"   {complexity.capitalize()}: {latency_ms:.0f}ms, {result.tokens_used or 0} tokens")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all performance metric tests."""
    print("\n📈 Structured Extractor - Performance Metrics")
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

    # Initialize metrics collector
    collector = MetricsCollector()

    # Run all performance tests
    run_performance_tests(extractor, collector)
    run_latency_benchmark(extractor, collector, iterations=5)
    run_throughput_test(extractor, collector, num_extractions=10)
    run_schema_complexity_test(extractor, collector)

    # Generate and print report
    print("\n")
    report = collector.generate_report()
    print(report)

    # Export to JSON
    output_dir = Path(__file__).parent / "metrics_output"
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    collector.export_json(json_path)
    print(f"\n📁 Metrics exported to: {json_path}")

    report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    collector.generate_report(report_path)
    print(f"📁 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
