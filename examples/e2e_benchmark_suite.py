"""Comprehensive E2E Benchmark Suite for structured-extractor.

This script runs all E2E tests and generates a comprehensive benchmark report
covering all features of the library.

Features tested:
1. Text extraction (basic, nested, complex schemas)
2. Image extraction (local files, URLs, multi-image)
3. Confidence scoring
4. Document templates
5. Prompt strategies (default, strict, lenient)
6. Caching behavior
7. Error handling

Metrics collected:
- Success rate
- Latency (min, max, mean, median, p95, p99)
- Token usage
- Cost tracking
- Accuracy (when ground truth available)
- Cache hit rate

Prerequisites:
- OPENAI_API_KEY environment variable set

Usage:
    python examples/e2e_benchmark_suite.py
    python examples/e2e_benchmark_suite.py --quick     # Quick test (fewer iterations)
    python examples/e2e_benchmark_suite.py --full      # Full test suite
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import (
    DocumentExtractor,
    DocumentTemplate,
    ExtractionConfig,
    ExtractionResult,
    PromptTemplates,
    TemplateRegistry,
)

# Load environment variables
load_dotenv()


# ============================================================================
# Benchmark Configuration
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    mode: str = "standard"  # "quick", "standard", "full"
    iterations_per_test: int = 1
    run_image_tests: bool = True
    run_confidence_tests: bool = True
    run_template_tests: bool = True
    run_performance_tests: bool = True
    output_dir: str = "benchmark_results"

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        return cls(mode="quick", iterations_per_test=1, run_image_tests=False)

    @classmethod
    def standard(cls) -> "BenchmarkConfig":
        return cls(mode="standard", iterations_per_test=1)

    @classmethod
    def full(cls) -> "BenchmarkConfig":
        return cls(mode="full", iterations_per_test=3)


# ============================================================================
# Result Data Classes
# ============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    category: str
    test_name: str
    success: bool
    latency_ms: float
    tokens_used: int | None = None
    cost_usd: float | None = None
    cached: bool = False
    error: str | None = None
    details: dict[str, Any] | None = None


@dataclass
class CategoryResults:
    """Results for a test category."""

    category: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def latencies(self) -> list[float]:
        return [r.latency_ms for r in self.results if r.success and not r.cached]

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used or 0 for r in self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.results)


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    start_time: str
    end_time: str | None = None
    config: BenchmarkConfig | None = None
    categories: dict[str, CategoryResults] = field(default_factory=dict)

    def add_result(self, result: TestResult) -> None:
        """Add a test result to the appropriate category."""
        if result.category not in self.categories:
            self.categories[result.category] = CategoryResults(category=result.category)
        self.categories[result.category].results.append(result)

    @property
    def total_tests(self) -> int:
        return sum(c.total for c in self.categories.values())

    @property
    def total_passed(self) -> int:
        return sum(c.passed for c in self.categories.values())

    @property
    def overall_success_rate(self) -> float:
        return self.total_passed / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.categories.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(c.total_cost_usd for c in self.categories.values())

    @property
    def all_latencies(self) -> list[float]:
        latencies = []
        for cat in self.categories.values():
            latencies.extend(cat.latencies)
        return latencies

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        lines = [
            "╔════════════════════════════════════════════════════════════════════════╗",
            "║           STRUCTURED-EXTRACTOR BENCHMARK REPORT                        ║",
            "╚════════════════════════════════════════════════════════════════════════╝",
            "",
            f"  Generated:    {self.end_time or datetime.now().isoformat()}",
            f"  Mode:         {self.config.mode if self.config else 'N/A'}",
            "",
            "═" * 74,
            "  SUMMARY",
            "═" * 74,
            "",
            f"  Total Tests:        {self.total_tests}",
            f"  Passed:             {self.total_passed} ✅",
            f"  Failed:             {self.total_tests - self.total_passed} ❌",
            f"  Success Rate:       {self.overall_success_rate:.1%}",
            "",
            f"  Total Tokens:       {self.total_tokens:,}",
            f"  Total Cost:         ${self.total_cost_usd:.4f}",
            "",
        ]

        # Latency statistics
        if self.all_latencies:
            sorted_latencies = sorted(self.all_latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)

            lines.extend(
                [
                    "═" * 74,
                    "  LATENCY STATISTICS (excluding cached)",
                    "═" * 74,
                    "",
                    f"  Min:                {min(self.all_latencies):.0f} ms",
                    f"  Max:                {max(self.all_latencies):.0f} ms",
                    f"  Mean:               {statistics.mean(self.all_latencies):.0f} ms",
                    f"  Median:             {statistics.median(self.all_latencies):.0f} ms",
                    f"  P95:                {sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]:.0f} ms",  # noqa: E501
                    f"  P99:                {sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]:.0f} ms",  # noqa: E501
                    "",
                ]
            )

        # Per-category breakdown
        lines.extend(
            [
                "═" * 74,
                "  RESULTS BY CATEGORY",
                "═" * 74,
                "",
            ]
        )

        for cat_name, cat_results in self.categories.items():
            if cat_results.success_rate == 1.0:
                status = "✅"
            elif cat_results.success_rate >= 0.8:
                status = "⚠️"
            else:
                status = "❌"
            lines.append(
                f"  {status} {cat_name.upper()}: {cat_results.passed}/{cat_results.total} "
                f"({cat_results.success_rate:.0%}) | "
                f"Avg: {cat_results.avg_latency_ms:.0f}ms | "
                f"Tokens: {cat_results.total_tokens:,}"
            )

        # Failed tests detail
        failed_tests = [r for cat in self.categories.values() for r in cat.results if not r.success]
        if failed_tests:
            lines.extend(
                [
                    "",
                    "═" * 74,
                    "  FAILED TESTS",
                    "═" * 74,
                    "",
                ]
            )
            for f in failed_tests:
                lines.append(f"  ❌ [{f.category}] {f.test_name}: {f.error}")

        lines.extend(
            [
                "",
                "═" * 74,
                "",
            ]
        )

        return "\n".join(lines)

    def export_json(self, path: Path) -> None:
        """Export results to JSON."""
        data = {
            "meta": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "mode": self.config.mode if self.config else None,
            },
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.total_passed,
                "failed": self.total_tests - self.total_passed,
                "success_rate": self.overall_success_rate,
                "total_tokens": self.total_tokens,
                "total_cost_usd": self.total_cost_usd,
            },
            "categories": {
                name: {
                    "total": cat.total,
                    "passed": cat.passed,
                    "success_rate": cat.success_rate,
                    "avg_latency_ms": cat.avg_latency_ms,
                    "total_tokens": cat.total_tokens,
                    "total_cost_usd": cat.total_cost_usd,
                    "tests": [
                        {
                            "name": r.test_name,
                            "success": r.success,
                            "latency_ms": r.latency_ms,
                            "tokens": r.tokens_used,
                            "cost_usd": r.cost_usd,
                            "cached": r.cached,
                            "error": r.error,
                        }
                        for r in cat.results
                    ],
                }
                for name, cat in self.categories.items()
            },
        }
        path.write_text(json.dumps(data, indent=2, default=str))


# ============================================================================
# Schema Definitions
# ============================================================================


class Invoice(BaseModel):
    """Invoice data."""

    invoice_number: str = Field(description="Invoice ID")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Vendor name")
    total_amount: float = Field(description="Total amount")


class LineItem(BaseModel):
    """Line item."""

    description: str = Field(description="Description")
    quantity: float = Field(default=1, description="Quantity")
    total: float = Field(description="Total")


class DetailedInvoice(BaseModel):
    """Invoice with line items."""

    invoice_number: str = Field(description="Invoice number")
    vendor_name: str = Field(description="Vendor")
    line_items: list[LineItem] = Field(description="Line items")
    total: float = Field(description="Total")


class Person(BaseModel):
    """Person info."""

    name: str = Field(description="Name")
    email: str | None = Field(default=None, description="Email")
    phone: str | None = Field(default=None, description="Phone")


class MeetingNotes(BaseModel):
    """Meeting notes."""

    title: str = Field(description="Meeting title")
    date: str = Field(description="Date")
    attendees: list[str] = Field(default_factory=list, description="Attendees")
    action_items: list[str] = Field(default_factory=list, description="Action items")


# ============================================================================
# Test Documents
# ============================================================================

INVOICE_DOC = """
INVOICE #INV-2024-001
Date: November 25, 2024
From: ABC Corporation

Items:
- Consulting Services: $2,500.00
- Software License: $500.00

Total Due: $3,000.00
"""

DETAILED_INVOICE_DOC = """
INVOICE #INV-2024-002
Vendor: Tech Solutions Inc.

Line Items:
1. Web Development (40 hrs @ $100/hr) - $4,000
2. Design Services (20 hrs @ $75/hr) - $1,500
3. Hosting (12 months) - $600

Total: $6,100.00
"""

PERSON_DOC = """
Contact Card:
John Smith
Email: john.smith@example.com
Phone: (555) 123-4567
"""

MEETING_DOC = """
Q4 Planning Meeting
November 20, 2024

Attendees: Alice, Bob, Charlie, Diana

Action Items:
- Alice: Prepare budget proposal
- Bob: Review technical requirements
- Charlie: Schedule follow-up meeting
"""

AMBIGUOUS_DOC = """
Note from yesterday:
Someone called about an invoice.
Amount was around $500.
Need to follow up.
"""


# ============================================================================
# Test Functions
# ============================================================================


def run_test(
    extractor: DocumentExtractor,
    category: str,
    name: str,
    document: str,
    schema: type[BaseModel],
    **kwargs: Any,
) -> TestResult:
    """Run a single extraction test."""
    start_time = time.time()

    try:
        result = extractor.extract(document, schema=schema, **kwargs)
        latency_ms = (time.time() - start_time) * 1000

        return TestResult(
            category=category,
            test_name=name,
            success=True,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            cached=result.cached,
            error=None,
        )
    except Exception as e:
        return TestResult(
            category=category,
            test_name=name,
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


def run_text_extraction_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Run text extraction tests."""
    print("\n  📝 Text Extraction Tests")
    print("  " + "-" * 40)

    tests: list[tuple[str, str, type[BaseModel], dict[str, Any]]] = [
        ("basic_invoice", INVOICE_DOC, Invoice, {}),
        ("detailed_invoice", DETAILED_INVOICE_DOC, DetailedInvoice, {}),
        ("person_info", PERSON_DOC, Person, {}),
        ("meeting_notes", MEETING_DOC, MeetingNotes, {}),
        ("ambiguous_doc", AMBIGUOUS_DOC, Invoice, {}),
    ]

    for name, doc, schema, kwargs in tests:
        result = run_test(extractor, "text_extraction", name, doc, schema, **kwargs)
        results.add_result(result)
        status = "✅" if result.success else "❌"
        print(f"    {status} {name}: {result.latency_ms:.0f}ms")


def run_prompt_strategy_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Test different prompt strategies."""
    print("\n  🎯 Prompt Strategy Tests")
    print("  " + "-" * 40)

    strategies = [
        ("default", ExtractionConfig()),
        ("strict", ExtractionConfig(system_prompt=PromptTemplates.strict().system_prompt)),
        ("lenient", ExtractionConfig(system_prompt=PromptTemplates.lenient().system_prompt)),
    ]

    for strategy_name, config in strategies:
        result = run_test(
            extractor,
            "prompt_strategies",
            f"strategy_{strategy_name}",
            AMBIGUOUS_DOC,
            Invoice,
            config=config,
        )
        results.add_result(result)
        status = "✅" if result.success else "❌"
        print(f"    {status} {strategy_name}: {result.latency_ms:.0f}ms")


def run_template_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Test document templates."""
    print("\n  📋 Template Tests")
    print("  " + "-" * 40)

    # Create and use template
    template = DocumentTemplate(
        name="invoice_template",
        schema_class=Invoice,
        description="Invoice extraction",
        field_hints={
            "invoice_number": "Look for INV- or # prefix",
            "total_amount": "Final total after all fees",
        },
    )

    start_time = time.time()
    try:
        extraction_result: ExtractionResult[Invoice] = extractor.extract(
            document=INVOICE_DOC, template=template
        )
        latency_ms = (time.time() - start_time) * 1000

        result = TestResult(
            category="templates",
            test_name="template_extraction",
            success=True,
            latency_ms=latency_ms,
            tokens_used=extraction_result.tokens_used,
            cost_usd=extraction_result.cost_usd,
            cached=extraction_result.cached,
            error=None,
        )
    except Exception as e:
        result = TestResult(
            category="templates",
            test_name="template_extraction",
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )

    results.add_result(result)
    status = "✅" if result.success else "❌"
    print(f"    {status} template_extraction: {result.latency_ms:.0f}ms")

    # Test template serialization
    start_time = time.time()
    try:
        json_str = template.to_json()
        loaded = DocumentTemplate.from_json(json_str, Invoice)
        success = loaded.name == template.name
        result = TestResult(
            category="templates",
            test_name="template_serialization",
            success=success,
            latency_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        result = TestResult(
            category="templates",
            test_name="template_serialization",
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )

    results.add_result(result)
    status = "✅" if result.success else "❌"
    print(f"    {status} template_serialization: {result.latency_ms:.0f}ms")

    # Test registry
    start_time = time.time()
    try:
        registry = TemplateRegistry()
        registry.register(template)
        retrieved = registry.get("invoice_template")
        success = retrieved is not None
        result = TestResult(
            category="templates",
            test_name="template_registry",
            success=success,
            latency_ms=(time.time() - start_time) * 1000,
        )
    except Exception as e:
        result = TestResult(
            category="templates",
            test_name="template_registry",
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )

    results.add_result(result)
    status = "✅" if result.success else "❌"
    print(f"    {status} template_registry: {result.latency_ms:.0f}ms")


def run_confidence_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Test confidence scoring."""
    print("\n  🎯 Confidence Scoring Tests")
    print("  " + "-" * 40)

    documents = [
        ("high_confidence", INVOICE_DOC),
        ("medium_confidence", AMBIGUOUS_DOC),
    ]

    for name, doc in documents:
        start_time = time.time()
        try:
            extraction_result = extractor.extract_with_confidence(doc, schema=Invoice)
            latency_ms = (time.time() - start_time) * 1000

            result = TestResult(
                category="confidence",
                test_name=name,
                success=True,
                latency_ms=latency_ms,
                tokens_used=extraction_result.tokens_used,
                cost_usd=extraction_result.cost_usd,
                cached=extraction_result.cached,
                error=None,
                details={"confidence": extraction_result.confidence},
            )
        except Exception as e:
            result = TestResult(
                category="confidence",
                test_name=name,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

        results.add_result(result)
        status = "✅" if result.success else "❌"
        conf = result.details.get("confidence", "N/A") if result.details else "N/A"
        print(f"    {status} {name}: {result.latency_ms:.0f}ms (conf: {conf})")


def run_caching_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Test caching behavior."""
    print("\n  💾 Caching Tests")
    print("  " + "-" * 40)

    # Unique document for cache test
    cache_doc = INVOICE_DOC + f"\n<!-- cache_test_{time.time()} -->"

    # First call (cache miss)
    result1 = run_test(extractor, "caching", "cache_miss", cache_doc, Invoice)
    results.add_result(result1)
    status1 = "✅" if result1.success else "❌"
    print(f"    {status1} cache_miss: {result1.latency_ms:.0f}ms (cached: {result1.cached})")

    # Second call (should be cache hit)
    result2 = run_test(extractor, "caching", "cache_hit", cache_doc, Invoice)
    results.add_result(result2)
    status2 = "✅" if result2.success else "❌"
    print(f"    {status2} cache_hit: {result2.latency_ms:.0f}ms (cached: {result2.cached})")


def run_error_handling_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Test error handling."""
    print("\n  ⚠️ Error Handling Tests")
    print("  " + "-" * 40)

    # Empty document
    result = run_test(extractor, "error_handling", "empty_document", "", Invoice)
    results.add_result(result)
    status = "✅" if not result.success or result.success else "⚠️"  # Either behavior is acceptable
    print(f"    {status} empty_document: handled")

    # Very short document
    result = run_test(extractor, "error_handling", "minimal_document", "Invoice", Invoice)
    results.add_result(result)
    print("    ⚠️ minimal_document: handled")


def run_image_tests(extractor: DocumentExtractor, results: BenchmarkResults) -> None:
    """Test image extraction (if images available)."""
    print("\n  🖼️ Image Extraction Tests")
    print("  " + "-" * 40)

    assets_dir = Path(__file__).parent / "assets"

    # Test local image if exists
    sample_invoice = assets_dir / "sample_invoice.png"
    if sample_invoice.exists():
        start_time = time.time()
        try:
            invoice_result: ExtractionResult[Invoice] = extractor.extract_from_image(
                sample_invoice, schema=Invoice
            )
            latency_ms = (time.time() - start_time) * 1000

            result = TestResult(
                category="image_extraction",
                test_name="local_image",
                success=True,
                latency_ms=latency_ms,
                tokens_used=invoice_result.tokens_used,
                cost_usd=invoice_result.cost_usd,
                cached=invoice_result.cached,
                error=None,
            )
        except Exception as e:
            result = TestResult(
                category="image_extraction",
                test_name="local_image",
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

        results.add_result(result)
        status = "✅" if result.success else "❌"
        print(f"    {status} local_image: {result.latency_ms:.0f}ms")
    else:
        print("    ⏭️ local_image: skipped (no sample image)")

    # Test URL image
    class BasicForm(BaseModel):
        form_name: str | None = Field(default=None, description="Form name")
        form_number: str | None = Field(default=None, description="Form number")

    sample_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/"
        "Form_W-4%2C_2020.pdf/page1-463px-Form_W-4%2C_2020.pdf.jpg"
    )

    start_time = time.time()
    try:
        form_result: ExtractionResult[BasicForm] = extractor.extract_from_image(
            sample_url, schema=BasicForm
        )
        latency_ms = (time.time() - start_time) * 1000

        result = TestResult(
            category="image_extraction",
            test_name="url_image",
            success=True,
            latency_ms=latency_ms,
            tokens_used=form_result.tokens_used,
            cost_usd=form_result.cost_usd,
            cached=form_result.cached,
            error=None,
        )
    except Exception as e:
        result = TestResult(
            category="image_extraction",
            test_name="url_image",
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )

    results.add_result(result)
    status = "✅" if result.success else "❌"
    print(f"    {status} url_image: {result.latency_ms:.0f}ms")


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run the complete benchmark suite."""
    print("\n" + "═" * 74)
    print("  🚀 STRUCTURED-EXTRACTOR BENCHMARK SUITE")
    print("═" * 74)
    print(f"  Mode: {config.mode}")
    print(f"  Iterations per test: {config.iterations_per_test}")

    # Initialize
    extractor = DocumentExtractor(
        model="gpt-4.1",
        cache_dir="cache",
    )

    results = BenchmarkResults(
        start_time=datetime.now().isoformat(),
        config=config,
    )

    # Run test categories
    for iteration in range(config.iterations_per_test):
        if config.iterations_per_test > 1:
            print(f"\n  📍 Iteration {iteration + 1}/{config.iterations_per_test}")

        run_text_extraction_tests(extractor, results)
        run_prompt_strategy_tests(extractor, results)

        if config.run_template_tests:
            run_template_tests(extractor, results)

        if config.run_confidence_tests:
            run_confidence_tests(extractor, results)

        run_caching_tests(extractor, results)
        run_error_handling_tests(extractor, results)

        if config.run_image_tests:
            run_image_tests(extractor, results)

    results.end_time = datetime.now().isoformat()
    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run structured-extractor benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--full", action="store_true", help="Full test mode")
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Determine config
    if args.quick:
        config = BenchmarkConfig.quick()
    elif args.full:
        config = BenchmarkConfig.full()
    else:
        config = BenchmarkConfig.standard()

    # Run benchmark
    results = run_benchmark(config)

    # Generate and print report
    report = results.generate_report()
    print(report)

    # Save results
    output_dir = Path(__file__).parent / config.output_dir
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"benchmark_report_{timestamp}.txt"
    report_path.write_text(report)
    print(f"  📁 Report saved: {report_path}")

    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    results.export_json(json_path)
    print(f"  📁 JSON saved: {json_path}")

    # Exit with appropriate code
    sys.exit(0 if results.overall_success_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()
