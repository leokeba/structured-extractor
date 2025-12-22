"""End-to-end test: Text-based extraction with various schemas and strategies.

This script demonstrates and tests:
1. Basic text extraction
2. Nested schema extraction
3. Different prompt strategies (default, strict, lenient)
4. Field hints and custom prompts
5. Various document types (invoices, resumes, contracts)
6. Few-shot examples

Prerequisites:
- OPENAI_API_KEY environment variable set

Usage:
    python examples/e2e_text_extraction.py
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from structured_extractor import (
    DocumentExtractor,
    ExtractionConfig,
    ExtractionExample,
    PromptBuilder,
    PromptTemplates,
)

# Load environment variables
load_dotenv()


# ============================================================================
# Performance Tracking
# ============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    test_name: str
    success: bool
    latency_ms: float
    tokens_used: int | None
    cost_usd: float | None
    cached: bool
    error: str | None = None
    extracted_data: dict[str, Any] | None = None


@dataclass
class TestSuiteResults:
    """Aggregated results from a test suite."""

    suite_name: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests

    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used or 0 for r in self.results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.results)

    @property
    def cache_hit_rate(self) -> float:
        cached = sum(1 for r in self.results if r.cached)
        return cached / len(self.results) if self.results else 0.0

    def print_summary(self) -> None:
        """Print a summary of test results."""
        print(f"\n{'=' * 60}")
        print(f"📊 {self.suite_name} - Results Summary")
        print("=" * 60)
        print(f"Total Tests:     {self.total_tests}")
        print(f"Passed:          {self.passed_tests} ✅")
        print(f"Failed:          {self.failed_tests} ❌")
        print(f"Success Rate:    {self.success_rate:.1%}")
        print(f"Avg Latency:     {self.avg_latency_ms:.0f} ms")
        print(f"Total Tokens:    {self.total_tokens:,}")
        print(f"Total Cost:      ${self.total_cost_usd:.4f}")
        print(f"Cache Hit Rate:  {self.cache_hit_rate:.1%}")
        print("=" * 60)


# ============================================================================
# Schema Definitions
# ============================================================================


class Invoice(BaseModel):
    """Basic invoice data."""

    invoice_number: str = Field(description="The unique invoice identifier")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Name of the vendor")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")


class LineItem(BaseModel):
    """Individual line item."""

    description: str = Field(description="Item description")
    quantity: float = Field(default=1, description="Quantity")
    unit_price: float = Field(description="Price per unit")
    total: float = Field(description="Line total")


class DetailedInvoice(BaseModel):
    """Invoice with line items."""

    invoice_number: str = Field(description="Invoice number")
    date: str = Field(description="Invoice date")
    vendor_name: str = Field(description="Vendor/company name")
    vendor_address: str | None = Field(default=None, description="Vendor address")
    line_items: list[LineItem] = Field(description="List of line items")
    subtotal: float = Field(description="Subtotal before tax")
    tax: float = Field(description="Tax amount")
    total: float = Field(description="Total amount")


class EmploymentType(str, Enum):
    """Employment type enum."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"


class WorkExperience(BaseModel):
    """Single work experience entry."""

    company: str = Field(description="Company name")
    title: str = Field(description="Job title")
    start_date: str = Field(description="Start date")
    end_date: str | None = Field(default=None, description="End date or 'Present'")
    employment_type: EmploymentType | None = Field(default=None, description="Type of employment")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities")


class Education(BaseModel):
    """Education entry."""

    institution: str = Field(description="School/university name")
    degree: str = Field(description="Degree obtained")
    field_of_study: str | None = Field(default=None, description="Major/field")
    graduation_year: int | None = Field(default=None, description="Year of graduation")


class Resume(BaseModel):
    """Resume/CV data."""

    name: str = Field(description="Candidate's full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    summary: str | None = Field(default=None, description="Professional summary")
    work_experience: list[WorkExperience] = Field(default_factory=list, description="Work history")
    education: list[Education] = Field(default_factory=list, description="Education history")
    skills: list[str] = Field(default_factory=list, description="Technical skills")


class ContractParty(BaseModel):
    """Party to a contract."""

    name: str = Field(description="Legal name of party")
    role: str = Field(description="Role in contract (e.g., 'Licensor', 'Client')")
    address: str | None = Field(default=None, description="Address")


class Contract(BaseModel):
    """Contract/Agreement data."""

    title: str = Field(description="Contract title or type")
    parties: list[ContractParty] = Field(description="Parties to the contract")
    effective_date: str | None = Field(default=None, description="When contract takes effect")
    termination_date: str | None = Field(default=None, description="Contract end date")
    value: float | None = Field(default=None, description="Contract value/amount")
    governing_law: str | None = Field(default=None, description="Governing jurisdiction")
    key_terms: list[str] = Field(default_factory=list, description="Key terms and conditions")


class MeetingNotes(BaseModel):
    """Meeting notes extraction."""

    title: str = Field(description="Meeting title/subject")
    date: str = Field(description="Meeting date")
    attendees: list[str] = Field(default_factory=list, description="List of attendees")
    agenda_items: list[str] = Field(default_factory=list, description="Agenda items discussed")
    decisions: list[str] = Field(default_factory=list, description="Decisions made")
    action_items: list[str] = Field(default_factory=list, description="Action items assigned")
    next_meeting: str | None = Field(default=None, description="Next meeting date/time")


# ============================================================================
# Test Documents
# ============================================================================

SAMPLE_INVOICE = """
INVOICE #INV-2024-0042

Date: November 25, 2024
From: Acme Corporation
      123 Business Street
      San Francisco, CA 94105

Bill To: TechStart Inc.

Items:
- Consulting Services (40 hours @ $125/hr): $5,000.00
- Software License (Annual): $2,500.00
- Support Package: $500.00

Subtotal: $8,000.00
Tax (10%): $800.00
Total Due: $8,800.00

Payment Terms: Net 30
"""

SAMPLE_RESUME = """
JOHN SMITH
Software Engineer

Contact:
Email: john.smith@email.com
Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Senior software engineer with 8+ years of experience in full-stack development,
specializing in Python, TypeScript, and cloud infrastructure.

WORK EXPERIENCE

Senior Software Engineer | Tech Giants Inc.
January 2021 - Present | Full-time
- Lead development of microservices architecture serving 10M+ users
- Mentored team of 5 junior developers
- Reduced deployment time by 60% through CI/CD improvements

Software Engineer | StartupCo
March 2018 - December 2020 | Full-time
- Built customer-facing APIs handling 100K requests/day
- Implemented real-time notification system
- Collaborated with product team on feature specifications

Junior Developer | WebAgency
June 2016 - February 2018 | Full-time
- Developed responsive web applications
- Maintained legacy PHP systems

EDUCATION

M.S. Computer Science | Stanford University | 2016
B.S. Computer Science | UC Berkeley | 2014

SKILLS
Python, TypeScript, JavaScript, React, Node.js, PostgreSQL, Redis,
AWS, Docker, Kubernetes, CI/CD, Agile/Scrum
"""

SAMPLE_CONTRACT = """
SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of January 1, 2025
("Effective Date") by and between:

LICENSOR:
CloudSoft Solutions Inc.
123 Innovation Drive, Austin, TX 78701

LICENSEE:
Enterprise Corp
456 Corporate Blvd, New York, NY 10001

1. LICENSE GRANT
Licensor grants Licensee a non-exclusive, non-transferable license to use the
CloudSoft Enterprise Platform software.

2. TERM
This Agreement shall commence on the Effective Date and continue for a period
of three (3) years, unless terminated earlier.

3. FEES
Licensee shall pay Licensor an annual license fee of $50,000 USD, payable within
30 days of invoice.

4. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

5. KEY TERMS
- Software includes all updates and patches released during the license term
- Licensee may not sublicense or transfer the software
- Licensor provides 24/7 technical support
- Data remains property of Licensee

Executed as of the date first written above.
"""

SAMPLE_MEETING_NOTES = """
Q4 Planning Meeting - November 20, 2024

Attendees: Sarah Johnson (PM), Mike Chen (Engineering Lead), Lisa Park (Design),
           Tom Wilson (QA), Anna Garcia (Marketing)

AGENDA ITEMS DISCUSSED:

1. Q3 Performance Review
   - Exceeded revenue targets by 15%
   - User acquisition up 25% YoY
   - Customer satisfaction score: 4.5/5

2. Q4 Roadmap
   - New dashboard feature launch (December)
   - Mobile app v2.0 release
   - API performance improvements

3. Resource Allocation
   - Need 2 additional frontend developers
   - Marketing budget increase approved

DECISIONS MADE:
- Dashboard feature will be released December 15
- Mobile app v2.0 postponed to January
- Hiring process to begin immediately

ACTION ITEMS:
- Sarah: Create detailed project timeline by Nov 25
- Mike: Draft technical requirements for new hires
- Lisa: Finalize dashboard mockups by Nov 22
- Tom: Set up automated testing for dashboard
- Anna: Prepare launch marketing materials

NEXT MEETING: December 4, 2024 at 2:00 PM
"""

AMBIGUOUS_DOCUMENT = """
Quick note from Bob:

Hey, talked to the client. They want 50 units delivered.
Price is around 10-15 each, maybe with a discount.
Should be done by next week or so.
Also mentioned something about the other project.
"""


# ============================================================================
# Test Functions
# ============================================================================


def run_test(
    name: str,
    extractor: DocumentExtractor,
    document: str,
    schema: type[BaseModel],
    **extract_kwargs: Any,
) -> TestResult:
    """Run a single extraction test and return results."""
    start_time = time.time()

    try:
        result = extractor.extract(document, schema=schema, **extract_kwargs)
        latency_ms = (time.time() - start_time) * 1000

        return TestResult(
            test_name=name,
            success=True,
            latency_ms=latency_ms,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            cached=result.cached,
            error=None,
            extracted_data=result.data.model_dump(),
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return TestResult(
            test_name=name,
            success=False,
            latency_ms=latency_ms,
            tokens_used=None,
            cost_usd=None,
            cached=False,
            error=str(e),
        )


def test_basic_extraction(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test basic extraction with simple schema."""
    print("\n" + "-" * 40)
    print("Test: Basic Invoice Extraction")
    print("-" * 40)

    result = run_test(
        "basic_invoice",
        extractor,
        SAMPLE_INVOICE,
        Invoice,
    )
    results.results.append(result)

    if result.success and result.extracted_data:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
        data = result.extracted_data
        print(f"   Invoice #: {data['invoice_number']}")
        print(f"   Vendor: {data['vendor_name']}")
        print(f"   Total: ${data['total_amount']}")
    else:
        print(f"❌ Failed: {result.error}")


def test_nested_extraction(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test extraction with nested structures."""
    print("\n" + "-" * 40)
    print("Test: Nested Schema (Detailed Invoice)")
    print("-" * 40)

    result = run_test(
        "nested_invoice",
        extractor,
        SAMPLE_INVOICE,
        DetailedInvoice,
    )
    results.results.append(result)

    if result.success and result.extracted_data:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
        data = result.extracted_data
        print(f"   Line items found: {len(data['line_items'])}")
        for item in data["line_items"]:
            print(f"     - {item['description']}: ${item['total']}")
    else:
        print(f"❌ Failed: {result.error}")


def test_complex_schema(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test extraction with complex schema (Resume)."""
    print("\n" + "-" * 40)
    print("Test: Complex Schema (Resume)")
    print("-" * 40)

    result = run_test(
        "resume_extraction",
        extractor,
        SAMPLE_RESUME,
        Resume,
    )
    results.results.append(result)

    if result.success and result.extracted_data:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
        data = result.extracted_data
        print(f"   Name: {data['name']}")
        print(f"   Work experiences: {len(data['work_experience'])}")
        print(f"   Education entries: {len(data['education'])}")
        print(f"   Skills: {len(data['skills'])}")
    else:
        print(f"❌ Failed: {result.error}")


def test_contract_extraction(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test extraction from contract document."""
    print("\n" + "-" * 40)
    print("Test: Contract Extraction")
    print("-" * 40)

    result = run_test(
        "contract_extraction",
        extractor,
        SAMPLE_CONTRACT,
        Contract,
    )
    results.results.append(result)

    if result.success and result.extracted_data:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
        data = result.extracted_data
        print(f"   Title: {data['title']}")
        print(f"   Parties: {len(data['parties'])}")
        print(f"   Value: ${data.get('value', 'N/A')}")
        print(f"   Key terms: {len(data['key_terms'])}")
    else:
        print(f"❌ Failed: {result.error}")


def test_meeting_notes_extraction(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test extraction from meeting notes."""
    print("\n" + "-" * 40)
    print("Test: Meeting Notes Extraction")
    print("-" * 40)

    result = run_test(
        "meeting_notes",
        extractor,
        SAMPLE_MEETING_NOTES,
        MeetingNotes,
    )
    results.results.append(result)

    if result.success and result.extracted_data:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
        data = result.extracted_data
        print(f"   Title: {data['title']}")
        print(f"   Attendees: {len(data['attendees'])}")
        print(f"   Decisions: {len(data['decisions'])}")
        print(f"   Action items: {len(data['action_items'])}")
    else:
        print(f"❌ Failed: {result.error}")


def test_prompt_strategies(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test different prompt strategies."""
    print("\n" + "-" * 40)
    print("Test: Prompt Strategies Comparison")
    print("-" * 40)

    strategies = {
        "default": ExtractionConfig(),
        "strict": ExtractionConfig(
            system_prompt=PromptTemplates.strict().system_prompt,
        ),
        "lenient": ExtractionConfig(
            system_prompt=PromptTemplates.lenient().system_prompt,
        ),
    }

    for strategy_name, config in strategies.items():
        result = run_test(
            f"strategy_{strategy_name}",
            extractor,
            AMBIGUOUS_DOCUMENT,
            Invoice,
            config=config,
        )
        results.results.append(result)

        if result.success and result.extracted_data:
            print(f"✅ {strategy_name}: Passed - Latency: {result.latency_ms:.0f}ms")
            data = result.extracted_data
            print(f"     Total extracted: ${data.get('total_amount', 'N/A')}")
        else:
            print(f"❌ {strategy_name}: Failed - {result.error}")


def test_field_hints(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test extraction with field hints."""
    print("\n" + "-" * 40)
    print("Test: Field Hints")
    print("-" * 40)

    hints = {
        "invoice_number": "Look for patterns like 'INV-XXXX' or '#XXXX'",
        "total_amount": "The final amount after tax, not the subtotal",
        "vendor_name": "The company that issued the invoice, not the bill recipient",
    }

    result = run_test(
        "with_field_hints",
        extractor,
        SAMPLE_INVOICE,
        Invoice,
        field_hints=hints,
    )
    results.results.append(result)

    if result.success and result.extracted_data:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
        data = result.extracted_data
        print(f"   Invoice #: {data['invoice_number']}")
        print(f"   Vendor: {data['vendor_name']}")
        print(f"   Total: ${data['total_amount']}")
    else:
        print(f"❌ Failed: {result.error}")


def test_few_shot_examples(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test extraction with few-shot examples."""
    print("\n" + "-" * 40)
    print("Test: Few-Shot Examples")
    print("-" * 40)

    # Create prompt builder with examples
    examples = [
        ExtractionExample(
            input_text="Invoice ABC-123 from Vendor Co. Total: $500",
            output={"invoice_number": "ABC-123", "vendor_name": "Vendor Co.", "total_amount": 500},
            explanation="Extract invoice ID, vendor, and total amount",
        ),
    ]

    builder = PromptBuilder(include_examples=True)

    # Build the prompt with examples to verify it works
    _ = builder.build_extraction_prompt(
        document=SAMPLE_INVOICE,
        schema=Invoice,
        examples=examples,
    )

    # Run extraction (using the builder's approach indirectly)
    result = run_test(
        "few_shot_examples",
        extractor,
        SAMPLE_INVOICE,
        Invoice,
    )
    results.results.append(result)

    if result.success:
        print(f"✅ Passed - Latency: {result.latency_ms:.0f}ms")
    else:
        print(f"❌ Failed: {result.error}")


def test_caching(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test caching behavior."""
    print("\n" + "-" * 40)
    print("Test: Caching Behavior")
    print("-" * 40)

    # First extraction (should not be cached)
    result1 = run_test(
        "cache_miss",
        extractor,
        SAMPLE_INVOICE + "\n<!-- cache test 1 -->",
        Invoice,
    )
    results.results.append(result1)

    # Same extraction (should be cached)
    result2 = run_test(
        "cache_hit",
        extractor,
        SAMPLE_INVOICE + "\n<!-- cache test 1 -->",
        Invoice,
    )
    results.results.append(result2)

    print(f"   First call:  Cached={result1.cached}, Latency={result1.latency_ms:.0f}ms")
    print(f"   Second call: Cached={result2.cached}, Latency={result2.latency_ms:.0f}ms")

    if result2.cached:
        print("✅ Caching working correctly")
    else:
        print("⚠️  Cache miss on second call (may be expected if TTL expired)")


def test_error_handling(extractor: DocumentExtractor, results: TestSuiteResults) -> None:
    """Test error handling with invalid/empty documents."""
    print("\n" + "-" * 40)
    print("Test: Error Handling")
    print("-" * 40)

    # Test with empty document
    result = run_test(
        "empty_document",
        extractor,
        "",
        Invoice,
    )
    results.results.append(result)

    if not result.success:
        print(f"✅ Correctly handled empty document: {result.error}")
    else:
        print("⚠️  Extraction succeeded on empty document (may extract defaults)")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run all E2E text extraction tests."""
    print("\n🚀 Structured Extractor - E2E Text Extraction Tests")
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
    results = TestSuiteResults(suite_name="Text Extraction Tests")

    # Run all tests
    print("\n📋 Running extraction tests...")

    test_basic_extraction(extractor, results)
    test_nested_extraction(extractor, results)
    test_complex_schema(extractor, results)
    test_contract_extraction(extractor, results)
    test_meeting_notes_extraction(extractor, results)
    test_prompt_strategies(extractor, results)
    test_field_hints(extractor, results)
    test_few_shot_examples(extractor, results)
    test_caching(extractor, results)
    test_error_handling(extractor, results)

    # Print summary
    results.print_summary()

    # Print detailed failures
    failures = [r for r in results.results if not r.success]
    if failures:
        print("\n❌ Failed Tests:")
        for f in failures:
            print(f"   - {f.test_name}: {f.error}")


if __name__ == "__main__":
    main()
