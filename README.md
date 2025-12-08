# 📄 structured-extractor

A powerful LLM-driven structured data extractor for document parsing, built on top of [seeds-clients](https://github.com/leokeba/seeds-clients) and Pydantic.

## 🎯 Overview

`structured-extractor` provides a high-level, declarative API for extracting structured data from documents using Large Language Models. Define your data schema with Pydantic models, and let the library handle the complexity of prompt engineering, chunking, validation, and multi-field extraction.

### Key Features

- 🔧 **Declarative Schema Definition** - Use Pydantic models to define extraction templates
- 📝 **Built-in Templates** - Pre-configured templates for invoices, receipts, resumes, and contracts
- ✅ **Validation & Coercion** - Automatic type validation and data coercion via Pydantic
- 💾 **Smart Caching** - Leverage seeds-clients caching for repeated extractions
- 🌍 **Carbon Tracking** - Built-in environmental impact tracking from seeds-clients
- 📊 **Confidence Scores** - Field-level and overall confidence scoring
- 📈 **Quality Metrics** - Completeness ratios, warnings, and review flags
- 🎯 **Performance Evaluation** - Precision, recall, F1, accuracy metrics against ground truth
- 🔄 **Human-in-the-Loop** - Callback hooks for low-confidence review
- 🔌 **Extensible** - Template inheritance, custom prompts, and registries
- 🤖 **Multi-Provider Support** - Use OpenAI, Anthropic, Google, OpenRouter, or any seeds-clients provider

## 📦 Installation

```bash
# Using uv (recommended)
uv add structured-extractor

# Using pip
pip install structured-extractor
```

## 🚀 Quick Start

### Basic Extraction

```python
from pydantic import BaseModel, Field
from structured_extractor import DocumentExtractor

# Define your extraction schema
class Invoice(BaseModel):
    """Invoice data extraction template."""
    invoice_number: str = Field(description="The unique invoice identifier")
    date: str = Field(description="Invoice date in ISO format (YYYY-MM-DD)")
    vendor_name: str = Field(description="Name of the vendor/supplier")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")

# Initialize the extractor
extractor = DocumentExtractor(
    api_key="your-openai-key",  # or use OPENAI_API_KEY env var
    model="gpt-4.1",
)

# Extract data from a document
document_text = """
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

result = extractor.extract(document_text, schema=Invoice)

print(f"Invoice: {result.data.invoice_number}")
print(f"Vendor: {result.data.vendor_name}")
print(f"Total: {result.data.currency} {result.data.total_amount}")
# Output:
# Invoice: INV-2024-0042
# Vendor: Acme Corporation
# Total: USD 8250.0
```

### Using Different LLM Providers

`structured-extractor` supports any client from the [seeds-clients](https://github.com/leokeba/seeds-clients) library, giving you flexibility to use OpenAI, Anthropic, Google, OpenRouter, or other providers:

```python
from pydantic import BaseModel, Field
from structured_extractor import DocumentExtractor

class Invoice(BaseModel):
    invoice_number: str = Field(description="The invoice ID")
    total_amount: float = Field(description="Total amount due")

# Using Anthropic Claude
from seeds_clients import AnthropicClient

client = AnthropicClient(
    model="claude-sonnet-4-20250514",
    cache_dir="./cache",
)
extractor = DocumentExtractor(client=client)
result = extractor.extract(document_text, schema=Invoice)

# Using Google Gemini
from seeds_clients import GoogleClient

client = GoogleClient(
    model="gemini-2.5-flash",
    cache_dir="./cache",
)
extractor = DocumentExtractor(client=client)

# Using OpenRouter (access multiple providers)
from seeds_clients import OpenRouterClient

client = OpenRouterClient(
    model="anthropic/claude-3-5-sonnet",  # provider/model format
    cache_dir="./cache",
)
extractor = DocumentExtractor(client=client)
```

When you provide a `client` parameter, the `api_key`, `model`, `cache_dir`, and `cache_ttl_hours` parameters are ignored since the client is already configured.

### Nested Structures

```python
from pydantic import BaseModel, Field

class LineItem(BaseModel):
    """Individual line item in an invoice."""
    description: str
    quantity: int = Field(default=1)
    unit_price: float
    total: float

class Address(BaseModel):
    """Postal address."""
    street: str
    city: str
    state: str | None = None
    postal_code: str
    country: str = Field(default="USA")

class DetailedInvoice(BaseModel):
    """Detailed invoice with nested structures."""
    invoice_number: str
    date: str
    vendor: str
    vendor_address: Address
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float

result = extractor.extract(document_text, schema=DetailedInvoice)

for item in result.data.line_items:
    print(f"- {item.description}: ${item.total}")
```

### Batch Extraction

```python
from structured_extractor import BatchExtractor

batch_extractor = BatchExtractor(
    api_key="your-openai-key",
    model="gpt-4.1",
    max_concurrent=5,  # Parallel extraction
)

documents = [
    "Invoice #001...",
    "Invoice #002...",
    "Invoice #003...",
]

results = batch_extractor.extract_many(documents, schema=Invoice)

for result in results:
    if result.success:
        print(f"Extracted: {result.data.invoice_number}")
    else:
        print(f"Failed: {result.error}")
```

### Document Templates

```python
from structured_extractor import DocumentTemplate, FieldConfig, InvoiceSchema

# Define a reusable template
invoice_template = DocumentTemplate(
    name="invoice",
    schema_class=InvoiceSchema,
    system_prompt="You are an expert invoice data extractor. Extract data precisely.",
    field_hints={
        "date": "Look for date formats like MM/DD/YYYY or YYYY-MM-DD",
        "total_amount": "The final total after taxes and discounts",
    },
)

result = extractor.extract(document_text, template=invoice_template)
```

### Built-in Templates

```python
from structured_extractor import BuiltinTemplates, DocumentExtractor

# Use pre-configured templates for common document types
extractor = DocumentExtractor(model="gpt-4.1")

# Invoice extraction with optimized prompts
invoice_template = BuiltinTemplates.invoice()
result = extractor.extract(invoice_text, template=invoice_template)
print(result.data.total_amount)

# Also available: receipt, resume, contract
receipt_template = BuiltinTemplates.receipt()
resume_template = BuiltinTemplates.resume()
contract_template = BuiltinTemplates.contract()

# Create a registry with all built-in templates
registry = BuiltinTemplates.create_registry()
template = registry.get("builtin_invoice")
```

### Using Prompt Templates

```python
from structured_extractor import (
    DocumentExtractor,
    PromptBuilder,
    PromptTemplates,
    PromptStrategy,
    ExtractionExample,
)

# Use a built-in template for invoice extraction
builder = PromptBuilder(template=PromptTemplates.invoice())
extractor = DocumentExtractor(prompt_builder=builder)

# Or use strict mode for high-accuracy extraction
strict_strategy = PromptStrategy(strict_mode=True, include_reasoning=True)
builder = PromptBuilder(strategy=strict_strategy)

# Add few-shot examples for better results
examples = [
    ExtractionExample(
        input_text="Invoice #12345\nDate: 2024-01-15\nTotal: $500.00",
        output={"invoice_number": "12345", "date": "2024-01-15", "total": 500.0},
        explanation="Extracted invoice number, date, and total from header",
    )
]

result = extractor.extract(
    document_text,
    schema=Invoice,
    field_hints={"date": "Look for invoice date, not due date"},
    examples=examples,
)
```

### Extraction with Confidence Scores

```python
from structured_extractor import DocumentExtractor, ExtractionConfig

extractor = DocumentExtractor(
    api_key="your-openai-key",
    model="gpt-4.1",
)

# Use extract_with_confidence for confidence-scored extraction
result = extractor.extract_with_confidence(document_text, schema=Invoice)

print(f"Data: {result.data}")
print(f"Overall Confidence: {result.confidence}")
print(f"Field confidences: {result.field_confidences}")
print(f"Low confidence fields: {result.low_confidence_fields}")
# Output:
# Overall Confidence: 0.95
# Field confidences: {'invoice_number': 0.99, 'date': 0.95, 'vendor_name': 0.92, ...}
```

### Quality Metrics

```python
from structured_extractor import DocumentExtractor, ExtractionConfig

config = ExtractionConfig(
    compute_quality_metrics=True,
    confidence_threshold=0.8,
)

extractor = DocumentExtractor(model="gpt-4.1", default_config=config)
result = extractor.extract_with_confidence(document_text, schema=Invoice)

# Access detailed quality metrics
metrics = result.quality_metrics
print(f"Completeness: {metrics.completeness_ratio:.0%}")
print(f"Average confidence: {metrics.average_confidence:.2f}")
print(f"Needs review: {metrics.needs_review}")
print(f"Warnings: {metrics.quality_warnings}")
```

### Human-in-the-Loop Callbacks

```python
from structured_extractor import DocumentExtractor, ExtractionConfig, ExtractionResult

def review_low_confidence(result: ExtractionResult) -> ExtractionResult | None:
    """Called when extraction confidence is below threshold."""
    print(f"Low confidence extraction: {result.confidence}")
    # Return modified result, or None to keep original
    return None

def review_required(result: ExtractionResult) -> ExtractionResult | None:
    """Called when quality metrics indicate review is needed."""
    if result.quality_metrics and result.quality_metrics.needs_review:
        print(f"Review needed: {result.quality_metrics.quality_warnings}")
    return None

config = ExtractionConfig(
    compute_quality_metrics=True,
    on_low_confidence=review_low_confidence,
    on_review_required=review_required,
    review_confidence_threshold=0.5,
)

extractor = DocumentExtractor(model="gpt-4.1", default_config=config)
result = extractor.extract_with_confidence(document_text, schema=Invoice)
```

### Performance Evaluation

Evaluate extraction quality against ground truth data with precision, recall, F1, and accuracy metrics:

```python
from pydantic import BaseModel, Field
from structured_extractor import DocumentExtractor
from structured_extractor.evaluation import (
    ExtractionEvaluator,
    NumericComparator,
    StringComparator,
    EvaluationReporter,
)

class Invoice(BaseModel):
    invoice_number: str = Field(description="Invoice ID")
    total_amount: float = Field(description="Total amount")
    vendor_name: str = Field(description="Vendor name")

# Create evaluator with an extractor
extractor = DocumentExtractor(model="gpt-4.1")
evaluator = ExtractionEvaluator(
    extractor,
    default_string_threshold=0.85,   # 85% similarity for string match
    default_numeric_tolerance=0.02,  # 2% tolerance for numbers
)

# Evaluate a single extraction
result = evaluator.evaluate(
    document="Invoice #INV-001\nTotal: $1,250.00\nFrom: Acme Corp",
    schema=Invoice,
    ground_truth={
        "invoice_number": "INV-001",
        "total_amount": 1250.0,
        "vendor_name": "Acme Corp",
    },
)

# Binary metrics (for classification)
print(f"Precision: {result.precision:.2%}")
print(f"Recall: {result.recall:.2%}")
print(f"F1 Score: {result.f1_score:.2%}")
print(f"Accuracy: {result.accuracy:.2%}")

# Continuous score metrics (for similarity)
print(f"Mean Score: {result.mean_score:.2%}")

# Per-field breakdown
for match in result.field_matches:
    status = "✓" if match.matched else "✗"
    print(f"  {match.field_name}: {status} (score: {match.score:.2f})")
```

#### Custom Comparators

Use field-specific comparators with configurable thresholds:

```python
from structured_extractor.evaluation import (
    ExactComparator,      # Exact equality
    NumericComparator,    # Numeric with tolerance
    StringComparator,     # Levenshtein similarity
    ContainsComparator,   # Substring match
    DateComparator,       # Date format normalization
    ListComparator,       # List comparison
)

result = evaluator.evaluate(
    document=doc,
    schema=Invoice,
    ground_truth=ground_truth,
    field_comparators={
        "invoice_number": ExactComparator(),  # Must match exactly
        "total_amount": NumericComparator(threshold=0.05),  # 5% tolerance
        "vendor_name": StringComparator(threshold=0.8),  # 80% similarity
    },
)
```

#### Batch Evaluation and Reports

```python
# Evaluate multiple test cases
metrics = evaluator.compute_aggregate_metrics()

print(f"Micro F1: {metrics.micro_f1:.2%}")  # Global F1 across all fields
print(f"Macro F1: {metrics.macro_f1:.2%}")  # Average F1 per evaluation

# Generate reports in multiple formats
reporter = EvaluationReporter(evaluator.evaluations, metrics)
reporter.save("report.json")   # JSON format
reporter.save("report.md")     # Markdown format
reporter.print_summary()       # Console output
```

### Carbon Tracking & BoAmps Reporting

Track environmental impact and generate standardized energy consumption reports following the [BoAmps format](https://github.com/Boavizta/BoAmps):

```python
from pydantic import BaseModel, Field
from structured_extractor import DocumentExtractor

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float

extractor = DocumentExtractor(model="gpt-4.1")

# Process multiple documents
documents = ["Invoice #001...", "Invoice #002...", "Invoice #003..."]
for doc in documents:
    extractor.extract(doc, schema=Invoice)

# Check cumulative tracking stats
tracking = extractor.cumulative_tracking
print(f"Total requests: {tracking.total_request_count}")
print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")
print(f"Total cost: ${tracking.total_cost_usd:.4f}")
print(f"Carbon emissions: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
print(f"Emissions avoided (cache): {tracking.emissions_avoided_kgco2eq:.6f} kgCO2eq")

# Export BoAmps-compliant report
report = extractor.export_boamps_report(
    "extraction_report.json",
    publisher_name="My Company",
    project_name="Document Processing Pipeline",
    task_description="Invoice data extraction from PDF documents",
)

# Access report data programmatically
print(f"Total energy: {report.measures[0].powerConsumption} kWh")
print(f"Model: {report.task.algorithms[0].foundationModelName}")
```

For more control, use the `BoAmpsReporter` directly:

```python
from structured_extractor import BoAmpsReporter

reporter = BoAmpsReporter(
    client=extractor._client,  # Access underlying client
    publisher_name="Research Organization",
    publisher_division="AI Team",
    project_name="Sustainability Study",
    task_description="Document extraction benchmark",
    infrastructure_type="publicCloud",
    quality="high",
)

report = reporter.generate_report()
report.save("detailed_report.json")
```

### Large Document Handling

```python
from structured_extractor import DocumentExtractor, ChunkingStrategy

extractor = DocumentExtractor(
    api_key="your-openai-key",
    model="gpt-4.1",
    chunking_strategy=ChunkingStrategy.SEMANTIC,  # or FIXED, PARAGRAPH, PAGE
    chunk_size=4000,  # tokens per chunk
    chunk_overlap=200,  # overlap for context continuity
)

# Works automatically with large documents
large_document = open("large_report.txt").read()
result = extractor.extract(large_document, schema=ReportData)
```

## 🏗️ Architecture

```
structured_extractor/
├── __init__.py              # Public API exports
├── core/
│   ├── extractor.py         # Main DocumentExtractor class
│   ├── templates.py         # DocumentTemplate, TemplateRegistry
│   └── config.py            # ExtractionConfig, FieldConfig, callbacks
├── prompts/
│   └── builder.py           # PromptBuilder, PromptStrategy, PromptTemplates
├── results/
│   ├── types.py             # ExtractionResult, FieldResult
│   └── confidence.py        # Confidence scoring, ExtractionQualityMetrics
├── evaluation/              # Performance evaluation module
│   ├── types.py             # EvaluationMetrics, FieldMatch, ExtractionEvaluation
│   ├── comparators.py       # Field comparators (Exact, Numeric, String, etc.)
│   ├── evaluator.py         # ExtractionEvaluator class
│   └── reporters.py         # EvaluationReporter (JSON, Markdown, text)
├── schemas/                 # Built-in Pydantic schemas for common documents
│   ├── invoice.py           # InvoiceSchema, InvoiceLineItem
│   ├── receipt.py           # ReceiptSchema
│   ├── resume.py            # ResumeSchema, WorkExperience, Education
│   └── contract.py          # ContractSchema, ContractParty
└── templates/               # Pre-configured extraction templates
    └── builtins.py          # BuiltinTemplates factory class
```

## 📋 Implementation Roadmap

### Phase 1: Core Foundation ✅

**Goals**: Set up project structure and basic extraction

- [x] Project scaffolding with uv
- [x] Basic `DocumentExtractor` class
- [x] Integration with seeds-clients `OpenAIClient`
- [x] Simple schema-to-prompt conversion
- [x] Basic `ExtractionResult` type
- [x] Unit test infrastructure with pytest

**Deliverables**:
- ✅ Working basic extraction with Pydantic models
- ✅ Simple prompt generation from schema
- ✅ Basic test coverage (37 tests)

### Phase 2: Advanced Schema Support ✅

**Goals**: Support complex Pydantic schemas

- [x] Nested model support
- [x] List/array field extraction
- [x] Optional fields and defaults
- [x] Field descriptions in prompts
- [x] Union types support
- [x] Enum extraction
- [x] Literal types support
- [x] Field constraints (ge, le, min_length, max_length, pattern)

**Deliverables**:
- ✅ Full Pydantic model support
- ✅ Complex nested structure extraction
- ✅ Comprehensive validation (67 tests total)

### Phase 3: Prompt Engineering ✅

**Goals**: Optimize extraction quality

- [x] Dynamic prompt builder enhancements
- [x] Field-specific extraction hints
- [x] Few-shot examples support
- [x] System prompt customization
- [x] Prompt templates library (invoice, resume, contract, strict, lenient)
- [x] Schema-aware prompt optimization

**Deliverables**:
- ✅ `PromptStrategy` for configuring prompt behavior
- ✅ `PromptTemplate` for reusable extraction scenarios
- ✅ `PromptTemplates` with 6 built-in templates
- ✅ `ExtractionExample` for few-shot learning
- ✅ Reasoning mode for complex extractions
- ✅ 106 tests total

### Phase 4: Confidence & Quality ✅

**Goals**: Add confidence scoring and quality metrics

- [x] Confidence score generation via `extract_with_confidence()`
- [x] Field-level confidence scores
- [x] Low-confidence field flagging
- [x] Extraction quality metrics
- [x] Retry logic for errors
- [x] Human-in-the-loop hooks

**Deliverables**:
- ✅ `extract_with_confidence()` method for confidence-scored extractions
- ✅ `ConfidenceAssessment` and `FieldConfidence` types
- ✅ `low_confidence_fields` in `ExtractionResult`
- ✅ `ExtractionQualityMetrics` for detailed quality assessment
- ✅ `compute_quality_metrics()` function
- ✅ Human-in-the-loop callbacks: `on_low_confidence`, `on_review_required`, `on_validation_error`
- ✅ Retry mechanisms
- ✅ 175 tests total

### Phase 5: Templates & Presets ✅

**Goals**: Reusable extraction templates

- [x] `DocumentTemplate` class enhancements
- [x] Template serialization (YAML/JSON)
- [x] Built-in templates (invoice, receipt, contract, resume)
- [x] Template inheritance
- [x] Template validation
- [x] Template registry

**Deliverables**:
- ✅ `DocumentTemplate` with serialization to JSON/YAML
- ✅ `TemplateRegistry` for managing and discovering templates
- ✅ Template inheritance via `merge_with_parent()`
- ✅ `BuiltinTemplates` factory with invoice, receipt, resume, contract templates
- ✅ Built-in schemas: `InvoiceSchema`, `ReceiptSchema`, `ResumeSchema`, `ContractSchema`
- ✅ Template validation and field hint validation
- ✅ 216 tests total

### Phase 6: Batch Processing (Low Priority)

**Goals**: Efficient multi-document processing

- [ ] `BatchExtractor` class
- [ ] Async extraction with `aextract()` method
- [ ] Parallel extraction with asyncio
- [ ] Progress tracking via callbacks
- [ ] Error handling and partial results
- [ ] Rate limiting
- [ ] Cost estimation

**Deliverables**:
- Batch processing capability
- Async extraction support
- Production-ready error handling

### Phase 7: Document Chunking (Low Priority)

**Goals**: Handle large documents

- [ ] Token counting utilities
- [ ] Fixed-size chunking strategy
- [ ] Paragraph-aware chunking
- [ ] Semantic chunking (sentence boundaries)
- [ ] Chunk overlap management
- [ ] Result merging strategies
- [ ] Deduplication logic

**Deliverables**:
- Large document support (>100k tokens)
- Multiple chunking strategies
- Intelligent result merging

### Phase 8: Documentation & Polish

**Goals**: Production readiness

- [ ] Comprehensive documentation
- [ ] API reference
- [ ] Usage examples
- [ ] Performance benchmarks
- [ ] Integration tests
- [ ] PyPI publication

**Deliverables**:
- Complete documentation
- Published package
- Example notebooks

## 🔧 Configuration

### Environment Variables

```bash
# Required (when not using client injection)
OPENAI_API_KEY=your-api-key

# For other providers
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
OPENROUTER_API_KEY=your-openrouter-key

# Optional
STRUCTURED_EXTRACTOR_CACHE_DIR=./cache
STRUCTURED_EXTRACTOR_CACHE_TTL=24  # hours
STRUCTURED_EXTRACTOR_DEFAULT_MODEL=gpt-4.1
```

### Extractor Configuration

```python
from structured_extractor import DocumentExtractor, ExtractionConfig

# Option 1: Simple initialization (uses OpenAI)
extractor = DocumentExtractor(
    api_key="your-key",  # or use OPENAI_API_KEY env var
    model="gpt-4.1",
    cache_dir="./cache",
    cache_ttl_hours=24.0,
)

# Option 2: Inject a pre-configured client (any provider)
from seeds_clients import AnthropicClient

client = AnthropicClient(
    model="claude-sonnet-4-20250514",
    cache_dir="./cache",
    ttl_hours=24.0,
)
extractor = DocumentExtractor(client=client)

# Option 3: With custom extraction configuration
config = ExtractionConfig(
    temperature=0.0,  # Deterministic extraction
    max_retries=3,
    retry_on_validation_error=True,
    compute_quality_metrics=True,
    confidence_threshold=0.8,
)
extractor = DocumentExtractor(
    client=client,
    default_config=config,
)
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [seeds-clients](https://github.com/leokeba/seeds-clients) - LLM client library with carbon tracking
- [Pydantic](https://pydantic.dev/) - Data validation using Python type hints
