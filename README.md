# 📄 structured-extractor

A powerful LLM-driven structured data extractor for document parsing, built on top of [seeds-clients](https://github.com/leokeba/seeds-clients) and Pydantic.

## 🎯 Overview

`structured-extractor` provides a high-level, declarative API for extracting structured data from documents using Large Language Models. Define your data schema with Pydantic models, and let the library handle the complexity of prompt engineering, chunking, validation, and multi-field extraction.

### Key Features

- 🔧 **Declarative Schema Definition** - Use Pydantic models to define extraction templates
- 📑 **Document Chunking** - Automatic handling of large documents that exceed context limits
- 🔄 **Multi-pass Extraction** - Extract different data types in separate optimized passes
- ✅ **Validation & Coercion** - Automatic type validation and data coercion via Pydantic
- 💾 **Smart Caching** - Leverage seeds-clients caching for repeated extractions
- 🌍 **Carbon Tracking** - Built-in environmental impact tracking from seeds-clients
- 📊 **Confidence Scores** - Optional confidence scoring for extracted fields
- 🔌 **Extensible** - Support for custom extractors and post-processors

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
from structured_extractor import DocumentTemplate, FieldConfig

# Define a reusable template
invoice_template = DocumentTemplate(
    name="invoice",
    schema=Invoice,
    system_prompt="You are an expert invoice data extractor. Extract data precisely.",
    field_configs={
        "date": FieldConfig(
            extraction_hint="Look for date formats like MM/DD/YYYY or YYYY-MM-DD",
            required=True,
        ),
        "total_amount": FieldConfig(
            extraction_hint="The final total after taxes and discounts",
            required=True,
        ),
    },
)

result = extractor.extract(document_text, template=invoice_template)
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

config = ExtractionConfig(
    include_confidence=True,
    confidence_threshold=0.8,  # Flag low-confidence extractions
)

result = extractor.extract(document_text, schema=Invoice, config=config)

print(f"Data: {result.data}")
print(f"Confidence: {result.confidence}")
print(f"Field confidences: {result.field_confidences}")
# Output:
# Confidence: 0.95
# Field confidences: {'invoice_number': 0.99, 'date': 0.95, 'vendor_name': 0.92, ...}
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
│   ├── batch.py             # BatchExtractor for parallel processing
│   ├── templates.py         # DocumentTemplate definitions
│   └── config.py            # ExtractionConfig and options
├── chunking/
│   ├── strategies.py        # Chunking strategy implementations
│   ├── semantic.py          # Semantic chunking (sentence-aware)
│   └── merger.py            # Result merging from chunks
├── prompts/
│   ├── builder.py           # Dynamic prompt construction
│   ├── templates.py         # Prompt templates
│   └── optimizer.py         # Prompt optimization utilities
├── validation/
│   ├── validators.py        # Custom validation rules
│   └── coercion.py          # Type coercion utilities
├── results/
│   ├── types.py             # ExtractionResult, FieldResult types
│   └── confidence.py        # Confidence scoring logic
└── utils/
    ├── tokens.py            # Token counting utilities
    └── text.py              # Text preprocessing utilities
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
- [ ] Extraction quality metrics
- [x] Retry logic for errors
- [ ] Human-in-the-loop hooks

**Deliverables**:
- ✅ `extract_with_confidence()` method for confidence-scored extractions
- ✅ `ConfidenceAssessment` and `FieldConfidence` types
- ✅ `low_confidence_fields` in `ExtractionResult`
- ✅ Retry mechanisms
- ✅ 127 tests total

### Phase 5: Templates & Presets

**Goals**: Reusable extraction templates

- [ ] `DocumentTemplate` class enhancements
- [ ] Template serialization (YAML/JSON)
- [ ] Built-in templates (invoice, receipt, contract, resume)
- [ ] Template inheritance
- [ ] Template validation
- [ ] Template registry

**Deliverables**:
- Reusable template system
- Common document templates
- Template management utilities

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
# Required
OPENAI_API_KEY=your-api-key

# Optional
STRUCTURED_EXTRACTOR_CACHE_DIR=./cache
STRUCTURED_EXTRACTOR_CACHE_TTL=24  # hours
STRUCTURED_EXTRACTOR_DEFAULT_MODEL=gpt-4.1
```

### Extractor Configuration

```python
from structured_extractor import DocumentExtractor, ExtractionConfig

extractor = DocumentExtractor(
    # LLM Configuration
    api_key="your-key",
    model="gpt-4.1",
    temperature=0.0,  # Deterministic extraction
    
    # Caching
    cache_dir="./cache",
    cache_ttl_hours=24.0,
    
    # Chunking
    chunking_strategy="semantic",
    chunk_size=4000,
    chunk_overlap=200,
    
    # Extraction
    max_retries=3,
    retry_on_validation_error=True,
)
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [seeds-clients](https://github.com/leokeba/seeds-clients) - LLM client library with carbon tracking
- [Pydantic](https://pydantic.dev/) - Data validation using Python type hints
