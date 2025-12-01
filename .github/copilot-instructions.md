# Copilot Instructions for structured-extractor

## Project Overview

This is a Python library for LLM-driven structured data extraction from documents. It uses Pydantic models to define extraction schemas and leverages [seeds-clients](https://github.com/leokeba/seeds-clients) for LLM interactions with built-in carbon tracking, cost monitoring, and smart caching.

## Development Environment

### Package Manager: uv

We use **uv** for all Python-related tasks:

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Run Python scripts
uv run python script.py

# Run pytest
uv run python -m pytest tests/

# Run a specific test file
uv run python -m pytest tests/test_extractor.py -v

# Run with coverage
uv run python -m pytest tests/ --cov=structured_extractor
```

**Do NOT use:**
- `pip install` - use `uv add` or `uv sync` instead
- `python` directly - use `uv run python` instead
- `pytest` directly - use `uv run python -m pytest` instead

### Environment Variables

API keys are loaded from `.env` file using `python-dotenv`. The `.env` file is gitignored.

Required:
- `OPENAI_API_KEY` - OpenAI API key for LLM calls

## Code Quality Requirements

### Before Committing

**Always run tests and check for errors before committing:**

```bash
# Run all tests
uv run python -m pytest tests/ -v --tb=short

# Check for lint errors
uv run ruff check src/ tests/

# Run type checking
uv run ty check src/
```

- All tests must pass
- No new lint errors should be introduced
- Type checking should pass
- Coverage should not decrease significantly

### Linting and Formatting

We use **ruff** for linting with the following rules enabled:
- E, F: pyflakes and pycodestyle errors
- I: isort import sorting
- N: pep8-naming
- W: pycodestyle warnings
- UP: pyupgrade
- B: flake8-bugbear
- C4: flake8-comprehensions
- SIM: flake8-simplify

Line length is 100 characters.

### Type Checking

We use **ty** for type checking. All code must be fully typed.

## Project Structure

```
src/structured_extractor/
├── __init__.py          # Public API exports
├── core/
│   ├── config.py        # ExtractionConfig, FieldConfig
│   ├── extractor.py     # Main DocumentExtractor class
│   └── templates.py     # DocumentTemplate definitions
├── prompts/
│   └── builder.py       # PromptBuilder, PromptStrategy, PromptTemplates
└── results/
    └── types.py         # ExtractionResult, FieldResult types
tests/                   # Test files (test_*.py)
examples/                # Usage examples
```

## Key Patterns

### Schema Definition
- Extraction schemas are defined using **Pydantic models**
- Use `Field(description="...")` to provide extraction hints to the LLM
- Support for nested models, lists, optionals, enums, literals, and unions

### Core Classes
- `DocumentExtractor` - Main class for extracting data from documents
- `PromptBuilder` - Constructs prompts from schemas with customizable strategies
- `PromptStrategy` - Configures prompt behavior (strict mode, reasoning, etc.)
- `PromptTemplates` - Built-in templates for common extraction scenarios
- `ExtractionResult` - Contains extracted data with metadata

### Testing Patterns
- Tests use `unittest.mock` to mock the `OpenAIClient` from seeds-clients
- Mock responses should return valid JSON matching the expected schema
- Use `pytest.mark.asyncio` for async test methods
- Test files follow the pattern `test_<module>.py`

### Example Test Pattern

```python
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field
from structured_extractor import DocumentExtractor

class Invoice(BaseModel):
    invoice_number: str = Field(description="The invoice ID")
    total: float = Field(description="Total amount")

def test_extraction():
    with patch("structured_extractor.core.extractor.OpenAIClient") as mock_client:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = '{"invoice_number": "INV-001", "total": 100.0}'
        mock_client.return_value.generate.return_value = mock_response
        
        # Test extraction
        extractor = DocumentExtractor(api_key="test-key")
        result = extractor.extract("Invoice #INV-001, Total: $100", schema=Invoice)
        
        assert result.data.invoice_number == "INV-001"
```

## Default Model for Testing

We use **`gpt-4.1`** as the default model. When adding new tests or examples, use mocks for unit tests to avoid API calls.

## Dependencies

This project depends on:
- `seeds-clients` - LLM client library with carbon tracking (from GitHub)
- `pydantic>=2.0` - Data validation and schema definition

Dev dependencies:
- `pytest>=8.0` - Testing framework
- `pytest-asyncio>=0.23` - Async test support
- `pytest-cov>=4.0` - Coverage reporting
- `ruff>=0.4` - Linting
- `ty` - Type checking

## Python Version

This project requires **Python 3.13+** (`requires-python = ">=3.13"`).
