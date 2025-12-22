"""Tests for exception handling and fail-fast behavior."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from structured_extractor import (
    DocumentExtractor,
    ExtractionConfig,
    ExtractionValidationError,
    LLMError,
)


class SimpleSchema(BaseModel):
    """Simple schema for testing."""

    name: str
    age: int


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock()
    client.model = "gpt-4.1"
    return client


@pytest.fixture
def extractor(mock_client: MagicMock) -> DocumentExtractor:
    """Create a DocumentExtractor with a mock client."""
    return DocumentExtractor(client=mock_client)


def test_always_raises_on_llm_failure(extractor: DocumentExtractor, mock_client: MagicMock) -> None:
    """Test that LLMError is always raised on failure."""
    mock_client.generate.side_effect = Exception("API Connection Failed")

    config = ExtractionConfig(max_retries=1)

    with pytest.raises(LLMError) as excinfo:
        extractor.extract("Test doc", schema=SimpleSchema, config=config)

    assert "LLM call failed" in str(excinfo.value)
    assert "API Connection Failed" in str(excinfo.value)


def test_always_raises_on_validation_failure(
    extractor: DocumentExtractor, mock_client: MagicMock
) -> None:
    """Test that ExtractionValidationError is always raised on validation failure."""
    # Mock a response that fails Pydantic validation
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.content = '{"name": "John"}'  # Missing 'age'
    mock_client.generate.return_value = mock_response

    config = ExtractionConfig(max_retries=1)

    with pytest.raises(ExtractionValidationError) as excinfo:
        extractor.extract("Test doc", schema=SimpleSchema, config=config)

    assert "LLM did not return parsed data matching the schema" in str(excinfo.value)
    assert excinfo.value.raw_response == '{"name": "John"}'


def test_retry_on_validation_error_behavior(
    extractor: DocumentExtractor, mock_client: MagicMock
) -> None:
    """Test that retry_on_validation_error works with the new logic."""
    mock_response = MagicMock()
    mock_response.parsed = SimpleSchema(name="John", age=30)
    mock_response.content = '{"name": "John", "age": 30}'
    mock_response.model = "gpt-4.1"
    mock_response.cached = False
    mock_response.usage.total_tokens = 100
    mock_response.tracking = None

    # First call returns None for parsed (validation failure), second succeeds
    mock_fail_response = MagicMock()
    mock_fail_response.parsed = None
    mock_fail_response.content = '{"invalid": "data"}'

    mock_client.generate.side_effect = [mock_fail_response, mock_response]

    config = ExtractionConfig(retry_on_validation_error=True, max_retries=2)
    result = extractor.extract("Test doc", schema=SimpleSchema, config=config)

    assert result.data.name == "John"
    assert mock_client.generate.call_count == 2
