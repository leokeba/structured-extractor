"""Tests for the DocumentExtractor class."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from structured_extractor import BaseClient, DocumentExtractor, ExtractionConfig


class Invoice(BaseModel):
    """Test invoice model."""

    invoice_number: str = Field(description="The invoice ID")
    total_amount: float = Field(description="Total amount due")
    vendor_name: str = Field(description="Name of the vendor")


class Person(BaseModel):
    """Test person model."""

    name: str
    age: int


class TestDocumentExtractorInit:
    """Tests for DocumentExtractor initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        with patch("structured_extractor.core.extractor.OpenAIClient"):
            extractor = DocumentExtractor(api_key="test-key")

            assert extractor.model == "gpt-4.1"
            assert extractor.default_config is not None
            assert extractor.default_config.temperature == 0.0

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        with patch("structured_extractor.core.extractor.OpenAIClient"):
            extractor = DocumentExtractor(
                api_key="test-key",
                model="gpt-4.1-mini",
            )

            assert extractor.model == "gpt-4.1-mini"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ExtractionConfig(
            temperature=0.5,
            max_retries=5,
        )

        with patch("structured_extractor.core.extractor.OpenAIClient"):
            extractor = DocumentExtractor(
                api_key="test-key",
                default_config=config,
            )

            assert extractor.default_config.temperature == 0.5
            assert extractor.default_config.max_retries == 5

    def test_init_with_client_injection(self) -> None:
        """Test initialization with injected client."""
        mock_client = MagicMock(spec=BaseClient)
        mock_client.model = "claude-sonnet-4-20250514"

        extractor = DocumentExtractor(client=mock_client)

        assert extractor._client is mock_client
        assert extractor.model == "claude-sonnet-4-20250514"

    def test_init_with_client_ignores_other_params(self) -> None:
        """Test that client parameter takes precedence over other parameters."""
        mock_client = MagicMock(spec=BaseClient)
        mock_client.model = "custom-model"

        # These parameters should be ignored when client is provided
        with patch("structured_extractor.core.extractor.OpenAIClient") as mock_openai:
            extractor = DocumentExtractor(
                client=mock_client,
                api_key="should-be-ignored",
                model="should-be-ignored",
                cache_dir="should-be-ignored",
            )

            # OpenAIClient should not be instantiated
            mock_openai.assert_not_called()
            assert extractor._client is mock_client
            assert extractor.model == "custom-model"


class TestDocumentExtractorWithInjectedClient:
    """Tests for DocumentExtractor with injected client."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client that simulates any BaseClient."""
        client = MagicMock(spec=BaseClient)
        client.model = "mock-model"
        return client

    @pytest.fixture
    def extractor(self, mock_client: MagicMock) -> DocumentExtractor:
        """Create an extractor with injected mock client."""
        return DocumentExtractor(client=mock_client)

    def test_extract_with_injected_client(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test extraction works with an injected client."""
        mock_response = MagicMock()
        mock_response.parsed = Person(name="Alice", age=25)
        mock_response.content = json.dumps({"name": "Alice", "age": 25})
        mock_response.model = "mock-model"
        mock_response.cached = False
        mock_response.usage.total_tokens = 100
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        result = extractor.extract("Alice is 25 years old.", schema=Person)

        assert result.success is True
        assert result.data.name == "Alice"
        assert result.data.age == 25
        assert result.model_used == "mock-model"
        mock_client.generate.assert_called_once()

    def test_extract_with_confidence_injected_client(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test extract_with_confidence works with an injected client."""
        # Mock the confidence-wrapped response
        mock_response = MagicMock()
        mock_response.parsed = MagicMock()
        mock_response.parsed.extracted_data = Person(name="Bob", age=30)
        mock_response.parsed.confidence_assessment = MagicMock()
        mock_response.parsed.confidence_assessment.overall_confidence = 0.95
        mock_response.parsed.confidence_assessment.field_confidences = [
            MagicMock(field_name="name", confidence=0.98),
            MagicMock(field_name="age", confidence=0.92),
        ]
        mock_response.content = "{}"
        mock_response.model = "mock-model"
        mock_response.cached = False
        mock_response.usage.total_tokens = 150
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        result = extractor.extract_with_confidence("Bob is 30.", schema=Person)

        assert result.success is True
        assert result.data.name == "Bob"
        assert result.confidence == 0.95
        mock_client.generate.assert_called_once()

    def test_cumulative_tracking_with_injected_client(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test cumulative_tracking property works with injected client."""
        mock_tracking = MagicMock()
        mock_tracking.total_request_count = 5
        mock_client.cumulative_tracking = mock_tracking

        tracking = extractor.cumulative_tracking

        assert tracking.total_request_count == 5


class TestDocumentExtractorExtract:
    """Tests for DocumentExtractor.extract method."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_client: MagicMock) -> DocumentExtractor:
        """Create an extractor with mocked client."""
        with patch("structured_extractor.core.extractor.OpenAIClient", return_value=mock_client):
            return DocumentExtractor(api_key="test-key")

    def test_extract_requires_schema_or_template(self, extractor: DocumentExtractor) -> None:
        """Test that extract raises error without schema or template."""
        with pytest.raises(ValueError, match="schema.*template"):
            extractor.extract("Some document")

    def test_extract_simple_document(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test extracting from a simple document."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.parsed = Person(name="John Doe", age=30)
        mock_response.content = json.dumps({"name": "John Doe", "age": 30})
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 100
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        document = "John Doe is 30 years old."
        result = extractor.extract(document, schema=Person)

        assert result.success is True
        assert result.data.name == "John Doe"
        assert result.data.age == 30
        assert result.model_used == "gpt-4.1"

    def test_extract_invoice(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test extracting invoice data."""
        mock_response = MagicMock()
        mock_response.parsed = Invoice(
            invoice_number="INV-001",
            total_amount=1500.00,
            vendor_name="Acme Corp",
        )
        mock_response.content = json.dumps({
            "invoice_number": "INV-001",
            "total_amount": 1500.00,
            "vendor_name": "Acme Corp",
        })
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 150
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        document = """
        INVOICE #INV-001
        From: Acme Corp
        Total: $1,500.00
        """

        result = extractor.extract(document, schema=Invoice)

        assert result.success is True
        assert result.data.invoice_number == "INV-001"
        assert result.data.total_amount == 1500.00
        assert result.data.vendor_name == "Acme Corp"

    def test_extract_uses_response_format(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that extract passes response_format to LLM client."""
        mock_response = MagicMock()
        mock_response.parsed = Person(name="Test", age=25)
        mock_response.content = "{}"
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 50
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        extractor.extract("Test doc", schema=Person)

        # Check that generate was called with response_format
        call_kwargs = mock_client.generate.call_args.kwargs
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"] == Person

    def test_extract_with_custom_config(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test extract with custom configuration."""
        mock_response = MagicMock()
        mock_response.parsed = Person(name="Test", age=25)
        mock_response.content = "{}"
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 50
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        config = ExtractionConfig(temperature=0.5, max_tokens=500)
        extractor.extract("Test doc", schema=Person, config=config)

        call_kwargs = mock_client.generate.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500

    def test_extract_cached_response(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that cached responses are properly indicated."""
        mock_response = MagicMock()
        mock_response.parsed = Person(name="Cached", age=99)
        mock_response.content = "{}"
        mock_response.model = "gpt-4.1"
        mock_response.cached = True
        mock_response.usage.total_tokens = 0
        mock_response.tracking = None
        mock_client.generate.return_value = mock_response

        result = extractor.extract("Test doc", schema=Person)

        assert result.cached is True

    def test_extract_handles_error(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that extraction errors are handled gracefully."""
        mock_client.generate.side_effect = Exception("API Error")

        result = extractor.extract("Test doc", schema=Person)

        assert result.success is False
        assert "API Error" in str(result.error)

    def test_extract_retries_on_error(
        self, extractor: DocumentExtractor, mock_client: MagicMock
    ) -> None:
        """Test that extraction retries on failure."""
        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.parsed = Person(name="Success", age=1)
        mock_response.content = "{}"
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 50
        mock_response.tracking = None

        mock_client.generate.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            mock_response,
        ]

        result = extractor.extract("Test doc", schema=Person)

        assert result.success is True
        assert mock_client.generate.call_count == 3
