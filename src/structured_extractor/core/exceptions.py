"""Custom exceptions for structured-extractor."""

from typing import Any


class StructuredExtractorError(Exception):
    """Base exception for all structured-extractor errors."""

    pass


class ExtractionError(StructuredExtractorError):
    """Raised when extraction fails."""

    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        last_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.last_error = last_error


class ExtractionValidationError(ExtractionError):
    """Raised when extracted data fails Pydantic validation."""

    def __init__(
        self,
        message: str,
        validation_errors: Any = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message, raw_response=raw_response)
        self.validation_errors = validation_errors


class LLMError(ExtractionError):
    """Raised when the LLM client fails or returns an invalid response."""

    pass


class ConfigurationError(StructuredExtractorError):
    """Raised when extraction configuration is invalid."""

    pass


class TemplateValidationError(StructuredExtractorError):
    """Raised when template validation fails."""

    pass
