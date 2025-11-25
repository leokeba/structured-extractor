"""Core extraction functionality."""

from structured_extractor.core.config import ExtractionConfig, FieldConfig
from structured_extractor.core.extractor import DocumentExtractor
from structured_extractor.core.templates import (
    DocumentTemplate,
    TemplateRegistry,
    TemplateValidationError,
)

__all__ = [
    "DocumentExtractor",
    "DocumentTemplate",
    "ExtractionConfig",
    "FieldConfig",
    "TemplateRegistry",
    "TemplateValidationError",
]
