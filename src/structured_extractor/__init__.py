"""
structured-extractor: A LLM-driven structured data extractor for document parsing.
"""

from structured_extractor.core.config import ExtractionConfig, FieldConfig
from structured_extractor.core.extractor import DocumentExtractor
from structured_extractor.core.templates import DocumentTemplate
from structured_extractor.prompts.builder import (
    ExtractionExample,
    PromptBuilder,
    PromptStrategy,
    PromptTemplate,
    PromptTemplates,
)
from structured_extractor.results.confidence import (
    ConfidenceAssessment,
    FieldConfidence,
)
from structured_extractor.results.types import ExtractionResult, FieldResult

__version__ = "0.1.0"

__all__ = [
    # Core
    "DocumentExtractor",
    "DocumentTemplate",
    # Config
    "ExtractionConfig",
    "FieldConfig",
    # Prompts
    "PromptBuilder",
    "PromptStrategy",
    "PromptTemplate",
    "PromptTemplates",
    "ExtractionExample",
    # Results
    "ExtractionResult",
    "FieldResult",
    # Confidence
    "ConfidenceAssessment",
    "FieldConfidence",
]
