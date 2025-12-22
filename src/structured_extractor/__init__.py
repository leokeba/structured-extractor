"""
structured-extractor: A LLM-driven structured data extractor for document parsing.
"""

from seeds_clients.core.base_client import BaseClient
from seeds_clients.core.types import CumulativeTracking
from seeds_clients.tracking.boamps_reporter import BoAmpsReport, BoAmpsReporter

from structured_extractor.core.config import ExtractionConfig, FieldConfig
from structured_extractor.core.exceptions import (
    ConfigurationError,
    ExtractionError,
    ExtractionValidationError,
    LLMError,
    StructuredExtractorError,
    TemplateValidationError,
)
from structured_extractor.core.extractor import DocumentExtractor
from structured_extractor.core.templates import (
    DocumentTemplate,
    TemplateRegistry,
)

# Evaluation
from structured_extractor.evaluation import (
    ComparisonResult,
    ContainsComparator,
    DateComparator,
    EvaluationMetrics,
    EvaluationReporter,
    EvaluationTestCase,
    ExactComparator,
    ExtractionEvaluation,
    ExtractionEvaluator,
    FieldComparator,
    FieldEvaluationMetrics,
    FieldMatch,
    ListComparator,
    MatchStatus,
    NestedModelComparator,
    NumericComparator,
    StringComparator,
    get_default_comparator,
)
from structured_extractor.prompts.builder import (
    ExtractionExample,
    PromptBuilder,
    PromptStrategy,
    PromptTemplate,
    PromptTemplates,
)
from structured_extractor.results.confidence import (
    ConfidenceAssessment,
    ExtractionQualityMetrics,
    FieldConfidence,
    compute_quality_metrics,
)
from structured_extractor.results.types import ExtractionResult, FieldResult

# Built-in schemas
from structured_extractor.schemas import (
    ContractParty,
    ContractSchema,
    Education,
    InvoiceLineItem,
    InvoiceSchema,
    ReceiptSchema,
    ResumeSchema,
    WorkExperience,
)

# Built-in templates
from structured_extractor.templates import BuiltinTemplates

__version__ = "0.1.0"

__all__ = [
    # Core
    "DocumentExtractor",
    "DocumentTemplate",
    "TemplateRegistry",
    "StructuredExtractorError",
    "ExtractionError",
    "ExtractionValidationError",
    "LLMError",
    "ConfigurationError",
    "TemplateValidationError",
    "BaseClient",  # For type hints when injecting clients
    # Built-in Templates
    "BuiltinTemplates",
    # Built-in Schemas
    "InvoiceSchema",
    "InvoiceLineItem",
    "ReceiptSchema",
    "ResumeSchema",
    "WorkExperience",
    "Education",
    "ContractSchema",
    "ContractParty",
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
    # Confidence & Quality
    "ConfidenceAssessment",
    "FieldConfidence",
    "ExtractionQualityMetrics",
    "compute_quality_metrics",
    # Evaluation
    "MatchStatus",
    "FieldMatch",
    "ExtractionEvaluation",
    "FieldEvaluationMetrics",
    "EvaluationMetrics",
    "ComparisonResult",
    "FieldComparator",
    "ExactComparator",
    "NumericComparator",
    "StringComparator",
    "ContainsComparator",
    "DateComparator",
    "ListComparator",
    "NestedModelComparator",
    "get_default_comparator",
    "ExtractionEvaluator",
    "EvaluationTestCase",
    "EvaluationReporter",
    # Tracking & Reporting
    "CumulativeTracking",
    "BoAmpsReport",
    "BoAmpsReporter",
]
