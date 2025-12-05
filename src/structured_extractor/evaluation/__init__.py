"""Evaluation module for structured-extractor.

This module provides tools for evaluating extraction quality against
ground truth data, with support for precision, recall, F1, and accuracy
metrics, as well as continuous similarity scores.

Example:
    ```python
    from structured_extractor import DocumentExtractor
    from structured_extractor.evaluation import (
        ExtractionEvaluator,
        NumericComparator,
        StringComparator,
        EvaluationReporter,
    )

    # Create evaluator
    extractor = DocumentExtractor(model="gpt-4.1")
    evaluator = ExtractionEvaluator(extractor)

    # Evaluate extraction
    result = evaluator.evaluate(
        document="Invoice #INV-001\\nTotal: $1,250.00",
        schema=Invoice,
        ground_truth={"invoice_number": "INV-001", "total_amount": 1250.0},
    )

    print(f"F1 Score: {result.f1_score:.2%}")
    print(f"Mean Score: {result.mean_score:.2%}")

    # Generate report
    metrics = evaluator.compute_aggregate_metrics()
    reporter = EvaluationReporter(evaluator.evaluations, metrics)
    reporter.save("report.md")
    ```
"""

# Types
# Comparators
from structured_extractor.evaluation.comparators import (
    ComparisonResult,
    ContainsComparator,
    DateComparator,
    ExactComparator,
    FieldComparator,
    ListComparator,
    NestedModelComparator,
    NumericComparator,
    StringComparator,
    get_default_comparator,
)

# Evaluator
from structured_extractor.evaluation.evaluator import (
    EvaluationTestCase,
    ExtractionEvaluator,
)

# Reporters
from structured_extractor.evaluation.reporters import EvaluationReporter
from structured_extractor.evaluation.types import (
    EvaluationMetrics,
    ExtractionEvaluation,
    FieldEvaluationMetrics,
    FieldMatch,
    MatchStatus,
)

__all__ = [
    # Types
    "MatchStatus",
    "FieldMatch",
    "ExtractionEvaluation",
    "FieldEvaluationMetrics",
    "EvaluationMetrics",
    # Comparators
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
    # Evaluator
    "ExtractionEvaluator",
    "EvaluationTestCase",
    # Reporters
    "EvaluationReporter",
]
