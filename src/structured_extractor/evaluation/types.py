"""Type definitions for extraction evaluation."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MatchStatus(str, Enum):
    """Classification status for a field match in confusion matrix terms."""

    TRUE_POSITIVE = "tp"  # Expected field correctly extracted
    FALSE_POSITIVE = "fp"  # Field extracted but with incorrect value
    FALSE_NEGATIVE = "fn"  # Expected field not extracted or wrong value
    TRUE_NEGATIVE = "tn"  # Field not expected and not extracted (rare)


class FieldMatch(BaseModel):
    """Match result for a single field with both binary and score outputs.

    Provides dual output for flexibility:
    - `matched`: Binary decision for precision/recall/F1 calculations
    - `score`: Continuous similarity score (0.0-1.0) for weighted metrics
    """

    field_name: str = Field(description="Name of the evaluated field")

    # Dual output
    matched: bool = Field(description="Binary match decision (for P/R/F1)")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Continuous similarity score 0.0-1.0 (for averaging)",
    )

    # Classification
    status: MatchStatus = Field(description="Confusion matrix classification")

    # Values
    expected: Any = Field(description="Expected value from ground truth")
    actual: Any = Field(description="Actual extracted value")

    # Comparison details
    threshold_used: float | None = Field(
        default=None,
        description="Threshold used for binary match decision",
    )
    comparator_type: str | None = Field(
        default=None,
        description="Type of comparator used",
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of the comparison result",
    )


class ExtractionEvaluation(BaseModel):
    """Full evaluation of an extraction against ground truth.

    Contains both binary metrics (precision/recall/F1/accuracy) and
    score-based metrics (mean/min scores) for comprehensive evaluation.

    Example:
        ```python
        evaluation = evaluator.evaluate(doc, schema, ground_truth)
        print(f"F1: {evaluation.f1_score:.2%}")
        print(f"Mean Score: {evaluation.mean_score:.2%}")
        ```
    """

    test_id: str = Field(description="Unique identifier for this test case")
    schema_name: str = Field(description="Name of the schema used for extraction")
    extraction_success: bool = Field(
        description="Whether the extraction itself succeeded (not the evaluation)"
    )

    # Field-level results
    field_matches: list[FieldMatch] = Field(description="Detailed match results for each field")

    # === Binary metrics (for P/R/F1/Accuracy) ===
    true_positives: int = Field(default=0, description="Count of correctly extracted fields")
    false_positives: int = Field(default=0, description="Count of incorrectly extracted fields")
    false_negatives: int = Field(
        default=0, description="Count of expected but missing/wrong fields"
    )
    true_negatives: int = Field(default=0, description="Count of correctly absent fields (rare)")

    precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="TP / (TP + FP) - accuracy of positive predictions",
    )
    recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="TP / (TP + FN) - coverage of expected values",
    )
    f1_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of precision and recall",
    )
    accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="(TP + TN) / total",
    )

    # === Score-based metrics (for weighted evaluation) ===
    mean_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average score across all evaluated fields",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Lowest field score in this evaluation",
    )
    max_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest field score in this evaluation",
    )

    # Performance metrics
    latency_ms: float | None = Field(
        default=None,
        description="Extraction latency in milliseconds",
    )
    tokens_used: int | None = Field(
        default=None,
        description="Total tokens used for extraction",
    )
    cost_usd: float | None = Field(
        default=None,
        description="Cost of extraction in USD",
    )
    cached: bool = Field(
        default=False,
        description="Whether the extraction was served from cache",
    )


class FieldEvaluationMetrics(BaseModel):
    """Aggregate metrics for a specific field across multiple evaluations.

    Useful for identifying which fields are consistently problematic.
    """

    field_name: str = Field(description="Name of the field")
    evaluation_count: int = Field(description="Number of evaluations including this field")

    # Binary metrics
    true_positives: int = Field(default=0)
    false_positives: int = Field(default=0)
    false_negatives: int = Field(default=0)
    true_negatives: int = Field(default=0)

    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # Score metrics
    mean_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average score for this field"
    )
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score observed")
    max_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Maximum score observed")
    std_score: float | None = Field(default=None, description="Standard deviation of scores")


class EvaluationMetrics(BaseModel):
    """Aggregate metrics across multiple extraction evaluations.

    Provides both micro-averaged (global) and macro-averaged (per-extraction)
    metrics for comprehensive performance assessment.

    Example:
        ```python
        metrics = evaluator.compute_aggregate_metrics(evaluations)
        print(f"Micro F1: {metrics.micro_f1:.2%}")
        print(f"Macro F1: {metrics.macro_f1:.2%}")
        print(f"Mean Score: {metrics.mean_score:.2%}")
        ```
    """

    total_evaluations: int = Field(description="Total number of evaluations")
    total_fields: int = Field(description="Total number of fields evaluated")
    successful_extractions: int = Field(default=0, description="Number of successful extractions")
    failed_extractions: int = Field(default=0, description="Number of failed extractions")

    # === Aggregate binary metrics ===
    total_true_positives: int = Field(default=0)
    total_false_positives: int = Field(default=0)
    total_false_negatives: int = Field(default=0)
    total_true_negatives: int = Field(default=0)

    # Micro-averaged (computed across all fields globally)
    micro_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Global precision across all fields",
    )
    micro_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Global recall across all fields",
    )
    micro_f1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Global F1 score across all fields",
    )
    micro_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Global accuracy across all fields",
    )

    # Macro-averaged (average of per-extraction metrics)
    macro_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average precision across evaluations",
    )
    macro_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average recall across evaluations",
    )
    macro_f1: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average F1 across evaluations",
    )
    macro_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average accuracy across evaluations",
    )

    # === Aggregate score metrics ===
    mean_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Global average similarity score"
    )
    median_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Median similarity score")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score observed")
    max_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Maximum score observed")
    std_score: float | None = Field(default=None, description="Standard deviation of scores")

    # Per-field breakdown
    field_metrics: dict[str, FieldEvaluationMetrics] = Field(
        default_factory=dict,
        description="Metrics broken down by field name",
    )

    # Performance stats
    latency_mean_ms: float = Field(default=0.0, description="Average extraction latency")
    latency_median_ms: float = Field(default=0.0, description="Median extraction latency")
    latency_p95_ms: float = Field(default=0.0, description="95th percentile latency")
    latency_p99_ms: float = Field(default=0.0, description="99th percentile latency")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    cached_count: int = Field(default=0, description="Number of cached extractions")
