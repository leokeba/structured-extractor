"""Extraction evaluator for comparing extractions against ground truth."""

import statistics
import time
import uuid
from typing import Any, TypeVar

from pydantic import BaseModel

from structured_extractor.core.extractor import DocumentExtractor
from structured_extractor.evaluation.comparators import (
    ComparisonResult,
    ExactComparator,
    FieldComparator,
    get_default_comparator,
)
from structured_extractor.evaluation.types import (
    EvaluationMetrics,
    ExtractionEvaluation,
    FieldEvaluationMetrics,
    FieldMatch,
    MatchStatus,
)

T = TypeVar("T", bound=BaseModel)


class EvaluationTestCase(BaseModel):
    """A test case for evaluation with ground truth.

    Example:
        ```python
        test_case = EvaluationTestCase(
            id="invoice_001",
            document="Invoice #INV-001...",
            schema_class=Invoice,
            ground_truth={"invoice_number": "INV-001", "total": 1250.0}
        )
        ```
    """

    id: str
    document: str
    schema_class: type[BaseModel]
    ground_truth: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True}


class ExtractionEvaluator:
    """Evaluates document extractions against ground truth.

    Provides precision, recall, F1, and accuracy metrics along with
    continuous similarity scores for comprehensive evaluation.

    Example:
        ```python
        from structured_extractor import DocumentExtractor
        from structured_extractor.evaluation import (
            ExtractionEvaluator,
            NumericComparator,
            StringComparator,
        )

        extractor = DocumentExtractor(model="gpt-4.1")
        evaluator = ExtractionEvaluator(extractor)

        result = evaluator.evaluate(
            document="Invoice #INV-001\\nTotal: $1,250.00",
            schema=Invoice,
            ground_truth={"invoice_number": "INV-001", "total_amount": 1250.0},
        )

        print(f"Precision: {result.precision:.2%}")
        print(f"Recall: {result.recall:.2%}")
        print(f"F1: {result.f1_score:.2%}")
        print(f"Mean Score: {result.mean_score:.2%}")
        ```
    """

    def __init__(
        self,
        extractor: DocumentExtractor,
        default_comparators: dict[str, FieldComparator] | None = None,
        default_string_threshold: float = 0.9,
        default_numeric_tolerance: float = 0.01,
        auto_select_comparators: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            extractor: DocumentExtractor instance for performing extractions.
            default_comparators: Default comparators for specific field names.
            default_string_threshold: Default threshold for string comparisons.
            default_numeric_tolerance: Default tolerance for numeric comparisons.
            auto_select_comparators: If True, automatically select comparators
                based on ground truth value types.
        """
        self.extractor = extractor
        self.default_comparators = default_comparators or {}
        self.default_string_threshold = default_string_threshold
        self.default_numeric_tolerance = default_numeric_tolerance
        self.auto_select_comparators = auto_select_comparators

        # Store evaluations for aggregate metrics
        self._evaluations: list[ExtractionEvaluation] = []

    def evaluate(
        self,
        document: str,
        schema: type[T],
        ground_truth: dict[str, Any] | T,
        test_id: str | None = None,
        field_comparators: dict[str, FieldComparator] | None = None,
    ) -> ExtractionEvaluation:
        """Evaluate a single extraction against ground truth.

        Args:
            document: The document text to extract from.
            schema: Pydantic model defining the extraction schema.
            ground_truth: Expected values (dict or Pydantic model).
            test_id: Optional unique identifier for this test case.
            field_comparators: Override comparators for specific fields.

        Returns:
            ExtractionEvaluation with precision/recall/F1/accuracy and scores.
        """
        # Generate test ID if not provided
        if test_id is None:
            test_id = str(uuid.uuid4())[:8]

        # Convert ground truth to dict if needed
        if isinstance(ground_truth, BaseModel):
            gt_dict = ground_truth.model_dump()
        else:
            gt_dict = dict(ground_truth)

        # Perform extraction with timing
        start_time = time.perf_counter()
        extraction_result = self.extractor.extract(document, schema=schema)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Get extracted data as dict
        if extraction_result.success and extraction_result.data is not None:
            extracted_dict = extraction_result.data.model_dump()
            extraction_success = True
        else:
            extracted_dict = {}
            extraction_success = False

        # Evaluate fields
        field_matches = self._evaluate_fields(
            ground_truth=gt_dict,
            extracted=extracted_dict,
            schema=schema,
            field_comparators=field_comparators,
        )

        # Compute metrics
        evaluation = self._build_evaluation(
            test_id=test_id,
            schema_name=schema.__name__,
            extraction_success=extraction_success,
            field_matches=field_matches,
            latency_ms=latency_ms,
            tokens_used=extraction_result.tokens_used,
            cost_usd=extraction_result.cost_usd,
            cached=extraction_result.cached,
        )

        # Store for aggregate metrics
        self._evaluations.append(evaluation)

        return evaluation

    def evaluate_batch(
        self,
        test_cases: list[EvaluationTestCase],
        field_comparators: dict[str, FieldComparator] | None = None,
    ) -> list[ExtractionEvaluation]:
        """Evaluate multiple test cases.

        Args:
            test_cases: List of test cases to evaluate.
            field_comparators: Override comparators for specific fields.

        Returns:
            List of ExtractionEvaluation results.
        """
        results = []
        for test_case in test_cases:
            result = self.evaluate(
                document=test_case.document,
                schema=test_case.schema_class,
                ground_truth=test_case.ground_truth,
                test_id=test_case.id,
                field_comparators=field_comparators,
            )
            results.append(result)
        return results

    def evaluate_without_extraction(
        self,
        extracted_data: dict[str, Any] | T,
        ground_truth: dict[str, Any] | T,
        schema: type[T],
        test_id: str | None = None,
        field_comparators: dict[str, FieldComparator] | None = None,
    ) -> ExtractionEvaluation:
        """Evaluate pre-extracted data against ground truth.

        Useful when you already have extraction results and want to
        evaluate them without re-running the extraction.

        Args:
            extracted_data: Pre-extracted data (dict or Pydantic model).
            ground_truth: Expected values (dict or Pydantic model).
            schema: Pydantic model defining the schema.
            test_id: Optional unique identifier.
            field_comparators: Override comparators for specific fields.

        Returns:
            ExtractionEvaluation with metrics.
        """
        if test_id is None:
            test_id = str(uuid.uuid4())[:8]

        # Convert to dicts
        if isinstance(ground_truth, BaseModel):
            gt_dict = ground_truth.model_dump()
        else:
            gt_dict = dict(ground_truth)

        if isinstance(extracted_data, BaseModel):
            extracted_dict = extracted_data.model_dump()
        else:
            extracted_dict = dict(extracted_data)

        # Evaluate fields
        field_matches = self._evaluate_fields(
            ground_truth=gt_dict,
            extracted=extracted_dict,
            schema=schema,
            field_comparators=field_comparators,
        )

        # Build evaluation (no performance metrics)
        evaluation = self._build_evaluation(
            test_id=test_id,
            schema_name=schema.__name__,
            extraction_success=True,
            field_matches=field_matches,
        )

        self._evaluations.append(evaluation)
        return evaluation

    def _evaluate_fields(
        self,
        ground_truth: dict[str, Any],
        extracted: dict[str, Any],
        schema: type[BaseModel],
        field_comparators: dict[str, FieldComparator] | None = None,
    ) -> list[FieldMatch]:
        """Compare each field in ground truth against extracted values."""
        field_matches = []
        field_comparators = field_comparators or {}

        # Get schema field info
        schema_fields = schema.model_fields

        for field_name, expected_value in ground_truth.items():
            actual_value = extracted.get(field_name)

            # Determine comparator
            comparator = self._get_comparator(
                field_name=field_name,
                expected_value=expected_value,
                schema_fields=schema_fields,
                override_comparators=field_comparators,
            )

            # Handle missing field
            if field_name not in extracted:
                field_matches.append(
                    FieldMatch(
                        field_name=field_name,
                        matched=False,
                        score=0.0,
                        status=MatchStatus.FALSE_NEGATIVE,
                        expected=expected_value,
                        actual=None,
                        comparator_type=type(comparator).__name__,
                        reasoning="Field not extracted",
                    )
                )
                continue

            # Handle None expected value (shouldn't be extracted)
            if expected_value is None:
                if actual_value is None:
                    status = MatchStatus.TRUE_NEGATIVE
                    matched = True
                    score = 1.0
                    reasoning = "Both None as expected"
                else:
                    status = MatchStatus.FALSE_POSITIVE
                    matched = False
                    score = 0.0
                    reasoning = f"Expected None, got {actual_value!r}"

                field_matches.append(
                    FieldMatch(
                        field_name=field_name,
                        matched=matched,
                        score=score,
                        status=status,
                        expected=expected_value,
                        actual=actual_value,
                        comparator_type=type(comparator).__name__,
                        reasoning=reasoning,
                    )
                )
                continue

            # Handle None actual value (should have been extracted)
            if actual_value is None:
                field_matches.append(
                    FieldMatch(
                        field_name=field_name,
                        matched=False,
                        score=0.0,
                        status=MatchStatus.FALSE_NEGATIVE,
                        expected=expected_value,
                        actual=None,
                        comparator_type=type(comparator).__name__,
                        reasoning="Expected value but got None",
                    )
                )
                continue

            # Compare values
            result: ComparisonResult = comparator.compare(expected_value, actual_value)

            # Determine status (TP if matched, FN if value was extracted but incorrect)
            status = MatchStatus.TRUE_POSITIVE if result.matched else MatchStatus.FALSE_NEGATIVE

            field_matches.append(
                FieldMatch(
                    field_name=field_name,
                    matched=result.matched,
                    score=result.score,
                    status=status,
                    expected=expected_value,
                    actual=actual_value,
                    threshold_used=result.threshold_used,
                    comparator_type=type(comparator).__name__,
                    reasoning=result.reasoning,
                )
            )

        return field_matches

    def _get_comparator(
        self,
        field_name: str,
        expected_value: Any,
        schema_fields: dict,
        override_comparators: dict[str, FieldComparator],
    ) -> FieldComparator:
        """Get the appropriate comparator for a field."""
        # Priority 1: Override from evaluate() call
        if field_name in override_comparators:
            return override_comparators[field_name]

        # Priority 2: Default comparator for this field name
        if field_name in self.default_comparators:
            return self.default_comparators[field_name]

        # Priority 3: Auto-select based on value type
        if self.auto_select_comparators:
            return get_default_comparator(
                expected_value,
                string_threshold=self.default_string_threshold,
                numeric_threshold=self.default_numeric_tolerance,
            )

        # Fallback: Exact comparator
        return ExactComparator()

    def _build_evaluation(
        self,
        test_id: str,
        schema_name: str,
        extraction_success: bool,
        field_matches: list[FieldMatch],
        latency_ms: float | None = None,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        cached: bool = False,
    ) -> ExtractionEvaluation:
        """Build an ExtractionEvaluation from field matches."""
        # Count confusion matrix elements
        tp = sum(1 for m in field_matches if m.status == MatchStatus.TRUE_POSITIVE)
        fp = sum(1 for m in field_matches if m.status == MatchStatus.FALSE_POSITIVE)
        fn = sum(1 for m in field_matches if m.status == MatchStatus.FALSE_NEGATIVE)
        tn = sum(1 for m in field_matches if m.status == MatchStatus.TRUE_NEGATIVE)

        # Compute binary metrics
        precision, recall, f1, accuracy = self._compute_binary_metrics(tp, fp, fn, tn)

        # Compute score metrics
        scores = [m.score for m in field_matches]
        mean_score = statistics.mean(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        return ExtractionEvaluation(
            test_id=test_id,
            schema_name=schema_name,
            extraction_success=extraction_success,
            field_matches=field_matches,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            mean_score=mean_score,
            min_score=min_score,
            max_score=max_score,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            cached=cached,
        )

    def _compute_binary_metrics(
        self, tp: int, fp: int, fn: int, tn: int
    ) -> tuple[float, float, float, float]:
        """Compute precision, recall, F1, and accuracy from confusion matrix."""
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1: Harmonic mean of precision and recall
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy: (TP + TN) / total
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return precision, recall, f1, accuracy

    def compute_aggregate_metrics(
        self,
        evaluations: list[ExtractionEvaluation] | None = None,
    ) -> EvaluationMetrics:
        """Compute aggregate metrics across multiple evaluations.

        Args:
            evaluations: List of evaluations. If None, uses stored evaluations.

        Returns:
            EvaluationMetrics with micro/macro averaged metrics.
        """
        if evaluations is None:
            evaluations = self._evaluations

        if not evaluations:
            return EvaluationMetrics(
                total_evaluations=0,
                total_fields=0,
            )

        # Aggregate counts
        total_tp = sum(e.true_positives for e in evaluations)
        total_fp = sum(e.false_positives for e in evaluations)
        total_fn = sum(e.false_negatives for e in evaluations)
        total_tn = sum(e.true_negatives for e in evaluations)

        # Micro-averaged metrics (global)
        micro_p, micro_r, micro_f1, micro_acc = self._compute_binary_metrics(
            total_tp, total_fp, total_fn, total_tn
        )

        # Macro-averaged metrics (average of per-evaluation metrics)
        macro_precision = statistics.mean(e.precision for e in evaluations)
        macro_recall = statistics.mean(e.recall for e in evaluations)
        macro_f1 = statistics.mean(e.f1_score for e in evaluations)
        macro_accuracy = statistics.mean(e.accuracy for e in evaluations)

        # Score metrics
        all_scores = [m.score for e in evaluations for m in e.field_matches]
        mean_score = statistics.mean(all_scores) if all_scores else 0.0
        median_score = statistics.median(all_scores) if all_scores else 0.0
        min_score = min(all_scores) if all_scores else 0.0
        max_score = max(all_scores) if all_scores else 0.0
        std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else None

        # Per-field metrics
        field_metrics = self._compute_field_metrics(evaluations)

        # Latency stats
        latencies = [e.latency_ms for e in evaluations if e.latency_ms is not None]
        latency_mean = statistics.mean(latencies) if latencies else 0.0
        latency_median = statistics.median(latencies) if latencies else 0.0
        sorted_latencies = sorted(latencies) if latencies else []
        latency_p95 = (
            sorted_latencies[int(len(sorted_latencies) * 0.95)]
            if len(sorted_latencies) > 0
            else 0.0
        )
        latency_p99 = (
            sorted_latencies[int(len(sorted_latencies) * 0.99)]
            if len(sorted_latencies) > 0
            else 0.0
        )

        # Cost/token totals
        total_tokens = sum(e.tokens_used or 0 for e in evaluations)
        total_cost = sum(e.cost_usd or 0.0 for e in evaluations)
        cached_count = sum(1 for e in evaluations if e.cached)

        return EvaluationMetrics(
            total_evaluations=len(evaluations),
            total_fields=sum(len(e.field_matches) for e in evaluations),
            successful_extractions=sum(1 for e in evaluations if e.extraction_success),
            failed_extractions=sum(1 for e in evaluations if not e.extraction_success),
            total_true_positives=total_tp,
            total_false_positives=total_fp,
            total_false_negatives=total_fn,
            total_true_negatives=total_tn,
            micro_precision=micro_p,
            micro_recall=micro_r,
            micro_f1=micro_f1,
            micro_accuracy=micro_acc,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            macro_accuracy=macro_accuracy,
            mean_score=mean_score,
            median_score=median_score,
            min_score=min_score,
            max_score=max_score,
            std_score=std_score,
            field_metrics=field_metrics,
            latency_mean_ms=latency_mean,
            latency_median_ms=latency_median,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            cached_count=cached_count,
        )

    def _compute_field_metrics(
        self, evaluations: list[ExtractionEvaluation]
    ) -> dict[str, FieldEvaluationMetrics]:
        """Compute per-field metrics across evaluations."""
        # Collect field data
        field_data: dict[str, list[FieldMatch]] = {}
        for evaluation in evaluations:
            for match in evaluation.field_matches:
                if match.field_name not in field_data:
                    field_data[match.field_name] = []
                field_data[match.field_name].append(match)

        # Compute metrics for each field
        field_metrics = {}
        for field_name, matches in field_data.items():
            tp = sum(1 for m in matches if m.status == MatchStatus.TRUE_POSITIVE)
            fp = sum(1 for m in matches if m.status == MatchStatus.FALSE_POSITIVE)
            fn = sum(1 for m in matches if m.status == MatchStatus.FALSE_NEGATIVE)
            tn = sum(1 for m in matches if m.status == MatchStatus.TRUE_NEGATIVE)

            precision, recall, f1, accuracy = self._compute_binary_metrics(tp, fp, fn, tn)

            scores = [m.score for m in matches]
            mean_score = statistics.mean(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 1.0
            std_score = statistics.stdev(scores) if len(scores) > 1 else None

            field_metrics[field_name] = FieldEvaluationMetrics(
                field_name=field_name,
                evaluation_count=len(matches),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                mean_score=mean_score,
                min_score=min_score,
                max_score=max_score,
                std_score=std_score,
            )

        return field_metrics

    def clear_evaluations(self) -> None:
        """Clear stored evaluations."""
        self._evaluations.clear()

    @property
    def evaluations(self) -> list[ExtractionEvaluation]:
        """Get stored evaluations."""
        return list(self._evaluations)
