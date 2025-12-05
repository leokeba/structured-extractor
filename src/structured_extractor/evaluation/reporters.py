"""Evaluation reporters for generating reports in various formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from structured_extractor.evaluation.types import (
    EvaluationMetrics,
    ExtractionEvaluation,
)


class EvaluationReporter:
    """Generate evaluation reports in various formats.

    Supports text, JSON, and Markdown output formats.

    Example:
        ```python
        evaluations = evaluator.evaluate_batch(test_cases)
        metrics = evaluator.compute_aggregate_metrics(evaluations)

        reporter = EvaluationReporter(evaluations, metrics)
        reporter.save("report.md")  # Auto-detects format from extension

        # Or get report as string
        print(reporter.to_text())
        ```
    """

    def __init__(
        self,
        evaluations: list[ExtractionEvaluation],
        metrics: EvaluationMetrics | None = None,
        title: str = "Extraction Evaluation Report",
    ) -> None:
        """Initialize the reporter.

        Args:
            evaluations: List of evaluation results.
            metrics: Pre-computed aggregate metrics. Computed if not provided.
            title: Title for the report.
        """
        self.evaluations = evaluations
        self.title = title

        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> EvaluationMetrics:
        """Compute aggregate metrics from evaluations."""
        import statistics

        if not self.evaluations:
            return EvaluationMetrics(total_evaluations=0, total_fields=0)

        # This duplicates logic from ExtractionEvaluator but allows
        # standalone use of the reporter
        total_tp = sum(e.true_positives for e in self.evaluations)
        total_fp = sum(e.false_positives for e in self.evaluations)
        total_fn = sum(e.false_negatives for e in self.evaluations)
        total_tn = sum(e.true_negatives for e in self.evaluations)

        # Micro metrics
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r) > 0
            else 0.0
        )
        total = total_tp + total_fp + total_fn + total_tn
        micro_acc = (total_tp + total_tn) / total if total > 0 else 0.0

        # Macro metrics
        macro_p = statistics.mean(e.precision for e in self.evaluations)
        macro_r = statistics.mean(e.recall for e in self.evaluations)
        macro_f1 = statistics.mean(e.f1_score for e in self.evaluations)
        macro_acc = statistics.mean(e.accuracy for e in self.evaluations)

        # Score metrics
        all_scores = [m.score for e in self.evaluations for m in e.field_matches]
        mean_score = statistics.mean(all_scores) if all_scores else 0.0
        median_score = statistics.median(all_scores) if all_scores else 0.0
        min_score = min(all_scores) if all_scores else 0.0
        max_score = max(all_scores) if all_scores else 0.0

        # Latency
        latencies = [e.latency_ms for e in self.evaluations if e.latency_ms]
        lat_mean = statistics.mean(latencies) if latencies else 0.0
        lat_median = statistics.median(latencies) if latencies else 0.0

        return EvaluationMetrics(
            total_evaluations=len(self.evaluations),
            total_fields=sum(len(e.field_matches) for e in self.evaluations),
            successful_extractions=sum(1 for e in self.evaluations if e.extraction_success),
            failed_extractions=sum(1 for e in self.evaluations if not e.extraction_success),
            total_true_positives=total_tp,
            total_false_positives=total_fp,
            total_false_negatives=total_fn,
            total_true_negatives=total_tn,
            micro_precision=micro_p,
            micro_recall=micro_r,
            micro_f1=micro_f1,
            micro_accuracy=micro_acc,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            macro_accuracy=macro_acc,
            mean_score=mean_score,
            median_score=median_score,
            min_score=min_score,
            max_score=max_score,
            latency_mean_ms=lat_mean,
            latency_median_ms=lat_median,
            total_tokens=sum(e.tokens_used or 0 for e in self.evaluations),
            total_cost_usd=sum(e.cost_usd or 0.0 for e in self.evaluations),
        )

    def to_text(self, include_details: bool = True) -> str:
        """Generate a plain text report.

        Args:
            include_details: Whether to include per-evaluation details.

        Returns:
            Plain text report string.
        """
        lines = [
            "=" * 70,
            f"  {self.title}",
            "=" * 70,
            f"  Generated: {datetime.now().isoformat()}",
            "",
            "-" * 70,
            "  SUMMARY",
            "-" * 70,
            f"  Total Evaluations:    {self.metrics.total_evaluations}",
            f"  Total Fields:         {self.metrics.total_fields}",
            f"  Successful:           {self.metrics.successful_extractions}",
            f"  Failed:               {self.metrics.failed_extractions}",
            "",
            "-" * 70,
            "  BINARY METRICS (Precision / Recall / F1 / Accuracy)",
            "-" * 70,
            "",
            "  Micro-averaged (global across all fields):",
            f"    Precision:          {self.metrics.micro_precision:.2%}",
            f"    Recall:             {self.metrics.micro_recall:.2%}",
            f"    F1 Score:           {self.metrics.micro_f1:.2%}",
            f"    Accuracy:           {self.metrics.micro_accuracy:.2%}",
            "",
            "  Macro-averaged (average per evaluation):",
            f"    Precision:          {self.metrics.macro_precision:.2%}",
            f"    Recall:             {self.metrics.macro_recall:.2%}",
            f"    F1 Score:           {self.metrics.macro_f1:.2%}",
            f"    Accuracy:           {self.metrics.macro_accuracy:.2%}",
            "",
            "-" * 70,
            "  CONFUSION MATRIX",
            "-" * 70,
            f"  True Positives:       {self.metrics.total_true_positives}",
            f"  False Positives:      {self.metrics.total_false_positives}",
            f"  False Negatives:      {self.metrics.total_false_negatives}",
            f"  True Negatives:       {self.metrics.total_true_negatives}",
            "",
            "-" * 70,
            "  SCORE METRICS (Continuous Similarity)",
            "-" * 70,
            f"  Mean Score:           {self.metrics.mean_score:.2%}",
            f"  Median Score:         {self.metrics.median_score:.2%}",
            f"  Min Score:            {self.metrics.min_score:.2%}",
            f"  Max Score:            {self.metrics.max_score:.2%}",
        ]

        if self.metrics.latency_mean_ms > 0:
            lines.extend([
                "",
                "-" * 70,
                "  PERFORMANCE",
                "-" * 70,
                f"  Mean Latency:         {self.metrics.latency_mean_ms:.0f} ms",
                f"  Median Latency:       {self.metrics.latency_median_ms:.0f} ms",
                f"  P95 Latency:          {self.metrics.latency_p95_ms:.0f} ms",
                f"  Total Tokens:         {self.metrics.total_tokens:,}",
                f"  Total Cost:           ${self.metrics.total_cost_usd:.4f}",
            ])

        if self.metrics.field_metrics and include_details:
            lines.extend([
                "",
                "-" * 70,
                "  PER-FIELD METRICS",
                "-" * 70,
            ])
            for field_name, fm in sorted(self.metrics.field_metrics.items()):
                lines.append(
                    f"  {field_name:20} P:{fm.precision:.0%} R:{fm.recall:.0%} "
                    f"F1:{fm.f1_score:.0%} Score:{fm.mean_score:.0%}"
                )

        if include_details and self.evaluations:
            lines.extend([
                "",
                "-" * 70,
                "  EVALUATION DETAILS",
                "-" * 70,
            ])
            for eval_result in self.evaluations:
                status = "✓" if eval_result.extraction_success else "✗"
                lines.append(
                    f"\n  [{status}] {eval_result.test_id} ({eval_result.schema_name})"
                )
                lines.append(
                    f"      P:{eval_result.precision:.0%} R:{eval_result.recall:.0%} "
                    f"F1:{eval_result.f1_score:.0%} Score:{eval_result.mean_score:.0%}"
                )
                for match in eval_result.field_matches:
                    match_status = "✓" if match.matched else "✗"
                    lines.append(
                        f"      {match_status} {match.field_name}: "
                        f"score={match.score:.2f} ({match.status.value})"
                    )

        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Generate a JSON-serializable report dictionary.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "title": self.title,
            "generated": datetime.now().isoformat(),
            "summary": {
                "total_evaluations": self.metrics.total_evaluations,
                "total_fields": self.metrics.total_fields,
                "successful_extractions": self.metrics.successful_extractions,
                "failed_extractions": self.metrics.failed_extractions,
            },
            "binary_metrics": {
                "micro": {
                    "precision": self.metrics.micro_precision,
                    "recall": self.metrics.micro_recall,
                    "f1_score": self.metrics.micro_f1,
                    "accuracy": self.metrics.micro_accuracy,
                },
                "macro": {
                    "precision": self.metrics.macro_precision,
                    "recall": self.metrics.macro_recall,
                    "f1_score": self.metrics.macro_f1,
                    "accuracy": self.metrics.macro_accuracy,
                },
            },
            "confusion_matrix": {
                "true_positives": self.metrics.total_true_positives,
                "false_positives": self.metrics.total_false_positives,
                "false_negatives": self.metrics.total_false_negatives,
                "true_negatives": self.metrics.total_true_negatives,
            },
            "score_metrics": {
                "mean": self.metrics.mean_score,
                "median": self.metrics.median_score,
                "min": self.metrics.min_score,
                "max": self.metrics.max_score,
                "std": self.metrics.std_score,
            },
            "performance": {
                "latency_mean_ms": self.metrics.latency_mean_ms,
                "latency_median_ms": self.metrics.latency_median_ms,
                "latency_p95_ms": self.metrics.latency_p95_ms,
                "latency_p99_ms": self.metrics.latency_p99_ms,
                "total_tokens": self.metrics.total_tokens,
                "total_cost_usd": self.metrics.total_cost_usd,
                "cached_count": self.metrics.cached_count,
            },
            "field_metrics": {
                name: {
                    "evaluation_count": fm.evaluation_count,
                    "precision": fm.precision,
                    "recall": fm.recall,
                    "f1_score": fm.f1_score,
                    "accuracy": fm.accuracy,
                    "mean_score": fm.mean_score,
                    "min_score": fm.min_score,
                    "max_score": fm.max_score,
                }
                for name, fm in self.metrics.field_metrics.items()
            },
            "evaluations": [
                {
                    "test_id": e.test_id,
                    "schema_name": e.schema_name,
                    "extraction_success": e.extraction_success,
                    "precision": e.precision,
                    "recall": e.recall,
                    "f1_score": e.f1_score,
                    "accuracy": e.accuracy,
                    "mean_score": e.mean_score,
                    "latency_ms": e.latency_ms,
                    "tokens_used": e.tokens_used,
                    "cost_usd": e.cost_usd,
                    "field_matches": [
                        {
                            "field_name": m.field_name,
                            "matched": m.matched,
                            "score": m.score,
                            "status": m.status.value,
                            "expected": self._serialize_value(m.expected),
                            "actual": self._serialize_value(m.actual),
                            "reasoning": m.reasoning,
                        }
                        for m in e.field_matches
                    ],
                }
                for e in self.evaluations
            ],
        }

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON output."""
        if hasattr(value, "model_dump"):
            return value.model_dump()
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

    def to_markdown(self, include_details: bool = True) -> str:
        """Generate a Markdown report.

        Args:
            include_details: Whether to include per-evaluation details.

        Returns:
            Markdown formatted report string.
        """
        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Evaluations | {self.metrics.total_evaluations} |",
            f"| Total Fields | {self.metrics.total_fields} |",
            f"| Successful | {self.metrics.successful_extractions} |",
            f"| Failed | {self.metrics.failed_extractions} |",
            "",
            "## Binary Metrics",
            "",
            "### Micro-averaged (global)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {self.metrics.micro_precision:.2%} |",
            f"| Recall | {self.metrics.micro_recall:.2%} |",
            f"| F1 Score | {self.metrics.micro_f1:.2%} |",
            f"| Accuracy | {self.metrics.micro_accuracy:.2%} |",
            "",
            "### Macro-averaged (per evaluation)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {self.metrics.macro_precision:.2%} |",
            f"| Recall | {self.metrics.macro_recall:.2%} |",
            f"| F1 Score | {self.metrics.macro_f1:.2%} |",
            f"| Accuracy | {self.metrics.macro_accuracy:.2%} |",
            "",
            "## Confusion Matrix",
            "",
            "| | Predicted Positive | Predicted Negative |",
            "|---|---|---|",
            f"| **Actual Positive** | TP: {self.metrics.total_true_positives} | "
            f"FN: {self.metrics.total_false_negatives} |",
            f"| **Actual Negative** | FP: {self.metrics.total_false_positives} | "
            f"TN: {self.metrics.total_true_negatives} |",
            "",
            "## Score Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Score | {self.metrics.mean_score:.2%} |",
            f"| Median Score | {self.metrics.median_score:.2%} |",
            f"| Min Score | {self.metrics.min_score:.2%} |",
            f"| Max Score | {self.metrics.max_score:.2%} |",
        ]

        if self.metrics.latency_mean_ms > 0:
            lines.extend([
                "",
                "## Performance",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean Latency | {self.metrics.latency_mean_ms:.0f} ms |",
                f"| Median Latency | {self.metrics.latency_median_ms:.0f} ms |",
                f"| P95 Latency | {self.metrics.latency_p95_ms:.0f} ms |",
                f"| Total Tokens | {self.metrics.total_tokens:,} |",
                f"| Total Cost | ${self.metrics.total_cost_usd:.4f} |",
            ])

        if self.metrics.field_metrics and include_details:
            lines.extend([
                "",
                "## Per-Field Metrics",
                "",
                "| Field | Precision | Recall | F1 | Mean Score |",
                "|-------|-----------|--------|----|-----------:|",
            ])
            for field_name, fm in sorted(self.metrics.field_metrics.items()):
                lines.append(
                    f"| {field_name} | {fm.precision:.0%} | {fm.recall:.0%} | "
                    f"{fm.f1_score:.0%} | {fm.mean_score:.0%} |"
                )

        if include_details and self.evaluations:
            lines.extend([
                "",
                "## Evaluation Details",
                "",
            ])
            for eval_result in self.evaluations:
                status = "✅" if eval_result.extraction_success else "❌"
                lines.extend([
                    f"### {status} {eval_result.test_id}",
                    "",
                    f"**Schema:** {eval_result.schema_name}",
                    "",
                    "| Precision | Recall | F1 | Mean Score |",
                    "|-----------|--------|----|-----------:|",
                    f"| {eval_result.precision:.0%} | {eval_result.recall:.0%} | "
                    f"{eval_result.f1_score:.0%} | {eval_result.mean_score:.0%} |",
                    "",
                    "| Field | Match | Score | Status | Reasoning |",
                    "|-------|-------|------:|--------|-----------|",
                ])
                for match in eval_result.field_matches:
                    match_icon = "✓" if match.matched else "✗"
                    reasoning = (match.reasoning or "")[:50]
                    lines.append(
                        f"| {match.field_name} | {match_icon} | {match.score:.2f} | "
                        f"{match.status.value} | {reasoning} |"
                    )
                lines.append("")

        return "\n".join(lines)

    def save(
        self,
        path: str | Path,
        format: str = "auto",
        include_details: bool = True,
    ) -> None:
        """Save the report to a file.

        Args:
            path: Output file path.
            format: Output format ("text", "json", "markdown", or "auto").
                   "auto" detects from file extension.
            include_details: Whether to include per-evaluation details.
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix in (".md", ".markdown"):
                format = "markdown"
            else:
                format = "text"

        # Generate content
        if format == "json":
            content = json.dumps(self.to_json(), indent=2, default=str)
        elif format == "markdown":
            content = self.to_markdown(include_details=include_details)
        else:
            content = self.to_text(include_details=include_details)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        path.write_text(content)

    def print_summary(self) -> None:
        """Print a concise summary to stdout."""
        print(f"\n{'=' * 50}")
        print(f"  {self.title}")
        print(f"{'=' * 50}")
        print(f"  Evaluations: {self.metrics.total_evaluations}")
        print(f"  Fields:      {self.metrics.total_fields}")
        print()
        print("  Binary Metrics (Micro):")
        print(f"    Precision: {self.metrics.micro_precision:.2%}")
        print(f"    Recall:    {self.metrics.micro_recall:.2%}")
        print(f"    F1 Score:  {self.metrics.micro_f1:.2%}")
        print(f"    Accuracy:  {self.metrics.micro_accuracy:.2%}")
        print()
        print("  Score Metrics:")
        print(f"    Mean:      {self.metrics.mean_score:.2%}")
        print(f"    Median:    {self.metrics.median_score:.2%}")
        print(f"{'=' * 50}\n")
