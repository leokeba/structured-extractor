"""Field comparators for evaluation.

Each comparator outputs both:
- `matched`: Binary decision (for precision/recall/F1)
- `score`: Continuous similarity score 0.0-1.0 (for weighted metrics)

Comparators:
- ExactComparator: Exact equality
- NumericComparator: Numeric comparison with tolerance
- StringComparator: Fuzzy string matching (Levenshtein)
- ContainsComparator: Substring containment
- DateComparator: Date comparison with format normalization
- ListComparator: List comparison with item matching
- NestedModelComparator: Recursive comparison of nested structures
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class ComparisonResult:
    """Result of comparing expected vs actual value.

    Attributes:
        matched: Binary match decision (True/False)
        score: Continuous similarity score (0.0-1.0)
        threshold_used: The threshold used for binary decision (if applicable)
        reasoning: Human-readable explanation of the comparison
    """

    matched: bool
    score: float
    threshold_used: float | None = None
    reasoning: str | None = None

    def __post_init__(self) -> None:
        """Validate score is in valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


@runtime_checkable
class FieldComparator(Protocol):
    """Protocol for field comparison strategies.

    All comparators must implement this interface, returning both
    a binary match decision and a continuous similarity score.
    """

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare expected and actual values.

        Args:
            expected: The ground truth value
            actual: The extracted value

        Returns:
            ComparisonResult with both binary match and score
        """
        ...


class ExactComparator:
    """Exact equality comparison.

    Returns score of 1.0 for exact match, 0.0 otherwise.
    No threshold is applicable - match is always binary.

    Example:
        ```python
        comparator = ExactComparator()
        result = comparator.compare("hello", "hello")
        assert result.matched is True
        assert result.score == 1.0
        ```
    """

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare values for exact equality."""
        matched = expected == actual
        return ComparisonResult(
            matched=matched,
            score=1.0 if matched else 0.0,
            reasoning="Exact match" if matched else f"Expected {expected!r}, got {actual!r}",
        )


class NumericComparator:
    """Numeric comparison with configurable tolerance.

    Supports both relative (percentage) and absolute tolerance modes.
    Score decays linearly based on the difference.

    Example:
        ```python
        # 5% relative tolerance
        comparator = NumericComparator(threshold=0.05, mode="relative")
        result = comparator.compare(100.0, 103.0)  # 3% diff
        assert result.matched is True
        assert result.score > 0.95

        # Absolute tolerance
        comparator = NumericComparator(threshold=10.0, mode="absolute")
        result = comparator.compare(100.0, 105.0)  # diff of 5
        assert result.matched is True
        ```
    """

    def __init__(
        self,
        threshold: float = 0.01,
        mode: Literal["relative", "absolute"] = "relative",
    ) -> None:
        """Initialize numeric comparator.

        Args:
            threshold: Maximum allowed difference for binary match.
                - Relative mode: 0.01 = 1% difference allowed
                - Absolute mode: exact difference allowed (e.g., 10.0)
            mode: "relative" for percentage-based, "absolute" for fixed difference
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = threshold
        self.mode = mode

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare numeric values within tolerance."""
        # Type checking
        if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Non-numeric values: expected {type(expected).__name__}, "
                f"actual {type(actual).__name__}",
            )

        # Handle NaN
        if expected != expected or actual != actual:  # NaN check
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning="NaN values cannot be compared",
            )

        # Exact match (handles both zero)
        if expected == actual:
            return ComparisonResult(
                matched=True,
                score=1.0,
                threshold_used=self.threshold,
                reasoning="Exact numeric match",
            )

        # Calculate difference and score
        if self.mode == "relative":
            if expected == 0:
                # Can't compute relative diff when expected is 0
                # Fall back to absolute comparison with small threshold
                diff = abs(actual)
                matched = diff <= self.threshold
                score = max(0.0, 1.0 - diff) if diff <= 1 else 0.0
                reasoning = f"Expected 0, absolute diff: {diff:.6g}"
            else:
                diff = abs(actual - expected) / abs(expected)
                score = max(0.0, 1.0 - diff)
                matched = diff <= self.threshold
                cmp = "≤" if matched else ">"
                reasoning = f"Relative diff: {diff:.2%} ({cmp} {self.threshold:.2%})"
        else:  # absolute
            diff = abs(actual - expected)
            matched = diff <= self.threshold
            # Normalize score: 1.0 at diff=0, 0.0 at diff=2*threshold
            if self.threshold > 0:
                score = max(0.0, 1.0 - diff / (2 * self.threshold))
            else:
                score = 1.0 if diff == 0 else 0.0
            cmp = "≤" if matched else ">"
            reasoning = f"Absolute diff: {diff:.6g} ({cmp} {self.threshold:.6g})"

        return ComparisonResult(
            matched=matched,
            score=score,
            threshold_used=self.threshold,
            reasoning=reasoning,
        )


class StringComparator:
    """String comparison using Levenshtein similarity.

    Computes normalized edit distance and compares against threshold.
    Score is the raw similarity value (0.0-1.0).

    Example:
        ```python
        comparator = StringComparator(threshold=0.8)
        result = comparator.compare("hello world", "hello word")
        assert result.score > 0.9  # High similarity
        assert result.matched is True  # Above 80% threshold
        ```
    """

    def __init__(
        self,
        threshold: float = 0.9,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize string comparator.

        Args:
            threshold: Minimum similarity (0.0-1.0) for binary match.
            case_sensitive: Whether comparison is case-sensitive.
            strip_whitespace: Whether to strip leading/trailing whitespace.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare strings using Levenshtein similarity."""
        # Convert to strings if possible
        if not isinstance(expected, str):
            expected = str(expected) if expected is not None else ""
        if not isinstance(actual, str):
            actual = str(actual) if actual is not None else ""

        # Preprocess
        exp = expected.strip() if self.strip_whitespace else expected
        act = actual.strip() if self.strip_whitespace else actual

        if not self.case_sensitive:
            exp = exp.lower()
            act = act.lower()

        # Calculate similarity
        score = self._levenshtein_similarity(exp, act)
        matched = score >= self.threshold

        return ComparisonResult(
            matched=matched,
            score=score,
            threshold_used=self.threshold,
            reasoning=f"Similarity: {score:.1%} ({'≥' if matched else '<'} {self.threshold:.0%})",
        )

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity (0.0-1.0).

        Uses dynamic programming for efficient computation.
        """
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)

        # Optimize for very different length strings
        if abs(m - n) > max(m, n) * 0.5:
            return 0.0

        # DP table for edit distance
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + cost,  # substitution
                )

        distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len)


class ContainsComparator:
    """Substring containment comparison.

    Matches if either string contains the other as a substring.
    Score reflects the coverage ratio.

    Example:
        ```python
        comparator = ContainsComparator()
        result = comparator.compare("John", "John Smith")
        assert result.matched is True  # "John" in "John Smith"
        assert result.score < 1.0  # Partial coverage
        ```
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        """Initialize contains comparator.

        Args:
            case_sensitive: Whether comparison is case-sensitive.
        """
        self.case_sensitive = case_sensitive

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Check if one string contains the other."""
        # Convert to strings
        if not isinstance(expected, str):
            expected = str(expected) if expected is not None else ""
        if not isinstance(actual, str):
            actual = str(actual) if actual is not None else ""

        exp = expected if self.case_sensitive else expected.lower()
        act = actual if self.case_sensitive else actual.lower()

        # Check exact match first
        if exp == act:
            return ComparisonResult(
                matched=True,
                score=1.0,
                reasoning="Exact match",
            )

        # Check containment
        if exp and act:
            if exp in act:
                # Expected is substring of actual
                score = len(exp) / len(act)
                return ComparisonResult(
                    matched=True,
                    score=score,
                    reasoning=f"Expected found in actual (coverage: {score:.1%})",
                )
            elif act in exp:
                # Actual is substring of expected
                score = len(act) / len(exp)
                return ComparisonResult(
                    matched=True,
                    score=score,
                    reasoning=f"Actual found in expected (coverage: {score:.1%})",
                )

        return ComparisonResult(
            matched=False,
            score=0.0,
            reasoning="No substring match found",
        )


class DateComparator:
    """Date comparison with format normalization.

    Parses dates in various formats and compares the underlying dates.
    Score decays based on the number of days difference.

    Example:
        ```python
        comparator = DateComparator()
        result = comparator.compare("2024-01-15", "January 15, 2024")
        assert result.matched is True  # Same date, different format
        assert result.score == 1.0
        ```
    """

    DEFAULT_FORMATS = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%d %b %Y",
        "%Y%m%d",
        "%d.%m.%Y",
    ]

    def __init__(
        self,
        formats: list[str] | None = None,
        max_days_diff: int = 30,
    ) -> None:
        """Initialize date comparator.

        Args:
            formats: Date formats to try when parsing. Uses defaults if None.
            max_days_diff: Maximum days difference for score decay calculation.
        """
        self.formats = formats or self.DEFAULT_FORMATS
        self.max_days_diff = max_days_diff

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare dates after normalizing formats."""
        exp_date = self._parse_date(str(expected))
        act_date = self._parse_date(str(actual))

        if exp_date is None:
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Could not parse expected date: {expected!r}",
            )

        if act_date is None:
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Could not parse actual date: {actual!r}",
            )

        if exp_date == act_date:
            return ComparisonResult(
                matched=True,
                score=1.0,
                reasoning=f"Dates match: {exp_date}",
            )

        # Calculate days difference and score
        days_diff = abs((exp_date - act_date).days)
        score = max(0.0, 1.0 - days_diff / self.max_days_diff)

        return ComparisonResult(
            matched=False,
            score=score,
            reasoning=f"Dates differ by {days_diff} days: {exp_date} vs {act_date}",
        )

    def _parse_date(self, value: str) -> datetime | None:
        """Try to parse a date string using configured formats."""
        value = value.strip()
        for fmt in self.formats:
            try:
                return datetime.strptime(value, fmt).date()  # type: ignore[return-value]
            except ValueError:
                continue
        return None


class ListComparator:
    """List comparison with item-level matching.

    Supports both order-sensitive and order-insensitive comparison.
    Score reflects the ratio of matched items.

    Example:
        ```python
        comparator = ListComparator(order_sensitive=False, match_threshold=0.8)
        result = comparator.compare(["a", "b", "c"], ["c", "b", "a"])
        assert result.matched is True  # All items match
        assert result.score == 1.0
        ```
    """

    def __init__(
        self,
        item_comparator: FieldComparator | None = None,
        order_sensitive: bool = False,
        match_threshold: float = 1.0,
    ) -> None:
        """Initialize list comparator.

        Args:
            item_comparator: Comparator for individual items. Defaults to ExactComparator.
            order_sensitive: Whether item order matters.
            match_threshold: Minimum match ratio (0.0-1.0) for binary match.
        """
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be between 0.0 and 1.0")
        self.item_comparator = item_comparator or ExactComparator()
        self.order_sensitive = order_sensitive
        self.match_threshold = match_threshold

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare lists with configurable matching strategy."""
        # Type checking
        if not isinstance(expected, (list, tuple)):
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Expected is not a list: {type(expected).__name__}",
            )
        if not isinstance(actual, (list, tuple)):
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Actual is not a list: {type(actual).__name__}",
            )

        expected = list(expected)
        actual = list(actual)

        # Handle empty lists
        if len(expected) == 0 and len(actual) == 0:
            return ComparisonResult(
                matched=True,
                score=1.0,
                reasoning="Both lists empty",
            )

        if len(expected) == 0 or len(actual) == 0:
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"List length mismatch: expected {len(expected)}, actual {len(actual)}",
            )

        # Compare items
        if self.order_sensitive:
            score = self._compare_ordered(expected, actual)
        else:
            score = self._compare_unordered(expected, actual)

        matched = score >= self.match_threshold

        return ComparisonResult(
            matched=matched,
            score=score,
            threshold_used=self.match_threshold,
            reasoning=f"List match ratio: {score:.1%} ({'≥' if matched else '<'} "
            f"{self.match_threshold:.0%})",
        )

    def _compare_ordered(self, expected: list, actual: list) -> float:
        """Compare lists in order."""
        scores = []
        min_len = min(len(expected), len(actual))

        for i in range(min_len):
            result = self.item_comparator.compare(expected[i], actual[i])
            scores.append(result.score)

        if not scores:
            return 0.0

        # Penalize length mismatch
        max_len = max(len(expected), len(actual))
        length_penalty = min_len / max_len
        avg_score = sum(scores) / len(scores)

        return avg_score * length_penalty

    def _compare_unordered(self, expected: list, actual: list) -> float:
        """Compare lists without regard to order (greedy best-match)."""
        scores = []
        remaining_indices = list(range(len(actual)))

        for exp_item in expected:
            best_score = 0.0
            best_idx = -1

            for i in remaining_indices:
                result = self.item_comparator.compare(exp_item, actual[i])
                if result.score > best_score:
                    best_score = result.score
                    best_idx = i

            scores.append(best_score)
            if best_idx >= 0 and best_score > 0:
                remaining_indices.remove(best_idx)

        # Account for all items in both lists
        total_items = max(len(expected), len(actual))
        return sum(scores) / total_items if total_items > 0 else 0.0


class NestedModelComparator:
    """Compare nested Pydantic models or dictionaries field by field.

    Recursively compares nested structures using configurable comparators
    for each field.

    Example:
        ```python
        comparator = NestedModelComparator(
            field_comparators={"amount": NumericComparator(threshold=0.01)}
        )
        result = comparator.compare(
            {"name": "Test", "amount": 100.0},
            {"name": "Test", "amount": 100.5}
        )
        assert result.matched is True  # amount within tolerance
        ```
    """

    def __init__(
        self,
        field_comparators: dict[str, FieldComparator] | None = None,
        default_comparator: FieldComparator | None = None,
        match_threshold: float = 1.0,
    ) -> None:
        """Initialize nested model comparator.

        Args:
            field_comparators: Specific comparators for named fields.
            default_comparator: Fallback comparator. Defaults to ExactComparator.
            match_threshold: Minimum average score for binary match.
        """
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be between 0.0 and 1.0")
        self.field_comparators = field_comparators or {}
        self.default_comparator = default_comparator or ExactComparator()
        self.match_threshold = match_threshold

    def compare(self, expected: Any, actual: Any) -> ComparisonResult:
        """Compare nested structures field by field."""
        # Convert Pydantic models to dicts
        if hasattr(expected, "model_dump"):
            expected = expected.model_dump()
        if hasattr(actual, "model_dump"):
            actual = actual.model_dump()

        # Type checking
        if not isinstance(expected, dict):
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Expected is not a dict: {type(expected).__name__}",
            )
        if not isinstance(actual, dict):
            return ComparisonResult(
                matched=False,
                score=0.0,
                reasoning=f"Actual is not a dict: {type(actual).__name__}",
            )

        # Get all keys
        all_keys = set(expected.keys()) | set(actual.keys())

        if not all_keys:
            return ComparisonResult(
                matched=True,
                score=1.0,
                reasoning="Both dicts empty",
            )

        # Compare each field
        scores = []
        details = []

        for key in sorted(all_keys):
            exp_val = expected.get(key)
            act_val = actual.get(key)

            # Handle missing keys
            if key not in expected:
                scores.append(0.0)
                details.append(f"{key}: missing in expected")
                continue
            if key not in actual:
                scores.append(0.0)
                details.append(f"{key}: missing in actual")
                continue

            # Both None is a match
            if exp_val is None and act_val is None:
                scores.append(1.0)
                details.append(f"{key}: both None")
                continue

            # One None is a mismatch
            if exp_val is None or act_val is None:
                scores.append(0.0)
                details.append(f"{key}: None mismatch")
                continue

            # Compare values
            comparator = self.field_comparators.get(key, self.default_comparator)
            result = comparator.compare(exp_val, act_val)
            scores.append(result.score)
            details.append(f"{key}: {result.score:.2f}")

        score = sum(scores) / len(scores)
        matched = score >= self.match_threshold

        return ComparisonResult(
            matched=matched,
            score=score,
            threshold_used=self.match_threshold,
            reasoning=f"Nested match: {score:.1%} across {len(all_keys)} fields",
        )


# Convenience function to get default comparator based on type
def get_default_comparator(
    value: Any,
    string_threshold: float = 0.9,
    numeric_threshold: float = 0.01,
) -> FieldComparator:
    """Get a sensible default comparator based on value type.

    Args:
        value: The value to determine comparator for
        string_threshold: Threshold for string comparator
        numeric_threshold: Threshold for numeric comparator

    Returns:
        Appropriate FieldComparator for the value type
    """
    if isinstance(value, bool):
        return ExactComparator()
    elif isinstance(value, (int, float)):
        return NumericComparator(threshold=numeric_threshold)
    elif isinstance(value, str):
        return StringComparator(threshold=string_threshold)
    elif isinstance(value, (list, tuple)):
        return ListComparator()
    elif isinstance(value, dict) or hasattr(value, "model_dump"):
        return NestedModelComparator()
    else:
        return ExactComparator()
