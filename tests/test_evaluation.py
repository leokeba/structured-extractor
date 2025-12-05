"""Tests for the evaluation module."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from structured_extractor.evaluation import (
    # Comparators
    ComparisonResult,
    ContainsComparator,
    DateComparator,
    EvaluationReporter,
    ExactComparator,
    ExtractionEvaluation,
    # Evaluator
    ExtractionEvaluator,
    FieldMatch,
    ListComparator,
    # Types
    MatchStatus,
    NestedModelComparator,
    NumericComparator,
    StringComparator,
    get_default_comparator,
)

# ============================================================================
# Test Schemas
# ============================================================================


class SimpleInvoice(BaseModel):
    """Simple invoice for testing."""

    invoice_number: str = Field(description="Invoice ID")
    total_amount: float = Field(description="Total amount")
    vendor_name: str = Field(description="Vendor name")


class PersonInfo(BaseModel):
    """Person info for testing."""

    name: str = Field(description="Full name")
    age: int | None = Field(default=None, description="Age")
    email: str | None = Field(default=None, description="Email")


class NestedSchema(BaseModel):
    """Schema with nested structure."""

    title: str
    items: list[str]
    metadata: dict[str, str] | None = None


# ============================================================================
# ComparisonResult Tests
# ============================================================================


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid comparison result."""
        result = ComparisonResult(matched=True, score=0.95, reasoning="Close match")
        assert result.matched is True
        assert result.score == 0.95
        assert result.reasoning == "Close match"

    def test_score_validation(self):
        """Test that score must be between 0 and 1."""
        with pytest.raises(ValueError, match="Score must be between"):
            ComparisonResult(matched=True, score=1.5)

        with pytest.raises(ValueError, match="Score must be between"):
            ComparisonResult(matched=False, score=-0.1)

    def test_boundary_scores(self):
        """Test boundary values for score."""
        result_zero = ComparisonResult(matched=False, score=0.0)
        assert result_zero.score == 0.0

        result_one = ComparisonResult(matched=True, score=1.0)
        assert result_one.score == 1.0


# ============================================================================
# ExactComparator Tests
# ============================================================================


class TestExactComparator:
    """Tests for ExactComparator."""

    def test_exact_match_strings(self):
        """Test exact string match."""
        comp = ExactComparator()
        result = comp.compare("hello", "hello")
        assert result.matched is True
        assert result.score == 1.0

    def test_no_match_strings(self):
        """Test string mismatch."""
        comp = ExactComparator()
        result = comp.compare("hello", "world")
        assert result.matched is False
        assert result.score == 0.0

    def test_exact_match_numbers(self):
        """Test exact number match."""
        comp = ExactComparator()
        result = comp.compare(100, 100)
        assert result.matched is True
        assert result.score == 1.0

    def test_exact_match_lists(self):
        """Test exact list match."""
        comp = ExactComparator()
        result = comp.compare([1, 2, 3], [1, 2, 3])
        assert result.matched is True

    def test_no_match_different_types(self):
        """Test different types don't match."""
        comp = ExactComparator()
        result = comp.compare("100", 100)
        assert result.matched is False


# ============================================================================
# NumericComparator Tests
# ============================================================================


class TestNumericComparator:
    """Tests for NumericComparator."""

    def test_exact_numeric_match(self):
        """Test exact numeric match."""
        comp = NumericComparator()
        result = comp.compare(100.0, 100.0)
        assert result.matched is True
        assert result.score == 1.0

    def test_within_relative_tolerance(self):
        """Test match within relative tolerance."""
        comp = NumericComparator(threshold=0.05, mode="relative")  # 5%
        result = comp.compare(100.0, 103.0)  # 3% difference
        assert result.matched is True
        assert result.score > 0.95

    def test_exceeds_relative_tolerance(self):
        """Test mismatch exceeding relative tolerance."""
        comp = NumericComparator(threshold=0.01, mode="relative")  # 1%
        result = comp.compare(100.0, 110.0)  # 10% difference
        assert result.matched is False
        assert result.score < 0.95

    def test_absolute_tolerance(self):
        """Test absolute tolerance mode."""
        comp = NumericComparator(threshold=5.0, mode="absolute")
        result_match = comp.compare(100.0, 103.0)  # diff of 3
        result_nomatch = comp.compare(100.0, 110.0)  # diff of 10

        assert result_match.matched is True
        assert result_nomatch.matched is False

    def test_zero_expected_value(self):
        """Test comparison with zero expected value."""
        comp = NumericComparator(threshold=0.1)
        result = comp.compare(0, 0)
        assert result.matched is True
        assert result.score == 1.0

    def test_non_numeric_values(self):
        """Test with non-numeric values."""
        comp = NumericComparator()
        result = comp.compare("100", 100)
        assert result.matched is False
        assert result.score == 0.0

    def test_negative_threshold_raises(self):
        """Test that negative threshold raises error."""
        with pytest.raises(ValueError):
            NumericComparator(threshold=-0.1)


# ============================================================================
# StringComparator Tests
# ============================================================================


class TestStringComparator:
    """Tests for StringComparator."""

    def test_exact_string_match(self):
        """Test exact string match."""
        comp = StringComparator(threshold=0.9)
        result = comp.compare("hello world", "hello world")
        assert result.matched is True
        assert result.score == 1.0

    def test_similar_strings_above_threshold(self):
        """Test similar strings above threshold."""
        comp = StringComparator(threshold=0.8)
        result = comp.compare("hello world", "hello word")  # One char difference
        assert result.matched is True
        assert result.score > 0.8

    def test_different_strings_below_threshold(self):
        """Test different strings below threshold."""
        comp = StringComparator(threshold=0.9)
        result = comp.compare("hello", "goodbye")
        assert result.matched is False
        assert result.score < 0.9

    def test_case_insensitive(self):
        """Test case-insensitive comparison."""
        comp = StringComparator(case_sensitive=False)
        result = comp.compare("Hello World", "hello world")
        assert result.matched is True
        assert result.score == 1.0

    def test_case_sensitive(self):
        """Test case-sensitive comparison."""
        comp = StringComparator(case_sensitive=True)
        result = comp.compare("Hello World", "hello world")
        assert result.matched is False
        assert result.score < 1.0

    def test_whitespace_stripping(self):
        """Test whitespace stripping."""
        comp = StringComparator(strip_whitespace=True)
        result = comp.compare("  hello  ", "hello")
        assert result.matched is True
        assert result.score == 1.0

    def test_empty_strings(self):
        """Test empty string comparison."""
        comp = StringComparator()
        result = comp.compare("", "")
        assert result.matched is True
        assert result.score == 1.0

    def test_invalid_threshold(self):
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError):
            StringComparator(threshold=1.5)


# ============================================================================
# ContainsComparator Tests
# ============================================================================


class TestContainsComparator:
    """Tests for ContainsComparator."""

    def test_exact_match(self):
        """Test exact match."""
        comp = ContainsComparator()
        result = comp.compare("hello", "hello")
        assert result.matched is True
        assert result.score == 1.0

    def test_expected_in_actual(self):
        """Test expected is substring of actual."""
        comp = ContainsComparator()
        result = comp.compare("John", "John Smith")
        assert result.matched is True
        assert 0 < result.score < 1.0

    def test_actual_in_expected(self):
        """Test actual is substring of expected."""
        comp = ContainsComparator()
        result = comp.compare("John Smith", "John")
        assert result.matched is True
        assert 0 < result.score < 1.0

    def test_no_containment(self):
        """Test no substring relationship."""
        comp = ContainsComparator()
        result = comp.compare("hello", "world")
        assert result.matched is False
        assert result.score == 0.0

    def test_case_sensitivity(self):
        """Test case sensitivity option."""
        comp_insensitive = ContainsComparator(case_sensitive=False)
        comp_sensitive = ContainsComparator(case_sensitive=True)

        result_insensitive = comp_insensitive.compare("JOHN", "john smith")
        result_sensitive = comp_sensitive.compare("JOHN", "john smith")

        assert result_insensitive.matched is True
        assert result_sensitive.matched is False


# ============================================================================
# DateComparator Tests
# ============================================================================


class TestDateComparator:
    """Tests for DateComparator."""

    def test_same_date_same_format(self):
        """Test same date with same format."""
        comp = DateComparator()
        result = comp.compare("2024-01-15", "2024-01-15")
        assert result.matched is True
        assert result.score == 1.0

    def test_same_date_different_formats(self):
        """Test same date with different formats."""
        comp = DateComparator()
        result = comp.compare("2024-01-15", "15/01/2024")
        assert result.matched is True
        assert result.score == 1.0

    def test_different_dates(self):
        """Test different dates."""
        comp = DateComparator()
        result = comp.compare("2024-01-15", "2024-01-20")
        assert result.matched is False
        assert result.score < 1.0
        assert result.score > 0.0  # Within 30 days

    def test_invalid_date_format(self):
        """Test invalid date format."""
        comp = DateComparator()
        result = comp.compare("not-a-date", "2024-01-15")
        assert result.matched is False
        assert result.score == 0.0

    def test_custom_formats(self):
        """Test custom date formats."""
        comp = DateComparator(formats=["%d.%m.%Y"])
        result = comp.compare("15.01.2024", "15.01.2024")
        assert result.matched is True


# ============================================================================
# ListComparator Tests
# ============================================================================


class TestListComparator:
    """Tests for ListComparator."""

    def test_exact_list_match(self):
        """Test exact list match."""
        comp = ListComparator()
        result = comp.compare(["a", "b", "c"], ["a", "b", "c"])
        assert result.matched is True
        assert result.score == 1.0

    def test_same_items_different_order(self):
        """Test same items in different order (order insensitive)."""
        comp = ListComparator(order_sensitive=False)
        result = comp.compare(["a", "b", "c"], ["c", "b", "a"])
        assert result.matched is True
        assert result.score == 1.0

    def test_same_items_different_order_sensitive(self):
        """Test same items in different order (order sensitive)."""
        comp = ListComparator(order_sensitive=True)
        result = comp.compare(["a", "b", "c"], ["c", "b", "a"])
        assert result.matched is False
        assert result.score < 1.0

    def test_partial_match(self):
        """Test partial list match."""
        comp = ListComparator(match_threshold=0.5)
        result = comp.compare(["a", "b", "c", "d"], ["a", "b", "x", "y"])
        assert result.score >= 0.5

    def test_empty_lists(self):
        """Test empty list comparison."""
        comp = ListComparator()
        result = comp.compare([], [])
        assert result.matched is True
        assert result.score == 1.0

    def test_non_list_values(self):
        """Test with non-list values."""
        comp = ListComparator()
        result = comp.compare("not a list", ["a"])
        assert result.matched is False
        assert result.score == 0.0


# ============================================================================
# NestedModelComparator Tests
# ============================================================================


class TestNestedModelComparator:
    """Tests for NestedModelComparator."""

    def test_exact_dict_match(self):
        """Test exact dictionary match."""
        comp = NestedModelComparator()
        result = comp.compare({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert result.matched is True
        assert result.score == 1.0

    def test_partial_dict_match(self):
        """Test partial dictionary match."""
        comp = NestedModelComparator(match_threshold=0.5)
        result = comp.compare({"a": 1, "b": 2}, {"a": 1, "b": 999})
        assert result.score == 0.5  # One field matches

    def test_with_field_comparators(self):
        """Test with custom field comparators."""
        comp = NestedModelComparator(
            field_comparators={"amount": NumericComparator(threshold=0.1)},
            match_threshold=0.9,  # Allow 90% match for overall binary decision
        )
        result = comp.compare(
            {"name": "Test", "amount": 100.0},
            {"name": "Test", "amount": 105.0},  # 5% diff
        )
        assert result.matched is True

    def test_pydantic_model_comparison(self):
        """Test comparison of Pydantic models."""
        comp = NestedModelComparator()
        model1 = SimpleInvoice(invoice_number="INV-001", total_amount=100.0, vendor_name="Test")
        model2 = SimpleInvoice(invoice_number="INV-001", total_amount=100.0, vendor_name="Test")
        result = comp.compare(model1, model2)
        assert result.matched is True
        assert result.score == 1.0


# ============================================================================
# get_default_comparator Tests
# ============================================================================


class TestGetDefaultComparator:
    """Tests for get_default_comparator function."""

    def test_string_returns_string_comparator(self):
        """Test string value returns StringComparator."""
        comp = get_default_comparator("hello")
        assert isinstance(comp, StringComparator)

    def test_int_returns_numeric_comparator(self):
        """Test int value returns NumericComparator."""
        comp = get_default_comparator(42)
        assert isinstance(comp, NumericComparator)

    def test_float_returns_numeric_comparator(self):
        """Test float value returns NumericComparator."""
        comp = get_default_comparator(3.14)
        assert isinstance(comp, NumericComparator)

    def test_list_returns_list_comparator(self):
        """Test list value returns ListComparator."""
        comp = get_default_comparator([1, 2, 3])
        assert isinstance(comp, ListComparator)

    def test_dict_returns_nested_comparator(self):
        """Test dict value returns NestedModelComparator."""
        comp = get_default_comparator({"key": "value"})
        assert isinstance(comp, NestedModelComparator)

    def test_bool_returns_exact_comparator(self):
        """Test bool value returns ExactComparator."""
        comp = get_default_comparator(True)
        assert isinstance(comp, ExactComparator)


# ============================================================================
# FieldMatch Tests
# ============================================================================


class TestFieldMatch:
    """Tests for FieldMatch model."""

    def test_create_field_match(self):
        """Test creating a FieldMatch."""
        match = FieldMatch(
            field_name="invoice_number",
            matched=True,
            score=1.0,
            status=MatchStatus.TRUE_POSITIVE,
            expected="INV-001",
            actual="INV-001",
        )
        assert match.field_name == "invoice_number"
        assert match.matched is True
        assert match.status == MatchStatus.TRUE_POSITIVE

    def test_field_match_with_details(self):
        """Test FieldMatch with all optional fields."""
        match = FieldMatch(
            field_name="amount",
            matched=True,
            score=0.98,
            status=MatchStatus.TRUE_POSITIVE,
            expected=100.0,
            actual=101.0,
            threshold_used=0.05,
            comparator_type="NumericComparator",
            reasoning="Within 5% tolerance",
        )
        assert match.threshold_used == 0.05
        assert match.comparator_type == "NumericComparator"


# ============================================================================
# ExtractionEvaluation Tests
# ============================================================================


class TestExtractionEvaluation:
    """Tests for ExtractionEvaluation model."""

    def test_create_evaluation(self):
        """Test creating an ExtractionEvaluation."""
        field_matches = [
            FieldMatch(
                field_name="invoice_number",
                matched=True,
                score=1.0,
                status=MatchStatus.TRUE_POSITIVE,
                expected="INV-001",
                actual="INV-001",
            ),
            FieldMatch(
                field_name="total_amount",
                matched=False,
                score=0.5,
                status=MatchStatus.FALSE_NEGATIVE,
                expected=100.0,
                actual=50.0,
            ),
        ]

        evaluation = ExtractionEvaluation(
            test_id="test_001",
            schema_name="SimpleInvoice",
            extraction_success=True,
            field_matches=field_matches,
            true_positives=1,
            false_positives=0,
            false_negatives=1,
            true_negatives=0,
            precision=1.0,
            recall=0.5,
            f1_score=0.667,
            accuracy=0.5,
            mean_score=0.75,
            min_score=0.5,
            max_score=1.0,
        )

        assert evaluation.test_id == "test_001"
        assert evaluation.precision == 1.0
        assert evaluation.recall == 0.5
        assert len(evaluation.field_matches) == 2


# ============================================================================
# ExtractionEvaluator Tests
# ============================================================================


class TestExtractionEvaluator:
    """Tests for ExtractionEvaluator class."""

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock DocumentExtractor."""
        with patch("structured_extractor.evaluation.evaluator.DocumentExtractor") as mock:
            extractor = MagicMock()
            mock.return_value = extractor
            yield extractor

    def test_evaluate_without_extraction(self, mock_extractor):
        """Test evaluate_without_extraction method."""
        evaluator = ExtractionEvaluator(mock_extractor)

        extracted_data = {
            "invoice_number": "INV-001",
            "total_amount": 100.0,
            "vendor_name": "Test Corp",
        }
        ground_truth = {
            "invoice_number": "INV-001",
            "total_amount": 100.0,
            "vendor_name": "Test Corp",
        }

        result = evaluator.evaluate_without_extraction(
            extracted_data=extracted_data,
            ground_truth=ground_truth,
            schema=SimpleInvoice,
            test_id="test_001",
        )

        assert result.test_id == "test_001"
        assert result.extraction_success is True
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.mean_score == 1.0

    def test_evaluate_partial_match(self, mock_extractor):
        """Test evaluation with partial matches."""
        evaluator = ExtractionEvaluator(
            mock_extractor,
            default_numeric_tolerance=0.01,  # 1%
        )

        extracted_data = {
            "invoice_number": "INV-001",
            "total_amount": 105.0,  # 5% off
            "vendor_name": "Test Corporation",  # Different
        }
        ground_truth = {
            "invoice_number": "INV-001",
            "total_amount": 100.0,
            "vendor_name": "Test Corp",
        }

        result = evaluator.evaluate_without_extraction(
            extracted_data=extracted_data,
            ground_truth=ground_truth,
            schema=SimpleInvoice,
        )

        # Only invoice_number should match exactly
        assert result.true_positives >= 1
        assert result.false_negatives >= 1
        assert result.f1_score < 1.0

    def test_evaluate_with_custom_comparators(self, mock_extractor):
        """Test evaluation with custom field comparators."""
        evaluator = ExtractionEvaluator(mock_extractor)

        extracted_data = {
            "invoice_number": "INV-001",
            "total_amount": 105.0,  # 5% off
            "vendor_name": "Test Corp",
        }
        ground_truth = {
            "invoice_number": "INV-001",
            "total_amount": 100.0,
            "vendor_name": "Test Corp",
        }

        # Use lenient numeric comparator
        result = evaluator.evaluate_without_extraction(
            extracted_data=extracted_data,
            ground_truth=ground_truth,
            schema=SimpleInvoice,
            field_comparators={
                "total_amount": NumericComparator(threshold=0.1),  # 10% tolerance
            },
        )

        # All should match with lenient comparator
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_evaluate_missing_field(self, mock_extractor):
        """Test evaluation with missing extracted field."""
        evaluator = ExtractionEvaluator(mock_extractor)

        extracted_data = {
            "invoice_number": "INV-001",
            # total_amount is missing
            "vendor_name": "Test Corp",
        }
        ground_truth = {
            "invoice_number": "INV-001",
            "total_amount": 100.0,
            "vendor_name": "Test Corp",
        }

        result = evaluator.evaluate_without_extraction(
            extracted_data=extracted_data,
            ground_truth=ground_truth,
            schema=SimpleInvoice,
        )

        assert result.false_negatives >= 1
        assert result.recall < 1.0

    def test_compute_aggregate_metrics(self, mock_extractor):
        """Test computing aggregate metrics."""
        evaluator = ExtractionEvaluator(mock_extractor)

        # Add multiple evaluations
        for i in range(3):
            evaluator.evaluate_without_extraction(
                extracted_data={
                    "invoice_number": f"INV-00{i}",
                    "total_amount": 100.0,
                    "vendor_name": "Test Corp",
                },
                ground_truth={
                    "invoice_number": f"INV-00{i}",
                    "total_amount": 100.0,
                    "vendor_name": "Test Corp",
                },
                schema=SimpleInvoice,
            )

        metrics = evaluator.compute_aggregate_metrics()

        assert metrics.total_evaluations == 3
        assert metrics.total_fields == 9  # 3 fields x 3 evaluations
        assert metrics.micro_precision == 1.0
        assert metrics.micro_recall == 1.0
        assert metrics.micro_f1 == 1.0

    def test_clear_evaluations(self, mock_extractor):
        """Test clearing stored evaluations."""
        evaluator = ExtractionEvaluator(mock_extractor)

        evaluator.evaluate_without_extraction(
            extracted_data={
                "invoice_number": "INV-001",
                "total_amount": 100.0,
                "vendor_name": "Test",
            },
            ground_truth={
                "invoice_number": "INV-001",
                "total_amount": 100.0,
                "vendor_name": "Test",
            },
            schema=SimpleInvoice,
        )

        assert len(evaluator.evaluations) == 1

        evaluator.clear_evaluations()
        assert len(evaluator.evaluations) == 0


# ============================================================================
# EvaluationReporter Tests
# ============================================================================


class TestEvaluationReporter:
    """Tests for EvaluationReporter class."""

    @pytest.fixture
    def sample_evaluations(self):
        """Create sample evaluations for testing."""
        return [
            ExtractionEvaluation(
                test_id="test_001",
                schema_name="SimpleInvoice",
                extraction_success=True,
                field_matches=[
                    FieldMatch(
                        field_name="invoice_number",
                        matched=True,
                        score=1.0,
                        status=MatchStatus.TRUE_POSITIVE,
                        expected="INV-001",
                        actual="INV-001",
                    ),
                    FieldMatch(
                        field_name="total_amount",
                        matched=True,
                        score=0.95,
                        status=MatchStatus.TRUE_POSITIVE,
                        expected=100.0,
                        actual=102.0,
                    ),
                ],
                true_positives=2,
                false_positives=0,
                false_negatives=0,
                precision=1.0,
                recall=1.0,
                f1_score=1.0,
                accuracy=1.0,
                mean_score=0.975,
                min_score=0.95,
                max_score=1.0,
                latency_ms=150.0,
                tokens_used=500,
                cost_usd=0.001,
            ),
        ]

    def test_to_text(self, sample_evaluations):
        """Test generating text report."""
        reporter = EvaluationReporter(sample_evaluations)
        text = reporter.to_text()

        assert "Extraction Evaluation Report" in text
        assert "Precision" in text
        assert "Recall" in text
        assert "F1" in text

    def test_to_json(self, sample_evaluations):
        """Test generating JSON report."""
        reporter = EvaluationReporter(sample_evaluations)
        data = reporter.to_json()

        assert "title" in data
        assert "summary" in data
        assert "binary_metrics" in data
        assert "score_metrics" in data
        assert data["summary"]["total_evaluations"] == 1

    def test_to_markdown(self, sample_evaluations):
        """Test generating Markdown report."""
        reporter = EvaluationReporter(sample_evaluations)
        md = reporter.to_markdown()

        assert "# Extraction Evaluation Report" in md
        assert "## Binary Metrics" in md
        assert "| Precision |" in md

    def test_save_json(self, sample_evaluations, tmp_path):
        """Test saving JSON report."""
        reporter = EvaluationReporter(sample_evaluations)
        output_path = tmp_path / "report.json"
        reporter.save(output_path)

        assert output_path.exists()
        import json

        with open(output_path) as f:
            data = json.load(f)
        assert "evaluations" in data

    def test_save_markdown(self, sample_evaluations, tmp_path):
        """Test saving Markdown report."""
        reporter = EvaluationReporter(sample_evaluations)
        output_path = tmp_path / "report.md"
        reporter.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Extraction Evaluation Report" in content

    def test_save_text(self, sample_evaluations, tmp_path):
        """Test saving text report."""
        reporter = EvaluationReporter(sample_evaluations)
        output_path = tmp_path / "report.txt"
        reporter.save(output_path)

        assert output_path.exists()

    def test_print_summary(self, sample_evaluations, capsys):
        """Test printing summary."""
        reporter = EvaluationReporter(sample_evaluations)
        reporter.print_summary()

        captured = capsys.readouterr()
        assert "Extraction Evaluation Report" in captured.out
        assert "Precision" in captured.out


# ============================================================================
# Integration Tests
# ============================================================================


class TestEvaluationIntegration:
    """Integration tests for the evaluation module."""

    def test_full_workflow(self):
        """Test full evaluation workflow without actual extraction."""
        # Mock extractor
        mock_extractor = MagicMock()

        # Create evaluator
        evaluator = ExtractionEvaluator(
            mock_extractor,
            default_string_threshold=0.8,
            default_numeric_tolerance=0.05,
        )

        # Test cases
        test_data = [
            {
                "extracted": {
                    "invoice_number": "INV-001",
                    "total_amount": 100.0,
                    "vendor_name": "Acme Corp",
                },
                "ground_truth": {
                    "invoice_number": "INV-001",
                    "total_amount": 100.0,
                    "vendor_name": "Acme Corp",
                },
            },
            {
                "extracted": {
                    "invoice_number": "INV-002",
                    "total_amount": 205.0,  # 2.5% off from 200
                    "vendor_name": "Test Corporation",
                },
                "ground_truth": {
                    "invoice_number": "INV-002",
                    "total_amount": 200.0,
                    "vendor_name": "Test Corp",  # Different name
                },
            },
        ]

        # Run evaluations
        for i, data in enumerate(test_data):
            evaluator.evaluate_without_extraction(
                extracted_data=data["extracted"],
                ground_truth=data["ground_truth"],
                schema=SimpleInvoice,
                test_id=f"test_{i}",
            )

        # Get metrics
        metrics = evaluator.compute_aggregate_metrics()

        assert metrics.total_evaluations == 2
        assert metrics.total_fields == 6
        assert 0 < metrics.micro_f1 <= 1.0

        # Generate report
        reporter = EvaluationReporter(evaluator.evaluations, metrics)
        json_report = reporter.to_json()

        assert json_report["summary"]["total_evaluations"] == 2
        assert "binary_metrics" in json_report
        assert "score_metrics" in json_report
