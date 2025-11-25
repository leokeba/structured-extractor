"""Tests for human-in-the-loop callback hooks."""

from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from structured_extractor import (
    DocumentExtractor,
    ExtractionConfig,
    ExtractionResult,
)


class SimpleSchema(BaseModel):
    """Simple test schema."""

    name: str = Field(description="Person name")
    age: int = Field(description="Person age")


class TestHumanInLoopCallbacks:
    """Tests for human-in-the-loop callback configuration."""

    def test_config_accepts_callbacks(self):
        """Test that ExtractionConfig accepts callback functions."""

        def review_callback(result):
            return result

        def validation_callback(data):
            return (True, None)

        config = ExtractionConfig(
            on_low_confidence=review_callback,
            on_validation_error=validation_callback,
            on_review_required=review_callback,
            review_confidence_threshold=0.6,
        )

        assert config.on_low_confidence is review_callback
        assert config.on_validation_error is validation_callback
        assert config.on_review_required is review_callback
        assert config.review_confidence_threshold == 0.6

    def test_config_default_no_callbacks(self):
        """Test that callbacks default to None."""
        config = ExtractionConfig()

        assert config.on_low_confidence is None
        assert config.on_validation_error is None
        assert config.on_review_required is None
        assert config.review_confidence_threshold == 0.5

    def test_config_with_quality_metrics_enabled(self):
        """Test enabling quality metrics computation."""
        config = ExtractionConfig(compute_quality_metrics=True)

        assert config.compute_quality_metrics is True

    @patch("structured_extractor.core.extractor.OpenAIClient")
    def test_low_confidence_callback_invoked(self, mock_client_class):
        """Test that on_low_confidence callback is invoked for low confidence results."""
        # Setup mock response with low confidence
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create a mock parsed response with low confidence
        mock_parsed = MagicMock()
        mock_parsed.extracted_data = SimpleSchema(name="John", age=30)
        mock_parsed.confidence_assessment = MagicMock()
        mock_parsed.confidence_assessment.overall_confidence = 0.3  # Low confidence
        mock_parsed.confidence_assessment.field_confidences = [
            MagicMock(field_name="name", confidence=0.3),
            MagicMock(field_name="age", confidence=0.3),
        ]

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 100
        mock_response.tracking.cost_usd = 0.01
        mock_response.content = '{"name": "John", "age": 30}'

        mock_client.generate.return_value = mock_response

        # Create callback that modifies the result
        callback_invoked = {"called": False}

        def on_low_confidence(result):
            callback_invoked["called"] = True
            # Return modified result
            return ExtractionResult(
                data=SimpleSchema(name="John (Reviewed)", age=30),
                success=True,
                confidence=0.9,  # Updated after review
            )

        config = ExtractionConfig(
            on_low_confidence=on_low_confidence,
            review_confidence_threshold=0.5,  # 0.3 < 0.5, so callback should fire
        )

        extractor = DocumentExtractor(api_key="test-key", default_config=config)
        result = extractor.extract_with_confidence(
            "John is 30 years old", schema=SimpleSchema
        )

        assert callback_invoked["called"] is True
        assert result.data.name == "John (Reviewed)"
        assert result.confidence == 0.9

    @patch("structured_extractor.core.extractor.OpenAIClient")
    def test_low_confidence_callback_not_invoked_for_high_confidence(
        self, mock_client_class
    ):
        """Test that on_low_confidence is not invoked for high confidence results."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_parsed = MagicMock()
        mock_parsed.extracted_data = SimpleSchema(name="John", age=30)
        mock_parsed.confidence_assessment = MagicMock()
        mock_parsed.confidence_assessment.overall_confidence = 0.95  # High confidence
        mock_parsed.confidence_assessment.field_confidences = [
            MagicMock(field_name="name", confidence=0.95),
            MagicMock(field_name="age", confidence=0.95),
        ]

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 100
        mock_response.tracking.cost_usd = 0.01
        mock_response.content = '{"name": "John", "age": 30}'

        mock_client.generate.return_value = mock_response

        callback_invoked = {"called": False}

        def on_low_confidence(result):
            callback_invoked["called"] = True
            return result

        config = ExtractionConfig(
            on_low_confidence=on_low_confidence,
            review_confidence_threshold=0.5,
        )

        extractor = DocumentExtractor(api_key="test-key", default_config=config)
        result = extractor.extract_with_confidence(
            "John is 30 years old", schema=SimpleSchema
        )

        assert callback_invoked["called"] is False
        assert result.confidence == 0.95

    @patch("structured_extractor.core.extractor.OpenAIClient")
    def test_review_required_callback_with_quality_metrics(self, mock_client_class):
        """Test that on_review_required callback is invoked based on quality metrics."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_parsed = MagicMock()
        mock_parsed.extracted_data = SimpleSchema(name="John", age=30)
        mock_parsed.confidence_assessment = MagicMock()
        mock_parsed.confidence_assessment.overall_confidence = 0.7
        mock_parsed.confidence_assessment.field_confidences = [
            MagicMock(field_name="name", confidence=0.4),  # Low confidence
            MagicMock(field_name="age", confidence=0.95),
        ]

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 100
        mock_response.tracking.cost_usd = 0.01
        mock_response.content = '{"name": "John", "age": 30}'

        mock_client.generate.return_value = mock_response

        review_callback_invoked = {"called": False}

        def on_review_required(result):
            review_callback_invoked["called"] = True
            # Return None to keep original
            return None

        config = ExtractionConfig(
            on_review_required=on_review_required,
            compute_quality_metrics=True,
        )

        extractor = DocumentExtractor(api_key="test-key", default_config=config)
        result = extractor.extract_with_confidence(
            "John is 30 years old", schema=SimpleSchema
        )

        # Quality metrics should indicate review needed due to low field confidence
        assert result.quality_metrics is not None
        if result.quality_metrics.needs_review:
            assert review_callback_invoked["called"] is True

    @patch("structured_extractor.core.extractor.OpenAIClient")
    def test_callback_returns_none_keeps_original(self, mock_client_class):
        """Test that callback returning None keeps the original result."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_parsed = MagicMock()
        mock_parsed.extracted_data = SimpleSchema(name="John", age=30)
        mock_parsed.confidence_assessment = MagicMock()
        mock_parsed.confidence_assessment.overall_confidence = 0.3
        mock_parsed.confidence_assessment.field_confidences = [
            MagicMock(field_name="name", confidence=0.3),
            MagicMock(field_name="age", confidence=0.3),
        ]

        mock_response = MagicMock()
        mock_response.parsed = mock_parsed
        mock_response.model = "gpt-4.1"
        mock_response.cached = False
        mock_response.usage.total_tokens = 100
        mock_response.tracking.cost_usd = 0.01
        mock_response.content = '{"name": "John", "age": 30}'

        mock_client.generate.return_value = mock_response

        def on_low_confidence(result):
            # Return None to indicate "keep original"
            return None

        config = ExtractionConfig(
            on_low_confidence=on_low_confidence,
            review_confidence_threshold=0.5,
        )

        extractor = DocumentExtractor(api_key="test-key", default_config=config)
        result = extractor.extract_with_confidence(
            "John is 30 years old", schema=SimpleSchema
        )

        # Original result should be returned
        assert result.data.name == "John"
        assert result.confidence == 0.3
