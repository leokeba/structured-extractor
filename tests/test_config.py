"""Tests for configuration classes."""

import pytest
from pydantic import ValidationError

from structured_extractor.core.config import ExtractionConfig, FieldConfig


class TestFieldConfig:
    """Tests for FieldConfig class."""

    def test_default_values(self) -> None:
        """Test default field config values."""
        config = FieldConfig()

        assert config.extraction_hint is None
        assert config.required is True
        assert config.examples is None
        assert config.fallback_value is None

    def test_with_hint(self) -> None:
        """Test field config with extraction hint."""
        config = FieldConfig(
            extraction_hint="Look for ISO date format",
            required=True,
        )

        assert config.extraction_hint == "Look for ISO date format"
        assert config.required is True

    def test_with_examples(self) -> None:
        """Test field config with examples."""
        config = FieldConfig(
            examples=["2024-01-01", "2024-12-31"],
        )

        assert config.examples == ["2024-01-01", "2024-12-31"]


class TestExtractionConfig:
    """Tests for ExtractionConfig class."""

    def test_default_values(self) -> None:
        """Test default extraction config values."""
        config = ExtractionConfig()

        assert config.include_confidence is False
        assert config.confidence_threshold == 0.8
        assert config.max_retries == 3
        assert config.retry_on_validation_error is True
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.include_field_descriptions is True

    def test_custom_values(self) -> None:
        """Test extraction config with custom values."""
        config = ExtractionConfig(
            include_confidence=True,
            confidence_threshold=0.9,
            max_retries=5,
            temperature=0.2,
            system_prompt="Custom prompt",
        )

        assert config.include_confidence is True
        assert config.confidence_threshold == 0.9
        assert config.max_retries == 5
        assert config.temperature == 0.2
        assert config.system_prompt == "Custom prompt"

    def test_confidence_threshold_validation(self) -> None:
        """Test confidence threshold must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ExtractionConfig(confidence_threshold=1.5)

        with pytest.raises(ValidationError):
            ExtractionConfig(confidence_threshold=-0.1)

    def test_temperature_validation(self) -> None:
        """Test temperature must be between 0 and 2."""
        # Valid temperatures
        ExtractionConfig(temperature=0.0)
        ExtractionConfig(temperature=1.0)
        ExtractionConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValidationError):
            ExtractionConfig(temperature=2.5)

        with pytest.raises(ValidationError):
            ExtractionConfig(temperature=-0.1)

    def test_max_retries_validation(self) -> None:
        """Test max_retries must be at least 1."""
        ExtractionConfig(max_retries=1)
        ExtractionConfig(max_retries=10)

        with pytest.raises(ValidationError):
            ExtractionConfig(max_retries=0)
