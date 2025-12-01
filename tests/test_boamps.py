"""Tests for BoAmps report generation integration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from structured_extractor import (
    BoAmpsReport,
    BoAmpsReporter,
    CumulativeTracking,
    DocumentExtractor,
)


class SimpleInvoice(BaseModel):
    """Simple invoice for testing."""

    invoice_number: str = Field(description="Invoice ID")
    total: float = Field(description="Total amount")


class TestCumulativeTracking:
    """Tests for cumulative tracking access."""

    def test_cumulative_tracking_property(self) -> None:
        """Test that cumulative_tracking property returns client tracking."""
        with patch("structured_extractor.core.extractor.OpenAIClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_tracking = CumulativeTracking(
                api_request_count=5,
                cached_request_count=3,
                api_energy_kwh=0.001,
                api_gwp_kgco2eq=0.0005,
                api_cost_usd=0.10,
                api_prompt_tokens=500,
                api_completion_tokens=200,
            )
            mock_client.cumulative_tracking = mock_tracking
            mock_client_cls.return_value = mock_client

            extractor = DocumentExtractor(api_key="test-key")
            tracking = extractor.cumulative_tracking

            assert tracking.api_request_count == 5
            assert tracking.cached_request_count == 3
            assert tracking.total_request_count == 8
            assert tracking.api_energy_kwh == 0.001
            assert tracking.api_cost_usd == 0.10

    def test_reset_cumulative_tracking(self) -> None:
        """Test resetting cumulative tracking."""
        with patch("structured_extractor.core.extractor.OpenAIClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            extractor = DocumentExtractor(api_key="test-key")
            extractor.reset_cumulative_tracking()

            mock_client.reset_cumulative_tracking.assert_called_once()


class TestBoAmpsExport:
    """Tests for BoAmps report export."""

    @pytest.fixture
    def mock_extractor(self) -> DocumentExtractor:
        """Create an extractor with mocked client."""
        with patch("structured_extractor.core.extractor.OpenAIClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.model = "gpt-4.1"
            mock_client.tracking_method = "ecologits"
            mock_client.electricity_mix_zone = "FRA"
            mock_client._get_provider_name.return_value = "openai"

            # Setup cumulative tracking
            mock_tracking = CumulativeTracking(
                api_request_count=10,
                cached_request_count=5,
                api_energy_kwh=0.005,
                api_gwp_kgco2eq=0.0025,
                api_cost_usd=0.50,
                api_prompt_tokens=1000,
                api_completion_tokens=500,
            )
            mock_client.cumulative_tracking = mock_tracking
            mock_client._cumulative_tracking = mock_tracking
            mock_client_cls.return_value = mock_client

            extractor = DocumentExtractor(api_key="test-key")
            return extractor

    def test_export_boamps_report_creates_file(self, mock_extractor: DocumentExtractor) -> None:
        """Test that export creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            mock_extractor.export_boamps_report(
                report_path,
                publisher_name="Test Org",
                include_summary=False,
            )

            assert report_path.exists()
            with open(report_path) as f:
                data = json.load(f)

            assert "header" in data
            assert "task" in data
            assert "measures" in data

    def test_export_boamps_report_with_publisher_info(
        self, mock_extractor: DocumentExtractor
    ) -> None:
        """Test export with publisher information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            report = mock_extractor.export_boamps_report(
                report_path,
                publisher_name="Test Organization",
                publisher_division="AI Team",
                project_name="Document Extraction",
                task_description="Invoice parsing pipeline",
                include_summary=False,
            )

            assert report.header.publisher is not None
            assert report.header.publisher.name == "Test Organization"
            assert report.header.publisher.division == "AI Team"
            assert report.header.publisher.projectName == "Document Extraction"

    def test_export_boamps_report_task_info(self, mock_extractor: DocumentExtractor) -> None:
        """Test that task info is populated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            report = mock_extractor.export_boamps_report(
                report_path,
                task_description="Test extraction",
                include_summary=False,
            )

            # Check task section
            assert report.task.taskStage == "inference"
            assert report.task.nbRequest == 15  # 10 API + 5 cached
            assert len(report.task.algorithms) == 1
            assert report.task.algorithms[0].foundationModelName == "gpt-4.1"
            assert report.task.taskDescription == "Test extraction"

    def test_export_boamps_report_measures(self, mock_extractor: DocumentExtractor) -> None:
        """Test that measures are populated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            report = mock_extractor.export_boamps_report(
                report_path,
                include_summary=False,
            )

            # Check measures
            assert len(report.measures) == 1
            assert report.measures[0].measurementMethod == "ecologits"
            assert report.measures[0].powerConsumption == pytest.approx(0.005)

    def test_export_boamps_report_with_summary(
        self, mock_extractor: DocumentExtractor, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test export with summary output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"

            mock_extractor.export_boamps_report(
                report_path,
                include_summary=True,
            )

            captured = capsys.readouterr()
            assert "BoAmps Report saved to:" in captured.out
            assert "Total requests: 15" in captured.out


class TestBoAmpsReporterDirectUse:
    """Tests for using BoAmpsReporter directly."""

    def test_reporter_with_extractor_client(self) -> None:
        """Test using BoAmpsReporter with extractor's internal client."""
        with patch("structured_extractor.core.extractor.OpenAIClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.model = "gpt-4.1"
            mock_client.tracking_method = "ecologits"
            mock_client.electricity_mix_zone = "WOR"
            mock_client._get_provider_name.return_value = "openai"

            mock_tracking = CumulativeTracking(
                api_request_count=3,
                cached_request_count=1,
                api_energy_kwh=0.002,
                api_gwp_kgco2eq=0.001,
                api_cost_usd=0.15,
                api_prompt_tokens=300,
                api_completion_tokens=100,
            )
            mock_client.cumulative_tracking = mock_tracking
            mock_client._cumulative_tracking = mock_tracking
            mock_client_cls.return_value = mock_client

            extractor = DocumentExtractor(api_key="test-key")

            # Use reporter directly with extractor's client
            reporter = BoAmpsReporter(
                client=extractor._client,
                publisher_name="Direct Test",
                quality="high",
            )

            report = reporter.generate_report()

            assert report.quality == "high"
            assert report.header.publisher is not None
            assert report.header.publisher.name == "Direct Test"
            assert report.task.nbRequest == 4


class TestBoAmpsReportModel:
    """Tests for BoAmpsReport model usage."""

    def test_report_to_json(self) -> None:
        """Test converting report to JSON."""
        from seeds_clients.tracking.boamps_reporter import Measure

        report = BoAmpsReport(
            measures=[
                Measure(
                    measurementMethod="ecologits",
                    powerConsumption=0.001,
                )
            ]
        )

        json_str = report.to_json()
        data = json.loads(json_str)

        assert "header" in data
        assert "measures" in data
        assert data["measures"][0]["powerConsumption"] == 0.001

    def test_report_save(self) -> None:
        """Test saving report to file."""
        from seeds_clients.tracking.boamps_reporter import Measure

        report = BoAmpsReport(
            measures=[
                Measure(
                    measurementMethod="ecologits",
                    powerConsumption=0.002,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_report.json"
            report.save(filepath)

            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["measures"][0]["powerConsumption"] == 0.002


class TestCumulativeTrackingModel:
    """Tests for CumulativeTracking model usage."""

    def test_cumulative_tracking_properties(self) -> None:
        """Test CumulativeTracking computed properties."""
        tracking = CumulativeTracking(
            api_request_count=10,
            cached_request_count=5,
            api_energy_kwh=0.01,
            api_gwp_kgco2eq=0.005,
            api_cost_usd=1.0,
            api_prompt_tokens=1000,
            api_completion_tokens=500,
        )

        assert tracking.total_request_count == 15
        assert tracking.total_prompt_tokens == 1000
        assert tracking.total_completion_tokens == 500

    def test_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        tracking = CumulativeTracking(
            api_request_count=6,
            cached_request_count=4,
            api_energy_kwh=0.01,
            api_gwp_kgco2eq=0.005,
            api_cost_usd=1.0,
            api_prompt_tokens=1000,
            api_completion_tokens=500,
        )

        # Cache hit rate = cached / total = 4 / 10 = 0.4
        assert tracking.cache_hit_rate == pytest.approx(0.4)
