"""Main document extractor class."""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from PIL import Image
from pydantic import BaseModel, ValidationError
from seeds_clients import Message, OpenAIClient
from seeds_clients.core.base_client import BaseClient
from seeds_clients.core.types import CumulativeTracking

from structured_extractor.core.config import ExtractionConfig
from structured_extractor.core.exceptions import (
    ConfigurationError,
    ExtractionError,
    ExtractionValidationError,
    LLMError,
)
from structured_extractor.core.templates import DocumentTemplate
from structured_extractor.prompts.builder import PromptBuilder
from structured_extractor.results.types import ExtractionResult

if TYPE_CHECKING:
    from seeds_clients.tracking.boamps_reporter import BoAmpsReport

T = TypeVar("T", bound=BaseModel)

# Type alias for image inputs
ImageInput = str | Path | Image.Image | bytes

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """LLM-driven structured data extractor for documents.

    Uses seeds-clients for LLM integration and Pydantic for schema validation.
    Supports any client from seeds-clients (OpenAI, Anthropic, Google, OpenRouter, etc.).

    Example:
        ```python
        from pydantic import BaseModel, Field
        from structured_extractor import DocumentExtractor

        class Invoice(BaseModel):
            invoice_number: str
            total_amount: float
            vendor_name: str

        # Using default OpenAI client
        extractor = DocumentExtractor(model="gpt-4.1")
        result = extractor.extract(document_text, schema=Invoice)
        print(result.data.invoice_number)

        # Using Anthropic client
        from seeds_clients import AnthropicClient
        client = AnthropicClient(model="claude-sonnet-4-20250514")
        extractor = DocumentExtractor(client=client)
        result = extractor.extract(document_text, schema=Invoice)
        ```
    """

    _client: BaseClient

    def __init__(
        self,
        client: BaseClient | None = None,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        cache_dir: str = "cache",
        cache_ttl_hours: float | None = 24.0,
        default_config: ExtractionConfig | None = None,
    ) -> None:
        """Initialize the document extractor.

        Args:
            client: Pre-configured LLM client from seeds-clients. If provided,
                api_key, model, cache_dir, and cache_ttl_hours are ignored.
                Supports any client: OpenAIClient, AnthropicClient, GoogleClient,
                OpenRouterClient, ModelGardenClient.
            api_key: OpenAI API key. Only used if client is not provided.
                If not provided, uses OPENAI_API_KEY env var.
            model: LLM model to use. Only used if client is not provided.
            cache_dir: Directory for caching LLM responses. Only used if client is not provided.
            cache_ttl_hours: Cache TTL in hours. Only used if client is not provided.
            default_config: Default extraction configuration.
        """
        self.default_config = default_config or ExtractionConfig()

        if client is not None:
            # Use provided client (any BaseClient implementation)
            self._client = client
            self.model = client.model
        else:
            # Backwards-compatible: create OpenAIClient with provided parameters
            self.model = model
            self._client = OpenAIClient(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                model=model,
                cache_dir=cache_dir,
                ttl_hours=cache_ttl_hours,
            )

        # Initialize prompt builder
        self._prompt_builder = PromptBuilder(
            include_field_descriptions=self.default_config.include_field_descriptions,
        )

    def extract(
        self,
        document: str,
        schema: type[T] | None = None,
        template: DocumentTemplate | None = None,
        config: ExtractionConfig | None = None,
        field_hints: dict[str, str] | None = None,
        use_cache: bool | None = None,
    ) -> ExtractionResult[T]:
        """Extract structured data from a document.

        Args:
            document: The document text to extract from.
            schema: Pydantic model defining the extraction schema.
            template: Pre-configured document template (alternative to schema).
            config: Extraction configuration (overrides default).
            field_hints: Additional hints for specific fields.

        Returns:
            ExtractionResult containing the extracted data and metadata.

        Raises:
            ConfigurationError: If neither schema nor template is provided.
        """
        # Resolve schema and config from template if provided
        if template is not None:
            resolved_schema: type[BaseModel] = template.schema_class
            resolved_config = config or self.default_config
            resolved_hints = {**(template.field_configs or {}), **(field_hints or {})}
            system_prompt = template.system_prompt
            examples = template.examples
        elif schema is not None:
            resolved_schema = schema
            resolved_config = config or self.default_config
            resolved_hints = field_hints or {}
            system_prompt = resolved_config.system_prompt
            examples = None
        else:
            raise ConfigurationError("Either 'schema' or 'template' must be provided")

        # Build prompts
        system_message = self._prompt_builder.build_system_prompt(system_prompt)
        user_message = self._prompt_builder.build_extraction_prompt(
            document=document,
            schema=resolved_schema,
            field_hints=resolved_hints,
            examples=examples,
        )

        # Prepare messages
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=user_message),
        ]

        # Prepare LLM kwargs
        llm_kwargs: dict[str, Any] = {
            "temperature": resolved_config.temperature,
        }
        if resolved_config.max_tokens:
            llm_kwargs["max_tokens"] = resolved_config.max_tokens

        cache_enabled = resolved_config.use_cache if use_cache is None else use_cache

        # Call LLM with retry logic
        response = self._call_llm_with_retry(
            messages=messages,
            schema=resolved_schema,
            config=resolved_config,
            use_cache=use_cache,
            **llm_kwargs,
        )

        return ExtractionResult(
            data=response.parsed,  # type: ignore[arg-type]
            model_used=response.model,
            cached=response.cached,
            tokens_used=response.usage.total_tokens,
            cost_usd=response.tracking.cost_usd if response.tracking else None,
            raw_response=response.content,
        )

    def extract_with_confidence(
        self,
        document: str,
        schema: type[T],
        config: ExtractionConfig | None = None,
        field_hints: dict[str, str] | None = None,
        use_cache: bool | None = None,
    ) -> ExtractionResult[T]:
        """Extract with confidence scoring for each field.

        This method asks the LLM to self-assess its confidence in each
        extracted field, returning both the data and confidence scores.

        If configured, this method will also:
        - Compute quality metrics for the extraction
        - Invoke human-in-the-loop callbacks for low confidence results

        Args:
            document: The document text to extract from.
            schema: Pydantic model defining the extraction schema.
            config: Extraction configuration.
            field_hints: Additional hints for specific fields.

        Returns:
            ExtractionResult with confidence scores populated.

        Example:
            ```python
            result = extractor.extract_with_confidence(doc, schema=Invoice)
            print(f"Overall confidence: {result.confidence}")
            print(f"Low confidence fields: {result.low_confidence_fields}")
            for field, conf in result.field_confidences.items():
                print(f"  {field}: {conf:.2f}")
            ```
        """
        from structured_extractor.results.confidence import (
            ConfidenceAssessment,
            build_confidence_schema,
            compute_quality_metrics,
            identify_low_confidence_fields,
        )

        effective_config = config or self.default_config
        resolved_hints = field_hints or {}

        # Build the confidence-wrapped schema
        confidence_schema = build_confidence_schema(schema)

        # Build prompts with confidence instructions
        system_prompt = self._build_confidence_system_prompt(effective_config.system_prompt)
        user_message = self._prompt_builder.build_extraction_prompt(
            document=document,
            schema=schema,
            field_hints=resolved_hints,
        )

        # Add confidence assessment instructions to the user prompt
        confidence_instructions = self._build_confidence_instructions(schema)
        user_message = f"{user_message}\n\n{confidence_instructions}"

        # Prepare messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ]

        # Prepare LLM kwargs
        llm_kwargs: dict[str, Any] = {
            "temperature": effective_config.temperature,
        }
        if effective_config.max_tokens:
            llm_kwargs["max_tokens"] = effective_config.max_tokens

        cache_enabled = effective_config.use_cache if use_cache is None else use_cache

        # Call LLM with retry logic
        response = self._call_llm_with_retry(
            messages=messages,
            schema=confidence_schema,
            config=effective_config,
            use_cache=use_cache,
            **llm_kwargs,
        )

        # Extract the data and confidence from the response
        parsed = response.parsed
        extracted_data = parsed.extracted_data
        confidence_assessment: ConfidenceAssessment = parsed.confidence_assessment

        # Convert field confidences to dict
        field_confidences = {
            fc.field_name: fc.confidence for fc in confidence_assessment.field_confidences
        }

        # Identify low confidence fields
        low_conf_fields = identify_low_confidence_fields(
            field_confidences,
            threshold=effective_config.confidence_threshold,
        )

        # Compute quality metrics if enabled
        quality_metrics = None
        if effective_config.compute_quality_metrics:
            quality_metrics = compute_quality_metrics(
                data=extracted_data,
                schema=schema,
                field_confidences=field_confidences,
                confidence_threshold=effective_config.confidence_threshold,
            )

        result: ExtractionResult[T] = ExtractionResult(
            data=extracted_data,
            confidence=confidence_assessment.overall_confidence,
            field_confidences=field_confidences,
            low_confidence_fields=low_conf_fields if low_conf_fields else None,
            quality_metrics=quality_metrics,
            model_used=response.model,
            cached=response.cached,
            tokens_used=response.usage.total_tokens,
            cost_usd=response.tracking.cost_usd if response.tracking else None,
            raw_response=response.content,
        )

        # Invoke human-in-the-loop callbacks
        result = self._invoke_callbacks(result, schema, effective_config)

        return result

    def _invoke_callbacks(
        self,
        result: ExtractionResult[T],
        schema: type[T],
        config: ExtractionConfig,
    ) -> ExtractionResult[T]:
        """Invoke human-in-the-loop callbacks based on result and config.

        Args:
            result: The extraction result.
            schema: The extraction schema.
            config: The extraction configuration.

        Returns:
            Potentially modified extraction result.
        """
        # Check for low confidence callback
        if (
            config.on_low_confidence
            and result.confidence is not None
            and result.confidence < config.review_confidence_threshold
        ):
            modified = config.on_low_confidence(result)
            if modified is not None:
                return modified

        # Check for review required callback (based on quality metrics)
        if (
            config.on_review_required
            and result.quality_metrics is not None
            and result.quality_metrics.needs_review
        ):
            modified = config.on_review_required(result)
            if modified is not None:
                return modified

        return result

    def _build_confidence_system_prompt(self, custom_prompt: str | None = None) -> str:
        """Build system prompt with confidence assessment instructions."""
        base_prompt = custom_prompt or self._prompt_builder.build_system_prompt()
        confidence_addition = (
            "\n\nIMPORTANT: For this extraction, you must also assess your confidence "
            "in each extracted field. Rate your confidence from 0.0 (no confidence, "
            "guessing) to 1.0 (completely certain the value is correct). Consider:\n"
            "- 1.0: Value is explicitly and clearly stated in the document\n"
            "- 0.8-0.9: Value is present but requires minor interpretation\n"
            "- 0.6-0.7: Value is inferred from context or partially visible\n"
            "- 0.4-0.5: Value is a reasonable guess based on limited information\n"
            "- 0.0-0.3: Value is largely unknown or fabricated"
        )
        return base_prompt + confidence_addition

    def _build_confidence_instructions(self, schema: type[BaseModel]) -> str:
        """Build instructions for confidence assessment."""
        field_names = list(schema.model_fields.keys())
        fields_list = ", ".join(f"'{f}'" for f in field_names)

        return (
            "## Confidence Assessment Required\n\n"
            "After extracting the data, provide a confidence assessment:\n"
            f"- Assess confidence for each field: {fields_list}\n"
            "- Provide an overall_confidence score (0.0-1.0)\n"
            "- Be honest about uncertainty - lower confidence is better than wrong data\n"
            "- If a field value is not found in the document, confidence should be 0.0-0.3"
        )

    def extract_from_image(
        self,
        image: ImageInput | Sequence[ImageInput],
        schema: type[T] | None = None,
        template: DocumentTemplate | None = None,
        config: ExtractionConfig | None = None,
        field_hints: dict[str, str] | None = None,
        additional_context: str | None = None,
        use_cache: bool | None = None,
    ) -> ExtractionResult[T]:
        """Extract structured data from an image or images.

        Uses GPT-4.1's vision capabilities to extract structured data
        directly from images of documents, forms, invoices, etc.

        Args:
            image: Image to extract from. Can be:
                - A file path (str or Path)
                - A PIL Image object
                - Raw bytes
                - A URL string
                - A list of any of the above for multi-image extraction
            schema: Pydantic model defining the extraction schema.
            template: Pre-configured document template (alternative to schema).
            config: Extraction configuration (overrides default).
            field_hints: Additional hints for specific fields.
            additional_context: Optional text context to help extraction.

        Returns:
            ExtractionResult containing the extracted data and metadata.

        Raises:
            ConfigurationError: If neither schema nor template is provided.

        Example:
            ```python
            from pathlib import Path
            from pydantic import BaseModel

            class InvoiceData(BaseModel):
                invoice_number: str
                total: float
                vendor: str

            extractor = DocumentExtractor(model="gpt-4.1")
            result = extractor.extract_from_image(
                Path("invoice.png"),
                schema=InvoiceData
            )
            print(result.data.invoice_number)
            ```
        """
        # Resolve schema and config from template if provided
        if template is not None:
            resolved_schema: type[BaseModel] = template.schema_class
            resolved_config = config or self.default_config
            resolved_hints = {**(template.field_configs or {}), **(field_hints or {})}
            system_prompt = template.system_prompt
            examples = template.examples
        elif schema is not None:
            resolved_schema = schema
            resolved_config = config or self.default_config
            resolved_hints = field_hints or {}
            system_prompt = resolved_config.system_prompt
            examples = None
        else:
            raise ConfigurationError("Either 'schema' or 'template' must be provided")

        # Build prompts - use empty string for document since we're using images
        system_message = self._prompt_builder.build_system_prompt(system_prompt)

        # Build extraction prompt without document content
        schema_description = self._prompt_builder._describe_schema(resolved_schema, resolved_hints)

        prompt_parts = [
            "## Extraction Schema",
            "",
            schema_description,
            "",
        ]

        if additional_context:
            prompt_parts.extend(
                [
                    "## Additional Context",
                    "",
                    additional_context,
                    "",
                ]
            )

        if examples:
            examples_text = self._prompt_builder._format_examples(examples)
            prompt_parts.extend(
                [
                    "## Examples",
                    "",
                    examples_text,
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Task",
                "",
                "Extract the structured data from the image(s) according to the schema. "
                "Return only the extracted data in the specified format.",
            ]
        )

        text_prompt = "\n".join(prompt_parts)

        # Build multimodal content
        content: list[dict[str, Any]] = [
            {"type": "text", "text": text_prompt},
        ]

        # Add images - handle single image vs sequence
        # Note: bytes is also a Sequence, but we want to treat it as a single image
        if isinstance(image, (str, Path, bytes, Image.Image)):
            images_to_process: list[ImageInput] = [image]
        else:
            images_to_process = cast(list[ImageInput], list(image))

        for img in images_to_process:
            img_source = self._normalize_image_source(img)
            content.append({"type": "image", "source": img_source})

        # Prepare messages
        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=content),
        ]

        # Prepare LLM kwargs
        llm_kwargs: dict[str, Any] = {
            "temperature": resolved_config.temperature,
        }
        if resolved_config.max_tokens:
            llm_kwargs["max_tokens"] = resolved_config.max_tokens

        cache_enabled = resolved_config.use_cache if use_cache is None else use_cache

        # Call LLM with retry logic
        response = self._call_llm_with_retry(
            messages=messages,
            schema=resolved_schema,
            config=resolved_config,
            use_cache=use_cache,
            **llm_kwargs,
        )

        return ExtractionResult(
            data=response.parsed,  # type: ignore[arg-type]
            model_used=response.model,
            cached=response.cached,
            tokens_used=response.usage.total_tokens,
            cost_usd=response.tracking.cost_usd if response.tracking else None,
            raw_response=response.content,
        )

    def extract_multimodal(
        self,
        document: str,
        images: ImageInput | Sequence[ImageInput],
        schema: type[T] | None = None,
        template: DocumentTemplate | None = None,
        config: ExtractionConfig | None = None,
        field_hints: dict[str, str] | None = None,
        additional_context: str | None = None,
        use_cache: bool | None = None,
    ) -> ExtractionResult[T]:
        """Extract structured data using both text and images.

        Combines a text document with one or more images in a single request.

        Args:
            document: The text to extract from (e.g., email body or notes).
            images: One or more images containing relevant content.
            schema: Pydantic model defining the extraction schema.
            template: Pre-configured document template (alternative to schema).
            config: Extraction configuration (overrides default).
            field_hints: Additional hints for specific fields.
            additional_context: Optional extra context appended to the prompt.

        Returns:
            ExtractionResult containing the extracted data and metadata.

        Raises:
            ValueError: If neither schema nor template is provided.
        """
        # Resolve schema and config from template if provided
        if template is not None:
            resolved_schema: type[BaseModel] = template.schema_class
            resolved_config = config or self.default_config
            resolved_hints = {**(template.field_configs or {}), **(field_hints or {})}
            system_prompt = template.system_prompt
            examples = template.examples
        elif schema is not None:
            resolved_schema = schema
            resolved_config = config or self.default_config
            resolved_hints = field_hints or {}
            system_prompt = resolved_config.system_prompt
            examples = None
        else:
            raise ConfigurationError("Either 'schema' or 'template' must be provided")

        # Build system and user prompts
        system_message = self._prompt_builder.build_system_prompt(system_prompt)
        user_prompt = self._prompt_builder.build_extraction_prompt(
            document=document,
            schema=resolved_schema,
            field_hints=resolved_hints,
            examples=examples,
        )

        if additional_context:
            user_prompt = f"{user_prompt}\n\n## Additional Context\n\n{additional_context}"

        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]

        if isinstance(images, (str, Path, bytes, Image.Image)):
            images_to_process: list[ImageInput] = [images]
        else:
            images_to_process = cast(list[ImageInput], list(images))

        for img in images_to_process:
            img_source = self._normalize_image_source(img)
            content.append({"type": "image", "source": img_source})

        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=content),
        ]

        llm_kwargs: dict[str, Any] = {
            "temperature": resolved_config.temperature,
        }
        if resolved_config.max_tokens:
            llm_kwargs["max_tokens"] = resolved_config.max_tokens

        # Call LLM with retry logic
        response = self._call_llm_with_retry(
            messages=messages,
            schema=resolved_schema,
            config=resolved_config,
            use_cache=use_cache,
            **llm_kwargs,
        )

        return ExtractionResult(
            data=response.parsed,  # type: ignore[arg-type]
            model_used=response.model,
            cached=response.cached,
            tokens_used=response.usage.total_tokens,
            cost_usd=response.tracking.cost_usd if response.tracking else None,
            raw_response=response.content,
        )

    def _call_llm_with_retry(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        config: ExtractionConfig,
        use_cache: bool | None = None,
        **llm_kwargs: Any,
    ) -> Any:
        """Call LLM with retry logic and specific error handling.

        Returns:
            The LLM response if successful.

        Raises:
            ExtractionValidationError: If validation fails.
            LLMError: If the LLM call fails.
            ExtractionError: For other extraction failures.
        """
        cache_enabled = config.use_cache if use_cache is None else use_cache
        last_error: Exception | None = None

        for attempt in range(config.max_retries):
            try:
                logger.debug(
                    "Extraction attempt %d/%d (model=%s)",
                    attempt + 1,
                    config.max_retries,
                    self.model,
                )
                response = self._client.generate(
                    messages,
                    use_cache=cache_enabled,
                    response_format=schema,
                    **llm_kwargs,
                )

                if response.parsed is not None:
                    return response

                last_error = ExtractionValidationError(
                    "LLM did not return parsed data matching the schema",
                    raw_response=response.content,
                )
                logger.warning("Validation failed on attempt %d: No parsed data", attempt + 1)
            except ValidationError as e:
                last_error = ExtractionValidationError(
                    f"Validation error on attempt {attempt + 1}: {str(e)}",
                    validation_errors=e.errors(),
                )
                logger.warning("Validation error on attempt %d: %s", attempt + 1, str(e))
                if not config.retry_on_validation_error:
                    break
            except Exception as e:
                last_error = LLMError(
                    f"LLM call failed on attempt {attempt + 1}: {str(e)}",
                    last_error=e,
                )
                logger.warning("LLM call failed on attempt %d: %s", attempt + 1, str(e))

        # If we're here, all retries failed
        logger.error("Extraction failed after %d attempts: %s", config.max_retries, str(last_error))
        if isinstance(last_error, ExtractionError):
            raise last_error
        raise ExtractionError(str(last_error)) from last_error

    def _normalize_image_source(self, image: ImageInput) -> str | bytes | Image.Image:
        """Normalize image input to a format seeds-clients can handle.

        Args:
            image: Image input in various formats.

        Returns:
            Normalized image source (URL string, file path, PIL Image, or bytes).
        """
        if isinstance(image, Path):
            return str(image)
        return image

    # =========================================================================
    # Tracking and Reporting
    # =========================================================================

    @property
    def cumulative_tracking(self) -> CumulativeTracking:
        """Get cumulative tracking data for all extraction requests.

        Returns tracking information across all extractions made with this
        extractor, including request counts, token usage, costs, and
        carbon emissions.

        Returns:
            CumulativeTracking object with aggregated metrics.

        Example:
            ```python
            extractor = DocumentExtractor(model="gpt-4.1")

            # Make several extractions
            for doc in documents:
                extractor.extract(doc, schema=Invoice)

            # Check cumulative stats
            tracking = extractor.cumulative_tracking
            print(f"Total requests: {tracking.total_request_count}")
            print(f"Cache hit rate: {tracking.cache_hit_rate:.1%}")
            print(f"Total cost: ${tracking.total_cost_usd:.4f}")
            print(f"Carbon: {tracking.api_gwp_kgco2eq:.6f} kgCO2eq")
            ```
        """
        return self._client.cumulative_tracking

    def reset_cumulative_tracking(self) -> None:
        """Reset cumulative tracking to start fresh measurements.

        Useful when you want to track metrics for a specific batch
        of extractions separately from previous work.

        Example:
            ```python
            extractor = DocumentExtractor(model="gpt-4.1")

            # Process first batch
            for doc in batch1:
                extractor.extract(doc, schema=Invoice)
            report1 = extractor.export_boamps_report("batch1_report.json")

            # Reset and process second batch
            extractor.reset_cumulative_tracking()
            for doc in batch2:
                extractor.extract(doc, schema=Invoice)
            report2 = extractor.export_boamps_report("batch2_report.json")
            ```
        """
        self._client.reset_cumulative_tracking()

    def export_boamps_report(
        self,
        output_path: str | Path,
        *,
        publisher_name: str | None = None,
        publisher_division: str | None = None,
        project_name: str | None = None,
        task_description: str | None = None,
        task_family: str = "textGeneration",
        include_summary: bool = True,
        **kwargs: Any,
    ) -> "BoAmpsReport":
        """Export a BoAmps-compliant energy consumption report.

        Generates a standardized JSON report following the BoAmps format
        for energy consumption of document extraction tasks. BoAmps is
        a standard developed by Boavizta for reporting AI environmental
        impact.

        See: https://github.com/Boavizta/BoAmps

        Args:
            output_path: Path where to save the JSON report.
            publisher_name: Name of the organization publishing the report.
            publisher_division: Division/department within the organization.
            project_name: Name of the project.
            task_description: Free-form description of the extraction task.
            task_family: Family of the task (default: "textGeneration").
            include_summary: Whether to print a summary to console.
            **kwargs: Additional arguments passed to BoAmpsReporter.

        Returns:
            BoAmpsReport object containing all energy consumption data.

        Example:
            ```python
            from pydantic import BaseModel

            class Invoice(BaseModel):
                invoice_number: str
                total: float

            extractor = DocumentExtractor(model="gpt-4.1")

            # Process documents
            for doc in documents:
                extractor.extract(doc, schema=Invoice)

            # Export BoAmps report
            report = extractor.export_boamps_report(
                "extraction_report.json",
                publisher_name="My Company",
                task_description="Invoice data extraction",
            )

            # Access report data programmatically
            print(f"Total energy: {report.measures[0].powerConsumption} kWh")
            print(f"Total requests: {report.task.nbRequest}")
            print(f"Model used: {report.task.algorithms[0].foundationModelName}")
            ```
        """
        from seeds_clients.tracking.boamps_reporter import BoAmpsReporter

        reporter = BoAmpsReporter(
            client=self._client,
            publisher_name=publisher_name,
            publisher_division=publisher_division,
            project_name=project_name,
            task_description=task_description,
            task_family=task_family,
            **kwargs,
        )
        return reporter.export(output_path, include_summary=include_summary)
