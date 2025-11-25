"""Main document extractor class."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from PIL import Image
from pydantic import BaseModel
from seeds_clients import Message, OpenAIClient

from structured_extractor.core.config import ExtractionConfig
from structured_extractor.core.templates import DocumentTemplate
from structured_extractor.prompts.builder import PromptBuilder
from structured_extractor.results.types import ExtractionResult

T = TypeVar("T", bound=BaseModel)

# Type alias for image inputs
ImageInput = str | Path | Image.Image | bytes


class DocumentExtractor:
    """LLM-driven structured data extractor for documents.

    Uses seeds-clients for LLM integration and Pydantic for schema validation.

    Example:
        ```python
        from pydantic import BaseModel, Field
        from structured_extractor import DocumentExtractor

        class Invoice(BaseModel):
            invoice_number: str
            total_amount: float
            vendor_name: str

        extractor = DocumentExtractor(model="gpt-4.1")
        result = extractor.extract(document_text, schema=Invoice)
        print(result.data.invoice_number)
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
        cache_dir: str = "cache",
        cache_ttl_hours: float | None = 24.0,
        default_config: ExtractionConfig | None = None,
    ) -> None:
        """Initialize the document extractor.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: LLM model to use for extraction.
            cache_dir: Directory for caching LLM responses.
            cache_ttl_hours: Cache TTL in hours. None disables caching.
            default_config: Default extraction configuration.
        """
        self.model = model
        self.default_config = default_config or ExtractionConfig()

        # Initialize the LLM client
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
            raise ValueError("Either 'schema' or 'template' must be provided")

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
            "response_format": resolved_schema,
            "temperature": resolved_config.temperature,
        }
        if resolved_config.max_tokens:
            llm_kwargs["max_tokens"] = resolved_config.max_tokens

        # Call LLM with retry logic
        last_error: Exception | None = None
        for _attempt in range(resolved_config.max_retries):
            try:
                response = self._client.generate(messages, **llm_kwargs)

                # Build result
                if response.parsed is not None:
                    return ExtractionResult(
                        data=response.parsed,  # type: ignore[arg-type]
                        success=True,
                        model_used=response.model,
                        cached=response.cached,
                        tokens_used=response.usage.total_tokens,
                        cost_usd=response.tracking.cost_usd if response.tracking else None,
                        raw_response=response.content,
                    )
                else:
                    # This shouldn't happen with response_format, but handle gracefully
                    last_error = ValueError("LLM did not return parsed data")

            except Exception as e:
                last_error = e
                if not resolved_config.retry_on_validation_error:
                    break

        # All retries failed
        # Create a minimal "empty" instance for the error case
        # We need to return something, so we create with required fields only
        error_data = self._create_error_placeholder(resolved_schema)

        return ExtractionResult[T](
            data=cast(T, error_data),
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            model_used=self.model,
        )

    def _create_error_placeholder(self, schema: type[BaseModel]) -> BaseModel:
        """Create a placeholder instance for error cases.

        This creates an instance with None/default values for all fields.
        """
        # Try to create with all None values
        field_values: dict[str, Any] = {}
        for field_name, field_info in schema.model_fields.items():
            if not field_info.is_required():
                field_values[field_name] = field_info.default
            else:
                # Use None for required fields (will fail validation but gives structure)
                field_values[field_name] = None

        try:
            return schema.model_construct(**field_values)
        except Exception:
            # Last resort: construct without validation
            return schema.model_construct(_fields_set=set(), **field_values)

    def extract_with_confidence(
        self,
        document: str,
        schema: type[T],
        config: ExtractionConfig | None = None,
    ) -> ExtractionResult[T]:
        """Extract with confidence scoring (placeholder for Phase 5).

        Args:
            document: The document text to extract from.
            schema: Pydantic model defining the extraction schema.
            config: Extraction configuration.

        Returns:
            ExtractionResult with confidence scores populated.
        """
        # For now, delegate to regular extract
        # Phase 5 will implement proper confidence scoring
        effective_config = config or self.default_config
        effective_config = ExtractionConfig(
            **effective_config.model_dump(),
            include_confidence=True,
        )
        return self.extract(document, schema=schema, config=effective_config)

    def extract_from_image(
        self,
        image: ImageInput | Sequence[ImageInput],
        schema: type[T] | None = None,
        template: DocumentTemplate | None = None,
        config: ExtractionConfig | None = None,
        field_hints: dict[str, str] | None = None,
        additional_context: str | None = None,
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
            ValueError: If neither schema nor template is provided.

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
            raise ValueError("Either 'schema' or 'template' must be provided")

        # Build prompts - use empty string for document since we're using images
        system_message = self._prompt_builder.build_system_prompt(system_prompt)

        # Build extraction prompt without document content
        schema_description = self._prompt_builder._describe_schema(
            resolved_schema, resolved_hints
        )

        prompt_parts = [
            "## Extraction Schema",
            "",
            schema_description,
            "",
        ]

        if additional_context:
            prompt_parts.extend([
                "## Additional Context",
                "",
                additional_context,
                "",
            ])

        if examples:
            examples_text = self._prompt_builder._format_examples(examples)
            prompt_parts.extend([
                "## Examples",
                "",
                examples_text,
                "",
            ])

        prompt_parts.extend([
            "## Task",
            "",
            "Extract the structured data from the image(s) according to the schema. "
            "Return only the extracted data in the specified format.",
        ])

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
            "response_format": resolved_schema,
            "temperature": resolved_config.temperature,
        }
        if resolved_config.max_tokens:
            llm_kwargs["max_tokens"] = resolved_config.max_tokens

        # Call LLM with retry logic
        last_error: Exception | None = None
        for _attempt in range(resolved_config.max_retries):
            try:
                response = self._client.generate(messages, **llm_kwargs)

                # Build result
                if response.parsed is not None:
                    return ExtractionResult(
                        data=response.parsed,  # type: ignore[arg-type]
                        success=True,
                        model_used=response.model,
                        cached=response.cached,
                        tokens_used=response.usage.total_tokens,
                        cost_usd=response.tracking.cost_usd if response.tracking else None,
                        raw_response=response.content,
                    )
                else:
                    last_error = ValueError("LLM did not return parsed data")

            except Exception as e:
                last_error = e
                if not resolved_config.retry_on_validation_error:
                    break

        # All retries failed
        error_data = self._create_error_placeholder(resolved_schema)

        return ExtractionResult[T](
            data=cast(T, error_data),
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            model_used=self.model,
        )

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
