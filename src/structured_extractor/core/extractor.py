"""Main document extractor class."""

import os
from typing import Any, TypeVar

from pydantic import BaseModel
from seeds_clients import Message, OpenAIClient

from structured_extractor.core.config import ExtractionConfig
from structured_extractor.core.templates import DocumentTemplate
from structured_extractor.prompts.builder import PromptBuilder
from structured_extractor.results.types import ExtractionResult

T = TypeVar("T", bound=BaseModel)


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

        extractor = DocumentExtractor(model="gpt-4o")
        result = extractor.extract(document_text, schema=Invoice)
        print(result.data.invoice_number)
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
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

        return ExtractionResult(
            data=error_data,  # type: ignore[arg-type]
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
