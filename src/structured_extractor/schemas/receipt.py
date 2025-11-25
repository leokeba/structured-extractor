"""Receipt extraction schema.

Provides a standard schema for extracting structured data from
receipts and purchase records.
"""

from pydantic import BaseModel, Field


class ReceiptSchema(BaseModel):
    """Standard receipt extraction schema.

    Use this schema to extract structured data from receipts,
    purchase records, and point-of-sale documents.

    Example:
        ```python
        from structured_extractor import DocumentExtractor, ReceiptSchema

        extractor = DocumentExtractor()
        result = extractor.extract(receipt_text, schema=ReceiptSchema)
        print(result.data.total)
        ```
    """

    merchant_name: str = Field(description="Name of the merchant/store")
    merchant_address: str | None = Field(default=None, description="Store address")
    date: str = Field(description="Transaction date in ISO format")
    time: str | None = Field(default=None, description="Transaction time")
    items: list[str] = Field(default_factory=list, description="List of purchased items")
    subtotal: float | None = Field(default=None, description="Subtotal before tax")
    tax: float | None = Field(default=None, description="Tax amount")
    total: float = Field(description="Total amount paid")
    payment_method: str | None = Field(default=None, description="Payment method used")
    card_last_four: str | None = Field(default=None, description="Last 4 digits of card")
