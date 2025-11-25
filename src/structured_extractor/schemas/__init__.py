"""Built-in extraction schemas for common document types.

This module provides pre-defined Pydantic schemas for extracting
structured data from common document types like invoices, receipts,
resumes, and contracts.
"""

from structured_extractor.schemas.contract import ContractParty, ContractSchema
from structured_extractor.schemas.invoice import InvoiceLineItem, InvoiceSchema
from structured_extractor.schemas.receipt import ReceiptSchema
from structured_extractor.schemas.resume import Education, ResumeSchema, WorkExperience

__all__ = [
    # Invoice
    "InvoiceSchema",
    "InvoiceLineItem",
    # Receipt
    "ReceiptSchema",
    # Resume
    "ResumeSchema",
    "WorkExperience",
    "Education",
    # Contract
    "ContractSchema",
    "ContractParty",
]
