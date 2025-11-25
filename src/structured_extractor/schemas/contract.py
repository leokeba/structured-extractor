"""Contract extraction schema.

Provides a standard schema for extracting structured data from
legal contracts and agreements.
"""

from pydantic import BaseModel, Field


class ContractParty(BaseModel):
    """A party to a contract."""

    name: str = Field(description="Legal name of the party")
    role: str | None = Field(default=None, description="Role in contract (e.g., Buyer)")
    address: str | None = Field(default=None, description="Address")


class ContractSchema(BaseModel):
    """Standard contract extraction schema.

    Use this schema to extract structured data from legal contracts,
    agreements, and similar legal documents.

    Example:
        ```python
        from structured_extractor import DocumentExtractor, ContractSchema

        extractor = DocumentExtractor()
        result = extractor.extract(contract_text, schema=ContractSchema)
        for party in result.data.parties:
            print(f"{party.name} ({party.role})")
        ```
    """

    title: str | None = Field(default=None, description="Contract title")
    contract_type: str | None = Field(default=None, description="Type of contract")
    parties: list[ContractParty] = Field(
        default_factory=list, description="Parties to the contract"
    )
    effective_date: str | None = Field(default=None, description="When contract takes effect")
    termination_date: str | None = Field(default=None, description="Contract end date")
    term_duration: str | None = Field(default=None, description="Duration of the agreement")
    total_value: float | None = Field(default=None, description="Total contract value")
    currency: str | None = Field(default=None, description="Currency for monetary values")
    payment_terms: str | None = Field(default=None, description="Payment schedule/terms")
    governing_law: str | None = Field(default=None, description="Governing jurisdiction")
    key_obligations: list[str] = Field(
        default_factory=list, description="Main obligations of parties"
    )
    termination_clauses: list[str] = Field(
        default_factory=list, description="Termination conditions"
    )
    special_conditions: list[str] = Field(
        default_factory=list, description="Special terms or conditions"
    )
