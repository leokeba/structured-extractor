"""Resume/CV extraction schema.

Provides a standard schema for extracting structured data from
resumes, CVs, and professional profiles.
"""

from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    """Work experience entry in a resume."""

    company: str = Field(description="Company or organization name")
    title: str = Field(description="Job title/position")
    start_date: str | None = Field(default=None, description="Start date")
    end_date: str | None = Field(default=None, description="End date or 'Present'")
    description: str | None = Field(default=None, description="Role description")
    achievements: list[str] = Field(default_factory=list, description="Key achievements")


class Education(BaseModel):
    """Education entry in a resume."""

    institution: str = Field(description="School/university name")
    degree: str = Field(description="Degree or certification")
    field_of_study: str | None = Field(default=None, description="Major/field of study")
    graduation_date: str | None = Field(default=None, description="Graduation date")
    gpa: float | None = Field(default=None, description="GPA if mentioned")


class ResumeSchema(BaseModel):
    """Standard resume/CV extraction schema.

    Use this schema to extract structured data from resumes,
    CVs, and professional profile documents.

    Example:
        ```python
        from structured_extractor import DocumentExtractor, ResumeSchema

        extractor = DocumentExtractor()
        result = extractor.extract(resume_text, schema=ResumeSchema)
        print(result.data.name)
        for job in result.data.work_experience:
            print(f"{job.title} at {job.company}")
        ```
    """

    name: str = Field(description="Candidate's full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    location: str | None = Field(default=None, description="City/location")
    linkedin: str | None = Field(default=None, description="LinkedIn profile URL")
    summary: str | None = Field(default=None, description="Professional summary")
    work_experience: list[WorkExperience] = Field(
        default_factory=list, description="Work history"
    )
    education: list[Education] = Field(
        default_factory=list, description="Educational background"
    )
    skills: list[str] = Field(default_factory=list, description="Technical and soft skills")
    certifications: list[str] = Field(
        default_factory=list, description="Professional certifications"
    )
    languages: list[str] = Field(default_factory=list, description="Languages spoken")
