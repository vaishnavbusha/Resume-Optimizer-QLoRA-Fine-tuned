"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl


# ========================
# Request Schemas
# ========================

class JobDescriptionRequest(BaseModel):
    """Request schema when providing job description text."""
    job_description: str = Field(
        ...,
        description="The job description text",
        min_length=50,
        examples=["We are looking for a Senior Software Engineer with 5+ years of experience in Python..."]
    )


class JobLinkRequest(BaseModel):
    """Request schema when providing a job posting URL."""
    job_url: str = Field(
        ...,
        description="URL of the job posting",
        examples=["https://www.linkedin.com/jobs/view/1234567890"]
    )


# ========================
# Response Schemas
# ========================

class SocialLink(BaseModel):
    """Social media link."""
    name: str
    link: str


class PersonalInfo(BaseModel):
    """Personal information section."""
    name: str
    email: str
    phone: str
    location: str
    socials: Optional[List[SocialLink]] = None


class Experience(BaseModel):
    """Work experience entry."""
    designation: str
    companyName: str
    location: str
    start_date: str
    end_date: Optional[str] = None
    points: Optional[List[str]] = None


class Education(BaseModel):
    """Education entry."""
    institution: str
    degree: str
    location: str
    start_date: str
    end_date: Optional[str] = None
    gpa: Optional[str] = None


class Skill(BaseModel):
    """Skill category."""
    name: str
    data: List[str]


class Project(BaseModel):
    """Project entry."""
    projectName: str
    caption: Optional[str] = None
    location: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    url: Optional[str] = None
    projectDetails: List[str]
    externalSources: Optional[List[SocialLink]] = None
    technologiesUsed: Optional[List[str]] = None


class Certification(BaseModel):
    """Certification entry."""
    name: str
    issuing_organization: str
    issue_date: str
    expiration_date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None


class Award(BaseModel):
    """Award entry."""
    name: str
    type: str
    location: str
    date: str
    description: str


class Achievement(BaseModel):
    """Extracurricular achievement entry."""
    name: str
    type: str
    location: str
    date: str
    description: str


class Language(BaseModel):
    """Language proficiency."""
    language: str
    proficiency: str


class TailoredResume(BaseModel):
    """Complete tailored resume response."""
    personal_information: PersonalInfo
    summary: Optional[str] = None
    experiences: Optional[List[Experience]] = None
    education: List[Education]
    skills: List[Skill]
    projects: Optional[List[Project]] = None
    certifications: Optional[List[Certification]] = None
    awards: Optional[List[Award]] = None
    extracurricular_achievements: List[Achievement]
    languages: Optional[List[Language]] = None


class ResumeOptimizeResponse(BaseModel):
    """API response for resume optimization."""
    success: bool = Field(..., description="Whether the optimization was successful")
    message: str = Field(..., description="Status message")
    tailored_resume: Optional[TailoredResume] = Field(None, description="The tailored resume JSON")
    raw_output: Optional[str] = Field(None, description="Raw model output (if JSON parsing fails)")
    processing_time_seconds: float = Field(..., description="Time taken to process the request")
    saved_to: Optional[str] = Field(None, description="Path where the response was saved")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = False
    error: str
    detail: Optional[str] = None
