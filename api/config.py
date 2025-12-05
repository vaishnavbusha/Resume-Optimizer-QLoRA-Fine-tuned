"""
Configuration settings for the Resume Optimizer API.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    APP_NAME: str = "Resume Optimizer API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for generating tailored resumes using QLoRA fine-tuned Qwen3-4B"
    DEBUG: bool = False
    
    # Model Settings
    MODEL_NAME: str = "Qwen/Qwen3-4B-Instruct-2507"
    LORA_ADAPTER_PATH: str = "./qwen3-resume-lora-single-gpu"
    MAX_NEW_TOKENS: int = 4096
    TEMPERATURE: float = 0.0  # Deterministic for JSON output
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"pdf", "docx", "doc", "txt"}
    UPLOAD_DIR: str = "./api/uploads"
    RESPONSE_DIR: str = "./api/responses"  # Directory to save generated responses
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.RESPONSE_DIR).mkdir(parents=True, exist_ok=True)


# Resume JSON Schema
RESUME_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "personal_information": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "location": {"type": "string"},
                "socials": {
                    "type": "array",
                    "items": [{
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "link": {"type": "string"}
                        },
                        "required": ["name", "link"]
                    }]
                }
            },
            "required": ["name", "email", "phone", "location"]
        },
        "summary": {"type": "string"},
        "experiences": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "designation": {"type": "string"},
                    "companyName": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "points": {"type": "array", "items": [{"type": "string"}]}
                },
                "required": ["designation", "companyName", "location", "start_date"]
            }]
        },
        "education": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "gpa": {"type": "string"}
                },
                "required": ["institution", "degree", "location", "start_date", "gpa"]
            }]
        },
        "skills": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "data": {"type": "array", "items": [{"type": "string"}]}
                },
                "required": ["name", "data"]
            }]
        },
        "projects": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "projectName": {"type": "string"},
                    "caption": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "url": {"type": "string"},
                    "projectDetails": {"type": "array", "items": [{"type": "string"}]},
                    "externalSources": {
                        "type": "array",
                        "items": [{
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "link": {"type": "string"}
                            },
                            "required": ["name", "link"]
                        }]
                    },
                    "technologiesUsed": {"type": "array", "items": [{"type": "string"}]}
                },
                "required": ["projectName", "location", "projectDetails"]
            }]
        },
        "certifications": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuing_organization": {"type": "string"},
                    "issue_date": {"type": "string"},
                    "expiration_date": {"type": "string"},
                    "credential_id": {"type": "string"},
                    "url": {"type": "string"}
                },
                "required": ["name", "issuing_organization", "issue_date", "expiration_date", "credential_id", "url"]
            }]
        },
        "awards": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "location": {"type": "string"},
                    "date": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type", "location", "date", "description"]
            }]
        },
        "extracurricular_achievements": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "location": {"type": "string"},
                    "date": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type", "location", "date", "description"]
            }]
        },
        "languages": {
            "type": "array",
            "items": [{
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "proficiency": {"type": "string"}
                },
                "required": ["language", "proficiency"]
            }]
        }
    },
    "required": ["personal_information", "education", "skills", "extracurricular_achievements"]
}


# System prompt for the model
SYSTEM_PROMPT = """You create a tailored resume based on the job description.

Your task:
1. Read the RESUME_TEXT.
2. Read the JOB_DESCRIPTION.
3. Use only the information inside these two.
4. Follow the SCHEMA exactly.
5. Write a tailored resume in JSON using the SCHEMA.
6. Do not output anything outside the JSON.
7. If a field is missing in the resume, write a short, safe placeholder that fits the job.

"""
