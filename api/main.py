"""
FastAPI Application for Resume Optimization.
Main entry point for the API.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings, RESUME_SCHEMA
from .schemas import (
    ResumeOptimizeResponse,
    HealthResponse,
    ErrorResponse,
    TailoredResume,
)
from .resume_parser import resume_parser
from .job_scraper import job_scraper
from .model_service import model_service


# ========================
# Helper Functions
# ========================

def save_uploaded_file(filename: str, content: bytes) -> str:
    """
    Save the uploaded resume file to disk.
    
    Args:
        filename: Original uploaded filename
        content: File content as bytes
    
    Returns:
        Path to the saved file
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean filename (replace special chars)
    name_part = Path(filename).stem
    ext_part = Path(filename).suffix
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name_part)
    
    # Create output filename with timestamp
    output_filename = f"{clean_name}_{timestamp}{ext_part}"
    output_path = Path(settings.UPLOAD_DIR) / output_filename
    
    # Save to file
    with open(output_path, "wb") as f:
        f.write(content)
    
    print(f"📄 Uploaded file saved to: {output_path}")
    return str(output_path)


def save_response(filename: str, response_data: dict, raw_output: str = None, uploaded_file_path: str = None) -> str:
    """
    Save the generated response to a file.
    
    Args:
        filename: Original uploaded filename (without extension)
        response_data: The parsed JSON response or raw output
        raw_output: Raw model output (used if JSON parsing failed)
        uploaded_file_path: Path where the uploaded file was saved
    
    Returns:
        Path to the saved file
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean filename (remove extension, replace special chars)
    clean_name = Path(filename).stem
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in clean_name)
    
    # Create output filename
    output_filename = f"{clean_name}_{timestamp}.json"
    output_path = Path(settings.RESPONSE_DIR) / output_filename
    
    # Prepare data to save
    save_data = {
        "original_filename": filename,
        "uploaded_file_path": uploaded_file_path,
        "generated_at": datetime.now().isoformat(),
        "tailored_resume": response_data,
    }
    
    # If we have raw output (JSON parsing failed), save that too
    if raw_output:
        save_data["raw_output"] = raw_output
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Response saved to: {output_path}")
    return str(output_path)


# ========================
# Lifespan Context Manager
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the model on startup and unloads on shutdown.
    """
    # Startup: Load the model
    print("🚀 Starting Resume Optimizer API...")
    print(f"📁 LoRA adapter path: {settings.LORA_ADAPTER_PATH}")
    
    # Load model (this may take a minute)
    success = model_service.load_model()
    if not success:
        print("⚠️ Model failed to load. API will return errors for optimization requests.")
    
    yield
    
    # Shutdown: Unload the model
    print("🛑 Shutting down Resume Optimizer API...")
    model_service.unload_model()


# ========================
# FastAPI App
# ========================

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# Endpoints
# ========================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        gpu_available=model_service.gpu_available,
        gpu_name=model_service.gpu_name,
    )


@app.get("/schema", tags=["Schema"])
async def get_schema():
    """Get the JSON schema for the tailored resume output."""
    return RESUME_SCHEMA


@app.post(
    "/optimize",
    response_model=ResumeOptimizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Resume Optimization"],
)
async def optimize_resume(
    resume_file: UploadFile = File(..., description="Resume file (PDF, DOCX, DOC, or TXT)"),
    job_description: Optional[str] = Form(None, description="Job description text"),
    job_url: Optional[str] = Form(None, description="URL of the job posting"),
):
    """
    Optimize a resume for a specific job.
    
    Upload a resume file and provide either:
    - `job_description`: The job description text directly
    - `job_url`: URL of the job posting (will be scraped)
    
    Returns a tailored resume in JSON format.
    """
    start_time = time.time()
    
    # Validate inputs
    if not job_description and not job_url:
        raise HTTPException(
            status_code=400,
            detail="Either job_description or job_url must be provided"
        )
    
    # Check model status
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later."
        )
    
    # Validate file size
    file_size = 0
    content = await resume_file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    # Validate file extension
    filename = resume_file.filename or "resume.pdf"
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Save uploaded file
    uploaded_file_path = save_uploaded_file(filename, content)
    
    try:
        # 1. Parse resume
        resume_text = resume_parser.parse(file_bytes=content, filename=filename)
        
        if not resume_text or len(resume_text) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from the resume. Please try a different file format."
            )
        
        # 2. Get job description
        if job_url:
            try:
                job_desc = job_scraper.scrape(job_url)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to scrape job URL: {str(e)}. Please provide the job description text directly."
                )
        else:
            job_desc = job_description
        
        # 3. Generate tailored resume
        result = model_service.generate(
            resume_text=resume_text,
            job_description=job_desc,
        )
        
        processing_time = time.time() - start_time
        
        # 4. Build response
        if result["parsed_json"]:
            try:
                tailored_resume = TailoredResume(**result["parsed_json"])
                
                # Save response to file
                saved_path = save_response(
                    filename=filename,
                    response_data=result["parsed_json"],
                    uploaded_file_path=uploaded_file_path,
                )
                
                return ResumeOptimizeResponse(
                    success=True,
                    message="Resume optimized successfully",
                    tailored_resume=tailored_resume,
                    raw_output=None,
                    processing_time_seconds=processing_time,
                    saved_to=saved_path,
                )
            except Exception as e:
                # JSON parsed but doesn't match schema - still save it
                saved_path = save_response(
                    filename=filename,
                    response_data=result["parsed_json"],
                    raw_output=result["raw_output"],
                    uploaded_file_path=uploaded_file_path,
                )
                
                return ResumeOptimizeResponse(
                    success=True,
                    message=f"Resume generated but schema validation failed: {str(e)}",
                    tailored_resume=None,
                    raw_output=result["raw_output"],
                    processing_time_seconds=processing_time,
                    saved_to=saved_path,
                )
        else:
            # Save raw output even if JSON parsing failed
            saved_path = save_response(
                filename=filename,
                response_data=None,
                raw_output=result["raw_output"],
                uploaded_file_path=uploaded_file_path,
            )
            
            return ResumeOptimizeResponse(
                success=False,
                message=f"Generated output is not valid JSON: {result['parse_error']}",
                tailored_resume=None,
                raw_output=result["raw_output"],
                processing_time_seconds=processing_time,
                saved_to=saved_path,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@app.post(
    "/optimize/text",
    response_model=ResumeOptimizeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Resume Optimization"],
)
async def optimize_resume_text(
    resume_text: str = Form(..., description="Resume text content"),
    job_description: Optional[str] = Form(None, description="Job description text"),
    job_url: Optional[str] = Form(None, description="URL of the job posting"),
):
    """
    Optimize a resume from plain text input.
    
    Provide resume text directly along with either:
    - `job_description`: The job description text directly
    - `job_url`: URL of the job posting (will be scraped)
    
    Returns a tailored resume in JSON format.
    """
    start_time = time.time()
    
    # Validate inputs
    if not job_description and not job_url:
        raise HTTPException(
            status_code=400,
            detail="Either job_description or job_url must be provided"
        )
    
    if len(resume_text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Resume text is too short. Please provide more content."
        )
    
    # Check model status
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later."
        )
    
    try:
        # Get job description
        if job_url:
            try:
                job_desc = job_scraper.scrape(job_url)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to scrape job URL: {str(e)}. Please provide the job description text directly."
                )
        else:
            job_desc = job_description
        
        # Generate tailored resume
        result = model_service.generate(
            resume_text=resume_text,
            job_description=job_desc,
        )
        
        processing_time = time.time() - start_time
        
        # Generate a filename for text input
        text_filename = f"text_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build response
        if result["parsed_json"]:
            try:
                tailored_resume = TailoredResume(**result["parsed_json"])
                
                # Save response to file
                saved_path = save_response(
                    filename=text_filename,
                    response_data=result["parsed_json"],
                )
                
                return ResumeOptimizeResponse(
                    success=True,
                    message="Resume optimized successfully",
                    tailored_resume=tailored_resume,
                    raw_output=None,
                    processing_time_seconds=processing_time,
                    saved_to=saved_path,
                )
            except Exception as e:
                # Save even if schema validation failed
                saved_path = save_response(
                    filename=text_filename,
                    response_data=result["parsed_json"],
                    raw_output=result["raw_output"],
                )
                
                return ResumeOptimizeResponse(
                    success=True,
                    message=f"Resume generated but schema validation failed: {str(e)}",
                    tailored_resume=None,
                    raw_output=result["raw_output"],
                    processing_time_seconds=processing_time,
                    saved_to=saved_path,
                )
        else:
            # Save raw output even if JSON parsing failed
            saved_path = save_response(
                filename=text_filename,
                response_data=None,
                raw_output=result["raw_output"],
            )
            
            return ResumeOptimizeResponse(
                success=False,
                message=f"Generated output is not valid JSON: {result['parse_error']}",
                tailored_resume=None,
                raw_output=result["raw_output"],
                processing_time_seconds=processing_time,
                saved_to=saved_path,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


# ========================
# Error Handlers
# ========================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "detail": None,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None,
        }
    )
