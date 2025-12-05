"""
Resume Parser Module.
Handles extraction of text from various resume file formats (PDF, DOCX, DOC, TXT).
"""

import io
import os
import re
import tempfile
from pathlib import Path
from typing import Optional
import subprocess

# PDF extraction
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# DOCX extraction
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class ResumeParser:
    """
    Parser for extracting text from resume files.
    Supports PDF, DOCX, DOC, and TXT formats.
    """
    
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
    
    def __init__(self):
        """Initialize the resume parser."""
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if not PDF_AVAILABLE:
            print("⚠️ PyPDF2 not installed. PDF parsing will not work.")
        if not DOCX_AVAILABLE:
            print("⚠️ python-docx not installed. DOCX parsing will not work.")
    
    def parse(self, file_path: str = None, file_bytes: bytes = None, filename: str = None) -> str:
        """
        Extract text from a resume file.
        
        Args:
            file_path: Path to the file on disk
            file_bytes: File content as bytes (for uploaded files)
            filename: Original filename (required if using file_bytes)
        
        Returns:
            Extracted text from the resume
        
        Raises:
            ValueError: If file format is not supported or extraction fails
        """
        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
            
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        elif file_bytes and filename:
            ext = Path(filename).suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
        else:
            raise ValueError("Either file_path or (file_bytes + filename) must be provided")
        
        # Extract based on file type
        if ext == ".pdf":
            return self._extract_from_pdf(file_bytes)
        elif ext == ".docx":
            return self._extract_from_docx(file_bytes)
        elif ext == ".doc":
            return self._extract_from_doc(file_bytes)
        elif ext == ".txt":
            return self._extract_from_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            raise ValueError("PyPDF2 is not installed. Install with: pip install PyPDF2")
        
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            text = "\n".join(text_parts)
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file."""
        if not DOCX_AVAILABLE:
            raise ValueError("python-docx is not installed. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(io.BytesIO(file_bytes))
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        text_parts.append(row_text)
            
            text = "\n".join(text_parts)
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
    
    def _extract_from_doc(self, file_bytes: bytes) -> str:
        """
        Extract text from DOC file (legacy Word format).
        Uses antiword or catdoc if available, otherwise tries conversion.
        """
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            # Try antiword first
            try:
                result = subprocess.run(
                    ["antiword", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    return self._clean_text(result.stdout)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Try catdoc
            try:
                result = subprocess.run(
                    ["catdoc", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    return self._clean_text(result.stdout)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # If no tool available, raise error
            raise ValueError(
                "Cannot extract text from .doc file. "
                "Please convert to .docx or .pdf format, or install antiword/catdoc."
            )
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def _extract_from_txt(self, file_bytes: bytes) -> str:
        """Extract text from TXT file."""
        # Try different encodings
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(encoding)
                return self._clean_text(text)
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Failed to decode text file. Please use UTF-8 encoding.")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and normalizing.
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Strip leading/trailing whitespace from entire text
        text = text.strip()
        
        return text


# Singleton instance
resume_parser = ResumeParser()
