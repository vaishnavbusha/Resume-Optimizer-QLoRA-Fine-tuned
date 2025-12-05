# Resume Optimizer API

A FastAPI-based REST API for generating tailored resumes using the QLoRA fine-tuned Qwen3-4B model.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r api_requirements.txt
```

### 2. Start the Server

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`.

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check (model status) |
| `/schema` | GET | Get resume JSON schema |
| `/optimize` | POST | Optimize resume from file upload |
| `/optimize/text` | POST | Optimize resume from text input |

## 📤 Usage Examples

### Using cURL

#### Upload Resume File + Job Description

```bash
curl -X POST "http://localhost:8000/optimize" \
  -F "resume_file=@/path/to/resume.pdf" \
  -F "job_description=We are looking for a Senior Software Engineer with 5+ years of experience in Python..."
```

#### Upload Resume File + Job URL

```bash
curl -X POST "http://localhost:8000/optimize" \
  -F "resume_file=@/path/to/resume.pdf" \
  -F "job_url=https://www.linkedin.com/jobs/view/1234567890"
```

#### Resume Text + Job Description

```bash
curl -X POST "http://localhost:8000/optimize/text" \
  -F "resume_text=John Doe, Software Engineer, 5 years experience in Python, Django, AWS..." \
  -F "job_description=We are looking for a Senior Software Engineer..."
```

### Using Python

```python
import requests

# Upload file with job description
url = "http://localhost:8000/optimize"

with open("resume.pdf", "rb") as f:
    response = requests.post(
        url,
        files={"resume_file": ("resume.pdf", f, "application/pdf")},
        data={"job_description": "We are looking for a Senior Software Engineer..."}
    )

result = response.json()

if result["success"]:
    print("Tailored Resume:")
    print(result["tailored_resume"])
else:
    print("Error:", result["message"])
    print("Raw output:", result["raw_output"])
```

### Using JavaScript (Fetch)

```javascript
const formData = new FormData();
formData.append('resume_file', fileInput.files[0]);
formData.append('job_description', 'We are looking for a Senior Software Engineer...');

const response = await fetch('http://localhost:8000/optimize', {
    method: 'POST',
    body: formData
});

const result = await response.json();

if (result.success) {
    console.log('Tailored Resume:', result.tailored_resume);
} else {
    console.error('Error:', result.message);
}
```

## 📁 Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Requires PyPDF2 |
| Word (modern) | `.docx` | Requires python-docx |
| Word (legacy) | `.doc` | Requires antiword or catdoc |
| Plain Text | `.txt` | UTF-8 encoding recommended |

## 📊 Response Format

### Success Response

```json
{
  "success": true,
  "message": "Resume optimized successfully",
  "tailored_resume": {
    "personal_information": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "+1234567890",
      "location": "San Francisco, CA"
    },
    "summary": "Experienced software engineer...",
    "experiences": [...],
    "education": [...],
    "skills": [...],
    ...
  },
  "raw_output": null,
  "processing_time_seconds": 4.25
}
```

### Error Response

```json
{
  "success": false,
  "error": "Error description",
  "detail": "Additional details (debug mode only)"
}
```

## ⚙️ Configuration

Create a `.env` file to customize settings:

```env
# Model Settings
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
LORA_ADAPTER_PATH=./qwen3-resume-lora-single-gpu
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0

# Server Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
```

## 🔧 Command Line Options

```bash
python run_api.py --help

Options:
  --host TEXT     Host to bind to (default: 0.0.0.0)
  --port INTEGER  Port to bind to (default: 8000)
  --reload        Enable auto-reload (development mode)
  --workers INT   Number of worker processes (default: 1)
```

## 📝 API Architecture

```
api/
├── __init__.py          # Package init
├── config.py            # Settings and configuration
├── schemas.py           # Pydantic models for request/response
├── resume_parser.py     # Resume file parsing (PDF, DOCX, etc.)
├── job_scraper.py       # Job URL scraping
├── model_service.py     # ML model loading and inference
└── main.py              # FastAPI application and endpoints
```

## 🐛 Troubleshooting

### Model Not Loading

1. Ensure the LoRA adapter path is correct
2. Check GPU memory (requires ~5GB for inference)
3. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### PDF Parsing Issues

Install PyPDF2:
```bash
pip install PyPDF2
```

### DOC File Support

For legacy `.doc` files, install antiword:
```bash
# Ubuntu/Debian
sudo apt-get install antiword

# macOS
brew install antiword
```

### Job URL Scraping Fails

Many job sites block scraping. If URL scraping fails:
1. Copy the job description text manually
2. Use the `job_description` parameter instead of `job_url`

## 📄 License

MIT License - See main project LICENSE file.
