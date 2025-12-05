# 🎯 Resume Optimizer - QLoRA Fine-tuned Qwen3-4B

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end pipeline for building a resume optimization system using QLoRA fine-tuning on Qwen3-4B-Instruct.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Pipeline Workflow](#-pipeline-workflow)
- [Dataset Format](#-dataset-format)
- [Training](#-training)
- [Inference](#-inference)
- [Rest API for model inference](#-rest-api)
- [Model Performance](#-model-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project demonstrates a complete pipeline for building a **resume optimization system** that:

| Step | Description |
|------|-------------|
| 📄 **Extract** | Parse text from 1800+ resumes (PDF, DOCX, DOC) |
| 🔍 **Scrape** | Collect job descriptions from job posting websites |
| 🤖 **Generate** | Create tailored resumes using LLMs (Ollama + Gemini) |
| 🎯 **Fine-tune** | Train Qwen3-4B with QLoRA for optimized JSON output |
| 🚀 **Deploy** | Run inference with the fine-tuned model |

The model uses **QLoRA** (Quantized Low-Rank Adaptation) with 4-bit quantization, enabling efficient fine-tuning on consumer GPUs (tested on **NVIDIA RTX 3090** with 24GB VRAM).

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📦 **QLoRA Training** | Memory-efficient 4-bit NF4 quantization with LoRA adapters |
| 📝 **Structured JSON** | Generates valid JSON following a comprehensive resume schema |
| 💬 **Chat Template** | Uses Qwen's native chat template for training and inference |
| ⚡ **Flash Attention 2** | Optional faster inference on compatible GPUs |
| 🔗 **Merge & Unload** | Faster inference with merged LoRA weights |
| ☁️ **Batch Processing** | Gemini Batch API support for large-scale processing |
| 🏠 **Local Processing** | Ollama integration for privacy-focused generation |
| 🌐 **REST API** | FastAPI server for resume optimization via HTTP |

---

## 📁 Project Structure

```
QLoRA-finetuning/
│
├── 📓 final_project.ipynb              # Main notebook (11 sections)
├── 📄 README.md                         # This file
├── 📦 requirements.txt                  # Python dependencies
├── 🐍 environment.yml                   # Conda environment
├── 🚀 run_api.py                        # API server launcher
├── 📦 api_requirements.txt              # API dependencies
│
├── 🌐 api/                              # FastAPI REST API
│   ├── main.py                         # API endpoints
│   ├── config.py                       # Settings & schema
│   ├── schemas.py                      # Request/response models
│   ├── model_service.py                # ML inference service
│   ├── resume_parser.py                # PDF/DOCX parsing
│   ├── job_scraper.py                  # Job URL scraping
│   └── README.md                       # API documentation
│
├── 📂 files/
│   ├── dataset/
│   │   ├── batch_requests/             # Gemini batch request files
│   │   ├── batch_results/              # Batch processing results
│   │   ├── extracted-resume-data/      # Raw extracted resume text
│   │   ├── final-training-data/        # Cleaned training dataset
│   │   └── resume-data-with-job-description/
│   ├── job-scraper/                    # Scraped job data
│   └── prompt/                         # Prompt templates
│
├── 🤖 qwen3-resume-lora-single-gpu/    # Fine-tuned LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── checkpoint-384/
│
└── 📊 results/
    ├── basemodel/                      # Base model outputs
    └── finetuned/                      # Fine-tuned model outputs
```

---

## 🛠️ Installation

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required |
| CUDA | 11.8+ | For GPU acceleration |
| GPU VRAM | 24GB+ | Recommended (RTX 3090/4090) |
| Conda | Latest | For environment management |

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Abhinav1426/QLoRA-finetuning.git
cd QLoRA-finetuning

# 2. Create conda environment
conda env create -f environment.yml
conda activate finetune

# 3. Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Manual Installation

```bash
conda create -n finetune python=3.11
conda activate finetune

# Core ML packages
pip install torch transformers datasets accelerate trl peft bitsandbytes

# Data processing
pip install pandas numpy tqdm requests PyPDF2 python-docx pyarrow

# Web scraping (optional)
pip install selenium webdriver-manager beautifulsoup4 lxml

# Google AI (optional)
pip install google-genai
```

---

## 🔄 Pipeline Workflow

The notebook is organized into **11 sections**:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Setup          →  2. Resume Extraction  →  3. Job Scraping  │
│         ↓                                                       │
│  4. Data Cleaning  →  5. Ollama Generation  →  6. Gemini Batch  │
│         ↓                                                       │
│  7. Training Prep  →  8. Base Model Test    →  9. LoRA Training │
│         ↓                                                       │
│  10. Inference     →  11. Alternative Training                  │
└─────────────────────────────────────────────────────────────────┘
```

| # | Section | Description |
|:-:|---------|-------------|
| 1 | **Setup & Dependencies** | Install packages, import libraries |
| 2 | **Dataset Creation** | Extract text from 1800+ resumes (PDF/DOCX/DOC) |
| 3 | **Job Scraper** | Scrape job descriptions using Selenium |
| 4 | **Data Cleaning** | Clean, filter, and combine resume-job pairs |
| 5 | **Ollama Generator** | Generate tailored resumes locally |
| 6 | **Gemini Batch API** | Large-scale processing via Google Cloud |
| 7 | **Training Preparation** | Convert to chat format for fine-tuning |
| 8 | **Base Model Testing** | Establish baseline performance |
| 9 | **LoRA Fine-tuning** | Train with QLoRA technique |
| 10 | **Inference** | Generate resumes with fine-tuned model |
| 11 | **Alternative Training** | Stable single-GPU configuration |

---

## 📊 Dataset Format

### Training Data Structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You create a tailored resume based on the job description..."
    },
    {
      "role": "user", 
      "content": "RESUME_TEXT:\n\"\"\"...\"\"\"\n\nJOB_DESCRIPTION:\n\"\"\"...\"\"\"\n\nSCHEMA:\n\"\"\"...\"\"\""
    },
    {
      "role": "assistant",
      "content": "{...structured JSON resume...}"
    }
  ]
}
```

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `personal_information` | object | Name, email, phone, location, socials |
| `summary` | string | Professional summary tailored to job |
| `experiences` | array | Work history with bullet points |
| `education` | array | Degrees, institutions, GPA |
| `skills` | array | Categorized skills |
| `projects` | array | Project details with technologies |
| `certifications` | array | Professional certifications |
| `awards` | array | Awards and recognitions |
| `extracurricular_achievements` | array | Additional achievements |
| `languages` | array | Language proficiencies |

---

## 🎯 Training

### Configuration Summary

```python
# Model
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Quantization (4-bit QLoRA)
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.float16
bnb_4bit_use_double_quant = True

# LoRA
r = 16                    # Rank
lora_alpha = 32           # Scaling factor
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
num_train_epochs = 2
per_device_train_batch_size = 1
gradient_accumulation_steps = 8   # Effective batch = 8
learning_rate = 2e-4
max_seq_length = 4096
fp16 = True
```

### Memory Requirements

| Stage | VRAM Usage |
|-------|------------|
| Model Loading | ~4-5 GB |
| Training Peak | ~18-22 GB |
| Inference | ~4-5 GB |

### Run Training

```python
# In the notebook (Section 9):
trainer.train()
trainer.save_model("./qwen3-resume-lora-single-gpu")
tokenizer.save_pretrained("./qwen3-resume-lora-single-gpu")
```

---

## 🌐 REST API

The project includes a FastAPI-based REST API for easy integration.

### Quick Start

```bash
# Install API dependencies
pip install -r api_requirements.txt

# Start the server
python run_api.py
```

API will be available at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check model status |
| `/schema` | GET | Get resume JSON schema |
| `/optimize` | POST | Upload resume file + job description |
| `/optimize/text` | POST | Submit resume text + job description |

### Example Usage

```bash
# Upload resume file with job description
curl -X POST "http://localhost:8000/optimize" \
  -F "resume_file=@resume.pdf" \
  -F "job_description=We are looking for a Senior Software Engineer..."

# Or with job URL (auto-scraped)
curl -X POST "http://localhost:8000/optimize" \
  -F "resume_file=@resume.pdf" \
  -F "job_url=https://linkedin.com/jobs/view/123456"
```

📖 **Full API Documentation:** See [`api/README.md`](api/README.md) or visit `/docs` when server is running.

---

## 🔍 Inference

### Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LORA_PATH = "./qwen3-resume-lora-single-gpu"

# Load with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load and merge LoRA for faster inference
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.eval()

# Generate function
def generate(messages):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,  # Deterministic for JSON
            do_sample=False,
        )
    
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

### Example Output

```json
{
  "personal_information": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "location": "San Francisco, CA"
  },
  "summary": "Experienced software engineer with 5+ years...",
  "experiences": [
    {
      "designation": "Senior Software Engineer",
      "companyName": "Tech Corp",
      "location": "San Francisco, CA",
      "start_date": "2020",
      "end_date": "Present",
      "points": [
        "Built scalable APIs serving 1M+ requests/day",
        "Led team of 5 developers"
      ]
    }
  ],
  "skills": [
    {"name": "Programming Languages", "data": ["Python", "JavaScript", "Go"]}
  ]
}
```

---

## 📈 Model Performance

### Evaluation Results (GPT-5.1 Thinking LLM)

🔗 [Full Evaluation Link](https://chatgpt.com/s/t_69309bb9c0a0819191373e1d9cbe86d9)

| Output | Source | Score | Verdict |
|--------|--------|:-----:|---------|
| **Output 4** | Fine-tuned (best params) | ⭐ **9.5/10** | Best - schema consistent, no hallucinations |
| **Output 2** | Base model | ⭐ **9/10** | Strong baseline |
| **Output 1** | Fine-tuned (bad data) | ⭐ **7/10** | Data quality issues |
| **Output 3** | Old training data | ⭐ **2/10** | Invalid JSON, hallucinations |

### Quality Criteria

| ✅ Good Training Data | ❌ Bad Training Data |
|----------------------|---------------------|
| Valid JSON structure | Invalid JSON syntax |
| Schema adherence | Schema violations |
| No hallucinations | Invented data |
| Job-aligned content | Generic content |
| Strong action verbs | Weak phrasing |

### Performance Stats

| Metric | Value |
|--------|-------|
| Generation Time | ~3-5 seconds (RTX 3090) |
| Max Sequence Length | 4096 tokens |
| Output Format | Deterministic JSON |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | Reduce `gradient_accumulation_steps` to 4 |
| **Flash Attention unavailable** | Falls back to "eager" automatically |
| **JSON parse failures** | Use `temperature=0.0`, increase `max_new_tokens` |
| **LoRA merge issues** | Update to `transformers>=4.35`, `peft>=0.6` |

### Memory Cleanup

```python
import gc, torch
del trainer, model
gc.collect()
torch.cuda.empty_cache()
```

---

## 🤝 Contributing

Contributions welcome! Ideas for improvement:

- [ ] Longer context window support
- [ ] Multi-language resume optimization
- [ ] Industry-specific fine-tuning
- [ ] Web interface
- [ ] GGUF export for llama.cpp

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

> **Note:** The Qwen3 base model has its own [license terms](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

---

## 🙏 Acknowledgments

- **Qwen Team** - Qwen3-4B-Instruct base model
- **Hugging Face** - Transformers, PEFT, TRL libraries
- **Tim Dettmers** - bitsandbytes quantization

---

## 📚 References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Efficient Fine-tuning of Quantized LLMs
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) - Base Model
- [TRL Docs](https://huggingface.co/docs/trl) - Training Library
- [PEFT Docs](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning

---

<p align="center">
  <b>⭐ If you find this project helpful, please give it a star!</b>
</p>
