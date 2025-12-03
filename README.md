# Resume Optimizer - QLoRA Fine-tuned Qwen3-4B-Instruct

A fine-tuned language model for generating tailored resumes based on job descriptions. This project uses QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune the Qwen3-4B-Instruct model for resume optimization and JSON structured output.

## 📋 Overview

This project fine-tunes a 4B parameter language model to:
- **Tailor resumes** to match specific job descriptions
- **Generate structured JSON** output following a predefined schema
- **Extract and reorganize** resume content for optimal job matching

The model is trained using QLoRA with 4-bit quantization, enabling efficient fine-tuning on consumer GPUs (tested on NVIDIA RTX 3090 with 24GB VRAM).

## 🚀 Features

- **QLoRA Training**: Memory-efficient 4-bit NF4 quantization with LoRA adapters
- **Structured JSON Output**: Generates valid JSON following a comprehensive resume schema
- **Chat Template Format**: Uses Qwen's native chat template for training and inference
- **Flash Attention 2**: Optional support for faster inference on compatible GPUs
- **Merge and Unload**: Faster inference with merged LoRA weights

## 📁 Project Structure

```
QLoRA-finetuning/
├── final_project.ipynb                            # Main training/inference notebook
├── dataset/
│   ├── training/
│   │   └── final_training_dataset.jsonl          # Training data
│   ├── final_resume_dataset.jsonl                # Resume dataset
│   └── batch_requests/                           # Batch processing files
├── qwen3-resume-lora/                            # LoRA adapter weights (generated)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── environment.yml                                # Conda environment
└── README.md                                      # This file
```

## 🛠️ Installation

### Prerequisites

- **Python**: 3.11+
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU**: NVIDIA GPU with 24GB+ VRAM recommended
- **Conda**: For environment management

### Setup Environment

1. **Clone the repository**
```bash
git clone https://github.com/Abhinav1426/QLoRA-finetuning.git
cd QLoRA-finetuning
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate finetune
```

Or install dependencies manually:
```bash
conda create -n finetune python=3.11
conda activate finetune
pip install torch transformers datasets accelerate trl peft bitsandbytes
```

3. **Verify GPU availability**
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```

## 📊 Dataset Format

The training dataset uses JSONL format with chat-style messages:

```json
{
  "messages": [
    {"role": "system", "content": "You create a tailored resume based on the job description..."},
    {"role": "user", "content": "RESUME_TEXT:\n...\n\nJOB_DESCRIPTION:\n...\n\nSCHEMA:\n..."},
    {"role": "assistant", "content": "{...JSON resume output...}"}
  ]
}
```

### Output Schema

The model generates JSON following this schema structure:
- `personal_information`: name, email, phone, location, socials
- `summary`: Professional summary tailored to the job
- `experiences`: Work experience with designations, companies, dates, and bullet points
- `education`: Degrees, institutions, locations, dates, GPA
- `skills`: Categorized skills relevant to the job
- `projects`: Project details with technologies used
- `certifications`: Professional certifications
- `awards`: Awards and recognitions
- `extracurricular_achievements`: Additional achievements
- `languages`: Language proficiencies

## 🎯 Training

### Quick Start

1. **Prepare your dataset** in the required JSONL format
2. **Run the training cell** in the notebook:

```python
#!/usr/bin/env python
"""
Single-GPU QLoRA fine-tuning for Qwen3-4B-Instruct on resume->JSON data.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATA_PATH = "./train_clean.jsonl"
OUTPUT_DIR = "./qwen3-resume-lora-single-gpu"
MAX_SEQ_LENGTH = 4096

# Training configuration
trainer.train()
trainer.save_model(OUTPUT_DIR)
```

### Training Configuration

```python
Model: Qwen/Qwen3-4B-Instruct-2507
Quantization: 4-bit NF4 with double quantization
Compute dtype: float16

LoRA Config:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Bias: none
  - Task type: CAUSAL_LM
  - Target modules: q_proj, k_proj, v_proj, o_proj

Training Parameters:
  - Batch size: 1 (per device)
  - Gradient accumulation: 8 steps
  - Effective batch size: 8
  - Learning rate: 2e-4
  - Epochs: 2
  - Max sequence length: 4096 tokens
  - FP16: Enabled
  - Warmup steps: 50
  - Logging steps: 20
  - Save steps: 500
```

### Memory Requirements

- **Model loading**: ~4-5 GB VRAM (4-bit quantized)
- **Training peak**: ~18-22 GB VRAM
- **Inference**: ~4-5 GB VRAM

## 🔍 Usage

### Inference with LoRA Adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import importlib.util

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LORA_PATH = "./qwen3-resume-lora-single-gpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Check if flash_attn is available
flash_attn_available = importlib.util.find_spec("flash_attn") is not None
use_flash_attn = flash_attn_available and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter and merge for faster inference
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
model.eval()

def generate(messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,      # Deterministic for JSON output
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return output_text

# Example usage
messages = [
    {"role": "system", "content": "You create a tailored resume based on the job description..."},
    {"role": "user", "content": "RESUME_TEXT:\n...\n\nJOB_DESCRIPTION:\n...\n\nSCHEMA:\n..."},
]

output = generate(messages)
```

### Expected Output Format

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
  "education": [...],
  "skills": [
    {
      "name": "Programming Languages",
      "data": ["Python", "JavaScript", "Go"]
    }
  ],
  "projects": [...],
  "certifications": [...],
  "extracurricular_achievements": [...]
}
```

## 🧹 Memory Management

### Clean GPU Memory After Training

```python
import torch
import gc

# Delete model objects to free memory
if 'trainer' in dir():
    del trainer
if 'model' in dir():
    del model
if 'base_model' in dir():
    del base_model
if 'inference_model' in dir():
    del inference_model

# Clear Python garbage collector
gc.collect()

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## 📈 Model Performance

### Key Metrics

- **Average Generation Time**: ~3-5 seconds per resume (on RTX 3090)
- **Max Sequence Length**: 4096 tokens
- **Output Format**: Deterministic JSON (temperature=0.0)

### Tips for Best Results

1. Use `temperature=0.0` and `do_sample=False` for consistent JSON output
2. Use `merge_and_unload()` for faster, more stable inference
3. Ensure the prompt follows the exact training format with RESUME_TEXT, JOB_DESCRIPTION, and SCHEMA sections
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size or sequence length
per_device_train_batch_size=1  # Already at minimum
gradient_accumulation_steps=4   # Reduce from 8
```

**2. Flash Attention Not Available**
```python
# The code automatically falls back to "eager" attention
# Or install flash-attn:
pip install flash-attn --no-build-isolation
```

**3. JSON Parse Failures**
```python
# Solution: Ensure temperature=0.0 for deterministic output
# Increase max_new_tokens if output is truncated
```

**4. LoRA Merge Issues with Quantized Models**
```python
# Note: merge_and_unload() works with 4-bit quantized models in newer versions
# Ensure you have transformers>=4.35 and peft>=0.6
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for longer context windows
- [ ] Multi-language resume optimization
- [ ] Fine-tuning on industry-specific datasets
- [ ] Web interface for easy inference
- [ ] Batch processing scripts
- [ ] GGUF export for llama.cpp deployment

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

**Note**: The Qwen3-4B-Instruct base model has its own license terms. Please review the [Qwen license](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) before commercial use.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen3-4B-Instruct base model
- **Hugging Face** for transformers, PEFT, and TRL libraries
- **Tim Dettmers** for bitsandbytes quantization

## 📧 Contact

- **Author**: Abhinav
- **GitHub**: [@Abhinav1426](https://github.com/Abhinav1426)
- **Repository**: [QLoRA-finetuning](https://github.com/Abhinav1426/QLoRA-finetuning)

## 📚 References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [TRL Library](https://huggingface.co/docs/trl)
- [PEFT Library](https://huggingface.co/docs/peft)

---

**⭐ If you find this project helpful, please consider giving it a star!**
