"""
Model Service Module.
Handles loading and inference with the QLoRA fine-tuned model.
"""

import json
import gc
import importlib.util
from typing import Optional, Dict, Any
import time

import torch

# Transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from .config import settings, RESUME_SCHEMA, SYSTEM_PROMPT


class ModelService:
    """
    Service for loading and running inference with the fine-tuned model.
    Uses singleton pattern to ensure model is loaded only once.
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    _is_loaded = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model service."""
        pass  # Actual initialization in load_model()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()
    
    @property
    def gpu_name(self) -> Optional[str]:
        """Get GPU name if available."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return None
    
    def load_model(self) -> bool:
        """
        Load the model and tokenizer.
        
        Returns:
            True if loading was successful
        """
        if self._is_loaded:
            print("✅ Model already loaded")
            return True
        
        try:
            print(f"🔄 Loading model: {settings.MODEL_NAME}")
            print(f"🔄 LoRA adapter: {settings.LORA_ADAPTER_PATH}")
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Check Flash Attention availability
            flash_attn_available = importlib.util.find_spec("flash_attn") is not None
            use_flash_attn = (
                flash_attn_available 
                and torch.cuda.is_available() 
                and torch.cuda.get_device_properties(0).major >= 8
            )
            
            if use_flash_attn:
                print("✅ Using Flash Attention 2")
            else:
                print("ℹ️ Using eager attention")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="flash_attention_2" if use_flash_attn else "eager",
                torch_dtype=torch.bfloat16,
            )
            
            # Load LoRA adapter
            print("🔄 Loading LoRA adapter...")
            self._model = PeftModel.from_pretrained(base_model, settings.LORA_ADAPTER_PATH)
            
            # Merge and unload for faster inference
            print("🔄 Merging LoRA weights...")
            self._model = self._model.merge_and_unload()
            self._model.eval()
            
            # Clean up base model reference
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Load tokenizer
            print("🔄 Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(settings.LORA_ADAPTER_PATH)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._is_loaded = True
            print("✅ Model loaded successfully!")
            
            # Print memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            self._is_loaded = False
            return False
    
    def generate(
        self,
        resume_text: str,
        job_description: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate a tailored resume.
        
        Args:
            resume_text: Extracted text from the resume
            job_description: Job description text
            max_new_tokens: Maximum tokens to generate (default from settings)
            temperature: Sampling temperature (default from settings)
        
        Returns:
            Dictionary with generation results
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Build the prompt
        user_content = self._build_user_prompt(resume_text, job_description)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        
        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or settings.MAX_NEW_TOKENS,
                temperature=temperature or settings.TEMPERATURE,
                do_sample=False if (temperature or settings.TEMPERATURE) == 0 else True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        # Try to parse as JSON
        parsed_json = None
        parse_error = None
        
        try:
            # Clean up response (remove markdown code blocks if present)
            cleaned_response = self._clean_json_response(response)
            parsed_json = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            parse_error = str(e)
        
        return {
            "raw_output": response,
            "parsed_json": parsed_json,
            "parse_error": parse_error,
            "generation_time": generation_time,
            "tokens_generated": len(generated_tokens),
        }
    
    def _build_user_prompt(self, resume_text: str, job_description: str) -> str:
        """Build the user prompt with resume, job description, and schema."""
        schema_str = json.dumps(RESUME_SCHEMA)
        
        return f"""
RESUME_TEXT:
\"\"\"
{resume_text}
\"\"\"

JOB_DESCRIPTION:
\"\"\"
{job_description}
\"\"\"

SCHEMA:
\"\"\"
{schema_str}
\"\"\"

Create a tailored resume in JSON following the SCHEMA exactly.
Use only content from the RESUME_TEXT but rewrite it to match the JOB_DESCRIPTION.
Do not add extra lines or explanation.
Output only JSON.
"""
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown code blocks."""
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        return response.strip()
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        self._is_loaded = False
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("✅ Model unloaded")


# Singleton instance
model_service = ModelService()
