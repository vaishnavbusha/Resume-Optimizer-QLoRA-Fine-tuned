

import pandas as pd
import requests
import json
import os
import time
import re
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from pprint import pprint

RESUME_SCHEMA = '''{"$schema":"http://json-schema.org/draft-04/schema#","type":"object","properties":{"personal_information":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"},"phone":{"type":"string"},"location":{"type":"string"},"socials":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"link":{"type":"string"}},"required":["name","link"]}]}},"required":["name","email","phone","location"]},"summary":{"type":"string"},"experiences":{"type":"array","items":[{"type":"object","properties":{"designation":{"type":"string"},"companyName":{"type":"string"},"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"points":{"type":"array","items":[{"type":"string"}]}},"required":["designation","companyName","location","start_date"]}]},"education":{"type":"array","items":[{"type":"object","properties":{"institution":{"type":"string"},"degree":{"type":"string"},"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"gpa":{"type":"string"}},"required":["institution","degree","location","start_date","gpa"]}]},"skills":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"data":{"type":"array","items":[{"type":"string"}]}},"required":["name","data"]}]},"projects":{"type":"array","items":[{"type":"object","properties":{"projectName":{"type":"string"},"caption":{"type":"string"},"location":{"type":"string"},"start_date":{"type":"string"},"end_date":{"type":"string"},"url":{"type":"string"},"projectDetails":{"type":"array","items":[{"type":"string"}]},"externalSources":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"link":{"type":"string"}},"required":["name","link"]}]},"technologiesUsed":{"type":"array","items":[{"type":"string"}]}},"required":["projectName","location","projectDetails"]}]},"certifications":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"issuing_organization":{"type":"string"},"issue_date":{"type":"string"},"expiration_date":{"type":"string"},"credential_id":{"type":"string"},"url":{"type":"string"}},"required":["name","issuing_organization","issue_date","expiration_date","credential_id","url"]}]},"awards":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"type":{"type":"string"},"location":{"type":"string"},"date":{"type":"string"},"description":{"type":"string"}},"required":["name","type","location","date","description"]}]},"extracurricular_achievements":{"type":"array","items":[{"type":"object","properties":{"name":{"type":"string"},"type":{"type":"string"},"location":{"type":"string"},"date":{"type":"string"},"description":{"type":"string"}},"required":["name","type","location","date","description"]}]},"languages":{"type":"array","items":[{"type":"object","properties":{"language":{"type":"string"},"proficiency":{"type":"string"}},"required":["language","proficiency"]}]}},"required":["personal_information","education","skills","extracurricular_achievements"]}'''

# Load prompt template
with open('dataset/prompt.txt', 'r', encoding='utf-8') as f:
    PROMPT_TEMPLATE = f.read()

class OllamaResumeGenerator:
    """Class to generate tailored resumes using local Ollama model (supports thinking models)."""
    
    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        output_path: str = "dataset/tailored_resumes",  # Base name without extension
        timeout: int = 300,  # Increased for thinking models
        max_retries: int = 3,
        enable_thinking: bool = True  # Enable thinking mode for supported models
    ):
        """
        Initialize the Ollama Resume Generator.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
            output_path: Base path for output file (timestamp will be added)
            timeout: Request timeout in seconds (higher for thinking models)
            max_retries: Maximum number of retries on failure
            enable_thinking: Whether to enable thinking mode (adds /think suffix)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        
        # Add timestamp to output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = Path(f"{output_path}_{timestamp}.jsonl")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store session start time
        self.session_start = datetime.now().isoformat()
        
    def _build_prompt(self, resume_text: str, job_description: str) -> str:
        """Build the prompt by filling in the template."""
        prompt = PROMPT_TEMPLATE.replace("<<PASTE RESUME HERE>>", resume_text)
        prompt = prompt.replace("<<PASTE JD HERE>>", job_description)
        prompt = prompt.replace("<<PASTE JSON SCHEMA HERE>>", RESUME_SCHEMA)
        pprint(prompt)
        return prompt
    
    def _extract_thinking_and_response(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract thinking content and actual response from thinking model output.
        
        Returns:
            Tuple of (thinking_content, actual_response)
        """
        if not response:
            return None, ""
        
        thinking_content = None
        actual_response = response
        
        # Pattern to match <think>...</think> blocks
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response, re.DOTALL)
        
        if think_match:
            thinking_content = think_match.group(1).strip()
            # Remove the thinking block from response
            actual_response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
        
        return thinking_content, actual_response
    
    def _call_ollama(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Make a request to the Ollama API.
        
        Returns:
            Tuple of (response, thinking_content)
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 8192  # Increased for thinking models
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                raw_response = result.get("response", "")
                
                # Extract thinking and actual response
                thinking, actual_response = self._extract_thinking_and_response(raw_response)
                
                return actual_response, thinking
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None, None
    
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from the model response."""
        if not response:
            return None
        
        # Try to find JSON in the response
        response = response.strip()
        
        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try generic code blocks
        if "```" in response:
            start = response.find("```") + 3
            # Skip language identifier if present
            newline_pos = response.find("\n", start)
            if newline_pos > start:
                start = newline_pos + 1
            end = response.find("```", start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Try to extract JSON between curly braces
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass
        
        return None
    
    def generate_single(self, resume_text: str, job_description: str, filename: str) -> Dict[str, Any]:
        """Generate a tailored resume for a single resume-job pair."""
        start_time = datetime.now()
        prompt = self._build_prompt(resume_text, job_description)
        response, thinking_content = self._call_ollama(prompt)
        end_time = datetime.now()
        
        result = {
            "filename": filename,
            "original_resume": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
            "job_description": job_description[:500] + "..." if len(job_description) > 500 else job_description,
            "status": "success",
            "tailored_resume": None,
            "raw_response": None,
            "thinking_content": thinking_content,  # Store the model's reasoning
            "timestamp": end_time.isoformat(),
            "processing_time_seconds": (end_time - start_time).total_seconds()
        }
        
        if response:
            parsed_json = self._extract_json(response)
            if parsed_json:
                result["tailored_resume"] = parsed_json
            else:
                result["status"] = "json_parse_error"
                result["raw_response"] = response[:2000] if len(response) > 2000 else response
        else:
            result["status"] = "api_error"
        
        return result
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        resume_col: str = "resume_text",
        job_col: str = "job_description",
        filename_col: str = "filename",
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        save_every: int = 10
    ) -> None:
        """
        Process a DataFrame and generate tailored resumes.
        
        Args:
            df: DataFrame with resume and job description columns
            resume_col: Name of the resume text column
            job_col: Name of the job description column
            filename_col: Name of the filename column
            start_idx: Starting index for processing
            end_idx: Ending index for processing (None = process all)
            save_every: Save progress after every N records
        """
        if end_idx is None:
            end_idx = len(df)
        
        df_subset = df.iloc[start_idx:end_idx]
        
        successful = 0
        failed = 0
        
        batch_start_time = datetime.now()
        print(f"{'='*50}")
        print(f"Starting processing at: {batch_start_time.isoformat()}")
        print(f"Model: {self.model_name}")
        print(f"Thinking mode: {'Enabled' if self.enable_thinking else 'Disabled'}")
        print(f"Output file: {self.output_path}")
        print(f"Processing records {start_idx} to {end_idx} ({len(df_subset)} total)")
        print(f"{'='*50}\n")
        
        # Open file in append mode
        with open(self.output_path, 'a', encoding='utf-8') as fh:
            for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Generating resumes"):
                resume_text = row[resume_col]
                job_description = row[job_col]
                filename = row[filename_col]
                
                result = self.generate_single(resume_text, job_description, filename)
                result["original_index"] = idx
                result["session_start"] = self.session_start
                result["model_used"] = self.model_name
                
                # Write to file immediately
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                if result["status"] == "success":
                    successful += 1
                else:
                    failed += 1
                
                # Flush periodically
                if (successful + failed) % save_every == 0:
                    fh.flush()
        
        batch_end_time = datetime.now()
        total_time = (batch_end_time - batch_start_time).total_seconds()
        
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Model:      {self.model_name}")
        print(f"Started:    {batch_start_time.isoformat()}")
        print(f"Finished:   {batch_end_time.isoformat()}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Successful: {successful}")
        print(f"Failed:     {failed}")
        if successful + failed > 0:
            print(f"Avg time/record: {total_time/(successful+failed):.2f} seconds")
        print(f"Results saved to: {self.output_path}")
        print(f"{'='*50}")
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            model_base_names = [m.split(":")[0] for m in model_names]
            
            print(f"Ollama is running. Available models: {model_names}")
            
            model_base = self.model_name.split(":")[0]
            if self.model_name in model_names or model_base in model_base_names:
                print(f"✓ Model '{self.model_name}' is available")
                if "qwen3" in self.model_name.lower():
                    print(f"  Note: Qwen3 thinking model detected - extended timeout set to {self.timeout}s")
                return True
            else:
                print(f"✗ Model '{self.model_name}' not found. Please run: ollama pull {self.model_name}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to Ollama at {self.base_url}")
            print(f"  Error: {e}")
            print(f"  Make sure Ollama is running: ollama serve")
            return False


print("OllamaResumeGenerator class defined successfully!")


generator = OllamaResumeGenerator(
    model_name="qwen3:8b",  # Change to your preferred model
    output_path="dataset/tailored_resumes/tailored_resumes",
    timeout=180,
    max_retries=3
)

# Check if Ollama is running
generator.check_ollama_status()
DATA_DIR = ".//dataset"
df = pd.read_parquet(os.path.join(DATA_DIR, 'resume_dataset.parquet'), engine='pyarrow')
generator.process_dataframe(
    df,
    resume_col="resume_text",
    job_col="job_description",
    filename_col="filename",
    start_idx=0,
    end_idx=1,  # Change to None to process all 1565 records
    save_every=1
)
print("Processing complete!")