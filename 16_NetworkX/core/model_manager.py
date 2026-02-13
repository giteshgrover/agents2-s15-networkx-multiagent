import os
import time
import asyncio
import json
import yaml
import requests
from pathlib import Path
from google import genai
from google.genai.errors import ServerError
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import pdb

load_dotenv()

ROOT = Path(__file__).parent.parent
MODELS_JSON = ROOT / "config" / "models.json"
PROFILE_YAML = ROOT / "config" / "profiles.yaml"

class ModelManager:
    def __init__(self, model_name: str = None):
        self.config = json.loads(MODELS_JSON.read_text())
        self.profile = yaml.safe_load(PROFILE_YAML.read_text())

        # 🎯 NEW: Use provided model_name or fall back to profile default
        if model_name:
            self.text_model_key = model_name
        else:
            self.text_model_key = self.profile["llm"]["text_generation"]
        
        # Validate that the model exists in config
        if self.text_model_key not in self.config["models"]:
            available_models = list(self.config["models"].keys())
            raise ValueError(f"Model '{self.text_model_key}' not found in models.json. Available: {available_models}")
            
        self.model_info = self.config["models"][self.text_model_key]
        self.model_type = self.model_info["type"]

        # pdb.set_trace()
        # Initialize client based on model type
        if self.model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
        # Add other model types as needed

    async def generate_text(self, prompt: str) -> str:
        if self.model_type == "gemini":
            return await self._gemini_generate(prompt)

        elif self.model_type == "ollama":
            return await self._ollama_generate(prompt)

        raise NotImplementedError(f"Unsupported model type: {self.model_type}")

    async def generate_content(self, contents: list) -> str:
        """Generate content with support for text and images"""
        if self.model_type == "gemini":
            return await self._gemini_generate_content(contents)
        elif self.model_type == "ollama":
            # Ollama doesn't support images, fall back to text-only
            text_content = ""
            for content in contents:
                if isinstance(content, str):
                    text_content += content
            return await self._ollama_generate(text_content)
        
        raise NotImplementedError(f"Unsupported model type: {self.model_type}")

    # --- Rate Limiting Helper ---
    _last_call = time.time()
    _lock = asyncio.Lock()
    _min_interval = 4.0  # Minimum 4 seconds between calls (15 RPM = 4s interval)

    async def _wait_for_rate_limit(self):
        """Enforce rate limit for Gemini - ensures sequential API calls"""
        async with ModelManager._lock:
            now = time.time()
            elapsed = now - ModelManager._last_call
            if elapsed < ModelManager._min_interval:
                sleep_time = ModelManager._min_interval - elapsed
                print(f"[Rate Limit] Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            # Don't update _last_call here - update it after successful API call


    async def _gemini_generate(self, prompt: str) -> str:
        """Generate text with retry logic for rate limiting"""
        return await self._gemini_generate_with_retry(
            lambda: self.client.aio.models.generate_content(
                model=self.model_info["model"],
                contents=prompt
            )
        )

    async def _gemini_generate_with_retry(self, api_call_func, max_retries=5):
        """Execute Gemini API call with exponential backoff retry for 429 errors"""
        await self._wait_for_rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = await api_call_func()
                # Update timestamp AFTER successful call
                async with ModelManager._lock:
                    ModelManager._last_call = time.time()
                return response.text.strip()
            
            except ServerError as e:
                error_str = str(e)
                error_code = getattr(e, 'code', None)
                
                # Check if it's a 429 RESOURCE_EXHAUSTED error
                # The error can appear as: "429 RESOURCE_EXHAUSTED" or in error dict
                is_rate_limit_error = (
                    error_code == 429 or
                    '429' in error_str or
                    'RESOURCE_EXHAUSTED' in error_str.upper() or
                    'Resource exhausted' in error_str
                )
                
                if is_rate_limit_error:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2^attempt * base_delay
                        base_delay = 30  # Start with 30 seconds
                        backoff_delay = min(base_delay * (2 ** attempt), 300)  # Cap at 5 minutes
                        print(f"[Rate Limit] 429 error on attempt {attempt + 1}/{max_retries}. Retrying in {backoff_delay}s...")
                        
                        # Update last_call to account for the backoff delay
                        async with ModelManager._lock:
                            ModelManager._last_call = time.time() + backoff_delay
                        
                        await asyncio.sleep(backoff_delay)
                        # Re-acquire rate limit before retrying
                        await self._wait_for_rate_limit()
                        continue
                    else:
                        raise RuntimeError(f"Gemini generation failed after {max_retries} retries: {error_str}")
                else:
                    # For non-429 errors, raise immediately
                    raise RuntimeError(f"Gemini generation failed: {error_str}")
            
            except Exception as e:
                error_str = str(e)
                # Also check for 429 in generic exceptions (in case error is wrapped)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str.upper():
                    if attempt < max_retries - 1:
                        base_delay = 30
                        backoff_delay = min(base_delay * (2 ** attempt), 300)
                        print(f"[Rate Limit] 429 error (wrapped) on attempt {attempt + 1}/{max_retries}. Retrying in {backoff_delay}s...")
                        
                        async with ModelManager._lock:
                            ModelManager._last_call = time.time() + backoff_delay
                        
                        await asyncio.sleep(backoff_delay)
                        await self._wait_for_rate_limit()
                        continue
                
                # For other exceptions, raise immediately
                raise RuntimeError(f"Gemini generation failed: {error_str}")
        
        raise RuntimeError(f"Gemini generation failed after {max_retries} attempts")

    async def _gemini_generate_content(self, contents: list) -> str:
        """Generate content with support for text and images using Gemini"""
        return await self._gemini_generate_with_retry(
            lambda: self.client.aio.models.generate_content(
                model=self.model_info["model"],
                contents=contents
            )
        )

    async def _ollama_generate(self, prompt: str) -> str:
        try:
            # ✅ Use aiohttp for truly async requests
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.model_info["url"]["generate"],
                    json={"model": self.model_info["model"], "prompt": prompt, "stream": False}
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
