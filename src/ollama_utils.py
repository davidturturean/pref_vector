"""
Ollama-based utilities for model loading and inference.
Replaces the transformers-based approach with Ollama API calls.
"""

import json
import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .config import get_model_config, get_ollama_model_name

logger = logging.getLogger(__name__)

@dataclass
class OllamaResponse:
    """Structured response from Ollama API."""
    text: str
    tokens: List[str]
    logits: Optional[List[float]] = None
    embeddings: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class OllamaClient:
    """Client for interacting with Ollama models."""
    
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_name = get_ollama_model_name(model_name)
        self.host = host.rstrip('/')
        self.config = get_model_config(model_name)
        
        # Ensure model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Pull model if not available locally."""
        try:
            # Check if model exists
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.ollama_name not in model_names:
                    logger.info(f"Pulling model {self.ollama_name}...")
                    self._pull_model()
                else:
                    logger.info(f"Model {self.ollama_name} already available")
            else:
                logger.warning(f"Could not check model availability: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            raise
    
    def _pull_model(self):
        """Pull model from Ollama registry."""
        try:
            data = {"name": self.ollama_name}
            response = requests.post(f"{self.host}/api/pull", json=data, stream=True)
            
            for line in response.iter_lines():
                if line:
                    status = json.loads(line)
                    if 'status' in status:
                        logger.info(f"Pull status: {status['status']}")
                    if status.get('status') == 'success':
                        logger.info(f"Successfully pulled {self.ollama_name}")
                        break
                        
        except Exception as e:
            logger.error(f"Failed to pull model {self.ollama_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> OllamaResponse:
        """Generate text using Ollama model."""
        try:
            data = {
                "model": self.ollama_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "top_p": kwargs.get('top_p', self.config.top_p),
                    "num_ctx": kwargs.get('num_ctx', self.config.num_ctx),
                }
            }
            
            if max_tokens:
                data["options"]["num_predict"] = max_tokens
            
            response = requests.post(f"{self.host}/api/generate", json=data)
            response.raise_for_status()
            
            result = response.json()
            
            return OllamaResponse(
                text=result.get('response', ''),
                tokens=result.get('response', '').split(),  # Simple tokenization
                metadata={
                    'model': self.ollama_name,
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'eval_count': result.get('eval_count', 0),
                    'total_duration': result.get('total_duration', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama."""
        try:
            data = {
                "model": self.ollama_name,
                "prompt": text
            }
            
            response = requests.post(f"{self.host}/api/embeddings", json=data)
            response.raise_for_status()
            
            result = response.json()
            return result.get('embedding', [])
            
        except Exception as e:
            logger.warning(f"Embeddings not available for {self.ollama_name}: {e}")
            return []
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[OllamaResponse]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
            time.sleep(0.1)  # Small delay to avoid overwhelming the API
        return results

def load_ollama_model(model_name: str, **kwargs) -> OllamaClient:
    """Load an Ollama model client."""
    logger.info(f"Loading Ollama model: {model_name}")
    return OllamaClient(model_name, **kwargs)

def check_ollama_status(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_available_models(host: str = "http://localhost:11434") -> List[str]:
    """List all available Ollama models."""
    try:
        response = requests.get(f"{host}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        return []
    except:
        return []

# Compatibility layer for existing transformers-based code
class OllamaModelWrapper:
    """Wrapper to make Ollama client compatible with transformers-style usage."""
    
    def __init__(self, model_name: str):
        self.client = OllamaClient(model_name)
        self.model_name = model_name
        self.config = self.client.config
    
    def generate(self, input_ids=None, attention_mask=None, max_length=None, **kwargs):
        """Transformers-style generate method."""
        if hasattr(input_ids, 'input_ids'):
            # If it's a tokenizer output, extract the text
            prompt = str(input_ids)  # Simplified - would need proper detokenization
        else:
            prompt = str(input_ids) if input_ids else ""
        
        response = self.client.generate(prompt, max_tokens=max_length, **kwargs)
        return MockGenerateOutput(response.text)
    
    def __call__(self, text: str, **kwargs):
        """Direct text generation."""
        return self.client.generate(text, **kwargs)

class MockGenerateOutput:
    """Mock output to match transformers GenerateOutput interface."""
    def __init__(self, text: str):
        self.sequences = [text]
        self.text = text