"""
Flexible Model Loading Infrastructure for Cross-Model Analysis

This module provides a unified interface for loading and managing different LLMs
through Ollama, enabling seamless switching between model families and architectures.
"""

import time
import requests
import subprocess
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
from pathlib import Path

from .config import (
    get_model_config, get_ollama_model_name, HF_TO_OLLAMA_MODELS,
    MODEL_FAMILIES, ModelConfig
)

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    hf_name: str
    ollama_name: str
    family: str
    size: str  # e.g., "7b", "13b"
    architecture: str  # e.g., "llama", "mistral", "gemma"
    hidden_size: int
    num_layers: int
    vocab_size: Optional[int] = None
    context_length: Optional[int] = None
    is_instruct: bool = False
    is_loaded: bool = False

class OllamaModelManager:
    """Manages Ollama server and model lifecycle."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.server_process = None
        self.loaded_models = {}
        self._ensure_server_running()
    
    def _ensure_server_running(self) -> bool:
        """Ensure Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server is already running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info("Starting Ollama server...")
        try:
            # Try to start ollama serve in background
            self.server_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.host}/api/tags", timeout=2)
                    if response.status_code == 200:
                        logger.info("Ollama server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            logger.error("Ollama server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama server: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """List models available in Ollama."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not already available."""
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                # Stream the response to show progress
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            if data["status"] == "success":
                                logger.info(f"Successfully pulled {model_name}")
                                return True
                            elif "error" in data:
                                logger.error(f"Error pulling {model_name}: {data['error']}")
                                return False
            return False
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate(self, model_name: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using specified model."""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = requests.post(f"{self.host}/api/generate", json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a model."""
        try:
            response = requests.post(
                f"{self.host}/api/show",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the Ollama server if we started it."""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("Ollama server shut down")

class ModelLoader:
    """Main interface for loading and managing models across different families."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.ollama_manager = OllamaModelManager(host)
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_cache: Dict[str, Any] = {}
    
    def load_model(self, model_name: str, **config_overrides) -> ModelInfo:
        """Load a model and return ModelInfo object."""
        
        # Check if already loaded
        if model_name in self.loaded_models and self.loaded_models[model_name].is_loaded:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        # Get model configuration
        model_config = get_model_config(model_name, **config_overrides)
        ollama_name = model_config.ollama_name
        
        logger.info(f"Loading model: {model_name} -> {ollama_name}")
        
        # Check if model is available, pull if necessary
        available_models = self.ollama_manager.list_available_models()
        if ollama_name not in available_models:
            logger.info(f"Model {ollama_name} not found locally, pulling...")
            if not self.ollama_manager.pull_model(ollama_name):
                raise RuntimeError(f"Failed to pull model {ollama_name}")
        
        # Extract model information
        model_info = self._extract_model_info(model_name, ollama_name)
        
        # Test the model to ensure it's working
        if self._test_model(ollama_name):
            model_info.is_loaded = True
            self.loaded_models[model_name] = model_info
            logger.info(f"Successfully loaded {model_name}")
        else:
            raise RuntimeError(f"Model {model_name} failed validation test")
        
        return model_info
    
    def _extract_model_info(self, hf_name: str, ollama_name: str) -> ModelInfo:
        """Extract detailed information about a model."""
        
        # Get detailed model info from Ollama
        ollama_info = self.ollama_manager.get_model_info(ollama_name)
        
        # Parse model name to extract information
        family = self._get_model_family(hf_name)
        size = self._extract_model_size(hf_name)
        architecture = self._get_model_architecture(hf_name)
        is_instruct = "instruct" in hf_name.lower() or "chat" in hf_name.lower() or "it" in hf_name.lower()
        
        # Default hidden sizes based on model size
        hidden_size_map = {
            "0.5b": 1024, "1b": 2048, "2b": 2048, "3b": 3072,
            "6b": 4096, "7b": 4096, "8b": 4096, "9b": 4608,
            "13b": 5120, "14b": 5120, "20b": 6144, "30b": 6656,
            "34b": 7168, "65b": 8192, "70b": 8192, "72b": 8192
        }
        
        hidden_size = hidden_size_map.get(size, 4096)  # Default to 4096 for 7B-class models
        
        # Estimate number of layers (rough approximation)
        layer_map = {
            "0.5b": 16, "1b": 22, "2b": 24, "3b": 26,
            "6b": 32, "7b": 32, "8b": 32, "9b": 32,
            "13b": 40, "14b": 40, "20b": 44, "30b": 48,
            "34b": 48, "65b": 80, "70b": 80, "72b": 80
        }
        
        num_layers = layer_map.get(size, 32)
        
        return ModelInfo(
            hf_name=hf_name,
            ollama_name=ollama_name,
            family=family,
            size=size,
            architecture=architecture,
            hidden_size=hidden_size,
            num_layers=num_layers,
            is_instruct=is_instruct,
            context_length=4096  # Default context length
        )
    
    def _get_model_family(self, model_name: str) -> str:
        """Determine model family from name."""
        for family, models in MODEL_FAMILIES.items():
            if model_name in models:
                return family
        
        # Fallback based on name patterns
        name_lower = model_name.lower()
        if "mistral" in name_lower or "mixtral" in name_lower:
            return "mistral"
        elif "gemma" in name_lower:
            return "gemma"
        elif "llama" in name_lower:
            return "llama"
        elif "qwen" in name_lower:
            return "qwen"
        elif "yi" in name_lower:
            return "yi"
        elif "code" in name_lower:
            return "code"
        else:
            return "unknown"
    
    def _extract_model_size(self, model_name: str) -> str:
        """Extract model size from name."""
        import re
        # Look for patterns like "7b", "13b", "70b", etc.
        size_pattern = r"(\d+(?:\.\d+)?[bB])"
        match = re.search(size_pattern, model_name)
        if match:
            return match.group(1).lower()
        
        # Fallback patterns
        if "small" in model_name.lower():
            return "2b"
        elif "medium" in model_name.lower():
            return "7b"
        elif "large" in model_name.lower():
            return "13b"
        else:
            return "7b"  # Default assumption
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine model architecture."""
        name_lower = model_name.lower()
        if "mistral" in name_lower:
            return "mistral"
        elif "gemma" in name_lower:
            return "gemma"
        elif "llama" in name_lower:
            return "llama"
        elif "qwen" in name_lower:
            return "qwen"
        elif "yi" in name_lower:
            return "yi"
        else:
            return "transformer"
    
    def _test_model(self, ollama_name: str) -> bool:
        """Test if model is working correctly."""
        try:
            response = self.ollama_manager.generate(
                ollama_name, 
                "Hello, how are you?",
                options={"num_predict": 10}
            )
            return response is not None and len(response.strip()) > 0
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False
    
    def generate_text(self, model_name: str, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using a loaded model."""
        if model_name not in self.loaded_models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        model_info = self.loaded_models[model_name]
        return self.ollama_manager.generate(model_info.ollama_name, prompt, **kwargs)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        return self.loaded_models.get(model_name)
    
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models."""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str):
        """Unload a model (remove from cache)."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.model_cache:
                del self.model_cache[model_name]
            logger.info(f"Unloaded model {model_name}")
    
    def load_model_family(self, family_name: str) -> List[ModelInfo]:
        """Load all models from a specific family."""
        if family_name not in MODEL_FAMILIES:
            raise ValueError(f"Unknown model family: {family_name}")
        
        models = MODEL_FAMILIES[family_name]
        loaded_models = []
        
        for model_name in models:
            try:
                model_info = self.load_model(model_name)
                loaded_models.append(model_info)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
        
        return loaded_models
    
    def get_compatible_models(self, reference_model: str, max_size_diff: str = "2x") -> List[str]:
        """Get models compatible with reference model for comparison."""
        if reference_model not in self.loaded_models:
            raise ValueError(f"Reference model {reference_model} not loaded")
        
        ref_info = self.loaded_models[reference_model]
        compatible = []
        
        for model_name, model_info in self.loaded_models.items():
            if model_name == reference_model:
                continue
            
            # Check architecture compatibility
            if model_info.architecture == ref_info.architecture:
                compatible.append(model_name)
            # Cross-architecture compatibility based on size
            elif self._size_compatible(ref_info.size, model_info.size, max_size_diff):
                compatible.append(model_name)
        
        return compatible
    
    def _size_compatible(self, size1: str, size2: str, max_diff: str) -> bool:
        """Check if two model sizes are compatible."""
        def parse_size(size_str: str) -> float:
            return float(size_str.replace('b', ''))
        
        try:
            s1, s2 = parse_size(size1), parse_size(size2)
            ratio = max(s1, s2) / min(s1, s2)
            
            if max_diff == "2x":
                return ratio <= 2.0
            elif max_diff == "4x":
                return ratio <= 4.0
            elif max_diff == "any":
                return True
            else:
                return ratio <= float(max_diff.replace('x', ''))
        except:
            return False
    
    def shutdown(self):
        """Shutdown the model loader and cleanup resources."""
        self.loaded_models.clear()
        self.model_cache.clear()
        self.ollama_manager.shutdown()
        logger.info("ModelLoader shut down")

# Global model loader instance
_global_loader = None

def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ModelLoader()
    return _global_loader

# Convenience functions
def load_model(model_name: str, **kwargs) -> ModelInfo:
    """Load a model using the global loader."""
    return get_model_loader().load_model(model_name, **kwargs)

def generate_text(model_name: str, prompt: str, **kwargs) -> Optional[str]:
    """Generate text using the global loader."""
    return get_model_loader().generate_text(model_name, prompt, **kwargs)

def list_available_models() -> List[str]:
    """List all models available through HuggingFace-Ollama mapping."""
    return list(HF_TO_OLLAMA_MODELS.keys())