#!/usr/bin/env python3
"""
Shared utilities to eliminate code duplication across the project.
Refactored to use Ollama instead of transformers.
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from functools import wraps

from .config import get_model_config
from .ollama_utils import load_ollama_model, OllamaClient, check_ollama_status

logger = logging.getLogger(__name__)

# === SHARED CONSTANTS ===
DEFAULT_LAYER_INDICES = {
    'small': 8,      # For 1B-2B models
    'medium': 12,    # For 7B models  
    'large': 16      # For 13B+ models
}

DEFAULT_GENERATION_PARAMS = {
    'max_new_tokens': 150,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,
    'pad_token_id': None  # Set during model loading
}

TEXT_VALIDATION_PARAMS = {
    'min_length': 10,
    'max_length': 1000,
    'min_sentences': 2
}

# === SHARED ERROR HANDLING ===
def handle_errors(operation_name: str = "operation", continue_on_error: bool = True):
    """Decorator for consistent error handling across the codebase."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to {operation_name}: {e}")
                if continue_on_error:
                    return None
                else:
                    raise
        return wrapper
    return decorator

# === SHARED MODEL LOADER ===
class ModelLoader:
    """Centralized Ollama model loading with consistent configuration."""
    
    _loaded_models = {}  # Cache for loaded models
    
    @classmethod
    @handle_errors("load model")
    def load_model_and_tokenizer(cls, model_name: str, **config_overrides) -> Tuple[OllamaClient, None]:
        """Load Ollama model client. Returns (client, None) for compatibility."""
        
        # Check cache first
        cache_key = f"{model_name}_{hash(str(sorted(config_overrides.items())))}"
        if cache_key in cls._loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return cls._loaded_models[cache_key]
        
        logger.info(f"Loading Ollama model: {model_name}")
        
        # Check if Ollama is running
        if not check_ollama_status():
            raise RuntimeError("Ollama server is not running. Please start Ollama first.")
        
        # Load Ollama client
        client = load_ollama_model(model_name, **config_overrides)
        
        # Cache the loaded client
        cls._loaded_models[cache_key] = (client, None)  # None for tokenizer compatibility
        
        logger.info(f"Successfully loaded Ollama model: {model_name}")
        return client, None
    
    @classmethod
    def load_model(cls, model_name: str, **config_overrides) -> OllamaClient:
        """Load just the Ollama model client."""
        client, _ = cls.load_model_and_tokenizer(model_name, **config_overrides)
        return client
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache."""
        cls._loaded_models.clear()
        logger.info("Cleared Ollama model cache")

# === OLLAMA COMPATIBILITY ===
class DeviceManager:
    """Simplified device management for Ollama (server handles GPU allocation)."""
    
    @staticmethod
    def to_device(data: Any, device: str = None) -> Any:
        """For Ollama compatibility - data is handled server-side."""
        # Ollama handles device placement, just return data as-is
        return data
    
    @staticmethod
    def get_model_device(model: OllamaClient) -> str:
        """Get the device info for Ollama model (always returns 'ollama-server')."""
        return "ollama-server"
    
    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available (for Ollama this depends on server config)."""
        # For Ollama, we assume GPU is available if server is running
        return check_ollama_status()

# === SHARED TEXT VALIDATION ===
class TextValidator:
    """Centralized text validation logic."""
    
    @staticmethod
    def validate_text_basic(text: str, 
                           min_length: int = TEXT_VALIDATION_PARAMS['min_length'],
                           max_length: int = TEXT_VALIDATION_PARAMS['max_length']) -> bool:
        """Basic text validation."""
        if not text or len(text.strip()) == 0:
            return False
        
        word_count = len(text.split())
        return min_length <= word_count <= max_length
    
    @staticmethod
    def validate_text_quality(text: str) -> bool:
        """Enhanced text quality validation."""
        if not TextValidator.validate_text_basic(text):
            return False
        
        # Check for minimum sentence count
        sentence_count = len([s for s in text.split('.') if s.strip()])
        if sentence_count < TEXT_VALIDATION_PARAMS['min_sentences']:
            return False
        
        # Check for mostly ASCII content (basic language detection)
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if ascii_ratio < 0.8:
            return False
        
        return True
    
    @staticmethod
    def validate_summary_pair(concise: str, verbose: str) -> bool:
        """Validate a concise/verbose summary pair."""
        if not (TextValidator.validate_text_basic(concise) and 
                TextValidator.validate_text_basic(verbose)):
            return False
        
        concise_len = len(concise.split())
        verbose_len = len(verbose.split())
        
        # Verbose should be meaningfully longer
        if verbose_len <= concise_len * 1.2:
            return False
        
        return True

# === SHARED HOOK MANAGER ===
class HookManager:
    """Centralized activation hook management."""
    
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = {}
        self.hooks = []
    
    def get_activation_hook(self, name: str) -> Callable:
        """Create a hook function for capturing activations."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook_fn
    
    def register_hooks(self, model: nn.Module, layer_pattern: str = "layers"):
        """Register hooks on specified layers."""
        self.hooks = []
        
        for name, module in model.named_modules():
            if f"{layer_pattern}.{self.layer_idx}" in name and "mlp" not in name.lower():
                hook = module.register_forward_hook(self.get_activation_hook(name))
                self.hooks.append(hook)
                logger.debug(f"Registered hook on {name}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_residual_activation(self) -> Optional[torch.Tensor]:
        """Get the main residual stream activation."""
        for name, activation in self.activations.items():
            if "attention" not in name.lower() and "mlp" not in name.lower():
                return activation
        
        if self.activations:
            return list(self.activations.values())[0]
        return None
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()

# === SHARED GENERATION UTILITIES ===
class GenerationHelper:
    """Centralized text generation utilities."""
    
    @staticmethod
    def get_generation_params(tokenizer: AutoTokenizer, **overrides) -> Dict[str, Any]:
        """Get generation parameters with consistent defaults."""
        params = DEFAULT_GENERATION_PARAMS.copy()
        params['pad_token_id'] = tokenizer.eos_token_id
        params['eos_token_id'] = tokenizer.eos_token_id
        params.update(overrides)
        return params
    
    @staticmethod
    @handle_errors("generate text")
    def generate_text(model: AutoModelForCausalLM, 
                     tokenizer: AutoTokenizer,
                     prompt: str,
                     **generation_overrides) -> str:
        """Generate text with consistent parameters."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = DeviceManager.to_device(inputs, DeviceManager.get_model_device(model))
        
        generation_params = GenerationHelper.get_generation_params(tokenizer, **generation_overrides)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

# === SHARED LAYER UTILITIES ===
class LayerUtils:
    """Utilities for working with model layers."""
    
    @staticmethod
    def get_default_intervention_layer(model_name: str) -> int:
        """Get default intervention layer based on model size."""
        model_lower = model_name.lower()
        
        if any(size in model_lower for size in ["0.5b", "1b", "2b", "small"]):
            return DEFAULT_LAYER_INDICES['small']
        elif any(size in model_lower for size in ["7b", "8b", "medium"]):
            return DEFAULT_LAYER_INDICES['medium']
        else:
            return DEFAULT_LAYER_INDICES['large']
    
    @staticmethod
    def get_model_num_layers(model: nn.Module) -> int:
        """Get the number of layers in a model."""
        try:
            return model.config.num_hidden_layers
        except AttributeError:
            # Fallback: count layers manually
            layer_count = 0
            for name, _ in model.named_modules():
                if "layers." in name and name.count('.') == 1:
                    layer_count += 1
            return layer_count
    
    @staticmethod
    def map_functional_layer(source_layers: int, target_layers: int, source_layer: int) -> int:
        """Map layer based on relative position in architecture."""
        relative_position = source_layer / (source_layers - 1)
        target_layer = int(relative_position * (target_layers - 1))
        return min(target_layer, target_layers - 1)

# === SHARED NORMALIZATION UTILITIES ===
class VectorUtils:
    """Utilities for vector operations and normalization."""
    
    @staticmethod
    def normalize_vector(vector: torch.Tensor, method: str = "l2") -> torch.Tensor:
        """Normalize vector using specified method."""
        if method == "l2":
            return vector / vector.norm()
        elif method == "unit":
            return vector / vector.abs().max()
        elif method == "zscore":
            return (vector - vector.mean()) / vector.std()
        else:
            return vector
    
    @staticmethod
    def compute_vector_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute cosine similarity between vectors."""
        return torch.cosine_similarity(vec1, vec2, dim=0).item()
    
    @staticmethod
    def adapt_vector_dimension(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Adapt vector to target dimension."""
        current_dim = vector.shape[0]
        
        if current_dim == target_dim:
            return vector
        elif current_dim > target_dim:
            return vector[:target_dim]
        else:
            padding = torch.zeros(target_dim - current_dim)
            return torch.cat([vector, padding])