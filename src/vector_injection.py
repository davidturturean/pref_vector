import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import List, Dict, Tuple, Optional, Callable, Union
import logging
from contextlib import contextmanager
import json

from .config import get_model_config, EXPERIMENT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorInjectionHook:
    """Hook for injecting preference vectors into transformer layers during inference."""
    
    def __init__(self, layer_idx: int, vector: torch.Tensor, scale: float = 1.0):
        self.layer_idx = layer_idx
        self.vector = vector.to(vector.device)
        self.scale = scale
        self.hooks = []
        self.injection_applied = False
    
    def get_injection_hook(self, name: str) -> Callable:
        """Create hook function that injects the preference vector."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Inject vector into the last token's hidden state (for causal models)
                if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_dim]
                    hidden_states[:, -1, :] += self.scale * self.vector.to(hidden_states.device)
                    self.injection_applied = True
                return (hidden_states,) + output[1:]
            else:
                # Direct hidden state output
                if len(output.shape) == 3:
                    output[:, -1, :] += self.scale * self.vector.to(output.device)
                    self.injection_applied = True
                return output
        return hook_fn
    
    def register_hooks(self, model: nn.Module, layer_name_pattern: str = "layers"):
        """Register injection hooks on the specified layer."""
        self.hooks = []
        self.injection_applied = False
        
        target_layer_name = None
        for name, module in model.named_modules():
            if f"{layer_name_pattern}.{self.layer_idx}" in name:
                # Look for the main transformer block or attention output
                if any(block_type in name for block_type in ["block", "layer", "h."]):
                    if "attention" not in name.lower() or "output" in name.lower():
                        hook = module.register_forward_hook(
                            self.get_injection_hook(name)
                        )
                        self.hooks.append(hook)
                        target_layer_name = name
                        logger.debug(f"Registered injection hook on {name}")
                        break
        
        if not self.hooks:
            logger.warning(f"No suitable layer found for injection at layer {self.layer_idx}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.injection_applied = False

class SteerableModel:
    """Wrapper for language models that supports preference vector steering."""
    
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        model_config = get_model_config(model_name)
        self.device = device or model_config.device
        self.model = None
        self.tokenizer = None
        self.injection_hook = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading steerable model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_config = get_model_config(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_config.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    @contextmanager
    def steered_generation(self, 
                          preference_vector: torch.Tensor,
                          layer_idx: int,
                          scale: float = 1.0):
        """Context manager for steered text generation."""
        self.injection_hook = VectorInjectionHook(layer_idx, preference_vector, scale)
        
        try:
            self.injection_hook.register_hooks(self.model)
            yield self
        finally:
            if self.injection_hook:
                self.injection_hook.remove_hooks()
                self.injection_hook = None
    
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 150,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     do_sample: bool = True,
                     **kwargs) -> str:
        """Generate text with the current steering configuration."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode only the generated part (exclude the input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_steered(self,
                        prompt: str,
                        preference_vector: torch.Tensor,
                        layer_idx: int,
                        scale: float = 1.0,
                        **generation_kwargs) -> str:
        """Generate text with preference vector steering."""
        with self.steered_generation(preference_vector, layer_idx, scale):
            return self.generate_text(prompt, **generation_kwargs)
    
    def compare_generations(self,
                           prompt: str,
                           preference_vector: torch.Tensor,
                           layer_idx: int,
                           scales: List[float] = [-1.0, 0.0, 1.0],
                           **generation_kwargs) -> Dict[float, str]:
        """Generate text at different steering scales for comparison."""
        results = {}
        
        for scale in scales:
            logger.debug(f"Generating with scale {scale}")
            if scale == 0.0:
                # No steering
                text = self.generate_text(prompt, **generation_kwargs)
            else:
                text = self.generate_steered(
                    prompt, preference_vector, layer_idx, scale, **generation_kwargs
                )
            results[scale] = text
        
        return results

class CrossModelTransferer:
    """Handles transfer of preference vectors between different model architectures."""
    
    def __init__(self):
        self.models = {}
        self.loaded_vectors = {}
    
    def load_model(self, model_name: str, alias: str = None) -> SteerableModel:
        """Load a model for cross-model experiments."""
        alias = alias or model_name
        if alias not in self.models:
            self.models[alias] = SteerableModel(model_name)
        return self.models[alias]
    
    def load_vector(self, vector_path: str, alias: str = None) -> torch.Tensor:
        """Load a preference vector from file."""
        alias = alias or vector_path
        if alias not in self.loaded_vectors:
            with open(vector_path, 'r') as f:
                data = json.load(f)
            vector = torch.tensor(data['vector'], dtype=torch.float32)
            self.loaded_vectors[alias] = vector
        return self.loaded_vectors[alias]
    
    def test_direct_transfer(self,
                           source_vector: torch.Tensor,
                           target_model: SteerableModel,
                           test_prompts: List[str],
                           layer_idx: int,
                           scales: List[float] = [-1.0, 0.0, 1.0]) -> Dict:
        """Test direct transfer of preference vector to target model."""
        logger.info(f"Testing direct transfer to {target_model.model_name}")
        
        results = {
            'model_name': target_model.model_name,
            'layer_idx': layer_idx,
            'prompts': {},
            'successful_transfers': 0,
            'total_prompts': len(test_prompts)
        }
        
        for i, prompt in enumerate(test_prompts):
            try:
                generations = target_model.compare_generations(
                    prompt, source_vector, layer_idx, scales
                )
                results['prompts'][f'prompt_{i}'] = {
                    'text': prompt,
                    'generations': generations
                }
                
                # Check if transfer was successful (heuristic: different outputs)
                if len(set(generations.values())) > 1:
                    results['successful_transfers'] += 1
                    
            except Exception as e:
                logger.warning(f"Failed to generate for prompt {i}: {e}")
                results['prompts'][f'prompt_{i}'] = {
                    'text': prompt,
                    'error': str(e)
                }
        
        results['success_rate'] = results['successful_transfers'] / results['total_prompts']
        return results
    
    def test_multiple_models(self,
                           source_vector: torch.Tensor,
                           model_names: List[str],
                           test_prompts: List[str],
                           layer_idx: int) -> Dict:
        """Test preference vector transfer across multiple models."""
        all_results = {
            'source_vector_info': {
                'shape': list(source_vector.shape),
                'norm': source_vector.norm().item()
            },
            'layer_idx': layer_idx,
            'models': {}
        }
        
        for model_name in model_names:
            try:
                model = self.load_model(model_name)
                results = self.test_direct_transfer(
                    source_vector, model, test_prompts, layer_idx
                )
                all_results['models'][model_name] = results
                
            except Exception as e:
                logger.error(f"Failed to test model {model_name}: {e}")
                all_results['models'][model_name] = {'error': str(e)}
        
        return all_results

def create_test_prompts() -> List[str]:
    """Create standardized test prompts for evaluation."""
    return [
        "Summarize the key findings about climate change impacts on polar ice caps.",
        "Explain the process of photosynthesis in plants.",
        "Describe the main advantages and disadvantages of renewable energy sources.",
        "Summarize the plot of Shakespeare's Hamlet.",
        "Explain how machine learning algorithms work.",
        "Describe the causes and effects of the 2008 financial crisis.",
        "Summarize the benefits of regular exercise for human health.",
        "Explain the concept of supply and demand in economics."
    ]

def demonstrate_vector_injection():
    """Demonstration of preference vector injection functionality."""
    # This is a simplified demo using a smaller model
    logger.info("Demonstrating vector injection with a smaller model")
    
    model = SteerableModel("microsoft/DialoGPT-medium")
    
    # Create a dummy preference vector
    vector_dim = model.model.config.hidden_size
    dummy_vector = torch.randn(vector_dim) * 0.1
    
    test_prompt = "Summarize the benefits of artificial intelligence:"
    
    # Test generations at different scales
    results = model.compare_generations(
        test_prompt,
        dummy_vector,
        layer_idx=8,  # Middle layer for DialoGPT
        scales=[-0.5, 0.0, 0.5]
    )
    
    print("Vector injection demo results:")
    for scale, text in results.items():
        print(f"Scale {scale}: {text[:100]}...")
    
    return results

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_vector_injection()
    
    # Test cross-model transfer setup
    transferer = CrossModelTransferer()
    test_prompts = create_test_prompts()[:3]  # Use fewer prompts for demo
    
    print(f"\nCreated {len(test_prompts)} test prompts for evaluation")