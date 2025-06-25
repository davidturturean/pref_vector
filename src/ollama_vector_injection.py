"""
Ollama-compatible vector injection using prompt engineering and style transfer.
Since Ollama doesn't allow direct activation manipulation, we inject preferences through prompts.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .ollama_utils import OllamaClient
from .ollama_vector_extraction import BehavioralVector, load_behavioral_vector
from .utils import handle_errors

logger = logging.getLogger(__name__)

@dataclass 
class InjectionResult:
    """Result of preference vector injection."""
    original_response: str
    modified_response: str
    injection_method: str
    preference_type: str
    success_score: float  # 0-1 indicating how well the preference was applied
    metadata: Dict[str, Any]

class OllamaVectorInjector:
    """
    Inject preference vectors into Ollama models using prompt engineering techniques.
    """
    
    def __init__(self, model_client: OllamaClient):
        self.model = model_client
        self.loaded_vectors = {}
        
    def load_vector(self, vector_path: str) -> BehavioralVector:
        """Load a behavioral vector for injection."""
        vector = load_behavioral_vector(vector_path)
        self.loaded_vectors[vector.preference_type] = vector
        logger.info(f"Loaded {vector.preference_type} vector from {vector_path}")
        return vector
    
    def load_vector_object(self, vector: BehavioralVector):
        """Load a behavioral vector object directly."""
        self.loaded_vectors[vector.preference_type] = vector
        logger.info(f"Loaded {vector.preference_type} vector object")
    
    @handle_errors("inject preference vector")
    def inject_vector(self, 
                     prompt: str,
                     preference_type: str,
                     injection_strength: float = 1.0,
                     method: str = "prompt_engineering") -> InjectionResult:
        """
        Inject a preference vector into model response.
        
        Args:
            prompt: Original prompt
            preference_type: Type of preference to inject
            injection_strength: Strength of injection (0-1)
            method: Injection method to use
        """
        if preference_type not in self.loaded_vectors:
            raise ValueError(f"No vector loaded for preference type: {preference_type}")
        
        vector = self.loaded_vectors[preference_type]
        
        # Generate original response for comparison
        original_response = self.model.generate(prompt, temperature=0.7).text
        
        # Apply injection based on method
        if method == "prompt_engineering":
            modified_response = self._inject_via_prompt_engineering(
                prompt, vector, injection_strength
            )
        elif method == "example_based":
            modified_response = self._inject_via_examples(
                prompt, vector, injection_strength
            )
        elif method == "style_transfer":
            modified_response = self._inject_via_style_transfer(
                prompt, original_response, vector, injection_strength
            )
        else:
            raise ValueError(f"Unknown injection method: {method}")
        
        # Evaluate injection success
        success_score = self._evaluate_injection_success(
            original_response, modified_response, vector
        )
        
        return InjectionResult(
            original_response=original_response,
            modified_response=modified_response,
            injection_method=method,
            preference_type=preference_type,
            success_score=success_score,
            metadata={
                "injection_strength": injection_strength,
                "vector_id": vector.vector_id,
                "model": self.model.model_name
            }
        )
    
    def _inject_via_prompt_engineering(self, 
                                     prompt: str, 
                                     vector: BehavioralVector,
                                     strength: float) -> str:
        """Inject preference using prompt engineering."""
        
        # Create style instruction based on behavioral signature
        style_instruction = self._create_style_instruction(vector, strength)
        
        # Combine instruction with original prompt
        enhanced_prompt = f"{style_instruction}\n\nPlease respond to: {prompt}"
        
        # Generate response with style injection
        response = self.model.generate(enhanced_prompt, temperature=0.7)
        return response.text.strip()
    
    def _inject_via_examples(self, 
                           prompt: str,
                           vector: BehavioralVector,
                           strength: float) -> str:
        """Inject preference using few-shot examples."""
        
        if not vector.example_pairs:
            logger.warning("No example pairs available, falling back to prompt engineering")
            return self._inject_via_prompt_engineering(prompt, vector, strength)
        
        # Select examples based on strength
        num_examples = min(int(strength * 3) + 1, len(vector.example_pairs))
        selected_examples = vector.example_pairs[:num_examples]
        
        # Create few-shot prompt
        examples_text = ""
        for example_prompt, style_a, style_b in selected_examples:
            # Choose the preferred style based on vector type
            preferred_style = self._choose_preferred_style(vector, style_a, style_b)
            examples_text += f"Prompt: {example_prompt}\nResponse: {preferred_style}\n\n"
        
        # Add the actual prompt
        full_prompt = f"{examples_text}Prompt: {prompt}\nResponse:"
        
        response = self.model.generate(full_prompt, temperature=0.7)
        return response.text.strip()
    
    def _inject_via_style_transfer(self,
                                 prompt: str,
                                 original_response: str,
                                 vector: BehavioralVector,
                                 strength: float) -> str:
        """Inject preference by modifying the original response."""
        
        # Create transfer instruction
        transfer_instruction = self._create_transfer_instruction(vector, strength)
        
        # Ask model to modify the response
        transfer_prompt = f"""
{transfer_instruction}

Original response: {original_response}

Please rewrite this response following the style guidelines above:
"""
        
        response = self.model.generate(transfer_prompt, temperature=0.5)
        return response.text.strip()
    
    def _create_style_instruction(self, vector: BehavioralVector, strength: float) -> str:
        """Create style instruction based on behavioral signature."""
        
        base_instructions = {
            "verbosity": {
                "low": "Be concise and brief.",
                "high": "Be detailed and comprehensive."
            },
            "formality": {
                "low": "Use a casual, conversational tone.",
                "high": "Use a formal, professional tone."
            },
            "technical_complexity": {
                "low": "Explain in simple terms.",
                "high": "Use technical terminology and detailed explanations."
            },
            "certainty": {
                "low": "Express some uncertainty and mention limitations.",
                "high": "Be confident and definitive."
            }
        }
        
        if vector.preference_type not in base_instructions:
            return "Please respond in an appropriate style."
        
        # Determine style direction from behavioral signature
        style_level = self._determine_style_level(vector, strength)
        instructions = base_instructions[vector.preference_type]
        
        base_instruction = instructions.get(style_level, instructions["high"])
        
        # Enhance instruction based on strength
        if strength > 0.7:
            intensity = "strongly"
        elif strength > 0.4:
            intensity = "moderately"
        else:
            intensity = "slightly"
        
        return f"{base_instruction} Apply this style {intensity}."
    
    def _create_transfer_instruction(self, vector: BehavioralVector, strength: float) -> str:
        """Create instruction for style transfer."""
        
        transfer_templates = {
            "verbosity": "Rewrite to be {direction} verbose while maintaining the same information.",
            "formality": "Rewrite in a {direction} formal tone while keeping the same content.", 
            "technical_complexity": "Rewrite using {direction} technical language while preserving meaning.",
            "certainty": "Rewrite to express {direction} certainty while maintaining accuracy."
        }
        
        # Determine direction based on behavioral signature analysis
        direction = "more" if strength > 0.5 else "less"
        
        template = transfer_templates.get(
            vector.preference_type,
            "Rewrite in an appropriate style while maintaining the content."
        )
        
        return template.format(direction=direction)
    
    def _determine_style_level(self, vector: BehavioralVector, strength: float) -> str:
        """Determine style level from behavioral signature."""
        
        # Analyze the behavioral signature to determine which direction to go
        signature = vector.behavioral_signature
        
        if vector.preference_type == "verbosity":
            # Check if vector represents verbose or concise style
            verbose_ratio = signature.get("verbosity_ratio", 1.0)
            if verbose_ratio > 1.2:  # Vector represents verbose style
                return "high" if strength > 0.5 else "low"
            else:  # Vector represents concise style
                return "low" if strength > 0.5 else "high"
        
        # Default to high intensity for other types
        return "high" if strength > 0.5 else "low"
    
    def _choose_preferred_style(self, vector: BehavioralVector, style_a: str, style_b: str) -> str:
        """Choose the preferred style from two examples based on vector."""
        
        # Simple heuristic: choose based on length for verbosity
        if vector.preference_type == "verbosity":
            ratio = len(style_a.split()) / max(len(style_b.split()), 1)
            verbose_ratio = vector.behavioral_signature.get("verbosity_ratio", 1.0)
            
            if verbose_ratio > 1.2:  # Vector prefers verbose
                return style_a if ratio > 1 else style_b
            else:  # Vector prefers concise
                return style_a if ratio < 1 else style_b
        
        # For other types, default to first style
        return style_a
    
    def _evaluate_injection_success(self, 
                                  original: str,
                                  modified: str,
                                  vector: BehavioralVector) -> float:
        """Evaluate how successfully the preference was injected."""
        
        if not modified or modified == original:
            return 0.0
        
        success_score = 0.0
        
        if vector.preference_type == "verbosity":
            # Measure length change
            orig_words = len(original.split())
            mod_words = len(modified.split())
            
            # Expected direction based on vector
            verbose_ratio = vector.behavioral_signature.get("verbosity_ratio", 1.0)
            
            if verbose_ratio > 1.2:  # Should be more verbose
                success_score = min(mod_words / max(orig_words, 1), 2.0) - 1.0
            else:  # Should be more concise
                success_score = 2.0 - min(mod_words / max(orig_words, 1), 2.0)
            
            success_score = max(0.0, min(1.0, success_score))
        
        elif vector.preference_type == "formality":
            # Simple heuristic: count formal vs informal words
            formal_words = ["furthermore", "consequently", "therefore", "moreover"]
            informal_words = ["gonna", "wanna", "yeah", "ok", "cool"]
            
            orig_formal = sum(1 for word in formal_words if word in original.lower())
            mod_formal = sum(1 for word in formal_words if word in modified.lower())
            
            if mod_formal > orig_formal:
                success_score = 0.7
            elif mod_formal == orig_formal:
                success_score = 0.5
            else:
                success_score = 0.3
        
        else:
            # Default success evaluation based on response difference
            if len(modified) != len(original):
                success_score = 0.6
            else:
                success_score = 0.3
        
        return success_score

def batch_inject_vectors(injector: OllamaVectorInjector,
                        prompts: List[str],
                        preference_type: str,
                        **kwargs) -> List[InjectionResult]:
    """Inject vectors for multiple prompts."""
    results = []
    
    for prompt in prompts:
        try:
            result = injector.inject_vector(prompt, preference_type, **kwargs)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to inject vector for prompt '{prompt[:50]}...': {e}")
            # Create empty result for failed injection
            results.append(InjectionResult(
                original_response="",
                modified_response="",
                injection_method="failed",
                preference_type=preference_type,
                success_score=0.0,
                metadata={"error": str(e)}
            ))
    
    return results