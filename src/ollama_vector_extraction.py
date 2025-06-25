"""
Ollama-compatible vector extraction using behavioral differences instead of internal activations.
Since Ollama doesn't expose model internals, we extract preference vectors through output analysis.
"""

import json
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .ollama_utils import OllamaClient
from .utils import TextValidator, handle_errors

logger = logging.getLogger(__name__)

@dataclass
class BehavioralVector:
    """
    Represents a preference vector extracted from behavioral differences.
    Since we can't access internal activations with Ollama, we use output patterns.
    """
    vector_id: str
    preference_type: str  # "verbosity", "formality", "technical_complexity", etc.
    source_model: str
    extraction_method: str
    behavioral_signature: Dict[str, float]  # Behavioral characteristics
    example_pairs: List[Tuple[str, str, str]]  # (prompt, style_a, style_b)
    metadata: Dict[str, Any]

class OllamaBehavioralExtractor:
    """
    Extract preference vectors by analyzing behavioral differences in Ollama model outputs.
    """
    
    def __init__(self, model_client: OllamaClient, extraction_config: Dict[str, Any] = None):
        self.model = model_client
        self.config = extraction_config or self._get_default_config()
        self.extracted_vectors = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for behavioral extraction."""
        return {
            "num_samples": 50,
            "temperature_range": [0.3, 0.7, 1.0],
            "max_tokens": 200,
            "style_prompts": {
                "verbosity": {
                    "concise": "Be concise and brief in your response.",
                    "verbose": "Be detailed and comprehensive in your response."
                },
                "formality": {
                    "casual": "Respond in a casual, conversational tone.",
                    "formal": "Respond in a formal, professional tone."
                },
                "technical_complexity": {
                    "simple": "Explain in simple terms that anyone can understand.",
                    "technical": "Use technical terminology and detailed explanations."
                },
                "certainty": {
                    "uncertain": "Express uncertainty and mention potential limitations.",
                    "confident": "Be confident and definitive in your statements."
                }
            }
        }
    
    @handle_errors("extract behavioral vector")
    def extract_preference_vector(self, 
                                 preference_type: str,
                                 prompts: List[str],
                                 save_path: Optional[str] = None) -> BehavioralVector:
        """
        Extract a preference vector by analyzing output differences.
        
        Args:
            preference_type: Type of preference (verbosity, formality, etc.)
            prompts: List of prompts to test behavioral differences
            save_path: Optional path to save the extracted vector
        """
        logger.info(f"Extracting {preference_type} vector using behavioral analysis")
        
        if preference_type not in self.config["style_prompts"]:
            raise ValueError(f"Unknown preference type: {preference_type}")
        
        style_configs = self.config["style_prompts"][preference_type]
        behavioral_signature = {}
        example_pairs = []
        
        # Generate responses for each style
        style_responses = {}
        for style_name, style_prompt in style_configs.items():
            logger.info(f"Generating {style_name} responses")
            responses = self._generate_style_responses(prompts, style_prompt)
            style_responses[style_name] = responses
        
        # Analyze behavioral differences
        behavioral_signature = self._analyze_behavioral_differences(
            style_responses, preference_type
        )
        
        # Create example pairs for validation
        example_pairs = self._create_example_pairs(prompts, style_responses)
        
        # Create behavioral vector
        vector = BehavioralVector(
            vector_id=f"{self.model.model_name}_{preference_type}_{len(prompts)}",
            preference_type=preference_type,
            source_model=self.model.model_name,
            extraction_method="behavioral_analysis",
            behavioral_signature=behavioral_signature,
            example_pairs=example_pairs[:10],  # Store first 10 as examples
            metadata={
                "num_prompts": len(prompts),
                "extraction_config": self.config,
                "model_config": {
                    "ollama_name": self.model.ollama_name,
                    "host": self.model.host
                }
            }
        )
        
        # Cache the vector
        self.extracted_vectors[preference_type] = vector
        
        # Save if requested
        if save_path:
            self._save_vector(vector, save_path)
        
        logger.info(f"Successfully extracted {preference_type} vector")
        return vector
    
    def _generate_style_responses(self, prompts: List[str], style_prompt: str) -> List[str]:
        """Generate responses with a specific style prompt."""
        responses = []
        
        for prompt in prompts:
            # Combine style instruction with original prompt
            full_prompt = f"{style_prompt}\n\nPrompt: {prompt}\n\nResponse:"
            
            try:
                response = self.model.generate(
                    full_prompt,
                    max_tokens=self.config["max_tokens"],
                    temperature=0.7
                )
                responses.append(response.text.strip())
            except Exception as e:
                logger.warning(f"Failed to generate response for prompt: {e}")
                responses.append("")
        
        return responses
    
    def _analyze_behavioral_differences(self, 
                                      style_responses: Dict[str, List[str]], 
                                      preference_type: str) -> Dict[str, float]:
        """Analyze behavioral characteristics of different styles."""
        signature = {}
        
        # Analyze length differences
        if preference_type == "verbosity":
            signature.update(self._analyze_length_patterns(style_responses))
        
        # Analyze vocabulary complexity
        if preference_type in ["formality", "technical_complexity"]:
            signature.update(self._analyze_vocabulary_complexity(style_responses))
        
        # Analyze certainty markers
        if preference_type == "certainty":
            signature.update(self._analyze_certainty_markers(style_responses))
        
        # General linguistic features
        signature.update(self._analyze_general_features(style_responses))
        
        return signature
    
    def _analyze_length_patterns(self, style_responses: Dict[str, List[str]]) -> Dict[str, float]:
        """Analyze length patterns for verbosity."""
        patterns = {}
        
        for style_name, responses in style_responses.items():
            # Word count statistics
            word_counts = [len(resp.split()) for resp in responses if resp]
            if word_counts:
                patterns[f"{style_name}_avg_words"] = np.mean(word_counts)
                patterns[f"{style_name}_std_words"] = np.std(word_counts)
                
            # Sentence count statistics  
            sentence_counts = [len([s for s in resp.split('.') if s.strip()]) 
                             for resp in responses if resp]
            if sentence_counts:
                patterns[f"{style_name}_avg_sentences"] = np.mean(sentence_counts)
        
        # Calculate difference ratios
        if "verbose" in style_responses and "concise" in style_responses:
            verbose_words = patterns.get("verbose_avg_words", 0)
            concise_words = patterns.get("concise_avg_words", 1)
            patterns["verbosity_ratio"] = verbose_words / max(concise_words, 1)
        
        return patterns
    
    def _analyze_vocabulary_complexity(self, style_responses: Dict[str, List[str]]) -> Dict[str, float]:
        """Analyze vocabulary complexity patterns."""
        patterns = {}
        
        for style_name, responses in style_responses.items():
            if not responses:
                continue
                
            # Average word length
            all_words = []
            for resp in responses:
                all_words.extend(resp.split())
            
            if all_words:
                avg_word_length = np.mean([len(word) for word in all_words])
                patterns[f"{style_name}_avg_word_length"] = avg_word_length
                
                # Unique vocabulary size
                unique_words = set(word.lower() for word in all_words)
                patterns[f"{style_name}_vocab_diversity"] = len(unique_words) / max(len(all_words), 1)
        
        return patterns
    
    def _analyze_certainty_markers(self, style_responses: Dict[str, List[str]]) -> Dict[str, float]:
        """Analyze certainty/uncertainty markers."""
        patterns = {}
        
        uncertainty_markers = ["might", "could", "possibly", "perhaps", "may", "uncertain", "unclear"]
        certainty_markers = ["definitely", "certainly", "clearly", "obviously", "undoubtedly"]
        
        for style_name, responses in style_responses.items():
            if not responses:
                continue
                
            uncertainty_count = 0
            certainty_count = 0
            total_words = 0
            
            for resp in responses:
                words = resp.lower().split()
                total_words += len(words)
                uncertainty_count += sum(1 for word in words if word in uncertainty_markers)
                certainty_count += sum(1 for word in words if word in certainty_markers)
            
            if total_words > 0:
                patterns[f"{style_name}_uncertainty_rate"] = uncertainty_count / total_words
                patterns[f"{style_name}_certainty_rate"] = certainty_count / total_words
        
        return patterns
    
    def _analyze_general_features(self, style_responses: Dict[str, List[str]]) -> Dict[str, float]:
        """Analyze general linguistic features."""
        patterns = {}
        
        for style_name, responses in style_responses.items():
            if not responses:
                continue
                
            # Punctuation density
            total_chars = sum(len(resp) for resp in responses)
            punctuation_chars = sum(resp.count('.') + resp.count('!') + resp.count('?') 
                                  for resp in responses)
            
            if total_chars > 0:
                patterns[f"{style_name}_punctuation_density"] = punctuation_chars / total_chars
        
        return patterns
    
    def _create_example_pairs(self, prompts: List[str], 
                            style_responses: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
        """Create example pairs showing style differences."""
        pairs = []
        
        style_names = list(style_responses.keys())
        if len(style_names) >= 2:
            style_a, style_b = style_names[0], style_names[1]
            
            for i, prompt in enumerate(prompts[:10]):  # First 10 prompts
                if (i < len(style_responses[style_a]) and 
                    i < len(style_responses[style_b])):
                    pairs.append((
                        prompt,
                        style_responses[style_a][i],
                        style_responses[style_b][i]
                    ))
        
        return pairs
    
    def _save_vector(self, vector: BehavioralVector, save_path: str):
        """Save behavioral vector to file."""
        try:
            vector_data = {
                "vector_id": vector.vector_id,
                "preference_type": vector.preference_type,
                "source_model": vector.source_model,
                "extraction_method": vector.extraction_method,
                "behavioral_signature": vector.behavioral_signature,
                "example_pairs": vector.example_pairs,
                "metadata": vector.metadata
            }
            
            with open(save_path, 'w') as f:
                json.dump(vector_data, f, indent=2)
            
            logger.info(f"Saved behavioral vector to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector: {e}")

def load_behavioral_vector(file_path: str) -> BehavioralVector:
    """Load a behavioral vector from file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return BehavioralVector(
            vector_id=data["vector_id"],
            preference_type=data["preference_type"],
            source_model=data["source_model"],
            extraction_method=data["extraction_method"],
            behavioral_signature=data["behavioral_signature"],
            example_pairs=data["example_pairs"],
            metadata=data["metadata"]
        )
    except Exception as e:
        logger.error(f"Failed to load behavioral vector: {e}")
        raise