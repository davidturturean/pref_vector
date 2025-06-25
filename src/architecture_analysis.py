import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArchitecture:
    """Comprehensive model architecture analysis."""
    model_name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_type: str
    attention_type: str
    activation_function: str
    has_bias: bool
    tokenizer_type: str
    normalization_pattern: str  # pre_norm, post_norm, rms_norm, etc.
    
    # Functional layer mapping
    embedding_layers: List[str]
    attention_layers: List[str] 
    feedforward_layers: List[str]
    output_layers: List[str]
    
    # Hook points for intervention
    residual_stream_layers: List[str]
    attention_output_layers: List[str]
    mlp_output_layers: List[str]

class ArchitectureAnalyzer:
    """Analyzes and compares model architectures for cross-model compatibility."""
    
    def __init__(self):
        self.analyzed_models = {}
        self.compatibility_cache = {}
    
    def analyze_model_architecture(self, model_name: str) -> ModelArchitecture:
        """Comprehensive analysis of model architecture."""
        if model_name in self.analyzed_models:
            return self.analyzed_models[model_name]
        
        logger.info(f"Analyzing architecture for {model_name}")
        
        # Load config and model for inspection
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Extract basic architecture info
        arch_info = self._extract_basic_info(config, model)
        
        # Analyze layer structure
        layer_analysis = self._analyze_layer_structure(model)
        
        # Identify hook points
        hook_points = self._identify_hook_points(model)
        
        architecture = ModelArchitecture(
            model_name=model_name,
            **arch_info,
            **layer_analysis,
            **hook_points
        )
        
        self.analyzed_models[model_name] = architecture
        return architecture
    
    def _extract_basic_info(self, config: Any, model: nn.Module) -> Dict:
        """Extract basic architectural parameters."""
        info = {
            'hidden_size': getattr(config, 'hidden_size', getattr(config, 'd_model', 0)),
            'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0)),
            'num_attention_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_head', 0)),
            'intermediate_size': getattr(config, 'intermediate_size', getattr(config, 'd_ff', 0)),
            'vocab_size': getattr(config, 'vocab_size', 0),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 0),
            'has_bias': getattr(config, 'bias', True)
        }
        
        # Detect layer norm type
        info['layer_norm_type'] = self._detect_layer_norm_type(model)
        
        # Detect attention type
        info['attention_type'] = self._detect_attention_type(config, model)
        
        # Detect activation function
        info['activation_function'] = getattr(config, 'hidden_act', 'unknown')
        
        # Detect tokenizer type
        info['tokenizer_type'] = self._detect_tokenizer_type(config)
        
        # Detect normalization pattern
        info['normalization_pattern'] = self._detect_normalization_pattern(model)
        
        return info
    
    def _detect_layer_norm_type(self, model: nn.Module) -> str:
        """Detect the type of layer normalization used."""
        for name, module in model.named_modules():
            if 'norm' in name.lower():
                if 'rms' in str(type(module)).lower():
                    return 'rms_norm'
                elif 'layer_norm' in str(type(module)).lower():
                    return 'layer_norm'
        return 'unknown'
    
    def _detect_attention_type(self, config: Any, model: nn.Module) -> str:
        """Detect attention mechanism type."""
        # Check for specific attention implementations
        for name, module in model.named_modules():
            if 'attention' in name.lower():
                module_type = str(type(module)).lower()
                if 'multihead' in module_type:
                    return 'multi_head'
                elif 'grouped' in module_type:
                    return 'grouped_query'
                elif 'flash' in module_type:
                    return 'flash_attention'
        
        # Fallback to config analysis
        if hasattr(config, 'num_key_value_heads'):
            if config.num_key_value_heads != config.num_attention_heads:
                return 'grouped_query'
        
        return 'multi_head'  # Default assumption
    
    def _detect_tokenizer_type(self, config: Any) -> str:
        """Detect tokenizer type from config."""
        tokenizer_class = getattr(config, 'tokenizer_class', '') or ''
        if 'sentencepiece' in tokenizer_class.lower():
            return 'sentencepiece'
        elif 'byte' in tokenizer_class.lower():
            return 'byte_level'
        else:
            return 'standard'
    
    def _detect_normalization_pattern(self, model: nn.Module) -> str:
        """Detect whether model uses pre-norm or post-norm pattern."""
        # This is a heuristic based on common patterns
        layer_names = [name for name, _ in model.named_modules()]
        
        # Look for norm placement patterns
        if any('input_layernorm' in name for name in layer_names):
            return 'pre_norm'
        elif any('post_attention_layernorm' in name for name in layer_names):
            return 'post_norm'
        else:
            return 'unknown'
    
    def _analyze_layer_structure(self, model: nn.Module) -> Dict[str, List[str]]:
        """Analyze the functional layer structure."""
        layers = {
            'embedding_layers': [],
            'attention_layers': [],
            'feedforward_layers': [],
            'output_layers': []
        }
        
        for name, module in model.named_modules():
            name_lower = name.lower()
            
            if 'embed' in name_lower:
                layers['embedding_layers'].append(name)
            elif 'attention' in name_lower or 'attn' in name_lower:
                layers['attention_layers'].append(name)
            elif any(ff_term in name_lower for ff_term in ['feed_forward', 'mlp', 'ffn']):
                layers['feedforward_layers'].append(name)
            elif 'lm_head' in name_lower or 'output' in name_lower:
                layers['output_layers'].append(name)
        
        return layers
    
    def _identify_hook_points(self, model: nn.Module) -> Dict[str, List[str]]:
        """Identify optimal hook points for intervention."""
        hook_points = {
            'residual_stream_layers': [],
            'attention_output_layers': [],
            'mlp_output_layers': []
        }
        
        for name, module in model.named_modules():
            name_lower = name.lower()
            
            # Residual stream points (main transformer blocks)
            if any(block_term in name_lower for block_term in ['block', 'layer', 'h.']):
                if not any(submodule in name_lower for submodule in ['attention', 'mlp', 'norm']):
                    hook_points['residual_stream_layers'].append(name)
            
            # Attention output points
            if 'attention' in name_lower and 'output' in name_lower:
                hook_points['attention_output_layers'].append(name)
            
            # MLP output points
            if any(mlp_term in name_lower for mlp_term in ['mlp', 'feed_forward']) and 'output' in name_lower:
                hook_points['mlp_output_layers'].append(name)
        
        return hook_points
    
    def compute_compatibility_score(self, 
                                   arch1: ModelArchitecture, 
                                   arch2: ModelArchitecture) -> Dict[str, float]:
        """Compute compatibility scores between two architectures."""
        compatibility = {}
        
        # Dimension compatibility
        dim_ratio = min(arch1.hidden_size, arch2.hidden_size) / max(arch1.hidden_size, arch2.hidden_size)
        compatibility['dimension_compatibility'] = dim_ratio
        
        # Layer depth compatibility
        layer_ratio = min(arch1.num_layers, arch2.num_layers) / max(arch1.num_layers, arch2.num_layers)
        compatibility['depth_compatibility'] = layer_ratio
        
        # Attention compatibility
        if arch1.attention_type == arch2.attention_type:
            compatibility['attention_compatibility'] = 1.0
        elif 'multi_head' in [arch1.attention_type, arch2.attention_type]:
            compatibility['attention_compatibility'] = 0.7  # Partial compatibility
        else:
            compatibility['attention_compatibility'] = 0.3
        
        # Normalization compatibility
        if arch1.layer_norm_type == arch2.layer_norm_type:
            compatibility['norm_compatibility'] = 1.0
        elif 'norm' in [arch1.layer_norm_type, arch2.layer_norm_type]:
            compatibility['norm_compatibility'] = 0.6
        else:
            compatibility['norm_compatibility'] = 0.2
        
        # Overall compatibility (weighted average)
        weights = {
            'dimension_compatibility': 0.4,
            'depth_compatibility': 0.2,
            'attention_compatibility': 0.25,
            'norm_compatibility': 0.15
        }
        
        overall_score = sum(compatibility[key] * weights[key] for key in weights)
        compatibility['overall_compatibility'] = overall_score
        
        return compatibility
    
    def find_layer_correspondence(self, 
                                 source_arch: ModelArchitecture,
                                 target_arch: ModelArchitecture) -> Dict[int, int]:
        """Find functional correspondence between layers of different models."""
        source_layers = source_arch.num_layers
        target_layers = target_arch.num_layers
        
        # Simple proportional mapping
        correspondence = {}
        for source_layer in range(source_layers):
            # Map to proportional depth in target model
            relative_depth = source_layer / (source_layers - 1) if source_layers > 1 else 0
            target_layer = int(relative_depth * (target_layers - 1))
            correspondence[source_layer] = target_layer
        
        return correspondence
    
    def suggest_intervention_strategies(self, 
                                      source_arch: ModelArchitecture,
                                      target_arch: ModelArchitecture) -> Dict[str, Any]:
        """Suggest strategies for cross-model intervention based on compatibility."""
        compatibility = self.compute_compatibility_score(source_arch, target_arch)
        strategies = {}
        
        if compatibility['overall_compatibility'] > 0.8:
            strategies['direct_transfer'] = {
                'feasible': True,
                'confidence': 'high',
                'method': 'direct_injection',
                'recommended_layers': self.find_layer_correspondence(source_arch, target_arch)
            }
        elif compatibility['overall_compatibility'] > 0.5:
            strategies['direct_transfer'] = {
                'feasible': True,
                'confidence': 'medium',
                'method': 'scaled_injection',
                'scaling_factor': compatibility['dimension_compatibility']
            }
        else:
            strategies['direct_transfer'] = {
                'feasible': False,
                'confidence': 'low',
                'reason': 'architecture_mismatch'
            }
        
        # Adapter recommendations
        if compatibility['dimension_compatibility'] < 0.8:
            strategies['adapter_type'] = 'full_linear'
            strategies['adapter_complexity'] = 'high'
        elif compatibility['dimension_compatibility'] < 0.95:
            strategies['adapter_type'] = 'diagonal_scaling'
            strategies['adapter_complexity'] = 'medium'
        else:
            strategies['adapter_type'] = 'simple_rotation'
            strategies['adapter_complexity'] = 'low'
        
        return strategies
    
    def analyze_model_family(self, model_names: List[str]) -> Dict[str, Any]:
        """Analyze a family of models for cross-compatibility."""
        architectures = {}
        compatibility_matrix = {}
        
        # Analyze each model
        for model_name in model_names:
            try:
                architectures[model_name] = self.analyze_model_architecture(model_name)
            except Exception as e:
                logger.warning(f"Failed to analyze {model_name}: {e}")
                continue
        
        # Compute pairwise compatibility
        model_list = list(architectures.keys())
        for i, model1 in enumerate(model_list):
            for j, model2 in enumerate(model_list):
                if i != j:
                    compat = self.compute_compatibility_score(
                        architectures[model1], 
                        architectures[model2]
                    )
                    compatibility_matrix[f"{model1}_to_{model2}"] = compat
        
        return {
            'architectures': architectures,
            'compatibility_matrix': compatibility_matrix,
            'summary': self._summarize_family_analysis(architectures, compatibility_matrix)
        }
    
    def _summarize_family_analysis(self, 
                                  architectures: Dict[str, ModelArchitecture],
                                  compatibility_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Summarize the family analysis results."""
        summary = {
            'total_models': len(architectures),
            'dimension_ranges': {},
            'common_patterns': {},
            'best_compatibility_pairs': [],
            'problematic_pairs': []
        }
        
        # Dimension analysis
        hidden_sizes = [arch.hidden_size for arch in architectures.values()]
        summary['dimension_ranges'] = {
            'min_hidden_size': min(hidden_sizes),
            'max_hidden_size': max(hidden_sizes),
            'unique_hidden_sizes': len(set(hidden_sizes))
        }
        
        # Find best and worst compatibility pairs
        for pair, compat in compatibility_matrix.items():
            overall_score = compat['overall_compatibility']
            if overall_score > 0.8:
                summary['best_compatibility_pairs'].append((pair, overall_score))
            elif overall_score < 0.4:
                summary['problematic_pairs'].append((pair, overall_score))
        
        return summary

# Example usage and testing
def demonstrate_architecture_analysis():
    """Demonstrate architecture analysis capabilities."""
    analyzer = ArchitectureAnalyzer()
    
    # Test models (using smaller models for demonstration)
    test_models = [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small",
        "gpt2"
    ]
    
    print("Architecture Analysis Demonstration")
    print("=" * 50)
    
    try:
        # Analyze model family
        family_analysis = analyzer.analyze_model_family(test_models)
        
        print(f"Analyzed {family_analysis['summary']['total_models']} models")
        print(f"Hidden size range: {family_analysis['summary']['dimension_ranges']}")
        
        # Show compatibility matrix
        print("\nCompatibility Matrix:")
        for pair, compat in family_analysis['compatibility_matrix'].items():
            source, target = pair.split('_to_')
            print(f"{source} -> {target}: {compat['overall_compatibility']:.3f}")
        
        # Show intervention strategies
        architectures = family_analysis['architectures']
        if len(architectures) >= 2:
            arch_names = list(architectures.keys())
            source_arch = architectures[arch_names[0]]
            target_arch = architectures[arch_names[1]]
            
            strategies = analyzer.suggest_intervention_strategies(source_arch, target_arch)
            print(f"\nIntervention Strategies ({arch_names[0]} -> {arch_names[1]}):")
            print(json.dumps(strategies, indent=2))
    
    except Exception as e:
        print(f"Demo failed: {e}")
        logger.error(f"Architecture analysis demo error: {e}")

if __name__ == "__main__":
    demonstrate_architecture_analysis()