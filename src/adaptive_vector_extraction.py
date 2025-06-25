import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

from .architecture_analysis import ArchitectureAnalyzer, ModelArchitecture
from .vector_extraction import PreferenceVectorExtractor
from .data_preparation import SummaryPair
from .utils import LayerUtils, VectorUtils, handle_errors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdaptiveVector:
    """Architecture-aware preference vector with metadata."""
    vector: torch.Tensor
    source_architecture: ModelArchitecture
    extraction_layer: int
    extraction_method: str
    functional_layer_type: str  # 'residual', 'attention_output', 'mlp_output'
    normalization_applied: bool
    dimensionality_reduction: Optional[Dict] = None

class FunctionalLayerMapper:
    """Maps layers based on functional role rather than position."""
    
    def __init__(self):
        self.layer_functions = {
            'early_processing': 0.1,    # First 10% of layers
            'feature_extraction': 0.3,  # 10-40% of layers  
            'representation': 0.6,      # 40-70% of layers
            'decision_making': 0.9,     # 70-100% of layers
        }
    
    def get_functional_layer(self, 
                           architecture: ModelArchitecture,
                           function_type: str) -> int:
        """Get layer index based on functional role."""
        if function_type not in self.layer_functions:
            raise ValueError(f"Unknown function type: {function_type}")
        
        relative_depth = self.layer_functions[function_type]
        layer_index = int(relative_depth * architecture.num_layers)
        return min(layer_index, architecture.num_layers - 1)
    
    def map_functional_layers(self,
                            source_arch: ModelArchitecture,
                            target_arch: ModelArchitecture,
                            source_layer: int) -> int:
        """Map layer based on functional equivalence."""
        return LayerUtils.map_functional_layer(
            source_arch.num_layers, 
            target_arch.num_layers, 
            source_layer
        )

class MultiPointExtractor:
    """Extract vectors from multiple architectural points for robustness."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.analyzer = ArchitectureAnalyzer()
        self.architecture = self.analyzer.analyze_model_architecture(model_name)
        self.base_extractor = PreferenceVectorExtractor(model_name)
        self.functional_mapper = FunctionalLayerMapper()
    
    def extract_multi_point_vectors(self, 
                                   summary_pairs: List[SummaryPair],
                                   extraction_points: List[str] = None) -> Dict[str, AdaptiveVector]:
        """Extract vectors from multiple architectural points."""
        if extraction_points is None:
            extraction_points = ['feature_extraction', 'representation', 'decision_making']
        
        vectors = {}
        
        for point_name in extraction_points:
            try:
                # Get functional layer for this extraction point
                layer_idx = self.functional_mapper.get_functional_layer(
                    self.architecture, point_name
                )
                
                logger.info(f"Extracting vector from {point_name} layer (index {layer_idx})")
                
                # Extract vector at this layer
                self.base_extractor.intervention_layer = layer_idx
                vector = self.base_extractor.extract_preference_vector_from_pairs(
                    summary_pairs, method="difference"
                )
                
                # Create adaptive vector with metadata
                adaptive_vector = AdaptiveVector(
                    vector=vector,
                    source_architecture=self.architecture,
                    extraction_layer=layer_idx,
                    extraction_method="difference",
                    functional_layer_type=point_name,
                    normalization_applied=True
                )
                
                vectors[point_name] = adaptive_vector
                
            except Exception as e:
                logger.warning(f"Failed to extract vector from {point_name}: {e}")
                continue
        
        return vectors
    
    def extract_attention_vs_mlp_vectors(self, 
                                       summary_pairs: List[SummaryPair]) -> Dict[str, AdaptiveVector]:
        """Extract vectors specifically from attention vs MLP components."""
        vectors = {}
        
        # Try different component-specific extractions
        extraction_configs = [
            ('attention_mid', 'representation', 'attention_output'),
            ('mlp_mid', 'representation', 'mlp_output'),
            ('residual_mid', 'representation', 'residual')
        ]
        
        for config_name, functional_layer, component_type in extraction_configs:
            try:
                layer_idx = self.functional_mapper.get_functional_layer(
                    self.architecture, functional_layer
                )
                
                # Use component-specific extraction
                vector = self._extract_component_specific_vector(
                    summary_pairs, layer_idx, component_type
                )
                
                if vector is not None:
                    adaptive_vector = AdaptiveVector(
                        vector=vector,
                        source_architecture=self.architecture,
                        extraction_layer=layer_idx,
                        extraction_method="component_specific",
                        functional_layer_type=component_type,
                        normalization_applied=True
                    )
                    vectors[config_name] = adaptive_vector
                
            except Exception as e:
                logger.warning(f"Failed component extraction {config_name}: {e}")
                continue
        
        return vectors
    
    def _extract_component_specific_vector(self,
                                         summary_pairs: List[SummaryPair],
                                         layer_idx: int,
                                         component_type: str) -> Optional[torch.Tensor]:
        """Extract vector from specific architectural component."""
        # This would require specialized hooks for different components
        # For now, fallback to standard extraction
        self.base_extractor.intervention_layer = layer_idx
        return self.base_extractor.extract_preference_vector_from_pairs(
            summary_pairs, method="difference"
        )

class DimensionAdaptiveExtractor:
    """Handle extraction across different hidden dimensions."""
    
    def __init__(self):
        self.dimension_adapters = {}
    
    def extract_with_dimension_adaptation(self,
                                        extractor: MultiPointExtractor,
                                        summary_pairs: List[SummaryPair],
                                        target_dimension: int = None) -> Dict[str, AdaptiveVector]:
        """Extract vectors with automatic dimension adaptation."""
        vectors = extractor.extract_multi_point_vectors(summary_pairs)
        
        if target_dimension is None:
            return vectors
        
        adapted_vectors = {}
        for name, adaptive_vector in vectors.items():
            current_dim = adaptive_vector.vector.shape[0]
            
            if current_dim != target_dimension:
                logger.info(f"Adapting vector {name} from {current_dim}D to {target_dimension}D")
                
                # Apply dimension adaptation
                adapted_tensor, adaptation_info = self._adapt_dimension(
                    adaptive_vector.vector, target_dimension
                )
                
                # Update adaptive vector
                adaptive_vector.vector = adapted_tensor
                adaptive_vector.dimensionality_reduction = adaptation_info
            
            adapted_vectors[name] = adaptive_vector
        
        return adapted_vectors
    
    def _adapt_dimension(self, 
                        vector: torch.Tensor, 
                        target_dim: int) -> Tuple[torch.Tensor, Dict]:
        """Adapt vector dimension using various strategies."""
        current_dim = vector.shape[0]
        adaptation_info = {
            'original_dim': current_dim,
            'target_dim': target_dim,
            'method': None,
            'preservation_ratio': None
        }
        
        # Use shared utility for dimension adaptation
        adapted_vector = VectorUtils.adapt_vector_dimension(vector, target_dim)
        
        if current_dim > target_dim:
            adaptation_info['method'] = 'truncation'
            adaptation_info['preservation_ratio'] = target_dim / current_dim
        elif current_dim < target_dim:
            adaptation_info['method'] = 'zero_padding'
            adaptation_info['preservation_ratio'] = current_dim / target_dim
        else:
            adaptation_info['method'] = 'no_change'
            adaptation_info['preservation_ratio'] = 1.0
        
        return adapted_vector, adaptation_info

class ArchitectureAwareInjector:
    """Inject vectors across different architectures intelligently."""
    
    def __init__(self):
        self.analyzer = ArchitectureAnalyzer()
        self.functional_mapper = FunctionalLayerMapper()
        self.injection_strategies = {}
    
    def create_injection_strategy(self,
                                source_arch: ModelArchitecture,
                                target_arch: ModelArchitecture,
                                adaptive_vector: AdaptiveVector) -> Dict[str, Any]:
        """Create architecture-aware injection strategy."""
        compatibility = self.analyzer.compute_compatibility_score(source_arch, target_arch)
        
        strategy = {
            'compatibility_score': compatibility['overall_compatibility'],
            'injection_method': None,
            'target_layer': None,
            'scaling_factor': 1.0,
            'pre_processing': [],
            'post_processing': []
        }
        
        # Determine injection method based on compatibility
        if compatibility['overall_compatibility'] > 0.8:
            strategy['injection_method'] = 'direct'
            strategy['target_layer'] = self.functional_mapper.map_functional_layers(
                source_arch, target_arch, adaptive_vector.extraction_layer
            )
        
        elif compatibility['dimension_compatibility'] > 0.9:
            strategy['injection_method'] = 'scaled_direct'
            strategy['target_layer'] = self.functional_mapper.map_functional_layers(
                source_arch, target_arch, adaptive_vector.extraction_layer
            )
            strategy['scaling_factor'] = compatibility['dimension_compatibility']
        
        else:
            strategy['injection_method'] = 'adapted'
            strategy['pre_processing'].append('dimension_adaptation')
            if compatibility['norm_compatibility'] < 0.5:
                strategy['pre_processing'].append('normalization_adjustment')
        
        return strategy
    
    def inject_with_strategy(self,
                           target_model,
                           adaptive_vector: AdaptiveVector,
                           injection_strategy: Dict[str, Any],
                           prompt: str) -> str:
        """Inject vector using the determined strategy."""
        # Apply pre-processing
        processed_vector = adaptive_vector.vector.clone()
        
        for preprocessing_step in injection_strategy.get('pre_processing', []):
            processed_vector = self._apply_preprocessing(
                processed_vector, preprocessing_step, injection_strategy
            )
        
        # Apply scaling
        processed_vector *= injection_strategy.get('scaling_factor', 1.0)
        
        # Inject and generate
        if injection_strategy['injection_method'] in ['direct', 'scaled_direct']:
            return target_model.generate_steered(
                prompt,
                processed_vector,
                injection_strategy['target_layer']
            )
        else:
            # More complex injection strategies would be implemented here
            logger.warning("Advanced injection strategies not yet implemented")
            return target_model.generate_text(prompt)
    
    def _apply_preprocessing(self, 
                           vector: torch.Tensor, 
                           step: str, 
                           strategy: Dict) -> torch.Tensor:
        """Apply preprocessing step to vector."""
        if step == 'dimension_adaptation':
            # This would involve more sophisticated dimension adaptation
            return vector
        elif step == 'normalization_adjustment':
            # Adjust normalization to match target architecture
            return vector / vector.norm() * strategy.get('target_norm', 1.0)
        else:
            return vector

class RobustCrossModelExtractor:
    """Main class for robust cross-model preference vector extraction."""
    
    def __init__(self):
        self.analyzer = ArchitectureAnalyzer()
        self.extractors = {}
        self.injector = ArchitectureAwareInjector()
    
    def extract_robust_preference_vectors(self,
                                        model_name: str,
                                        summary_pairs: List[SummaryPair],
                                        target_architectures: List[str] = None) -> Dict[str, Dict[str, AdaptiveVector]]:
        """Extract preference vectors optimized for cross-model transfer."""
        if model_name not in self.extractors:
            self.extractors[model_name] = MultiPointExtractor(model_name)
        
        extractor = self.extractors[model_name]
        
        # Extract vectors from multiple architectural points
        multi_point_vectors = extractor.extract_multi_point_vectors(summary_pairs)
        
        # Extract component-specific vectors
        component_vectors = extractor.extract_attention_vs_mlp_vectors(summary_pairs)
        
        all_vectors = {**multi_point_vectors, **component_vectors}
        
        # If target architectures specified, create adapted versions
        if target_architectures:
            adapted_versions = {}
            for target_arch_name in target_architectures:
                try:
                    target_arch = self.analyzer.analyze_model_architecture(target_arch_name)
                    
                    # Create dimension-adapted versions
                    dimension_adapter = DimensionAdaptiveExtractor()
                    adapted_vectors = dimension_adapter.extract_with_dimension_adaptation(
                        extractor, summary_pairs, target_arch.hidden_size
                    )
                    
                    adapted_versions[target_arch_name] = adapted_vectors
                    
                except Exception as e:
                    logger.warning(f"Failed to create adapted vectors for {target_arch_name}: {e}")
                    continue
            
            return {
                'universal': all_vectors,
                'adapted': adapted_versions
            }
        
        return {'universal': all_vectors}
    
    def test_cross_architecture_transfer(self,
                                       source_vectors: Dict[str, AdaptiveVector],
                                       target_model,
                                       target_arch: ModelArchitecture,
                                       test_prompts: List[str]) -> Dict[str, Dict]:
        """Test transfer across architectures with multiple vector types."""
        results = {}
        
        for vector_name, adaptive_vector in source_vectors.items():
            logger.info(f"Testing transfer of {vector_name} vector")
            
            # Create injection strategy
            strategy = self.injector.create_injection_strategy(
                adaptive_vector.source_architecture,
                target_arch,
                adaptive_vector
            )
            
            vector_results = {
                'strategy': strategy,
                'generations': {},
                'success_metrics': {}
            }
            
            # Test on prompts
            for i, prompt in enumerate(test_prompts):
                try:
                    generated_text = self.injector.inject_with_strategy(
                        target_model, adaptive_vector, strategy, prompt
                    )
                    vector_results['generations'][f'prompt_{i}'] = generated_text
                    
                except Exception as e:
                    logger.warning(f"Failed injection for {vector_name}, prompt {i}: {e}")
                    vector_results['generations'][f'prompt_{i}'] = None
            
            results[vector_name] = vector_results
        
        return results

# Demonstration and testing
@handle_errors("demonstrate robust extraction")
def demonstrate_robust_extraction():
    """Demonstrate the robust extraction system."""
    print("Robust Cross-Model Vector Extraction Demo")
    print("=" * 50)
    
    try:
        # Initialize robust extractor
        robust_extractor = RobustCrossModelExtractor()
        
        # Test with small models
        source_model = "microsoft/DialoGPT-medium"
        target_models = ["microsoft/DialoGPT-small", "gpt2"]
        
        # Create mock summary pairs
        from .data_preparation import SummaryPair
        mock_pairs = [
            SummaryPair(
                source_text="Test article about AI development.",
                concise_summary="AI is advancing.",
                verbose_summary="Artificial intelligence technology continues to advance rapidly with new breakthroughs.",
                source_length=10, concise_length=3, verbose_length=12
            )
        ]
        
        # Extract robust vectors
        print(f"Extracting robust vectors from {source_model}...")
        vector_sets = robust_extractor.extract_robust_preference_vectors(
            source_model, mock_pairs, target_models
        )
        
        print(f"Extracted {len(vector_sets)} vector sets:")
        for set_name, vectors in vector_sets.items():
            if isinstance(vectors, dict):
                for vector_name, adaptive_vector in vectors.items():
                    if hasattr(adaptive_vector, 'vector'):
                        print(f"  {set_name}.{vector_name}: shape {adaptive_vector.vector.shape}")
        
        print("Robust extraction demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        logger.error(f"Robust extraction demo error: {e}")

if __name__ == "__main__":
    demonstrate_robust_extraction()