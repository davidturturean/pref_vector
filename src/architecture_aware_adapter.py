import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm
from dataclasses import dataclass
import json
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .architecture_analysis import ArchitectureAnalyzer, ModelArchitecture
from .adaptive_vector_extraction import AdaptiveVector, RobustCrossModelExtractor
from .linear_adapter import LinearAdapter, AdapterTrainingData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArchitecturalAlignment:
    """Information about alignment between two architectures."""
    source_arch: ModelArchitecture
    target_arch: ModelArchitecture
    dimension_mapping: Dict[int, int]
    layer_correspondence: Dict[int, int]
    compatibility_scores: Dict[str, float]
    recommended_adapter_type: str
    adaptation_complexity: str

class ArchitectureAwareAdapter(nn.Module):
    """Advanced adapter that handles architectural differences."""
    
    def __init__(self, 
                 source_arch: ModelArchitecture,
                 target_arch: ModelArchitecture,
                 adaptation_strategy: str = "auto"):
        super().__init__()
        
        self.source_arch = source_arch
        self.target_arch = target_arch
        self.adaptation_strategy = adaptation_strategy
        
        # Determine adaptation components needed
        self.components = self._design_adaptation_components()
        
        # Build the adapter network
        self._build_adapter_network()
    
    def _design_adaptation_components(self) -> Dict[str, bool]:
        """Design which adaptation components are needed."""
        components = {
            'dimension_transform': self.source_arch.hidden_size != self.target_arch.hidden_size,
            'attention_adaptation': self.source_arch.attention_type != self.target_arch.attention_type,
            'normalization_adaptation': self.source_arch.layer_norm_type != self.target_arch.layer_norm_type,
            'activation_adaptation': self.source_arch.activation_function != self.target_arch.activation_function,
            'positional_encoding': False,  # Advanced feature
            'tokenizer_alignment': self.source_arch.tokenizer_type != self.target_arch.tokenizer_type
        }
        
        logger.info(f"Adaptation components needed: {[k for k, v in components.items() if v]}")
        return components
    
    def _build_adapter_network(self):
        """Build the neural network components for adaptation."""
        layers = []
        
        current_dim = self.source_arch.hidden_size
        target_dim = self.target_arch.hidden_size
        
        # Dimension transformation
        if self.components['dimension_transform']:
            if self.adaptation_strategy in ["auto", "linear"]:
                self.dimension_transform = nn.Linear(current_dim, target_dim, bias=False)
            elif self.adaptation_strategy == "bottleneck":
                # Bottleneck architecture for regularization
                bottleneck_dim = min(current_dim, target_dim) // 2
                self.dimension_transform = nn.Sequential(
                    nn.Linear(current_dim, bottleneck_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(bottleneck_dim, target_dim, bias=False)
                )
            elif self.adaptation_strategy == "residual":
                # Residual connection for similar dimensions
                if abs(current_dim - target_dim) / max(current_dim, target_dim) < 0.2:
                    self.dimension_transform = ResidualAdapter(current_dim, target_dim)
                else:
                    self.dimension_transform = nn.Linear(current_dim, target_dim, bias=False)
        else:
            self.dimension_transform = nn.Identity()
        
        # Attention mechanism adaptation
        if self.components['attention_adaptation']:
            self.attention_adapter = AttentionMechanismAdapter(
                self.source_arch, self.target_arch
            )
        else:
            self.attention_adapter = nn.Identity()
        
        # Normalization adaptation
        if self.components['normalization_adaptation']:
            self.normalization_adapter = NormalizationAdapter(
                self.source_arch.layer_norm_type,
                self.target_arch.layer_norm_type,
                target_dim
            )
        else:
            self.normalization_adapter = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the architecture adapter."""
        # Apply dimension transformation
        x = self.dimension_transform(x)
        
        # Apply attention adaptation
        x = self.attention_adapter(x)
        
        # Apply normalization adaptation
        x = self.normalization_adapter(x)
        
        return x

class ResidualAdapter(nn.Module):
    """Residual adapter for similar-sized dimensions."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        if input_dim == output_dim:
            self.transform = nn.Identity()
        elif input_dim < output_dim:
            # Pad with learnable parameters
            self.pad_size = output_dim - input_dim
            self.padding = nn.Parameter(torch.randn(self.pad_size) * 0.01)
        else:
            # Project down with learnable weights
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim == self.output_dim:
            return x
        elif self.input_dim < self.output_dim:
            # Concatenate with learned padding
            batch_size = x.shape[0] if len(x.shape) > 1 else 1
            padding = self.padding.unsqueeze(0).expand(batch_size, -1)
            return torch.cat([x, padding], dim=-1)
        else:
            return self.projection(x)

class AttentionMechanismAdapter(nn.Module):
    """Adapter for different attention mechanisms."""
    
    def __init__(self, source_arch: ModelArchitecture, target_arch: ModelArchitecture):
        super().__init__()
        self.source_type = source_arch.attention_type
        self.target_type = target_arch.attention_type
        
        # Simple adaptation: just use learned scaling
        self.adaptation_scale = nn.Parameter(torch.ones(1))
        
        if self.source_type != self.target_type:
            logger.info(f"Adapting attention: {self.source_type} -> {self.target_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.adaptation_scale

class NormalizationAdapter(nn.Module):
    """Adapter for different normalization schemes."""
    
    def __init__(self, source_norm: str, target_norm: str, hidden_dim: int):
        super().__init__()
        self.source_norm = source_norm
        self.target_norm = target_norm
        
        if source_norm != target_norm:
            # Learn adaptation parameters
            self.scale_adaptation = nn.Parameter(torch.ones(hidden_dim))
            self.bias_adaptation = nn.Parameter(torch.zeros(hidden_dim))
            logger.info(f"Adapting normalization: {source_norm} -> {target_norm}")
        else:
            self.scale_adaptation = None
            self.bias_adaptation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_adaptation is not None:
            return x * self.scale_adaptation + self.bias_adaptation
        return x

class MultiScaleAdapter(nn.Module):
    """Adapter that works at multiple representation scales."""
    
    def __init__(self, 
                 source_arch: ModelArchitecture,
                 target_arch: ModelArchitecture,
                 num_scales: int = 3):
        super().__init__()
        
        self.num_scales = num_scales
        self.source_dim = source_arch.hidden_size
        self.target_dim = target_arch.hidden_size
        
        # Create multi-scale adaptation
        self.scale_adapters = nn.ModuleList()
        for i in range(num_scales):
            # Different scales process different frequency components
            scale_adapter = nn.Sequential(
                nn.Linear(self.source_dim, self.target_dim, bias=False),
                nn.Tanh()  # Bounded activation for stability
            )
            self.scale_adapters.append(scale_adapter)
        
        # Combine scales
        self.scale_combiner = nn.Linear(num_scales * self.target_dim, self.target_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_outputs = []
        
        for adapter in self.scale_adapters:
            scale_output = adapter(x)
            scale_outputs.append(scale_output)
        
        # Concatenate and combine
        combined = torch.cat(scale_outputs, dim=-1)
        return self.scale_combiner(combined)

class ArchitectureAwareAdapterTrainer:
    """Trainer for architecture-aware adapters."""
    
    def __init__(self):
        self.analyzer = ArchitectureAnalyzer()
        self.robust_extractor = RobustCrossModelExtractor()
    
    def analyze_architectural_alignment(self,
                                      source_model_name: str,
                                      target_model_name: str) -> ArchitecturalAlignment:
        """Analyze alignment between source and target architectures."""
        source_arch = self.analyzer.analyze_model_architecture(source_model_name)
        target_arch = self.analyzer.analyze_model_architecture(target_model_name)
        
        # Compute compatibility
        compatibility = self.analyzer.compute_compatibility_score(source_arch, target_arch)
        
        # Find layer correspondence
        layer_correspondence = {}
        for source_layer in range(source_arch.num_layers):
            relative_depth = source_layer / (source_arch.num_layers - 1) if source_arch.num_layers > 1 else 0
            target_layer = int(relative_depth * (target_arch.num_layers - 1))
            layer_correspondence[source_layer] = target_layer
        
        # Dimension mapping (for now, just identity or simple correspondence)
        dimension_mapping = {i: i for i in range(min(source_arch.hidden_size, target_arch.hidden_size))}
        
        # Recommend adapter type
        if compatibility['overall_compatibility'] > 0.8:
            adapter_type = "simple_linear"
            complexity = "low"
        elif compatibility['dimension_compatibility'] > 0.7:
            adapter_type = "residual"
            complexity = "medium"
        else:
            adapter_type = "multi_scale"
            complexity = "high"
        
        return ArchitecturalAlignment(
            source_arch=source_arch,
            target_arch=target_arch,
            dimension_mapping=dimension_mapping,
            layer_correspondence=layer_correspondence,
            compatibility_scores=compatibility,
            recommended_adapter_type=adapter_type,
            adaptation_complexity=complexity
        )
    
    def create_architecture_aware_adapter(self,
                                        alignment: ArchitecturalAlignment,
                                        adaptation_strategy: str = None) -> ArchitectureAwareAdapter:
        """Create adapter based on architectural alignment analysis."""
        if adaptation_strategy is None:
            adaptation_strategy = alignment.recommended_adapter_type
        
        if adaptation_strategy == "multi_scale":
            return MultiScaleAdapter(alignment.source_arch, alignment.target_arch)
        else:
            return ArchitectureAwareAdapter(
                alignment.source_arch, 
                alignment.target_arch, 
                adaptation_strategy
            )
    
    def train_with_architectural_awareness(self,
                                         source_model,
                                         target_model, 
                                         source_vectors: Dict[str, AdaptiveVector],
                                         training_prompts: List[str],
                                         alignment: ArchitecturalAlignment,
                                         num_epochs: int = 50) -> ArchitectureAwareAdapter:
        """Train adapter with architectural awareness."""
        
        # Create adapter
        adapter = self.create_architecture_aware_adapter(alignment)
        
        # Prepare training data based on architectural analysis
        training_data = self._prepare_architectural_training_data(
            source_model, target_model, source_vectors, training_prompts, alignment
        )
        
        # Train with architectural constraints
        trained_adapter = self._train_with_constraints(
            adapter, training_data, alignment, num_epochs
        )
        
        return trained_adapter
    
    def _prepare_architectural_training_data(self,
                                           source_model,
                                           target_model,
                                           source_vectors: Dict[str, AdaptiveVector],
                                           training_prompts: List[str],
                                           alignment: ArchitecturalAlignment) -> Dict[str, Any]:
        """Prepare training data considering architectural differences."""
        
        training_examples = []
        
        for vector_name, adaptive_vector in source_vectors.items():
            for prompt in training_prompts:
                try:
                    # Generate with source model using vector
                    source_layer = adaptive_vector.extraction_layer
                    target_layer = alignment.layer_correspondence.get(source_layer, source_layer)
                    
                    # Get baseline and steered generations from source
                    source_baseline = source_model.generate_text(prompt, max_new_tokens=50)
                    source_steered = source_model.generate_steered(
                        prompt, adaptive_vector.vector, source_layer, max_new_tokens=50
                    )
                    
                    # Get baseline from target
                    target_baseline = target_model.generate_text(prompt, max_new_tokens=50)
                    
                    # Measure effect in source model
                    source_effect = self._measure_steering_effect(source_baseline, source_steered)
                    
                    training_examples.append({
                        'source_vector': adaptive_vector.vector,
                        'target_effect': source_effect,  # What we want to achieve in target
                        'prompt': prompt,
                        'source_layer': source_layer,
                        'target_layer': target_layer,
                        'vector_type': vector_name
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to create training example: {e}")
                    continue
        
        return {
            'examples': training_examples,
            'alignment': alignment,
            'source_arch': alignment.source_arch,
            'target_arch': alignment.target_arch
        }
    
    def _measure_steering_effect(self, baseline: str, steered: str) -> torch.Tensor:
        """Measure the effect of steering as a feature vector."""
        # Simple metrics as features
        baseline_len = len(baseline.split())
        steered_len = len(steered.split())
        
        length_ratio = steered_len / max(1, baseline_len)
        length_diff = steered_len - baseline_len
        
        # Additional linguistic features could be added here
        effect_vector = torch.tensor([
            length_ratio,
            length_diff / 100.0,  # Normalized
            float(len(steered) > len(baseline)),  # Binary indicator
        ], dtype=torch.float32)
        
        return effect_vector
    
    def _train_with_constraints(self,
                              adapter: ArchitectureAwareAdapter,
                              training_data: Dict[str, Any],
                              alignment: ArchitecturalAlignment,
                              num_epochs: int) -> ArchitectureAwareAdapter:
        """Train adapter with architectural constraints."""
        
        optimizer = optim.Adam(adapter.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        examples = training_data['examples']
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for example in examples:
                optimizer.zero_grad()
                
                # Transform source vector
                source_vector = example['source_vector']
                transformed_vector = adapter(source_vector.unsqueeze(0)).squeeze(0)
                
                # Compute loss based on desired effect
                target_effect = example['target_effect']
                
                # Loss components
                effect_loss = self._compute_effect_preservation_loss(
                    transformed_vector, target_effect
                )
                
                # Architectural consistency loss
                arch_loss = self._compute_architectural_consistency_loss(
                    source_vector, transformed_vector, alignment
                )
                
                # Regularization loss
                reg_loss = self._compute_regularization_loss(adapter, alignment)
                
                # Combined loss
                total_loss_item = effect_loss + 0.1 * arch_loss + 0.01 * reg_loss
                total_loss_item.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                
                optimizer.step()
                total_loss += total_loss_item.item()
            
            scheduler.step()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(examples)
                logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
        
        return adapter
    
    def _compute_effect_preservation_loss(self,
                                        transformed_vector: torch.Tensor,
                                        target_effect: torch.Tensor) -> torch.Tensor:
        """Compute loss for preserving steering effect."""
        # Simple proxy: vector norm should relate to effect magnitude
        vector_norm = torch.norm(transformed_vector)
        effect_magnitude = torch.norm(target_effect)
        
        return nn.MSELoss()(vector_norm.unsqueeze(0), effect_magnitude.unsqueeze(0))
    
    def _compute_architectural_consistency_loss(self,
                                              source_vector: torch.Tensor,
                                              transformed_vector: torch.Tensor,
                                              alignment: ArchitecturalAlignment) -> torch.Tensor:
        """Compute loss for architectural consistency."""
        # Encourage similar relative magnitudes
        source_norm = torch.norm(source_vector)
        target_norm = torch.norm(transformed_vector)
        
        # Scale based on dimension compatibility
        expected_scale = alignment.compatibility_scores.get('dimension_compatibility', 1.0)
        expected_norm = source_norm * expected_scale
        
        return nn.MSELoss()(target_norm.unsqueeze(0), expected_norm.unsqueeze(0))
    
    def _compute_regularization_loss(self,
                                   adapter: ArchitectureAwareAdapter,
                                   alignment: ArchitecturalAlignment) -> torch.Tensor:
        """Compute regularization loss."""
        reg_loss = 0.0
        
        # L2 regularization on adapter parameters
        for param in adapter.parameters():
            reg_loss += torch.norm(param) ** 2
        
        return reg_loss

# Demonstration
def demonstrate_architecture_aware_adaptation():
    """Demonstrate architecture-aware adaptation."""
    print("Architecture-Aware Adapter Demo")
    print("=" * 40)
    
    try:
        trainer = ArchitectureAwareAdapterTrainer()
        
        # Test with available models
        source_model = "microsoft/DialoGPT-medium"
        target_model = "microsoft/DialoGPT-small"
        
        print(f"Analyzing alignment: {source_model} -> {target_model}")
        alignment = trainer.analyze_architectural_alignment(source_model, target_model)
        
        print(f"Compatibility scores:")
        for metric, score in alignment.compatibility_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        print(f"Recommended adapter: {alignment.recommended_adapter_type}")
        print(f"Adaptation complexity: {alignment.adaptation_complexity}")
        
        # Create adapter
        adapter = trainer.create_architecture_aware_adapter(alignment)
        print(f"Created adapter with {sum(p.numel() for p in adapter.parameters())} parameters")
        
        print("Architecture-aware adaptation demo completed!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        logger.error(f"Architecture-aware adaptation demo error: {e}")

if __name__ == "__main__":
    demonstrate_architecture_aware_adaptation()