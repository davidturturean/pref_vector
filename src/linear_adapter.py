import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm import tqdm
from dataclasses import dataclass
import json
import os

from .config import EXPERIMENT_CONFIG
from .vector_injection import SteerableModel
from .evaluation_metrics import PreferenceVectorEvaluator, EvaluationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdapterTrainingData:
    """Container for adapter training data."""
    source_vectors: torch.Tensor  # Vectors from source model space
    target_effects: torch.Tensor  # Desired effects in target model
    prompts: List[str]  # Associated prompts
    
class LinearAdapter(nn.Module):
    """Linear transformation layer for mapping vectors between model spaces."""
    
    def __init__(self, input_dim: int, output_dim: int = None, 
                 constraint_type: str = "none"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.constraint_type = constraint_type
        
        if constraint_type == "diagonal":
            # Diagonal scaling transformation
            self.transform = nn.Parameter(torch.ones(self.input_dim))
        elif constraint_type == "orthogonal":
            # Orthogonal transformation (rotation)
            self.transform = nn.Parameter(torch.eye(self.input_dim))
        else:
            # Full linear transformation
            self.transform = nn.Linear(self.input_dim, self.output_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation to input vector."""
        if self.constraint_type == "diagonal":
            return x * self.transform
        elif self.constraint_type == "orthogonal":
            return torch.matmul(x, self.transform)
        else:
            return self.transform(x)
    
    def get_transformation_matrix(self) -> torch.Tensor:
        """Get the transformation matrix."""
        if self.constraint_type == "diagonal":
            return torch.diag(self.transform)
        elif self.constraint_type == "orthogonal":
            return self.transform
        else:
            return self.transform.weight

class AdapterDataset(Dataset):
    """Dataset for training linear adapters."""
    
    def __init__(self, source_vectors: torch.Tensor, target_effects: torch.Tensor):
        self.source_vectors = source_vectors
        self.target_effects = target_effects
        
        assert len(source_vectors) == len(target_effects), \
            "Source vectors and target effects must have same length"
    
    def __len__(self):
        return len(self.source_vectors)
    
    def __getitem__(self, idx):
        return self.source_vectors[idx], self.target_effects[idx]

class AdapterTrainer:
    """Trains linear adapters for cross-model vector alignment."""
    
    def __init__(self, 
                 source_model: SteerableModel,
                 target_model: SteerableModel,
                 layer_idx: int):
        self.source_model = source_model
        self.target_model = target_model
        self.layer_idx = layer_idx
        self.evaluator = PreferenceVectorEvaluator()
        
        # Determine vector dimensions
        self.source_dim = source_model.model.config.hidden_size
        self.target_dim = target_model.model.config.hidden_size
        
        if self.source_dim != self.target_dim:
            logger.warning(f"Dimension mismatch: source={self.source_dim}, target={self.target_dim}")
    
    def generate_training_data(self,
                              base_prompts: List[str],
                              reference_vectors: List[torch.Tensor],
                              num_samples_per_vector: int = 5) -> AdapterTrainingData:
        """Generate training data by measuring effects of vectors on both models."""
        logger.info(f"Generating adapter training data from {len(reference_vectors)} reference vectors")
        
        source_vectors = []
        target_effects = []
        used_prompts = []
        
        for vector_idx, ref_vector in enumerate(reference_vectors):
            for prompt_idx, prompt in enumerate(base_prompts[:num_samples_per_vector]):
                try:
                    # Get baseline generations from both models
                    source_baseline = self.source_model.generate_text(prompt, max_new_tokens=100)
                    target_baseline = self.target_model.generate_text(prompt, max_new_tokens=100)
                    
                    # Get steered generation from source model
                    source_steered = self.source_model.generate_steered(
                        prompt, ref_vector, self.layer_idx, scale=1.0, max_new_tokens=100
                    )
                    
                    # Measure the effect in source model
                    source_effect = self._measure_generation_effect(
                        source_baseline, source_steered
                    )
                    
                    # Store the vector and its measured effect
                    source_vectors.append(ref_vector.clone())
                    target_effects.append(torch.tensor(source_effect, dtype=torch.float32))
                    used_prompts.append(prompt)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate training sample: {e}")
                    continue
        
        if not source_vectors:
            raise RuntimeError("No training data generated")
        
        return AdapterTrainingData(
            source_vectors=torch.stack(source_vectors),
            target_effects=torch.stack(target_effects),
            prompts=used_prompts
        )
    
    def _measure_generation_effect(self, baseline: str, steered: str) -> List[float]:
        """Measure the effect of steering as a feature vector."""
        from .evaluation_metrics import LengthAnalyzer, VerbosityScorer
        
        # Length changes
        baseline_length = LengthAnalyzer.word_count(baseline)
        steered_length = LengthAnalyzer.word_count(steered)
        length_ratio = steered_length / max(1, baseline_length)
        
        # Verbosity changes
        baseline_verbosity = VerbosityScorer.calculate_verbosity_score(baseline)
        steered_verbosity = VerbosityScorer.calculate_verbosity_score(steered)
        verbosity_change = steered_verbosity - baseline_verbosity
        
        # Complexity changes
        baseline_complexity = VerbosityScorer.complexity_score(baseline)
        steered_complexity = VerbosityScorer.complexity_score(steered)
        complexity_change = steered_complexity - baseline_complexity
        
        return [length_ratio, verbosity_change, complexity_change]
    
    def train_adapter(self,
                     training_data: AdapterTrainingData,
                     constraint_type: str = "none",
                     learning_rate: float = 1e-3,
                     num_epochs: int = 100,
                     batch_size: int = 8) -> LinearAdapter:
        """Train a linear adapter using the training data."""
        logger.info(f"Training adapter with {len(training_data.source_vectors)} samples")
        
        # Create adapter
        adapter = LinearAdapter(
            self.source_dim, 
            self.target_dim, 
            constraint_type=constraint_type
        )
        
        # Create dataset and dataloader
        dataset = AdapterDataset(training_data.source_vectors, training_data.target_effects)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(adapter.parameters(), lr=learning_rate)
        
        # Training loop
        adapter.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch_vectors, batch_effects in dataloader:
                optimizer.zero_grad()
                
                # Transform vectors
                transformed_vectors = adapter(batch_vectors)
                
                # Compute loss - we want the transformed vector to produce similar effects
                # This is a simplified proxy loss
                loss = self._compute_adapter_loss(
                    transformed_vectors, batch_effects, training_data.prompts[:len(batch_vectors)]
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
        
        adapter.eval()
        return adapter
    
    def _compute_adapter_loss(self, 
                             transformed_vectors: torch.Tensor,
                             target_effects: torch.Tensor,
                             prompts: List[str]) -> torch.Tensor:
        """Compute loss for adapter training."""
        # Simplified loss based on vector norms and target effects
        # In practice, this would involve generating text and measuring effects
        
        vector_norms = torch.norm(transformed_vectors, dim=-1)
        target_norms = torch.norm(target_effects, dim=-1)
        
        # Loss encourages transformed vectors to have similar norms to target effects
        norm_loss = nn.MSELoss()(vector_norms, target_norms)
        
        # Regularization to prevent degenerate solutions
        reg_loss = torch.mean(torch.norm(transformed_vectors, dim=-1))
        
        return norm_loss + 0.01 * reg_loss
    
    def evaluate_adapter(self,
                        adapter: LinearAdapter,
                        test_vector: torch.Tensor,
                        test_prompts: List[str],
                        scales: List[float] = [0.0, 1.0]) -> Dict:
        """Evaluate adapter performance on test prompts."""
        logger.info(f"Evaluating adapter on {len(test_prompts)} test prompts")
        
        # Transform the test vector
        with torch.no_grad():
            adapted_vector = adapter(test_vector.unsqueeze(0)).squeeze(0)
        
        results = {
            'original_vector': {
                'norm': test_vector.norm().item(),
                'generations': {}
            },
            'adapted_vector': {
                'norm': adapted_vector.norm().item(),
                'generations': {}
            },
            'evaluation_scores': {}
        }
        
        for prompt_idx, prompt in enumerate(test_prompts):
            try:
                # Test original vector (should have little effect)
                original_generations = self.target_model.compare_generations(
                    prompt, test_vector, self.layer_idx, scales
                )
                
                # Test adapted vector
                adapted_generations = self.target_model.compare_generations(
                    prompt, adapted_vector, self.layer_idx, scales
                )
                
                results['original_vector']['generations'][f'prompt_{prompt_idx}'] = original_generations
                results['adapted_vector']['generations'][f'prompt_{prompt_idx}'] = adapted_generations
                
                # Evaluate the adaptation
                if 0.0 in adapted_generations and 1.0 in adapted_generations:
                    eval_result = self.evaluator.evaluate_single_generation(
                        prompt,
                        adapted_generations[0.0],
                        adapted_generations[1.0],
                        "verbose"
                    )
                    results['evaluation_scores'][f'prompt_{prompt_idx}'] = eval_result.overall_score
                
            except Exception as e:
                logger.warning(f"Failed to evaluate prompt {prompt_idx}: {e}")
                continue
        
        # Compute average scores
        if results['evaluation_scores']:
            results['average_score'] = np.mean(list(results['evaluation_scores'].values()))
        else:
            results['average_score'] = 0.0
        
        return results

class AdapterLibrary:
    """Manages a library of trained adapters for different model pairs."""
    
    def __init__(self, library_path: str = "adapters/"):
        self.library_path = library_path
        os.makedirs(library_path, exist_ok=True)
    
    def save_adapter(self, 
                    adapter: LinearAdapter,
                    source_model_name: str,
                    target_model_name: str,
                    metadata: Dict = None) -> str:
        """Save an adapter to the library."""
        filename = f"{source_model_name.replace('/', '_')}_to_{target_model_name.replace('/', '_')}.pt"
        filepath = os.path.join(self.library_path, filename)
        
        save_data = {
            'adapter_state_dict': adapter.state_dict(),
            'adapter_config': {
                'input_dim': adapter.input_dim,
                'output_dim': adapter.output_dim,
                'constraint_type': adapter.constraint_type
            },
            'source_model': source_model_name,
            'target_model': target_model_name,
            'metadata': metadata or {}
        }
        
        torch.save(save_data, filepath)
        logger.info(f"Saved adapter to {filepath}")
        return filepath
    
    def load_adapter(self, 
                    source_model_name: str,
                    target_model_name: str) -> Tuple[LinearAdapter, Dict]:
        """Load an adapter from the library."""
        filename = f"{source_model_name.replace('/', '_')}_to_{target_model_name.replace('/', '_')}.pt"
        filepath = os.path.join(self.library_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Adapter not found: {filepath}")
        
        save_data = torch.load(filepath, map_location='cpu')
        
        # Reconstruct adapter
        config = save_data['adapter_config']
        adapter = LinearAdapter(
            config['input_dim'],
            config['output_dim'],
            config['constraint_type']
        )
        adapter.load_state_dict(save_data['adapter_state_dict'])
        
        metadata = save_data.get('metadata', {})
        logger.info(f"Loaded adapter from {filepath}")
        
        return adapter, metadata

def train_cross_model_adapter(source_model_name: str,
                             target_model_name: str,
                             reference_vectors: List[torch.Tensor],
                             test_prompts: List[str],
                             layer_idx: int = 16) -> Tuple[LinearAdapter, Dict]:
    """Main function to train and evaluate a cross-model adapter."""
    logger.info(f"Training adapter: {source_model_name} -> {target_model_name}")
    
    # Load models
    source_model = SteerableModel(source_model_name)
    target_model = SteerableModel(target_model_name)
    
    # Create trainer
    trainer = AdapterTrainer(source_model, target_model, layer_idx)
    
    # Generate training data
    training_data = trainer.generate_training_data(
        test_prompts[:5],  # Use subset for training
        reference_vectors
    )
    
    # Train adapter
    adapter = trainer.train_adapter(training_data)
    
    # Evaluate adapter
    if reference_vectors:
        test_vector = reference_vectors[0]  # Use first vector for testing
        evaluation_results = trainer.evaluate_adapter(
            adapter, test_vector, test_prompts[5:10]  # Use different prompts for testing
        )
    else:
        evaluation_results = {}
    
    return adapter, evaluation_results

if __name__ == "__main__":
    # Demo with smaller models
    source_model = "microsoft/DialoGPT-medium"
    target_model = "microsoft/DialoGPT-small"
    
    # Create dummy reference vectors
    dummy_vectors = [torch.randn(1024) * 0.1 for _ in range(3)]
    
    test_prompts = [
        "Explain artificial intelligence:",
        "Describe climate change:",
        "Summarize the benefits of exercise:"
    ]
    
    try:
        adapter, results = train_cross_model_adapter(
            source_model, target_model, dummy_vectors, test_prompts, layer_idx=8
        )
        print(f"Adapter training completed. Average evaluation score: {results.get('average_score', 0):.3f}")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("Adapter training demo failed - this is expected without proper model setup")