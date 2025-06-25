"""
Transfer Matrix Implementation for Cross-Model Preference Vector Analysis

This module implements the core mathematical framework for transferring style 
vectors between different language models, following Huang et al. (2025) methodology
with extensions for comprehensive Platonic Hypothesis analysis.
"""

import numpy as np

# Try to import scipy, fall back to numpy-only implementations if needed
try:
    from scipy.linalg import pinv, orthogonal_procrustes, svd, solve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
    # Numpy-only fallbacks
    def pinv(a, **kwargs):
        return np.linalg.pinv(a)
    
    def orthogonal_procrustes(A, B):
        # Simple procrustes using SVD
        U, s, Vt = np.linalg.svd(A.T @ B)
        R = U @ Vt
        return R, np.linalg.norm(A @ R - B, 'fro')**2
    
    def svd(a, **kwargs):
        return np.linalg.svd(a, full_matrices=False)
    
    def solve(a, b):
        return np.linalg.solve(a, b)
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import time
from collections import defaultdict

from .vector_extraction import VectorCollection, StyleVector

logger = logging.getLogger(__name__)

@dataclass
class TransferMatrix:
    """Container for a transfer matrix between two models."""
    source_model: str
    target_model: str
    matrix: np.ndarray
    method: str  # "pseudoinverse", "procrustes", "regularized_lstsq"
    traits_used: List[str]
    reconstruction_error: float
    condition_number: float
    rank: int
    is_orthogonal: bool = False
    orthogonality_score: float = 0.0  # ||T^T T - I||_F
    singular_values: Optional[np.ndarray] = None
    computation_time: float = 0.0
    regularization_strength: float = 0.0
    
    def __post_init__(self):
        """Compute derived properties."""
        if self.matrix is not None:
            self.condition_number = np.linalg.cond(self.matrix)
            self.rank = np.linalg.matrix_rank(self.matrix)
            
            # Check orthogonality
            if self.matrix.shape[0] == self.matrix.shape[1]:
                orthogonal_check = self.matrix.T @ self.matrix - np.eye(self.matrix.shape[0])
                self.orthogonality_score = np.linalg.norm(orthogonal_check, 'fro')
                self.is_orthogonal = self.orthogonality_score < 1e-10
            
            # Compute singular values
            self.singular_values = np.linalg.svd(self.matrix, compute_uv=False)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['matrix'] = self.matrix.tolist() if self.matrix is not None else None
        data['singular_values'] = self.singular_values.tolist() if self.singular_values is not None else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransferMatrix':
        """Create from dictionary."""
        if data['matrix'] is not None:
            data['matrix'] = np.array(data['matrix'])
        if data['singular_values'] is not None:
            data['singular_values'] = np.array(data['singular_values'])
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]):
        """Save transfer matrix to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TransferMatrix':
        """Load transfer matrix from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def apply_transfer(self, vector: np.ndarray) -> np.ndarray:
        """Apply this transfer matrix to a vector."""
        return vector @ self.matrix.T
    
    def compute_transfer_error(self, source_vector: np.ndarray, target_vector: np.ndarray) -> float:
        """Compute reconstruction error for a specific vector pair."""
        transferred = self.apply_transfer(source_vector)
        return np.linalg.norm(transferred - target_vector)

class TransferMatrixComputer:
    """Computes transfer matrices between model pairs using various methods."""
    
    def __init__(self, method: str = "pseudoinverse", regularization_strength: float = 1e-6):
        self.method = method
        self.regularization_strength = regularization_strength
        self.supported_methods = ["pseudoinverse", "procrustes", "regularized_lstsq"]
        
        if method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Use one of: {self.supported_methods}")
    
    def compute_transfer_matrix(self, source_vectors: np.ndarray, target_vectors: np.ndarray,
                               source_model: str, target_model: str, trait_names: List[str]) -> TransferMatrix:
        """
        Compute transfer matrix from source to target model vectors.
        
        Args:
            source_vectors: (n_traits, hidden_dim) array of source model vectors
            target_vectors: (n_traits, hidden_dim) array of target model vectors
            source_model: Name of source model
            target_model: Name of target model
            trait_names: List of trait names corresponding to vectors
        
        Returns:
            TransferMatrix object with computed transformation
        """
        start_time = time.time()
        
        if source_vectors.shape != target_vectors.shape:
            raise ValueError(f"Vector shapes don't match: {source_vectors.shape} vs {target_vectors.shape}")
        
        logger.info(f"Computing {self.method} transfer matrix: {source_model} -> {target_model}")
        
        if self.method == "pseudoinverse":
            matrix = self._compute_pseudoinverse_matrix(source_vectors, target_vectors)
        elif self.method == "procrustes":
            matrix = self._compute_procrustes_matrix(source_vectors, target_vectors)
        elif self.method == "regularized_lstsq":
            matrix = self._compute_regularized_lstsq_matrix(source_vectors, target_vectors)
        
        # Compute reconstruction error
        reconstruction_error = self._compute_reconstruction_error(source_vectors, target_vectors, matrix)
        
        computation_time = time.time() - start_time
        
        return TransferMatrix(
            source_model=source_model,
            target_model=target_model,
            matrix=matrix,
            method=self.method,
            traits_used=trait_names.copy(),
            reconstruction_error=reconstruction_error,
            condition_number=0.0,  # Will be computed in __post_init__
            rank=0,  # Will be computed in __post_init__
            computation_time=computation_time,
            regularization_strength=self.regularization_strength
        )
    
    def _compute_pseudoinverse_matrix(self, source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
        """
        Compute transfer matrix using pseudoinverse method.
        
        Solves: T = argmin_W ||V^(source) W^T - V^(target)||²_F
        Solution: T = (V^(source))⁺ V^(target)
        """
        # V^(source) is (n_traits, hidden_dim), we want (hidden_dim, hidden_dim) matrix
        # T @ V^(source).T ≈ V^(target).T
        # T ≈ V^(target).T @ pinv(V^(source).T) = V^(target).T @ (V^(source).T)⁺
        
        source_pinv = pinv(source_vectors.T)  # (n_traits, hidden_dim)
        matrix = target_vectors.T @ source_pinv  # (hidden_dim, hidden_dim)
        
        return matrix
    
    def _compute_procrustes_matrix(self, source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
        """
        Compute orthogonal transfer matrix using Procrustes analysis.
        
        Solves: Q = argmin_R∈O(d) ||V^(source) R - V^(target)||_F
        """
        # orthogonal_procrustes expects (n_points, n_dims) format
        R, _ = orthogonal_procrustes(source_vectors, target_vectors)
        return R.T  # Return transpose to match our convention
    
    def _compute_regularized_lstsq_matrix(self, source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
        """
        Compute transfer matrix with Tikhonov regularization.
        
        Solves: T = argmin_W ||V^(source) W^T - V^(target)||²_F + λ||W||²_F
        """
        n_traits, hidden_dim = source_vectors.shape
        
        # Ridge regression for each target dimension
        matrix = np.zeros((hidden_dim, hidden_dim))
        
        for i in range(hidden_dim):
            # Solve for i-th column of transfer matrix
            # (V^T V + λI) w = V^T t_i
            V = source_vectors  # (n_traits, hidden_dim)
            t_i = target_vectors[:, i]  # (n_traits,)
            
            A = V.T @ V + self.regularization_strength * np.eye(hidden_dim)
            b = V.T @ t_i
            
            matrix[i, :] = solve(A, b)
        
        return matrix
    
    def _compute_reconstruction_error(self, source_vectors: np.ndarray, target_vectors: np.ndarray, 
                                    matrix: np.ndarray) -> float:
        """Compute mean reconstruction error across all vectors."""
        errors = []
        for i in range(source_vectors.shape[0]):
            transferred = source_vectors[i] @ matrix.T
            error = np.linalg.norm(transferred - target_vectors[i])
            errors.append(error)
        return float(np.mean(errors))

class CrossModelTransferAnalyzer:
    """Analyzes transfer matrices across multiple model pairs."""
    
    def __init__(self, collection: VectorCollection, methods: List[str] = None):
        self.collection = collection
        self.methods = methods or ["pseudoinverse", "procrustes"]
        self.transfer_matrices: Dict[Tuple[str, str, str], TransferMatrix] = {}  # (source, target, method)
        self.computers = {method: TransferMatrixComputer(method) for method in self.methods}
    
    def compute_all_transfer_matrices(self, trait_names: List[str] = None) -> Dict[Tuple[str, str, str], TransferMatrix]:
        """Compute transfer matrices for all model pairs and methods."""
        
        models = list(self.collection.index.keys())
        if len(models) < 2:
            raise ValueError("Need at least 2 models for transfer analysis")
        
        # Get available traits across all models
        if trait_names is None:
            all_traits = set()
            for model_traits in self.collection.index.values():
                all_traits.update(model_traits.keys())
            trait_names = list(all_traits)
        
        logger.info(f"Computing transfer matrices for {len(models)} models and {len(trait_names)} traits")
        
        # Compute matrices for all ordered pairs
        for i, source_model in enumerate(models):
            for j, target_model in enumerate(models):
                if i == j:
                    continue  # Skip self-transfer
                
                logger.info(f"Processing pair: {source_model} -> {target_model}")
                
                # Get vector matrices for both models
                source_matrix, source_traits = self.collection.get_vector_matrix(source_model, trait_names)
                target_matrix, target_traits = self.collection.get_vector_matrix(target_model, trait_names)
                
                # Find common traits
                common_traits = list(set(source_traits) & set(target_traits))
                if len(common_traits) < 3:
                    logger.warning(f"Only {len(common_traits)} common traits between {source_model} and {target_model}")
                    continue
                
                # Extract vectors for common traits
                source_indices = [source_traits.index(trait) for trait in common_traits]
                target_indices = [target_traits.index(trait) for trait in common_traits]
                
                source_vectors = source_matrix[source_indices]
                target_vectors = target_matrix[target_indices]
                
                # Compute transfer matrix for each method
                for method in self.methods:
                    try:
                        transfer_matrix = self.computers[method].compute_transfer_matrix(
                            source_vectors, target_vectors, source_model, target_model, common_traits
                        )
                        
                        key = (source_model, target_model, method)
                        self.transfer_matrices[key] = transfer_matrix
                        
                        logger.info(f"  {method}: error={transfer_matrix.reconstruction_error:.4f}, "
                                  f"rank={transfer_matrix.rank}, cond={transfer_matrix.condition_number:.2e}")
                        
                    except Exception as e:
                        logger.error(f"Failed to compute {method} matrix for {source_model}->{target_model}: {e}")
        
        return self.transfer_matrices
    
    def analyze_transfer_quality(self) -> Dict[str, Any]:
        """Analyze quality metrics across all transfer matrices."""
        
        if not self.transfer_matrices:
            logger.warning("No transfer matrices computed yet")
            return {}
        
        analysis = {
            "method_comparison": defaultdict(list),
            "model_pair_analysis": defaultdict(dict),
            "trait_transfer_success": defaultdict(list),
            "geometric_properties": defaultdict(list)
        }
        
        # Group by method and model pairs
        for (source, target, method), matrix in self.transfer_matrices.items():
            pair = f"{source}->{target}"
            
            # Method comparison
            analysis["method_comparison"][method].append({
                "pair": pair,
                "reconstruction_error": matrix.reconstruction_error,
                "condition_number": matrix.condition_number,
                "rank": matrix.rank,
                "orthogonality_score": matrix.orthogonality_score
            })
            
            # Model pair analysis
            if pair not in analysis["model_pair_analysis"]:
                analysis["model_pair_analysis"][pair] = {}
            analysis["model_pair_analysis"][pair][method] = {
                "error": matrix.reconstruction_error,
                "rank": matrix.rank,
                "is_orthogonal": matrix.is_orthogonal
            }
            
            # Geometric properties
            analysis["geometric_properties"]["singular_values"].append(matrix.singular_values)
            analysis["geometric_properties"]["condition_numbers"].append(matrix.condition_number)
            analysis["geometric_properties"]["ranks"].append(matrix.rank)
        
        # Compute summary statistics (avoid modifying dict during iteration)
        method_items = list(analysis["method_comparison"].items())
        for method, data in method_items:
            errors = [d["reconstruction_error"] for d in data]
            analysis["method_comparison"][f"{method}_summary"] = {
                "mean_error": np.mean(errors),
                "std_error": np.std(errors),
                "min_error": np.min(errors),
                "max_error": np.max(errors)
            }
        
        return dict(analysis)
    
    def save_all_matrices(self, save_dir: Union[str, Path]):
        """Save all computed transfer matrices to files."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for (source, target, method), matrix in self.transfer_matrices.items():
            filename = f"{source.replace('/', '_')}_{target.replace('/', '_')}_{method}.json"
            filepath = save_dir / filename
            matrix.save(filepath)
            
        logger.info(f"Saved {len(self.transfer_matrices)} transfer matrices to {save_dir}")
    
    def evaluate_transfer_on_new_vectors(self, test_vectors: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Evaluate transfer matrices on held-out test vectors.
        
        Args:
            test_vectors: Dict[model_name][trait_name] -> vector
        """
        results = defaultdict(list)
        
        for (source, target, method), matrix in self.transfer_matrices.items():
            if source not in test_vectors or target not in test_vectors:
                continue
            
            # Find common traits in test set
            source_test_traits = set(test_vectors[source].keys())
            target_test_traits = set(test_vectors[target].keys())
            common_test_traits = source_test_traits & target_test_traits & set(matrix.traits_used)
            
            for trait in common_test_traits:
                source_vector = test_vectors[source][trait]
                target_vector = test_vectors[target][trait]
                
                transfer_error = matrix.compute_transfer_error(source_vector, target_vector)
                
                results[f"{source}->{target}_{method}"].append({
                    "trait": trait,
                    "error": transfer_error
                })
        
        return dict(results)

def analyze_procrustes_properties(transfer_matrices: Dict[Tuple[str, str, str], TransferMatrix]) -> Dict[str, Any]:
    """Analyze Procrustes-specific properties of transfer matrices."""
    
    procrustes_matrices = {key: matrix for key, matrix in transfer_matrices.items() if key[2] == "procrustes"}
    
    if not procrustes_matrices:
        return {}
    
    analysis = {
        "rotation_angles": [],
        "reflection_components": [],
        "orthogonality_scores": [],
        "comparison_with_general_linear": {}
    }
    
    for (source, target, _), matrix in procrustes_matrices.items():
        # Compute rotation angles from singular values
        if matrix.singular_values is not None:
            # For orthogonal matrices, singular values should all be 1
            analysis["rotation_angles"].append(matrix.singular_values)
        
        analysis["orthogonality_scores"].append(matrix.orthogonality_score)
        
        # Check if there's a corresponding pseudoinverse matrix for comparison
        pseudo_key = (source, target, "pseudoinverse")
        if pseudo_key in transfer_matrices:
            pseudo_matrix = transfer_matrices[pseudo_key]
            
            analysis["comparison_with_general_linear"][f"{source}->{target}"] = {
                "procrustes_error": matrix.reconstruction_error,
                "pseudoinverse_error": pseudo_matrix.reconstruction_error,
                "error_ratio": matrix.reconstruction_error / pseudo_matrix.reconstruction_error,
                "procrustes_orthogonal": matrix.is_orthogonal,
                "pseudoinverse_orthogonal": pseudo_matrix.is_orthogonal
            }
    
    return analysis