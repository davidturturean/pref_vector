"""
Platonic Hypothesis Analysis Framework

This module implements the core mathematical framework for testing whether 
large language models share a universal latent subspace for stylistic concepts,
extending the Platonic Representation Hypothesis to style transfer.

Key components:
- Subspace geometry analysis (SVD, principal angles, Grassmann distance)
- CKA similarity analysis
- Union basis construction
- Incremental trait analysis (Δ-MSE diagnostics)
"""

import numpy as np

# Try to import scipy, fall back to numpy-only implementations if needed
try:
    from scipy.linalg import subspace_angles, svd
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
    # Numpy-only fallbacks
    def subspace_angles(A, B):
        # Simplified subspace angle computation using SVD
        Q1, _ = np.linalg.qr(A)
        Q2, _ = np.linalg.qr(B)
        U, s, Vt = np.linalg.svd(Q1.T @ Q2)
        # Clamp to avoid numerical issues
        s = np.clip(s, 0, 1)
        return np.arccos(s)
    
    def svd(a, **kwargs):
        return np.linalg.svd(a, full_matrices=False)
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
from collections import defaultdict
import itertools

from .vector_extraction import VectorCollection

logger = logging.getLogger(__name__)

@dataclass
class SubspaceAnalysis:
    """Results from subspace geometry analysis between two models."""
    model_a: str
    model_b: str
    principal_angles: np.ndarray
    grassmann_distance: float
    mean_alignment: float
    cka_similarity: float
    subspace_dim_a: int
    subspace_dim_b: int
    variance_explained_a: float
    variance_explained_b: float
    basis_a: Optional[np.ndarray] = None
    basis_b: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        data['principal_angles'] = self.principal_angles.tolist() if self.principal_angles is not None else None
        data['basis_a'] = self.basis_a.tolist() if self.basis_a is not None else None
        data['basis_b'] = self.basis_b.tolist() if self.basis_b is not None else None
        return data

@dataclass
class UnionBasisAnalysis:
    """Results from union basis construction across all models."""
    models: List[str]
    union_basis: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    optimal_rank: int
    projection_residuals: Dict[str, Dict[str, float]]  # model -> trait -> residual
    residual_rankings: List[Tuple[str, str, float]]  # (model, trait, residual)
    total_variance_explained: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['union_basis'] = self.union_basis.tolist() if self.union_basis is not None else None
        data['explained_variance_ratio'] = self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None
        data['cumulative_variance'] = self.cumulative_variance.tolist() if self.cumulative_variance is not None else None
        return data

@dataclass
class IncrementalTraitAnalysis:
    """Results from incremental trait addition analysis."""
    base_traits: List[str]
    trait_contributions: List[Tuple[str, float]]  # (trait_name, delta_mse)
    mse_progression: List[float]
    optimal_trait_set: List[str]
    easy_traits: List[str]
    hard_traits: List[str]
    outlier_traits: List[str]
    improvement_threshold: float

class PlatonicAnalyzer:
    """Main class for Platonic Hypothesis analysis."""
    
    def __init__(self, collection: VectorCollection, variance_threshold: float = 0.95):
        self.collection = collection
        self.variance_threshold = variance_threshold
        self.subspace_analyses: Dict[Tuple[str, str], SubspaceAnalysis] = {}
        self.union_basis_analysis: Optional[UnionBasisAnalysis] = None
        self.incremental_analysis: Optional[IncrementalTraitAnalysis] = None
    
    def compute_subspace_geometry_analysis(self, trait_names: List[str] = None) -> Dict[Tuple[str, str], SubspaceAnalysis]:
        """
        Compute subspace geometry analysis for all model pairs.
        
        This implements the core mathematical analysis:
        1. SVD of each model's vector matrix: V^(M) = U_M Σ_M B_M^T
        2. Extract orthonormal basis B_M (retain variance_threshold variance)
        3. Compute principal angles between subspaces
        4. Calculate Grassmann distance and CKA similarity
        """
        models = list(self.collection.index.keys())
        if len(models) < 2:
            raise ValueError("Need at least 2 models for subspace analysis")
        
        logger.info(f"Computing subspace geometry analysis for {len(models)} models")
        
        # Get vector matrices for all models
        model_matrices = {}
        model_bases = {}
        
        for model in models:
            matrix, traits = self.collection.get_vector_matrix(model, trait_names)
            if matrix.size == 0:
                logger.warning(f"No vectors found for model {model}")
                continue
            
            # Compute SVD and extract basis
            U, s, Vt = svd(matrix, full_matrices=False)
            
            # Determine rank based on variance threshold
            total_variance = np.sum(s**2)
            cumulative_variance = np.cumsum(s**2) / total_variance
            rank = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            rank = min(rank, len(s))  # Don't exceed available dimensions
            
            # Extract orthonormal basis
            basis = Vt[:rank, :].T  # (hidden_dim, rank)
            
            model_matrices[model] = matrix
            model_bases[model] = {
                'basis': basis,
                'singular_values': s,
                'rank': rank,
                'variance_explained': cumulative_variance[rank-1] if rank > 0 else 0.0
            }
            
            logger.info(f"  {model}: rank={rank}, variance_explained={cumulative_variance[rank-1]:.3f}")
        
        # Compute pairwise subspace analyses
        for model_a, model_b in itertools.combinations(models, 2):
            if model_a not in model_bases or model_b not in model_bases:
                continue
            
            analysis = self._compute_pairwise_subspace_analysis(
                model_a, model_b, model_matrices, model_bases
            )
            
            self.subspace_analyses[(model_a, model_b)] = analysis
            
            logger.info(f"  {model_a} <-> {model_b}: "
                       f"Grassmann_dist={analysis.grassmann_distance:.4f}, "
                       f"CKA={analysis.cka_similarity:.4f}, "
                       f"mean_alignment={analysis.mean_alignment:.4f}")
        
        return self.subspace_analyses
    
    def _compute_pairwise_subspace_analysis(self, model_a: str, model_b: str,
                                          model_matrices: Dict[str, np.ndarray],
                                          model_bases: Dict[str, Dict]) -> SubspaceAnalysis:
        """Compute detailed subspace analysis between two models."""
        
        basis_a = model_bases[model_a]['basis']
        basis_b = model_bases[model_b]['basis']
        
        # Compute principal angles
        try:
            angles = subspace_angles(basis_a, basis_b)
        except Exception as e:
            logger.warning(f"Failed to compute subspace angles between {model_a} and {model_b}: {e}")
            angles = np.array([])
        
        # Compute Grassmann distance
        grassmann_distance = np.sum(angles**2) if len(angles) > 0 else np.inf
        
        # Compute mean alignment (cosine of angles)
        mean_alignment = np.mean(np.cos(angles)**2) if len(angles) > 0 else 0.0
        
        # Compute CKA similarity
        cka_similarity = self._compute_cka_similarity(
            model_matrices[model_a], model_matrices[model_b]
        )
        
        return SubspaceAnalysis(
            model_a=model_a,
            model_b=model_b,
            principal_angles=angles,
            grassmann_distance=grassmann_distance,
            mean_alignment=mean_alignment,
            cka_similarity=cka_similarity,
            subspace_dim_a=model_bases[model_a]['rank'],
            subspace_dim_b=model_bases[model_b]['rank'],
            variance_explained_a=model_bases[model_a]['variance_explained'],
            variance_explained_b=model_bases[model_b]['variance_explained'],
            basis_a=basis_a,
            basis_b=basis_b
        )
    
    def _compute_cka_similarity(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
        """
        Compute Centered Kernel Alignment (CKA) similarity.
        
        CKA(A,B) = ⟨G_A, G_B⟩_F / (||G_A||_F ||G_B||_F)
        where G_M = V^(M) (V^(M))^T is the Gram matrix
        """
        def center_gram_matrix(gram):
            """Center a Gram matrix."""
            n = gram.shape[0]
            ones = np.ones((n, n)) / n
            return gram - ones @ gram - gram @ ones + ones @ gram @ ones
        
        # Compute Gram matrices
        gram_a = matrix_a @ matrix_a.T
        gram_b = matrix_b @ matrix_b.T
        
        # Center the Gram matrices
        gram_a_centered = center_gram_matrix(gram_a)
        gram_b_centered = center_gram_matrix(gram_b)
        
        # Compute CKA
        numerator = np.trace(gram_a_centered @ gram_b_centered)
        denom_a = np.trace(gram_a_centered @ gram_a_centered)
        denom_b = np.trace(gram_b_centered @ gram_b_centered)
        
        if denom_a == 0 or denom_b == 0:
            return 0.0
        
        return numerator / np.sqrt(denom_a * denom_b)
    
    def construct_union_basis(self, trait_names: List[str] = None,
                            union_variance_threshold: float = 0.90) -> UnionBasisAnalysis:
        """
        Construct union basis across all models using concatenated SVD.
        
        Algorithm:
        1. Stack all model vectors: V_all = [V^(M1); V^(M2); ...; V^(Mq)]
        2. Compute SVD: V_all = U Σ B^T
        3. Extract union basis: B_union = B[:, 1:r*] (90% explained variance)
        4. Expected: r* << q×n if Platonic hypothesis holds
        """
        models = list(self.collection.index.keys())
        logger.info(f"Constructing union basis from {len(models)} models")
        
        # Collect all vectors
        all_vectors = []
        vector_metadata = []  # (model, trait) for each vector
        
        for model in models:
            matrix, traits = self.collection.get_vector_matrix(model, trait_names)
            if matrix.size == 0:
                continue
            
            for i, trait in enumerate(traits):
                all_vectors.append(matrix[i])
                vector_metadata.append((model, trait))
        
        if not all_vectors:
            raise ValueError("No vectors found for union basis construction")
        
        # Stack all vectors
        V_all = np.stack(all_vectors)  # (total_vectors, hidden_dim)
        logger.info(f"Stacked {V_all.shape[0]} vectors of dimension {V_all.shape[1]}")
        
        # Compute SVD
        U, s, Vt = svd(V_all, full_matrices=False)
        
        # Determine optimal rank based on variance threshold
        total_variance = np.sum(s**2)
        explained_variance_ratio = s**2 / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        optimal_rank = np.argmax(cumulative_variance >= union_variance_threshold) + 1
        optimal_rank = min(optimal_rank, len(s))
        
        # Extract union basis
        union_basis = Vt[:optimal_rank, :].T  # (hidden_dim, optimal_rank)
        
        logger.info(f"Union basis rank: {optimal_rank}, variance explained: {cumulative_variance[optimal_rank-1]:.3f}")
        
        # Compute projection residuals for each vector
        projection_residuals = defaultdict(dict)
        residual_rankings = []
        
        for i, (model, trait) in enumerate(vector_metadata):
            vector = all_vectors[i]
            
            # Project onto union basis and compute residual
            projection = union_basis @ (union_basis.T @ vector)
            residual = np.linalg.norm(vector - projection)
            
            projection_residuals[model][trait] = residual
            residual_rankings.append((model, trait, residual))
        
        # Sort by residual (highest first - these are outliers)
        residual_rankings.sort(key=lambda x: x[2], reverse=True)
        
        self.union_basis_analysis = UnionBasisAnalysis(
            models=models,
            union_basis=union_basis,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance=cumulative_variance,
            optimal_rank=optimal_rank,
            projection_residuals=dict(projection_residuals),
            residual_rankings=residual_rankings,
            total_variance_explained=cumulative_variance[optimal_rank-1]
        )
        
        return self.union_basis_analysis
    
    def incremental_trait_analysis(self, base_traits: List[str] = None,
                                 improvement_threshold: float = 0.01) -> IncrementalTraitAnalysis:
        """
        Perform incremental trait analysis to identify which traits fit the universal subspace model.
        
        Algorithm:
        1. Start with core trait set (verbosity, formality, technical_complexity)
        2. For each remaining trait, compute Δ-MSE when adding to current set
        3. Greedily add traits that improve (decrease) MSE
        4. Rank traits by their contribution to universality
        """
        if base_traits is None:
            base_traits = ["verbosity", "formality", "technical_complexity"]
        
        # Get all available traits
        all_traits = set()
        for model_traits in self.collection.index.values():
            all_traits.update(model_traits.keys())
        all_traits = list(all_traits)
        
        remaining_traits = [t for t in all_traits if t not in base_traits]
        
        logger.info(f"Incremental analysis: starting with {base_traits}, testing {len(remaining_traits)} additional traits")
        
        current_traits = base_traits.copy()
        trait_contributions = []
        mse_progression = []
        
        # Compute initial MSE with base traits
        base_mse = self._compute_transfer_mse(current_traits)
        mse_progression.append(base_mse)
        
        logger.info(f"Base MSE with {base_traits}: {base_mse:.4f}")
        
        # Test each remaining trait
        for trait in remaining_traits:
            if trait not in all_traits:
                continue
            
            # Compute MSE with current traits + new trait
            test_traits = current_traits + [trait]
            test_mse = self._compute_transfer_mse(test_traits)
            
            delta_mse = test_mse - mse_progression[-1]
            trait_contributions.append((trait, delta_mse))
            
            logger.info(f"  {trait}: Δ-MSE = {delta_mse:.4f}")
            
            # Greedily add if improvement
            if delta_mse < -improvement_threshold:
                current_traits.append(trait)
                mse_progression.append(test_mse)
                logger.info(f"    Added {trait} (improvement: {-delta_mse:.4f})")
        
        # Sort traits by contribution (most helpful first)
        trait_contributions.sort(key=lambda x: x[1])
        
        # Categorize traits
        easy_traits = [trait for trait, delta in trait_contributions if delta < -improvement_threshold]
        hard_traits = [trait for trait, delta in trait_contributions if -improvement_threshold <= delta <= improvement_threshold]
        outlier_traits = [trait for trait, delta in trait_contributions if delta > improvement_threshold]
        
        self.incremental_analysis = IncrementalTraitAnalysis(
            base_traits=base_traits,
            trait_contributions=trait_contributions,
            mse_progression=mse_progression,
            optimal_trait_set=current_traits,
            easy_traits=easy_traits,
            hard_traits=hard_traits,
            outlier_traits=outlier_traits,
            improvement_threshold=improvement_threshold
        )
        
        logger.info(f"Final trait set: {len(current_traits)} traits")
        logger.info(f"Easy traits: {easy_traits}")
        logger.info(f"Hard traits: {hard_traits}")
        logger.info(f"Outlier traits: {outlier_traits}")
        
        return self.incremental_analysis
    
    def _compute_transfer_mse(self, trait_names: List[str]) -> float:
        """
        Compute mean transfer MSE across all model pairs for given traits.
        
        This is a simplified version that uses Procrustes alignment.
        """
        from .transfer_matrices import TransferMatrixComputer
        
        models = list(self.collection.index.keys())
        if len(models) < 2:
            return np.inf
        
        computer = TransferMatrixComputer(method="procrustes")
        errors = []
        
        for i, source_model in enumerate(models):
            for j, target_model in enumerate(models):
                if i >= j:
                    continue
                
                try:
                    # Get vectors for both models
                    source_matrix, source_traits = self.collection.get_vector_matrix(source_model, trait_names)
                    target_matrix, target_traits = self.collection.get_vector_matrix(target_model, trait_names)
                    
                    # Find common traits
                    common_traits = list(set(source_traits) & set(target_traits))
                    if len(common_traits) < 2:
                        continue
                    
                    # Extract vectors for common traits
                    source_indices = [source_traits.index(trait) for trait in common_traits]
                    target_indices = [target_traits.index(trait) for trait in common_traits]
                    
                    source_vectors = source_matrix[source_indices]
                    target_vectors = target_matrix[target_indices]
                    
                    # Compute transfer matrix and error
                    transfer_matrix = computer.compute_transfer_matrix(
                        source_vectors, target_vectors, source_model, target_model, common_traits
                    )
                    
                    errors.append(transfer_matrix.reconstruction_error)
                    
                except Exception as e:
                    logger.debug(f"Failed to compute transfer MSE for {source_model}->{target_model}: {e}")
                    continue
        
        return np.mean(errors) if errors else np.inf
    
    def compute_universality_scores(self) -> Dict[str, float]:
        """
        Compute universality scores for each trait based on multiple criteria:
        1. Low Δ-MSE (helps with transfer)
        2. Low projection residual (fits in union subspace)
        3. Good clustering behavior (same trait clusters across models)
        """
        if not self.union_basis_analysis or not self.incremental_analysis:
            raise ValueError("Must run union basis and incremental analysis first")
        
        universality_scores = {}
        
        # Get all traits
        all_traits = set()
        for model_traits in self.collection.index.values():
            all_traits.update(model_traits.keys())
        
        for trait in all_traits:
            score = 0.0
            
            # Criterion 1: Δ-MSE contribution (lower is better)
            delta_mse_map = dict(self.incremental_analysis.trait_contributions)
            if trait in delta_mse_map:
                delta_mse = delta_mse_map[trait]
                # Convert to score (0-1, higher is better)
                max_delta = max(delta for _, delta in self.incremental_analysis.trait_contributions)
                min_delta = min(delta for _, delta in self.incremental_analysis.trait_contributions)
                if max_delta > min_delta:
                    delta_score = 1.0 - (delta_mse - min_delta) / (max_delta - min_delta)
                else:
                    delta_score = 0.5
                score += 0.4 * delta_score
            
            # Criterion 2: Projection residual (lower is better)
            residuals = []
            for model, trait_residuals in self.union_basis_analysis.projection_residuals.items():
                if trait in trait_residuals:
                    residuals.append(trait_residuals[trait])
            
            if residuals:
                mean_residual = np.mean(residuals)
                # Normalize by maximum residual across all traits
                max_residual = max(
                    residual for residuals in self.union_basis_analysis.projection_residuals.values()
                    for residual in residuals.values()
                )
                residual_score = 1.0 - (mean_residual / max_residual) if max_residual > 0 else 0.5
                score += 0.4 * residual_score
            
            # Criterion 3: Availability across models (more is better)
            models_with_trait = sum(
                1 for model_traits in self.collection.index.values()
                if trait in model_traits
            )
            total_models = len(self.collection.index)
            availability_score = models_with_trait / total_models if total_models > 0 else 0.0
            score += 0.2 * availability_score
            
            universality_scores[trait] = score
        
        return universality_scores
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary of Platonic analysis results."""
        
        report = {
            "overview": {
                "num_models": len(self.collection.index),
                "total_vectors": len(self.collection.vectors),
                "variance_threshold": self.variance_threshold
            },
            "subspace_analysis": {},
            "union_basis": {},
            "incremental_traits": {},
            "universality_scores": {},
            "platonic_hypothesis_assessment": {}
        }
        
        # Subspace analysis summary
        if self.subspace_analyses:
            grassmann_distances = [analysis.grassmann_distance for analysis in self.subspace_analyses.values()]
            cka_similarities = [analysis.cka_similarity for analysis in self.subspace_analyses.values()]
            mean_alignments = [analysis.mean_alignment for analysis in self.subspace_analyses.values()]
            
            report["subspace_analysis"] = {
                "num_pairs": len(self.subspace_analyses),
                "mean_grassmann_distance": np.mean(grassmann_distances),
                "mean_cka_similarity": np.mean(cka_similarities),
                "mean_alignment": np.mean(mean_alignments),
                "high_alignment_pairs": sum(1 for cka in cka_similarities if cka > 0.7)
            }
        
        # Union basis summary
        if self.union_basis_analysis:
            report["union_basis"] = {
                "optimal_rank": self.union_basis_analysis.optimal_rank,
                "variance_explained": self.union_basis_analysis.total_variance_explained,
                "top_outlier_traits": [
                    (trait, residual) for _, trait, residual in self.union_basis_analysis.residual_rankings[:5]
                ]
            }
        
        # Incremental analysis summary
        if self.incremental_analysis:
            report["incremental_traits"] = {
                "base_traits": self.incremental_analysis.base_traits,
                "optimal_trait_set_size": len(self.incremental_analysis.optimal_trait_set),
                "easy_traits": self.incremental_analysis.easy_traits,
                "hard_traits": self.incremental_analysis.hard_traits,
                "outlier_traits": self.incremental_analysis.outlier_traits
            }
        
        # Universality scores
        try:
            universality_scores = self.compute_universality_scores()
            sorted_scores = sorted(universality_scores.items(), key=lambda x: x[1], reverse=True)
            report["universality_scores"] = {
                "most_universal": sorted_scores[:5],
                "least_universal": sorted_scores[-5:],
                "mean_score": np.mean(list(universality_scores.values()))
            }
        except Exception as e:
            logger.warning(f"Failed to compute universality scores: {e}")
            report["universality_scores"] = {}
        
        # Platonic hypothesis assessment
        if self.subspace_analyses and self.union_basis_analysis:
            # Evidence for Platonic hypothesis:
            # 1. High CKA similarities (> 0.7)
            # 2. Low union basis rank relative to total dimensions
            # 3. High variance explained by union basis
            
            mean_cka = np.mean([a.cka_similarity for a in self.subspace_analyses.values()])
            high_cka_ratio = sum(1 for a in self.subspace_analyses.values() if a.cka_similarity > 0.7) / len(self.subspace_analyses)
            
            union_rank_ratio = self.union_basis_analysis.optimal_rank / self.union_basis_analysis.union_basis.shape[0]
            
            evidence_strength = 0.0
            if mean_cka > 0.7:
                evidence_strength += 0.4
            if high_cka_ratio > 0.5:
                evidence_strength += 0.3
            if self.union_basis_analysis.total_variance_explained > 0.85:
                evidence_strength += 0.2
            if union_rank_ratio < 0.1:  # Low-dimensional shared subspace
                evidence_strength += 0.1
            
            assessment = "Strong" if evidence_strength > 0.8 else "Moderate" if evidence_strength > 0.5 else "Weak"
            
            report["platonic_hypothesis_assessment"] = {
                "evidence_strength": evidence_strength,
                "assessment": assessment,
                "mean_cka_similarity": mean_cka,
                "high_alignment_ratio": high_cka_ratio,
                "union_rank_ratio": union_rank_ratio,
                "union_variance_explained": self.union_basis_analysis.total_variance_explained
            }
        
        return report