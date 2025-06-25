"""
Clustering and Semantic Analysis for Cross-Model Style Vectors

This module implements clustering analysis to test if same traits cluster together
across different models, and semantic factor analysis to discover underlying
structure in the style space.
"""

import numpy as np
import pandas as pd

# Try scipy imports with fallbacks
try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    SCIPY_CLUSTERING_AVAILABLE = True
except ImportError:
    SCIPY_CLUSTERING_AVAILABLE = False

# Try sklearn imports with fallbacks
try:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from .vector_extraction import VectorCollection
from .transfer_matrices import TransferMatrixComputer

logger = logging.getLogger(__name__)

@dataclass
class ClusteringResults:
    """Results from hierarchical clustering analysis."""
    method: str
    linkage_method: str
    similarity_matrix: np.ndarray
    cluster_labels: np.ndarray
    trait_labels: List[str]
    model_labels: List[str]
    silhouette_scores: Dict[str, float]  # trait-based vs model-based clustering
    purity_scores: Dict[str, float]
    dendrogram_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['similarity_matrix'] = self.similarity_matrix.tolist() if self.similarity_matrix is not None else None
        data['cluster_labels'] = self.cluster_labels.tolist() if self.cluster_labels is not None else None
        return data

@dataclass
class FactorAnalysisResults:
    """Results from semantic factor analysis."""
    n_factors: int
    factor_loadings: np.ndarray
    factor_names: List[str]
    explained_variance_ratio: np.ndarray
    trait_factor_assignments: Dict[str, int]
    cross_model_consistency: float
    factor_interpretations: Dict[str, List[str]]  # factor -> top traits
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['factor_loadings'] = self.factor_loadings.tolist() if self.factor_loadings is not None else None
        data['explained_variance_ratio'] = self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None
        return data

class CrossModelClusteringAnalyzer:
    """Analyzes clustering patterns across multiple models."""
    
    def __init__(self, collection: VectorCollection):
        self.collection = collection
        self.clustering_results: Dict[str, ClusteringResults] = {}
        self.factor_results: Optional[FactorAnalysisResults] = None
    
    def compute_cross_model_similarity_matrix(self, trait_names: List[str] = None,
                                            alignment_method: str = "procrustes") -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute similarity matrix for all model-trait pairs.
        
        Algorithm:
        1. Apply alignment transformations between models
        2. Compute pairwise similarities: s((M,t), (M',t')) = ⟨v_{M,t}, v_{M',t'}⟩
        3. Create (q×n) × (q×n) similarity matrix for all model-trait pairs
        """
        models = list(self.collection.index.keys())
        if len(models) < 2:
            raise ValueError("Need at least 2 models for clustering analysis")
        
        # Get all available traits if not specified
        if trait_names is None:
            all_traits = set()
            for model_traits in self.collection.index.values():
                all_traits.update(model_traits.keys())
            trait_names = list(all_traits)
        
        # Collect all model-trait pairs and their vectors
        model_trait_pairs = []
        vectors = []
        
        for model in models:
            matrix, traits = self.collection.get_vector_matrix(model, trait_names)
            if matrix.size == 0:
                continue
                
            for i, trait in enumerate(traits):
                model_trait_pairs.append((model, trait))
                vectors.append(matrix[i])
        
        if len(vectors) < 2:
            raise ValueError("Need at least 2 vectors for similarity analysis")
        
        logger.info(f"Computing similarity matrix for {len(model_trait_pairs)} model-trait pairs")
        
        # Stack all vectors
        vector_matrix = np.stack(vectors)
        
        # Apply alignment if specified
        if alignment_method == "procrustes":
            aligned_vectors = self._apply_procrustes_alignment(vector_matrix, model_trait_pairs, models)
        else:
            aligned_vectors = vector_matrix
        
        # Compute cosine similarity matrix
        # Normalize vectors
        norms = np.linalg.norm(aligned_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vectors = aligned_vectors / norms
        
        # Compute similarity matrix
        similarity_matrix = normalized_vectors @ normalized_vectors.T
        
        # Extract labels
        trait_labels = [pair[1] for pair in model_trait_pairs]
        model_labels = [pair[0] for pair in model_trait_pairs]
        
        return similarity_matrix, trait_labels, model_labels
    
    def _apply_procrustes_alignment(self, vectors: np.ndarray, pairs: List[Tuple[str, str]], 
                                  models: List[str]) -> np.ndarray:
        """Apply Procrustes alignment between models before computing similarities."""
        
        # Group vectors by model
        model_vectors = defaultdict(list)
        model_indices = defaultdict(list)
        
        for i, (model, trait) in enumerate(pairs):
            model_vectors[model].append(vectors[i])
            model_indices[model].append(i)
        
        # Convert to matrices
        for model in model_vectors:
            model_vectors[model] = np.stack(model_vectors[model])
        
        # Use first model as reference
        reference_model = models[0]
        aligned_vectors = vectors.copy()
        
        computer = TransferMatrixComputer(method="procrustes")
        
        # Align all other models to reference
        for model in models[1:]:
            if model not in model_vectors or reference_model not in model_vectors:
                continue
            
            try:
                # Find common traits between reference and current model
                ref_traits = [trait for ref_model, trait in pairs if ref_model == reference_model]
                cur_traits = [trait for cur_model, trait in pairs if cur_model == model]
                common_traits = list(set(ref_traits) & set(cur_traits))
                
                if len(common_traits) < 2:
                    continue
                
                # Get vectors for common traits
                ref_indices = [i for i, (m, t) in enumerate(pairs) if m == reference_model and t in common_traits]
                cur_indices = [i for i, (m, t) in enumerate(pairs) if m == model and t in common_traits]
                
                if len(ref_indices) != len(cur_indices):
                    continue
                
                ref_vectors = vectors[ref_indices]
                cur_vectors = vectors[cur_indices]
                
                # Compute Procrustes transformation
                transfer_matrix = computer.compute_transfer_matrix(
                    cur_vectors, ref_vectors, model, reference_model, common_traits
                )
                
                # Apply transformation to all vectors from this model
                for idx in model_indices[model]:
                    aligned_vectors[idx] = transfer_matrix.apply_transfer(vectors[idx])
                    
            except Exception as e:
                logger.warning(f"Failed to align {model} to {reference_model}: {e}")
                continue
        
        return aligned_vectors
    
    def hierarchical_clustering_analysis(self, trait_names: List[str] = None,
                                       linkage_method: str = "ward") -> ClusteringResults:
        """
        Perform hierarchical clustering analysis.
        
        Success criterion: Clusters should group by trait, not by model
        Example: {Mistral_verbosity, Gemma_verbosity, LLaMA_verbosity} form one cluster
        """
        
        similarity_matrix, trait_labels, model_labels = self.compute_cross_model_similarity_matrix(trait_names)
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
        
        # Perform hierarchical clustering
        # Convert to condensed distance matrix for scipy
        condensed_distances = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distances, method=linkage_method)
        
        # Generate dendrogram data
        dendrogram_data = dendrogram(linkage_matrix, no_plot=True, labels=trait_labels)
        
        # Compute cluster labels for different numbers of clusters
        n_clusters_range = range(2, min(len(set(trait_labels)) + 1, 20))
        best_trait_silhouette = -1
        best_n_clusters = 2
        
        silhouette_scores = {}
        purity_scores = {}
        
        for n_clusters in n_clusters_range:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Compute silhouette scores for trait-based and model-based clustering
            try:
                # Overall silhouette score
                sil_score = silhouette_score(similarity_matrix, cluster_labels)
                
                # Trait-based clustering quality
                trait_sil = self._compute_trait_clustering_quality(cluster_labels, trait_labels)
                silhouette_scores[f"n_clusters_{n_clusters}"] = {
                    "overall": sil_score,
                    "trait_based": trait_sil
                }
                
                # Purity scores
                trait_purity = self._compute_clustering_purity(cluster_labels, trait_labels)
                model_purity = self._compute_clustering_purity(cluster_labels, model_labels)
                purity_scores[f"n_clusters_{n_clusters}"] = {
                    "trait_purity": trait_purity,
                    "model_purity": model_purity
                }
                
                if trait_sil > best_trait_silhouette:
                    best_trait_silhouette = trait_sil
                    best_n_clusters = n_clusters
                    
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score for {n_clusters} clusters: {e}")
        
        # Use best clustering
        final_cluster_labels = fcluster(linkage_matrix, best_n_clusters, criterion='maxclust')
        
        results = ClusteringResults(
            method="hierarchical",
            linkage_method=linkage_method,
            similarity_matrix=similarity_matrix,
            cluster_labels=final_cluster_labels,
            trait_labels=trait_labels,
            model_labels=model_labels,
            silhouette_scores=silhouette_scores,
            purity_scores=purity_scores,
            dendrogram_data=dendrogram_data
        )
        
        self.clustering_results["hierarchical"] = results
        
        logger.info(f"Hierarchical clustering: {best_n_clusters} clusters, "
                   f"trait silhouette: {best_trait_silhouette:.3f}")
        
        return results
    
    def _compute_trait_clustering_quality(self, cluster_labels: np.ndarray, trait_labels: List[str]) -> float:
        """Compute how well traits cluster together (Anna Karenina validation)."""
        
        # For each trait, check if all instances are in the same cluster
        trait_cluster_consistency = []
        
        unique_traits = list(set(trait_labels))
        for trait in unique_traits:
            trait_indices = [i for i, t in enumerate(trait_labels) if t == trait]
            if len(trait_indices) > 1:
                # Check if all instances of this trait are in the same cluster
                trait_clusters = [cluster_labels[i] for i in trait_indices]
                consistency = len(set(trait_clusters)) == 1
                trait_cluster_consistency.append(consistency)
        
        return np.mean(trait_cluster_consistency) if trait_cluster_consistency else 0.0
    
    def _compute_clustering_purity(self, cluster_labels: np.ndarray, true_labels: List[str]) -> float:
        """Compute clustering purity score."""
        
        cluster_ids = np.unique(cluster_labels)
        total_correct = 0
        
        for cluster_id in cluster_ids:
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = [true_labels[i] for i in range(len(true_labels)) if cluster_mask[i]]
            
            if cluster_true_labels:
                # Find most common true label in this cluster
                from collections import Counter
                most_common_label = Counter(cluster_true_labels).most_common(1)[0][0]
                correct_in_cluster = sum(1 for label in cluster_true_labels if label == most_common_label)
                total_correct += correct_in_cluster
        
        return total_correct / len(true_labels)
    
    def semantic_factor_analysis(self, trait_names: List[str] = None, n_factors: int = None) -> FactorAnalysisResults:
        """
        Discover underlying semantic structure of style space using factor analysis.
        
        Expected factors:
        - "Formal-Academic" (formality + technicality + authority)
        - "Interpersonal" (politeness + empathy + humor)
        - "Linguistic-Surface" (verbosity + clarity + specificity)
        """
        
        # Collect vectors from all models for factor analysis
        all_vectors = []
        all_traits = []
        all_models = []
        
        models = list(self.collection.index.keys())
        
        if trait_names is None:
            trait_names = set()
            for model_traits in self.collection.index.values():
                trait_names.update(model_traits.keys())
            trait_names = list(trait_names)
        
        # Create a matrix where each row is a trait and each column is a model
        trait_model_matrix = {}
        
        for trait in trait_names:
            trait_vectors = []
            for model in models:
                matrix, traits = self.collection.get_vector_matrix(model, [trait])
                if matrix.size > 0:
                    trait_vectors.append(matrix[0])
                else:
                    # Fill with zeros if trait not available for this model
                    hidden_dim = 4096  # Default, should get from somewhere
                    trait_vectors.append(np.zeros(hidden_dim))
            
            if trait_vectors:
                trait_model_matrix[trait] = np.stack(trait_vectors)
        
        if not trait_model_matrix:
            raise ValueError("No trait vectors found for factor analysis")
        
        # Create feature matrix: each row is a trait, features are the union of all model representations
        trait_features = []
        trait_names_ordered = []
        
        for trait, vectors in trait_model_matrix.items():
            # Flatten all model vectors for this trait into one feature vector
            flattened = vectors.flatten()
            trait_features.append(flattened)
            trait_names_ordered.append(trait)
        
        feature_matrix = np.stack(trait_features)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_std = scaler.fit_transform(feature_matrix)
        
        # Determine number of factors if not specified
        if n_factors is None:
            # Use PCA to estimate good number of factors
            pca = PCA()
            pca.fit(feature_matrix_std)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_factors = np.argmax(cumvar >= 0.8) + 1
            n_factors = max(2, min(n_factors, 8))  # Between 2 and 8 factors
        
        logger.info(f"Performing factor analysis with {n_factors} factors on {len(trait_names_ordered)} traits")
        
        # Perform factor analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        factor_scores = fa.fit_transform(feature_matrix_std)
        factor_loadings = fa.components_.T  # (n_traits, n_factors)
        
        # Interpret factors by finding traits with highest loadings
        factor_interpretations = {}
        trait_factor_assignments = {}
        
        for factor_idx in range(n_factors):
            loadings = factor_loadings[:, factor_idx]
            # Ensure we don't take more traits than we have
            n_top_traits = min(5, len(trait_names_ordered))
            top_traits_idx = np.argsort(np.abs(loadings))[-n_top_traits:]
            # Ensure indices are within bounds
            top_traits_idx = [i for i in top_traits_idx if i < len(trait_names_ordered)]
            top_traits = [trait_names_ordered[i] for i in top_traits_idx]
            factor_interpretations[f"Factor_{factor_idx+1}"] = top_traits
        
        # Assign each trait to its dominant factor
        for i, trait in enumerate(trait_names_ordered):
            dominant_factor = np.argmax(np.abs(factor_loadings[i, :]))
            trait_factor_assignments[trait] = dominant_factor
        
        # Estimate cross-model consistency
        cross_model_consistency = self._estimate_factor_consistency(trait_model_matrix, factor_loadings)
        
        # Generate interpretable factor names
        factor_names = self._generate_factor_names(factor_interpretations)
        
        # Compute explained variance
        explained_variance_ratio = fa.noise_variance_  # This is actually noise, need to compute properly
        # For now, use a simple approximation
        total_variance = np.sum(np.var(feature_matrix_std, axis=0))
        factor_variances = np.var(factor_scores, axis=0)
        explained_variance_ratio = factor_variances / np.sum(factor_variances)
        
        results = FactorAnalysisResults(
            n_factors=n_factors,
            factor_loadings=factor_loadings,
            factor_names=factor_names,
            explained_variance_ratio=explained_variance_ratio,
            trait_factor_assignments=trait_factor_assignments,
            cross_model_consistency=cross_model_consistency,
            factor_interpretations=factor_interpretations
        )
        
        self.factor_results = results
        
        logger.info(f"Factor analysis complete:")
        for i, (factor_name, traits) in enumerate(zip(factor_names, factor_interpretations.values())):
            logger.info(f"  {factor_name}: {traits}")
        
        return results
    
    def _estimate_factor_consistency(self, trait_model_matrix: Dict[str, np.ndarray], 
                                   factor_loadings: np.ndarray) -> float:
        """Estimate how consistent factors are across different models."""
        
        # For each model, check if the same traits load on the same factors
        models = list(range(len(list(trait_model_matrix.values())[0])))
        
        if len(models) < 2:
            return 1.0
        
        consistency_scores = []
        
        # Compare factor structure between pairs of models
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                # Extract vectors for models i and j
                model_i_vectors = []
                model_j_vectors = []
                
                for trait_vectors in trait_model_matrix.values():
                    model_i_vectors.append(trait_vectors[i])
                    model_j_vectors.append(trait_vectors[j])
                
                model_i_matrix = np.stack(model_i_vectors)
                model_j_matrix = np.stack(model_j_vectors)
                
                # Compute factor analysis for each model separately
                scaler_i = StandardScaler()
                scaler_j = StandardScaler()
                
                matrix_i_std = scaler_i.fit_transform(model_i_matrix)
                matrix_j_std = scaler_j.fit_transform(model_j_matrix)
                
                try:
                    fa_i = FactorAnalysis(n_components=factor_loadings.shape[1], random_state=42)
                    fa_j = FactorAnalysis(n_components=factor_loadings.shape[1], random_state=42)
                    
                    fa_i.fit(matrix_i_std)
                    fa_j.fit(matrix_j_std)
                    
                    # Compare factor loadings using cosine similarity
                    loadings_i = fa_i.components_.T
                    loadings_j = fa_j.components_.T
                    
                    # Find best alignment between factors
                    similarity_matrix = np.abs(loadings_i.T @ loadings_j)
                    max_similarities = np.max(similarity_matrix, axis=1)
                    consistency = np.mean(max_similarities)
                    
                    consistency_scores.append(consistency)
                    
                except Exception as e:
                    logger.debug(f"Failed to compute factor consistency between models {i} and {j}: {e}")
                    continue
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _generate_factor_names(self, factor_interpretations: Dict[str, List[str]]) -> List[str]:
        """Generate interpretable names for factors based on top-loading traits."""
        
        factor_names = []
        
        for factor_id, top_traits in factor_interpretations.items():
            # Simple heuristic naming based on trait content
            if any(trait in ["formality", "technical_complexity", "authority", "register"] for trait in top_traits):
                name = "Formal-Academic"
            elif any(trait in ["politeness", "empathy", "humor", "emotional_tone"] for trait in top_traits):
                name = "Interpersonal"
            elif any(trait in ["verbosity", "clarity", "specificity", "concreteness"] for trait in top_traits):
                name = "Linguistic-Surface"
            elif any(trait in ["assertiveness", "certainty", "authority", "persuasiveness"] for trait in top_traits):
                name = "Assertive-Confident"
            elif any(trait in ["creativity", "humor", "optimism", "enthusiasm"] for trait in top_traits):
                name = "Creative-Positive"
            else:
                name = f"Factor_{len(factor_names)+1}"
            
            factor_names.append(name)
        
        return factor_names
    
    def analyze_outlier_traits(self) -> Dict[str, Any]:
        """
        Characterize traits that don't fit well in the universal subspace model.
        
        Combines evidence from:
        - High Δ-MSE (from incremental analysis)
        - High projection residuals (from union basis analysis)
        - Poor clustering behavior
        """
        
        outlier_analysis = {
            "clustering_outliers": [],
            "semantic_outliers": [],
            "universal_outliers": [],
            "outlier_characteristics": {}
        }
        
        # Clustering-based outliers: traits that don't cluster well across models
        if "hierarchical" in self.clustering_results:
            clustering_result = self.clustering_results["hierarchical"]
            
            # Find traits with low trait-clustering quality
            trait_qualities = {}
            unique_traits = list(set(clustering_result.trait_labels))
            
            for trait in unique_traits:
                trait_indices = [i for i, t in enumerate(clustering_result.trait_labels) if t == trait]
                if len(trait_indices) > 1:
                    trait_clusters = [clustering_result.cluster_labels[i] for i in trait_indices]
                    consistency = len(set(trait_clusters)) == 1
                    trait_qualities[trait] = consistency
            
            clustering_outliers = [trait for trait, quality in trait_qualities.items() if not quality]
            outlier_analysis["clustering_outliers"] = clustering_outliers
        
        # Semantic outliers: traits that don't load well on any factor
        if self.factor_results:
            factor_loadings = self.factor_results.factor_loadings
            trait_names = list(self.factor_results.trait_factor_assignments.keys())
            
            semantic_outliers = []
            for i, trait in enumerate(trait_names):
                max_loading = np.max(np.abs(factor_loadings[i, :]))
                if max_loading < 0.3:  # Weak loading on all factors
                    semantic_outliers.append(trait)
            
            outlier_analysis["semantic_outliers"] = semantic_outliers
        
        # Universal outliers: traits that appear in multiple outlier categories
        all_outliers = set(outlier_analysis["clustering_outliers"] + outlier_analysis["semantic_outliers"])
        universal_outliers = []
        
        for trait in all_outliers:
            outlier_count = 0
            if trait in outlier_analysis["clustering_outliers"]:
                outlier_count += 1
            if trait in outlier_analysis["semantic_outliers"]:
                outlier_count += 1
            
            if outlier_count >= 2:
                universal_outliers.append(trait)
        
        outlier_analysis["universal_outliers"] = universal_outliers
        
        # Characterize outlier properties
        outlier_characteristics = self._characterize_outlier_properties(universal_outliers)
        outlier_analysis["outlier_characteristics"] = outlier_characteristics
        
        logger.info(f"Outlier analysis:")
        logger.info(f"  Clustering outliers: {outlier_analysis['clustering_outliers']}")
        logger.info(f"  Semantic outliers: {outlier_analysis['semantic_outliers']}")
        logger.info(f"  Universal outliers: {universal_outliers}")
        
        return outlier_analysis
    
    def _characterize_outlier_properties(self, outlier_traits: List[str]) -> Dict[str, Any]:
        """Analyze common properties of outlier traits."""
        
        characteristics = {
            "linguistic_properties": {},
            "pragmatic_properties": {},
            "hypotheses": []
        }
        
        # Categorize traits by type
        surface_linguistic = ["verbosity", "clarity", "specificity", "concreteness", "register"]
        pragmatic_social = ["humor", "empathy", "politeness", "emotional_tone", "creativity"]
        cultural_specific = ["humor", "inclusivity", "assertiveness", "authority"]
        
        outlier_categories = {
            "surface_linguistic": [t for t in outlier_traits if t in surface_linguistic],
            "pragmatic_social": [t for t in outlier_traits if t in pragmatic_social],
            "cultural_specific": [t for t in outlier_traits if t in cultural_specific]
        }
        
        characteristics["linguistic_properties"] = outlier_categories
        
        # Generate hypotheses about why these traits are outliers
        hypotheses = []
        
        if len(outlier_categories["pragmatic_social"]) > len(outlier_categories["surface_linguistic"]):
            hypotheses.append("Pragmatic traits are more model-specific than surface linguistic traits")
        
        if "humor" in outlier_traits:
            hypotheses.append("Humor requires cultural/training-specific knowledge")
        
        if "authority" in outlier_traits or "assertiveness" in outlier_traits:
            hypotheses.append("Social power dynamics vary across model training")
        
        characteristics["hypotheses"] = hypotheses
        
        return characteristics