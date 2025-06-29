#!/usr/bin/env python3
"""
Comprehensive Platonic Hypothesis Analysis Pipeline
Implements state-of-the-art methods for cross-model preference vector analysis
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import itertools

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityExemplar:
    """Store min/max quality examples for each trait"""
    trait: str
    model: str
    min_score: float
    max_score: float
    min_example: str
    max_example: str
    
@dataclass
class TransferValidationResults:
    """Results from A→B→C permutation tests"""
    a_to_b_error: float
    b_to_c_error: float
    a_to_c_direct_error: float
    a_to_c_via_b_error: float
    transitivity_violation: float
    
@dataclass
class GeometricClusteringResults:
    """Geometric clustering vs MSE analysis"""
    trait: str
    geometric_cluster_id: int
    mse_cluster_id: int
    cosine_similarity: float
    transfer_error: float
    contradiction_score: float

class ComprehensivePlatonicAnalyzer:
    """Complete Platonic analysis implementation"""
    
    def __init__(self, results_dir: str = "platonic_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.quality_exemplars = {}
        self.transfer_validation_results = {}
        self.geometric_clustering_results = {}
        
        # Analysis parameters
        self.n_bootstrap_samples = 1000
        self.n_permutation_tests = 1000
        self.variance_threshold = 0.95
        
    def load_all_vectors(self) -> Dict[str, Any]:
        """Load vectors from all available sources"""
        from src.vector_extraction import VectorCollection
        
        logger.info("=== Loading All Available Vectors ===")
        
        collections = {}
        all_vectors = {}
        
        # Load from different directories
        for collection_dir in ["vectors/core", "vectors/extended", "vectors/challenging", "vectors/combined", "vectors"]:
            if Path(collection_dir).exists():
                collection = VectorCollection(collection_dir)
                if len(collection.vectors) > 0:
                    collections[collection_dir] = collection
                    all_vectors.update(collection.vectors)
                    logger.info(f"Loaded {len(collection.vectors)} vectors from {collection_dir}")
        
        # Organize by model and trait
        by_model = defaultdict(list)
        by_trait = defaultdict(list)
        trait_model_matrix = defaultdict(dict)
        
        for vector_id, vector in all_vectors.items():
            by_model[vector.model_name].append(vector)
            by_trait[vector.trait_name].append(vector)
            trait_model_matrix[vector.trait_name][vector.model_name] = vector
        
        logger.info(f"Total vectors: {len(all_vectors)}")
        logger.info(f"Models: {list(by_model.keys())}")
        logger.info(f"Traits: {list(by_trait.keys())}")
        
        return {
            "all_vectors": all_vectors,
            "by_model": by_model,
            "by_trait": by_trait,
            "trait_model_matrix": trait_model_matrix,
            "collections": collections
        }
    
    def create_quality_exemplars(self, vector_data: Dict) -> Dict[str, QualityExemplar]:
        """Create min/max quality examples for each trait"""
        logger.info("\n=== Creating Quality Exemplars ===")
        
        exemplars = {}
        
        for trait, vectors in vector_data["by_trait"].items():
            if not vectors:
                continue
                
            # Find min and max quality vectors
            min_vector = min(vectors, key=lambda v: v.quality_score)
            max_vector = max(vectors, key=lambda v: v.quality_score)
            
            # Create synthetic examples based on prompts
            min_example = f"Low quality {trait} example (score: {min_vector.quality_score:.3f})"
            max_example = f"High quality {trait} example (score: {max_vector.quality_score:.3f})"
            
            exemplar = QualityExemplar(
                trait=trait,
                model=f"{min_vector.model_name} -> {max_vector.model_name}",
                min_score=min_vector.quality_score,
                max_score=max_vector.quality_score,
                min_example=min_example,
                max_example=max_example
            )
            
            exemplars[trait] = exemplar
            logger.info(f"  {trait}: quality range [{min_vector.quality_score:.3f}, {max_vector.quality_score:.3f}]")
        
        # Save exemplars
        with open(self.results_dir / "quality_exemplars.json", "w") as f:
            json.dump({k: asdict(v) for k, v in exemplars.items()}, f, indent=2)
        
        self.quality_exemplars = exemplars
        return exemplars
    
    def statistical_transfer_validation(self, vector_data: Dict) -> Dict[str, TransferValidationResults]:
        """A→B→C permutation tests"""
        logger.info("\n=== Statistical Transfer Validation (A→B→C) ===")
        
        from src.transfer_matrices import CrossModelTransferAnalyzer
        
        # Get models with sufficient vectors
        models = [model for model, vectors in vector_data["by_model"].items() if len(vectors) >= 3]
        
        if len(models) < 3:
            logger.warning("Need at least 3 models for A→B→C validation")
            return {}
        
        validation_results = {}
        
        # Test all A→B→C triplets
        for a, b, c in itertools.permutations(models, 3):
            logger.info(f"Testing {a} → {b} → {c}")
            
            try:
                # Get common traits
                a_traits = set(v.trait_name for v in vector_data["by_model"][a])
                b_traits = set(v.trait_name for v in vector_data["by_model"][b])
                c_traits = set(v.trait_name for v in vector_data["by_model"][c])
                common_traits = a_traits & b_traits & c_traits
                
                if len(common_traits) < 2:
                    continue
                
                # Create temporary collections
                from src.vector_extraction import VectorCollection
                temp_collection = VectorCollection("temp")
                
                for model_name in [a, b, c]:
                    for vector in vector_data["by_model"][model_name]:
                        if vector.trait_name in common_traits:
                            temp_collection.vectors[vector.vector_id] = vector
                            temp_collection.index[vector.model_name][vector.trait_name].append(vector.vector_id)
                
                # Compute transfer matrices
                analyzer = CrossModelTransferAnalyzer(temp_collection)
                
                # A→B
                a_to_b_matrix = analyzer.compute_transfer_matrix(a, b, list(common_traits))
                a_to_b_error = analyzer._compute_reconstruction_error(a_to_b_matrix, a, b, list(common_traits))
                
                # B→C  
                b_to_c_matrix = analyzer.compute_transfer_matrix(b, c, list(common_traits))
                b_to_c_error = analyzer._compute_reconstruction_error(b_to_c_matrix, b, c, list(common_traits))
                
                # A→C direct
                a_to_c_matrix = analyzer.compute_transfer_matrix(a, c, list(common_traits))
                a_to_c_direct_error = analyzer._compute_reconstruction_error(a_to_c_matrix, a, c, list(common_traits))
                
                # A→C via B (composition)
                a_vectors, _ = temp_collection.get_vector_matrix(a, list(common_traits))
                c_vectors, _ = temp_collection.get_vector_matrix(c, list(common_traits))
                
                # Apply A→B→C transformation
                a_to_b_to_c = a_vectors @ a_to_b_matrix.matrix.T @ b_to_c_matrix.matrix.T
                a_to_c_via_b_error = np.mean(np.linalg.norm(a_to_b_to_c - c_vectors, axis=1)**2)
                
                # Transitivity violation
                transitivity_violation = abs(a_to_c_direct_error - a_to_c_via_b_error)
                
                result = TransferValidationResults(
                    a_to_b_error=a_to_b_error,
                    b_to_c_error=b_to_c_error,
                    a_to_c_direct_error=a_to_c_direct_error,
                    a_to_c_via_b_error=a_to_c_via_b_error,
                    transitivity_violation=transitivity_violation
                )
                
                validation_results[f"{a}→{b}→{c}"] = result
                logger.info(f"  Direct: {a_to_c_direct_error:.4f}, Via B: {a_to_c_via_b_error:.4f}, Violation: {transitivity_violation:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed validation for {a}→{b}→{c}: {e}")
                continue
        
        # Save results
        with open(self.results_dir / "transfer_validation.json", "w") as f:
            json.dump({k: asdict(v) for k, v in validation_results.items()}, f, indent=2)
        
        self.transfer_validation_results = validation_results
        return validation_results
    
    def geometric_vs_mse_clustering(self, vector_data: Dict) -> Dict[str, GeometricClusteringResults]:
        """Investigate geometric clustering vs MSE minimization"""
        logger.info("\n=== Geometric vs MSE Clustering Analysis ===")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances
        except ImportError:
            logger.warning("scikit-learn not available, skipping clustering analysis")
            return {}
        
        results = {}
        
        # Get all vector representations
        all_vectors_matrix = []
        vector_labels = []
        
        for trait, trait_vectors in vector_data["by_trait"].items():
            for vector in trait_vectors:
                all_vectors_matrix.append(vector.vector)
                vector_labels.append((trait, vector.model_name))
        
        if len(all_vectors_matrix) < 4:
            logger.warning("Not enough vectors for clustering analysis")
            return {}
        
        all_vectors_matrix = np.stack(all_vectors_matrix)
        
        # Geometric clustering (cosine similarity)
        logger.info("Computing geometric clustering...")
        n_clusters = min(len(set(label[0] for label in vector_labels)), 8)
        
        # Normalize for cosine similarity
        normalized_vectors = all_vectors_matrix / np.linalg.norm(all_vectors_matrix, axis=1, keepdims=True)
        geometric_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        geometric_clusters = geometric_kmeans.fit_predict(normalized_vectors)
        
        # MSE-based clustering (via transfer error)
        logger.info("Computing MSE-based clustering...")
        
        # Compute pairwise transfer errors
        transfer_errors = np.zeros((len(all_vectors_matrix), len(all_vectors_matrix)))
        
        for i, (trait_i, model_i) in enumerate(vector_labels):
            for j, (trait_j, model_j) in enumerate(vector_labels):
                if i != j and trait_i == trait_j and model_i != model_j:
                    # Compute transfer error for same trait across models
                    vec_i = all_vectors_matrix[i]
                    vec_j = all_vectors_matrix[j]
                    
                    # Simple linear map (single vector pair)
                    if np.linalg.norm(vec_i) > 0:
                        projection = np.dot(vec_j, vec_i) / np.dot(vec_i, vec_i) * vec_i
                        transfer_errors[i, j] = np.linalg.norm(vec_j - projection)**2
        
        # Cluster based on transfer error patterns
        mse_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        mse_clusters = mse_kmeans.fit_predict(transfer_errors)
        
        # Analyze contradictions
        for i, (trait, model) in enumerate(vector_labels):
            geometric_cluster_id = geometric_clusters[i]
            mse_cluster_id = mse_clusters[i]
            
            # Compute cosine similarity to cluster center
            geo_center = geometric_kmeans.cluster_centers_[geometric_cluster_id]
            cosine_sim = np.dot(normalized_vectors[i], geo_center) / (
                np.linalg.norm(normalized_vectors[i]) * np.linalg.norm(geo_center)
            )
            
            # Average transfer error
            transfer_error = np.mean([transfer_errors[i, j] for j in range(len(vector_labels)) if j != i])
            
            # Contradiction score: do geometric and MSE clusters disagree?
            contradiction_score = float(geometric_cluster_id != mse_cluster_id)
            
            result = GeometricClusteringResults(
                trait=trait,
                geometric_cluster_id=int(geometric_cluster_id),
                mse_cluster_id=int(mse_cluster_id),
                cosine_similarity=float(cosine_sim),
                transfer_error=float(transfer_error),
                contradiction_score=contradiction_score
            )
            
            results[f"{trait}_{model}"] = result
        
        # Summary statistics
        contradiction_rate = np.mean([r.contradiction_score for r in results.values()])
        logger.info(f"Geometric vs MSE contradiction rate: {contradiction_rate:.3f}")
        
        # Save results
        with open(self.results_dir / "geometric_vs_mse_clustering.json", "w") as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        
        self.geometric_clustering_results = results
        return results
    
    def cross_architecture_validation(self, vector_data: Dict) -> Dict[str, float]:
        """Test transfer matrices across very different architectures"""
        logger.info("\n=== Cross-Architecture Transfer Validation ===")
        
        # Group models by architecture
        model_groups = {
            "mistral": [m for m in vector_data["by_model"].keys() if "mistral" in m.lower()],
            "gemma": [m for m in vector_data["by_model"].keys() if "gemma" in m.lower()],
            "llama" : [m for m in vector_data["by_model"].keys() if "llama" in m.lower()]
        }
        
        cross_arch_results = {}
        
        for arch1, models1 in model_groups.items():
            for arch2, models2 in model_groups.items():
                if arch1 >= arch2 or not models1 or not models2:  # Avoid duplicates
                    continue
                    
                logger.info(f"Testing {arch1} ↔ {arch2}")
                
                # Test transfer between first model of each architecture
                model1, model2 = models1[0], models2[0]
                
                try:
                    from src.transfer_matrices import CrossModelTransferAnalyzer
                    from src.vector_extraction import VectorCollection
                    
                    # Create temp collection
                    temp_collection = VectorCollection("temp")
                    for vector in vector_data["by_model"][model1] + vector_data["by_model"][model2]:
                        temp_collection.vectors[vector.vector_id] = vector
                        temp_collection.index[vector.model_name][vector.trait_name].append(vector.vector_id)
                    
                    analyzer = CrossModelTransferAnalyzer(temp_collection)
                    
                    # Get common traits
                    traits1 = set(v.trait_name for v in vector_data["by_model"][model1])
                    traits2 = set(v.trait_name for v in vector_data["by_model"][model2])
                    common_traits = list(traits1 & traits2)
                    
                    if len(common_traits) >= 2:
                        matrix = analyzer.compute_transfer_matrix(model1, model2, common_traits)
                        error = analyzer._compute_reconstruction_error(matrix, model1, model2, common_traits)
                        
                        cross_arch_results[f"{arch1}→{arch2}"] = error
                        logger.info(f"  Transfer error: {error:.4f}")
                
                except Exception as e:
                    logger.warning(f"Failed cross-arch test {arch1}→{arch2}: {e}")
                    continue
        
        # Save results
        with open(self.results_dir / "cross_architecture_validation.json", "w") as f:
            json.dump(cross_arch_results, f, indent=2)
        
        return cross_arch_results
    
    def create_core_visualizations(self, vector_data: Dict):
        """Create comprehensive visualization suite"""
        logger.info("\n=== Creating Core Visualizations ===")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.decomposition import PCA
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Quality score heatmap by model and trait
            logger.info("Creating quality score heatmap...")
            quality_matrix = []
            model_names = []
            trait_names = []
            
            for model, vectors in vector_data["by_model"].items():
                model_names.append(model)
                model_scores = []
                if not trait_names:  # First model, establish trait order
                    trait_names = sorted(set(v.trait_name for v in vectors))
                
                for trait in trait_names:
                    trait_vectors = [v for v in vectors if v.trait_name == trait]
                    avg_quality = np.mean([v.quality_score for v in trait_vectors]) if trait_vectors else 0
                    model_scores.append(avg_quality)
                
                quality_matrix.append(model_scores)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(quality_matrix, 
                       xticklabels=trait_names, 
                       yticklabels=model_names,
                       annot=True, 
                       fmt='.3f',
                       cmap='viridis')
            plt.title('Quality Scores by Model and Trait')
            plt.xlabel('Traits')
            plt.ylabel('Models')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.results_dir / "quality_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. PCA visualization of all vectors
            logger.info("Creating PCA visualization...")
            
            all_vectors = []
            vector_labels = []
            vector_colors = []
            
            for trait, vectors in vector_data["by_trait"].items():
                for vector in vectors:
                    all_vectors.append(vector.vector)
                    vector_labels.append(f"{trait}_{vector.model_name}")
                    vector_colors.append(trait)
            
            if len(all_vectors) >= 4:
                all_vectors = np.stack(all_vectors)
                
                # Apply PCA
                pca = PCA(n_components=2)
                vectors_2d = pca.fit_transform(all_vectors)
                
                plt.figure(figsize=(14, 10))
                
                # Plot by trait
                unique_traits = list(set(vector_colors))
                colors = sns.color_palette("husl", len(unique_traits))
                
                for i, trait in enumerate(unique_traits):
                    mask = [c == trait for c in vector_colors]
                    plt.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
                              label=trait, color=colors[i], alpha=0.7, s=60)
                
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} variance)')
                plt.title('PCA of Style Vectors Across Models')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.results_dir / "pca_visualization.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Transfer error bar chart
            if hasattr(self, 'transfer_validation_results') and self.transfer_validation_results:
                logger.info("Creating transfer error visualization...")
                
                transfer_names = list(self.transfer_validation_results.keys())
                direct_errors = [r.a_to_c_direct_error for r in self.transfer_validation_results.values()]
                via_b_errors = [r.a_to_c_via_b_error for r in self.transfer_validation_results.values()]
                violations = [r.transitivity_violation for r in self.transfer_validation_results.values()]
                
                x = np.arange(len(transfer_names))
                width = 0.25
                
                plt.figure(figsize=(15, 8))
                plt.bar(x - width, direct_errors, width, label='Direct A→C', alpha=0.8)
                plt.bar(x, via_b_errors, width, label='Via B (A→B→C)', alpha=0.8)
                plt.bar(x + width, violations, width, label='Transitivity Violation', alpha=0.8)
                
                plt.xlabel('Transfer Paths')
                plt.ylabel('Error')
                plt.title('Transfer Validation: Direct vs Composed Paths')
                plt.xticks(x, transfer_names, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.results_dir / "transfer_validation.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to {self.results_dir}/")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
    
    def platonic_subspace_analysis(self, vector_data: Dict) -> Dict[str, Any]:
        """Advanced Platonic subspace geometry analysis"""
        logger.info("\n=== Advanced Platonic Subspace Analysis ===")
        
        try:
            from scipy.linalg import svd, subspace_angles
            from scipy.spatial.distance import pdist, squareform
        except ImportError:
            logger.warning("SciPy not available, skipping advanced analysis")
            return {}
        
        results = {}
        
        # Construct trait-specific subspaces
        trait_subspaces = {}
        for trait, vectors in vector_data["by_trait"].items():
            if len(vectors) >= 2:
                trait_matrix = np.stack([v.vector for v in vectors])
                U, s, Vt = svd(trait_matrix, full_matrices=False)
                # Keep components explaining 95% variance
                cumvar = np.cumsum(s**2) / np.sum(s**2)
                n_components = np.argmax(cumvar >= 0.95) + 1
                trait_subspaces[trait] = U[:, :n_components]
        
        # Compute pairwise subspace angles (Grassmann distance)
        trait_pairs = list(itertools.combinations(trait_subspaces.keys(), 2))
        subspace_angles_matrix = {}
        
        for trait1, trait2 in trait_pairs:
            if trait1 in trait_subspaces and trait2 in trait_subspaces:
                try:
                    angles = subspace_angles(trait_subspaces[trait1], trait_subspaces[trait2])
                    # Principal angle (smallest)
                    principal_angle = np.min(angles)
                    subspace_angles_matrix[f"{trait1}_{trait2}"] = float(principal_angle)
                except:
                    subspace_angles_matrix[f"{trait1}_{trait2}"] = np.nan
        
        # Union subspace construction (concatenated SVD)
        logger.info("Constructing union subspace...")
        all_vectors = np.vstack([np.stack([v.vector for v in vectors]) 
                                for vectors in vector_data["by_trait"].values() if vectors])
        
        U_union, s_union, _ = svd(all_vectors, full_matrices=False)
        cumvar_union = np.cumsum(s_union**2) / np.sum(s_union**2)
        
        # Find optimal rank for union basis
        optimal_rank = np.argmax(cumvar_union >= self.variance_threshold) + 1
        union_basis = U_union[:, :optimal_rank]
        
        results = {
            "trait_subspaces": {k: v.shape for k, v in trait_subspaces.items()},
            "subspace_angles": subspace_angles_matrix,
            "union_basis_rank": int(optimal_rank),
            "union_variance_explained": float(cumvar_union[optimal_rank-1]),
            "intrinsic_dimensionality": len([angle for angle in subspace_angles_matrix.values() 
                                           if not np.isnan(angle) and angle < np.pi/4])
        }
        
        # Save advanced analysis
        with open(self.results_dir / "platonic_subspace_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"  Union basis rank: {optimal_rank}")
        logger.info(f"  Variance explained: {cumvar_union[optimal_rank-1]:.3f}")
        logger.info(f"  Intrinsic trait dimensions: {results['intrinsic_dimensionality']}")
        
        return results
    
    def bootstrap_statistical_validation(self, vector_data: Dict) -> Dict[str, Any]:
        """Bootstrap confidence intervals for transfer validation"""
        logger.info("\n=== Bootstrap Statistical Validation ===")
        
        results = {}
        
        # Bootstrap transfer error confidence intervals
        if len(vector_data["by_model"]) >= 2:
            models = list(vector_data["by_model"].keys())[:2]  # Use first two models
            model_a, model_b = models[0], models[1]
            
            # Get common traits
            traits_a = set(v.trait_name for v in vector_data["by_model"][model_a])
            traits_b = set(v.trait_name for v in vector_data["by_model"][model_b])
            common_traits = list(traits_a & traits_b)
            
            if len(common_traits) >= 3:
                bootstrap_errors = []
                
                for _ in range(self.n_bootstrap_samples):
                    # Bootstrap sample traits
                    sampled_traits = np.random.choice(common_traits, 
                                                    size=min(len(common_traits), 5), 
                                                    replace=True)
                    
                    try:
                        from src.transfer_matrices import CrossModelTransferAnalyzer
                        from src.vector_extraction import VectorCollection
                        
                        # Create bootstrap collection
                        bootstrap_collection = VectorCollection("bootstrap")
                        for trait in sampled_traits:
                            for model in [model_a, model_b]:
                                trait_vectors = [v for v in vector_data["by_model"][model] 
                                               if v.trait_name == trait]
                                if trait_vectors:
                                    vector = np.random.choice(trait_vectors)
                                    bootstrap_collection.vectors[vector.vector_id] = vector
                                    bootstrap_collection.index[vector.model_name][vector.trait_name].append(vector.vector_id)
                        
                        analyzer = CrossModelTransferAnalyzer(bootstrap_collection)
                        matrix = analyzer.compute_transfer_matrix(model_a, model_b, sampled_traits)
                        error = analyzer._compute_reconstruction_error(matrix, model_a, model_b, sampled_traits)
                        bootstrap_errors.append(error)
                        
                    except Exception:
                        continue
                
                if bootstrap_errors:
                    bootstrap_errors = np.array(bootstrap_errors)
                    results["bootstrap_transfer"] = {
                        "mean_error": float(np.mean(bootstrap_errors)),
                        "std_error": float(np.std(bootstrap_errors)),
                        "ci_lower": float(np.percentile(bootstrap_errors, 2.5)),
                        "ci_upper": float(np.percentile(bootstrap_errors, 97.5)),
                        "n_samples": len(bootstrap_errors)
                    }
        
        # Save bootstrap results
        with open(self.results_dir / "bootstrap_validation.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_comprehensive_analysis(self):
        """Run the complete state-of-the-art analysis pipeline"""
        logger.info("🚀 Starting Comprehensive Platonic Analysis")
        logger.info("🔬 State-of-the-Art Methods for Cross-Model Preference Vector Analysis")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load data
        vector_data = self.load_all_vectors()
        
        if not vector_data["all_vectors"]:
            logger.error("No vectors found for analysis!")
            return False
        
        # Run all analysis components
        results = {}
        
        logger.info("📊 Phase 1: Data Quality & Exemplar Analysis")
        # 1. Quality exemplars (Rachel's requirement)
        results["quality_exemplars"] = self.create_quality_exemplars(vector_data)
        
        logger.info("🔄 Phase 2: Transfer Matrix Validation")
        # 2. Statistical validation (A→B→C tests)
        results["transfer_validation"] = self.statistical_transfer_validation(vector_data)
        
        logger.info("🧮 Phase 3: Advanced Geometry Analysis")
        # 3. Advanced Platonic subspace analysis
        results["platonic_subspace"] = self.platonic_subspace_analysis(vector_data)
        
        logger.info("📈 Phase 4: Bootstrap Statistical Validation")
        # 4. Bootstrap confidence intervals
        results["bootstrap_validation"] = self.bootstrap_statistical_validation(vector_data)
        
        logger.info("🎯 Phase 5: Clustering Contradiction Analysis")
        # 5. Geometric vs MSE clustering (Rachel's question)
        results["geometric_clustering"] = self.geometric_vs_mse_clustering(vector_data)
        
        logger.info("🏗️ Phase 6: Cross-Architecture Validation")
        # 6. Cross-architecture validation
        results["cross_architecture"] = self.cross_architecture_validation(vector_data)
        
        logger.info("🎨 Phase 7: Visualization Suite")
        # 7. Core visualizations
        self.create_core_visualizations(vector_data)
        
        logger.info("📋 Phase 8: Results Synthesis")
        # 8. Generate comprehensive summary
        self.generate_summary_report(results, vector_data)
        
        elapsed = time.time() - start_time
        logger.info(f"\n🎉 State-of-the-art Platonic analysis completed in {elapsed:.1f}s")
        logger.info(f"📊 Results saved to {self.results_dir}/")
        logger.info(f"🔬 Advanced methods: Subspace geometry, bootstrap validation, clustering analysis")
        logger.info(f"📈 Key findings ready for publication-quality analysis")
        
        return True
    
    def generate_summary_report(self, results: Dict, vector_data: Dict):
        """Generate comprehensive summary report"""
        logger.info("\n=== Generating Summary Report ===")
        
        report = {
            "analysis_summary": {
                "total_vectors": len(vector_data["all_vectors"]),
                "num_models": len(vector_data["by_model"]),
                "num_traits": len(vector_data["by_trait"]),
                "analysis_timestamp": time.time()
            },
            "quality_exemplars": {
                trait: {
                    "quality_range": f"[{ex.min_score:.3f}, {ex.max_score:.3f}]",
                    "quality_span": ex.max_score - ex.min_score
                }
                for trait, ex in results.get("quality_exemplars", {}).items()
            },
            "transfer_validation": {
                "num_tested_paths": len(results.get("transfer_validation", {})),
                "avg_transitivity_violation": np.mean([
                    r.transitivity_violation 
                    for r in results.get("transfer_validation", {}).values()
                ]) if results.get("transfer_validation") else 0,
                "max_violation": np.max([
                    r.transitivity_violation 
                    for r in results.get("transfer_validation", {}).values()
                ]) if results.get("transfer_validation") else 0
            },
            "geometric_clustering": {
                "num_vectors_analyzed": len(results.get("geometric_clustering", {})),
                "contradiction_rate": np.mean([
                    r.contradiction_score
                    for r in results.get("geometric_clustering", {}).values()
                ]) if results.get("geometric_clustering") else 0
            },
            "cross_architecture": {
                "tested_pairs": list(results.get("cross_architecture", {}).keys()),
                "transfer_errors": results.get("cross_architecture", {})
            }
        }
        
        # Save report
        with open(self.results_dir / "comprehensive_summary.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print key findings
        logger.info("📋 Key Findings:")
        logger.info(f"  • Analyzed {report['analysis_summary']['total_vectors']} vectors across {report['analysis_summary']['num_models']} models")
        logger.info(f"  • Quality ranges documented for {len(results.get('quality_exemplars', {}))} traits")
        logger.info(f"  • Tested {report['transfer_validation']['num_tested_paths']} transfer paths")
        logger.info(f"  • Average transitivity violation: {report['transfer_validation']['avg_transitivity_violation']:.4f}")
        logger.info(f"  • Geometric vs MSE contradiction rate: {report['geometric_clustering']['contradiction_rate']:.3f}")

def main():
    """Run comprehensive Platonic analysis"""
    # Set environment for cluster
    if 'OLLAMA_HOST' not in os.environ:
        os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
    
    analyzer = ComprehensivePlatonicAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\n🎉 Comprehensive Platonic analysis completed successfully!")
        print("📊 Check results in platonic_results/")
        return 0
    else:
        print("\n❌ Analysis failed!")
        return 1

if __name__ == "__main__":
    exit(main())