#!/usr/bin/env python3
"""
Real-time Platonic Hypothesis Analysis Pipeline
Runs analysis on extracted vectors as they become available
"""

import os
import sys
import logging
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_vector_availability():
    """Check how many vectors are available for analysis"""
    from src.vector_extraction import VectorCollection
    
    # Check different collection directories
    collections = {}
    for collection_dir in ["vectors/core", "vectors/extended", "vectors/challenging", "vectors"]:
        if Path(collection_dir).exists():
            collection = VectorCollection(collection_dir)
            if len(collection.vectors) > 0:
                collections[collection_dir] = collection
                logger.info(f"Found {len(collection.vectors)} vectors in {collection_dir}")
    
    return collections

def run_initial_platonic_analysis(collections):
    """Run initial Platonic analysis on available vectors"""
    logger.info("=== Initial Platonic Hypothesis Analysis ===")
    
    # Find the collection with the most vectors
    best_collection = max(collections.values(), key=lambda c: len(c.vectors))
    logger.info(f"Using collection with {len(best_collection.vectors)} vectors")
    
    # Get models and traits
    models = list(best_collection.index.keys())
    all_traits = set(v.trait_name for v in best_collection.vectors.values())
    
    logger.info(f"Models: {models}")
    logger.info(f"Traits: {sorted(all_traits)}")
    
    if len(models) < 2:
        logger.warning("Need at least 2 models for cross-model analysis")
        return False
    
    # Test 1: Transfer Matrix Analysis
    logger.info("\n--- Transfer Matrix Analysis ---")
    try:
        from src.transfer_matrices import CrossModelTransferAnalyzer
        
        transfer_analyzer = CrossModelTransferAnalyzer(best_collection)
        transfer_results = transfer_analyzer.compute_all_transfer_matrices()
        
        logger.info(f"✓ Computed {len(transfer_results)} transfer matrices")
        
        # Quick quality analysis
        analysis = transfer_analyzer.analyze_transfer_quality()
        logger.info("✓ Transfer quality analysis completed")
        
        # Show key results
        for method in ['pseudoinverse', 'procrustes']:
            if f"{method}_summary" in analysis.get("method_comparison", {}):
                summary = analysis["method_comparison"][f"{method}_summary"]
                logger.info(f"  {method}: mean_error={summary['mean_error']:.4f}")
        
    except Exception as e:
        logger.error(f"✗ Transfer matrix analysis failed: {e}")
        return False
    
    # Test 2: Basic Platonic Analysis
    logger.info("\n--- Basic Platonic Analysis ---")
    try:
        from src.platonic_analysis import PlatonicAnalyzer
        
        platonic_analyzer = PlatonicAnalyzer(best_collection)
        
        # Subspace geometry analysis
        subspace_results = platonic_analyzer.compute_subspace_geometry_analysis()
        logger.info(f"✓ Subspace analysis: {len(subspace_results)} model pairs")
        
        for pair_name, result in list(subspace_results.items())[:3]:  # Show first 3
            logger.info(f"  {pair_name}: CKA={result.cka_similarity:.3f}, Grassmann={result.grassmann_distance:.3f}")
        
        # Union basis construction
        union_results = platonic_analyzer.construct_union_basis()
        logger.info(f"✓ Union basis: rank={union_results.optimal_rank}, "
                   f"variance={union_results.total_variance_explained:.3f}")
        
        # Generate summary
        summary = platonic_analyzer.generate_summary_report()
        logger.info(f"✓ Platonic score: {summary.platonic_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Platonic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_clustering_analysis(collections):
    """Run clustering analysis to validate cross-model consistency"""
    logger.info("\n--- Cross-Model Clustering Analysis ---")
    
    best_collection = max(collections.values(), key=lambda c: len(c.vectors))
    
    try:
        from src.clustering_analysis import CrossModelClusteringAnalyzer
        
        clustering_analyzer = CrossModelClusteringAnalyzer(best_collection)
        
        # Hierarchical clustering
        clustering_results = clustering_analyzer.hierarchical_clustering_analysis()
        logger.info(f"✓ Hierarchical clustering: {clustering_results.n_clusters} clusters")
        
        # Factor analysis
        factor_results = clustering_analyzer.semantic_factor_analysis()
        logger.info(f"✓ Factor analysis: {factor_results.n_factors} factors")
        
        # Show factor interpretations
        for factor_name, traits in list(factor_results.factor_interpretations.items())[:3]:
            logger.info(f"  {factor_name}: {traits}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Clustering analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_and_analyze():
    """Monitor vector extraction and run analysis when ready"""
    logger.info("🔍 Monitoring vector extraction progress...")
    
    min_vectors_for_analysis = 6  # At least 2 models × 3 traits
    last_check_count = 0
    
    while True:
        collections = check_vector_availability()
        
        if not collections:
            logger.info("No vectors found yet, waiting...")
            time.sleep(30)
            continue
        
        total_vectors = sum(len(c.vectors) for c in collections.values())
        
        if total_vectors != last_check_count:
            logger.info(f"📊 Total vectors available: {total_vectors}")
            last_check_count = total_vectors
        
        if total_vectors >= min_vectors_for_analysis:
            logger.info(f"🎯 Sufficient vectors ({total_vectors}) available for analysis!")
            break
        
        time.sleep(30)
    
    # Run analysis
    logger.info("\n🚀 Starting Platonic Hypothesis Analysis...")
    
    success_count = 0
    
    # Run transfer matrix and platonic analysis
    if run_initial_platonic_analysis(collections):
        success_count += 1
    
    # Run clustering analysis
    if run_clustering_analysis(collections):
        success_count += 1
    
    # Summary
    logger.info(f"\n=== Analysis Complete ===")
    if success_count >= 2:
        logger.info("🎉 Platonic analysis pipeline completed successfully!")
        logger.info("✓ Transfer matrices computed")
        logger.info("✓ Subspace geometry analyzed") 
        logger.info("✓ Union basis constructed")
        logger.info("✓ Clustering analysis completed")
        return True
    else:
        logger.warning("⚠️  Some analysis components failed")
        return False

def main():
    """Main analysis pipeline"""
    logger.info("=== Real-time Platonic Analysis Pipeline ===")
    
    # Set environment for cluster if needed
    if 'OLLAMA_HOST' not in os.environ:
        os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
    
    success = monitor_and_analyze()
    
    if success:
        logger.info("🎉 Platonic hypothesis analysis completed!")
        logger.info("📊 Check results in the logs above")
        return 0
    else:
        logger.error("⚠️  Analysis pipeline failed")
        return 1

if __name__ == "__main__":
    exit(main())