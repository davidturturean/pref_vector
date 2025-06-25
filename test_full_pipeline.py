#!/usr/bin/env python3
"""
Complete Pipeline Test for Platonic Hypothesis Analysis
Tests the full pipeline with real extracted vectors.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.vector_extraction import VectorCollection
from src.transfer_matrices import CrossModelTransferAnalyzer
from src.platonic_analysis import PlatonicAnalyzer
from src.clustering_analysis import CrossModelClusteringAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the complete pipeline with real extracted vectors."""
    logger.info("=== Platonic Hypothesis Analysis Pipeline ===")
    
    # Load real vectors
    logger.info("Loading extracted vectors...")
    collection = VectorCollection("vectors")
    
    if len(collection.vectors) == 0:
        logger.error("No vectors found. Run vector extraction first.")
        return False
    
    # Show what we have
    models = set(v.model_name for v in collection.vectors.values())
    traits = set(v.trait_name for v in collection.vectors.values())
    
    logger.info(f"Found {len(collection.vectors)} vectors")
    logger.info(f"Models: {sorted(models)}")
    logger.info(f"Traits: {sorted(traits)}")
    
    success = True
    
    # Test 1: Transfer Matrix Analysis
    logger.info("\n--- Transfer Matrix Analysis ---")
    try:
        transfer_analyzer = CrossModelTransferAnalyzer(collection)
        transfer_results = transfer_analyzer.compute_all_transfer_matrices()
        
        logger.info(f"✓ Computed {len(transfer_results)} transfer matrices")
        
        # Test analysis
        analysis = transfer_analyzer.analyze_transfer_quality()
        logger.info("✓ Transfer quality analysis completed")
        
        # Show results
        for method in ['pseudoinverse', 'procrustes']:
            if f"{method}_summary" in analysis["method_comparison"]:
                summary = analysis["method_comparison"][f"{method}_summary"]
                logger.info(f"  {method}: mean_error={summary['mean_error']:.4f}")
        
    except Exception as e:
        logger.error(f"✗ Transfer matrix analysis failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Test 2: Platonic Hypothesis Analysis
    logger.info("\n--- Platonic Analysis ---")
    try:
        platonic_analyzer = PlatonicAnalyzer(collection)
        
        # Subspace geometry analysis
        subspace_results = platonic_analyzer.compute_subspace_geometry_analysis()
        logger.info(f"✓ Subspace analysis: {len(subspace_results)} model pairs")
        
        for pair_name, result in subspace_results.items():
            logger.info(f"  {pair_name}: CKA={result.cka_similarity:.3f}, Grassmann={result.grassmann_distance:.3f}")
        
        # Union basis construction
        union_results = platonic_analyzer.construct_union_basis()
        logger.info(f"✓ Union basis: rank={union_results.optimal_rank}, "
                   f"variance={union_results.total_variance_explained:.3f}")
        
        # Incremental trait analysis
        incremental_results = platonic_analyzer.incremental_trait_analysis()
        logger.info(f"✓ Incremental analysis: {len(incremental_results.optimal_trait_set)} optimal traits")
        
        # Generate summary
        summary = platonic_analyzer.generate_summary_report()
        logger.info(f"✓ Platonic score: {summary.platonic_score:.3f}")
        
    except Exception as e:
        logger.error(f"✗ Platonic analysis failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Test 3: Clustering Analysis
    logger.info("\n--- Clustering Analysis ---")
    try:
        clustering_analyzer = CrossModelClusteringAnalyzer(collection)
        
        # Hierarchical clustering
        clustering_results = clustering_analyzer.hierarchical_clustering_analysis()
        logger.info(f"✓ Hierarchical clustering: {clustering_results.n_clusters} clusters")
        
        # Factor analysis
        factor_results = clustering_analyzer.semantic_factor_analysis()
        logger.info(f"✓ Factor analysis: {factor_results.n_factors} factors")
        
        # Outlier analysis
        outlier_results = clustering_analyzer.analyze_outlier_traits()
        logger.info(f"✓ Outlier analysis: {len(outlier_results['universal_outliers'])} universal outliers")
        
    except Exception as e:
        logger.error(f"✗ Clustering analysis failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success

def main():
    """Run the pipeline test."""
    logger.info("=== Complete Platonic Analysis Pipeline Test ===")
    
    success = test_pipeline()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("=== FINAL RESULTS ===")
    
    if success:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("\n🎉 Pipeline is ready for Platonic Hypothesis analysis!")
    else:
        logger.info("✗ SOME TESTS FAILED")
        logger.info("\n⚠️  Some components need fixes before proceeding.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())