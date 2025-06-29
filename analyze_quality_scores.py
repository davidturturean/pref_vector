#!/usr/bin/env python3
"""
Quality Score Analysis for Extracted Vectors
Analyzes quality patterns across models and traits
"""

import os
import sys
import logging
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_quality_patterns():
    """Analyze quality score patterns across models and traits"""
    from src.vector_extraction import VectorCollection
    
    logger.info("=== Quality Score Analysis ===")
    
    # Load vectors from all available directories
    collections = []
    for collection_dir in ["vectors/core", "vectors/extended", "vectors/challenging", "vectors"]:
        if Path(collection_dir).exists():
            collection = VectorCollection(collection_dir)
            if len(collection.vectors) > 0:
                collections.append((collection_dir, collection))
                logger.info(f"Loaded {len(collection.vectors)} vectors from {collection_dir}")
    
    if not collections:
        logger.error("No vectors found for analysis")
        return {}
    
    # Combine all vectors
    all_vectors = {}
    for collection_dir, collection in collections:
        all_vectors.update(collection.vectors)
    
    logger.info(f"Total vectors for analysis: {len(all_vectors)}")
    
    # Organize by model and trait
    by_model = defaultdict(list)
    by_trait = defaultdict(list)
    by_model_trait = defaultdict(list)
    
    for vector_id, vector in all_vectors.items():
        by_model[vector.model_name].append(vector)
        by_trait[vector.trait_name].append(vector)
        by_model_trait[(vector.model_name, vector.trait_name)].append(vector)
    
    # Quality analysis
    analysis = {
        "overall_stats": {},
        "by_model": {},
        "by_trait": {},
        "by_model_trait": {},
        "quality_distribution": {},
        "recommendations": []
    }
    
    # Overall statistics
    all_qualities = [v.quality_score for v in all_vectors.values()]
    analysis["overall_stats"] = {
        "total_vectors": len(all_vectors),
        "mean_quality": np.mean(all_qualities),
        "std_quality": np.std(all_qualities),
        "min_quality": np.min(all_qualities),
        "max_quality": np.max(all_qualities),
        "median_quality": np.median(all_qualities)
    }
    
    logger.info(f"Overall quality: {analysis['overall_stats']['mean_quality']:.3f} ± {analysis['overall_stats']['std_quality']:.3f}")
    logger.info(f"Quality range: [{analysis['overall_stats']['min_quality']:.3f}, {analysis['overall_stats']['max_quality']:.3f}]")
    
    # By model analysis
    logger.info("\n--- Quality by Model ---")
    for model, vectors in by_model.items():
        qualities = [v.quality_score for v in vectors]
        model_stats = {
            "count": len(vectors),
            "mean_quality": np.mean(qualities),
            "std_quality": np.std(qualities),
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities)
        }
        analysis["by_model"][model] = model_stats
        logger.info(f"  {model}: {model_stats['mean_quality']:.3f} ± {model_stats['std_quality']:.3f} ({model_stats['count']} vectors)")
    
    # By trait analysis
    logger.info("\n--- Quality by Trait ---")
    for trait, vectors in by_trait.items():
        qualities = [v.quality_score for v in vectors]
        trait_stats = {
            "count": len(vectors),
            "mean_quality": np.mean(qualities),
            "std_quality": np.std(qualities),
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities)
        }
        analysis["by_trait"][trait] = trait_stats
        logger.info(f"  {trait}: {trait_stats['mean_quality']:.3f} ± {trait_stats['std_quality']:.3f} ({trait_stats['count']} vectors)")
    
    # Quality distribution
    quality_thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    distribution = {f"{quality_thresholds[i]:.1f}-{quality_thresholds[i+1]:.1f}": 0 
                   for i in range(len(quality_thresholds)-1)}
    
    for quality in all_qualities:
        for i in range(len(quality_thresholds)-1):
            if quality_thresholds[i] <= quality < quality_thresholds[i+1]:
                distribution[f"{quality_thresholds[i]:.1f}-{quality_thresholds[i+1]:.1f}"] += 1
                break
    
    analysis["quality_distribution"] = distribution
    
    logger.info("\n--- Quality Distribution ---")
    for range_str, count in distribution.items():
        percentage = count / len(all_qualities) * 100
        logger.info(f"  {range_str}: {count} vectors ({percentage:.1f}%)")
    
    # Generate recommendations
    recommendations = []
    
    # Check for low quality traits
    low_quality_traits = [trait for trait, stats in analysis["by_trait"].items() 
                         if stats["mean_quality"] < 0.4]
    if low_quality_traits:
        recommendations.append(f"Low quality traits detected: {low_quality_traits}")
        recommendations.append("Consider improving prompt templates or increasing sample size")
    
    # Check for model differences
    model_qualities = [stats["mean_quality"] for stats in analysis["by_model"].values()]
    if len(model_qualities) > 1 and np.std(model_qualities) > 0.1:
        recommendations.append("Significant quality differences between models detected")
        recommendations.append("Consider model-specific prompt optimization")
    
    # Check overall quality
    if analysis["overall_stats"]["mean_quality"] < 0.5:
        recommendations.append("Overall quality below 0.5 - consider improving extraction methodology")
    elif analysis["overall_stats"]["mean_quality"] > 0.7:
        recommendations.append("High quality extraction - proceed with Platonic analysis")
    
    analysis["recommendations"] = recommendations
    
    logger.info("\n--- Recommendations ---")
    for rec in recommendations:
        logger.info(f"  • {rec}")
    
    # Save analysis
    with open("quality_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    logger.info("\n📊 Quality analysis saved to quality_analysis.json")
    
    return analysis

def identify_best_vectors():
    """Identify the highest quality vectors for each trait"""
    from src.vector_extraction import VectorCollection
    
    logger.info("\n=== Best Vector Identification ===")
    
    # Load all vectors
    all_vectors = {}
    for collection_dir in ["vectors/core", "vectors/extended", "vectors/challenging", "vectors"]:
        if Path(collection_dir).exists():
            collection = VectorCollection(collection_dir)
            all_vectors.update(collection.vectors)
    
    # Find best vector for each trait
    by_trait = defaultdict(list)
    for vector in all_vectors.values():
        by_trait[vector.trait_name].append(vector)
    
    best_vectors = {}
    for trait, vectors in by_trait.items():
        best_vector = max(vectors, key=lambda v: v.quality_score)
        best_vectors[trait] = {
            "vector_id": best_vector.vector_id,
            "model": best_vector.model_name,
            "quality": best_vector.quality_score,
            "extraction_time": best_vector.extraction_time
        }
    
    logger.info("Best vectors by trait:")
    for trait, info in best_vectors.items():
        logger.info(f"  {trait}: {info['model']} (quality={info['quality']:.3f})")
    
    return best_vectors

def main():
    """Run quality analysis"""
    logger.info("🔍 Starting Quality Score Analysis...")
    
    analysis = analyze_quality_patterns()
    best_vectors = identify_best_vectors()
    
    if analysis:
        logger.info("✅ Quality analysis completed successfully!")
        return 0
    else:
        logger.error("❌ Quality analysis failed!")
        return 1

if __name__ == "__main__":
    exit(main())