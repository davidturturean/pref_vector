#!/usr/bin/env python3
"""
Batch Vector Extraction Script

Focus on data collection using the working vector extraction system.
Analysis components will be added once environment is fixed.
"""

import sys
import logging
import time
import json
from pathlib import Path
import numpy as np

# Add project root to path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Only import what works (vector extraction only)
from src.vector_extraction import extract_all_vectors
from src.config import get_available_models, CORE_STYLE_TRAITS, EXTENDED_STYLE_TRAITS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_core_traits_first():
    """Extract core traits (verbosity, formality, etc.) from all models first."""
    
    logger.info("=== Phase 1: Core Trait Extraction ===")
    
    models = get_available_models()
    core_traits = CORE_STYLE_TRAITS  # ["verbosity", "formality", "technical_complexity", "certainty"]
    
    logger.info(f"Extracting {len(core_traits)} core traits from {len(models)} models")
    logger.info(f"Models: {models}")
    logger.info(f"Core traits: {core_traits}")
    
    start_time = time.time()
    
    try:
        collection = extract_all_vectors(
            model_names=models,
            trait_names=core_traits,
            collection_dir="vectors/core"
        )
        
        extraction_time = time.time() - start_time
        
        logger.info(f"✅ Core extraction completed in {extraction_time:.1f}s")
        logger.info(f"Extracted {len(collection.vectors)} vectors total")
        
        # Report per-model results
        for model in models:
            model_vectors = [v for v in collection.vectors.values() if v.model_name == model]
            avg_quality = sum(v.quality_score for v in model_vectors) / len(model_vectors) if model_vectors else 0
            logger.info(f"  {model}: {len(model_vectors)} vectors, avg quality: {avg_quality:.3f}")
        
        return collection
        
    except Exception as e:
        logger.error(f"❌ Core extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_extended_traits():
    """Extract extended traits after core traits are successful."""
    
    logger.info("\n=== Phase 2: Extended Trait Extraction ===")
    
    models = get_available_models()
    
    # Select a subset of extended traits (top priority ones)
    priority_extended_traits = [
        "emotional_tone", "politeness", "assertiveness", "clarity", 
        "specificity", "objectivity", "creativity", "authority"
    ]
    
    logger.info(f"Extracting {len(priority_extended_traits)} extended traits")
    logger.info(f"Extended traits: {priority_extended_traits}")
    
    start_time = time.time()
    
    try:
        collection = extract_all_vectors(
            model_names=models,
            trait_names=priority_extended_traits,
            collection_dir="vectors/extended"
        )
        
        extraction_time = time.time() - start_time
        
        logger.info(f"✅ Extended extraction completed in {extraction_time:.1f}s")
        logger.info(f"Extracted {len(collection.vectors)} vectors total")
        
        return collection
        
    except Exception as e:
        logger.error(f"❌ Extended extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_challenging_traits():
    """Extract the most challenging traits (humor, cultural, etc.) last."""
    
    logger.info("\n=== Phase 3: Challenging Trait Extraction ===")
    
    models = get_available_models() 
    
    challenging_traits = [
        "humor", "empathy", "persuasiveness", "optimism",
        "urgency", "inclusivity", "concreteness"
    ]
    
    logger.info(f"Extracting {len(challenging_traits)} challenging traits")
    logger.info(f"Challenging traits: {challenging_traits}")
    
    start_time = time.time()
    
    try:
        collection = extract_all_vectors(
            model_names=models,
            trait_names=challenging_traits,
            collection_dir="vectors/challenging"
        )
        
        extraction_time = time.time() - start_time
        
        logger.info(f"✅ Challenging extraction completed in {extraction_time:.1f}s")
        logger.info(f"Extracted {len(collection.vectors)} vectors total")
        
        return collection
        
    except Exception as e:
        logger.error(f"❌ Challenging extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def combine_all_vectors():
    """Combine vectors from all phases into a single collection."""
    
    logger.info("\n=== Combining All Vectors ===")
    
    from src.vector_extraction import VectorCollection
    
    combined_collection = VectorCollection("vectors/combined")
    
    # Load from all directories
    vector_dirs = ["vectors/core", "vectors/extended", "vectors/challenging"]
    
    total_loaded = 0
    for vector_dir in vector_dirs:
        dir_path = Path(vector_dir)
        if dir_path.exists():
            for json_file in dir_path.glob("*.json"):
                try:
                    from src.vector_extraction import StyleVector
                    vector = StyleVector.load(json_file)
                    combined_collection.add_vector(vector, save=False)
                    total_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"✅ Loaded {total_loaded} vectors into combined collection")
    
    # Save summary
    summary = {
        "total_vectors": len(combined_collection.vectors),
        "models": list(combined_collection.index.keys()),
        "traits_per_model": {
            model: list(traits.keys()) 
            for model, traits in combined_collection.index.items()
        },
        "extraction_timestamp": time.time()
    }
    
    with open("vectors/combined/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return combined_collection

def generate_extraction_report(collection):
    """Generate a detailed extraction report."""
    
    logger.info("\n=== Generating Extraction Report ===")
    
    if not collection:
        logger.error("No collection provided for report")
        return
    
    report = {
        "extraction_summary": {
            "total_vectors": len(collection.vectors),
            "num_models": len(collection.index),
            "num_traits": len(set(v.trait_name for v in collection.vectors.values())),
            "avg_quality_score": sum(v.quality_score for v in collection.vectors.values()) / len(collection.vectors)
        },
        "model_breakdown": {},
        "trait_breakdown": {},
        "quality_analysis": {}
    }
    
    # Model breakdown
    for model in collection.index:
        model_vectors = [v for v in collection.vectors.values() if v.model_name == model]
        report["model_breakdown"][model] = {
            "num_vectors": len(model_vectors),
            "traits": list(collection.index[model].keys()),
            "avg_quality": sum(v.quality_score for v in model_vectors) / len(model_vectors),
            "avg_extraction_time": sum(v.extraction_time for v in model_vectors) / len(model_vectors)
        }
    
    # Trait breakdown  
    all_traits = set(v.trait_name for v in collection.vectors.values())
    for trait in all_traits:
        trait_vectors = [v for v in collection.vectors.values() if v.trait_name == trait]
        report["trait_breakdown"][trait] = {
            "num_models": len(trait_vectors),
            "avg_quality": sum(v.quality_score for v in trait_vectors) / len(trait_vectors),
            "quality_range": [min(v.quality_score for v in trait_vectors), max(v.quality_score for v in trait_vectors)]
        }
    
    # Quality analysis
    qualities = [v.quality_score for v in collection.vectors.values()]
    report["quality_analysis"] = {
        "mean": sum(qualities) / len(qualities),
        "min": min(qualities),
        "max": max(qualities),
        "high_quality_count": sum(1 for q in qualities if q > 0.7),
        "low_quality_count": sum(1 for q in qualities if q < 0.3)
    }
    
    # Save report
    with open("vectors/extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("📊 Extraction Report Summary:")
    logger.info(f"  Total vectors: {report['extraction_summary']['total_vectors']}")
    logger.info(f"  Models: {report['extraction_summary']['num_models']}")
    logger.info(f"  Traits: {report['extraction_summary']['num_traits']}")
    logger.info(f"  Avg quality: {report['extraction_summary']['avg_quality_score']:.3f}")
    logger.info(f"  High quality (>0.7): {report['quality_analysis']['high_quality_count']}")
    logger.info(f"  Low quality (<0.3): {report['quality_analysis']['low_quality_count']}")

def main():
    """Main extraction pipeline."""
    
    logger.info("🚀 Starting Batch Vector Extraction for Platonic Analysis")
    logger.info("=" * 60)
    
    overall_start = time.time()
    
    # Phase 1: Core traits (most important)
    core_collection = extract_core_traits_first()
    if not core_collection:
        logger.error("❌ Core extraction failed, stopping")
        return 1
    
    # Phase 2: Extended traits (medium priority)
    extended_collection = extract_extended_traits()
    
    # Phase 3: Challenging traits (experimental)
    challenging_collection = extract_challenging_traits() 
    
    # Combine all results
    final_collection = combine_all_vectors()
    
    # Generate comprehensive report
    generate_extraction_report(final_collection)
    
    overall_time = time.time() - overall_start
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 Batch extraction completed in {overall_time:.1f}s")
    logger.info(f"Next steps:")
    logger.info(f"1. Review extraction_report.json")
    logger.info(f"2. Fix scipy compatibility issues")
    logger.info(f"3. Run Platonic analysis: python3 scripts/run_platonic_analysis.py")
    
    return 0

if __name__ == "__main__":
    exit(main())