#!/usr/bin/env python3
"""
Vector Collection Script - Phase 1 Implementation

This script extracts style vectors from multiple models across different traits,
implementing the first phase of our research plan.
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import EXPERIMENT_CONFIG, MODEL_FAMILIES, CORE_STYLE_TRAITS, EXTENDED_STYLE_TRAITS
from src.vector_extraction import extract_all_vectors, VectorCollection, PromptGenerator, ContrastiveActivationExtractor
from src.model_loader import get_model_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Extract style vectors from language models")
    
    # Model selection
    parser.add_argument("--models", nargs="+", help="Specific models to use")
    parser.add_argument("--families", nargs="+", choices=list(MODEL_FAMILIES.keys()), 
                       help="Model families to include")
    parser.add_argument("--all-families", action="store_true", help="Use all available model families")
    
    # Trait selection
    parser.add_argument("--traits", nargs="+", help="Specific traits to extract")
    parser.add_argument("--core-only", action="store_true", help="Extract only core traits")
    parser.add_argument("--extended-only", action="store_true", help="Extract only extended traits")
    
    # Extraction parameters
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of contrastive pairs per trait")
    parser.add_argument("--collection-dir", default="vectors", help="Directory to save vectors")
    
    # Testing options
    parser.add_argument("--dry-run", action="store_true", help="Test setup without extraction")
    parser.add_argument("--quick-test", action="store_true", help="Extract from one model, one trait")
    
    args = parser.parse_args()
    
    # Determine models to use
    if args.models:
        models = args.models
    elif args.families:
        models = []
        for family in args.families:
            if family in MODEL_FAMILIES:
                models.extend(MODEL_FAMILIES[family])
    elif args.all_families:
        models = []
        for family_models in MODEL_FAMILIES.values():
            models.extend(family_models)
    else:
        # Default: use configured models
        models = [EXPERIMENT_CONFIG.source_model] + EXPERIMENT_CONFIG.target_models
    
    # Determine traits to use
    if args.traits:
        traits = args.traits
    elif args.core_only:
        traits = CORE_STYLE_TRAITS
    elif args.extended_only:
        traits = EXTENDED_STYLE_TRAITS
    else:
        # Default: use all traits
        traits = CORE_STYLE_TRAITS + EXTENDED_STYLE_TRAITS
    
    # Quick test mode
    if args.quick_test:
        models = [models[0]] if models else ["google/gemma-2b"]
        traits = ["verbosity"]
        logger.info("Quick test mode: using first model and verbosity trait only")
    
    logger.info(f"Will extract vectors for:")
    logger.info(f"  Models: {len(models)} models")
    for model in models:
        logger.info(f"    - {model}")
    logger.info(f"  Traits: {len(traits)} traits")
    for trait in traits:
        logger.info(f"    - {trait}")
    
    if args.dry_run:
        logger.info("Dry run mode - no actual extraction will be performed")
        return
    
    # Validate setup
    try:
        model_loader = get_model_loader()
        prompt_generator = PromptGenerator()
        
        # Test prompt generation
        logger.info("Testing prompt generation...")
        for trait in traits[:3]:  # Test first 3 traits
            pairs = prompt_generator.generate_contrastive_pairs(trait, num_pairs=2)
            if pairs:
                logger.info(f"  ✓ Generated {len(pairs)} pairs for {trait}")
            else:
                logger.warning(f"  ⚠️ No pairs generated for {trait}")
        
    except Exception as e:
        logger.error(f"Setup validation failed: {e}")
        return
    
    # Run extraction
    start_time = time.time()
    
    try:
        logger.info("Starting vector extraction...")
        collection = extract_all_vectors(
            model_names=models,
            trait_names=traits,
            collection_dir=args.collection_dir
        )
        
        extraction_time = time.time() - start_time
        
        # Report results
        logger.info(f"Extraction completed in {extraction_time:.1f} seconds")
        logger.info(f"Total vectors extracted: {len(collection.vectors)}")
        
        # Generate summary
        stats = {}
        for model in models:
            model_vectors = [v for v in collection.vectors.values() if v.model_name == model]
            stats[model] = len(model_vectors)
        
        logger.info("Extraction summary by model:")
        for model, count in stats.items():
            logger.info(f"  {model}: {count} vectors")
        
        # Quality summary
        quality_scores = [v.quality_score for v in collection.vectors.values()]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            logger.info(f"Average quality score: {avg_quality:.3f}")
        
        # Save summary
        summary_path = Path(args.collection_dir) / "extraction_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Vector Extraction Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Extraction time: {extraction_time:.1f} seconds\n")
            f.write(f"Total vectors: {len(collection.vectors)}\n")
            f.write(f"Average quality: {avg_quality:.3f}\n\n")
            f.write(f"Vectors by model:\n")
            for model, count in stats.items():
                f.write(f"  {model}: {count}\n")
        
        logger.info(f"Summary saved to {summary_path}")
        
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()