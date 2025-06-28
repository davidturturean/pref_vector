#!/usr/bin/env python3
"""
Cluster-specific vector extraction using actual Ollama model names
"""

import os
import sys
import logging
import time
from pathlib import Path

# Set environment variables BEFORE importing anything
os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
os.environ['OLLAMA_HOME'] = '/mnt/align4_drive2/davidct/.ollama'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_vectors_cluster():
    """Extract vectors using the actual model names available in cluster Ollama"""
    
    logger.info("=== Cluster Vector Extraction ===")
    logger.info(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")
    logger.info(f"OLLAMA_HOME: {os.environ.get('OLLAMA_HOME')}")
    
    try:
        from src.vector_extraction import VectorCollection, ContrastiveActivationExtractor, PromptGenerator
        from src.model_loader import get_model_loader
        
        # Get the model loader and check connection
        loader = get_model_loader()
        logger.info(f"Model loader host: {loader.ollama_manager.host}")
        
        # Get available models from Ollama directly
        available_models = loader.ollama_manager.list_available_models()
        logger.info(f"Available models: {available_models}")
        
        if not available_models:
            logger.error("No models available in Ollama!")
            return False
        
        # Use the actual model names from Ollama
        cluster_models = available_models  # ['mistral:7b-instruct', 'gemma:7b-instruct', 'llama2:7b-chat']
        
        # Start with core traits only
        core_traits = ["verbosity", "formality", "technical_complexity", "certainty"]
        
        logger.info(f"Extracting {len(core_traits)} traits from {len(cluster_models)} models")
        logger.info(f"Models: {cluster_models}")
        logger.info(f"Traits: {core_traits}")
        
        # Create collection
        collection = VectorCollection("vectors/cluster")
        
        # Extract vectors for each model-trait combination
        extractor = ContrastiveActivationExtractor(loader)
        prompt_generator = PromptGenerator()
        
        total_extracted = 0
        total_failed = 0
        
        for model_name in cluster_models:
            logger.info(f"\n--- Processing model: {model_name} ---")
            
            try:
                # Load the model using its Ollama name directly
                model_info = loader.load_model(model_name)
                logger.info(f"✓ Model loaded: {model_info.hf_name}")
                
                for trait in core_traits:
                    logger.info(f"Extracting {trait} vector...")
                    
                    try:
                        # Generate prompt pairs
                        prompt_pairs = prompt_generator.generate_contrastive_pairs(trait, num_pairs=5)
                        
                        if not prompt_pairs:
                            logger.warning(f"No prompt pairs for {trait}")
                            total_failed += 1
                            continue
                        
                        # Extract vector
                        vector = extractor.extract_vector(model_info, trait, prompt_pairs)
                        
                        # Save to collection
                        collection.add_vector(vector)
                        
                        logger.info(f"✓ {trait} extracted: quality={vector.quality_score:.3f}")
                        total_extracted += 1
                        
                    except Exception as e:
                        logger.error(f"✗ Failed to extract {trait}: {e}")
                        total_failed += 1
                        
            except Exception as e:
                logger.error(f"✗ Failed to load model {model_name}: {e}")
                continue
        
        # Summary
        logger.info(f"\n=== Extraction Complete ===")
        logger.info(f"✓ Successfully extracted: {total_extracted} vectors")
        logger.info(f"✗ Failed extractions: {total_failed}")
        logger.info(f"📁 Vectors saved to: vectors/cluster/")
        
        # Report quality scores
        if collection.vectors:
            qualities = [v.quality_score for v in collection.vectors.values()]
            logger.info(f"📊 Quality scores: avg={sum(qualities)/len(qualities):.3f}, min={min(qualities):.3f}, max={max(qualities):.3f}")
            
            # Show per-model results
            for model in cluster_models:
                model_vectors = [v for v in collection.vectors.values() if v.model_name == model]
                if model_vectors:
                    avg_quality = sum(v.quality_score for v in model_vectors) / len(model_vectors)
                    logger.info(f"  {model}: {len(model_vectors)} vectors, avg quality: {avg_quality:.3f}")
        
        return total_extracted > 0
        
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run cluster vector extraction"""
    
    start_time = time.time()
    
    success = extract_vectors_cluster()
    
    elapsed = time.time() - start_time
    
    if success:
        logger.info(f"🎉 Cluster extraction COMPLETED successfully in {elapsed:.1f}s!")
        return 0
    else:
        logger.error(f"⚠️  Cluster extraction FAILED after {elapsed:.1f}s!")
        return 1

if __name__ == "__main__":
    exit(main())