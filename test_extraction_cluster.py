#!/usr/bin/env python3
"""
Test vector extraction on cluster with fixed Ollama connection
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_extraction():
    """Test vector extraction with cluster Ollama instance"""
    
    # Force the custom port
    os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
    
    logger.info("=== Testing Vector Extraction on Cluster ===")
    logger.info(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")
    
    try:
        from src.vector_extraction import VectorCollection, extract_all_vectors
        
        # Test with just one model and one trait to start
        logger.info("Testing extraction with Mistral on 'verbosity' trait...")
        
        collection = extract_all_vectors(
            model_names=["mistral:7b-instruct"],
            trait_names=["verbosity"],
            collection_dir="vectors"
        )
        
        if len(collection.vectors) > 0:
            logger.info("✓ Vector extraction successful!")
            for vector_id, vector in collection.vectors.items():
                logger.info(f"  {vector.model_name} - {vector.trait_name}: quality={vector.quality_score:.3f}")
            return True
        else:
            logger.error("✗ No vectors extracted")
            return False
            
    except Exception as e:
        logger.error(f"✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run extraction test"""
    success = test_extraction()
    
    if success:
        logger.info("🎉 Cluster extraction test passed!")
        return 0
    else:
        logger.error("⚠️  Cluster extraction test failed!")
        return 1

if __name__ == "__main__":
    exit(main())