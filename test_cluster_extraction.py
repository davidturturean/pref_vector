#!/usr/bin/env python3
"""
Cluster-specific vector extraction test using actual Ollama model names
"""

import os
import sys
import logging
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

def test_cluster_extraction():
    """Test vector extraction using actual cluster model names"""
    
    logger.info("=== Cluster Vector Extraction Test ===")
    logger.info(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")
    logger.info(f"OLLAMA_HOME: {os.environ.get('OLLAMA_HOME')}")
    
    try:
        from src.vector_extraction import VectorCollection, ContrastiveActivationExtractor, PromptGenerator
        from src.model_loader import get_model_loader
        
        # Test connection first
        loader = get_model_loader()
        logger.info(f"Model loader host: {loader.ollama_manager.host}")
        
        # List available models
        available_models = loader.ollama_manager.list_available_models()
        logger.info(f"Available models: {available_models}")
        
        if not available_models:
            logger.error("No models available!")
            return False
        
        # Use the first available model for testing
        test_model = available_models[0]
        logger.info(f"Testing with model: {test_model}")
        
        # Test model loading with exact Ollama name
        try:
            model_info = loader.load_model(test_model)
            logger.info(f"✓ Model loaded: {model_info.hf_name}")
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            return False
        
        # Test single vector extraction
        logger.info("Testing single vector extraction...")
        
        extractor = ContrastiveActivationExtractor(loader)
        prompt_generator = PromptGenerator()
        
        # Generate prompts for verbosity
        prompt_pairs = prompt_generator.generate_contrastive_pairs("verbosity", num_pairs=3)
        logger.info(f"Generated {len(prompt_pairs)} prompt pairs")
        
        if not prompt_pairs:
            logger.error("No prompt pairs generated!")
            return False
        
        # Extract vector
        try:
            vector = extractor.extract_vector(model_info, "verbosity", prompt_pairs)
            logger.info(f"✓ Vector extracted: quality={vector.quality_score:.3f}")
            
            # Save vector
            collection = VectorCollection("vectors/test")
            collection.add_vector(vector)
            logger.info("✓ Vector saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Vector extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run cluster extraction test"""
    success = test_cluster_extraction()
    
    if success:
        logger.info("🎉 Cluster extraction test PASSED!")
        return 0
    else:
        logger.error("⚠️  Cluster extraction test FAILED!")
        return 1

if __name__ == "__main__":
    exit(main())