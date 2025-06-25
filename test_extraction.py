#!/usr/bin/env python3
"""
Test script for vector extraction system
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.vector_extraction import (
    extract_all_vectors, ContrastiveActivationExtractor, 
    PromptGenerator, VectorCollection
)
from src.model_loader import get_model_loader
from src.config import get_available_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_prompt_generation():
    """Test prompt generation for different traits."""
    logger.info("Testing prompt generation...")
    
    generator = PromptGenerator()
    
    test_traits = ["verbosity", "formality", "technical_complexity"]
    
    for trait in test_traits:
        pairs = generator.generate_contrastive_pairs(trait, num_pairs=3)
        logger.info(f"\n{trait.upper()} examples:")
        for i, (low, high) in enumerate(pairs[:2]):
            logger.info(f"  Pair {i+1}:")
            logger.info(f"    Low:  {low}")
            logger.info(f"    High: {high}")
    
    logger.info("✓ Prompt generation test passed")

def test_model_loader():
    """Test model loader functionality."""
    logger.info("Testing model loader...")
    
    model_loader = get_model_loader()
    available_models = get_available_models()
    
    logger.info(f"Available models: {available_models}")
    
    if available_models:
        test_model = available_models[0]
        logger.info(f"Testing with model: {test_model}")
        
        try:
            model_info = model_loader.load_model(test_model)
            logger.info(f"✓ Successfully loaded model: {model_info.hf_name}")
            logger.info(f"  Architecture: {model_info.architecture}")
            logger.info(f"  Hidden size: {model_info.hidden_size}")
            
            # Test text generation
            response = model_loader.generate_text(
                test_model, 
                "Explain machine learning briefly.", 
                options={"temperature": 0.1, "num_predict": 50}
            )
            logger.info(f"  Test generation: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            return False
    
    logger.info("✓ Model loader test passed")
    return True

def test_vector_extraction():
    """Test vector extraction for a single trait."""
    logger.info("Testing vector extraction...")
    
    available_models = get_available_models()
    if not available_models:
        logger.error("No models available for testing")
        return False
    
    test_model = available_models[0]
    test_trait = "verbosity"
    
    try:
        model_loader = get_model_loader()
        extractor = ContrastiveActivationExtractor(model_loader)
        generator = PromptGenerator()
        
        # Generate prompt pairs
        prompt_pairs = generator.generate_contrastive_pairs(test_trait, num_pairs=3)
        logger.info(f"Generated {len(prompt_pairs)} prompt pairs for {test_trait}")
        
        # Load model
        model_info = model_loader.load_model(test_model)
        
        # Extract vector
        logger.info(f"Extracting {test_trait} vector from {test_model}...")
        vector = extractor.extract_vector(model_info, test_trait, prompt_pairs)
        
        logger.info(f"✓ Vector extracted successfully:")
        logger.info(f"  Vector ID: {vector.vector_id}")
        logger.info(f"  Quality score: {vector.quality_score:.3f}")
        logger.info(f"  Vector norm: {vector.vector_norm:.3f}")
        logger.info(f"  Extraction time: {vector.extraction_time:.2f}s")
        logger.info(f"  Validation scores: {vector.validation_scores}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_collection():
    """Test vector collection functionality."""
    logger.info("Testing vector collection...")
    
    collection = VectorCollection("test_vectors")
    
    # Test with small subset of models and traits
    available_models = get_available_models()
    if not available_models:
        logger.error("No models available for testing")
        return False
    
    test_models = available_models[:1]  # Just one model for testing
    test_traits = ["verbosity", "formality"]  # Just two traits
    
    try:
        collection = extract_all_vectors(test_models, test_traits, "test_vectors")
        
        logger.info(f"✓ Collection created with {len(collection.vectors)} vectors")
        
        # Test matrix retrieval
        for model in test_models:
            matrix, traits = collection.get_vector_matrix(model)
            logger.info(f"  {model}: {matrix.shape} matrix for traits {traits}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Collection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("=== Vector Extraction System Test ===")
    
    tests = [
        ("Prompt Generation", test_prompt_generation),
        ("Model Loader", test_model_loader),
        ("Vector Extraction", test_vector_extraction),
        ("Vector Collection", test_vector_collection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n=== Test Results ===")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())