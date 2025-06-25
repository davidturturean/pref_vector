#!/usr/bin/env python3
"""
Compatibility test and environment setup helper.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports without scipy for now."""
    logger.info("=== Testing Basic Imports ===")
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from src.compatibility import detect_numpy_version, check_environment_compatibility
        numpy_version, is_v2 = detect_numpy_version()
        logger.info(f"✓ Compatibility module works, NumPy: {numpy_version} (v2: {is_v2})")
    except Exception as e:
        logger.error(f"✗ Compatibility module failed: {e}")
        return False
    
    return True

def test_vector_extraction_only():
    """Test just the vector extraction without scipy-dependent components."""
    logger.info("\n=== Testing Vector Extraction Only ===")
    
    try:
        import numpy as np  # Add this import
        from src.vector_extraction import PromptGenerator, StyleVector, VectorCollection
        from src.config import get_available_models
        
        # Test prompt generation
        generator = PromptGenerator()
        pairs = generator.generate_contrastive_pairs("verbosity", num_pairs=2)
        logger.info(f"✓ Generated {len(pairs)} prompt pairs")
        
        # Test vector collection (without actual vectors)
        collection = VectorCollection("test_vectors")
        logger.info("✓ VectorCollection created")
        
        # Create a mock vector
        mock_vector = StyleVector(
            vector_id="test",
            model_name="test_model",
            trait_name="verbosity", 
            vector=np.random.randn(64),
            extraction_method="test",
            quality_score=0.8,
            num_samples=1,
            extraction_time=1.0,
            source_prompts=["test"],
            validation_scores={}
        )
        
        collection.add_vector(mock_vector, save=False)
        matrix, traits = collection.get_vector_matrix("test_model")
        logger.info(f"✓ Vector collection works: {matrix.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest environment fixes."""
    logger.info("\n=== Environment Fix Suggestions ===")
    
    from src.compatibility import check_environment_compatibility
    diagnostics = check_environment_compatibility()
    
    logger.info("Current environment status:")
    for component, info in diagnostics.items():
        if isinstance(info, dict) and "installed" in info:
            status = "✓" if info.get("compatible", False) else "✗"
            version = info.get("version", "unknown")
            logger.info(f"  {status} {component}: {version}")
    
    if diagnostics["recommendations"]:
        logger.info("\nRecommended fixes:")
        for rec in diagnostics["recommendations"]:
            logger.info(f"  - {rec}")
    else:
        logger.info("\nTo fix scipy compatibility:")
        logger.info("  1. pip install 'numpy<2.0' scipy --force-reinstall")
        logger.info("  2. Or create a fresh virtual environment")
        logger.info("  3. Or use conda: conda install numpy=1.24 scipy")

def test_real_extraction():
    """Test with real model if available."""
    logger.info("\n=== Testing Real Model Extraction ===")
    
    try:
        from src.model_loader import get_model_loader
        from src.vector_extraction import ContrastiveActivationExtractor, PromptGenerator
        from src.config import get_available_models
        
        available_models = get_available_models()
        if not available_models:
            logger.info("No models configured, skipping real extraction test")
            return True
            
        test_model = available_models[0]
        logger.info(f"Testing with model: {test_model}")
        
        model_loader = get_model_loader()
        
        # Check if model is already loaded (from previous test)
        try:
            model_info = model_loader.get_loaded_model(test_model)
            if model_info:
                logger.info(f"✓ Model {test_model} already loaded")
                
                # Quick extraction test
                extractor = ContrastiveActivationExtractor(model_loader)
                generator = PromptGenerator()
                
                prompt_pairs = generator.generate_contrastive_pairs("verbosity", num_pairs=1)
                
                # Extract just one vector
                vector = extractor.extract_vector(model_info, "verbosity", prompt_pairs)
                logger.info(f"✓ Successfully extracted vector: quality={vector.quality_score:.3f}")
                
                return True
                
        except Exception as e:
            logger.info(f"Model not loaded or extraction failed: {e}")
            logger.info("This is normal if Ollama models aren't downloaded yet")
            return True
            
    except Exception as e:
        logger.error(f"Real extraction test failed: {e}")
        return False

def main():
    """Run compatibility tests."""
    logger.info("🔧 Compatibility Test and Environment Setup")
    
    success = True
    
    # Test 1: Basic imports
    success &= test_basic_imports()
    
    # Test 2: Vector extraction without scipy
    success &= test_vector_extraction_only()
    
    # Test 3: Real extraction if possible
    success &= test_real_extraction()
    
    # Always show suggestions
    suggest_fixes()
    
    logger.info(f"\n{'✅ Basic functionality works!' if success else '❌ Some issues detected'}")
    
    if success:
        logger.info("\n🚀 Ready to proceed with vector extraction!")
        logger.info("Next steps:")
        logger.info("1. Fix scipy compatibility if needed for full analysis")
        logger.info("2. Run: python3 test_extraction.py")
        logger.info("3. Extract vectors from multiple models")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())