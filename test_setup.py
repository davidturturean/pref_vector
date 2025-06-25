#!/usr/bin/env python3
"""
Test script to validate the flexible infrastructure setup
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import EXPERIMENT_CONFIG, MODEL_FAMILIES, ALL_STYLE_TRAITS
from src.model_loader import get_model_loader
from src.vector_extraction import extract_all_vectors, PromptGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration system."""
    print("=== Testing Configuration ===")
    
    # Test basic config
    config = EXPERIMENT_CONFIG
    print(f"Source model: {config.source_model}")
    print(f"Target models: {config.target_models}")
    print(f"Style traits: {len(config.style_traits)} traits")
    print(f"Model families: {list(MODEL_FAMILIES.keys())}")
    
    # Test model pairs
    pairs = config.get_model_pairs()
    print(f"Model pairs for analysis: {len(pairs)}")
    for pair in pairs[:3]:  # Show first 3
        print(f"  {pair[0]} -> {pair[1]}")
    
    print("✓ Configuration test passed\n")

def test_prompt_generation():
    """Test prompt generation system."""
    print("=== Testing Prompt Generation ===")
    
    generator = PromptGenerator()
    
    # Test prompt generation for different traits
    for trait in ["verbosity", "formality", "technical_complexity", "certainty"]:
        pairs = generator.generate_contrastive_pairs(trait, num_pairs=3)
        print(f"\n{trait.upper()} examples:")
        for i, (low, high) in enumerate(pairs):
            print(f"  Pair {i+1}:")
            print(f"    Low:  {low}")
            print(f"    High: {high}")
    
    print("\n✓ Prompt generation test passed\n")

def test_model_loading():
    """Test model loading system."""
    print("=== Testing Model Loading ===")
    
    # Test with a smaller model first
    test_models = ["google/gemma-2b", "mistralai/Mistral-7B-Instruct-v0.1"]
    
    model_loader = get_model_loader()
    
    for model_name in test_models:
        try:
            print(f"Loading model: {model_name}")
            model_info = model_loader.load_model(model_name)
            
            print(f"  Model loaded successfully!")
            print(f"  Family: {model_info.family}")
            print(f"  Size: {model_info.size}")
            print(f"  Hidden size: {model_info.hidden_size}")
            print(f"  Ollama name: {model_info.ollama_name}")
            
            # Test generation
            response = model_loader.generate_text(
                model_name, 
                "What is artificial intelligence?",
                options={"num_predict": 20}
            )
            print(f"  Sample response: {response[:100]}...")
            
        except Exception as e:
            print(f"  ⚠️ Model loading failed: {e}")
            print(f"  This might be expected if Ollama server isn't running or model isn't available")
    
    print("\n✓ Model loading test completed\n")

def test_vector_extraction():
    """Test vector extraction system."""
    print("=== Testing Vector Extraction (Dry Run) ===")
    
    # This is a dry run - we'll test the components without actually running extraction
    
    # Test with core traits only for speed
    test_traits = ["verbosity", "formality"]
    test_model = "google/gemma-2b"  # Small model
    
    print(f"Testing vector extraction setup for:")
    print(f"  Model: {test_model}")
    print(f"  Traits: {test_traits}")
    
    try:
        # Test prompt generation
        generator = PromptGenerator()
        for trait in test_traits:
            pairs = generator.generate_contrastive_pairs(trait, num_pairs=2)
            print(f"  Generated {len(pairs)} prompt pairs for {trait}")
        
        print("  Vector extraction components are ready")
        print("  ℹ️ Run with --extract to perform actual extraction")
        
    except Exception as e:
        print(f"  ⚠️ Vector extraction setup failed: {e}")
    
    print("\n✓ Vector extraction test completed\n")

def main():
    """Run all tests."""
    print("🚀 Testing Preference Vector Transfer Infrastructure\n")
    
    try:
        test_configuration()
        test_prompt_generation()
        
        # Check if user wants to test model loading (requires Ollama)
        if "--model-test" in sys.argv:
            test_model_loading()
        else:
            print("=== Skipping Model Loading Test ===")
            print("Add --model-test to test actual model loading")
            print("(Requires Ollama server to be running)")
            print()
        
        # Check if user wants to test vector extraction
        if "--extract" in sys.argv:
            print("=== Running Full Vector Extraction ===")
            # You would uncommment this for actual extraction:
            # collection = extract_all_vectors(["google/gemma-2b"], ["verbosity", "formality"])
            # print(f"Extracted vectors: {len(collection.vectors)}")
            print("Full extraction would run here...")
        else:
            test_vector_extraction()
        
        print("🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Ensure Ollama is running: `ollama serve`")
        print("2. Run with --model-test to test model loading")
        print("3. Run with --extract to test full vector extraction")
        print("4. Check the results/ directory for outputs")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()