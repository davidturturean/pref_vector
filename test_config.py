#!/usr/bin/env python3
"""
Simple test script to verify our config fix works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_config_import():
    """Test that our config imports work correctly."""
    try:
        from src.config import EXPERIMENT_CONFIG, get_model_config
        print("✅ Config imports successful!")
        
        # Test the factory function
        model_config = get_model_config("microsoft/DialoGPT-medium")
        print(f"✅ Factory function works! Model: {model_config.model_name}")
        print(f"   Device: {model_config.device}")
        print(f"   Dtype: {model_config.torch_dtype}")
        
        # Test experiment config
        print(f"✅ Experiment config loaded!")
        print(f"   Source model: {EXPERIMENT_CONFIG.source_model}")
        print(f"   Target models: {EXPERIMENT_CONFIG.target_models}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_imports():
    """Test basic imports without visualization dependencies."""
    try:
        from src.data_preparation import SummaryPair
        print("✅ Data preparation imports work!")
        
        from src.vector_extraction import PreferenceVectorExtractor
        print("✅ Vector extraction imports work!")
        
        from src.vector_injection import SteerableModel
        print("✅ Vector injection imports work!")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Preference Vector Transfer Configuration")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing config imports...")
    success &= test_config_import()
    
    print("\n2. Testing basic module imports...")
    success &= test_basic_imports()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The config fix works correctly.")
        print("\nNext steps:")
        print("1. Fix numpy version conflict for full functionality")
        print("2. Run: python test_config.py  # This test")
        print("3. Run: python run_experiment.py --validate  # Full validation")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())