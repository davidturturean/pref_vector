#!/usr/bin/env python3
"""
Test Ollama connection for cluster debugging
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_direct_api():
    """Test direct API connection to Ollama"""
    print("=== Testing Direct API Connection ===")
    
    # Get Ollama host
    ollama_host = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
    print(f"OLLAMA_HOST: {ollama_host}")
    
    # Test API endpoint
    url = f"http://{ollama_host}/api/tags"
    try:
        response = requests.get(url, timeout=10)
        print(f"API Response Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"Available models: {models}")
            return True
        else:
            print(f"API Error: {response.text}")
            return False
    except Exception as e:
        print(f"API Connection failed: {e}")
        return False

def test_model_loader():
    """Test ModelLoader connection"""
    print("\n=== Testing ModelLoader Connection ===")
    
    try:
        from src.model_loader import get_model_loader
        
        # Force the custom port
        os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
        
        loader = get_model_loader()
        print(f"ModelLoader created successfully")
        
        # Test loading a model
        try:
            model_info = loader.load_model("mistral:7b-instruct")
            print(f"✓ Model loaded: {model_info.hf_name}")
            return True
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ ModelLoader import failed: {e}")
        return False

def test_generation():
    """Test text generation"""
    print("\n=== Testing Text Generation ===")
    
    try:
        from src.model_loader import get_model_loader
        
        # Force the custom port
        os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
        
        loader = get_model_loader()
        
        # Test generation
        response = loader.generate_text(
            "mistral:7b-instruct", 
            "What is machine learning?",
            options={"temperature": 0.1, "num_predict": 50}
        )
        
        if response:
            print(f"✓ Generation successful: {response[:100]}...")
            return True
        else:
            print("✗ Generation returned empty response")
            return False
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Ollama Connection Test")
    print("=" * 50)
    
    results = []
    results.append(("Direct API", test_direct_api()))
    results.append(("ModelLoader", test_model_loader()))
    results.append(("Generation", test_generation()))
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:15} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ollama connection is working.")
    else:
        print("\n⚠️  Some tests failed. Check connection settings.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())