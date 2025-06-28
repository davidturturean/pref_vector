#!/usr/bin/env python3
import os
import sys
import requests

# Set the environment variable
os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'

print("=== Testing Fixed Connection ===")
print(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")

# Test 1: Direct API call
print("\n1. Testing direct API call...")
try:
    response = requests.get('http://127.0.0.1:11435/api/tags', timeout=5)
    if response.status_code == 200:
        models = response.json()['models']
        print(f"✓ API works! Found {len(models)} models: {[m['name'] for m in models]}")
    else:
        print(f"✗ API failed with status: {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"✗ API call failed: {e}")
    sys.exit(1)

# Test 2: Import model loader as a module
print("\n2. Testing model loader import...")
try:
    # Add project root to path so relative imports work
    sys.path.insert(0, os.getcwd())
    
    # Import the fixed model loader
    from src.model_loader import OllamaModelManager, get_model_loader
    print("✓ ModelLoader imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create manager and test connection
print("\n3. Testing OllamaModelManager...")
try:
    manager = OllamaModelManager()
    print(f"✓ Manager created with host: {manager.host}")
    
    # Test listing models
    models = manager.list_available_models()
    print(f"✓ Found models: {models}")
    
    if not models:
        print("⚠ Warning: No models found, but connection works")
    
except Exception as e:
    print(f"✗ Manager test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test model loading
print("\n4. Testing model loading...")
try:
    if models:
        test_model = models[0]
        print(f"Attempting to load model: {test_model}")
        
        # Load using exact Ollama name
        loader = get_model_loader()
        model_info = loader.load_model(test_model)
        print(f"✓ Model loaded: {model_info.hf_name}")
        
        # Test generation
        response = loader.generate_text(test_model, "Hello", options={"num_predict": 10})
        if response:
            print(f"✓ Generation test passed: {response[:50]}...")
        else:
            print("⚠ Generation returned empty response")
    else:
        print("⚠ Skipping model loading test - no models available")
        
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Connection test completed!")
print("If you see this message, the cluster connection is working!")
print("You can now run: python scripts/extract_vectors_batch.py")