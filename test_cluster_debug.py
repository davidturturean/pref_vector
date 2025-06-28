#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Set environment variables first
os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
os.environ['OLLAMA_HOME'] = '/mnt/align4_drive2/davidct/.ollama'

print(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST', 'not set')}")
print(f"OLLAMA_HOME: {os.environ.get('OLLAMA_HOME', 'not set')}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests

# Test 1: Direct API call
print("\n=== Test 1: Direct API Call ===")
try:
    response = requests.get('http://127.0.0.1:11435/api/tags')
    models = response.json()['models']
    print(f"✅ Direct API call successful: {len(models)} models")
    for model in models:
        print(f"  - {model['name']}")
except Exception as e:
    print(f"❌ Direct API call failed: {e}")

# Test 2: Model loader creation
print("\n=== Test 2: Model Loader Creation ===")
try:
    from src.model_loader import get_model_loader
    loader = get_model_loader()
    print("✅ Model loader created successfully")
    print(f"Ollama host: {loader.ollama_manager.host}")
except Exception as e:
    print(f"❌ Model loader creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: List available models through loader
print("\n=== Test 3: List Models Through Loader ===")
try:
    available_models = loader.ollama_manager.list_available_models()
    print(f"✅ Available models: {available_models}")
except Exception as e:
    print(f"❌ Failed to list models: {e}")

# Test 4: Try to load model with exact Ollama name
print("\n=== Test 4: Load Model with Ollama Name ===")
try:
    # Use the exact model name from Ollama
    model_info = loader.load_model('mistral:7b-instruct')
    print(f"✅ Model loaded: {model_info}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Simple generation test
print("\n=== Test 5: Generation Test ===")
try:
    response = loader.ollama_manager.generate(
        'mistral:7b-instruct',
        'Hello, how are you?',
        options={'num_predict': 20}
    )
    print(f"✅ Generation successful: {response}")
except Exception as e:
    print(f"❌ Generation failed: {e}")