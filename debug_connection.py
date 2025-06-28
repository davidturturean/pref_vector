#!/usr/bin/env python3
import os
import requests
import sys
from pathlib import Path

# Set environment variables
os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
os.environ['OLLAMA_HOME'] = '/mnt/align4_drive2/davidct/.ollama'

print("=== Environment ===")
print(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")
print(f"OLLAMA_HOME: {os.environ.get('OLLAMA_HOME')}")

# Test 1: Direct requests call
print("\n=== Test 1: Direct API Call ===")
url = 'http://127.0.0.1:11435/api/tags'
print(f"Testing URL: {url}")

try:
    response = requests.get(url, timeout=5)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Models: {[m['name'] for m in data['models']]}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")

# Test 2: Check host construction
print("\n=== Test 2: Host Construction ===")
sys.path.insert(0, str(Path(__file__).parent / "src"))

ollama_host = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
print(f"Raw OLLAMA_HOST: {ollama_host}")

if not ollama_host.startswith('http'):
    ollama_host = f"http://{ollama_host}"
print(f"Final host: {ollama_host}")

# Test 3: Test constructed URL
print("\n=== Test 3: Constructed URL ===")
test_url = f"{ollama_host}/api/tags"
print(f"Testing constructed URL: {test_url}")

try:
    response = requests.get(test_url, timeout=5)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ Connection successful with constructed URL")
    else:
        print(f"✗ Error: {response.text}")
except Exception as e:
    print(f"✗ Request failed: {e}")

# Test 4: Try ModelLoader
print("\n=== Test 4: ModelLoader ===")
try:
    from src.model_loader import OllamaModelManager
    
    print("Creating OllamaModelManager...")
    # This should use the environment variable
    manager = OllamaModelManager()
    print(f"Manager host: {manager.host}")
    
    # Try to list models
    models = manager.list_available_models()
    print(f"Available models: {models}")
    
except Exception as e:
    print(f"ModelLoader failed: {e}")
    import traceback
    traceback.print_exc()