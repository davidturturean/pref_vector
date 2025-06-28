#!/usr/bin/env python3
import os
import requests
import sys
from pathlib import Path

os.environ['OLLAMA_HOST'] = '127.0.0.1:11435'
os.environ['OLLAMA_HOME'] = '/mnt/align4_drive2/davidct/.ollama'

print("=== Environment ===")
print(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST')}")
print(f"OLLAMA_HOME: {os.environ.get('OLLAMA_HOME')}")

print("\n=== Direct API Call ===")
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

print("\n=== Host Construction ===")
sys.path.insert(0, str(Path(__file__).parent / "src"))

ollama_host = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
print(f"Raw OLLAMA_HOST: {ollama_host}")

if not ollama_host.startswith('http'):
    ollama_host = f"http://{ollama_host}"
print(f"Final host: {ollama_host}")

print("\n=== ModelLoader Test ===")
try:
    from src.model_loader import OllamaModelManager
    
    print("Creating OllamaModelManager...")
    manager = OllamaModelManager()
    print(f"Manager host: {manager.host}")
    
    models = manager.list_available_models()
    print(f"Available models: {models}")
    
except Exception as e:
    print(f"ModelLoader failed: {e}")
    import traceback
    traceback.print_exc()