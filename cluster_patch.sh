#!/bin/bash
# Automatic cluster file patcher
# This script will fix the model_loader.py file to work with OLLAMA_HOST

echo "Patching src/model_loader.py for cluster Ollama connection..."

# Create backup
cp src/model_loader.py src/model_loader.py.backup

# Apply the fix using sed
sed -i 's/def __init__(self, host: str = "http:\/\/localhost:11434"):/def __init__(self, host: str = None):/' src/model_loader.py

# Add the environment variable logic after the __init__ line
sed -i '/def __init__(self, host: str = None):/a\        # Use environment variable if set\n        if host is None:\n            ollama_host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")\n            if not ollama_host.startswith("http"):\n                ollama_host = f"http://{ollama_host}"\n            host = ollama_host' src/model_loader.py

# Add import os if not present
if ! grep -q "import os" src/model_loader.py; then
    sed -i '/import time/i import os' src/model_loader.py
fi

echo "Patch applied! Testing connection..."

# Test the fix
export OLLAMA_HOST="127.0.0.1:11435"
python3 -c "
import sys
sys.path.insert(0, 'src')
from model_loader import OllamaModelManager
print('Creating manager...')
manager = OllamaModelManager()
print(f'Manager host: {manager.host}')
models = manager.list_available_models()
print(f'Available models: {models}')
if models:
    print('SUCCESS: Cluster connection working!')
else:
    print('WARNING: Connected but no models found')
"