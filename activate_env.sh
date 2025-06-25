#!/bin/bash
# Activation script for Platonic Hypothesis Analysis project

echo "🚀 Activating Platonic Analysis Environment..."

# Check if virtual environment exists
if [ ! -d "pref_vector_env" ]; then
    echo "❌ Virtual environment not found. Creating it..."
    python3.9 -m venv pref_vector_env
    echo "✅ Virtual environment created."
fi

# Activate the environment
source pref_vector_env/bin/activate

# Check if packages are installed
python -c "import numpy, scipy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip install 'numpy<2.0' scipy scikit-learn requests matplotlib seaborn plotly tqdm python-dotenv
    echo "✅ Dependencies installed."
fi

echo "✅ Environment ready!"
echo "💡 Current Python: $(which python)"
echo "💡 To deactivate later: deactivate"

# Check Ollama status
if command -v ollama &> /dev/null; then
    echo "✅ Ollama available"
else
    echo "⚠️  Ollama not found - install from https://ollama.ai"
fi