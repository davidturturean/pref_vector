#!/bin/bash
# Project activation script

# Ensure conda is in PATH
if [[ -d "$HOME/miniconda3/bin" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# Activate environment
conda activate platonic_analysis

echo "✅ Platonic Analysis environment activated"
echo "💡 Python: $(which python)"
echo "💡 Packages: numpy=$(python -c 'import numpy; print(numpy.__version__)') scipy=$(python -c 'import scipy; print(scipy.__version__)')"

# Check Ollama
if command -v ollama >/dev/null 2>&1; then
    echo "✅ Ollama available"
else
    echo "⚠️  Ollama not found - install from https://ollama.ai"
fi
