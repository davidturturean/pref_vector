#!/bin/bash
# Robust Environment Setup for Platonic Hypothesis Analysis
# Works on laptop, cluster, and any Unix system

set -e  # Exit on any error

echo "🚀 Setting up Platonic Analysis Environment..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Miniconda if needed
install_miniconda() {
    echo "📦 Installing Miniconda..."
    
    # Detect OS and architecture
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    else
        echo "❌ Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    # Download and install
    wget -O miniconda.sh "$MINICONDA_URL"
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    
    # Add to PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
    
    echo "✅ Miniconda installed"
}

# Check if conda is available
if ! command_exists conda; then
    echo "⚠️  Conda not found. Installing Miniconda..."
    install_miniconda
    
    # Reload PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    if ! command_exists conda; then
        echo "❌ Conda installation failed"
        exit 1
    fi
fi

echo "✅ Conda available: $(conda --version)"

# Initialize conda for shell
conda init bash 2>/dev/null || true
conda init zsh 2>/dev/null || true

# Create/update environment from yml file
echo "🏗️  Setting up conda environment..."

if conda env list | grep -q "platonic_analysis"; then
    echo "📦 Updating existing environment..."
    conda env update -f environment.yml
else
    echo "📦 Creating new environment..."
    conda env create -f environment.yml
fi

echo "✅ Environment ready!"

# Create activation script
cat > activate_project.sh << 'EOF'
#!/bin/bash
# Project activation script

# Ensure conda is in PATH
if [[ -d "$HOME/miniconda3/bin" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

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
EOF

chmod +x activate_project.sh

echo ""
echo "🎯 Setup complete! To use the project:"
echo "   source activate_project.sh"
echo "   python test_full_pipeline.py"
echo ""
echo "💡 This works on laptop, cluster, and any Unix system!"