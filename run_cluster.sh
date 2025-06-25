#!/bin/bash
#SBATCH --job-name=pref_vector_ollama
#SBATCH -p sched_mit_mki_r8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Load modules (skip if not available)
# module load python/3.9
# module load cuda/11.8

# Set up environment variables
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_MODELS="/tmp/${USER}_ollama_models"

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p ${OLLAMA_MODELS}

# Navigate to project directory
cd /home/davidct/pref_vector

# Create clean virtual environment
python -m venv ~/experiment_env --clear
source ~/experiment_env/bin/activate

# Install minimal Python dependencies (no more transformers hell!)
pip install --upgrade pip setuptools wheel
pip install requests==2.31.0  # For Ollama API calls
pip install numpy==1.24.3
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install datasets==2.14.0  # HuggingFace datasets
pip install matplotlib==3.7.0
pip install seaborn==0.12.0
pip install pandas==2.0.0
pip install plotly==5.17.0
pip install scipy==1.10.0
pip install rouge-score==0.1.2
pip install nltk==3.8.1
pip install scikit-learn==1.3.0
pip install python-dotenv==1.0.0

# Install Ollama without sudo
echo "=== Installing Ollama ==="
mkdir -p ~/.local/bin
curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64 -o ~/.local/bin/ollama
chmod +x ~/.local/bin/ollama
export PATH="$HOME/.local/bin:$PATH"

# Start Ollama server
echo "=== Starting Ollama Server ==="
ollama serve &
OLLAMA_PID=$!

# Wait for server and download models
sleep 15
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN="${HF_TOKEN}"  # Set this environment variable

# Download models
echo "=== Downloading Models ==="
ollama pull mistral:7b-instruct
ollama pull gemma:7b
ollama pull llama2:7b-chat

# Set up Python environment
source ~/experiment_env/bin/activate
export PYTHONPATH="/home/davidct/pref_vector:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"

# Verify critical dependencies
echo "=== Dependency Versions ==="
pip list | grep -E "(requests|numpy|matplotlib|scipy)" | head -10

# Ensure virtual environment is active and download NLTK data
source ~/experiment_env/bin/activate

python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt', download_dir='/tmp/${USER}_nltk_data')
nltk.download('stopwords', download_dir='/tmp/${USER}_nltk_data')
"

# Set NLTK data path
export NLTK_DATA="/tmp/${USER}_nltk_data"

# Print system info
echo "=== System Information ==="
echo "Virtual env: $VIRTUAL_ENV"
echo "Ollama models: $(ollama list)"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "=========================="

# Run the experiment with virtual environment active
echo "Starting preference vector transfer experiment..."
source ~/experiment_env/bin/activate && python run_experiment.py --full

# Clean up Ollama server
echo "=== Cleaning up ==="
kill $OLLAMA_PID 2>/dev/null || true

echo "Experiment completed!"
echo "Check results in: results/"