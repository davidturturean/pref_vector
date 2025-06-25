# Environment Setup for Platonic Hypothesis Analysis

## Virtual Environment Setup

This project uses a dedicated virtual environment to avoid conflicts with other projects.

### Initial Setup
```bash
# Create virtual environment (do this once)
python3.9 -m venv pref_vector_env

# Activate environment (do this every time you work on the project)
source pref_vector_env/bin/activate

# Install dependencies
pip install 'numpy<2.0' scipy scikit-learn requests matplotlib seaborn plotly tqdm python-dotenv

# Verify setup
python -c "import numpy, scipy, sklearn; print('Environment ready!')"
```

### Daily Usage
```bash
# Always activate the environment first
source pref_vector_env/bin/activate

# Run project scripts
python test_full_pipeline.py
python scripts/extract_vectors_batch.py

# Deactivate when done
deactivate
```

### Dependencies
- Python 3.9+
- NumPy < 2.0 (for SciPy compatibility)
- SciPy 1.13+
- scikit-learn 1.6+
- Ollama (for model management)

### Troubleshooting
- If imports fail, ensure you're in the virtual environment: `which python` should show the venv path
- If Ollama models fail, ensure Ollama server is running: `ollama serve`
- Never install packages globally - always use the venv