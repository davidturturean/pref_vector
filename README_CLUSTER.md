# Running on MIT Engaging Cluster

This guide helps you run the preference vector transfer experiments on the MIT Engaging cluster with minimal setup.

## Quick Start

### 1. Initial Setup (Run Once)

**On your local machine:**
```bash
# Run the setup script to configure HuggingFace auth
python setup_cluster.py
```

**Copy project to cluster:**
```bash
# Use the automated upload script (uploads only essential files)
./upload_to_cluster.sh [your_username]

# Or manual upload (replace 'username' with your cluster username)
rsync -av --exclude='pref_vector_env/' --exclude='__pycache__/' --exclude='*.pyc' \
  . username@eofe7.mit.edu:/home/davidct/pref_vector/
```

### 2. HuggingFace Token Setup

**⚠️ REQUIRED: You need to provide your HuggingFace token for model access**

**Option A: Environment variable (recommended)**
```bash
# On the cluster, add to your ~/.bashrc:
export HF_TOKEN="hf_your_token_here"
```

**Option B: During setup**
- The `setup_cluster.py` script will prompt for your token
- Get your token from: https://huggingface.co/settings/tokens
- Make sure it has 'Read' permissions

### 3. Submit Job

**On the cluster:**
```bash
cd /home/davidct/pref_vector
python setup_cluster.py  # Run setup on cluster too
./submit_job.sh /home/davidct/pref_vector
```

### 4. Monitor Job

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/slurm_*.out

# View error logs
tail -f logs/slurm_*.err
```

## File Overview

- **`run_cluster.sh`**: SLURM job script with GPU allocation
- **`setup_cluster.py`**: One-time setup for HuggingFace auth
- **`submit_job.sh`**: Easy job submission script
- **`cluster_utils.py`**: Cluster-specific optimizations
- **`requirements_cluster.txt`**: Cluster-specific dependencies

## Key Configurations

### Resource Allocation
- **GPU**: 1 GPU (modify `#SBATCH --gres=gpu:1` if needed)
- **CPUs**: 8 cores
- **Memory**: 64GB RAM
- **Time**: 12 hours (adjust `#SBATCH --time=12:00:00`)

### Model Settings
- Uses GPU with mixed precision for speed
- Automatic memory optimization based on available GPU memory
- Fallback to CPU if GPU unavailable

## Customization

### Change Resource Requirements

Edit `run_cluster.sh`:
```bash
#SBATCH --gres=gpu:2        # Request 2 GPUs
#SBATCH --mem=128G          # Request 128GB RAM
#SBATCH --time=24:00:00     # Run for 24 hours
```

### Change Experiment Parameters

Edit the run command in `run_cluster.sh`:
```bash
python run_experiment.py --full \
    --num-pairs 50 \           # Generate 50 training pairs
    --eval-samples 25 \        # Use 25 evaluation samples
    --source-model "mistralai/Mistral-7B-Instruct-v0.1" \
    --target-models "google/gemma-7b" "meta-llama/Llama-2-7b-hf"
```

### Use Different Models

Modify `src/config.py` or pass arguments:
```bash
python run_experiment.py --full \
    --source-model "microsoft/DialoGPT-medium" \  # Smaller for testing
    --target-models "google/gemma-2b"
```

## Troubleshooting

### Common Issues

1. **HuggingFace Authentication Error**
   ```bash
   # Set token as environment variable
   export HF_TOKEN="hf_your_token_here"
   ```

2. **Out of Memory Error**
   ```bash
   # Use smaller models or reduce batch size
   python run_experiment.py --quick-test --num-pairs 5
   ```

3. **Module Not Found**
   ```bash
   # The job script installs dependencies automatically
   # If issues persist, check the SLURM output logs
   ```

4. **Job Pending Too Long**
   ```bash
   # Check queue status
   squeue
   # Try different partition if available
   # Edit run_cluster.sh: #SBATCH -p different_partition
   ```

### Performance Optimization

1. **For faster iteration**: Use smaller models during development
2. **For full experiments**: Request multiple GPUs if available
3. **For memory efficiency**: Enable 8-bit quantization (automatic based on GPU memory)

## Expected Runtime

- **Quick test** (5 pairs): ~30 minutes on GPU
- **Full experiment** (100 pairs): ~4-8 hours on GPU
- **CPU only**: 10-20x slower (not recommended for large models)

## Output Files

Results will be saved in:
```
results/cluster_exp_YYYYMMDD_HHMMSS/
├── experiment_results.json
├── summary_report.md
├── preference_vectors.json
└── figures/
    ├── transfer_success_matrix.png
    └── evaluation_dashboard.html
```

## Getting Results Back

```bash
# Copy results from cluster to local machine
scp -r username@eofe7.mit.edu:/home/davidct/pref_vector/results/ ./cluster_results/
```