# Cluster Testing Instructions

## Step 1: Pull Latest Changes
```bash
cd /mnt/align4_drive2/davidct/pref_vector
git pull origin main
```

## Step 2: Set Environment Variables
```bash
export OLLAMA_HOST="127.0.0.1:11435"
export OLLAMA_HOME="/mnt/align4_drive2/davidct/.ollama"
```

## Step 3: Activate Conda Environment
```bash
conda activate platonic_analysis
```

## Step 4: Test Connection Step by Step

### 4a: Test Direct Connection
```bash
python debug_connection.py
```
This should show:
- Environment variables
- Direct API call success
- Host construction logic
- ModelLoader creation

### 4b: Test Simple Extraction
```bash
python test_cluster_debug.py
```
This tests:
- Model loader creation
- Model listing
- Single model loading
- Simple generation

### 4c: Test Vector Extraction
```bash
python extract_cluster_vectors.py
```
This runs actual vector extraction using cluster models.

## Step 5: If Tests Pass - Run Full Extraction

### 5a: Fixed Batch Script (won't crash on empty collections)
```bash
python scripts/extract_vectors_batch.py
```

### 5b: Or the cluster-specific version
```bash
python extract_cluster_vectors.py
```

## Troubleshooting

### If "Cannot connect to Ollama server" error:
1. Check Ollama is running: `curl http://127.0.0.1:11435/api/tags`
2. Check environment variables: `echo $OLLAMA_HOST`
3. Try restarting Ollama with custom port: `OLLAMA_HOST=127.0.0.1:11435 ollama serve`

### If "No models available" error:
1. List models directly: `ollama list`
2. Check models in API: `curl http://127.0.0.1:11435/api/tags`
3. If models missing, pull them: `ollama pull mistral:7b-instruct`

### If import errors:
1. Check conda environment: `conda list | grep -E "(numpy|scipy|requests)"`
2. Reinstall if needed: `pip install requests numpy scipy`

## Expected Output

Successful extraction should show:
```
=== Cluster Vector Extraction ===
OLLAMA_HOST: 127.0.0.1:11435
Model loader host: http://127.0.0.1:11435
Available models: ['mistral:7b-instruct', 'gemma:7b-instruct', 'llama2:7b-chat']

--- Processing model: mistral:7b-instruct ---
✓ Model loaded: mistral:7b-instruct
Extracting verbosity vector...
✓ verbosity extracted: quality=0.742
...

=== Extraction Complete ===
✓ Successfully extracted: 12 vectors
```