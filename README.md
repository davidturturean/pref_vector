# Preference Vector Transfer (Ollama-Powered)

> **🚀 Now powered by Ollama for universal model compatibility!**
> 
> **✅ No more dependency hell**  
> **✅ Works with any model**  
> **✅ Zero version conflicts**

## Overview

This project investigates the transferability of preference vectors (behavioral patterns) across different large language model architectures. Using Ollama, we can test whether style preferences like "concise vs verbose" learned from one model (e.g., Mistral-7B) can be applied to different models (e.g., Gemma-7B, LLaMA-2) to achieve similar behavioral changes.

### Key Research Questions
- Can behavioral preferences transfer across different model architectures?
- Which style dimensions (verbosity, formality, technical complexity) transfer most successfully?
- How effective are different adaptation strategies for cross-model transfer?

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed
- GPU access (recommended)

### Installation

1. **Install Ollama**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. **Clone and setup**:
```bash
git clone <repository>
cd pref_vector
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

3. **Download models**:
```bash
ollama pull mistral:7b-instruct
ollama pull gemma:7b
ollama pull llama2:7b-chat
```

### Running Experiments

#### Local Testing
```bash
python run_experiment.py --quick-test
```

#### Full Experiment
```bash
python run_experiment.py --full
```

#### Cluster Deployment
```bash
./sync_to_cluster.sh
ssh cluster && sbatch run_cluster.sh
```

## Architecture

### Ollama-Based Approach
Unlike traditional activation patching, this project uses **behavioral analysis**:

1. **Vector Extraction**: Analyze output differences across style prompts
2. **Behavioral Signatures**: Capture patterns in length, vocabulary, certainty markers
3. **Cross-Model Transfer**: Apply behavioral patterns via prompt engineering
4. **Adaptation Strategies**: Optimize transfer effectiveness per model

### Key Components

```
src/
├── ollama_utils.py              # Ollama client and model management
├── ollama_vector_extraction.py  # Behavioral vector extraction
├── ollama_vector_injection.py   # Prompt-based style injection
├── experiment_pipeline.py       # Full experiment orchestration
├── data_preparation.py          # Dataset generation
└── evaluation_metrics.py        # Transfer success measurement
```

### Supported Models

All models available in Ollama are supported automatically:
- **Mistral**: `mistral:7b-instruct`
- **Gemma**: `gemma:7b`, `gemma:2b`
- **LLaMA**: `llama2:7b-chat`, `llama2:13b-chat`
- **Qwen**: `qwen:7b`, `qwen:14b`
- **Many more**: Any model in Ollama registry

## Configuration

### Model Mapping
Models are automatically mapped from HuggingFace names to Ollama names:

```python
HF_TO_OLLAMA_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral:7b-instruct",
    "google/gemma-7b": "gemma:7b",
    "meta-llama/Llama-2-7b-chat-hf": "llama2:7b-chat",
    "Qwen/Qwen-7B": "qwen:7b"
}
```

### Experiment Settings
Configure in `src/config.py`:

```python
@dataclass
class ExperimentConfig:
    source_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    target_models: List[str] = ["google/gemma-7b", "meta-llama/Llama-2-7b-chat-hf"]
    num_training_pairs: int = 100
    num_eval_samples: int = 50
```

## Style Dimensions

The system tests transfer across multiple preference dimensions:

### 1. **Verbosity**
- **Concise**: Brief, to-the-point responses
- **Verbose**: Detailed, comprehensive explanations

### 2. **Formality** 
- **Casual**: Conversational, informal tone
- **Formal**: Professional, academic style

### 3. **Technical Complexity**
- **Simple**: Basic terminology, accessible language
- **Technical**: Expert-level, specialized vocabulary

### 4. **Certainty**
- **Uncertain**: Hedged language, acknowledges limitations
- **Confident**: Definitive statements, assertive tone

## Transfer Methods

### 1. **Prompt Engineering**
Inject style preferences through instruction prompts:
```
"Be detailed and comprehensive in your response. [Original Prompt]"
```

### 2. **Example-Based**
Use few-shot examples showing desired style:
```
Prompt: What is photosynthesis?
Response: [Verbose example]

Prompt: [Your question]
Response:
```

### 3. **Style Transfer**
Post-generation rewriting:
```
"Rewrite this response to be more formal while maintaining the same content: [Original Response]"
```

## Results and Evaluation

### Metrics
- **Success Score**: 0-1 rating of transfer effectiveness
- **Style Consistency**: Maintenance of intended style
- **Content Preservation**: Factual accuracy retention
- **Cross-Model Alignment**: Transferability across architectures

### Output Files
- `results/experiment_results.json`: Complete numerical results
- `results/summary_report.md`: Human-readable findings
- `results/[style]_vector.json`: Extracted behavioral vectors
- `logs/slurm_*.out`: Execution logs

### Expected Findings
- **High Transfer**: Verbosity and formality (surface-level linguistic patterns)
- **Moderate Transfer**: Technical complexity (domain-dependent vocabulary)
- **Low Transfer**: Certainty (model-specific calibration and training)

## Cluster Deployment

### MIT Engaging Cluster
```bash
# Sync code
./sync_to_cluster.sh

# SSH and submit
ssh davidct@eofe7.mit.edu
cd pref_vector
sbatch run_cluster.sh

# Monitor progress
squeue -u davidct
tail -f logs/slurm_*.out
```

### SLURM Configuration
```bash
#SBATCH --job-name=pref_vector_ollama
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
```

## Troubleshooting

### Common Issues

**Ollama not found**:
```bash
ollama --version
# If not installed: curl -fsSL https://ollama.ai/install.sh | sh
```

**Model not available**:
```bash
ollama list
ollama pull mistral:7b-instruct
```

**Port conflicts**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
# Kill existing: pkill ollama
# Restart: ollama serve
```

**Memory issues**:
- Reduce `num_training_pairs` in config
- Use smaller models (`gemma:2b` instead of `gemma:7b`)
- Run with `--quick-test` flag

### Debug Mode
```bash
# Verbose logging
export OLLAMA_DEBUG=1
python run_experiment.py --validate

# Check Ollama status
python -c "from src.ollama_utils import check_ollama_status; print(check_ollama_status())"
```

## Research Applications

### Academic Research
- Cross-model behavioral analysis
- Style transfer in large language models
- Prompt engineering optimization
- Model architecture comparisons

### Practical Applications
- Multi-model style consistency
- Automated content adaptation
- Model-agnostic preference control
- Production deployment strategies

## Migration from Transformers

For users upgrading from the transformers-based version:

### What Changed
- **Vector representation**: Behavioral signatures instead of activation vectors
- **Injection method**: Prompt engineering instead of activation patching
- **Model loading**: Ollama API instead of transformers library
- **Dependencies**: Minimal Python packages instead of complex ML stack

### What Stayed the Same
- **Research questions**: Same transferability investigations
- **Evaluation metrics**: Same success measurement criteria
- **Experimental design**: Same multi-model, multi-style testing
- **Output format**: Same results structure and reporting

See `OLLAMA_MIGRATION.md` for detailed migration guide.

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
```

### Adding New Models
1. Add to `HF_TO_OLLAMA_MODELS` in `src/config.py`
2. Test with `ollama pull <model-name>`
3. Update target_models in experiment config

### Adding New Style Dimensions
1. Add to `style_prompts` in `src/ollama_vector_extraction.py`
2. Implement analysis method in `_analyze_behavioral_differences`
3. Add evaluation logic in `src/ollama_vector_injection.py`

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@misc{preference_vector_transfer_2025,
  title={Cross-Model Preference Vector Transfer via Behavioral Analysis},
  author={[Your Name]},
  year={2025},
  note={Ollama-powered implementation for universal model compatibility}
}
```

---

**Need help?** Check the troubleshooting section or open an issue with your specific error message and system details.