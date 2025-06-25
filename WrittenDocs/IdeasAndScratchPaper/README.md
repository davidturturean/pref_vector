# Preference Vector Transfer Between Language Models

This project investigates whether preference vectors learned in one language model can be transferred to other models to achieve similar behavioral changes. The research explores cross-model latent space alignment for high-level preferences like verbosity vs. conciseness in text generation.

## Research Question

**Does a preference vector transfer between models?** 

We investigate whether a latent direction representing "concise vs. verbose" style extracted from one model (e.g., Mistral-7B) can be directly applied to different model architectures (e.g., Gemma-7B, LLaMA-3) to elicit similar behavioral changes.

## Methodology

### 1. Preference Vector Extraction
- Generate paired concise/verbose summaries using the source model
- Extract preference vectors using activation differences at intermediate layers
- Validate the vector's effectiveness on the source model

### 2. Direct Transfer Testing
- Inject the extracted vector into target models at corresponding layers
- Measure behavioral changes in text generation
- Evaluate success using multiple metrics (length, verbosity, content preservation)

### 3. Linear Adapter Training
- When direct transfer fails, train lightweight linear adapters
- Learn mappings between source and target model latent spaces
- Test adapted vectors for improved transfer performance

## Key Components

### Core Modules
- `data_preparation.py`: Generate concise/verbose summary pairs
- `vector_extraction.py`: Extract preference vectors using activation differences
- `vector_injection.py`: Inject vectors into models during inference
- `evaluation_metrics.py`: Comprehensive evaluation framework
- `linear_adapter.py`: Train cross-model alignment adapters

### Experiment Pipeline
- `experiment_pipeline.py`: Orchestrate complete experiments
- `visualization.py`: Generate analysis plots and dashboards
- `run_experiment.py`: Main experiment runner script

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch 2.0+
- Transformers 4.35+
- Standard ML libraries (numpy, sklearn, matplotlib)
- Evaluation libraries (rouge-score, nltk)

### Hardware Requirements
- **Minimum**: 16GB RAM, 8GB GPU memory
- **Recommended**: 32GB RAM, 16GB+ GPU memory
- **Disk**: 50GB+ for models and results

## Usage

### Quick Test
```bash
python run_experiment.py --quick-test
```

### Full Experiment
```bash
python run_experiment.py --full
```

### Custom Configuration
```bash
python run_experiment.py --full \
    --source-model "mistralai/Mistral-7B-v0.1" \
    --target-models "google/gemma-7b" "meta-llama/Llama-2-7b-hf" \
    --num-pairs 100 \
    --intervention-layer 16
```

### Validation Only
```bash
python run_experiment.py --validate
```

### Generate Visualizations
```bash
python run_experiment.py --visualize-only results/experiment_id/
```

## Configuration

### Model Settings
```python
# Edit src/config.py
EXPERIMENT_CONFIG.source_model = "mistralai/Mistral-7B-v0.1"
EXPERIMENT_CONFIG.target_models = ["google/gemma-7b", "meta-llama/Llama-2-7b-hf"]
EXPERIMENT_CONFIG.intervention_layer = 16  # Middle layer
```

### Experiment Parameters
```python
EXPERIMENT_CONFIG.num_training_pairs = 100
EXPERIMENT_CONFIG.num_eval_samples = 50
EXPERIMENT_CONFIG.vector_dim = 4096  # 7B model hidden size
```

## Results Structure

```
results/experiment_id/
├── experiment_results.json      # Complete results data
├── summary_report.md           # Human-readable report
├── preference_vector.json      # Extracted vector
├── summary_pairs/             # Training data
└── figures/                   # Visualizations
    ├── transfer_success_matrix.png
    ├── evaluation_scores_comparison.png
    ├── model_performance_radar.png
    └── experiment_dashboard.html
```

## Evaluation Metrics

### Style Consistency
- **Length Direction Score**: Does steering change length as expected?
- **Verbosity Direction Score**: Does verbosity change appropriately?
- **Overall Consistency**: Combined style transfer effectiveness

### Content Preservation
- **ROUGE Scores**: Overlap with reference content
- **Semantic Similarity**: TF-IDF cosine similarity
- **Fact Preservation**: Named entity retention

### Transfer Success
- **Direct Transfer Rate**: Percentage of successful direct transfers
- **Adapter Improvement**: Performance gain from linear adapters
- **Cross-Model Alignment**: Similarity of vector effects across models

## Expected Outcomes

### Hypothesis 1: Within-Model Effectiveness
**Prediction**: Preference vectors will successfully steer behavior within the source model.
**Measurement**: Source validation score > 0.7

### Hypothesis 2: Direct Cross-Model Transfer
**Prediction**: Direct transfer will have limited success due to representational differences.
**Measurement**: Direct transfer success rate 10-30%

### Hypothesis 3: Linear Adapter Effectiveness
**Prediction**: Linear adapters will improve transfer performance significantly.
**Measurement**: Adapter success rate > Direct transfer rate + 20%

## Scientific Significance

### If Direct Transfer Succeeds
- Implies remarkable universality in preference representations
- Suggests convergent evolution of latent spaces across model families
- Opens possibility of model-agnostic preference alignment

### If Direct Transfer Fails But Adapters Work
- Confirms representational differences are primarily linear transformations
- Demonstrates feasibility of lightweight cross-model alignment
- Validates theoretical framework from word embedding alignment research

### If Both Approaches Fail
- Indicates fundamental representational incompatibilities
- Suggests need for more sophisticated alignment methods
- Provides bounds on cross-model transfer feasibility

## Limitations

- **Task Specificity**: Limited to summarization; may not generalize
- **Model Selection**: Constrained by available open-source models
- **Evaluation Subjectivity**: Metrics may not capture all preference aspects
- **Scale Constraints**: Hardware limits model size and sample size

## Extensions

### Multi-Dimensional Preferences
- Test transfer of multiple preference dimensions simultaneously
- Investigate interaction effects between different preference vectors

### Non-Linear Adapters
- Explore MLPs and other non-linear transformation methods
- Compare with linear adapter baselines

### Bidirectional Transfer
- Test if adapters work in both directions (A→B and B→A)
- Investigate symmetry in representational relationships

### Larger Model Families
- Test on larger models (13B, 30B+) when computationally feasible
- Investigate scaling effects on transfer success

## Contributing

This research framework is designed for extensibility:

1. **New Models**: Add models by updating `config.py`
2. **New Metrics**: Extend `evaluation_metrics.py`
3. **New Adapters**: Implement in `linear_adapter.py`
4. **New Tasks**: Modify `data_preparation.py`

## Citation

If you use this code for research, please cite:

```bibtex
@misc{preference_vector_transfer_2025,
  title={Cross-Model Preference Vector Transfer for Language Model Alignment},
  author={[Your Name]},
  year={2025},
  note={Research implementation for investigating cross-model latent space alignment}
}
```

## Contact

For questions about the research methodology or implementation, please open an issue or contact the research team.

---

*This project represents a systematic investigation into the transferability of high-level behavioral preferences across language model architectures, contributing to our understanding of representational alignment and cross-model transfer learning.*