import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Extended model mappings for comprehensive analysis
HF_TO_OLLAMA_MODELS = {
    # Mistral family
    "mistralai/Mistral-7B-v0.1": "mistral:7b",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral:7b-instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "mixtral:8x7b-instruct",
    
    # Google Gemma family
    "google/gemma-7b": "gemma:7b", 
    "google/gemma-2b": "gemma:2b",
    "google/gemma-7b-it": "gemma:7b-instruct",
    
    # Meta LLaMA family
    "meta-llama/Llama-2-7b-hf": "llama2:7b",
    "meta-llama/Llama-2-13b-hf": "llama2:13b",
    "meta-llama/Llama-2-7b-chat-hf": "llama2:7b-chat",
    "meta-llama/Llama-2-13b-chat-hf": "llama2:13b-chat",
    "meta-llama/Llama-3-8b": "llama3:8b",
    "meta-llama/Llama-3-8b-Instruct": "llama3:8b-instruct",
    
    # Qwen family
    "Qwen/Qwen-7B": "qwen:7b",
    "Qwen/Qwen-14B": "qwen:14b",
    "Qwen/Qwen2-7B": "qwen2:7b",
    
    # Yi family
    "01-ai/Yi-6B": "yi:6b",
    "01-ai/Yi-34B": "yi:34b",
    
    # Microsoft models
    "microsoft/DialoGPT-medium": "phi:2.7b",
    "microsoft/DialoGPT-small": "phi:2.7b",
    "microsoft/phi-2": "phi:2.7b",
    
    # Code models for testing
    "codellama/CodeLlama-7b-hf": "codellama:7b",
    "codellama/CodeLlama-13b-hf": "codellama:13b",
}

# Model family groupings for systematic analysis
MODEL_FAMILIES = {
    "mistral": ["mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    "gemma": ["google/gemma-2b", "google/gemma-7b", "google/gemma-7b-it"],
    "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-3-8b-Instruct"],
    "qwen": ["Qwen/Qwen-7B", "Qwen/Qwen2-7B"],
    "yi": ["01-ai/Yi-6B", "01-ai/Yi-34B"],
    "code": ["codellama/CodeLlama-7b-hf"]
}

# Extended stylistic traits for comprehensive analysis
CORE_STYLE_TRAITS = [
    "verbosity", "formality", "technical_complexity", "certainty"
]

EXTENDED_STYLE_TRAITS = [
    # Pragmatic and social dimensions
    "emotional_tone", "politeness", "assertiveness", "humor", 
    "objectivity", "specificity", "register", "persuasiveness",
    "empathy", "clarity", "creativity", "authority",
    "concreteness", "urgency", "inclusivity", "optimism",
    
    # Additional linguistic dimensions
    "hedging", "directness", "enthusiasm", "professionalism",
    "accessibility", "precision"
]

ALL_STYLE_TRAITS = CORE_STYLE_TRAITS + EXTENDED_STYLE_TRAITS

def get_ollama_model_name(hf_model_name: str) -> str:
    """Convert HuggingFace model name to Ollama model name."""
    return HF_TO_OLLAMA_MODELS.get(hf_model_name, hf_model_name.split('/')[-1].lower())

def get_available_models() -> List[str]:
    """Get list of available models for testing."""
    # Return a subset of models that are commonly available
    return [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "google/gemma-7b-it", 
        "meta-llama/Llama-2-7b-chat-hf"
    ]

@dataclass
class ModelConfig:
    """Configuration for Ollama model loading and inference."""
    model_name: str  # HuggingFace name
    ollama_name: str  # Corresponding Ollama name
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_ctx: int = 4096  # Ollama context window
    num_gpu: int = -1  # Use all available GPUs
    host: str = "http://localhost:11434"  # Ollama API endpoint

def get_model_config(model_name: str, **overrides) -> ModelConfig:
    """Create a ModelConfig instance for a specific model with Ollama defaults."""
    ollama_name = get_ollama_model_name(model_name)
    
    base_config = {
        "model_name": model_name,
        "ollama_name": ollama_name,
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_ctx": 4096,
        "num_gpu": -1,
        "host": "http://localhost:11434"
    }
    
    # Model-specific optimizations based on size
    model_lower = model_name.lower()
    
    # Large models - smaller context for memory efficiency
    if any(size in model_lower for size in ["70b", "72b", "67b", "34b"]):
        base_config["max_length"] = 256
        base_config["num_ctx"] = 2048
    
    # Small models - can handle larger context
    elif any(size in model_lower for size in ["0.5b", "2b", "small", "medium"]):
        base_config["max_length"] = 1024
        base_config["num_ctx"] = 8192
    
    # Apply any user overrides
    base_config.update(overrides)
    
    return ModelConfig(**base_config)

@dataclass
class ExperimentConfig:
    """Configuration for comprehensive preference vector experiments."""
    
    # Model configuration
    source_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    target_models: List[str] = field(default_factory=list)
    model_families_to_test: List[str] = field(default_factory=lambda: ["mistral", "gemma", "llama"])
    include_cross_family_analysis: bool = True
    include_scale_sensitivity: bool = True
    
    # Style trait configuration
    style_traits: List[str] = field(default_factory=lambda: ALL_STYLE_TRAITS)
    core_traits_only: bool = False  # Set to True for quick testing
    trait_extraction_method: str = "contrastive_activation"  # "contrastive_activation", "probe_based"
    
    # Vector extraction settings
    vector_dim: int = 4096  # Hidden dimension for 7B models
    num_contrastive_pairs: int = 10  # Number of prompt pairs per trait
    extraction_layer: str = "final"  # "final", "mid", "early", or specific layer number
    normalize_vectors: bool = True
    vector_quality_threshold: float = 0.3  # Minimum quality score to accept
    
    # Transfer matrix settings
    transfer_method: str = "pseudoinverse"  # "pseudoinverse", "procrustes", "regularized_lstsq"
    regularization_strength: float = 1e-6
    use_orthogonal_constraint: bool = False
    validate_transfer_matrices: bool = True
    
    # Platonic analysis settings
    subspace_variance_threshold: float = 0.95  # Retain components explaining this much variance
    union_basis_variance_threshold: float = 0.90  # For cross-model union basis
    max_subspace_rank: int = 15  # Maximum rank for style subspaces
    principal_angle_method: str = "svd"  # "svd", "qr"
    
    # Clustering and statistical settings
    clustering_method: str = "hierarchical"  # "hierarchical", "kmeans", "spectral"
    clustering_linkage: str = "ward"  # For hierarchical clustering
    statistical_significance_level: float = 0.05
    num_bootstrap_samples: int = 1000
    num_permutation_tests: int = 1000
    
    # Incremental analysis settings
    core_trait_set: List[str] = field(default_factory=lambda: CORE_STYLE_TRAITS)
    incremental_addition_strategy: str = "greedy"  # "greedy", "random", "semantic"
    mse_improvement_threshold: float = 0.01
    
    # Evaluation and validation
    num_evaluation_samples: int = 50
    use_human_evaluation: bool = False  # Set to True if human evaluation is available
    automated_metrics: List[str] = field(default_factory=lambda: ["cosine_similarity", "mse", "success_rate"])
    
    # Technical settings
    batch_size: int = 1  # Due to memory constraints with large models
    use_gpu: bool = True
    low_memory_mode: bool = False
    random_seed: int = 42
    
    # Output and logging
    results_dir: str = "results"
    save_intermediate_results: bool = True
    log_level: str = "INFO"
    create_visualizations: bool = True
    
    def __post_init__(self):
        """Set default values and validate configuration."""
        if not self.target_models:
            # Automatically populate based on selected families
            self.target_models = []
            for family in self.model_families_to_test:
                if family in MODEL_FAMILIES:
                    self.target_models.extend(MODEL_FAMILIES[family])
            
            # Remove source model from targets if present
            if self.source_model in self.target_models:
                self.target_models.remove(self.source_model)
        
        # Use only core traits for quick testing
        if self.core_traits_only:
            self.style_traits = CORE_STYLE_TRAITS.copy()
        
        # Ensure results directory exists
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        if not self.target_models:
            raise ValueError("No target models specified")
        
        if self.source_model in self.target_models:
            raise ValueError("Source model cannot be in target models list")
        
        if not self.style_traits:
            raise ValueError("No style traits specified")
        
        # Validate trait names
        invalid_traits = set(self.style_traits) - set(ALL_STYLE_TRAITS)
        if invalid_traits:
            raise ValueError(f"Invalid traits specified: {invalid_traits}")
    
    def get_model_pairs(self) -> List[Tuple[str, str]]:
        """Get all source-target model pairs for analysis."""
        pairs = []
        # Source to all targets
        for target in self.target_models:
            pairs.append((self.source_model, target))
        
        # All pairs if cross-family analysis is enabled
        if self.include_cross_family_analysis:
            all_models = [self.source_model] + self.target_models
            for i, model1 in enumerate(all_models):
                for model2 in all_models[i+1:]:
                    if (model1, model2) not in pairs and (model2, model1) not in pairs:
                        pairs.append((model1, model2))
        
        return pairs
    
    def get_family_for_model(self, model_name: str) -> Optional[str]:
        """Get the family name for a given model."""
        for family, models in MODEL_FAMILIES.items():
            if model_name in models:
                return family
        return None

@dataclass
class DataConfig:
    """Configuration for multi-domain data generation and processing."""
    
    # Primary datasets for style analysis
    datasets: List[str] = field(default_factory=lambda: [
        "cnn_dailymail",  # News summarization
        "eli5",           # Conversational explanations
        "scitldr",        # Technical abstracts
        "openwebtext",    # General web content
        "pubmed",         # Medical literature
        "wikihow"         # Instructional content
    ])
    
    # Text processing settings
    max_input_length: int = 1024
    min_output_length: int = 20
    max_output_length: int = 500
    target_length_short: int = 100
    target_length_long: int = 300
    
    # Style-specific length targets
    style_length_targets: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "verbosity": {"low": 50, "high": 200},
        "formality": {"low": 100, "high": 150},
        "technical_complexity": {"low": 80, "high": 180},
        "certainty": {"low": 120, "high": 120}  # Length doesn't vary much for certainty
    })
    
    # Sampling and validation
    samples_per_trait: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Quality filtering
    min_quality_score: float = 0.5
    filter_duplicate_prompts: bool = True
    max_prompt_similarity: float = 0.8  # Max cosine similarity between prompts
    
    # Dataset-specific settings
    dataset_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "cnn_dailymail": {
            "version": "3.0.0",
            "use_train_split": True,
            "max_samples": 10000
        },
        "eli5": {
            "category_filter": ["science", "technology", "history"],
            "max_samples": 5000
        },
        "scitldr": {
            "domains": ["cs", "bio", "med"],
            "max_samples": 3000
        },
        "openwebtext": {
            "min_length": 100,
            "max_samples": 8000
        },
        "pubmed": {
            "abstract_only": True,
            "max_samples": 4000
        },
        "wikihow": {
            "categories": ["health", "technology", "education"],
            "max_samples": 3000
        }
    })

# Datasets available for different style dimensions
TRAIT_OPTIMAL_DATASETS = {
    "verbosity": ["cnn_dailymail", "eli5", "wikihow"],
    "formality": ["scitldr", "pubmed", "openwebtext"],
    "technical_complexity": ["scitldr", "pubmed", "eli5"],
    "certainty": ["pubmed", "cnn_dailymail", "scitldr"],
    "emotional_tone": ["eli5", "openwebtext", "wikihow"],
    "politeness": ["eli5", "wikihow", "openwebtext"],
    "humor": ["openwebtext", "eli5"],
    "clarity": ["eli5", "wikihow", "scitldr"]
}

# Global configurations
EXPERIMENT_CONFIG = ExperimentConfig()
DATA_CONFIG = DataConfig()