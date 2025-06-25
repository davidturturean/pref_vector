import random
import numpy as np
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .config import DATA_CONFIG, get_model_config
from .utils import ModelLoader, TextValidator, handle_errors
from .ollama_utils import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummaryPair:
    """Container for paired concise/verbose summaries."""
    source_text: str
    concise_summary: str
    verbose_summary: str
    source_length: int
    concise_length: int
    verbose_length: int

class SummaryStyleGenerator:
    """Generates concise and verbose summaries using Ollama."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "mistralai/Mistral-7B-Instruct-v0.1"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Ollama model client for summary generation."""
        self.model = ModelLoader.load_model(self.model_name)
    
    def generate_summary(self, text: str, style: str, max_length: int = 150, trait: str = "verbosity") -> str:
        """Generate a summary with specified style and trait."""
        
        if trait == "verbosity":
            if style == "concise":
                prompt = f"""Summarize the following text in 1-2 sentences, focusing only on the main point:

Text: {text}

Brief summary:"""
            else:  # verbose
                prompt = f"""Provide a detailed summary of the following text, including background context, key details, and implications:

Text: {text}

Detailed summary:"""
        
        elif trait == "formality":
            if style == "concise":  # informal
                prompt = f"""Explain this topic casually, like you're talking to a friend:

Topic: {text}

Casual explanation:"""
            else:  # formal
                prompt = f"""Provide an academic, formal explanation of this topic:

Topic: {text}

Formal explanation:"""
        
        elif trait == "technical_complexity":
            if style == "concise":  # simple
                prompt = f"""Explain this in simple terms for a beginner:

Topic: {text}

Simple explanation:"""
            else:  # technical
                prompt = f"""Provide a technical, expert-level explanation:

Topic: {text}

Technical explanation:"""
        
        elif trait == "certainty":
            if style == "concise":  # uncertain
                prompt = f"""Discuss the uncertainties and limitations regarding:

Topic: {text}

Uncertain discussion:"""
            else:  # certain
                prompt = f"""State the definitive facts and established knowledge about:

Topic: {text}

Definitive facts:"""
        
        else:  # fallback to verbosity
            if style == "concise":
                prompt = f"""Provide a brief explanation of: {text}

Brief explanation:"""
            else:
                prompt = f"""Provide a detailed explanation of: {text}

Detailed explanation:"""
        
        try:
            response = self.model.generate(prompt, max_tokens=max_length, temperature=0.7)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate {style} summary: {e}")
            return ""
        

class DatasetPreparator:
    """Prepares paired concise/verbose summary datasets for preference vector learning."""
    
    def __init__(self, config: DATA_CONFIG = DATA_CONFIG):
        self.config = config
        self.generator = None
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    def load_source_dataset(self) -> Dataset:
        """Load the source dataset for generating summaries."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        if self.config.dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", self.config.dataset_version, split="train")
            # Filter for appropriate length articles
            dataset = dataset.filter(
                lambda x: len(x["article"].split()) >= 100 and len(x["article"].split()) <= 800
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        return dataset
    
    def generate_summary_pairs(self, num_pairs: int = 100, trait: str = "verbosity", model_name: str = None) -> List[SummaryPair]:
        """Generate paired summaries for specified trait."""
        if self.generator is None:
            self.generator = SummaryStyleGenerator(model_name)
        
        if trait == "verbosity":
            return self._generate_verbosity_pairs(num_pairs)
        else:
            return self._generate_trait_specific_pairs(num_pairs, trait)
    
    def _generate_verbosity_pairs(self, num_pairs: int) -> List[SummaryPair]:
        """Generate traditional verbosity pairs using dataset articles."""
        source_dataset = self.load_source_dataset()
        pairs = []
        
        logger.info(f"Generating {num_pairs} verbosity pairs...")
        
        # Sample random articles
        indices = random.sample(range(len(source_dataset)), min(num_pairs, len(source_dataset)))
        
        for idx in tqdm(indices, desc="Generating verbosity pairs"):
            article = source_dataset[idx]
            source_text = article["article"]
            
            # Skip articles that are too short or too long
            if len(source_text.split()) < 50 or len(source_text.split()) > 1000:
                continue
            
            try:
                # Generate both styles
                concise_summary = self.generator.generate_summary(
                    source_text, "concise", self.config.concise_target_length, "verbosity"
                )
                verbose_summary = self.generator.generate_summary(
                    source_text, "verbose", self.config.verbose_target_length, "verbosity"
                )
                
                # Validate the summaries
                if self._validate_summary_pair(concise_summary, verbose_summary):
                    pair = SummaryPair(
                        source_text=source_text,
                        concise_summary=concise_summary,
                        verbose_summary=verbose_summary,
                        source_length=len(source_text.split()),
                        concise_length=len(concise_summary.split()),
                        verbose_length=len(verbose_summary.split())
                    )
                    pairs.append(pair)
                    
                    if len(pairs) >= num_pairs:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to generate summary pair for article {idx}: {e}")
                continue
        
        return pairs
    
    def _generate_trait_specific_pairs(self, num_pairs: int, trait: str) -> List[SummaryPair]:
        """Generate pairs for non-verbosity traits using curated topics."""
        pairs = []
        
        # Define topics for each trait
        trait_topics = {
            "formality": [
                "machine learning algorithms", "climate change effects", "economic inflation",
                "vaccine effectiveness", "renewable energy", "artificial intelligence",
                "genetic engineering", "quantum computing", "blockchain technology",
                "sustainable development", "neural networks", "space exploration"
            ],
            "technical_complexity": [
                "photosynthesis process", "DNA replication", "rocket propulsion",
                "cryptographic algorithms", "neural network training", "quantum mechanics",
                "protein folding", "semiconductor manufacturing", "signal processing",
                "computer vision", "natural language processing", "database optimization"
            ],
            "certainty": [
                "dark matter properties", "consciousness mechanisms", "climate predictions",
                "artificial general intelligence", "cancer treatment", "economic forecasting",
                "earthquake prediction", "drug effectiveness", "social media impact",
                "technology adoption rates", "pandemic modeling", "asteroid threats"
            ]
        }
        
        topics = trait_topics.get(trait, trait_topics["formality"])
        
        logger.info(f"Generating {num_pairs} {trait} pairs...")
        
        # Cycle through topics if needed
        extended_topics = (topics * ((num_pairs // len(topics)) + 1))[:num_pairs]
        
        for i, topic in enumerate(tqdm(extended_topics, desc=f"Generating {trait} pairs")):
            try:
                # Generate both styles for this trait
                low_trait = self.generator.generate_summary(
                    topic, "concise", 100, trait
                )
                high_trait = self.generator.generate_summary(
                    topic, "verbose", 150, trait
                )
                
                # Validate the pair
                if self._validate_trait_pair(low_trait, high_trait, trait):
                    pair = SummaryPair(
                        source_text=topic,
                        concise_summary=low_trait,
                        verbose_summary=high_trait,
                        source_length=len(topic.split()),
                        concise_length=len(low_trait.split()),
                        verbose_length=len(high_trait.split())
                    )
                    pairs.append(pair)
                    
            except Exception as e:
                logger.warning(f"Failed to generate {trait} pair for {topic}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(pairs)} {trait} pairs")
        return pairs
    
    def _validate_trait_pair(self, low_response: str, high_response: str, trait: str) -> bool:
        """Validate trait-specific pairs."""
        if not (TextValidator.validate_text_basic(low_response) and 
                TextValidator.validate_text_basic(high_response)):
            return False
        
        if trait == "formality":
            formal_words = ['therefore', 'furthermore', 'consequently', 'methodology', 'investigation']
            informal_words = ['basically', 'like', 'pretty much', 'stuff', 'thing']
            
            high_formal = sum(1 for word in formal_words if word in high_response.lower())
            low_informal = sum(1 for word in informal_words if word in low_response.lower())
            
            return high_formal > 0 or low_informal > 0 or len(high_response) > len(low_response) * 1.2
        
        elif trait == "technical_complexity":
            return len(high_response.split()) > len(low_response.split()) * 1.3
        
        elif trait == "certainty":
            uncertain_words = ['might', 'could', 'possibly', 'perhaps', 'uncertain', 'may']
            certain_words = ['is', 'are', 'definitively', 'proven', 'established', 'confirmed']
            
            low_uncertain = sum(1 for word in uncertain_words if word in low_response.lower())
            high_certain = sum(1 for word in certain_words if word in high_response.lower())
            
            return low_uncertain > 0 or high_certain > 0
        
        return len(high_response.split()) > len(low_response.split()) * 1.1
    
    def _validate_summary_pair(self, concise: str, verbose: str) -> bool:
        """Validate that the summary pair meets quality criteria."""
        return TextValidator.validate_summary_pair(concise, verbose)
    
    def save_dataset(self, pairs: List[SummaryPair], filename: str):
        """Save the generated summary pairs to disk."""
        data = {
            "source_text": [p.source_text for p in pairs],
            "concise_summary": [p.concise_summary for p in pairs],
            "verbose_summary": [p.verbose_summary for p in pairs],
            "source_length": [p.source_length for p in pairs],
            "concise_length": [p.concise_length for p in pairs],
            "verbose_length": [p.verbose_length for p in pairs]
        }
        
        dataset = Dataset.from_dict(data)
        dataset.save_to_disk(filename)
        logger.info(f"Saved {len(pairs)} summary pairs to {filename}")
    
    def load_dataset(self, filename: str) -> List[SummaryPair]:
        """Load summary pairs from disk."""
        dataset = Dataset.load_from_disk(filename)
        
        pairs = []
        for i in range(len(dataset)):
            pairs.append(SummaryPair(
                source_text=dataset[i]["source_text"],
                concise_summary=dataset[i]["concise_summary"],
                verbose_summary=dataset[i]["verbose_summary"],
                source_length=dataset[i]["source_length"],
                concise_length=dataset[i]["concise_length"],
                verbose_length=dataset[i]["verbose_length"]
            ))
        
        logger.info(f"Loaded {len(pairs)} summary pairs from {filename}")
        return pairs

@handle_errors("create preference dataset")
def create_preference_dataset(num_pairs: int = 100, save_path: str = None, trait: str = "verbosity", model_name: str = None) -> List[SummaryPair]:
    """Create a dataset of preference pairs for training."""
    preparator = DatasetPreparator()
    pairs = preparator.generate_summary_pairs(num_pairs, trait, model_name)
    
    if save_path:
        preparator.save_dataset(pairs, save_path)
    
    return pairs

if __name__ == "__main__":
    # Generate and save dataset
    pairs = create_preference_dataset(num_pairs=50, save_path="data/summary_pairs")
    
    # Display sample
    if pairs:
        sample = pairs[0]
        print("Sample Summary Pair:")
        print(f"Source length: {sample.source_length} words")
        print(f"Concise ({sample.concise_length} words): {sample.concise_summary}")
        print(f"Verbose ({sample.verbose_length} words): {sample.verbose_summary}")