import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import nltk
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    # Fallback tokenization functions
    def word_tokenize(text):
        """Fallback word tokenization without NLTK."""
        import re
        # Simple regex-based tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def sent_tokenize(text):
        """Fallback sentence tokenization without NLTK."""
        import re
        # Simple sentence splitting on common punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from dataclasses import dataclass
from collections import Counter
import math
import warnings

# Suppress SSL warnings for cleaner output
warnings.filterwarnings("ignore", message=".*SSL.*")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        # SSL issues - use fallback tokenization (works fine)
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Container for evaluation metrics results."""
    length_metrics: Dict[str, float]
    verbosity_metrics: Dict[str, float]
    content_metrics: Dict[str, float]
    style_metrics: Dict[str, float]
    overall_score: float

class LengthAnalyzer:
    """Analyzes text length and related metrics."""
    
    @staticmethod
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(word_tokenize(text.lower()))
    
    @staticmethod
    def sentence_count(text: str) -> int:
        """Count sentences in text."""
        return len(sent_tokenize(text))
    
    @staticmethod
    def avg_sentence_length(text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        total_words = sum(len(word_tokenize(sent)) for sent in sentences)
        return total_words / len(sentences)
    
    @staticmethod
    def character_count(text: str) -> int:
        """Count characters excluding whitespace."""
        return len(re.sub(r'\s+', '', text))
    
    @classmethod
    def analyze_length(cls, text: str) -> Dict[str, float]:
        """Comprehensive length analysis."""
        return {
            'word_count': cls.word_count(text),
            'sentence_count': cls.sentence_count(text),
            'character_count': cls.character_count(text),
            'avg_sentence_length': cls.avg_sentence_length(text)
        }

class VerbosityScorer:
    """Measures verbosity and conciseness of text."""
    
    @staticmethod
    def redundancy_score(text: str) -> float:
        """Calculate redundancy based on repeated words and phrases."""
        words = word_tokenize(text.lower())
        if len(words) <= 1:
            return 0.0
        
        word_counts = Counter(words)
        total_repetitions = sum(max(0, count - 1) for count in word_counts.values())
        return total_repetitions / len(words)
    
    @staticmethod
    def information_density(text: str) -> float:
        """Calculate information density (unique words / total words)."""
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
        unique_words = len(set(words))
        return unique_words / len(words)
    
    @staticmethod
    def complexity_score(text: str) -> float:
        """Calculate complexity based on sentence structure and vocabulary."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        total_complexity = 0.0
        for sentence in sentences:
            words = word_tokenize(sentence)
            # Longer words indicate higher complexity
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            # More words per sentence indicate higher complexity
            sentence_length = len(words)
            
            sentence_complexity = (avg_word_length * 0.3) + (sentence_length * 0.1)
            total_complexity += sentence_complexity
        
        return total_complexity / len(sentences)
    
    @staticmethod
    def elaboration_score(text: str) -> float:
        """Score based on presence of elaborative words and phrases."""
        elaborative_words = [
            'furthermore', 'moreover', 'additionally', 'specifically', 'particularly',
            'notably', 'importantly', 'significantly', 'consequently', 'therefore',
            'however', 'nevertheless', 'nonetheless', 'meanwhile', 'subsequently',
            'for example', 'for instance', 'such as', 'in other words', 'that is'
        ]
        
        text_lower = text.lower()
        elaborative_count = sum(1 for phrase in elaborative_words if phrase in text_lower)
        word_count = len(word_tokenize(text))
        
        return elaborative_count / max(1, word_count / 100)  # Per 100 words
    
    @classmethod
    def calculate_verbosity_score(cls, text: str) -> float:
        """Calculate overall verbosity score (0 = concise, 1 = verbose)."""
        redundancy = cls.redundancy_score(text)
        inv_density = 1.0 - cls.information_density(text)
        complexity = cls.complexity_score(text) / 10.0  # Normalize
        elaboration = cls.elaboration_score(text)
        
        # Weighted combination
        verbosity = (redundancy * 0.3 + inv_density * 0.2 + 
                    complexity * 0.3 + elaboration * 0.2)
        
        return min(1.0, verbosity)
    
    @classmethod
    def analyze_verbosity(cls, text: str) -> Dict[str, float]:
        """Comprehensive verbosity analysis."""
        return {
            'redundancy_score': cls.redundancy_score(text),
            'information_density': cls.information_density(text),
            'complexity_score': cls.complexity_score(text),
            'elaboration_score': cls.elaboration_score(text),
            'overall_verbosity': cls.calculate_verbosity_score(text)
        }

class ContentPreservationEvaluator:
    """Evaluates how well content is preserved during style transfer."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores between reference and candidate."""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using TF-IDF cosine similarity."""
        try:
            # Fit on both texts
            corpus = [text1, text2]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def fact_preservation_score(self, original: str, generated: str) -> float:
        """Estimate fact preservation using named entity overlap."""
        # Simple heuristic: look for capitalized words and numbers
        def extract_entities(text):
            words = word_tokenize(text)
            entities = set()
            for word in words:
                # Capitalized words (potential proper nouns)
                if word[0].isupper() and len(word) > 2:
                    entities.add(word.lower())
                # Numbers
                if re.match(r'\d+', word):
                    entities.add(word)
            return entities
        
        orig_entities = extract_entities(original)
        gen_entities = extract_entities(generated)
        
        if not orig_entities:
            return 1.0  # No entities to preserve
        
        preserved = len(orig_entities.intersection(gen_entities))
        return preserved / len(orig_entities)
    
    def evaluate_content_preservation(self, original: str, generated: str) -> Dict[str, float]:
        """Comprehensive content preservation evaluation."""
        rouge_scores = self.rouge_scores(original, generated)
        semantic_sim = self.semantic_similarity(original, generated)
        fact_preservation = self.fact_preservation_score(original, generated)
        
        # Overall content score
        overall_content = (
            rouge_scores['rouge1_f'] * 0.3 +
            rouge_scores['rougeL_f'] * 0.3 +
            semantic_sim * 0.2 +
            fact_preservation * 0.2
        )
        
        return {
            **rouge_scores,
            'semantic_similarity': semantic_sim,
            'fact_preservation': fact_preservation,
            'overall_content_preservation': overall_content
        }

class StyleConsistencyEvaluator:
    """Evaluates consistency of style transfer."""
    
    @staticmethod
    def length_change_direction(baseline_length: int, steered_length: int, 
                               expected_direction: str) -> float:
        """Score based on whether length changed in expected direction."""
        length_ratio = steered_length / max(1, baseline_length)
        
        if expected_direction == "verbose":
            # Expect increase in length
            return min(1.0, max(0.0, (length_ratio - 1.0) / 1.0))
        elif expected_direction == "concise":
            # Expect decrease in length
            return min(1.0, max(0.0, (1.0 - length_ratio) / 0.5))
        else:
            return 0.0
    
    @staticmethod
    def verbosity_change_direction(baseline_verbosity: float, steered_verbosity: float,
                                  expected_direction: str) -> float:
        """Score based on whether verbosity changed in expected direction."""
        verbosity_change = steered_verbosity - baseline_verbosity
        
        if expected_direction == "verbose":
            return min(1.0, max(0.0, verbosity_change * 2.0))
        elif expected_direction == "concise":
            return min(1.0, max(0.0, -verbosity_change * 2.0))
        else:
            return 0.0
    
    @classmethod
    def evaluate_style_consistency(cls, baseline_text: str, steered_text: str,
                                  expected_direction: str) -> Dict[str, float]:
        """Evaluate consistency of style transfer."""
        # Length analysis
        baseline_length = LengthAnalyzer.word_count(baseline_text)
        steered_length = LengthAnalyzer.word_count(steered_text)
        length_direction_score = cls.length_change_direction(
            baseline_length, steered_length, expected_direction
        )
        
        # Verbosity analysis
        baseline_verbosity = VerbosityScorer.calculate_verbosity_score(baseline_text)
        steered_verbosity = VerbosityScorer.calculate_verbosity_score(steered_text)
        verbosity_direction_score = cls.verbosity_change_direction(
            baseline_verbosity, steered_verbosity, expected_direction
        )
        
        overall_consistency = (length_direction_score + verbosity_direction_score) / 2.0
        
        return {
            'length_direction_score': length_direction_score,
            'verbosity_direction_score': verbosity_direction_score,
            'baseline_length': baseline_length,
            'steered_length': steered_length,
            'baseline_verbosity': baseline_verbosity,
            'steered_verbosity': steered_verbosity,
            'overall_consistency': overall_consistency
        }

class PreferenceVectorEvaluator:
    """Main evaluator for preference vector transfer experiments."""
    
    def __init__(self):
        self.length_analyzer = LengthAnalyzer()
        self.verbosity_scorer = VerbosityScorer()
        self.content_evaluator = ContentPreservationEvaluator()
        self.style_evaluator = StyleConsistencyEvaluator()
    
    def evaluate_single_generation(self, 
                                  source_text: str,
                                  baseline_generation: str,
                                  steered_generation: str,
                                  expected_direction: str = "verbose") -> EvaluationResult:
        """Evaluate a single generation against baseline."""
        
        # Length metrics
        length_metrics = {
            'baseline': self.length_analyzer.analyze_length(baseline_generation),
            'steered': self.length_analyzer.analyze_length(steered_generation)
        }
        
        # Verbosity metrics
        verbosity_metrics = {
            'baseline': self.verbosity_scorer.analyze_verbosity(baseline_generation),
            'steered': self.verbosity_scorer.analyze_verbosity(steered_generation)
        }
        
        # Content preservation
        content_metrics = self.content_evaluator.evaluate_content_preservation(
            baseline_generation, steered_generation
        )
        
        # Style consistency
        style_metrics = self.style_evaluator.evaluate_style_consistency(
            baseline_generation, steered_generation, expected_direction
        )
        
        # Overall score calculation
        overall_score = (
            style_metrics['overall_consistency'] * 0.4 +
            content_metrics['overall_content_preservation'] * 0.4 +
            (1.0 - abs(verbosity_metrics['steered']['overall_verbosity'] - 
                      verbosity_metrics['baseline']['overall_verbosity'])) * 0.2
        )
        
        return EvaluationResult(
            length_metrics=length_metrics,
            verbosity_metrics=verbosity_metrics,
            content_metrics=content_metrics,
            style_metrics=style_metrics,
            overall_score=overall_score
        )
    
    def evaluate_experiment_results(self, 
                                   results: Dict,
                                   expected_direction: str = "verbose") -> Dict:
        """Evaluate results from a cross-model transfer experiment."""
        evaluation_summary = {
            'model_evaluations': {},
            'aggregate_metrics': {}
        }
        
        all_scores = []
        
        for model_name, model_results in results.get('models', {}).items():
            if 'error' in model_results:
                continue
            
            model_scores = []
            prompt_evaluations = {}
            
            for prompt_id, prompt_data in model_results.get('prompts', {}).items():
                if 'error' in prompt_data:
                    continue
                
                generations = prompt_data.get('generations', {})
                if 0.0 not in generations or 1.0 not in generations:
                    continue
                
                baseline = generations[0.0]
                steered = generations[1.0]
                
                # Evaluate this prompt
                eval_result = self.evaluate_single_generation(
                    prompt_data['text'], baseline, steered, expected_direction
                )
                
                prompt_evaluations[prompt_id] = {
                    'overall_score': eval_result.overall_score,
                    'style_consistency': eval_result.style_metrics['overall_consistency'],
                    'content_preservation': eval_result.content_metrics['overall_content_preservation']
                }
                
                model_scores.append(eval_result.overall_score)
                all_scores.append(eval_result.overall_score)
            
            if model_scores:
                evaluation_summary['model_evaluations'][model_name] = {
                    'prompts': prompt_evaluations,
                    'avg_score': np.mean(model_scores),
                    'std_score': np.std(model_scores),
                    'num_prompts': len(model_scores)
                }
        
        # Aggregate metrics across all models
        if all_scores:
            evaluation_summary['aggregate_metrics'] = {
                'overall_avg_score': np.mean(all_scores),
                'overall_std_score': np.std(all_scores),
                'total_evaluations': len(all_scores),
                'successful_models': len(evaluation_summary['model_evaluations'])
            }
        
        return evaluation_summary

def demonstrate_evaluation():
    """Demonstrate the evaluation metrics on sample texts."""
    baseline_text = "AI is useful for many applications. It can help with tasks."
    verbose_text = "Artificial intelligence represents a tremendously powerful and versatile technology that finds applications across numerous domains and industries. Specifically, AI systems can provide substantial assistance with a wide variety of complex tasks, ranging from data analysis to decision-making processes."
    
    evaluator = PreferenceVectorEvaluator()
    
    result = evaluator.evaluate_single_generation(
        source_text="Describe AI applications",
        baseline_generation=baseline_text,
        steered_generation=verbose_text,
        expected_direction="verbose"
    )
    
    print("Evaluation Demo Results:")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Style Consistency: {result.style_metrics['overall_consistency']:.3f}")
    print(f"Content Preservation: {result.content_metrics['overall_content_preservation']:.3f}")
    print(f"Verbosity Change: {result.verbosity_metrics['steered']['overall_verbosity'] - result.verbosity_metrics['baseline']['overall_verbosity']:.3f}")
    
    return result

if __name__ == "__main__":
    demo_result = demonstrate_evaluation()