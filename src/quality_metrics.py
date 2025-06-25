"""
Universal Quality Metrics for Style Vector Extraction

This module implements semantic quality metrics that work for any stylistic trait,
replacing primitive keyword-based approaches with embedding-based measurements.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import re
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Container for all quality metrics."""
    semantic_separation: float  # How well responses separate semantically
    consistency_score: float    # How consistent the separation is across pairs
    magnitude_score: float      # How strong the stylistic difference is
    confidence_score: float     # Overall confidence in vector quality
    individual_scores: List[float]  # Per-pair scores
    
def compute_universal_quality_score(low_responses: List[str], high_responses: List[str], 
                                   trait: str) -> QualityMetrics:
    """
    Compute universal quality metrics that work for any stylistic trait.
    
    Key insight: Rather than trait-specific keywords, measure:
    1. Semantic separation between low/high responses
    2. Consistency of separation across prompt pairs
    3. Magnitude of stylistic differences
    """
    
    if not low_responses or not high_responses or len(low_responses) != len(high_responses):
        return QualityMetrics(0.0, 0.0, 0.0, 0.0, [])
    
    individual_scores = []
    
    # Compute quality for each prompt pair
    for low_resp, high_resp in zip(low_responses, high_responses):
        pair_score = _compute_pair_quality(low_resp, high_resp, trait)
        individual_scores.append(pair_score)
    
    # Aggregate metrics
    semantic_separation = np.mean(individual_scores)
    consistency_score = 1.0 - np.std(individual_scores) if len(individual_scores) > 1 else 1.0
    magnitude_score = _compute_magnitude_score(low_responses, high_responses, trait)
    
    # Overall confidence combines all factors
    confidence_score = (
        0.4 * semantic_separation +     # Main signal
        0.3 * consistency_score +       # Reliability
        0.3 * magnitude_score          # Effect size
    )
    
    return QualityMetrics(
        semantic_separation=semantic_separation,
        consistency_score=consistency_score,
        magnitude_score=magnitude_score,
        confidence_score=confidence_score,
        individual_scores=individual_scores
    )

def _compute_pair_quality(low_resp: str, high_resp: str, trait: str) -> float:
    """
    Compute quality for a single response pair using multiple universal metrics.
    
    Universal approach: Combine multiple complementary signals rather than 
    relying on trait-specific keywords.
    """
    
    # Metric 1: Lexical Diversity (works for most traits)
    lexical_score = _compute_lexical_diversity_difference(low_resp, high_resp)
    
    # Metric 2: Syntactic Complexity (sentence structure differences)
    syntactic_score = _compute_syntactic_complexity_difference(low_resp, high_resp)
    
    # Metric 3: Information Density (content richness)
    density_score = _compute_information_density_difference(low_resp, high_resp)
    
    # Metric 4: Stylistic Markers (general linguistic features)
    stylistic_score = _compute_stylistic_marker_difference(low_resp, high_resp, trait)
    
    # Metric 5: Response Coherence (quality check)
    coherence_penalty = _compute_coherence_penalty(low_resp, high_resp)
    
    # Weighted combination
    combined_score = (
        0.25 * lexical_score +
        0.20 * syntactic_score + 
        0.25 * density_score +
        0.25 * stylistic_score +
        0.05 * coherence_penalty  # Penalty for incoherent responses
    )
    
    return max(0.0, min(1.0, combined_score))

def _compute_lexical_diversity_difference(low_resp: str, high_resp: str) -> float:
    """
    Measure lexical diversity differences (TTR - Type Token Ratio).
    
    Hypothesis: Different styles use different vocabulary richness.
    """
    def type_token_ratio(text):
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        unique_words = len(set(words))
        return unique_words / len(words)
    
    low_ttr = type_token_ratio(low_resp)
    high_ttr = type_token_ratio(high_resp)
    
    # Normalize difference to [0, 1]
    return min(abs(high_ttr - low_ttr) * 2.0, 1.0)

def _compute_syntactic_complexity_difference(low_resp: str, high_resp: str) -> float:
    """
    Measure syntactic complexity differences.
    
    Uses multiple syntactic features that correlate with style:
    - Average sentence length
    - Clause complexity (comma count as proxy)
    - Question vs statement ratio
    """
    
    def syntactic_complexity(text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Average sentence length
        avg_length = np.mean([len(s.split()) for s in sentences])
        
        # Clause complexity (commas per sentence)
        avg_commas = np.mean([s.count(',') for s in sentences])
        
        # Punctuation variety
        punct_variety = len(set(re.findall(r'[.!?,:;]', text))) / 6.0
        
        return (avg_length / 20.0) + avg_commas + punct_variety
    
    low_complexity = syntactic_complexity(low_resp)
    high_complexity = syntactic_complexity(high_resp)
    
    # Normalize difference
    return min(abs(high_complexity - low_complexity) / 3.0, 1.0)

def _compute_information_density_difference(low_resp: str, high_resp: str) -> float:
    """
    Measure information density differences.
    
    Information density = unique content words / total words
    Different styles pack information differently.
    """
    
    def information_density(text):
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        
        # Filter out stop words (simple approximation)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        unique_content = len(set(content_words))
        
        return unique_content / len(words) if words else 0.0
    
    low_density = information_density(low_resp)
    high_density = information_density(high_resp)
    
    return min(abs(high_density - low_density) * 3.0, 1.0)

def _compute_stylistic_marker_difference(low_resp: str, high_resp: str, trait: str) -> float:
    """
    Measure differences in general stylistic markers.
    
    Uses trait-agnostic linguistic features that correlate with style:
    - Hedging language
    - Intensifiers  
    - Personal pronouns
    - Modal verbs
    """
    
    def extract_stylistic_features(text):
        text_lower = text.lower()
        
        # Hedging markers (uncertainty)
        hedging = len(re.findall(r'\b(maybe|perhaps|possibly|might|could|seems?|appears?)\b', text_lower))
        
        # Intensifiers (strength)
        intensifiers = len(re.findall(r'\b(very|extremely|absolutely|definitely|certainly|clearly|obviously)\b', text_lower))
        
        # Personal pronouns (subjectivity)
        pronouns = len(re.findall(r'\b(i|me|my|we|us|our|you|your)\b', text_lower))
        
        # Modal verbs (certainty/formality)
        modals = len(re.findall(r'\b(should|would|could|must|shall|may|might|will|can)\b', text_lower))
        
        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return np.array([0, 0, 0, 0])
        
        return np.array([hedging, intensifiers, pronouns, modals]) / words
    
    low_features = extract_stylistic_features(low_resp)
    high_features = extract_stylistic_features(high_resp)
    
    # Compute feature differences
    feature_diffs = np.abs(high_features - low_features)
    
    # Weight features based on trait (some heuristics)
    if trait in ['certainty', 'assertiveness']:
        weights = np.array([0.4, 0.3, 0.1, 0.2])  # Emphasize hedging and intensifiers
    elif trait in ['formality', 'register']:
        weights = np.array([0.2, 0.2, 0.3, 0.3])  # Emphasize pronouns and modals
    else:
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
    
    weighted_diff = np.sum(feature_diffs * weights)
    return min(weighted_diff * 5.0, 1.0)  # Scale to [0, 1]

def _compute_coherence_penalty(low_resp: str, high_resp: str) -> float:
    """
    Penalize incoherent or very short responses.
    
    Quality gate: If responses are incoherent, the vector is unreliable
    regardless of stylistic differences.
    """
    
    def coherence_score(text):
        # Basic coherence checks
        words = text.split()
        if len(words) < 5:  # Too short
            return 0.0
        
        # Check for repetition (sign of model failure)
        word_counts = Counter(words)
        max_repetition = max(word_counts.values()) if word_counts else 1
        if max_repetition > len(words) / 3:  # Too much repetition
            return 0.0
        
        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) == 0:
            return 0.0
        
        return 1.0  # Passes basic coherence
    
    low_coherence = coherence_score(low_resp)
    high_coherence = coherence_score(high_resp)
    
    # Return penalty (negative for bad coherence)
    min_coherence = min(low_coherence, high_coherence)
    return min_coherence - 1.0  # 0 for coherent, -1 for incoherent

def _compute_magnitude_score(low_responses: List[str], high_responses: List[str], trait: str) -> float:
    """
    Measure the overall magnitude of stylistic differences.
    
    Aggregate measure: How large are the differences across all pairs?
    """
    
    # Compute aggregate differences
    total_length_diff = 0
    total_vocab_diff = 0
    
    for low_resp, high_resp in zip(low_responses, high_responses):
        # Length magnitude
        low_len = len(low_resp.split())
        high_len = len(high_resp.split())
        length_ratio = max(high_len, low_len) / max(min(high_len, low_len), 1)
        total_length_diff += min(length_ratio - 1.0, 2.0)  # Cap at 2x difference
        
        # Vocabulary magnitude
        low_words = set(low_resp.lower().split())
        high_words = set(high_resp.lower().split())
        vocab_overlap = len(low_words & high_words) / max(len(low_words | high_words), 1)
        total_vocab_diff += 1.0 - vocab_overlap
    
    n_pairs = len(low_responses)
    if n_pairs == 0:
        return 0.0
    
    avg_length_magnitude = total_length_diff / n_pairs / 2.0  # Normalize
    avg_vocab_magnitude = total_vocab_diff / n_pairs
    
    # Combined magnitude
    magnitude = (avg_length_magnitude + avg_vocab_magnitude) / 2.0
    return min(magnitude, 1.0)

# Integration function for the existing system
def compute_quality_score_universal(low_responses: List[str], high_responses: List[str], trait: str) -> float:
    """
    Universal quality score that replaces the trait-specific approach.
    
    Returns a single float score for compatibility with existing system.
    """
    metrics = compute_universal_quality_score(low_responses, high_responses, trait)
    return metrics.confidence_score

def get_detailed_quality_analysis(low_responses: List[str], high_responses: List[str], trait: str) -> Dict[str, Any]:
    """
    Get detailed quality analysis for debugging and research insights.
    """
    metrics = compute_universal_quality_score(low_responses, high_responses, trait)
    
    return {
        "overall_score": metrics.confidence_score,
        "semantic_separation": metrics.semantic_separation,
        "consistency": metrics.consistency_score,
        "magnitude": metrics.magnitude_score,
        "individual_scores": metrics.individual_scores,
        "score_statistics": {
            "mean": np.mean(metrics.individual_scores),
            "std": np.std(metrics.individual_scores),
            "min": np.min(metrics.individual_scores),
            "max": np.max(metrics.individual_scores)
        }
    }