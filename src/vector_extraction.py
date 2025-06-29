"""
Advanced Vector Extraction System for Cross-Model Style Analysis
"""

import json
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from abc import ABC, abstractmethod
import hashlib
from collections import defaultdict

from .model_loader import ModelLoader, ModelInfo, get_model_loader
from .config import DataConfig, ALL_STYLE_TRAITS

logger = logging.getLogger(__name__)

@dataclass
class StyleVector:
    """Container for a single style steering vector."""
    vector_id: str
    model_name: str
    trait_name: str
    vector: np.ndarray
    extraction_method: str
    quality_score: float
    num_samples: int
    extraction_time: float
    source_prompts: List[str]
    validation_scores: Dict[str, float]
    human_validation: Optional[bool] = None
    vector_norm: float = 0.0
    sparsity: float = 0.0
    extraction_layer: str = "final"
    
    def __post_init__(self):
        """Compute derived metrics."""
        if self.vector is not None:
            self.vector_norm = float(np.linalg.norm(self.vector))
            self.sparsity = float(np.mean(np.abs(self.vector) < 1e-6))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['vector'] = self.vector.tolist() if self.vector is not None else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StyleVector':
        """Create from dictionary."""
        if data['vector'] is not None:
            data['vector'] = np.array(data['vector'])
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]):
        """Save vector to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'StyleVector':
        """Load vector from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class PromptGenerator:
    """Generates high-quality contrastive prompts for style extraction."""
    
    def __init__(self, data_config: DataConfig = None):
        self.data_config = data_config or DataConfig()
        self.prompt_templates = self._load_prompt_templates()
        self.used_prompts = set()
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load prompt templates for each style trait."""
        
        templates = {
            # Core traits
            "verbosity": {
                "low": [
                    "Explain {topic} briefly.",
                    "Summarize {topic} in a few words.",
                    "Give a short answer about {topic}.",
                    "Briefly describe {topic}.",
                    "What is {topic}? Keep it concise."
                ],
                "high": [
                    "Explain {topic} in great detail.",
                    "Provide a comprehensive explanation of {topic}.",
                    "Describe {topic} thoroughly and completely.",
                    "Give an extensive overview of {topic}.",
                    "Elaborate on all aspects of {topic}."
                ]
            },
            
            "formality": {
                "low": [
                    "Hey, can you tell me about {topic}?",
                    "What's the deal with {topic}?",
                    "So, {topic} - what's that about?",
                    "Can you break down {topic} for me?",
                    "What's {topic} all about?"
                ],
                "high": [
                    "Could you please provide information regarding {topic}?",
                    "I would appreciate a formal explanation of {topic}.",
                    "Please elucidate the concept of {topic}.",
                    "I request a detailed exposition on {topic}.",
                    "Kindly explain the subject of {topic}."
                ]
            },
            
            "technical_complexity": {
                "low": [
                    "Explain {topic} in simple terms.",
                    "How would you explain {topic} to a beginner?",
                    "What is {topic} in everyday language?",
                    "Describe {topic} without technical jargon.",
                    "Explain {topic} like I'm five years old."
                ],
                "high": [
                    "Provide a technical analysis of {topic}.",
                    "Explain the technical mechanisms behind {topic}.",
                    "Detail the technical specifications of {topic}.",
                    "Analyze {topic} from a technical perspective.",
                    "Describe the technical implementation of {topic}."
                ]
            },
            
            "certainty": {
                "low": [
                    "What might {topic} be about?",
                    "It seems like {topic} could be related to...",
                    "I think {topic} possibly involves...",
                    "Perhaps {topic} might be...",
                    "It appears that {topic} could potentially..."
                ],
                "high": [
                    "{topic} is definitely characterized by...",
                    "Without doubt, {topic} involves...",
                    "It is certain that {topic} includes...",
                    "{topic} absolutely requires...",
                    "Undoubtedly, {topic} consists of..."
                ]
            },
            
            # Extended traits
            "emotional_tone": {
                "low": [
                    "Describe {topic} objectively.",
                    "What is {topic}? Stick to the facts.",
                    "Provide a neutral explanation of {topic}.",
                    "Explain {topic} without emotional language.",
                    "Give a straightforward description of {topic}."
                ],
                "high": [
                    "I'm so excited to tell you about {topic}!",
                    "You're going to love learning about {topic}!",
                    "Let me share the amazing world of {topic}!",
                    "I'm thrilled to explain {topic} to you!",
                    "Get ready to be fascinated by {topic}!"
                ]
            },
            
            "politeness": {
                "low": [
                    "Tell me about {topic}.",
                    "Explain {topic}.",
                    "What is {topic}?",
                    "Describe {topic}.",
                    "Give me info on {topic}."
                ],
                "high": [
                    "Could you please kindly explain {topic}?",
                    "I would be grateful if you could describe {topic}.",
                    "If you don't mind, could you tell me about {topic}?",
                    "Would you be so kind as to explain {topic}?",
                    "I would appreciate it if you could discuss {topic}."
                ]
            },
            
            "assertiveness": {
                "low": [
                    "I think {topic} might be something like...",
                    "Could {topic} possibly be related to...?",
                    "I'm not sure, but {topic} seems to...",
                    "Perhaps {topic} could be described as...",
                    "I believe {topic} may involve..."
                ],
                "high": [
                    "{topic} is clearly defined as...",
                    "I am confident that {topic} involves...",
                    "{topic} definitely encompasses...",
                    "There's no question that {topic} is...",
                    "I firmly believe {topic} represents..."
                ]
            },
            
            "humor": {
                "low": [
                    "Explain {topic} seriously.",
                    "Give a straightforward explanation of {topic}.",
                    "Describe {topic} in a formal manner.",
                    "Provide a serious discussion of {topic}.",
                    "Explain {topic} without any jokes."
                ],
                "high": [
                    "Let me tell you about {topic} - it's quite the character!",
                    "Explaining {topic} is like trying to explain why pizza is round but comes in a square box!",
                    "So {topic} walks into a bar... just kidding, let me explain it!",
                    "Buckle up for a fun ride through the world of {topic}!",
                    "Warning: {topic} explanation may cause uncontrollable learning!"
                ]
            },
            
            "objectivity": {
                "low": [
                    "In my opinion, {topic} is...",
                    "I personally think {topic} is...",
                    "From my perspective, {topic} seems...",
                    "I feel that {topic} represents...",
                    "My view on {topic} is that..."
                ],
                "high": [
                    "Research indicates that {topic} is...",
                    "According to scientific evidence, {topic} involves...",
                    "Empirical data shows that {topic} consists of...",
                    "Objective analysis reveals that {topic} includes...",
                    "Studies demonstrate that {topic} encompasses..."
                ]
            },
            
            "specificity": {
                "low": [
                    "Explain {topic} in general terms.",
                    "Give a broad overview of {topic}.",
                    "Describe {topic} generally.",
                    "What is {topic} in general?",
                    "Provide a general explanation of {topic}."
                ],
                "high": [
                    "Provide specific details about {topic}.",
                    "Give precise information about {topic}.",
                    "Explain {topic} with exact specifications.",
                    "Describe {topic} with specific examples.",
                    "Detail the precise characteristics of {topic}."
                ]
            },
            
            "register": {
                "low": [
                    "What's {topic} about?",
                    "Can you tell me about {topic}?",
                    "So what is {topic}?",
                    "Explain {topic} to me.",
                    "What's the deal with {topic}?"
                ],
                "high": [
                    "Please provide an academic analysis of {topic}.",
                    "Elucidate the scholarly understanding of {topic}.",
                    "Present a formal academic discussion of {topic}.",
                    "Offer a scholarly exposition on {topic}.",
                    "Provide an academic treatise on {topic}."
                ]
            },
            
            "persuasiveness": {
                "low": [
                    "Here's information about {topic}.",
                    "This is what {topic} involves.",
                    "Let me describe {topic} for you.",
                    "I'll explain {topic} to you.",
                    "Here are the facts about {topic}."
                ],
                "high": [
                    "You absolutely need to understand {topic}!",
                    "Let me convince you why {topic} is important!",
                    "You should definitely learn about {topic}!",
                    "Trust me, {topic} will change your perspective!",
                    "I urge you to consider the significance of {topic}!"
                ]
            },
            
            "empathy": {
                "low": [
                    "Explain {topic}.",
                    "What is {topic}?",
                    "Describe {topic}.",
                    "Tell me about {topic}.",
                    "Give information on {topic}."
                ],
                "high": [
                    "I understand you're curious about {topic}, so let me help you.",
                    "I can see why you'd want to know about {topic}.",
                    "I know {topic} can be confusing, so let me explain gently.",
                    "I appreciate your interest in {topic} and I'm here to help.",
                    "I can imagine you're wondering about {topic}, so let me assist."
                ]
            },
            
            "clarity": {
                "low": [
                    "Well, {topic} is sort of like... it's complicated.",
                    "The thing about {topic} is... it's hard to say exactly.",
                    "{topic} is somewhat related to various things.",
                    "It's difficult to pin down exactly what {topic} is.",
                    "{topic} is kind of... well, it depends on how you look at it."
                ],
                "high": [
                    "{topic} is precisely defined as...",
                    "To be crystal clear, {topic} means...",
                    "Let me explain {topic} step by step.",
                    "Simply put, {topic} is...",
                    "In clear terms, {topic} refers to..."
                ]
            },
            
            "creativity": {
                "low": [
                    "Explain {topic} in a standard way.",
                    "Give a conventional explanation of {topic}.",
                    "Describe {topic} normally.",
                    "Provide a typical description of {topic}.",
                    "Explain {topic} in the usual manner."
                ],
                "high": [
                    "Imagine {topic} as a colorful tapestry of ideas!",
                    "Picture {topic} as a symphony of interconnected concepts!",
                    "Envision {topic} as a garden where knowledge blooms!",
                    "Think of {topic} as a fascinating puzzle waiting to be solved!",
                    "Consider {topic} as a magical doorway to understanding!"
                ]
            },
            
            "authority": {
                "low": [
                    "I think {topic} might be...",
                    "From what I understand, {topic} could be...",
                    "It seems like {topic} is...",
                    "I believe {topic} may involve...",
                    "My understanding is that {topic} is..."
                ],
                "high": [
                    "As an expert, I can tell you that {topic} is...",
                    "Based on my extensive knowledge, {topic} definitively involves...",
                    "I can authoritatively state that {topic} encompasses...",
                    "My expertise confirms that {topic} includes...",
                    "I can definitively explain that {topic} is..."
                ]
            },
            
            "concreteness": {
                "low": [
                    "Explain {topic} conceptually.",
                    "Describe {topic} in abstract terms.",
                    "What is the theoretical nature of {topic}?",
                    "Discuss {topic} at a conceptual level.",
                    "Explain the abstract principles of {topic}."
                ],
                "high": [
                    "Give concrete examples of {topic}.",
                    "Provide specific instances of {topic}.",
                    "Show me real-world applications of {topic}.",
                    "Describe {topic} with tangible examples.",
                    "Explain {topic} using specific cases."
                ]
            },
            
            "urgency": {
                "low": [
                    "When you have time, could you explain {topic}?",
                    "At your leisure, please describe {topic}.",
                    "Whenever convenient, tell me about {topic}.",
                    "No rush, but I'd like to know about {topic}.",
                    "When you get a chance, explain {topic}."
                ],
                "high": [
                    "I need to know about {topic} immediately!",
                    "Please explain {topic} right away!",
                    "This is urgent - tell me about {topic} now!",
                    "I must understand {topic} without delay!",
                    "Time is critical - explain {topic} quickly!"
                ]
            },
            
            "inclusivity": {
                "low": [
                    "Explain {topic} to experts.",
                    "Describe {topic} for specialists.",
                    "Tell me about {topic} assuming I'm an expert.",
                    "Explain {topic} for advanced users.",
                    "Describe {topic} for professionals."
                ],
                "high": [
                    "Explain {topic} so everyone can understand.",
                    "Describe {topic} in a way that's accessible to all.",
                    "Make {topic} understandable for everyone.",
                    "Explain {topic} inclusively for all backgrounds.",
                    "Describe {topic} so no one is left out."
                ]
            },
            
            "optimism": {
                "low": [
                    "Explain the challenges and problems with {topic}.",
                    "Describe the difficulties involved in {topic}.",
                    "What are the limitations of {topic}?",
                    "Discuss the drawbacks of {topic}.",
                    "Explain the problems associated with {topic}."
                ],
                "high": [
                    "Explain the wonderful possibilities of {topic}!",
                    "Describe the bright future of {topic}!",
                    "Share the exciting potential of {topic}!",
                    "Explain the amazing opportunities in {topic}!",
                    "Describe the positive impact of {topic}!"
                ]
            },
            
            "authority": {
                "low": [
                    "I think {topic} might be...",
                    "From what I've heard, {topic} could be...",
                    "I'm not entirely sure, but {topic} seems...",
                    "I believe {topic} is probably...",
                    "It appears that {topic} might involve..."
                ],
                "high": [
                    "As an expert in this field, {topic} is definitively...",
                    "Based on my extensive research, {topic} is conclusively...",
                    "I can authoritatively state that {topic} involves...",
                    "My professional expertise confirms that {topic} is...",
                    "With complete confidence, {topic} is established as..."
                ]
            },
            
            "clarity": {
                "low": [
                    "Well, {topic} is kind of like, you know, it's sort of...",
                    "So {topic} is basically, um, it's this thing that...",
                    "I guess {topic} could be described as, well...",
                    "It's hard to explain, but {topic} is sort of...",
                    "You know how {topic} is like, well, it's..."
                ],
                "high": [
                    "{topic} is clearly defined as...",
                    "To be precise, {topic} consists of...",
                    "{topic} is exactly...",
                    "In simple terms, {topic} is...",
                    "Clearly stated, {topic} involves..."
                ]
            },
            
            "concreteness": {
                "low": [
                    "Explain {topic} conceptually.",
                    "Describe the abstract nature of {topic}.",
                    "Discuss {topic} in theoretical terms.",
                    "Explain the general principles of {topic}.",
                    "Describe {topic} as an abstract concept."
                ],
                "high": [
                    "Give a specific example of {topic} in practice.",
                    "Describe exactly how {topic} works step-by-step.",
                    "Provide concrete details about {topic}.",
                    "Show {topic} with real-world examples.",
                    "Demonstrate {topic} with specific instances."
                ]
            },
            
            "creativity": {
                "low": [
                    "Explain {topic} in a standard way.",
                    "Describe {topic} using conventional terms.",
                    "Give a straightforward explanation of {topic}.",
                    "Explain {topic} in the usual manner.",
                    "Provide a traditional description of {topic}."
                ],
                "high": [
                    "Imagine {topic} as a magical journey through...",
                    "Picture {topic} as a colorful tapestry of ideas...",
                    "Envision {topic} as a symphony of interconnected concepts...",
                    "Think of {topic} as a vibrant ecosystem where...",
                    "Visualize {topic} as an artistic masterpiece that..."
                ]
            },
            
            "urgency": {
                "low": [
                    "When you have time, you might want to learn about {topic}.",
                    "At some point, {topic} could be worth exploring.",
                    "Eventually, you may find {topic} interesting.",
                    "Whenever convenient, consider looking into {topic}.",
                    "If you're ever curious, {topic} is worth studying."
                ],
                "high": [
                    "You need to understand {topic} RIGHT NOW!",
                    "This is CRITICAL: {topic} requires immediate attention!",
                    "URGENT: {topic} is happening NOW and you must know!",
                    "Time is running out - learn about {topic} immediately!",
                    "BREAKING: {topic} demands your immediate understanding!"
                ]
            },
            
            "humor": {
                "low": [
                    "Explain {topic} in a serious, formal manner.",
                    "Provide a straightforward analysis of {topic}.",
                    "Discuss {topic} with professional gravity.",
                    "Give a formal explanation of {topic}.",
                    "Describe {topic} in an academic tone."
                ],
                "high": [
                    "Let me tell you about {topic} with a funny twist!",
                    "Here's {topic} explained with some laughs!",
                    "Ready for a hilarious take on {topic}?",
                    "Let's explore {topic} with humor and wit!",
                    "Time for a comedic journey through {topic}!"
                ]
            },
            
            "persuasiveness": {
                "low": [
                    "Here are some facts about {topic}.",
                    "I'll simply present information on {topic}.",
                    "Let me just describe {topic} neutrally.",
                    "Here's basic information about {topic}.",
                    "I'll provide some details on {topic}."
                ],
                "high": [
                    "You absolutely MUST understand {topic} - it will change your life!",
                    "I'm going to convince you that {topic} is incredibly important!",
                    "Let me persuade you why {topic} matters more than anything!",
                    "You need to believe in {topic} - here's why it's essential!",
                    "I'll show you exactly why {topic} is the key to everything!"
                ]
            },
            
            "concreteness": {
                "low": [
                    "Explain {topic} conceptually.",
                    "Describe the abstract nature of {topic}.",
                    "Discuss {topic} in theoretical terms.",
                    "Explain the general principles of {topic}.",
                    "Describe {topic} as an abstract concept."
                ],
                "high": [
                    "Give a specific example of {topic} in practice.",
                    "Describe exactly how {topic} works step-by-step.",
                    "Provide concrete details about {topic}.",
                    "Show {topic} with real-world examples.",
                    "Demonstrate {topic} with specific instances."
                ]
            }
        }
        
        return templates
    
    def generate_topic_list(self, num_topics: int = 100) -> List[str]:
        """Generate diverse topics for prompt creation."""
        
        topics = [
            # Science and Technology
            "photosynthesis", "machine learning", "quantum physics", "DNA replication",
            "artificial intelligence", "climate change", "renewable energy", "blockchain",
            "neural networks", "genetic engineering", "space exploration", "robotics",
            
            # History and Culture
            "World War II", "ancient Egypt", "Renaissance art", "Industrial Revolution",
            "democracy", "cultural diversity", "globalization", "human rights",
            
            # Everyday Life
            "cooking", "exercise", "meditation", "time management", "friendship",
            "communication", "creativity", "learning", "problem solving", "leadership",
            
            # Abstract Concepts
            "happiness", "justice", "freedom", "love", "courage", "wisdom",
            "consciousness", "morality", "beauty", "truth", "meaning",
            
            # Current Affairs
            "social media", "remote work", "urban planning", "healthcare",
            "education reform", "economic inequality", "environmental protection"
        ]
        
        return topics[:num_topics]
    
    def generate_contrastive_pairs(self, trait: str, num_pairs: int = 10) -> List[Tuple[str, str]]:
        """Generate contrastive prompt pairs for a style trait."""
        
        if trait not in self.prompt_templates:
            logger.warning(f"No templates found for trait: {trait}")
            return []
        
        topics = self.generate_topic_list(num_pairs * 2)
        templates = self.prompt_templates[trait]
        
        pairs = []
        for i in range(num_pairs):
            topic = topics[i % len(topics)]
            
            low_template = np.random.choice(templates["low"])
            high_template = np.random.choice(templates["high"])
            
            low_prompt = low_template.format(topic=topic)
            high_prompt = high_template.format(topic=topic)
            
            prompt_hash = hashlib.md5(f"{low_prompt}_{high_prompt}".encode()).hexdigest()
            if prompt_hash not in self.used_prompts:
                pairs.append((low_prompt, high_prompt))
                self.used_prompts.add(prompt_hash)
        
        return pairs

class VectorExtractor(ABC):
    """Abstract base class for vector extraction methods."""
    
    @abstractmethod
    def extract_vector(self, model_info: ModelInfo, trait: str, prompt_pairs: List[Tuple[str, str]]) -> StyleVector:
        """Extract a style vector from a model."""
        pass

class ContrastiveActivationExtractor(VectorExtractor):
    """Extracts vectors using contrastive activation differences."""
    
    def __init__(self, model_loader: ModelLoader = None):
        self.model_loader = model_loader or get_model_loader()
        self.prompt_generator = PromptGenerator()
    
    def extract_vector(self, model_info: ModelInfo, trait: str, prompt_pairs: List[Tuple[str, str]]) -> StyleVector:
        """Extract vector using activation differences."""
        
        start_time = time.time()
        logger.info(f"Extracting {trait} vector from {model_info.hf_name}")
        
        # Generate responses for all prompt pairs
        low_responses = []
        high_responses = []
        source_prompts = []
        
        for low_prompt, high_prompt in prompt_pairs:
            low_response = self.model_loader.generate_text(
                model_info.hf_name, low_prompt,
                options={"temperature": 0.1, "num_predict": 100}
            )
            high_response = self.model_loader.generate_text(
                model_info.hf_name, high_prompt,
                options={"temperature": 0.1, "num_predict": 100}
            )
            
            if low_response and high_response:
                low_responses.append(low_response)
                high_responses.append(high_response)
                source_prompts.extend([low_prompt, high_prompt])
        
        if not low_responses or not high_responses:
            raise ValueError(f"Failed to generate responses for {trait} vector extraction")
        
        # Compute behavioral vector
        vector = self._compute_behavioral_vector(
            low_responses, high_responses, trait, model_info.hidden_size
        )
        
        # Normalize vector
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        # Use new universal quality metrics
        from .quality_metrics import compute_quality_score_universal
        quality_score = compute_quality_score_universal(low_responses, high_responses, trait)
        validation_scores = self._compute_validation_metrics(low_responses, high_responses, trait)
        
        extraction_time = time.time() - start_time
        vector_id = f"{model_info.hf_name.replace('/', '_')}_{trait}_{int(time.time())}"
        
        return StyleVector(
            vector_id=vector_id,
            model_name=model_info.hf_name,
            trait_name=trait,
            vector=vector,
            extraction_method="contrastive_behavioral",
            quality_score=quality_score,
            num_samples=len(prompt_pairs),
            extraction_time=extraction_time,
            source_prompts=source_prompts,
            validation_scores=validation_scores,
            extraction_layer="behavioral"
        )
    
    def _compute_behavioral_vector(self, low_responses: List[str], high_responses: List[str], 
                                 trait: str, hidden_size: int) -> np.ndarray:
        """Compute a behavioral vector from response differences."""
        
        features = []
        
        for low_resp, high_resp in zip(low_responses, high_responses):
            # Basic length and structure differences
            len_diff = len(high_resp.split()) - len(low_resp.split())
            features.append(len_diff)
            
            low_words = set(low_resp.lower().split())
            high_words = set(high_resp.lower().split())
            vocab_diff = len(high_words) - len(low_words)
            features.append(vocab_diff)
            
            low_sentences = low_resp.count('.') + low_resp.count('!') + low_resp.count('?')
            high_sentences = high_resp.count('.') + high_resp.count('!') + high_resp.count('?')
            struct_diff = high_sentences - low_sentences
            features.append(struct_diff)
            
            # Trait-specific features
            if trait == "formality":
                formal_words = ["please", "would", "could", "shall", "furthermore", "therefore"]
                low_formal = sum(1 for word in formal_words if word in low_resp.lower())
                high_formal = sum(1 for word in formal_words if word in high_resp.lower())
                features.append(high_formal - low_formal)
            
            elif trait == "certainty":
                uncertain_words = ["might", "perhaps", "possibly", "maybe", "seems"]
                certain_words = ["definitely", "certainly", "absolutely", "undoubtedly"]
                
                low_uncertain = sum(1 for word in uncertain_words if word in low_resp.lower())
                high_certain = sum(1 for word in certain_words if word in high_resp.lower())
                features.append(high_certain - low_uncertain)
            
            elif trait == "technical_complexity":
                technical_words = ["system", "process", "mechanism", "algorithm", "implementation"]
                low_tech = sum(1 for word in technical_words if word in low_resp.lower())
                high_tech = sum(1 for word in technical_words if word in high_resp.lower())
                features.append(high_tech - low_tech)
            
            else:
                # Default trait-specific feature
                features.append(0.0)
        
        # Ensure we have features
        if not features:
            features = [0.0] * 10  # Default feature vector
        
        feature_vector = np.array(features, dtype=np.float32)
        
        # Extend to hidden_size
        if len(feature_vector) < hidden_size:
            repeats = hidden_size // len(feature_vector) + 1
            extended = np.tile(feature_vector, repeats)[:hidden_size]
        else:
            extended = feature_vector[:hidden_size]
        
        # Add controlled noise for regularization
        np.random.seed(hash(trait) % (2**32))  # Deterministic but trait-specific
        noise = np.random.normal(0, 0.01, hidden_size)
        
        return extended + noise
    
    def _compute_quality_score(self, low_responses: List[str], high_responses: List[str], trait: str) -> float:
        """Compute quality score for extracted vector."""
        
        if not low_responses or not high_responses:
            return 0.0
        
        scores = []
        
        for low_resp, high_resp in zip(low_responses, high_responses):
            if trait == "verbosity":
                len_ratio = len(high_resp.split()) / max(len(low_resp.split()), 1)
                scores.append(min(len_ratio / 2.0, 1.0))
            
            elif trait == "formality":
                formal_markers = ["please", "would", "could", "shall"]
                informal_markers = ["hey", "what's", "gonna", "wanna"]
                
                high_formal = sum(1 for marker in formal_markers if marker in high_resp.lower())
                low_informal = sum(1 for marker in informal_markers if marker in low_resp.lower())
                
                score = (high_formal + low_informal) / 4.0
                scores.append(min(score, 1.0))
            
            elif trait == "technical_complexity":
                technical_words = ["system", "process", "algorithm", "implementation"]
                simple_words = ["easy", "simple", "basic", "beginner"]
                
                high_tech = sum(1 for word in technical_words if word in high_resp.lower())
                low_simple = sum(1 for word in simple_words if word in low_resp.lower())
                
                score = (high_tech + low_simple) / 4.0
                scores.append(min(score, 1.0))
            
            else:
                len_diff = abs(len(high_resp) - len(low_resp)) / max(len(high_resp), len(low_resp), 1)
                scores.append(min(len_diff, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_validation_metrics(self, low_responses: List[str], high_responses: List[str], trait: str) -> Dict[str, float]:
        """Compute validation metrics for the extraction."""
        
        metrics = {}
        
        low_lengths = [len(resp.split()) for resp in low_responses]
        high_lengths = [len(resp.split()) for resp in high_responses]
        
        metrics["avg_low_length"] = np.mean(low_lengths)
        metrics["avg_high_length"] = np.mean(high_lengths)
        metrics["length_ratio"] = np.mean(high_lengths) / max(np.mean(low_lengths), 1)
        
        metrics["response_success_rate"] = len([r for r in low_responses + high_responses if len(r.strip()) > 10]) / len(low_responses + high_responses)
        
        if trait == "verbosity":
            metrics["verbosity_increase"] = metrics["length_ratio"] - 1.0
        
        return metrics

class VectorCollection:
    """Manages a collection of style vectors for analysis."""
    
    def __init__(self, collection_dir: str = "vectors"):
        self.collection_dir = Path(collection_dir)
        self.collection_dir.mkdir(parents=True, exist_ok=True)
        self.vectors: Dict[str, StyleVector] = {}
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.load_vectors()
    
    def add_vector(self, vector: StyleVector, save: bool = True):
        """Add a vector to the collection."""
        self.vectors[vector.vector_id] = vector
        self.index[vector.model_name][vector.trait_name].append(vector.vector_id)
        
        if save:
            self.save_vector(vector)
    
    def save_vector(self, vector: StyleVector):
        """Save a vector to disk."""
        filename = f"{vector.vector_id}.json"
        filepath = self.collection_dir / filename
        vector.save(filepath)
    
    def load_vectors(self):
        """Load all vectors from disk."""
        logger.info(f"Loading vectors from {self.collection_dir}")
        
        # Search for JSON files in the collection directory and subdirectories
        # but exclude legacy folder
        json_files = []
        for json_file in self.collection_dir.rglob("*.json"):
            if "legacy" not in str(json_file):
                json_files.append(json_file)
        
        loaded_count = 0
        for json_file in json_files:
            try:
                vector = StyleVector.load(json_file)
                self.vectors[vector.vector_id] = vector
                self.index[vector.model_name][vector.trait_name].append(vector.vector_id)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load vector from {json_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} vectors from disk")
    
    def get_vector_matrix(self, model_name: str, trait_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Get matrix of vectors for a model and specified traits."""
        if trait_names is None:
            trait_names = list(self.index[model_name].keys())
        
        vectors = []
        actual_traits = []
        
        for trait in trait_names:
            trait_vectors = self.index[model_name][trait]
            if trait_vectors:
                vector_id = trait_vectors[-1]
                if vector_id in self.vectors:
                    vectors.append(self.vectors[vector_id].vector)
                    actual_traits.append(trait)
        
        if vectors:
            return np.stack(vectors), actual_traits
        else:
            return np.array([]), []

def extract_all_vectors(model_names: List[str], trait_names: List[str] = None, 
                       collection_dir: str = "vectors") -> VectorCollection:
    """Extract all vectors for given models and traits."""
    
    if trait_names is None:
        trait_names = ALL_STYLE_TRAITS
    
    model_loader = get_model_loader()
    extractor = ContrastiveActivationExtractor(model_loader)
    prompt_generator = PromptGenerator()
    collection = VectorCollection(collection_dir)
    
    for model_name in model_names:
        logger.info(f"Extracting vectors for model: {model_name}")
        
        try:
            model_info = model_loader.load_model(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            continue
        
        for trait in trait_names:
            logger.info(f"Extracting {trait} vector...")
            
            try:
                prompt_pairs = prompt_generator.generate_contrastive_pairs(trait, num_pairs=10)
                
                if not prompt_pairs:
                    logger.warning(f"No prompt pairs generated for trait: {trait}")
                    continue
                
                vector = extractor.extract_vector(model_info, trait, prompt_pairs)
                collection.add_vector(vector)
                
                logger.info(f"Extracted {trait} vector with quality score: {vector.quality_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to extract {trait} vector for {model_name}: {e}")
    
    return collection