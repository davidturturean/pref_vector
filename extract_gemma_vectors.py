#!/usr/bin/env python3
"""
Extract preference vectors from Gemma model and save them.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import json
from typing import List
from src.vector_extraction import extract_verbosity_vector
from src.adaptive_vector_extraction import RobustCrossModelExtractor, MultiPointExtractor
from src.data_preparation import create_preference_dataset, SummaryPair
from src.vector_injection import SteerableModel
from src.config import get_model_config
from src.utils import LayerUtils, handle_errors

def create_directories():
    """Create necessary directories for saving results."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("vectors", exist_ok=True)
    os.makedirs("results", exist_ok=True)

def step1_basic_extraction():
    """Step 1: Basic vector extraction from Gemma."""
    print("="*60)
    print("STEP 1: Basic Vector Extraction from Gemma")
    print("="*60)
    
    try:
        # Use smaller Gemma model for testing (fallback to even smaller if needed)
        model_name = "google/gemma-2b"
        print(f"Using model: {model_name}")
        
        # Get default intervention layer using shared utility
        intervention_layer = LayerUtils.get_default_intervention_layer(model_name)
        print(f"Using intervention layer {intervention_layer} for extraction")
        
        # Generate training data using methodologically sound dataset selection
        print("Creating methodologically diverse summary pairs...")
        from src.data_preparation import DatasetPreparator, SummaryStyleGenerator
        from src.dataset_selection import create_methodologically_sound_dataset
        
        try:
            preparator = DatasetPreparator()
            # Use Gemma for generating the summary pairs
            preparator.generator = SummaryStyleGenerator(model_name=model_name)
            pairs = preparator.generate_summary_pairs(num_pairs=10)
            preparator.save_dataset(pairs, "data/gemma_pairs")
            print(f"Created {len(pairs)} summary pairs")
        except Exception as e:
            if "memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                print(f"⚠️  Memory issue with {model_name}, trying smaller model...")
                # Fallback to a smaller model for data generation
                preparator.generator = SummaryStyleGenerator(model_name="microsoft/DialoGPT-medium")
                pairs = preparator.generate_summary_pairs(num_pairs=10)
                preparator.save_dataset(pairs, "data/gemma_pairs")
                print(f"Created {len(pairs)} summary pairs with fallback model")
            else:
                raise
        
        # Extract preference vector from Gemma at the determined layer
        print(f"Extracting preference vector from Gemma at layer {intervention_layer}...")
        from src.vector_extraction import PreferenceVectorExtractor
        
        extractor = PreferenceVectorExtractor(model_name, intervention_layer)
        vector = extractor.extract_preference_vector_from_pairs(pairs, method="difference")
        
        # Save the vector manually
        extractor.save_vector(
            vector, 
            "vectors/gemma_verbosity_vector.json",
            metadata={
                'model_name': model_name,
                'intervention_layer': intervention_layer,
                'num_pairs': len(pairs),
                'method': 'difference'
            }
        )
        
        print(f"✅ Step 1 Complete!")
        print(f"   Vector shape: {vector.shape}")
        print(f"   Vector norm: {vector.norm().item():.4f}")
        print(f"   Saved to: vectors/gemma_verbosity_vector.json")
        
        return vector, pairs
        
    except Exception as e:
        print(f"❌ Step 1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def step2_robust_extraction(pairs):
    """Step 2: Architecture-aware robust extraction."""
    print("\n" + "="*60)
    print("STEP 2: Architecture-Aware Robust Extraction")
    print("="*60)
    
    try:
        # Create robust extractor
        extractor = RobustCrossModelExtractor()
        
        # Extract from multiple architectural points in Gemma
        print("Extracting robust preference vectors from Gemma...")
        vector_sets = extractor.extract_robust_preference_vectors(
            model_name="google/gemma-2b",
            summary_pairs=pairs,
            target_architectures=["microsoft/DialoGPT-medium", "google/gemma-7b"]  # Include other Gemma sizes
        )
        
        # Save the results
        results_file = "vectors/gemma_robust_vectors.json"
        
        # Convert to serializable format
        serializable_data = {}
        for set_name, vectors in vector_sets.items():
            serializable_data[set_name] = {}
            if isinstance(vectors, dict):
                for vector_name, adaptive_vector in vectors.items():
                    if hasattr(adaptive_vector, 'vector'):
                        serializable_data[set_name][vector_name] = {
                            'vector': adaptive_vector.vector.tolist(),
                            'shape': list(adaptive_vector.vector.shape),
                            'extraction_layer': adaptive_vector.extraction_layer,
                            'extraction_method': adaptive_vector.extraction_method,
                            'functional_layer_type': adaptive_vector.functional_layer_type
                        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Step 2 Complete!")
        print(f"   Extracted vector sets: {list(vector_sets.keys())}")
        for set_name, vectors in vector_sets.items():
            if isinstance(vectors, dict):
                print(f"   {set_name} vectors:")
                for vector_name, adaptive_vector in vectors.items():
                    if hasattr(adaptive_vector, 'vector'):
                        print(f"     {vector_name}: {adaptive_vector.vector.shape}")
        print(f"   Saved to: {results_file}")
        
        return vector_sets
        
    except Exception as e:
        print(f"❌ Step 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def step3_multipoint_extraction(pairs):
    """Step 3: Multi-point extraction from specific layers."""
    print("\n" + "="*60)
    print("STEP 3: Multi-Point Layer-Specific Extraction")
    print("="*60)
    
    try:
        # Create targeted extractor
        extractor = MultiPointExtractor("google/gemma-2b")
        
        # Extract from different functional layers
        vectors = extractor.extract_multi_point_vectors(pairs, [
            'feature_extraction',  # Early processing
            'representation',      # Mid-level 
            'decision_making'      # Late processing
        ])
        
        # Save results
        results_file = "vectors/gemma_multipoint_vectors.json"
        serializable_data = {}
        
        for name, adaptive_vector in vectors.items():
            serializable_data[name] = {
                'vector': adaptive_vector.vector.tolist(),
                'shape': list(adaptive_vector.vector.shape),
                'extraction_layer': adaptive_vector.extraction_layer,
                'extraction_method': adaptive_vector.extraction_method,
                'functional_layer_type': adaptive_vector.functional_layer_type,
                'normalization_applied': adaptive_vector.normalization_applied
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Step 3 Complete!")
        print("   Gemma preference vectors extracted:")
        for name, adaptive_vector in vectors.items():
            print(f"     {name}: layer {adaptive_vector.extraction_layer}, shape {adaptive_vector.vector.shape}")
        print(f"   Saved to: {results_file}")
        
        return vectors
        
    except Exception as e:
        print(f"❌ Step 3 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def step4_validation_test():
    """Step 4: Test extracted vector on Gemma itself."""
    print("\n" + "="*60)
    print("STEP 4: Validation Test on Gemma")
    print("="*60)
    
    try:
        # Load the basic vector
        vector_file = "vectors/gemma_verbosity_vector.json"
        if not os.path.exists(vector_file):
            print(f"❌ Vector file not found: {vector_file}")
            return None
        
        with open(vector_file, 'r') as f:
            data = json.load(f)
        vector = torch.tensor(data['vector'])
        
        print(f"Loaded vector: {vector.shape}")
        
        # Test on Gemma
        print("Loading Gemma model for testing...")
        model = SteerableModel("google/gemma-2b")
        
        test_prompt = "Explain the benefits of renewable energy:"
        
        # Generate at different steering scales
        print("Testing vector with different scales...")
        results = model.compare_generations(
            test_prompt, 
            vector, 
            layer_idx=12,  # Middle layer for Gemma-2B (assuming ~24 layers)
            scales=[-1.0, 0.0, 1.0]
        )
        
        # Save test results
        test_results = {
            'prompt': test_prompt,
            'vector_info': {
                'shape': list(vector.shape),
                'norm': vector.norm().item()
            },
            'generations': results
        }
        
        with open("results/gemma_validation_test.json", 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"✅ Step 4 Complete!")
        print("   Gemma steering test results:")
        for scale, text in results.items():
            print(f"     Scale {scale}: {text[:80]}...")
        print("   Saved to: results/gemma_validation_test.json")
        
        return results
        
    except Exception as e:
        print(f"❌ Step 4 Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_custom_preference_example():
    """Create example of custom preference pairs."""
    print("\n" + "="*60)
    print("BONUS: Custom Preference Example")
    print("="*60)
    
    # Create custom technical vs simple preference pairs
    tech_pairs = [
        SummaryPair(
            source_text="Photosynthesis is the process by which plants convert light energy into chemical energy",
            concise_summary="Plants make food from sunlight using chlorophyll",
            verbose_summary="Chloroplasts utilize photons to drive ATP synthesis via the electron transport chain, converting CO2 and H2O into glucose through the Calvin cycle",
            source_length=15, concise_length=8, verbose_length=22
        ),
        SummaryPair(
            source_text="Machine learning algorithms learn patterns from data to make predictions",
            concise_summary="AI learns from examples to make guesses about new data",
            verbose_summary="Neural networks employ gradient descent optimization to minimize loss functions across training datasets, enabling pattern recognition through weighted feature extraction",
            source_length=12, concise_length=10, verbose_length=20
        ),
        SummaryPair(
            source_text="Climate change refers to long-term shifts in global weather patterns",
            concise_summary="Earth's weather is changing over many years due to human activities",
            verbose_summary="Anthropogenic greenhouse gas emissions alter radiative forcing dynamics, resulting in measurable increases in global mean surface temperature and associated climatological phenomena",
            source_length=11, concise_length=11, verbose_length=21
        )
    ]
    
    # Save custom preference data
    custom_data = {
        'preference_type': 'technical_vs_simple',
        'description': 'Technical jargon vs simple explanations',
        'pairs': []
    }
    
    for pair in tech_pairs:
        custom_data['pairs'].append({
            'source_text': pair.source_text,
            'simple_summary': pair.concise_summary,
            'technical_summary': pair.verbose_summary,
            'source_length': pair.source_length,
            'simple_length': pair.concise_length,
            'technical_length': pair.verbose_length
        })
    
    with open("data/custom_technical_preference_pairs.json", 'w') as f:
        json.dump(custom_data, f, indent=2)
    
    print("✅ Custom preference example created!")
    print(f"   Created {len(tech_pairs)} technical vs simple pairs")
    print("   Saved to: data/custom_technical_preference_pairs.json")
    print("   You can use these for extracting technical/simple preference vectors!")

@handle_errors("extract multiple traits")
def extract_multiple_traits(model_name: str = "google/gemma-2b", 
                          traits: List[str] = None,
                          pairs_per_trait: int = 10):
    """Extract vectors for multiple traits efficiently."""
    if traits is None:
        traits = ["verbosity", "formality", "technical_complexity", "certainty"]
    
    print(f"🎯 MULTI-TRAIT EXTRACTION: {', '.join(traits)}")
    print("="*60)
    
    results = {}
    
    for trait in traits:
        print(f"\n📊 Extracting {trait} preference vector...")
        
        try:
            # Generate trait-specific pairs
            from src.data_preparation import create_preference_dataset
            pairs = create_preference_dataset(
                num_pairs=pairs_per_trait, 
                trait=trait,
                save_path=f"data/gemma_{trait}_pairs",
                model_name=model_name
            )
            
            # Extract vector for this trait
            from src.vector_extraction import extract_preference_vector
            vector = extract_preference_vector(
                model_name=model_name,
                summary_pairs=pairs,
                method="difference",
                trait=trait,
                save_path=f"vectors/gemma_{trait}_vector.json"
            )
            
            results[trait] = {
                'vector': vector,
                'pairs_generated': len(pairs),
                'vector_norm': vector.norm().item()
            }
            
            print(f"✅ {trait}: shape {vector.shape}, norm {vector.norm().item():.3f}")
            
        except Exception as e:
            print(f"❌ Failed to extract {trait}: {e}")
            results[trait] = {'error': str(e)}
    
    return results

def main():
    """Run all extraction steps."""
    print("🚀 GEMMA MULTI-TRAIT PREFERENCE VECTOR EXTRACTION")
    print("Starting comprehensive preference vector extraction from Gemma...")
    
    # Create directories
    create_directories()
    
    # Extract multiple traits efficiently
    trait_results = extract_multiple_traits()
    
    # Legacy single-trait extraction for detailed analysis
    vector, pairs = step1_basic_extraction()
    
    if pairs is not None:
        # Step 2: Robust extraction
        vector_sets = step2_robust_extraction(pairs)
        
        # Step 3: Multi-point extraction
        multipoint_vectors = step3_multipoint_extraction(pairs)
    
    # Step 4: Validation test
    validation_results = step4_validation_test()
    
    # Bonus: Custom preference example
    create_custom_preference_example()
    
    print("\n" + "="*60)
    print("🎉 GEMMA MULTI-TRAIT EXTRACTION COMPLETE!")
    print("="*60)
    print("Multi-trait results:")
    for trait, result in trait_results.items():
        if 'error' in result:
            print(f"  ❌ {trait}: {result['error']}")
        else:
            print(f"  ✅ {trait}: {result['pairs_generated']} pairs, norm {result['vector_norm']:.3f}")
    
    print("\nFiles created:")
    print("  📁 data/gemma_*_pairs/ - Trait-specific training pairs")
    print("  📁 vectors/gemma_*_vector.json - Trait-specific preference vectors")
    print("  📁 vectors/gemma_robust_vectors.json - Architecture-aware vectors")
    print("  📁 results/gemma_validation_test.json - Validation test results")
    print("\nNext steps:")
    print("  1. Transfer these vectors to other models")
    print("  2. Test cross-model compatibility")
    print("  3. Train linear adapters if needed")

if __name__ == "__main__":
    main()