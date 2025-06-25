#!/usr/bin/env python3
"""
Main script to run the complete preference vector transfer experiment.

This script orchestrates the entire research pipeline from data preparation
to final analysis and visualization.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import EXPERIMENT_CONFIG, get_model_config
from src.experiment_pipeline import ExperimentPipeline, run_lightweight_experiment
from src.visualization import visualize_experiment_results
from src.data_preparation import create_preference_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the experimental environment."""
    # Check for CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("adapters", exist_ok=True)
    
    # Check disk space (basic check)
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (1024**3)
    logger.info(f"Available disk space: {free_gb} GB")
    
    if free_gb < 10:
        logger.warning("Low disk space - models and results may require significant storage")

def run_full_experiment(config_overrides=None):
    """Run the complete preference vector transfer experiment."""
    logger.info("Starting full preference vector transfer experiment")
    
    # Override configuration if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(EXPERIMENT_CONFIG, key):
                setattr(EXPERIMENT_CONFIG, key, value)
                logger.info(f"Overrode config: {key} = {value}")
    
    try:
        # Initialize pipeline
        pipeline = ExperimentPipeline(EXPERIMENT_CONFIG)
        
        # Run complete experiment
        results = pipeline.run_complete_experiment(save_results=True)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualize_experiment_results(pipeline.results_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment ID: {results.experiment_id}")
        print(f"Results saved to: {pipeline.results_dir}")
        
        # Print key findings
        eval_summary = results.evaluation_summary
        if eval_summary:
            print(f"\nKEY FINDINGS:")
            print(f"- Source validation score: {eval_summary.get('source_model_validation', {}).get('score', 0):.3f}")
            print(f"- Direct transfer success rate: {eval_summary.get('overall_findings', {}).get('direct_transfer_success_rate', 0):.2%}")
            print(f"- Adapter transfer success rate: {eval_summary.get('overall_findings', {}).get('adapter_transfer_success_rate', 0):.2%}")
        
        # Print conclusions
        conclusions = results.conclusions
        if conclusions.get('findings'):
            print(f"\nCONCLUSIONS:")
            for finding in conclusions['findings'][:3]:  # Show top 3 findings
                print(f"- {finding}")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

def run_quick_test():
    """Run a quick test with minimal resources."""
    logger.info("Running quick test experiment")
    
    try:
        results = run_lightweight_experiment()
        if results:
            print(f"\nQuick test completed successfully!")
            print(f"Results: {results.experiment_id}")
            return results
        else:
            print("Quick test failed")
            return None
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return None

def validate_setup():
    """Validate that the setup is correct for running experiments."""
    logger.info("Validating experimental setup...")
    
    issues = []
    
    # Check Python packages
    try:
        import transformers
        import torch
        import datasets
        logger.info(f"✓ Core packages available (transformers {transformers.__version__})")
    except ImportError as e:
        issues.append(f"Missing required package: {e}")
    
    # Check model access
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        logger.info("✓ Model access working")
    except Exception as e:
        issues.append(f"Model access issue: {e}")
    
    # Check disk space
    import shutil
    free_gb = shutil.disk_usage(".")[2] // (1024**3)
    if free_gb < 5:
        issues.append(f"Low disk space: {free_gb} GB available")
    else:
        logger.info(f"✓ Sufficient disk space: {free_gb} GB")
    
    # Check memory
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✓ GPU memory: {memory_gb:.1f} GB")
        if memory_gb < 8:
            issues.append(f"Limited GPU memory: {memory_gb:.1f} GB")
    else:
        logger.info("! No GPU available - will use CPU (slower)")
    
    if issues:
        print("\nSETUP ISSUES FOUND:")
        for issue in issues:
            print(f"- {issue}")
        return False
    else:
        print("\n✓ Setup validation passed - ready to run experiments!")
        return True

def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run preference vector transfer experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --validate                    # Check setup
  python run_experiment.py --quick-test                  # Quick test
  python run_experiment.py --full                        # Full experiment
  python run_experiment.py --full --source-model gpt2    # Custom source model
        """
    )
    
    parser.add_argument('--validate', action='store_true',
                       help='Validate setup and requirements')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with lightweight models')
    parser.add_argument('--full', action='store_true',
                       help='Run full experiment')
    parser.add_argument('--visualize-only', type=str, metavar='RESULTS_DIR',
                       help='Only generate visualizations for existing results')
    
    # Configuration overrides
    parser.add_argument('--source-model', type=str,
                       help='Override source model name')
    parser.add_argument('--target-models', nargs='+',
                       help='Override target model names')
    parser.add_argument('--num-pairs', type=int,
                       help='Override number of training pairs')
    parser.add_argument('--intervention-layer', type=int,
                       help='Override intervention layer index')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Handle different modes
    if args.validate:
        validate_setup()
        return
    
    if args.visualize_only:
        if os.path.exists(args.visualize_only):
            visualize_experiment_results(args.visualize_only)
            print(f"Visualizations generated for {args.visualize_only}")
        else:
            print(f"Results directory not found: {args.visualize_only}")
        return
    
    if args.quick_test:
        if not validate_setup():
            print("Setup validation failed - cannot run experiment")
            return
        run_quick_test()
        return
    
    if args.full:
        if not validate_setup():
            print("Setup validation failed - cannot run experiment")
            return
        
        # Prepare configuration overrides
        config_overrides = {}
        if args.source_model:
            config_overrides['source_model'] = args.source_model
        if args.target_models:
            config_overrides['target_models'] = args.target_models
        if args.num_pairs:
            config_overrides['num_training_pairs'] = args.num_pairs
        if args.intervention_layer:
            config_overrides['intervention_layer'] = args.intervention_layer
        
        run_full_experiment(config_overrides)
        return
    
    # Default: show help
    parser.print_help()

if __name__ == "__main__":
    main()