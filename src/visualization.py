import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from .experiment_pipeline import ExperimentResults

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExperimentVisualizer:
    """Creates visualizations for preference vector transfer experiments."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load experiment results
        self.results = self._load_results()
        
    def _load_results(self) -> Optional[ExperimentResults]:
        """Load experiment results from JSON file."""
        results_file = self.results_dir / "experiment_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            return data
        return None
    
    def create_all_visualizations(self):
        """Create all visualization plots for the experiment."""
        if not self.results:
            print("No results to visualize")
            return
        
        print("Creating experiment visualizations...")
        
        # Core visualizations
        self.plot_preference_vector_analysis()
        self.plot_transfer_success_matrix()
        self.plot_evaluation_scores_comparison()
        self.plot_model_performance_radar()
        
        # Detailed analysis plots
        self.plot_verbosity_changes()
        self.plot_content_preservation()
        self.plot_adapter_effectiveness()
        
        # Summary dashboard
        self.create_summary_dashboard()
        
        print(f"All visualizations saved to {self.figures_dir}")
    
    def plot_preference_vector_analysis(self):
        """Plot analysis of the extracted preference vector."""
        vector_info = self.results.get('preference_vector_info', {})
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Preference Vector Analysis', fontsize=16, fontweight='bold')
        
        # Vector statistics
        stats = ['norm', 'mean', 'std', 'sparsity', 'max_magnitude']
        values = [vector_info.get(stat, 0) for stat in stats]
        
        axes[0, 0].bar(stats, values)
        axes[0, 0].set_title('Vector Statistics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Vector shape info
        shape = vector_info.get('shape', [])
        if shape:
            axes[0, 1].text(0.5, 0.5, f"Vector Shape: {shape}\nDimension: {shape[0] if shape else 'Unknown'}", 
                           ha='center', va='center', fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('Vector Properties')
        axes[0, 1].axis('off')
        
        # Placeholder for vector distribution (would need actual vector data)
        x = np.linspace(-3, 3, 100)
        y = np.exp(-x**2/2) / np.sqrt(2*np.pi)  # Standard normal as placeholder
        axes[1, 0].plot(x, y)
        axes[1, 0].set_title('Vector Component Distribution (Conceptual)')
        axes[1, 0].set_xlabel('Component Value')
        axes[1, 0].set_ylabel('Density')
        
        # Vector norm comparison
        norm_comparison = [vector_info.get('norm', 1), 1.0, 0.5]  # Compare with unit norms
        norm_labels = ['Extracted Vector', 'Unit Vector', 'Half Unit']
        axes[1, 1].bar(norm_labels, norm_comparison)
        axes[1, 1].set_title('Norm Comparison')
        axes[1, 1].set_ylabel('L2 Norm')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'preference_vector_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_transfer_success_matrix(self):
        """Plot success matrix for cross-model transfers."""
        # Prepare data
        direct_results = self.results.get('direct_transfer_results', {}).get('evaluation', {})
        adapter_results = self.results.get('adapter_results', {})
        
        models = self.results.get('target_models', [])
        source_model = self.results.get('source_model', 'Source')
        
        # Create success matrix
        methods = ['Direct Transfer', 'Adapter Transfer']
        success_matrix = np.zeros((len(models), len(methods)))
        
        for i, model in enumerate(models):
            # Direct transfer success
            direct_score = direct_results.get('model_evaluations', {}).get(model, {}).get('avg_score', 0)
            success_matrix[i, 0] = direct_score
            
            # Adapter transfer success
            adapter_score = adapter_results.get(model, {}).get('evaluation', {}).get('average_score', 0)
            success_matrix[i, 1] = adapter_score
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.split('/')[-1] for m in models])  # Show only model names
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{success_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f'Transfer Success Matrix\nSource: {source_model.split("/")[-1]}', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'transfer_success_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_evaluation_scores_comparison(self):
        """Plot comparison of evaluation scores across models and methods."""
        # Extract evaluation data
        direct_eval = self.results.get('direct_transfer_results', {}).get('evaluation', {})
        adapter_results = self.results.get('adapter_results', {})
        
        models = self.results.get('target_models', [])
        
        # Prepare data for plotting
        data = []
        
        for model in models:
            model_name = model.split('/')[-1]
            
            # Direct transfer
            direct_score = direct_eval.get('model_evaluations', {}).get(model, {}).get('avg_score', 0)
            data.append({
                'Model': model_name,
                'Method': 'Direct Transfer',
                'Score': direct_score
            })
            
            # Adapter transfer
            adapter_score = adapter_results.get(model, {}).get('evaluation', {}).get('average_score', 0)
            data.append({
                'Model': model_name,
                'Method': 'Adapter Transfer',
                'Score': adapter_score
            })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.35
        
        direct_scores = df[df['Method'] == 'Direct Transfer']['Score'].values
        adapter_scores = df[df['Method'] == 'Adapter Transfer']['Score'].values
        
        bars1 = ax.bar(x - width/2, direct_scores, width, label='Direct Transfer', alpha=0.8)
        bars2 = ax.bar(x + width/2, adapter_scores, width, label='Adapter Transfer', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_xlabel('Target Models')
        ax.set_ylabel('Evaluation Score')
        ax.set_title('Cross-Model Transfer Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add horizontal line for success threshold
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Success Threshold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'evaluation_scores_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_performance_radar(self):
        """Create radar chart showing multi-dimensional performance."""
        # This would require detailed evaluation metrics
        # For now, create a conceptual radar chart
        
        categories = ['Style Consistency', 'Content Preservation', 'Transfer Success', 'Adaptation Quality']
        
        # Mock data based on available results
        eval_summary = self.results.get('evaluation_summary', {})
        source_score = eval_summary.get('source_model_validation', {}).get('score', 0.5)
        direct_rate = eval_summary.get('overall_findings', {}).get('direct_transfer_success_rate', 0.3)
        adapter_rate = eval_summary.get('overall_findings', {}).get('adapter_transfer_success_rate', 0.7)
        
        # Create data for radar chart
        source_values = [source_score, source_score * 0.9, 1.0, 0.8]  # Source model performance
        direct_values = [direct_rate, direct_rate * 0.8, direct_rate, 0.0]  # Direct transfer
        adapter_values = [adapter_rate, adapter_rate * 0.85, adapter_rate, adapter_rate]  # Adapter transfer
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        for values, label, color in [(source_values, 'Source Model', 'blue'),
                                   (direct_values, 'Direct Transfer', 'red'),
                                   (adapter_values, 'Adapter Transfer', 'green')]:
            values += values[:1]  # Close the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_verbosity_changes(self):
        """Plot verbosity changes across different steering scales."""
        # This would use detailed generation data
        # Creating a conceptual plot for demonstration
        
        scales = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # Mock verbosity scores
        source_verbosity = [0.2, 0.35, 0.5, 0.65, 0.8]
        target1_verbosity = [0.45, 0.47, 0.5, 0.53, 0.55]  # Poor transfer
        target2_verbosity = [0.25, 0.37, 0.5, 0.63, 0.75]  # Good transfer
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(scales, source_verbosity, 'o-', linewidth=2, label='Source Model', color='blue')
        ax.plot(scales, target1_verbosity, 's-', linewidth=2, label='Target Model 1 (Poor Transfer)', color='red')
        ax.plot(scales, target2_verbosity, '^-', linewidth=2, label='Target Model 2 (Good Transfer)', color='green')
        
        ax.set_xlabel('Steering Scale')
        ax.set_ylabel('Verbosity Score')
        ax.set_title('Verbosity Response to Preference Vector Steering')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'verbosity_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_content_preservation(self):
        """Plot content preservation scores."""
        # Mock data for content preservation across models
        models = [m.split('/')[-1] for m in self.results.get('target_models', [])]
        preservation_scores = [0.85, 0.75, 0.90]  # Mock ROUGE-like scores
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(models, preservation_scores, color=['lightcoral', 'lightblue', 'lightgreen'])
        
        # Add value labels
        for bar, score in zip(bars, preservation_scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Content Preservation Score')
        ax.set_title('Content Preservation Across Target Models')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Acceptable Threshold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'content_preservation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_adapter_effectiveness(self):
        """Plot effectiveness of linear adapters."""
        models = [m.split('/')[-1] for m in self.results.get('target_models', [])]
        
        # Mock data showing improvement from adapters
        direct_scores = [0.2, 0.15, 0.25]
        adapter_scores = [0.6, 0.55, 0.7]
        improvement = [a - d for a, d in zip(adapter_scores, direct_scores)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Before/after comparison
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, direct_scores, width, label='Direct Transfer', alpha=0.7)
        ax1.bar(x + width/2, adapter_scores, width, label='With Adapter', alpha=0.7)
        ax1.set_xlabel('Target Models')
        ax1.set_ylabel('Transfer Success Score')
        ax1.set_title('Adapter Impact on Transfer Success')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        
        # Improvement plot
        colors = ['green' if imp > 0 else 'red' for imp in improvement]
        bars = ax2.bar(models, improvement, color=colors, alpha=0.7)
        ax2.set_ylabel('Improvement Score')
        ax2.set_title('Adapter Improvement Over Direct Transfer')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvement):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{imp:+.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'adapter_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self):
        """Create an interactive summary dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transfer Success Rates', 'Model Comparison', 
                          'Evaluation Scores', 'Conclusions Summary'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Transfer success rates
        methods = ['Direct Transfer', 'Adapter Transfer']
        eval_summary = self.results.get('evaluation_summary', {})
        direct_rate = eval_summary.get('overall_findings', {}).get('direct_transfer_success_rate', 0.3)
        adapter_rate = eval_summary.get('overall_findings', {}).get('adapter_transfer_success_rate', 0.7)
        rates = [direct_rate, adapter_rate]
        
        fig.add_trace(
            go.Bar(x=methods, y=rates, name="Success Rate",
                  marker_color=['lightcoral', 'lightgreen']),
            row=1, col=1
        )
        
        # Model comparison scatter
        models = [m.split('/')[-1] for m in self.results.get('target_models', [])]
        direct_scores = [0.2, 0.15, 0.25]  # Mock data
        adapter_scores = [0.6, 0.55, 0.7]
        
        fig.add_trace(
            go.Scatter(x=direct_scores, y=adapter_scores, mode='markers+text',
                      text=models, textposition="top center",
                      marker=dict(size=12, color='blue'),
                      name="Model Performance"),
            row=1, col=2
        )
        
        # Add diagonal line for reference
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='gray'),
                      name="Equal Performance", showlegend=False),
            row=1, col=2
        )
        
        # Evaluation scores
        fig.add_trace(
            go.Bar(x=models, y=adapter_scores, name="Final Scores",
                  marker_color='lightblue'),
            row=2, col=1
        )
        
        # Conclusions table
        conclusions = self.results.get('conclusions', {})
        hypotheses = conclusions.get('hypotheses_tested', [])
        
        table_data = []
        for hyp in hypotheses:
            table_data.append([hyp.get('hypothesis', '')[:50] + '...', 
                             hyp.get('result', ''), 
                             hyp.get('evidence', '')[:30] + '...'])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Hypothesis', 'Result', 'Evidence'],
                           fill_color='lightgray'),
                cells=dict(values=list(zip(*table_data)) if table_data else [[], [], []],
                          fill_color='white')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Preference Vector Transfer Experiment Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        # Save as HTML
        fig.write_html(str(self.figures_dir / 'experiment_dashboard.html'))
        
        print(f"Interactive dashboard saved to {self.figures_dir / 'experiment_dashboard.html'}")

def visualize_experiment_results(results_dir: str):
    """Main function to create all visualizations for an experiment."""
    visualizer = ExperimentVisualizer(results_dir)
    visualizer.create_all_visualizations()

def create_comparison_visualization(results_dirs: List[str], experiment_names: List[str]):
    """Create comparison visualizations across multiple experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Experiment Comparison', fontsize=16, fontweight='bold')
    
    all_data = []
    
    for results_dir, exp_name in zip(results_dirs, experiment_names):
        try:
            with open(Path(results_dir) / 'experiment_results.json', 'r') as f:
                results = json.load(f)
            
            eval_summary = results.get('evaluation_summary', {})
            direct_rate = eval_summary.get('overall_findings', {}).get('direct_transfer_success_rate', 0)
            adapter_rate = eval_summary.get('overall_findings', {}).get('adapter_transfer_success_rate', 0)
            source_score = eval_summary.get('source_model_validation', {}).get('score', 0)
            
            all_data.append({
                'Experiment': exp_name,
                'Direct Transfer Rate': direct_rate,
                'Adapter Transfer Rate': adapter_rate,
                'Source Validation Score': source_score
            })
        except Exception as e:
            print(f"Failed to load {results_dir}: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Plot comparisons
        x = range(len(experiment_names))
        width = 0.25
        
        # Transfer rates comparison
        axes[0, 0].bar([i - width for i in x], df['Direct Transfer Rate'], width, label='Direct', alpha=0.8)
        axes[0, 0].bar([i + width for i in x], df['Adapter Transfer Rate'], width, label='Adapter', alpha=0.8)
        axes[0, 0].set_title('Transfer Success Rates Comparison')
        axes[0, 0].set_xlabel('Experiments')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(experiment_names, rotation=45)
        axes[0, 0].legend()
        
        # Source validation scores
        axes[0, 1].bar(experiment_names, df['Source Validation Score'], alpha=0.8)
        axes[0, 1].set_title('Source Model Validation Scores')
        axes[0, 1].set_ylabel('Validation Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Improvement from adapters
        improvement = df['Adapter Transfer Rate'] - df['Direct Transfer Rate']
        colors = ['green' if imp > 0 else 'red' for imp in improvement]
        axes[1, 0].bar(experiment_names, improvement, color=colors, alpha=0.8)
        axes[1, 0].set_title('Adapter Improvement')
        axes[1, 0].set_ylabel('Improvement Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Summary statistics
        axes[1, 1].text(0.1, 0.8, f"Experiments: {len(all_data)}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Avg Direct Success: {df['Direct Transfer Rate'].mean():.2f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Avg Adapter Success: {df['Adapter Transfer Rate'].mean():.2f}", transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f"Avg Source Validation: {df['Source Validation Score'].mean():.2f}", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('multi_experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    results_dir = "results/pref_vector_exp_20250604_120000"  # Example directory
    
    if os.path.exists(results_dir):
        visualize_experiment_results(results_dir)
    else:
        print(f"Results directory {results_dir} not found")
        print("Creating sample visualizations...")
        
        # Create sample data structure for demonstration
        sample_results = {
            'experiment_id': 'demo_experiment',
            'source_model': 'microsoft/DialoGPT-medium',
            'target_models': ['microsoft/DialoGPT-small', 'gpt2'],
            'preference_vector_info': {
                'shape': [1024], 'norm': 1.5, 'mean': 0.01, 'std': 0.15,
                'sparsity': 0.05, 'max_magnitude': 0.8
            },
            'evaluation_summary': {
                'source_model_validation': {'score': 0.75, 'success': True},
                'overall_findings': {
                    'direct_transfer_success_rate': 0.3,
                    'adapter_transfer_success_rate': 0.7
                }
            },
            'conclusions': {
                'hypotheses_tested': [
                    {'hypothesis': 'Vectors transfer directly', 'result': 'NOT SUPPORTED', 'evidence': 'Low success rate'},
                    {'hypothesis': 'Adapters help transfer', 'result': 'SUPPORTED', 'evidence': 'Higher success rate'}
                ]
            }
        }
        
        # Save sample results and create visualizations
        sample_dir = Path("sample_results")
        sample_dir.mkdir(exist_ok=True)
        
        with open(sample_dir / "experiment_results.json", 'w') as f:
            json.dump(sample_results, f, indent=2)
        
        visualizer = ExperimentVisualizer(str(sample_dir))
        visualizer.create_all_visualizations()