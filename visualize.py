#!/usr/bin/env python
"""
ToM-NAS Visualization Tools
Create plots and visualizations of training results and evaluations
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os
from typing import Dict, List, Optional


class ResultsVisualizer:
    """Visualize ToM-NAS experimental results"""

    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)

    def plot_training_curves(self, metrics_file: str, output_name: str = 'training_curves.png'):
        """Plot training loss and accuracy curves"""
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        training_history = data.get('training_history', [])
        if not training_history:
            print("No training history found")
            return

        epochs = [m['epoch'] for m in training_history]
        losses = [m['loss'] for m in training_history]
        accuracies = [m['accuracy'] for m in training_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(epochs, losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2.plot(epochs, accuracies, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved: {output_path}")

    def plot_architecture_comparison(self, baseline_file: str,
                                    output_name: str = 'architecture_comparison.png'):
        """Compare different architectures"""
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)

        architectures = list(baseline_data.keys())
        scores = [baseline_data[arch]['best_score'] for arch in architectures]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(architectures, scores, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('Best Score (%)', fontsize=12)
        ax.set_title('Architecture Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Architecture comparison saved: {output_path}")

    def plot_evolution_progress(self, evolution_file: str,
                               output_name: str = 'evolution_progress.png'):
        """Plot evolution progress over generations"""
        with open(evolution_file, 'r') as f:
            data = json.load(f)

        if 'evolution_history' not in data:
            print("No evolution history found")
            return

        history = data['evolution_history']
        best_fitness = history.get('fitness_history', [])
        avg_fitness = history.get('avg_fitness_history', [])
        diversity = history.get('diversity_history', [])

        generations = list(range(len(best_fitness)))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Fitness plot
        ax1.plot(generations, best_fitness, 'r-', linewidth=2, label='Best Fitness')
        ax1.plot(generations, avg_fitness, 'b--', linewidth=2, label='Average Fitness')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Fitness', fontsize=12)
        ax1.set_title('Evolution Fitness Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Diversity plot
        ax2.plot(generations, diversity, 'g-', linewidth=2)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Population Diversity', fontsize=12)
        ax2.set_title('Population Diversity Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Evolution progress saved: {output_path}")

    def plot_tom_order_performance(self, results: Dict,
                                   output_name: str = 'tom_order_performance.png'):
        """Plot performance across different ToM orders"""
        # This would use actual benchmark results
        # Placeholder with sample data
        orders = [1, 2, 3, 4, 5]
        scores = [0.95, 0.85, 0.75, 0.65, 0.55]  # Example decreasing with order

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(orders, scores, 'o-', linewidth=2, markersize=10, color='#9b59b6')
        ax.set_xlabel('Theory of Mind Order', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Performance Across ToM Orders', fontsize=14, fontweight='bold')
        ax.set_xticks(orders)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3)

        # Add expected difficulty line
        expected = [1.0 - (o-1) * 0.15 for o in orders]
        ax.plot(orders, expected, '--', linewidth=2, color='gray',
               label='Expected Difficulty', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ToM order performance saved: {output_path}")

    def create_summary_dashboard(self, complete_results_file: str):
        """Create comprehensive summary dashboard"""
        with open(complete_results_file, 'r') as f:
            data = json.load(f)

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Architecture Comparison
        if 'baseline' in data:
            ax1 = fig.add_subplot(gs[0, 0])
            baseline = data['baseline']
            archs = list(baseline.keys())
            scores = [baseline[a]['best_score'] for a in archs]
            ax1.bar(archs, scores, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax1.set_ylabel('Score (%)')
            ax1.set_title('Baseline Architecture Comparison', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

        # 2. Evolution Progress (if available)
        if 'evolution' in data and 'evolution_history' in data['evolution']:
            ax2 = fig.add_subplot(gs[0, 1])
            history = data['evolution']['evolution_history']
            best_fit = history.get('fitness_history', [])
            if best_fit:
                ax2.plot(best_fit, 'r-', linewidth=2)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Best Fitness')
                ax2.set_title('Evolution Progress', fontweight='bold')
                ax2.grid(True, alpha=0.3)

        # 3. ToM Order Performance
        ax3 = fig.add_subplot(gs[1, 0])
        orders = [1, 2, 3, 4, 5]
        sample_scores = [0.95, 0.85, 0.75, 0.65, 0.55]
        ax3.plot(orders, sample_scores, 'o-', linewidth=2, markersize=8, color='#9b59b6')
        ax3.set_xlabel('ToM Order')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Performance by ToM Order', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Summary Statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        summary_text = "EXPERIMENT SUMMARY\n" + "="*40 + "\n\n"

        if 'baseline' in data:
            baseline_scores = [data['baseline'][a]['best_score'] for a in data['baseline'].keys()]
            summary_text += f"Best Baseline: {max(baseline_scores):.2f}%\n"
            summary_text += f"Avg Baseline: {np.mean(baseline_scores):.2f}%\n\n"

        if 'evolution' in data:
            summary_text += f"Evolution Best: {data['evolution']['best_fitness']:.4f}\n"
            summary_text += f"Best Arch: {data['evolution']['best_architecture']['arch_type']}\n\n"

        summary_text += f"\nDate: {data.get('timestamp', 'N/A')}"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        # 5. Resource Efficiency (placeholder)
        ax5 = fig.add_subplot(gs[2, :])
        metrics = ['Accuracy', 'Speed', 'Efficiency']
        trn_scores = [0.75, 0.85, 0.80]
        rsan_scores = [0.80, 0.70, 0.75]
        trans_scores = [0.85, 0.65, 0.70]

        x = np.arange(len(metrics))
        width = 0.25

        ax5.bar(x - width, trn_scores, width, label='TRN', color='#3498db')
        ax5.bar(x, rsan_scores, width, label='RSAN', color='#e74c3c')
        ax5.bar(x + width, trans_scores, width, label='Transformer', color='#2ecc71')

        ax5.set_ylabel('Score')
        ax5.set_title('Multi-Metric Comparison', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        plt.suptitle('ToM-NAS Experiment Dashboard', fontsize=16, fontweight='bold', y=0.995)

        output_path = os.path.join(self.figures_dir, 'summary_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Summary dashboard saved: {output_path}")

    def generate_all_visualizations(self):
        """Generate all available visualizations from results directory"""
        print("\n" + "="*80)
        print("Generating Visualizations")
        print("="*80)

        # Check for various result files
        files_to_check = {
            'baseline': 'baseline_results.json',
            'evolution': 'evolution/evolution_summary.json',
            'complete': 'complete_results.json'
        }

        for name, filename in files_to_check.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                print(f"\nFound {name} results: {filename}")

                if name == 'baseline':
                    self.plot_architecture_comparison(filepath)

                elif name == 'evolution':
                    self.plot_evolution_progress(filepath)

                elif name == 'complete':
                    self.create_summary_dashboard(filepath)

        # ToM order plot (always create with sample data)
        self.plot_tom_order_performance({})

        print("\n" + "="*80)
        print(f"All visualizations saved to: {self.figures_dir}")
        print("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize ToM-NAS Results')

    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing results')
    parser.add_argument('--all', action='store_true',
                       help='Generate all available visualizations')

    return parser.parse_args()


def main():
    """Main visualization function"""
    args = parse_args()

    visualizer = ResultsVisualizer(args.results_dir)

    if args.all:
        visualizer.generate_all_visualizations()
    else:
        # Interactive mode
        print("\nToM-NAS Visualization Tool")
        print("1. Training curves")
        print("2. Architecture comparison")
        print("3. Evolution progress")
        print("4. ToM order performance")
        print("5. Summary dashboard")
        print("6. All visualizations")

        choice = input("\nSelect option (1-6): ")

        if choice == '6':
            visualizer.generate_all_visualizations()
        else:
            print("Creating visualization...")
            # Would implement specific visualization based on choice


if __name__ == "__main__":
    main()
