#!/usr/bin/env python
"""
ToM-NAS Experiment Runner
Run complete experiments including evolution, training, and evaluation
"""
import torch
import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.world.social_world import SocialWorld4
from src.evolution.nas_engine import NASEngine, EvolutionConfig
from src.evaluation.benchmarks import BenchmarkSuite
from src.evaluation.metrics import MetricsTracker, ResultsAggregator
from train import ToMTrainer


class ExperimentRunner:
    """Run complete ToM-NAS experiments"""

    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = config.get('results_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.results_aggregator = ResultsAggregator()

    def run_baseline_experiment(self) -> Dict:
        """Run baseline experiment with fixed architecture"""
        print("\n" + "="*80)
        print("Running Baseline Experiment")
        print("="*80)

        results = {}

        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            print(f"\n{'='*80}")
            print(f"Training {arch_type}")
            print(f"{'='*80}")

            config = {
                'architecture': arch_type,
                'num_agents': 6,
                'ontology_dim': 181,
                'input_dim': 191,
                'hidden_dim': 128,
                'max_belief_order': 5,
                'num_zombies': 2,
                'batch_size': 32,
                'sequence_length': 20,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'device': self.config.get('device', 'cpu'),
                'checkpoint_dir': os.path.join(self.results_dir, f'baseline_{arch_type}'),
                'batches_per_epoch': 50,
                'eval_interval': 10,
                'checkpoint_interval': 20,
                'early_stopping': True,
                'patience': 20
            }

            trainer = ToMTrainer(config)
            best_score = trainer.train(self.config.get('baseline_epochs', 50))

            results[arch_type] = {
                'best_score': best_score,
                'architecture': arch_type,
                'final_metrics': trainer.metrics.get_training_summary()
            }

            self.results_aggregator.add_run(results[arch_type], f'baseline_{arch_type}')

        # Save baseline results
        with open(os.path.join(self.results_dir, 'baseline_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_evolution_experiment(self) -> Dict:
        """Run evolution/NAS experiment"""
        print("\n" + "="*80)
        print("Running Evolution Experiment")
        print("="*80)

        # Initialize world and belief network for evolution
        world = SocialWorld4(
            num_agents=6,
            ontology_dim=181,
            num_zombies=2
        )

        belief_network = BeliefNetwork(
            num_agents=6,
            ontology_dim=181,
            max_order=5
        )

        # Evolution configuration
        evo_config = EvolutionConfig(
            population_size=self.config.get('population_size', 20),
            num_generations=self.config.get('num_generations', 50),
            elite_size=2,
            tournament_size=3,
            mutation_rate=0.1,
            crossover_rate=0.7,
            weight_mutation_prob=0.3,
            use_speciation=True,
            use_coevolution=True,
            fitness_episodes=5,
            device=self.config.get('device', 'cpu'),
            checkpoint_interval=10
        )

        # Create and run NAS engine
        nas_engine = NASEngine(evo_config, world, belief_network)
        best_individual = nas_engine.run(evo_config.num_generations)

        # Save evolved model
        evolution_dir = os.path.join(self.results_dir, 'evolution')
        os.makedirs(evolution_dir, exist_ok=True)
        nas_engine.save_checkpoint(os.path.join(evolution_dir, 'final_evolution.pt'))

        # Get summary
        evolution_summary = nas_engine.get_evolution_summary()

        with open(os.path.join(evolution_dir, 'evolution_summary.json'), 'w') as f:
            json.dump(evolution_summary, f, indent=2)

        return {
            'best_fitness': evolution_summary['best_fitness'],
            'best_architecture': evolution_summary['best_architecture'],
            'evolution_history': evolution_summary
        }

    def run_comparison_experiment(self) -> Dict:
        """Run complete comparison between baseline and evolved"""
        print("\n" + "="*80)
        print("Running Complete Comparison Experiment")
        print("="*80)

        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }

        # Run baseline
        baseline_results = self.run_baseline_experiment()
        results['baseline'] = baseline_results

        # Run evolution
        if self.config.get('run_evolution', True):
            evolution_results = self.run_evolution_experiment()
            results['evolution'] = evolution_results

        # Generate comparison report
        self._generate_comparison_report(results)

        # Save complete results
        with open(os.path.join(self.results_dir, 'complete_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _generate_comparison_report(self, results: Dict):
        """Generate text report comparing results"""
        report_path = os.path.join(self.results_dir, 'comparison_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ToM-NAS Experiment Comparison Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Date: {results['timestamp']}\n\n")

            # Baseline results
            f.write("BASELINE ARCHITECTURES\n")
            f.write("-"*80 + "\n")
            for arch, res in results.get('baseline', {}).items():
                f.write(f"\n{arch}:\n")
                f.write(f"  Best Score: {res['best_score']:.2f}%\n")

            # Evolution results
            if 'evolution' in results:
                f.write("\n\nEVOLVED ARCHITECTURE\n")
                f.write("-"*80 + "\n")
                evo = results['evolution']
                f.write(f"  Best Fitness: {evo['best_fitness']:.4f}\n")
                f.write(f"  Architecture: {evo['best_architecture']['arch_type']}\n")
                f.write(f"  Hidden Dim: {evo['best_architecture']['hidden_dim']}\n")
                f.write(f"  Num Layers: {evo['best_architecture']['num_layers']}\n")

            # Summary
            f.write("\n\nSUMMARY\n")
            f.write("-"*80 + "\n")

            baseline_scores = [res['best_score'] for res in results.get('baseline', {}).values()]
            if baseline_scores:
                f.write(f"Best Baseline Score: {max(baseline_scores):.2f}%\n")

            if 'evolution' in results:
                f.write(f"Evolution Best: {results['evolution']['best_fitness']:.4f}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"\nComparison report saved: {report_path}")

    def run_ablation_study(self) -> Dict:
        """Run ablation study to understand component contributions"""
        print("\n" + "="*80)
        print("Running Ablation Study")
        print("="*80)

        configs_to_test = [
            {'name': 'full_system', 'hidden_dim': 128, 'num_layers': 2},
            {'name': 'reduced_hidden', 'hidden_dim': 64, 'num_layers': 2},
            {'name': 'single_layer', 'hidden_dim': 128, 'num_layers': 1},
            {'name': 'increased_hidden', 'hidden_dim': 256, 'num_layers': 2},
        ]

        results = {}

        for test_config in configs_to_test:
            print(f"\nTesting: {test_config['name']}")
            print("-"*80)

            config = {
                'architecture': 'TRN',
                'num_agents': 6,
                'ontology_dim': 181,
                'input_dim': 191,
                'hidden_dim': test_config['hidden_dim'],
                'num_layers': test_config['num_layers'],
                'max_belief_order': 5,
                'num_zombies': 2,
                'batch_size': 32,
                'sequence_length': 20,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'device': self.config.get('device', 'cpu'),
                'checkpoint_dir': os.path.join(self.results_dir, f"ablation_{test_config['name']}"),
                'batches_per_epoch': 30,
                'eval_interval': 10,
                'early_stopping': True,
                'patience': 15
            }

            trainer = ToMTrainer(config)
            best_score = trainer.train(30)  # Shorter for ablation

            results[test_config['name']] = {
                'config': test_config,
                'best_score': best_score
            }

        # Save ablation results
        with open(os.path.join(self.results_dir, 'ablation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run ToM-NAS Experiments')

    parser.add_argument('--experiment', type=str, default='baseline',
                       choices=['baseline', 'evolution', 'comparison', 'ablation'],
                       help='Type of experiment to run')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory for results')
    parser.add_argument('--baseline-epochs', type=int, default=50,
                       help='Epochs for baseline training')
    parser.add_argument('--population-size', type=int, default=20,
                       help='Evolution population size')
    parser.add_argument('--num-generations', type=int, default=50,
                       help='Number of evolution generations')
    parser.add_argument('--run-evolution', action='store_true',
                       help='Include evolution in comparison')

    return parser.parse_args()


def main():
    """Main experiment runner"""
    args = parse_args()

    config = {
        'device': args.device,
        'results_dir': args.results_dir,
        'baseline_epochs': args.baseline_epochs,
        'population_size': args.population_size,
        'num_generations': args.num_generations,
        'run_evolution': args.run_evolution
    }

    runner = ExperimentRunner(config)

    if args.experiment == 'baseline':
        results = runner.run_baseline_experiment()
    elif args.experiment == 'evolution':
        results = runner.run_evolution_experiment()
    elif args.experiment == 'comparison':
        results = runner.run_comparison_experiment()
    elif args.experiment == 'ablation':
        results = runner.run_ablation_study()

    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
