#!/usr/bin/env python3
"""
Main Experiment Script for ToM-NAS

Run complete experiment suite for testing hypotheses about
neural architecture requirements for Theory of Mind.

Usage:
    python scripts/run_experiments.py --mode full
    python scripts/run_experiments.py --mode quick --tasks tomi hitom_2
    python scripts/run_experiments.py --mode ablation
    python scripts/run_experiments.py --mode baseline
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.experiment_runner import (
    ExperimentConfig,
    NASExperimentRunner,
    create_dummy_fitness_factory,
    run_quick_experiment,
)
from src.evolution.nas_bench import run_benchmark_baseline_study
from src.evolution.ablations import AblationStudy, AblationConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ToM-NAS experiments"
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'full', 'hypothesis', 'method_comparison', 'ablation', 'baseline'],
        help='Experiment mode'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='Specific tasks to run'
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=['CMA_ES'],
        help='NAS methods to use'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456],
        help='Random seeds for multiple runs'
    )

    parser.add_argument(
        '--population',
        type=int,
        default=128,
        help='Population size for evolution'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Number of generations'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory'
    )

    return parser.parse_args()


def run_full_suite(args):
    """Run complete experiment suite"""
    print("\n" + "#"*60)
    print("FULL EXPERIMENT SUITE")
    print("#"*60)

    config = ExperimentConfig(
        population_size=args.population,
        num_generations=args.generations,
        seeds=args.seeds,
        output_dir=f"{args.output}/full_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    runner = NASExperimentRunner(config, create_dummy_fitness_factory())
    results = runner.run_task_suite(
        tasks=args.tasks,
        methods=args.methods,
        seeds=args.seeds,
    )

    runner.print_summary()
    return results


def run_hypothesis_focused(args):
    """Run hypothesis-focused experiments"""
    print("\n" + "#"*60)
    print("HYPOTHESIS-FOCUSED EXPERIMENTS")
    print("#"*60)

    config = ExperimentConfig(
        population_size=args.population,
        num_generations=args.generations,
        seeds=args.seeds,
        output_dir=f"{args.output}/hypothesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    runner = NASExperimentRunner(config, create_dummy_fitness_factory())
    results = runner.run_hypothesis_focused_suite()

    runner.print_summary()
    return results


def run_method_comparison(args):
    """Compare NAS methods"""
    print("\n" + "#"*60)
    print("NAS METHOD COMPARISON")
    print("#"*60)

    config = ExperimentConfig(
        population_size=args.population,
        num_generations=args.generations,
        nas_methods=['CMA_ES', 'OpenES', 'PGPE', 'RandomSearch'],
        output_dir=f"{args.output}/methods_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    runner = NASExperimentRunner(config, create_dummy_fitness_factory())
    results = runner.run_method_comparison()

    runner.print_summary()
    return results


def run_ablation_study(args):
    """Run ablation studies"""
    print("\n" + "#"*60)
    print("ABLATION STUDIES")
    print("#"*60)

    tasks = args.tasks or ['simple_sequence', 'tomi', 'hitom_2', 'hitom_4']
    ablations = ['full', 'no_skip', 'no_attention', 'no_recurrence']

    config = AblationConfig(
        task_types=tasks,
        ablation_types=ablations,
        seeds=args.seeds[:3],  # Fewer seeds for ablations
        population_size=args.population // 2,
        num_generations=args.generations // 2,
    )

    study = AblationStudy(
        config,
        create_dummy_fitness_factory(),
        output_dir=f"{args.output}/ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    results = study.run_full_study()

    print("\n" + results.get('summary', ''))
    return results


def run_baseline_study(args):
    """Run NAS-Bench baseline study"""
    print("\n" + "#"*60)
    print("NAS-BENCH BASELINE STUDY")
    print("#"*60)

    results = run_benchmark_baseline_study(n_samples=5000, seed=42)

    print("\n--- Results ---")
    print(f"NAS-Bench-201 skip correlation: {results['nasbench201']['correlations']['skip_vs_accuracy']:.4f}")
    print(f"Top 10% avg skips: {results['nasbench201']['top_vs_bottom']['top_10_avg_skip_connections']:.2f}")

    return results


def run_quick_mode(args):
    """Run quick experiment for testing"""
    print("\n" + "#"*60)
    print("QUICK TEST MODE")
    print("#"*60)

    tasks = args.tasks or ['simple_sequence', 'tomi', 'hitom_2']

    results = run_quick_experiment(
        tasks=tasks,
        output_dir=f"{args.output}/quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    print("\nQuick experiment complete!")
    print(f"Tasks: {tasks}")
    print(f"Total experiments: {results.get('total_experiments', 0)}")

    return results


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("ToM-NAS Experiment Runner")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print("="*60)

    # Create output directory
    Path(args.output).mkdir(exist_ok=True, parents=True)

    if args.mode == 'quick':
        results = run_quick_mode(args)
    elif args.mode == 'full':
        results = run_full_suite(args)
    elif args.mode == 'hypothesis':
        results = run_hypothesis_focused(args)
    elif args.mode == 'method_comparison':
        results = run_method_comparison(args)
    elif args.mode == 'ablation':
        results = run_ablation_study(args)
    elif args.mode == 'baseline':
        results = run_baseline_study(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
