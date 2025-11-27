"""
Run Scientific Validation on ToM-NAS Architectures

This script runs rigorous scientific validation following the methodology
documented in SCIENTIFIC_METHODOLOGY.md

Usage:
    python run_scientific_validation.py [--architecture ARCH] [--seed SEED]

Arguments:
    --architecture: TRN, RSAN, Transformer, or all (default: all)
    --seed: Random seed for reproducibility (default: 42)
    --output-dir: Directory for results (default: validation_results/)
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.core.ontology import SoulMapOntology
from src.agents.architectures import TRN, RSAN, TransformerToM
from src.evaluation.scientific_validation import (
    run_scientific_validation,
    ToMBenchmark,
    StatisticalAnalyzer,
    AblationStudy,
    ReproducibilityManager
)

def main():
    parser = argparse.ArgumentParser(description='Run scientific validation on ToM architectures')
    parser.add_argument('--architecture', type=str, default='all',
                       choices=['TRN', 'RSAN', 'Transformer', 'all'],
                       help='Architecture to validate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Output directory for results')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of trials per test')

    args = parser.parse_args()

    # Set up reproducibility
    print("=" * 70)
    print("SCIENTIFIC VALIDATION OF ToM-NAS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture: {args.architecture}")
    print(f"  Random Seed: {args.seed}")
    print(f"  Trials per test: {args.n_trials}")
    print(f"  Output directory: {args.output_dir}")

    repro = ReproducibilityManager(seed=args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save configuration
    config_file = output_dir / "experiment_config.json"
    repro.save_config(str(config_file))
    print(f"\n✓ Configuration saved to {config_file}")

    # Initialize ontology
    ontology = SoulMapOntology()

    # Determine which architectures to test
    if args.architecture == 'all':
        architectures_to_test = ['TRN', 'RSAN', 'Transformer']
    else:
        architectures_to_test = [args.architecture]

    # Results storage
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': vars(args),
        'architectures': {}
    }

    # Test each architecture
    for arch_name in architectures_to_test:
        print("\n" + "=" * 70)
        print(f"TESTING: {arch_name}")
        print("=" * 70)

        # Create agent
        hidden_dim = 256
        input_dim = ontology.total_dims

        if arch_name == 'TRN':
            agent = TRN(input_dim=input_dim, hidden_dim=hidden_dim)
        elif arch_name == 'RSAN':
            agent = RSAN(input_dim=input_dim, hidden_dim=hidden_dim, num_recursions=3)
        elif arch_name == 'Transformer':
            agent = TransformerToM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=8,
                num_layers=4
            )

        agent.eval()  # Evaluation mode

        # Run validation
        results = run_scientific_validation(
            agent,
            arch_name,
            output_file=str(output_dir / f"{arch_name}_validation.json")
        )

        all_results['architectures'][arch_name] = results

    # Comparative analysis if multiple architectures
    if len(architectures_to_test) > 1:
        print("\n" + "=" * 70)
        print("COMPARATIVE ANALYSIS")
        print("=" * 70)

        analyzer = StatisticalAnalyzer()

        # Compare first-order ToM performance
        print("\n1. Sally-Anne Test Comparison:")
        sally_anne_scores = {}

        for arch_name in architectures_to_test:
            sally_anne_result = all_results['architectures'][arch_name]['tests'][0]
            sally_anne_scores[arch_name] = sally_anne_result['accuracy']
            print(f"   {arch_name}: {sally_anne_result['accuracy']:.1%}")

        # Pairwise comparisons
        if len(architectures_to_test) == 2:
            arch1, arch2 = architectures_to_test
            score1 = sally_anne_scores[arch1]
            score2 = sally_anne_scores[arch2]

            # Note: This is simplified - full implementation would use
            # trial-level data for proper statistical testing
            if abs(score1 - score2) > 0.05:
                better = arch1 if score1 > score2 else arch2
                worse = arch2 if score1 > score2 else arch1
                diff = abs(score1 - score2)
                print(f"\n   → {better} outperforms {worse} by {diff:.1%}")
            else:
                print(f"\n   → No substantial difference ({abs(score1-score2):.1%})")

        # Overall summary
        print("\n2. Overall Performance Summary:")

        for arch_name in architectures_to_test:
            summary = all_results['architectures'][arch_name]['summary']
            print(f"\n   {arch_name}:")
            print(f"     Mean accuracy: {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
            print(f"     Tests above chance: {summary['tests_above_chance']}/5")
            print(f"     Tests above human: {summary['tests_above_human']}/5")

        # Save comparative results
        comparison_file = output_dir / "comparative_analysis.json"
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Comparative analysis saved to {comparison_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to: {output_dir}/")
    print("\nFiles generated:")
    for arch_name in architectures_to_test:
        print(f"  - {arch_name}_validation.json")

    if len(architectures_to_test) > 1:
        print(f"  - comparative_analysis.json")

    print(f"  - experiment_config.json")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDELINES")
    print("=" * 70)
    print("\nBefore making scientific claims, verify:")
    print("  1. Statistical significance (p < 0.05 with corrections)")
    print("  2. Effect size is meaningful (Cohen's d > 0.3)")
    print("  3. Confidence intervals reported")
    print("  4. Performance replicable across multiple runs")
    print("  5. Limitations acknowledged")
    print("\nSee SCIENTIFIC_METHODOLOGY.md for complete guidelines.")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
