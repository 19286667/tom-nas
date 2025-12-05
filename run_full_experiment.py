#!/usr/bin/env python3
"""
ToM-NAS Complete Automated Experiment
=====================================
Runs everything automatically:
1. System validation
2. Full evolution experiment
3. Comprehensive analysis
4. Report generation
5. Results export

Just run: python run_full_experiment.py
"""
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# Configuration
CONFIG = {
    'population_size': 15,
    'generations': 25,
    'mutation_rate': 0.1,
    'elite_size': 2,
    'output_dir': 'experiment_results',
    'save_checkpoints': True,
    'verbose': True
}


def print_banner(text, char='='):
    """Print formatted banner."""
    width = 70
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_step(step_num, total, description):
    """Print step progress."""
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 50)


class AutomatedExperiment:
    """Fully automated ToM-NAS experiment."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.results = {}
        self.start_time = None
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)

    def run(self):
        """Run complete automated experiment."""
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        print_banner("ToM-NAS Automated Experiment")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")

        total_steps = 6

        # Step 1: System Validation
        print_step(1, total_steps, "Validating System")
        validation_passed = self.run_validation()
        if not validation_passed:
            print("\n[WARNING] Some validations failed - continuing anyway")

        # Step 2: Initialize Components
        print_step(2, total_steps, "Initializing Components")
        self.initialize_components()

        # Step 3: Run Evolution
        print_step(3, total_steps, "Running Evolution")
        self.run_evolution()

        # Step 4: Final Evaluation
        print_step(4, total_steps, "Final Evaluation")
        self.run_final_evaluation()

        # Step 5: Generate Report
        print_step(5, total_steps, "Generating Report")
        report_path = self.generate_report(timestamp)

        # Step 6: Save Results
        print_step(6, total_steps, "Saving Results")
        self.save_results(timestamp)

        # Summary
        self.print_summary(report_path)

        return self.results

    def run_validation(self):
        """Run system validation."""
        try:
            from validate_system import ValidationSuite
            suite = ValidationSuite()
            success = suite.run_all()
            self.results['validation'] = {
                'passed': suite.passed,
                'warnings': suite.warnings,
                'failed': suite.failed,
                'success': success
            }
            return success
        except Exception as e:
            print(f"Validation error: {e}")
            self.results['validation'] = {'error': str(e)}
            return False

    def initialize_components(self):
        """Initialize all system components."""
        from src.core.ontology import SoulMapOntology
        from src.core.beliefs import BeliefNetwork
        from src.world.social_world import SocialWorld4
        from src.evolution.nas_engine import NASEngine, EvolutionConfig
        from src.evaluation.tom_benchmarks import ToMBenchmarkSuite
        from src.evaluation.zombie_detection import ZombieDetectionSuite

        print("  Creating ontology...")
        self.ontology = SoulMapOntology()
        print(f"    - {self.ontology.total_dims} dimensions")

        print("  Creating social world...")
        self.world = SocialWorld4(num_agents=10, ontology_dim=181, num_zombies=2)
        print(f"    - {self.world.num_agents} agents, 2 zombies")

        print("  Creating belief network...")
        self.belief_net = BeliefNetwork(num_agents=10, ontology_dim=181, max_order=5)
        print(f"    - 5th-order ToM")

        print("  Creating evolution engine...")
        self.evolution_config = EvolutionConfig(
            population_size=self.config['population_size'],
            num_generations=self.config['generations'],
            mutation_rate=self.config['mutation_rate'],
            elite_size=self.config['elite_size']
        )
        self.engine = NASEngine(self.evolution_config, self.world, self.belief_net)

        print("  Creating evaluation suites...")
        self.tom_suite = ToMBenchmarkSuite(input_dim=191)
        self.zombie_suite = ZombieDetectionSuite()

        print("  Initializing population...")
        self.engine.initialize_population()

        self.results['initialization'] = {
            'ontology_dims': self.ontology.total_dims,
            'num_agents': self.world.num_agents,
            'population_size': self.config['population_size'],
            'generations': self.config['generations']
        }

        print("\n  [OK] All components initialized")

    def run_evolution(self):
        """Run the evolutionary process."""
        num_gens = self.config['generations']
        history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'species_count': [],
            'tom_scores': [],
            'zombie_scores': []
        }

        print(f"\n  Running {num_gens} generations...")
        print(f"  Population: {self.config['population_size']}")
        print()

        evolution_start = time.time()

        for gen in range(num_gens):
            gen_start = time.time()

            # Evolve
            self.engine.evolve_generation()

            # Get stats
            best = self.engine.history['best_fitness'][-1] if self.engine.history['best_fitness'] else 0
            avg = self.engine.history['avg_fitness'][-1] if self.engine.history['avg_fitness'] else 0
            div = self.engine.history['diversity'][-1] if self.engine.history['diversity'] else 0
            species = self.engine.species_manager.get_species_count() if self.engine.species_manager else 1

            history['best_fitness'].append(best)
            history['avg_fitness'].append(avg)
            history['diversity'].append(div)
            history['species_count'].append(species)

            # Evaluate best individual every 5 generations
            if self.engine.best_individual and (gen + 1) % 5 == 0:
                tom_results = self.tom_suite.run_full_evaluation(self.engine.best_individual.model)
                zombie_results = self.zombie_suite.run_full_evaluation(
                    self.engine.best_individual.model, {'input_dim': 191}
                )
                history['tom_scores'].append(tom_results['overall_score'])
                history['zombie_scores'].append(1 - zombie_results.get('zombie_probability', 0.5))

            gen_time = time.time() - gen_start

            # Progress display
            bar_len = 30
            filled = int(bar_len * (gen + 1) / num_gens)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  [{bar}] Gen {gen+1:3d}/{num_gens} | "
                  f"Best: {best:.3f} | Avg: {avg:.3f} | "
                  f"Species: {species} | {gen_time:.1f}s", end='', flush=True)

            # Every 5 generations, show detailed population breakdown
            if (gen + 1) % 5 == 0:
                print()  # New line
                # Show architecture distribution
                arch_counts = {}
                for ind in self.engine.population:
                    arch = ind.gene.gene_dict.get('arch_type', 'Unknown')
                    arch_counts[arch] = arch_counts.get(arch, 0) + 1
                arch_str = ", ".join([f"{k}:{v}" for k, v in arch_counts.items()])
                print(f"       Population: {arch_str}")

                # Show top 3 individuals with actual fitness values
                sorted_pop = sorted([p for p in self.engine.population if p.fitness is not None],
                                   key=lambda x: x.fitness, reverse=True)[:3]
                if sorted_pop:
                    top_str = " | ".join([f"{ind.gene.gene_dict.get('arch_type', '?')}:{ind.fitness:.4f}"
                                         for ind in sorted_pop])
                    print(f"       Top 3: {top_str}")

        print()  # New line after progress bar

        evolution_time = time.time() - evolution_start

        self.results['evolution'] = {
            'history': history,
            'total_time': evolution_time,
            'best_fitness': max(history['best_fitness']),
            'final_fitness': history['best_fitness'][-1],
            'final_diversity': history['diversity'][-1]
        }

        print(f"\n  [OK] Evolution complete in {evolution_time:.1f}s")
        print(f"       Best fitness: {max(history['best_fitness']):.4f}")

    def run_final_evaluation(self):
        """Comprehensive evaluation of best individual."""
        if not self.engine.best_individual:
            print("  [ERROR] No best individual found")
            return

        best = self.engine.best_individual
        print(f"  Evaluating best individual (Architecture: {best.gene.gene_dict['arch_type']})")

        # ToM evaluation
        print("  Running ToM benchmarks...")
        tom_results = self.tom_suite.run_full_evaluation(best.model, "best_agent")

        # Zombie evaluation
        print("  Running zombie detection...")
        zombie_results = self.zombie_suite.run_full_evaluation(best.model, {'input_dim': 191})

        self.results['final_evaluation'] = {
            'best_fitness': best.fitness,
            'architecture': best.gene.gene_dict['arch_type'],
            'gene_config': best.gene.gene_dict,
            'tom': {
                'overall_score': tom_results['overall_score'],
                'max_order': tom_results['max_tom_order'],
                'progression': tom_results['sally_anne_progression'],
                'hierarchy_valid': tom_results['hierarchy_valid'],
                'num_passed': tom_results['num_passed'],
                'num_total': tom_results['num_total']
            },
            'zombie': {
                'probability': zombie_results.get('zombie_probability', 0.5),
                'test_results': {k: v.get('score', 0) for k, v in zombie_results.get('test_results', {}).items()}
            }
        }

        # Print summary
        print(f"\n  ToM Results:")
        print(f"    Overall Score: {tom_results['overall_score']:.3f}")
        print(f"    Max Order Passed: {tom_results['max_tom_order']}")
        print(f"    Sally-Anne Progression: {[f'{s:.2f}' for s in tom_results['sally_anne_progression']]}")
        print(f"    Hierarchy Valid: {tom_results['hierarchy_valid']}")

        print(f"\n  Zombie Detection:")
        print(f"    Human-likeness: {1 - zombie_results.get('zombie_probability', 0.5):.3f}")
        for test, result in zombie_results.get('test_results', {}).items():
            status = "PASS" if result.get('passed', False) else "FAIL"
            print(f"    {test}: {result.get('score', 0):.2f} [{status}]")

        print("\n  [OK] Evaluation complete")

    def generate_report(self, timestamp):
        """Generate comprehensive report."""
        report_path = self.output_dir / f"report_{timestamp}.txt"

        total_time = (datetime.now() - self.start_time).total_seconds()

        report = f"""
{'='*70}
ToM-NAS EXPERIMENT REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runtime: {total_time:.1f} seconds

{'='*70}
CONFIGURATION
{'='*70}
Population Size: {self.config['population_size']}
Generations: {self.config['generations']}
Mutation Rate: {self.config['mutation_rate']}

{'='*70}
VALIDATION RESULTS
{'='*70}
"""
        if 'validation' in self.results:
            v = self.results['validation']
            if 'error' in v:
                report += f"Error: {v['error']}\n"
            else:
                report += f"Passed: {v['passed']}\n"
                report += f"Warnings: {v['warnings']}\n"
                report += f"Failed: {v['failed']}\n"

        report += f"""
{'='*70}
EVOLUTION RESULTS
{'='*70}
"""
        if 'evolution' in self.results:
            e = self.results['evolution']
            report += f"Evolution Time: {e['total_time']:.1f}s\n"
            report += f"Best Fitness Achieved: {e['best_fitness']:.4f}\n"
            report += f"Final Fitness: {e['final_fitness']:.4f}\n"
            report += f"Final Diversity: {e['final_diversity']:.4f}\n"

        report += f"""
{'='*70}
BEST INDIVIDUAL
{'='*70}
"""
        if 'final_evaluation' in self.results:
            f = self.results['final_evaluation']
            report += f"Architecture: {f['architecture']}\n"
            report += f"Fitness: {f['best_fitness']:.4f}\n"
            report += f"\nArchitecture Configuration:\n"
            for k, v in f['gene_config'].items():
                report += f"  {k}: {v}\n"

            report += f"\nTheory of Mind:\n"
            report += f"  Overall Score: {f['tom']['overall_score']:.3f}\n"
            report += f"  Max Order Passed: {f['tom']['max_order']}\n"
            report += f"  Tests Passed: {f['tom']['num_passed']}/{f['tom']['num_total']}\n"
            report += f"  Hierarchy Valid: {f['tom']['hierarchy_valid']}\n"
            report += f"\n  Sally-Anne Progression:\n"
            for i, score in enumerate(f['tom']['progression']):
                status = "PASS" if score > 0.5 else "FAIL"
                report += f"    Order {i}: {score:.3f} [{status}]\n"

            report += f"\nZombie Detection:\n"
            report += f"  Human-likeness Score: {1 - f['zombie']['probability']:.3f}\n"
            for test, score in f['zombie']['test_results'].items():
                status = "PASS" if score > 0.5 else "FAIL"
                report += f"  {test}: {score:.3f} [{status}]\n"

        report += f"""
{'='*70}
INTERPRETATION (Plain English)
{'='*70}
"""
        if 'final_evaluation' in self.results:
            f = self.results['final_evaluation']
            max_order = f['tom']['max_order']

            if max_order >= 3:
                report += """
EXCELLENT RESULTS!

The AI has developed genuine Theory of Mind capabilities. It can:
- Track what others believe (even when those beliefs are false)
- Model what others think about what you think
- Handle nested belief reasoning at multiple levels

This is comparable to human-like social cognition. The AI isn't just
pattern-matching - it's actually reasoning about mental states.
"""
            elif max_order >= 1:
                report += """
GOOD RESULTS!

The AI has basic Theory of Mind - it can track that others have
different beliefs than reality. This is the foundation of social
intelligence.

For deeper reasoning (knowing that you know that I know...),
try running more generations or increasing population size.
"""
            else:
                report += """
DEVELOPING RESULTS

The AI is still learning to understand others' minds. This is
expected for early evolution stages.

Recommendations:
- Run more generations (50+)
- Increase population size
- Try different architecture combinations
"""

        report += f"""
{'='*70}
FILES GENERATED
{'='*70}
Report: {report_path.name}
Results JSON: results_{timestamp}.json
Evolution Plot: evolution_{timestamp}.png (if matplotlib available)

{'='*70}
END OF REPORT
{'='*70}
"""

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"  Report saved to: {report_path}")
        return report_path

    def save_results(self, timestamp):
        """Save all results to files."""
        # Save JSON results
        json_path = self.output_dir / f"results_{timestamp}.json"

        # Make results JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            return obj

        serializable_results = make_serializable(self.results)

        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"  Results saved to: {json_path}")

        # Save plot
        try:
            import matplotlib.pyplot as plt

            if 'evolution' in self.results:
                history = self.results['evolution']['history']

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('ToM-NAS Evolution Results', fontsize=14, fontweight='bold')

                # Fitness
                ax1 = axes[0, 0]
                ax1.plot(history['best_fitness'], 'b-', linewidth=2, label='Best')
                ax1.plot(history['avg_fitness'], 'g--', linewidth=1.5, label='Average')
                ax1.fill_between(range(len(history['best_fitness'])),
                                history['avg_fitness'], history['best_fitness'], alpha=0.3)
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Fitness')
                ax1.set_title('Fitness Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Diversity
                ax2 = axes[0, 1]
                ax2.plot(history['diversity'], 'r-', linewidth=2)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Diversity')
                ax2.set_title('Population Diversity')
                ax2.grid(True, alpha=0.3)

                # Species
                ax3 = axes[1, 0]
                ax3.plot(history['species_count'], 'purple', linewidth=2, marker='o', markersize=3)
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('Species Count')
                ax3.set_title('Architecture Diversity')
                ax3.set_ylim(0, 5)
                ax3.grid(True, alpha=0.3)

                # ToM Progression
                ax4 = axes[1, 1]
                if 'final_evaluation' in self.results:
                    progression = self.results['final_evaluation']['tom']['progression']
                    orders = list(range(len(progression)))
                    colors = ['green' if s > 0.5 else 'red' for s in progression]
                    ax4.bar(orders, progression, color=colors, alpha=0.7)
                    ax4.axhline(y=0.5, color='black', linestyle='--', label='Pass threshold')
                    ax4.set_xlabel('ToM Order')
                    ax4.set_ylabel('Score')
                    ax4.set_title('Sally-Anne Test Results')
                    ax4.set_xticks(orders)
                    ax4.legend()
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()

                plot_path = self.output_dir / f"evolution_{timestamp}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Plot saved to: {plot_path}")

        except Exception as e:
            print(f"  Could not save plot: {e}")

        # Save best model
        if hasattr(self, 'engine') and self.engine.best_individual:
            model_path = self.output_dir / f"best_model_{timestamp}.pt"
            torch.save({
                'model_state': self.engine.best_individual.model.state_dict(),
                'gene_config': self.engine.best_individual.gene.gene_dict,
                'fitness': self.engine.best_individual.fitness
            }, model_path)
            print(f"  Model saved to: {model_path}")

    def print_summary(self, report_path):
        """Print final summary."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        print_banner("EXPERIMENT COMPLETE", '=')

        print(f"\nTotal Runtime: {total_time:.1f} seconds")

        if 'final_evaluation' in self.results:
            f = self.results['final_evaluation']
            print(f"\nKey Results:")
            print(f"  Best Fitness: {f['best_fitness']:.4f}")
            print(f"  Architecture: {f['architecture']}")
            print(f"  Max ToM Order: {f['tom']['max_order']}")
            print(f"  Human-likeness: {1 - f['zombie']['probability']:.3f}")

        print(f"\nOutput Files:")
        print(f"  {self.output_dir}/")
        for file in sorted(self.output_dir.iterdir()):
            print(f"    - {file.name}")

        print(f"\nFull Report: {report_path}")
        print("\n" + "=" * 70)


def run_validation_check():
    """Run the new comprehensive validation suite before experiment."""
    try:
        from validate_baselines import run_full_validation
        print("\n  Running comprehensive baseline validation...")
        report = run_full_validation(output_file=None, verbose=False)

        if report.overall_validity == "VALID":
            print("  [OK] Validation passed - tests appear scientifically sound")
            return True
        elif report.overall_validity == "NEEDS_REVIEW":
            print("  [WARNING] Some validation concerns found:")
            for rec in report.recommendations[:3]:
                print(f"    - {rec}")
            return True
        else:
            print("  [ERROR] Validation failed - review test suite before using results")
            for rec in report.recommendations:
                print(f"    - {rec}")
            return False
    except Exception as e:
        print(f"  [WARNING] Could not run validation: {e}")
        return True  # Continue anyway


def main():
    """Run automated experiment."""
    # Parse command line args for config overrides
    import argparse
    parser = argparse.ArgumentParser(description='Run ToM-NAS Automated Experiment')
    parser.add_argument('-g', '--generations', type=int, default=25,
                        help='Number of generations (default: 25)')
    parser.add_argument('-p', '--population', type=int, default=15,
                        help='Population size (default: 15)')
    parser.add_argument('-m', '--mutation', type=float, default=0.1,
                        help='Mutation rate (default: 0.1)')
    parser.add_argument('-o', '--output', type=str, default='experiment_results',
                        help='Output directory (default: experiment_results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer generations')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip comprehensive validation check')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation, no experiment')

    args = parser.parse_args()

    # Option to run validation only
    if args.validate_only:
        print_banner("ToM-NAS Validation Only Mode")
        from validate_baselines import run_full_validation
        run_full_validation(output_file="validation_report.json", verbose=True)
        return 0

    # Run validation check before experiment (unless skipped)
    if not args.skip_validation:
        print_banner("Pre-Experiment Validation")
        if not run_validation_check():
            print("\n[ERROR] Fix validation issues before running experiment")
            print("Use --skip-validation to bypass this check")
            return 1

    config = CONFIG.copy()
    config['generations'] = 10 if args.quick else args.generations
    config['population_size'] = 8 if args.quick else args.population
    config['mutation_rate'] = args.mutation
    config['output_dir'] = args.output

    experiment = AutomatedExperiment(config)
    results = experiment.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
