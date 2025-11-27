"""
Unified Experiment Runner for ToM-NAS

Orchestrates NAS experiments across multiple tasks and methods.
Provides controlled comparisons for hypothesis testing.

Key features:
1. Runs NAS on task taxonomy (simple -> higher-order ToM)
2. Compares multiple NAS methods (CMA-ES, OpenES, DARTS, etc.)
3. Collects architecture metrics for hypothesis testing
4. Generates statistical analysis and reports
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np

# Import components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evolution.search_space import (
    ArchitectureGenome, ArchitectureMetrics, OPERATION_SPACE,
    SearchSpaceFactory, create_random_genome
)
from src.evolution.evo_nas_jax import (
    EvosaxConfig, run_evolutionary_search, create_evolution_engine
)
from src.evolution.multi_objective import run_pareto_search, analyze_pareto_front
from src.evolution.ablations import AblationStudy, AblationConfig
from src.evolution.nas_bench import run_benchmark_baseline_study

from src.analysis.statistical_analysis import (
    run_hypothesis_tests, compute_effect_sizes, generate_statistical_report
)
from src.analysis.architecture_analysis import (
    analyze_architecture_families, compare_task_architectures
)
from src.analysis.rsc_metrics import analyze_rsc_across_tasks, compare_rsc_tom_vs_control


@dataclass
class ExperimentConfig:
    """Configuration for experiment suite"""
    # Task taxonomy
    task_conditions: Dict[str, Dict] = field(default_factory=lambda: {
        'simple_sequence': {
            'expected_complexity': 'low',
            'tom_order': 0,
        },
        'babi_1': {
            'expected_complexity': 'low',
            'tom_order': 0,
        },
        'babi_4': {
            'expected_complexity': 'medium',
            'tom_order': 0,
        },
        'tomi': {
            'expected_complexity': 'medium',
            'tom_order': 1,
        },
        'bigtom': {
            'expected_complexity': 'medium-high',
            'tom_order': 1,
        },
        'hitom_2': {
            'expected_complexity': 'high',
            'tom_order': 2,
        },
        'hitom_3': {
            'expected_complexity': 'very_high',
            'tom_order': 3,
        },
        'hitom_4': {
            'expected_complexity': 'extreme',
            'tom_order': 4,
        },
        'opentom': {
            'expected_complexity': 'high',
            'tom_order': 1,
        },
        'socialqa': {
            'expected_complexity': 'high',
            'tom_order': 1,
        },
    })

    # NAS methods to compare
    nas_methods: List[str] = field(default_factory=lambda: [
        'CMA_ES', 'OpenES', 'PGPE', 'RandomSearch'
    ])

    # Evolution parameters
    population_size: int = 128
    num_generations: int = 50

    # Seeds for multiple runs
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])

    # Output directory
    output_dir: str = "experiments"


class NASExperimentRunner:
    """
    Orchestrates NAS experiments across multiple tasks and methods.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        fitness_fn_factory: Callable[[str], Callable[[ArchitectureGenome], float]],
    ):
        """
        Args:
            config: Experiment configuration
            fitness_fn_factory: Factory creating fitness functions per task
        """
        self.config = config
        self.fitness_fn_factory = fitness_fn_factory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.results: Dict[str, List[Dict]] = {}
        self.summary_stats: Dict[str, Any] = {}

    def run_single_experiment(
        self,
        task_name: str,
        nas_method: str,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run NAS for a single task-method-seed combination.
        """
        print(f"\n{'='*60}")
        print(f"Running: {task_name} with {nas_method} (seed={seed})")
        print(f"{'='*60}\n")

        task_config = self.config.task_conditions.get(task_name, {})
        fitness_fn = self.fitness_fn_factory(task_name)

        # Run appropriate NAS method
        if nas_method in ['CMA_ES', 'OpenES', 'PGPE']:
            results = run_evolutionary_search(
                task_type=task_name,
                fitness_fn=fitness_fn,
                strategy=nas_method,
                population_size=self.config.population_size,
                num_generations=self.config.num_generations,
                seed=seed,
                use_jax=False,  # Use NumPy for broader compatibility
            )
        elif nas_method == 'RandomSearch':
            results = self._run_random_search(
                task_name, fitness_fn, seed
            )
        else:
            # Default to CMA-ES
            results = run_evolutionary_search(
                task_type=task_name,
                fitness_fn=fitness_fn,
                strategy='CMA_ES',
                population_size=self.config.population_size,
                num_generations=self.config.num_generations,
                seed=seed,
            )

        # Add metadata
        results['task_name'] = task_name
        results['nas_method'] = nas_method
        results['seed'] = seed
        results['tom_order'] = task_config.get('tom_order', 0)
        results['expected_complexity'] = task_config.get('expected_complexity', 'unknown')
        results['timestamp'] = datetime.now().isoformat()

        # Save individual result
        self._save_result(results, task_name, nas_method, seed)

        return results

    def _run_random_search(
        self,
        task_name: str,
        fitness_fn: Callable,
        seed: int,
    ) -> Dict[str, Any]:
        """Run random architecture search as baseline"""
        np.random.seed(seed)

        best_fitness = float('-inf')
        best_genome = None
        fitness_history = []

        total_samples = self.config.population_size * self.config.num_generations

        for i in range(total_samples):
            genome = create_random_genome(seed=seed + i)

            try:
                fitness = fitness_fn(genome)
            except Exception:
                fitness = 0.0

            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome

            if i % 100 == 0:
                fitness_history.append(best_fitness)

        return {
            'best_genome': best_genome.to_dict() if best_genome else None,
            'best_fitness': best_fitness,
            'evolution_log': {'best_fitness': fitness_history},
            'final_metrics': ArchitectureMetrics(best_genome).compute_all() if best_genome else {},
        }

    def run_task_suite(
        self,
        tasks: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run experiments across multiple tasks, methods, and seeds.
        """
        tasks = tasks or list(self.config.task_conditions.keys())
        methods = methods or self.config.nas_methods
        seeds = seeds or self.config.seeds

        total_runs = len(tasks) * len(methods) * len(seeds)
        current_run = 0

        print(f"\n{'#'*60}")
        print("STARTING FULL EXPERIMENT SUITE")
        print(f"Tasks: {len(tasks)}, Methods: {len(methods)}, Seeds: {len(seeds)}")
        print(f"Total runs: {total_runs}")
        print(f"{'#'*60}\n")

        for task in tasks:
            self.results[task] = []

            for method in methods:
                for seed in seeds:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}]")

                    try:
                        result = self.run_single_experiment(task, method, seed)
                        self.results[task].append(result)
                    except Exception as e:
                        print(f"ERROR in {task}/{method}/seed{seed}: {e}")
                        continue

        # Generate analysis
        self._generate_analysis()

        return self.get_summary()

    def run_hypothesis_focused_suite(self) -> Dict[str, Any]:
        """
        Run experiments specifically designed to test hypotheses.

        H1: Skip connections
        H2: Attention mechanisms
        H3: Complexity correlation
        H4: Recursive structure
        """
        print("\n" + "="*60)
        print("HYPOTHESIS-FOCUSED EXPERIMENT SUITE")
        print("="*60)

        # Select key tasks for hypothesis testing
        hypothesis_tasks = {
            'control': ['simple_sequence', 'babi_1'],
            'tom_first_order': ['tomi', 'bigtom'],
            'tom_higher_order': ['hitom_2', 'hitom_4'],
        }

        all_tasks = []
        for group in hypothesis_tasks.values():
            all_tasks.extend(group)

        # Run with primary method and multiple seeds
        return self.run_task_suite(
            tasks=all_tasks,
            methods=['CMA_ES'],
            seeds=self.config.seeds,
        )

    def run_method_comparison(self) -> Dict[str, Any]:
        """
        Compare different NAS methods on a subset of tasks.
        """
        print("\n" + "="*60)
        print("NAS METHOD COMPARISON")
        print("="*60)

        comparison_tasks = ['simple_sequence', 'tomi', 'hitom_2']

        return self.run_task_suite(
            tasks=comparison_tasks,
            methods=self.config.nas_methods,
            seeds=[42, 123, 456],  # Fewer seeds for method comparison
        )

    def _generate_analysis(self):
        """Generate comprehensive analysis of results"""
        print("\n" + "="*60)
        print("GENERATING ANALYSIS")
        print("="*60)

        # Statistical hypothesis tests
        print("\nRunning hypothesis tests...")
        self.summary_stats['hypothesis_tests'] = run_hypothesis_tests(self.results)

        # Effect sizes
        print("Computing effect sizes...")
        self.summary_stats['effect_sizes'] = compute_effect_sizes(self.results)

        # Architecture family analysis
        print("Analyzing architecture families...")
        all_architectures = []
        for task_results in self.results.values():
            for r in task_results:
                if r.get('best_genome'):
                    all_architectures.append(r['best_genome'])

        if all_architectures:
            self.summary_stats['architecture_clusters'] = analyze_architecture_families(
                all_architectures, n_clusters=4
            )

        # Task comparison
        print("Comparing architectures across tasks...")
        self.summary_stats['task_comparison'] = compare_task_architectures(self.results)

        # RSC analysis
        print("Computing RSC metrics...")
        self.summary_stats['rsc_analysis'] = analyze_rsc_across_tasks(self.results)

        # Generate report
        print("Generating report...")
        report = generate_statistical_report(self.results)
        self.summary_stats['report'] = report

        # Save analysis
        self._save_analysis()

    def _save_result(self, result: Dict, task: str, method: str, seed: int):
        """Save individual experiment result"""
        filename = f"{task}_{method}_seed{seed}.json"
        filepath = self.output_dir / filename

        # Make serializable
        serializable = self._make_serializable(result)

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

    def _save_analysis(self):
        """Save analysis results"""
        # Save full results
        all_results_path = self.output_dir / 'all_results.json'
        with open(all_results_path, 'w') as f:
            json.dump(self._make_serializable(self.results), f, indent=2)

        # Save summary stats
        summary_path = self.output_dir / 'summary_stats.json'
        with open(summary_path, 'w') as f:
            json.dump(self._make_serializable(self.summary_stats), f, indent=2)

        # Save text report
        if 'report' in self.summary_stats:
            report_path = self.output_dir / 'statistical_report.md'
            with open(report_path, 'w') as f:
                f.write(self.summary_stats['report'])

        print(f"\nResults saved to {self.output_dir}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        summary = {
            'total_experiments': sum(len(r) for r in self.results.values()),
            'tasks': list(self.results.keys()),
            'summary_stats': self.summary_stats,
        }

        # Per-task summary
        per_task = {}
        for task, task_results in self.results.items():
            fitnesses = [r.get('best_fitness', 0) for r in task_results]
            per_task[task] = {
                'n_experiments': len(task_results),
                'mean_fitness': float(np.mean(fitnesses)) if fitnesses else 0,
                'std_fitness': float(np.std(fitnesses)) if fitnesses else 0,
                'max_fitness': float(np.max(fitnesses)) if fitnesses else 0,
            }

        summary['per_task'] = per_task

        return summary

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)

        print(f"\nTotal experiments: {summary['total_experiments']}")
        print(f"Tasks: {', '.join(summary['tasks'])}")

        print("\n--- Per-Task Results ---")
        for task, stats in summary.get('per_task', {}).items():
            print(f"\n{task}:")
            print(f"  Experiments: {stats['n_experiments']}")
            print(f"  Mean fitness: {stats['mean_fitness']:.4f} +/- {stats['std_fitness']:.4f}")
            print(f"  Max fitness: {stats['max_fitness']:.4f}")

        if 'hypothesis_tests' in self.summary_stats:
            print("\n--- Hypothesis Test Results ---")
            for h_name, result in self.summary_stats['hypothesis_tests'].items():
                if hasattr(result, 'significant'):
                    sig = "SIGNIFICANT" if result.significant else "not significant"
                    print(f"  {h_name}: {sig} (p={result.p_value:.4f})")


def create_dummy_fitness_factory():
    """Create a dummy fitness factory for testing"""

    def fitness_fn_factory(task_name: str):
        """Create fitness function for task"""

        def fitness_fn(genome: ArchitectureGenome) -> float:
            """Dummy fitness based on architecture metrics"""
            metrics = ArchitectureMetrics(genome).compute_all()

            # Base fitness
            base = 0.5

            # Task-specific bonuses
            if 'hitom' in task_name or 'tom' in task_name:
                # ToM tasks prefer skip connections and attention
                base += 0.1 * metrics['num_skip_connections'] / 10
                base += 0.1 * metrics['num_attention_ops'] / 10
                base += 0.05 * metrics['recursive_depth'] / 10
            else:
                # Control tasks are simpler
                base += 0.1 - 0.01 * metrics['total_parameters'] / 1e6

            # Add noise
            base += np.random.normal(0, 0.05)

            return float(np.clip(base, 0, 1))

        return fitness_fn

    return fitness_fn_factory


def run_quick_experiment(
    tasks: Optional[List[str]] = None,
    output_dir: str = "quick_experiment",
) -> Dict[str, Any]:
    """
    Run a quick experiment for testing.

    Args:
        tasks: List of tasks to run
        output_dir: Output directory

    Returns:
        Experiment results
    """
    tasks = tasks or ['simple_sequence', 'tomi', 'hitom_2']

    config = ExperimentConfig(
        population_size=32,
        num_generations=10,
        seeds=[42],
        output_dir=output_dir,
    )

    # Filter task conditions
    config.task_conditions = {
        k: v for k, v in config.task_conditions.items() if k in tasks
    }

    runner = NASExperimentRunner(config, create_dummy_fitness_factory())

    return runner.run_task_suite(
        tasks=tasks,
        methods=['CMA_ES'],
        seeds=[42],
    )
