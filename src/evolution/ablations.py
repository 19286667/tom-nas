"""
Ablation Study Framework for ToM-NAS Experiments

Systematic ablation studies to test hypotheses:
- H1: Removing skip connections hurts ToM performance more than control tasks
- H2: Removing attention mechanisms hurts ToM performance
- H4: Removing recursive operations hurts higher-order ToM

Ablation types:
1. Operation-level: Remove specific operation types from search space
2. Component-level: Remove architectural components (attention, recurrence)
3. Scale-level: Vary architecture size parameters
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from scipy import stats

from .search_space import (
    ArchitectureGenome, ArchitectureMetrics,
    OPERATION_SPACE, SearchSpaceFactory,
    create_random_genome
)
from .evo_nas_jax import (
    EvosaxConfig, run_evolutionary_search, create_evolution_engine
)


@dataclass
class AblationConfig:
    """Configuration for ablation experiments"""
    task_types: List[str] = field(default_factory=lambda: [
        'simple_sequence', 'babi_1', 'tomi', 'hitom_2', 'hitom_4'
    ])
    ablation_types: List[str] = field(default_factory=lambda: [
        'full', 'no_skip', 'no_attention', 'no_recurrence', 'minimal'
    ])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    population_size: int = 128
    num_generations: int = 50


@dataclass
class AblationResult:
    """Result from a single ablation experiment"""
    task_type: str
    ablation_type: str
    seed: int
    best_fitness: float
    final_metrics: Dict[str, Any]
    evolution_log: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'task_type': self.task_type,
            'ablation_type': self.ablation_type,
            'seed': self.seed,
            'best_fitness': self.best_fitness,
            'final_metrics': self.final_metrics,
            'evolution_log': self.evolution_log,
            'timestamp': self.timestamp,
        }


class AblationStudy:
    """
    Systematic ablation study framework.

    Runs NAS with different operation spaces and compares results
    to test hypotheses about which operations are essential for ToM.
    """

    def __init__(
        self,
        config: AblationConfig,
        fitness_fn_factory: Callable[[str], Callable[[ArchitectureGenome], float]],
        output_dir: Optional[str] = None,
    ):
        """
        Args:
            config: Ablation configuration
            fitness_fn_factory: Factory function that creates fitness function for a task
            output_dir: Directory to save results
        """
        self.config = config
        self.fitness_fn_factory = fitness_fn_factory
        self.output_dir = Path(output_dir) if output_dir else Path("ablation_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.results: List[AblationResult] = []

    def get_ablated_space(self, ablation_type: str) -> Dict:
        """Get operation space for ablation type"""
        if ablation_type == 'full':
            return SearchSpaceFactory.full_space()
        elif ablation_type == 'no_skip':
            return SearchSpaceFactory.no_skip_space()
        elif ablation_type == 'no_attention':
            return SearchSpaceFactory.no_attention_space()
        elif ablation_type == 'no_recurrence':
            return SearchSpaceFactory.no_recurrence_space()
        elif ablation_type == 'minimal':
            return SearchSpaceFactory.minimal_space()
        elif ablation_type == 'conv_only':
            return SearchSpaceFactory.conv_only_space()
        else:
            return SearchSpaceFactory.full_space()

    def run_single_ablation(
        self,
        task_type: str,
        ablation_type: str,
        seed: int,
    ) -> AblationResult:
        """Run a single ablation experiment"""
        print(f"\n--- Ablation: {task_type} / {ablation_type} / seed={seed} ---")

        # Get ablated operation space
        operation_space = self.get_ablated_space(ablation_type)

        # Get fitness function for task
        fitness_fn = self.fitness_fn_factory(task_type)

        # Run evolution
        results = run_evolutionary_search(
            task_type=task_type,
            fitness_fn=fitness_fn,
            strategy="CMA_ES",
            population_size=self.config.population_size,
            num_generations=self.config.num_generations,
            operation_space=operation_space,
            seed=seed,
            use_jax=False,  # Use NumPy for broader compatibility
        )

        # Create result
        ablation_result = AblationResult(
            task_type=task_type,
            ablation_type=ablation_type,
            seed=seed,
            best_fitness=results.get('best_fitness', 0.0),
            final_metrics=results.get('final_metrics', {}),
            evolution_log=results.get('evolution_log', {}),
        )

        self.results.append(ablation_result)

        # Save individual result
        self._save_result(ablation_result)

        return ablation_result

    def run_full_study(self) -> Dict[str, Any]:
        """Run complete ablation study across all configurations"""
        print("\n" + "="*60)
        print("Starting Full Ablation Study")
        print(f"Tasks: {self.config.task_types}")
        print(f"Ablations: {self.config.ablation_types}")
        print(f"Seeds: {self.config.seeds}")
        print("="*60)

        total_runs = len(self.config.task_types) * len(self.config.ablation_types) * len(self.config.seeds)
        current_run = 0

        for task_type in self.config.task_types:
            for ablation_type in self.config.ablation_types:
                for seed in self.config.seeds:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}] Running ablation...")

                    try:
                        self.run_single_ablation(task_type, ablation_type, seed)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue

        # Generate analysis
        analysis = self.analyze_results()

        # Save full results
        self._save_full_results(analysis)

        return analysis

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze ablation study results"""
        if not self.results:
            return {'error': 'No results to analyze'}

        # Organize results by task and ablation type
        organized = {}
        for result in self.results:
            key = (result.task_type, result.ablation_type)
            if key not in organized:
                organized[key] = []
            organized[key].append(result.best_fitness)

        # Calculate performance drops
        performance_matrix = {}
        for task_type in self.config.task_types:
            performance_matrix[task_type] = {}

            # Get full space baseline
            full_key = (task_type, 'full')
            if full_key in organized:
                baseline = np.mean(organized[full_key])
            else:
                baseline = 0.0

            for ablation_type in self.config.ablation_types:
                key = (task_type, ablation_type)
                if key in organized:
                    mean_fitness = np.mean(organized[key])
                    std_fitness = np.std(organized[key])
                    drop = baseline - mean_fitness

                    performance_matrix[task_type][ablation_type] = {
                        'mean_fitness': float(mean_fitness),
                        'std_fitness': float(std_fitness),
                        'performance_drop': float(drop),
                        'relative_drop': float(drop / baseline) if baseline > 0 else 0.0,
                    }

        # Statistical tests
        statistical_tests = self._run_statistical_tests(organized)

        # Hypothesis-specific analysis
        hypothesis_analysis = self._analyze_hypotheses(organized)

        return {
            'performance_matrix': performance_matrix,
            'statistical_tests': statistical_tests,
            'hypothesis_analysis': hypothesis_analysis,
            'summary': self._generate_summary(performance_matrix, hypothesis_analysis),
        }

    def _run_statistical_tests(self, organized: Dict) -> Dict[str, Any]:
        """Run statistical tests on ablation results"""
        tests = {}

        # Compare ToM vs non-ToM tasks for skip connection ablation
        tom_tasks = ['tomi', 'hitom_2', 'hitom_4']
        non_tom_tasks = ['simple_sequence', 'babi_1']

        for ablation_type in ['no_skip', 'no_attention', 'no_recurrence']:
            tom_drops = []
            non_tom_drops = []

            for task in tom_tasks:
                full_key = (task, 'full')
                ablated_key = (task, ablation_type)

                if full_key in organized and ablated_key in organized:
                    baseline = np.mean(organized[full_key])
                    ablated = np.mean(organized[ablated_key])
                    tom_drops.append(baseline - ablated)

            for task in non_tom_tasks:
                full_key = (task, 'full')
                ablated_key = (task, ablation_type)

                if full_key in organized and ablated_key in organized:
                    baseline = np.mean(organized[full_key])
                    ablated = np.mean(organized[ablated_key])
                    non_tom_drops.append(baseline - ablated)

            if tom_drops and non_tom_drops:
                # T-test: Do ToM tasks suffer more from this ablation?
                t_stat, p_value = stats.ttest_ind(tom_drops, non_tom_drops)

                tests[f'{ablation_type}_tom_vs_control'] = {
                    'tom_mean_drop': float(np.mean(tom_drops)),
                    'control_mean_drop': float(np.mean(non_tom_drops)),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                }

        return tests

    def _analyze_hypotheses(self, organized: Dict) -> Dict[str, Any]:
        """Analyze results in context of research hypotheses"""
        analysis = {}

        # H1: Skip connections more important for ToM
        h1_data = self._compute_hypothesis_data(organized, 'no_skip')
        analysis['H1_skip_connections'] = {
            'data': h1_data,
            'support': self._assess_hypothesis_support(h1_data),
        }

        # H2: Attention more important for ToM
        h2_data = self._compute_hypothesis_data(organized, 'no_attention')
        analysis['H2_attention'] = {
            'data': h2_data,
            'support': self._assess_hypothesis_support(h2_data),
        }

        # H4: Recursive structures more important for higher-order ToM
        h4_data = self._compute_higher_order_effect(organized)
        analysis['H4_recursive'] = {
            'data': h4_data,
            'support': self._assess_h4_support(h4_data),
        }

        return analysis

    def _compute_hypothesis_data(
        self,
        organized: Dict,
        ablation_type: str
    ) -> Dict[str, float]:
        """Compute performance drops for hypothesis testing"""
        tom_drops = []
        control_drops = []

        for task in ['tomi', 'hitom_2', 'hitom_4']:
            full_key = (task, 'full')
            ablated_key = (task, ablation_type)
            if full_key in organized and ablated_key in organized:
                baseline = np.mean(organized[full_key])
                ablated = np.mean(organized[ablated_key])
                tom_drops.append((baseline - ablated) / baseline if baseline > 0 else 0)

        for task in ['simple_sequence', 'babi_1']:
            full_key = (task, 'full')
            ablated_key = (task, ablation_type)
            if full_key in organized and ablated_key in organized:
                baseline = np.mean(organized[full_key])
                ablated = np.mean(organized[ablated_key])
                control_drops.append((baseline - ablated) / baseline if baseline > 0 else 0)

        return {
            'tom_relative_drop': float(np.mean(tom_drops)) if tom_drops else 0.0,
            'control_relative_drop': float(np.mean(control_drops)) if control_drops else 0.0,
            'difference': float(np.mean(tom_drops) - np.mean(control_drops)) if tom_drops and control_drops else 0.0,
        }

    def _compute_higher_order_effect(self, organized: Dict) -> Dict[str, float]:
        """Compute effect of recurrence ablation across ToM orders"""
        order_drops = {}

        for task, order in [('tomi', 1), ('hitom_2', 2), ('hitom_4', 4)]:
            full_key = (task, 'full')
            ablated_key = (task, 'no_recurrence')

            if full_key in organized and ablated_key in organized:
                baseline = np.mean(organized[full_key])
                ablated = np.mean(organized[ablated_key])
                order_drops[order] = (baseline - ablated) / baseline if baseline > 0 else 0

        # Compute correlation between order and drop
        if len(order_drops) >= 2:
            orders = list(order_drops.keys())
            drops = [order_drops[o] for o in orders]
            correlation = np.corrcoef(orders, drops)[0, 1] if len(orders) > 1 else 0.0
        else:
            correlation = 0.0

        return {
            'order_drops': order_drops,
            'order_drop_correlation': float(correlation),
        }

    def _assess_hypothesis_support(self, data: Dict[str, float]) -> str:
        """Assess level of support for hypothesis"""
        diff = data['difference']

        if diff > 0.1:
            return "strong_support"
        elif diff > 0.05:
            return "moderate_support"
        elif diff > 0.02:
            return "weak_support"
        elif diff > -0.02:
            return "no_difference"
        else:
            return "contradicted"

    def _assess_h4_support(self, data: Dict[str, float]) -> str:
        """Assess H4 (recursive structure) support"""
        corr = data['order_drop_correlation']

        if corr > 0.7:
            return "strong_support"
        elif corr > 0.4:
            return "moderate_support"
        elif corr > 0.2:
            return "weak_support"
        else:
            return "no_support"

    def _generate_summary(
        self,
        performance_matrix: Dict,
        hypothesis_analysis: Dict
    ) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("ABLATION STUDY SUMMARY")
        lines.append("=" * 40)

        # H1 Summary
        h1 = hypothesis_analysis.get('H1_skip_connections', {})
        lines.append(f"\nH1 (Skip Connections): {h1.get('support', 'unknown')}")
        if 'data' in h1:
            lines.append(f"  ToM relative drop: {h1['data']['tom_relative_drop']:.2%}")
            lines.append(f"  Control relative drop: {h1['data']['control_relative_drop']:.2%}")

        # H2 Summary
        h2 = hypothesis_analysis.get('H2_attention', {})
        lines.append(f"\nH2 (Attention): {h2.get('support', 'unknown')}")
        if 'data' in h2:
            lines.append(f"  ToM relative drop: {h2['data']['tom_relative_drop']:.2%}")
            lines.append(f"  Control relative drop: {h2['data']['control_relative_drop']:.2%}")

        # H4 Summary
        h4 = hypothesis_analysis.get('H4_recursive', {})
        lines.append(f"\nH4 (Recursive Structure): {h4.get('support', 'unknown')}")
        if 'data' in h4:
            lines.append(f"  Order-drop correlation: {h4['data']['order_drop_correlation']:.3f}")

        return '\n'.join(lines)

    def _save_result(self, result: AblationResult):
        """Save individual ablation result"""
        filename = f"{result.task_type}_{result.ablation_type}_seed{result.seed}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_full_results(self, analysis: Dict):
        """Save complete analysis"""
        # Save all results
        all_results = [r.to_dict() for r in self.results]
        with open(self.output_dir / 'all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        # Save analysis
        with open(self.output_dir / 'analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)

        # Save summary
        with open(self.output_dir / 'summary.txt', 'w') as f:
            f.write(analysis.get('summary', ''))

        print(f"\nResults saved to {self.output_dir}")


def ablation_remove_skip_connections(
    task_type: str,
    fitness_fn: Callable[[ArchitectureGenome], float],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ablation: What happens when skip connections are removed?

    Hypothesis: Performance on ToM tasks will drop more than on simple tasks.
    """
    # Full space
    results_full = run_evolutionary_search(
        task_type=task_type,
        fitness_fn=fitness_fn,
        strategy='CMA_ES',
        population_size=128,
        num_generations=50,
        operation_space=SearchSpaceFactory.full_space(),
        seed=seed,
    )

    # Ablated space
    results_ablated = run_evolutionary_search(
        task_type=task_type,
        fitness_fn=fitness_fn,
        strategy='CMA_ES',
        population_size=128,
        num_generations=50,
        operation_space=SearchSpaceFactory.no_skip_space(),
        seed=seed,
    )

    performance_drop = results_full['best_fitness'] - results_ablated['best_fitness']

    return {
        'ablation_type': 'no_skip',
        'task_type': task_type,
        'full_fitness': results_full['best_fitness'],
        'ablated_fitness': results_ablated['best_fitness'],
        'performance_drop': performance_drop,
        'relative_drop': performance_drop / results_full['best_fitness'] if results_full['best_fitness'] > 0 else 0,
        'full_results': results_full,
        'ablated_results': results_ablated,
    }


def ablation_remove_attention(
    task_type: str,
    fitness_fn: Callable[[ArchitectureGenome], float],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Ablation: What happens when attention mechanisms are removed?
    """
    results_full = run_evolutionary_search(
        task_type=task_type,
        fitness_fn=fitness_fn,
        strategy='CMA_ES',
        population_size=128,
        num_generations=50,
        operation_space=SearchSpaceFactory.full_space(),
        seed=seed,
    )

    results_ablated = run_evolutionary_search(
        task_type=task_type,
        fitness_fn=fitness_fn,
        strategy='CMA_ES',
        population_size=128,
        num_generations=50,
        operation_space=SearchSpaceFactory.no_attention_space(),
        seed=seed,
    )

    performance_drop = results_full['best_fitness'] - results_ablated['best_fitness']

    return {
        'ablation_type': 'no_attention',
        'task_type': task_type,
        'full_fitness': results_full['best_fitness'],
        'ablated_fitness': results_ablated['best_fitness'],
        'performance_drop': performance_drop,
        'relative_drop': performance_drop / results_full['best_fitness'] if results_full['best_fitness'] > 0 else 0,
    }


def run_quick_ablation_study(
    fitness_fn_factory: Callable[[str], Callable],
    tasks: Optional[List[str]] = None,
    ablations: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a quick ablation study with minimal configuration.

    Args:
        fitness_fn_factory: Factory function creating fitness evaluators
        tasks: List of task types
        ablations: List of ablation types
        seed: Random seed

    Returns:
        Ablation study results
    """
    tasks = tasks or ['simple_sequence', 'tomi', 'hitom_2']
    ablations = ablations or ['full', 'no_skip', 'no_attention']

    config = AblationConfig(
        task_types=tasks,
        ablation_types=ablations,
        seeds=[seed],
        population_size=64,
        num_generations=25,
    )

    study = AblationStudy(config, fitness_fn_factory)
    return study.run_full_study()
