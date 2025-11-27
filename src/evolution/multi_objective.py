"""
Multi-Objective Neural Architecture Search for ToM

Implements Pareto optimization for discovering architectures that balance:
1. Task performance (accuracy on ToM tasks)
2. Architecture simplicity (fewer parameters = more interpretable)
3. Skip connection ratio (hypothesis-driven objective)
4. Recursive structure score (hypothesis-driven objective)

Uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) for
Pareto-optimal architecture discovery.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import copy
from datetime import datetime

from .search_space import (
    ArchitectureGenome, ArchitectureMetrics, CellArchitecture, CellNode,
    OPERATION_SPACE, create_random_genome, mutate_genome, crossover_genomes,
    genome_distance
)


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective evolution"""
    population_size: int = 256
    num_generations: int = 100
    num_objectives: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9

    # Crowding distance parameters
    archive_size: int = 100

    # Objectives to optimize (maximize)
    objectives: List[str] = field(default_factory=lambda: [
        'task_performance',
        'simplicity',
        'recursive_score',
    ])

    seed: int = 42


@dataclass
class Individual:
    """Individual in multi-objective population"""
    genome: ArchitectureGenome
    objectives: np.ndarray  # Objective values
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0

    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another"""
        better_in_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] < other.objectives[i]:
                return False
            if self.objectives[i] > other.objectives[i]:
                better_in_one = True
        return better_in_one

    def __lt__(self, other: 'Individual') -> bool:
        """Comparison for sorting"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance


class NSGA2Engine:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) Implementation

    Multi-objective optimization for neural architecture search.
    Returns a Pareto front of architectures balancing multiple objectives.
    """

    def __init__(
        self,
        config: MultiObjectiveConfig,
        objective_fns: Dict[str, Callable[[ArchitectureGenome], float]],
        operation_space: Optional[Dict] = None,
    ):
        """
        Args:
            config: Multi-objective configuration
            objective_fns: Dictionary of objective functions
            operation_space: Custom operation space
        """
        self.config = config
        self.objective_fns = objective_fns
        self.operation_space = operation_space or OPERATION_SPACE

        np.random.seed(config.seed)

        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.archive: List[Individual] = []
        self.generation = 0

        # History tracking
        self.history = {
            'pareto_sizes': [],
            'hypervolume': [],
            'objective_stats': {obj: [] for obj in config.objectives},
        }

    def initialize_population(self):
        """Initialize random population"""
        self.population = []

        for _ in range(self.config.population_size):
            genome = create_random_genome(operation_space=self.operation_space)
            objectives = self._evaluate_objectives(genome)
            individual = Individual(genome=genome, objectives=objectives)
            self.population.append(individual)

        # Initial ranking
        self._fast_non_dominated_sort(self.population)
        self._assign_crowding_distance(self.population)

    def _evaluate_objectives(self, genome: ArchitectureGenome) -> np.ndarray:
        """Evaluate all objectives for a genome"""
        objectives = np.zeros(len(self.config.objectives))

        for i, obj_name in enumerate(self.config.objectives):
            if obj_name in self.objective_fns:
                try:
                    objectives[i] = self.objective_fns[obj_name](genome)
                except Exception as e:
                    print(f"Objective evaluation failed for {obj_name}: {e}")
                    objectives[i] = 0.0
            else:
                # Built-in objectives based on architecture metrics
                objectives[i] = self._compute_builtin_objective(obj_name, genome)

        return objectives

    def _compute_builtin_objective(self, obj_name: str, genome: ArchitectureGenome) -> float:
        """Compute built-in objectives from architecture metrics"""
        metrics = ArchitectureMetrics(genome).compute_all()

        if obj_name == 'simplicity':
            # Negative parameter count (we maximize)
            return -metrics['total_parameters'] / 1e6

        elif obj_name == 'skip_ratio':
            # Ratio of skip connections
            total_ops = genome.num_nodes * 2 * genome.num_cells
            return metrics['num_skip_connections'] / max(total_ops, 1)

        elif obj_name == 'attention_ratio':
            total_ops = genome.num_nodes * 2 * genome.num_cells
            return metrics['num_attention_ops'] / max(total_ops, 1)

        elif obj_name == 'recursive_score':
            return metrics['recursive_depth'] / max(genome.num_cells, 1)

        elif obj_name == 'effective_depth':
            return metrics['effective_depth']

        elif obj_name == 'compression_ratio':
            return metrics['compression_ratio']

        return 0.0

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting algorithm.

        Returns list of fronts, where fronts[0] is the Pareto front.
        """
        fronts: List[List[Individual]] = [[]]

        # For each individual
        dominated_by: Dict[int, List[int]] = {i: [] for i in range(len(population))}
        domination_count: Dict[int, int] = {i: 0 for i in range(len(population))}

        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i == j:
                    continue

                if p.dominates(q):
                    dominated_by[i].append(j)
                elif q.dominates(p):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                p.rank = 0
                fronts[0].append(p)

        # Generate subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for p_idx in range(len(population)):
                if population[p_idx] in fronts[current_front]:
                    for q_idx in dominated_by[p_idx]:
                        domination_count[q_idx] -= 1
                        if domination_count[q_idx] == 0:
                            population[q_idx].rank = current_front + 1
                            next_front.append(population[q_idx])

            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _assign_crowding_distance(self, population: List[Individual]):
        """Assign crowding distance to individuals"""
        if len(population) <= 2:
            for ind in population:
                ind.crowding_distance = float('inf')
            return

        n = len(population)
        for ind in population:
            ind.crowding_distance = 0.0

        # For each objective
        for m in range(len(self.config.objectives)):
            # Sort by objective
            population.sort(key=lambda x: x.objectives[m])

            # Boundary points get infinite distance
            population[0].crowding_distance = float('inf')
            population[-1].crowding_distance = float('inf')

            # Range of objective values
            obj_min = population[0].objectives[m]
            obj_max = population[-1].objectives[m]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Assign distances
            for i in range(1, n - 1):
                distance = (population[i + 1].objectives[m] - population[i - 1].objectives[m]) / obj_range
                population[i].crowding_distance += distance

    def _tournament_selection(self, population: List[Individual], tournament_size: int = 2) -> Individual:
        """Binary tournament selection based on rank and crowding distance"""
        selected = np.random.choice(len(population), size=tournament_size, replace=False)
        candidates = [population[i] for i in selected]
        return min(candidates)  # Uses __lt__ which compares rank then crowding distance

    def _create_offspring(self) -> List[Individual]:
        """Create offspring population through selection, crossover, mutation"""
        offspring = []

        while len(offspring) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection(self.population)
            parent2 = self._tournament_selection(self.population)

            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1_genome, child2_genome = crossover_genomes(
                    parent1.genome, parent2.genome, self.operation_space
                )
            else:
                child1_genome = copy.deepcopy(parent1.genome)
                child2_genome = copy.deepcopy(parent2.genome)

            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child1_genome = mutate_genome(child1_genome, self.config.mutation_rate, self.operation_space)
            if np.random.random() < self.config.mutation_rate:
                child2_genome = mutate_genome(child2_genome, self.config.mutation_rate, self.operation_space)

            # Evaluate and create individuals
            for genome in [child1_genome, child2_genome]:
                if len(offspring) >= self.config.population_size:
                    break
                objectives = self._evaluate_objectives(genome)
                offspring.append(Individual(genome=genome, objectives=objectives))

        return offspring[:self.config.population_size]

    def _environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        """Select next generation from combined parent + offspring population"""
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)

        # Select individuals
        next_population = []

        for front in fronts:
            if len(next_population) + len(front) <= self.config.population_size:
                next_population.extend(front)
            else:
                # Need to select subset from this front
                remaining = self.config.population_size - len(next_population)
                self._assign_crowding_distance(front)
                front.sort()  # Sort by rank then crowding distance
                next_population.extend(front[:remaining])
                break

        return next_population

    def step(self) -> Dict[str, Any]:
        """Execute one generation of NSGA-II"""
        if not self.population:
            self.initialize_population()

        # Create offspring
        offspring = self._create_offspring()

        # Combine parent and offspring
        combined = self.population + offspring

        # Environmental selection
        self.population = self._environmental_selection(combined)

        # Update Pareto front
        self._update_pareto_front()

        # Update archive
        self._update_archive()

        # Record history
        self._record_history()

        result = {
            'generation': self.generation,
            'pareto_size': len(self.pareto_front),
            'archive_size': len(self.archive),
        }

        self.generation += 1

        return result

    def _update_pareto_front(self):
        """Extract current Pareto front from population"""
        self.pareto_front = [ind for ind in self.population if ind.rank == 0]

    def _update_archive(self):
        """Update archive with best solutions found"""
        for ind in self.pareto_front:
            # Check if dominated by any archive member
            is_dominated = any(arch.dominates(ind) for arch in self.archive)

            if not is_dominated:
                # Remove dominated archive members
                self.archive = [arch for arch in self.archive if not ind.dominates(arch)]
                self.archive.append(copy.deepcopy(ind))

        # Trim archive if too large
        if len(self.archive) > self.config.archive_size:
            self._assign_crowding_distance(self.archive)
            self.archive.sort()
            self.archive = self.archive[:self.config.archive_size]

    def _record_history(self):
        """Record evolution history"""
        self.history['pareto_sizes'].append(len(self.pareto_front))

        # Record objective statistics
        for i, obj_name in enumerate(self.config.objectives):
            obj_values = [ind.objectives[i] for ind in self.pareto_front]
            self.history['objective_stats'][obj_name].append({
                'mean': float(np.mean(obj_values)),
                'max': float(np.max(obj_values)),
                'min': float(np.min(obj_values)),
            })

    def run(self, num_generations: Optional[int] = None) -> Dict[str, Any]:
        """Run full NSGA-II evolution"""
        num_generations = num_generations or self.config.num_generations

        print(f"\n{'='*60}")
        print("Starting NSGA-II Multi-Objective Evolution")
        print(f"Objectives: {self.config.objectives}")
        print(f"Population: {self.config.population_size}, Generations: {num_generations}")
        print(f"{'='*60}\n")

        for gen in range(num_generations):
            result = self.step()

            if gen % 10 == 0:
                print(f"Gen {gen:4d} | Pareto size: {result['pareto_size']} | Archive: {result['archive_size']}")

        print(f"\n{'='*60}")
        print("Evolution Complete!")
        print(f"Final Pareto front size: {len(self.pareto_front)}")
        print(f"Archive size: {len(self.archive)}")
        print(f"{'='*60}\n")

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get final results"""
        pareto_results = []
        for ind in self.pareto_front:
            metrics = ArchitectureMetrics(ind.genome)
            pareto_results.append({
                'genome': ind.genome.to_dict(),
                'objectives': ind.objectives.tolist(),
                'objective_names': self.config.objectives,
                'metrics': metrics.compute_all(),
                'hypothesis_metrics': metrics.get_hypothesis_metrics(),
            })

        archive_results = []
        for ind in self.archive:
            metrics = ArchitectureMetrics(ind.genome)
            archive_results.append({
                'genome': ind.genome.to_dict(),
                'objectives': ind.objectives.tolist(),
                'metrics': metrics.compute_all(),
            })

        return {
            'pareto_front': pareto_results,
            'archive': archive_results,
            'history': self.history,
            'config': {
                'population_size': self.config.population_size,
                'num_generations': self.config.num_generations,
                'objectives': self.config.objectives,
            },
            'timestamp': datetime.now().isoformat(),
        }

    def get_pareto_front(self) -> List[Tuple[ArchitectureGenome, np.ndarray]]:
        """Get Pareto front as list of (genome, objectives) tuples"""
        return [(ind.genome, ind.objectives) for ind in self.pareto_front]


def multi_objective_fitness(
    genome: ArchitectureGenome,
    task_evaluator: Callable[[ArchitectureGenome], float],
) -> Dict[str, float]:
    """
    Multi-objective fitness function combining:
    1. Task performance (from evaluator)
    2. Architecture simplicity (negative parameters)
    3. Recursive structure score

    Args:
        genome: Architecture genome
        task_evaluator: Function to evaluate task performance

    Returns:
        Dictionary of objective values
    """
    metrics = ArchitectureMetrics(genome).compute_all()

    return {
        'task_performance': task_evaluator(genome),
        'simplicity': -metrics['total_parameters'] / 1e6,
        'recursive_score': metrics['recursive_depth'] / max(genome.num_cells, 1),
    }


def run_pareto_search(
    task_evaluator: Callable[[ArchitectureGenome], float],
    objectives: Optional[List[str]] = None,
    population_size: int = 256,
    num_generations: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run Pareto-optimal architecture search.

    Args:
        task_evaluator: Function evaluating task performance
        objectives: List of objective names
        population_size: Population size
        num_generations: Number of generations
        seed: Random seed

    Returns:
        Dictionary with Pareto front and evolution history
    """
    objectives = objectives or ['task_performance', 'simplicity', 'recursive_score']

    config = MultiObjectiveConfig(
        population_size=population_size,
        num_generations=num_generations,
        objectives=objectives,
        seed=seed,
    )

    # Create objective functions
    objective_fns = {
        'task_performance': task_evaluator,
    }

    engine = NSGA2Engine(config, objective_fns)
    results = engine.run(num_generations)

    return results


def analyze_pareto_front(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze Pareto front for hypothesis testing.

    Returns analysis of:
    - Skip connection prevalence
    - Attention mechanism usage
    - Recursive structure patterns
    """
    pareto = results['pareto_front']

    if not pareto:
        return {'error': 'Empty Pareto front'}

    # Collect metrics
    skip_connections = [p['metrics']['num_skip_connections'] for p in pareto]
    attention_ops = [p['metrics']['num_attention_ops'] for p in pareto]
    recursive_depths = [p['metrics']['recursive_depth'] for p in pareto]
    total_params = [p['metrics']['total_parameters'] for p in pareto]

    analysis = {
        'pareto_size': len(pareto),

        'skip_connections': {
            'mean': float(np.mean(skip_connections)),
            'std': float(np.std(skip_connections)),
            'max': float(np.max(skip_connections)),
            'prevalence': float(np.mean([s > 0 for s in skip_connections])),
        },

        'attention_ops': {
            'mean': float(np.mean(attention_ops)),
            'std': float(np.std(attention_ops)),
            'max': float(np.max(attention_ops)),
            'prevalence': float(np.mean([a > 0 for a in attention_ops])),
        },

        'recursive_depth': {
            'mean': float(np.mean(recursive_depths)),
            'std': float(np.std(recursive_depths)),
            'max': float(np.max(recursive_depths)),
        },

        'parameters': {
            'mean': float(np.mean(total_params)),
            'std': float(np.std(total_params)),
            'range': (float(np.min(total_params)), float(np.max(total_params))),
        },

        # Correlation analysis
        'correlations': {
            'skip_vs_performance': _compute_correlation(
                [p['objectives'][0] for p in pareto], skip_connections
            ),
            'attention_vs_performance': _compute_correlation(
                [p['objectives'][0] for p in pareto], attention_ops
            ),
            'params_vs_performance': _compute_correlation(
                [p['objectives'][0] for p in pareto], total_params
            ),
        }
    }

    return analysis


def _compute_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient"""
    if len(x) < 2:
        return 0.0

    x = np.array(x)
    y = np.array(y)

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])
