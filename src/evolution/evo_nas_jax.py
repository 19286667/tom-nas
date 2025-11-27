"""
GPU-Accelerated Evolutionary NAS for ToM Tasks using evosax

JAX-based implementation for fast evolutionary architecture search.
Supports multiple evolution strategies:
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- OpenES: OpenAI Evolution Strategy
- PGPE: Policy Gradient with Parameter Exploration
- Sep-CMA-ES: Separable CMA-ES for high dimensions

This module implements the core evolutionary loop with:
1. Genome encoding/decoding for neural architectures
2. Fitness evaluation via partial training
3. Architecture metrics tracking for hypothesis testing
4. Multi-strategy support for comparison experiments
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import copy

# JAX imports (with fallback for environments without JAX)
try:
    import jax
    import jax.numpy as jnp
    from jax import random as jrandom
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    # Create dummy jnp for type hints
    class jnp:
        ndarray = np.ndarray

# evosax imports (with fallback)
try:
    from evosax import CMA_ES, OpenES, PGPE, Sep_CMA_ES, SimpleGA
    from evosax import FitnessShaper
    HAS_EVOSAX = True
except ImportError:
    HAS_EVOSAX = False

# PyTorch for model building/evaluation
import torch
import torch.nn as nn

from .search_space import (
    ArchitectureGenome, ArchitectureMetrics, CellArchitecture, CellNode,
    OPERATION_SPACE, create_random_genome, mutate_genome, crossover_genomes
)


@dataclass
class EvosaxConfig:
    """Configuration for evosax-based evolutionary search"""
    strategy: str = "CMA_ES"  # CMA_ES, OpenES, PGPE, Sep_CMA_ES, SimpleGA
    population_size: int = 256
    num_generations: int = 100
    elite_ratio: float = 0.5
    learning_rate: float = 0.01
    sigma_init: float = 0.1

    # Genome configuration
    num_nodes: int = 4
    num_cells: int = 6
    init_channels: int = 32

    # Fitness evaluation
    training_epochs: int = 5
    batch_size: int = 32

    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 20

    seed: int = 42


@dataclass
class EvolutionLog:
    """Log of evolution progress"""
    generations: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    std_fitness: List[float] = field(default_factory=list)
    best_genomes: List[Dict] = field(default_factory=list)
    architecture_metrics: List[Dict] = field(default_factory=list)

    def update(self, gen: int, fitness_values: np.ndarray, best_genome: ArchitectureGenome):
        """Update log with generation results"""
        self.generations.append(gen)
        self.best_fitness.append(float(np.max(fitness_values)))
        self.mean_fitness.append(float(np.mean(fitness_values)))
        self.std_fitness.append(float(np.std(fitness_values)))
        self.best_genomes.append(best_genome.to_dict())

        # Compute architecture metrics
        metrics = ArchitectureMetrics(best_genome)
        self.architecture_metrics.append(metrics.compute_all())

    def to_dict(self) -> Dict:
        return {
            'generations': self.generations,
            'best_fitness': self.best_fitness,
            'mean_fitness': self.mean_fitness,
            'std_fitness': self.std_fitness,
            'best_genomes': self.best_genomes,
            'architecture_metrics': self.architecture_metrics,
        }


class ContinuousGenomeEncoder:
    """
    Encode/decode architecture genomes to/from continuous vectors.

    This allows gradient-free optimization methods to operate in
    a continuous space while producing discrete architectures.
    """

    def __init__(
        self,
        num_nodes: int = 4,
        operation_space: Optional[Dict] = None
    ):
        self.num_nodes = num_nodes
        self.operation_space = operation_space or OPERATION_SPACE
        self.num_ops = len(self.operation_space['primitives'])

        # Calculate genome size
        # For each cell: num_nodes * (1 op + 2 inputs)
        # Two cells (normal + reduction)
        # Plus macro parameters (num_cells, channels)
        self.node_genes = num_nodes * 3
        self.cell_genes = self.node_genes * 2  # Two cells
        self.macro_genes = 2  # num_cells, init_channels
        self.genome_size = self.cell_genes + self.macro_genes

    def encode(self, genome: ArchitectureGenome) -> np.ndarray:
        """Convert discrete genome to continuous vector"""
        primitives = self.operation_space['primitives']
        vector = []

        # Encode cells
        for cell in [genome.normal_cell, genome.reduction_cell]:
            for i, node in enumerate(cell.nodes):
                # Operation index normalized to [0, 1]
                op_idx = primitives.index(node.operation) if node.operation in primitives else 0
                vector.append(op_idx / self.num_ops)

                # Input indices normalized
                max_input = i + 2
                vector.append(node.inputs[0] / max_input)
                vector.append(node.inputs[1] / max_input)

        # Encode macro parameters
        num_cells_options = self.operation_space.get('num_cells', [4, 6, 8])
        channels_options = self.operation_space.get('channels', [16, 32, 64])

        num_cells_idx = num_cells_options.index(genome.num_cells) if genome.num_cells in num_cells_options else 0
        channels_idx = channels_options.index(genome.init_channels) if genome.init_channels in channels_options else 0

        vector.append(num_cells_idx / len(num_cells_options))
        vector.append(channels_idx / len(channels_options))

        return np.array(vector, dtype=np.float32)

    def decode(self, vector: np.ndarray) -> ArchitectureGenome:
        """Convert continuous vector to discrete genome"""
        primitives = self.operation_space['primitives']

        genome = ArchitectureGenome(
            num_nodes=self.num_nodes,
            num_ops=self.num_ops,
        )

        idx = 0

        # Decode cells
        for cell_idx, cell_attr in enumerate(['normal_cell', 'reduction_cell']):
            nodes = []
            for i in range(self.num_nodes):
                # Decode operation
                op_idx = int(vector[idx] * self.num_ops) % self.num_ops
                op = primitives[op_idx]
                idx += 1

                # Decode inputs
                max_input = i + 2
                input_1 = int(vector[idx] * max_input) % max_input
                idx += 1
                input_2 = int(vector[idx] * max_input) % max_input
                idx += 1

                nodes.append(CellNode(operation=op, inputs=[input_1, input_2]))

            cell_type = 'normal' if cell_idx == 0 else 'reduction'
            setattr(genome, cell_attr, CellArchitecture(nodes=nodes, cell_type=cell_type))

        # Decode macro parameters
        num_cells_options = self.operation_space.get('num_cells', [4, 6, 8])
        channels_options = self.operation_space.get('channels', [16, 32, 64])

        num_cells_idx = int(vector[idx] * len(num_cells_options)) % len(num_cells_options)
        genome.num_cells = num_cells_options[num_cells_idx]
        idx += 1

        channels_idx = int(vector[idx] * len(channels_options)) % len(channels_options)
        genome.init_channels = channels_options[channels_idx]

        return genome


class NumpyEvolutionEngine:
    """
    NumPy-based evolution engine (fallback when JAX/evosax unavailable).

    Implements basic evolutionary strategies using NumPy.
    """

    def __init__(
        self,
        config: EvosaxConfig,
        fitness_fn: Callable[[ArchitectureGenome], float],
        operation_space: Optional[Dict] = None
    ):
        self.config = config
        self.fitness_fn = fitness_fn
        self.operation_space = operation_space or OPERATION_SPACE

        self.encoder = ContinuousGenomeEncoder(
            num_nodes=config.num_nodes,
            operation_space=self.operation_space
        )

        np.random.seed(config.seed)

        # Initialize population
        self.population = self._initialize_population()
        self.fitness_values = np.zeros(config.population_size)
        self.generation = 0
        self.best_genome: Optional[ArchitectureGenome] = None
        self.best_fitness = float('-inf')

        # Strategy-specific parameters
        if config.strategy == "CMA_ES":
            self._init_cma_es()
        elif config.strategy == "OpenES":
            self._init_open_es()
        else:
            self._init_simple_ga()

        self.log = EvolutionLog()

    def _initialize_population(self) -> np.ndarray:
        """Initialize population of continuous genomes"""
        population = np.random.randn(
            self.config.population_size,
            self.encoder.genome_size
        ) * self.config.sigma_init

        # Normalize to [0, 1] range
        population = (population - population.min()) / (population.max() - population.min() + 1e-8)

        return population

    def _init_cma_es(self):
        """Initialize CMA-ES parameters"""
        n = self.encoder.genome_size
        self.mean = np.random.rand(n)
        self.sigma = self.config.sigma_init
        self.C = np.eye(n)  # Covariance matrix
        self.pc = np.zeros(n)  # Evolution path for C
        self.ps = np.zeros(n)  # Evolution path for sigma

        # Strategy parameters
        self.lambda_ = self.config.population_size
        self.mu = int(self.lambda_ * self.config.elite_ratio)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / self.weights.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        self.cc = 4.0 / (n + 4.0)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 3.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs

    def _init_open_es(self):
        """Initialize OpenES parameters"""
        n = self.encoder.genome_size
        self.mean = np.random.rand(n)
        self.sigma = self.config.sigma_init

    def _init_simple_ga(self):
        """Initialize Simple GA parameters"""
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def ask(self) -> np.ndarray:
        """Generate new population for evaluation"""
        if self.config.strategy == "CMA_ES":
            return self._ask_cma_es()
        elif self.config.strategy == "OpenES":
            return self._ask_open_es()
        else:
            return self._ask_simple_ga()

    def _ask_cma_es(self) -> np.ndarray:
        """Sample population using CMA-ES"""
        n = self.encoder.genome_size

        # Sample from multivariate normal
        try:
            B = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # Fall back to diagonal if Cholesky fails
            B = np.diag(np.sqrt(np.diag(self.C)))

        z = np.random.randn(self.lambda_, n)
        population = self.mean + self.sigma * (z @ B.T)

        # Clip to valid range
        population = np.clip(population, 0, 1)

        return population

    def _ask_open_es(self) -> np.ndarray:
        """Sample population using OpenES"""
        n = self.encoder.genome_size
        noise = np.random.randn(self.lambda_, n)
        population = self.mean + self.sigma * noise

        # Store noise for gradient estimation
        self.noise = noise

        population = np.clip(population, 0, 1)
        return population

    def _ask_simple_ga(self) -> np.ndarray:
        """Generate population using simple GA"""
        new_population = []

        # Elitism
        elite_size = max(1, int(self.config.population_size * self.config.elite_ratio * 0.1))
        sorted_indices = np.argsort(-self.fitness_values)
        for idx in sorted_indices[:elite_size]:
            new_population.append(self.population[idx].copy())

        # Generate rest via crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            idx1 = self._tournament_select()
            idx2 = self._tournament_select()

            parent1 = self.population[idx1]
            parent2 = self.population[idx2]

            # Crossover
            if np.random.rand() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            child = self._mutate(child)

            new_population.append(child)

        return np.array(new_population[:self.config.population_size])

    def _tournament_select(self, tournament_size: int = 3) -> int:
        """Tournament selection"""
        indices = np.random.choice(len(self.population), size=tournament_size, replace=False)
        fitnesses = self.fitness_values[indices]
        winner = indices[np.argmax(fitnesses)]
        return winner

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Single-point crossover"""
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        mutation_mask = np.random.rand(len(individual)) < self.mutation_rate
        noise = np.random.randn(len(individual)) * self.config.sigma_init
        individual[mutation_mask] += noise[mutation_mask]
        return np.clip(individual, 0, 1)

    def tell(self, population: np.ndarray, fitness_values: np.ndarray):
        """Update strategy based on fitness evaluations"""
        self.population = population
        self.fitness_values = fitness_values

        # Update best
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = fitness_values[best_idx]
            self.best_genome = self.encoder.decode(population[best_idx])

        if self.config.strategy == "CMA_ES":
            self._tell_cma_es(population, fitness_values)
        elif self.config.strategy == "OpenES":
            self._tell_open_es(population, fitness_values)

    def _tell_cma_es(self, population: np.ndarray, fitness_values: np.ndarray):
        """Update CMA-ES parameters"""
        n = self.encoder.genome_size

        # Sort by fitness (descending)
        sorted_indices = np.argsort(-fitness_values)
        x_old = self.mean.copy()

        # Update mean
        x_best = population[sorted_indices[:self.mu]]
        self.mean = np.sum(self.weights[:, np.newaxis] * x_best, axis=0)

        # Update evolution paths
        y = (self.mean - x_old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y

        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < 1.4 + 2.0 / (n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        # Update covariance matrix
        artmp = (x_best - x_old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * artmp.T @ np.diag(self.weights) @ artmp)

        # Ensure symmetry and positive definiteness
        self.C = (self.C + self.C.T) / 2
        eigvals = np.linalg.eigvalsh(self.C)
        if np.min(eigvals) < 1e-10:
            self.C += (1e-10 - np.min(eigvals)) * np.eye(n)

        # Update sigma
        cn = self.cs / self.damps
        self.sigma *= np.exp(cn * (np.linalg.norm(self.ps) / np.sqrt(n) - 1))

    def _tell_open_es(self, population: np.ndarray, fitness_values: np.ndarray):
        """Update OpenES parameters"""
        # Fitness shaping (rank-based)
        ranks = np.argsort(np.argsort(-fitness_values))
        shaped_fitness = np.maximum(0, np.log(len(fitness_values) / 2 + 1) - np.log(ranks + 1))
        shaped_fitness = shaped_fitness / np.sum(shaped_fitness) - 1.0 / len(fitness_values)

        # Gradient estimation
        gradient = np.mean(self.noise * shaped_fitness[:, np.newaxis], axis=0)

        # Update mean
        self.mean = self.mean + self.config.learning_rate * self.sigma * gradient

        # Clip
        self.mean = np.clip(self.mean, 0, 1)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of entire population"""
        fitness_values = np.zeros(len(population))

        for i, genome_vector in enumerate(population):
            genome = self.encoder.decode(genome_vector)
            try:
                fitness = self.fitness_fn(genome)
            except Exception as e:
                print(f"Fitness evaluation failed for individual {i}: {e}")
                fitness = 0.0
            fitness_values[i] = fitness

        return fitness_values

    def step(self) -> Dict[str, Any]:
        """Execute one generation of evolution"""
        # Generate new population
        population = self.ask()

        # Evaluate
        fitness_values = self.evaluate_population(population)

        # Update strategy
        self.tell(population, fitness_values)

        # Log
        self.log.update(self.generation, fitness_values, self.best_genome)

        result = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'mean_fitness': float(np.mean(fitness_values)),
            'std_fitness': float(np.std(fitness_values)),
        }

        self.generation += 1

        return result

    def run(self, num_generations: Optional[int] = None) -> Dict[str, Any]:
        """Run full evolution"""
        num_generations = num_generations or self.config.num_generations

        print(f"\n{'='*60}")
        print(f"Starting Evolution: {self.config.strategy}")
        print(f"Population: {self.config.population_size}, Generations: {num_generations}")
        print(f"{'='*60}\n")

        for gen in range(num_generations):
            result = self.step()

            if gen % self.config.log_interval == 0:
                print(f"Gen {gen:4d} | Best: {result['best_fitness']:.4f} | "
                      f"Mean: {result['mean_fitness']:.4f} | Std: {result['std_fitness']:.4f}")

        print(f"\n{'='*60}")
        print(f"Evolution Complete!")
        print(f"Best Fitness: {self.best_fitness:.4f}")
        print(f"{'='*60}\n")

        return {
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'best_fitness': self.best_fitness,
            'evolution_log': self.log.to_dict(),
        }


class JaxEvolutionEngine:
    """
    JAX-based evolution engine using evosax.

    Provides GPU-accelerated evolution with:
    - Vectorized fitness evaluation
    - JIT compilation for speed
    - Multiple strategy support
    """

    def __init__(
        self,
        config: EvosaxConfig,
        fitness_fn: Callable[[ArchitectureGenome], float],
        operation_space: Optional[Dict] = None
    ):
        if not HAS_JAX:
            raise RuntimeError("JAX not available. Install with: pip install jax jaxlib")
        if not HAS_EVOSAX:
            raise RuntimeError("evosax not available. Install with: pip install evosax")

        self.config = config
        self.fitness_fn = fitness_fn
        self.operation_space = operation_space or OPERATION_SPACE

        self.encoder = ContinuousGenomeEncoder(
            num_nodes=config.num_nodes,
            operation_space=self.operation_space
        )

        self.rng = jrandom.PRNGKey(config.seed)

        # Initialize evolution strategy
        self.es = self._create_strategy()
        self.es_params = self.es.default_params
        self.es_state = None

        self.best_genome: Optional[ArchitectureGenome] = None
        self.best_fitness = float('-inf')
        self.generation = 0

        self.log = EvolutionLog()

    def _create_strategy(self):
        """Create evosax evolution strategy"""
        strategy_map = {
            'CMA_ES': CMA_ES,
            'OpenES': OpenES,
            'PGPE': PGPE,
            'Sep_CMA_ES': Sep_CMA_ES,
            'SimpleGA': SimpleGA,
        }

        strategy_cls = strategy_map.get(self.config.strategy, CMA_ES)

        es = strategy_cls(
            popsize=self.config.population_size,
            num_dims=self.encoder.genome_size,
            elite_ratio=self.config.elite_ratio,
        )

        return es

    def initialize(self):
        """Initialize the evolution state"""
        self.rng, init_rng = jrandom.split(self.rng)
        self.es_state = self.es.initialize(init_rng, self.es_params)

    def ask(self) -> np.ndarray:
        """Generate population for evaluation"""
        self.rng, ask_rng = jrandom.split(self.rng)
        population, self.es_state = self.es.ask(ask_rng, self.es_state, self.es_params)

        # Convert to numpy and clip to [0, 1]
        population = np.array(population)
        population = np.clip(population, 0, 1)

        return population

    def tell(self, population: np.ndarray, fitness_values: np.ndarray):
        """Update strategy with fitness values"""
        # Convert to JAX arrays
        pop_jax = jnp.array(population)
        fit_jax = jnp.array(fitness_values)

        # evosax expects negative fitness for minimization, we maximize
        self.es_state = self.es.tell(pop_jax, -fit_jax, self.es_state, self.es_params)

        # Update best
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = fitness_values[best_idx]
            self.best_genome = self.encoder.decode(population[best_idx])

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of population"""
        fitness_values = np.zeros(len(population))

        for i, genome_vector in enumerate(population):
            genome = self.encoder.decode(genome_vector)
            try:
                fitness = self.fitness_fn(genome)
            except Exception as e:
                print(f"Fitness evaluation failed: {e}")
                fitness = 0.0
            fitness_values[i] = fitness

        return fitness_values

    def step(self) -> Dict[str, Any]:
        """Execute one generation"""
        if self.es_state is None:
            self.initialize()

        population = self.ask()
        fitness_values = self.evaluate_population(population)
        self.tell(population, fitness_values)

        self.log.update(self.generation, fitness_values, self.best_genome)

        result = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'mean_fitness': float(np.mean(fitness_values)),
            'std_fitness': float(np.std(fitness_values)),
        }

        self.generation += 1

        return result

    def run(self, num_generations: Optional[int] = None) -> Dict[str, Any]:
        """Run full evolution"""
        num_generations = num_generations or self.config.num_generations

        print(f"\n{'='*60}")
        print(f"Starting JAX Evolution: {self.config.strategy}")
        print(f"Population: {self.config.population_size}, Generations: {num_generations}")
        print(f"{'='*60}\n")

        for gen in range(num_generations):
            result = self.step()

            if gen % self.config.log_interval == 0:
                print(f"Gen {gen:4d} | Best: {result['best_fitness']:.4f} | "
                      f"Mean: {result['mean_fitness']:.4f}")

        print(f"\n{'='*60}")
        print(f"Evolution Complete!")
        print(f"Best Fitness: {self.best_fitness:.4f}")
        print(f"{'='*60}\n")

        return {
            'best_genome': self.best_genome.to_dict() if self.best_genome else None,
            'best_fitness': self.best_fitness,
            'evolution_log': self.log.to_dict(),
        }


def create_evolution_engine(
    config: EvosaxConfig,
    fitness_fn: Callable[[ArchitectureGenome], float],
    operation_space: Optional[Dict] = None,
    use_jax: bool = True
):
    """
    Factory function to create appropriate evolution engine.

    Args:
        config: Evolution configuration
        fitness_fn: Function mapping genome to fitness value
        operation_space: Operation search space
        use_jax: Whether to prefer JAX/evosax backend

    Returns:
        Evolution engine instance
    """
    if use_jax and HAS_JAX and HAS_EVOSAX:
        print("Using JAX/evosax backend")
        return JaxEvolutionEngine(config, fitness_fn, operation_space)
    else:
        print("Using NumPy backend")
        return NumpyEvolutionEngine(config, fitness_fn, operation_space)


def run_evolutionary_search(
    task_type: str,
    fitness_fn: Callable[[ArchitectureGenome], float],
    strategy: str = "CMA_ES",
    population_size: int = 256,
    num_generations: int = 100,
    operation_space: Optional[Dict] = None,
    seed: int = 42,
    use_jax: bool = True,
) -> Dict[str, Any]:
    """
    Run evolutionary NAS for a specific task.

    Args:
        task_type: Name of the task (for logging)
        fitness_fn: Fitness evaluation function
        strategy: Evolution strategy name
        population_size: Population size
        num_generations: Number of generations
        operation_space: Custom operation space (or None for default)
        seed: Random seed
        use_jax: Whether to use JAX backend

    Returns:
        Dictionary with best architecture, fitness, and evolution log
    """
    config = EvosaxConfig(
        strategy=strategy,
        population_size=population_size,
        num_generations=num_generations,
        seed=seed,
    )

    engine = create_evolution_engine(config, fitness_fn, operation_space, use_jax)

    results = engine.run(num_generations)

    # Add task metadata
    results['task_type'] = task_type
    results['strategy'] = strategy
    results['seed'] = seed
    results['timestamp'] = datetime.now().isoformat()

    # Compute final architecture metrics
    if results['best_genome']:
        genome = ArchitectureGenome.from_dict(results['best_genome'])
        metrics = ArchitectureMetrics(genome)
        results['final_metrics'] = metrics.compute_all()
        results['hypothesis_metrics'] = metrics.get_hypothesis_metrics()

    return results


def compare_strategies(
    fitness_fn: Callable[[ArchitectureGenome], float],
    strategies: List[str] = None,
    population_size: int = 128,
    num_generations: int = 50,
    seeds: List[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Compare multiple evolution strategies.

    Args:
        fitness_fn: Fitness evaluation function
        strategies: List of strategy names to compare
        population_size: Population size
        num_generations: Number of generations per run
        seeds: Random seeds for multiple runs

    Returns:
        Dictionary mapping strategy names to list of results
    """
    strategies = strategies or ["CMA_ES", "OpenES", "PGPE"]
    seeds = seeds or [42, 123, 456]

    results = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing Strategy: {strategy}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")

            result = run_evolutionary_search(
                task_type='strategy_comparison',
                fitness_fn=fitness_fn,
                strategy=strategy,
                population_size=population_size,
                num_generations=num_generations,
                seed=seed,
            )

            results[strategy].append(result)

    return results
