"""
POET Co-Evolution Engine

Paired Open-Ended Trailblazer for ToM Neural Architecture Search.
Co-evolves agent architectures with environment complexity.

Key innovation: ToM emerges because social survival requires it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
from collections import defaultdict
import copy


class ArchitectureType(Enum):
    """Neural architecture types for ToM agents"""
    TRN = "transparent_rnn"  # Interpretable reasoning
    RSAN = "recursive_self_attention"  # Hierarchical attention
    TRANSFORMER = "transformer"  # Communication-focused
    HYBRID = "hybrid"  # Evolved combination


@dataclass
class ArchitectureGene:
    """Genome for agent architecture (from existing NAS)"""
    arch_type: ArchitectureType = ArchitectureType.TRN
    num_layers: int = 2
    hidden_dim: int = 128
    num_heads: int = 4  # For attention architectures
    max_recursion: int = 5  # For ToM depth
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # ToM-specific parameters
    tom_depth: int = 2
    belief_update_rate: float = 0.7
    memory_capacity: int = 100

    def mutate(self, mutation_rate: float = 0.1,
              rng: Optional[np.random.Generator] = None) -> 'ArchitectureGene':
        """Create mutated copy of gene"""
        if rng is None:
            rng = np.random.default_rng()

        gene = copy.deepcopy(self)

        if rng.random() < mutation_rate:
            gene.arch_type = rng.choice(list(ArchitectureType))

        if rng.random() < mutation_rate:
            gene.num_layers = int(np.clip(gene.num_layers + rng.integers(-1, 2), 1, 6))

        if rng.random() < mutation_rate:
            gene.hidden_dim = int(np.clip(
                gene.hidden_dim * rng.uniform(0.5, 2.0), 32, 512
            ))

        if rng.random() < mutation_rate:
            gene.num_heads = rng.choice([2, 4, 8, 16])

        if rng.random() < mutation_rate:
            gene.max_recursion = int(np.clip(gene.max_recursion + rng.integers(-1, 2), 1, 7))

        if rng.random() < mutation_rate:
            gene.tom_depth = int(np.clip(gene.tom_depth + rng.integers(-1, 2), 0, 5))

        if rng.random() < mutation_rate:
            gene.learning_rate = float(np.clip(
                gene.learning_rate * rng.uniform(0.5, 2.0), 1e-5, 1e-2
            ))

        if rng.random() < mutation_rate:
            gene.belief_update_rate = float(np.clip(
                gene.belief_update_rate + rng.uniform(-0.1, 0.1), 0.1, 0.99
            ))

        if rng.random() < mutation_rate:
            gene.memory_capacity = int(np.clip(
                gene.memory_capacity * rng.uniform(0.5, 2.0), 20, 500
            ))

        return gene

    def crossover(self, other: 'ArchitectureGene',
                 rng: Optional[np.random.Generator] = None) -> 'ArchitectureGene':
        """Create offspring by combining two genes"""
        if rng is None:
            rng = np.random.default_rng()

        child = ArchitectureGene()

        # Randomly select each attribute from one parent
        for attr in ['arch_type', 'num_layers', 'hidden_dim', 'num_heads',
                     'max_recursion', 'dropout_rate', 'learning_rate',
                     'weight_decay', 'tom_depth', 'belief_update_rate', 'memory_capacity']:
            if rng.random() < 0.5:
                setattr(child, attr, getattr(self, attr))
            else:
                setattr(child, attr, getattr(other, attr))

        return child

    def complexity_score(self) -> float:
        """Calculate architecture complexity (for regularization)"""
        return (
            self.num_layers * 0.2 +
            self.hidden_dim / 512 * 0.3 +
            self.max_recursion * 0.1 +
            self.tom_depth * 0.2 +
            self.memory_capacity / 500 * 0.2
        )


@dataclass
class EnvironmentGene:
    """Genome for environment configuration"""
    # Size and structure
    world_size: int = 50
    wall_density: float = 0.05
    resource_density: float = 0.02

    # Social complexity
    num_agents: int = 10
    num_zombies: int = 2
    zombie_types: List[str] = field(default_factory=lambda: ['behavioral'])

    # Game mechanics
    payoff_type: str = "prisoners_dilemma"
    communication_enabled: bool = True
    coalition_enabled: bool = False
    deception_enabled: bool = False

    # Information asymmetry
    observation_radius: float = 10.0
    memory_noise: float = 0.1
    reputation_visible: bool = True

    # Benchmark embedding
    benchmark_rate: float = 0.1

    # Difficulty scalar (for curriculum)
    difficulty: float = 0.5

    def mutate(self, mutation_rate: float = 0.1,
              rng: Optional[np.random.Generator] = None) -> 'EnvironmentGene':
        """Create mutated copy of gene"""
        if rng is None:
            rng = np.random.default_rng()

        gene = copy.deepcopy(self)

        if rng.random() < mutation_rate:
            gene.world_size = int(np.clip(gene.world_size + rng.integers(-10, 11), 20, 100))

        if rng.random() < mutation_rate:
            gene.wall_density = float(np.clip(
                gene.wall_density + rng.uniform(-0.02, 0.02), 0, 0.2
            ))

        if rng.random() < mutation_rate:
            gene.num_agents = int(np.clip(gene.num_agents + rng.integers(-2, 3), 4, 30))

        if rng.random() < mutation_rate:
            gene.num_zombies = int(np.clip(gene.num_zombies + rng.integers(-1, 2), 0, gene.num_agents // 2))

        if rng.random() < mutation_rate:
            gene.observation_radius = float(np.clip(
                gene.observation_radius + rng.uniform(-2, 2), 3, 20
            ))

        if rng.random() < mutation_rate:
            gene.coalition_enabled = not gene.coalition_enabled

        if rng.random() < mutation_rate:
            gene.deception_enabled = not gene.deception_enabled

        if rng.random() < mutation_rate:
            gene.benchmark_rate = float(np.clip(
                gene.benchmark_rate + rng.uniform(-0.05, 0.05), 0.05, 0.3
            ))

        if rng.random() < mutation_rate:
            gene.difficulty = float(np.clip(
                gene.difficulty + rng.uniform(-0.1, 0.1), 0.1, 1.0
            ))

        return gene

    def get_tom_pressure(self) -> float:
        """Calculate how much ToM is needed for this environment"""
        pressure = 0.0

        # More agents = more social complexity
        pressure += min(1.0, self.num_agents / 20) * 0.2

        # Information asymmetry requires ToM
        pressure += (1.0 - self.observation_radius / 20) * 0.2

        # Deception requires ToM
        if self.deception_enabled:
            pressure += 0.2

        # Coalitions require social reasoning
        if self.coalition_enabled:
            pressure += 0.15

        # Benchmark embedding creates ToM tasks
        pressure += self.benchmark_rate * 0.25

        return min(1.0, pressure)


@dataclass
class AgentArchitectureGenome:
    """Combined genome for an agent (architecture + psychosocial)"""
    architecture: ArchitectureGene = field(default_factory=ArchitectureGene)
    archetype: str = "everyman"
    profile_seed: int = 0

    def mutate(self, rate: float = 0.1,
              rng: Optional[np.random.Generator] = None) -> 'AgentArchitectureGenome':
        """Mutate the combined genome"""
        if rng is None:
            rng = np.random.default_rng()

        genome = AgentArchitectureGenome(
            architecture=self.architecture.mutate(rate, rng),
            archetype=self.archetype,
            profile_seed=self.profile_seed,
        )

        if rng.random() < rate:
            archetypes = ['hero', 'caregiver', 'sage', 'rebel', 'creator',
                         'ruler', 'innocent', 'explorer', 'everyman', 'jester',
                         'lover', 'magician', 'outlaw']
            genome.archetype = rng.choice(archetypes)

        if rng.random() < rate:
            genome.profile_seed = rng.integers(0, 1000000)

        return genome


@dataclass
class CoevolutionPair:
    """A paired agent-environment for co-evolution"""
    agent_genome: AgentArchitectureGenome
    env_genome: EnvironmentGene
    fitness: float = 0.0
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    def evaluate(self, evaluator: Callable) -> float:
        """Evaluate this pair using provided evaluator"""
        self.fitness, self.benchmark_scores = evaluator(
            self.agent_genome, self.env_genome
        )
        return self.fitness


@dataclass
class POETEngine:
    """
    POET (Paired Open-Ended Trailblazer) for ToM-NAS.

    Co-evolves:
    - Agent architectures (what neural structure enables ToM)
    - Environments (what social complexity creates ToM pressure)

    Key insight: ToM emerges as adaptation to social environment.
    """
    # Population configuration
    population_size: int = 20
    elite_size: int = 2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

    # Evolution tracking
    generation: int = 0
    population: List[CoevolutionPair] = field(default_factory=list)
    archive: List[CoevolutionPair] = field(default_factory=list)  # Hall of fame

    # Curriculum settings
    min_difficulty: float = 0.2
    max_difficulty: float = 1.0
    difficulty_increase_rate: float = 0.02

    # Fitness weights
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        'survival': 0.3,
        'cooperation': 0.2,
        'zombie_detection': 0.2,
        'benchmark': 0.2,
        'complexity_penalty': 0.1,
    })

    # Random number generator
    _rng: np.random.Generator = field(init=False, repr=False)
    seed: Optional[int] = None

    # History
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def initialize_population(self):
        """Initialize diverse population of agent-environment pairs"""
        self.population = []

        for i in range(self.population_size):
            # Create diverse starting points
            agent_genome = AgentArchitectureGenome(
                architecture=ArchitectureGene(
                    arch_type=self._rng.choice(list(ArchitectureType)),
                    num_layers=self._rng.integers(1, 4),
                    hidden_dim=self._rng.choice([64, 128, 256]),
                    tom_depth=self._rng.integers(0, 4),
                ),
                archetype=self._rng.choice([
                    'hero', 'caregiver', 'sage', 'rebel', 'everyman'
                ]),
                profile_seed=self._rng.integers(0, 1000000),
            )

            env_genome = EnvironmentGene(
                world_size=self._rng.integers(30, 60),
                num_agents=self._rng.integers(6, 15),
                num_zombies=self._rng.integers(1, 3),
                difficulty=self.min_difficulty + self._rng.uniform(0, 0.2),
            )

            self.population.append(CoevolutionPair(
                agent_genome=agent_genome,
                env_genome=env_genome,
                generation=0,
            ))

    def evolve_generation(self, evaluator: Callable) -> Dict[str, Any]:
        """
        Evolve one generation.
        Returns statistics about the generation.
        """
        self.generation += 1

        # Evaluate all pairs
        for pair in self.population:
            pair.evaluate(evaluator)

        # Sort by fitness
        self.population.sort(key=lambda p: p.fitness, reverse=True)

        # Record statistics
        fitnesses = [p.fitness for p in self.population]
        self.fitness_history.append(np.mean(fitnesses))
        self.diversity_history.append(self._compute_diversity())

        # Archive best if novel
        best = self.population[0]
        if self._is_novel(best):
            self.archive.append(copy.deepcopy(best))

        # Selection and reproduction
        new_population = []

        # Keep elites
        for i in range(self.elite_size):
            elite = copy.deepcopy(self.population[i])
            elite.generation = self.generation
            new_population.append(elite)

        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            if self._rng.random() < self.crossover_rate:
                child_agent = parent1.agent_genome.architecture.crossover(
                    parent2.agent_genome.architecture, self._rng
                )
                child_genome = AgentArchitectureGenome(
                    architecture=child_agent,
                    archetype=self._rng.choice([
                        parent1.agent_genome.archetype,
                        parent2.agent_genome.archetype
                    ]),
                    profile_seed=self._rng.integers(0, 1000000),
                )
            else:
                child_genome = copy.deepcopy(parent1.agent_genome)

            # Mutate
            child_genome = child_genome.mutate(self.mutation_rate, self._rng)

            # Environment co-evolution: adapt to agent's capabilities
            child_env = self._coevolve_environment(child_genome, parent1.env_genome)

            new_population.append(CoevolutionPair(
                agent_genome=child_genome,
                env_genome=child_env,
                generation=self.generation,
            ))

        self.population = new_population

        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'diversity': self.diversity_history[-1],
            'archive_size': len(self.archive),
            'best_benchmark_scores': self.population[0].benchmark_scores,
        }

    def _tournament_select(self, tournament_size: int = 3) -> CoevolutionPair:
        """Tournament selection"""
        candidates = self._rng.choice(self.population, size=tournament_size, replace=False)
        return max(candidates, key=lambda p: p.fitness)

    def _coevolve_environment(self, agent_genome: AgentArchitectureGenome,
                             parent_env: EnvironmentGene) -> EnvironmentGene:
        """
        Co-evolve environment based on agent capabilities.
        Key POET insight: environment adapts to challenge agents.
        """
        env = parent_env.mutate(self.mutation_rate, self._rng)

        # Increase difficulty if agent has high ToM
        if agent_genome.architecture.tom_depth >= 3:
            env.difficulty = min(
                self.max_difficulty,
                env.difficulty + self.difficulty_increase_rate
            )
            env.deception_enabled = True

        # Adjust ToM pressure to match agent capability (slightly above)
        target_tom = agent_genome.architecture.tom_depth + 1
        if target_tom <= 2:
            env.deception_enabled = False
            env.coalition_enabled = False
        elif target_tom <= 4:
            env.coalition_enabled = True
        else:
            env.deception_enabled = True
            env.benchmark_rate = min(0.3, env.benchmark_rate + 0.05)

        return env

    def _compute_diversity(self) -> float:
        """Compute population diversity"""
        if len(self.population) < 2:
            return 0.0

        # Measure diversity in architecture parameters
        tom_depths = [p.agent_genome.architecture.tom_depth for p in self.population]
        hidden_dims = [p.agent_genome.architecture.hidden_dim for p in self.population]
        difficulties = [p.env_genome.difficulty for p in self.population]

        diversity = (
            np.std(tom_depths) / 2.5 +
            np.std(hidden_dims) / 200 +
            np.std(difficulties)
        ) / 3

        return diversity

    def _is_novel(self, pair: CoevolutionPair, threshold: float = 0.1) -> bool:
        """Check if pair is sufficiently novel for archive"""
        if not self.archive:
            return True

        for archived in self.archive:
            # Compare architecture
            arch_diff = abs(
                pair.agent_genome.architecture.tom_depth -
                archived.agent_genome.architecture.tom_depth
            )
            arch_diff += abs(
                pair.agent_genome.architecture.hidden_dim -
                archived.agent_genome.architecture.hidden_dim
            ) / 512

            # Compare environment
            env_diff = abs(
                pair.env_genome.difficulty -
                archived.env_genome.difficulty
            )

            if arch_diff + env_diff < threshold:
                return False

        return True

    def get_best_agent(self) -> AgentArchitectureGenome:
        """Get the best agent genome from current population"""
        if not self.population:
            return AgentArchitectureGenome()
        return max(self.population, key=lambda p: p.fitness).agent_genome

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'archive_size': len(self.archive),
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'best_fitness': max(p.fitness for p in self.population) if self.population else 0,
            'mean_fitness': np.mean([p.fitness for p in self.population]) if self.population else 0,
            'best_tom_depth': max(
                p.agent_genome.architecture.tom_depth for p in self.population
            ) if self.population else 0,
            'mean_difficulty': np.mean(
                [p.env_genome.difficulty for p in self.population]
            ) if self.population else 0,
        }

    def describe(self) -> str:
        """Generate human-readable description"""
        stats = self.get_statistics()
        lines = [
            "=== POET Co-Evolution Engine ===",
            f"Generation: {stats['generation']}",
            f"Population: {stats['population_size']}",
            f"Archive: {stats['archive_size']}",
            "",
            "--- Current Best ---",
        ]

        if self.population:
            best = max(self.population, key=lambda p: p.fitness)
            lines.extend([
                f"Fitness: {best.fitness:.3f}",
                f"Architecture: {best.agent_genome.architecture.arch_type.value}",
                f"ToM Depth: {best.agent_genome.architecture.tom_depth}",
                f"Hidden Dim: {best.agent_genome.architecture.hidden_dim}",
                f"Env Difficulty: {best.env_genome.difficulty:.2f}",
                "",
                "Benchmark Scores:",
            ])
            for k, v in best.benchmark_scores.items():
                lines.append(f"  {k}: {v:.3f}")

        return '\n'.join(lines)
