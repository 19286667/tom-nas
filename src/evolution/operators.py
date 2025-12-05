"""
Evolutionary Operators for ToM-NAS
Mutation and crossover for neural architectures
"""
import torch
import torch.nn as nn
import copy
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class ArchitectureGene:
    """Represents genetic encoding of architecture"""

    def __init__(self):
        self.gene_dict = {
            # Architecture type
            'arch_type': 'TRN',  # TRN, RSAN, Transformer, or Hybrid

            # Layer configuration
            'num_layers': 2,
            'hidden_dim': 128,
            'num_heads': 4,  # For attention-based models
            'max_recursion': 5,  # For RSAN

            # Component toggles
            'use_layer_norm': True,
            'use_dropout': True,
            'dropout_rate': 0.1,

            # Gating mechanisms (for TRN)
            'use_update_gate': True,
            'use_reset_gate': True,

            # Output configuration
            'belief_head_layers': 1,
            'action_head_layers': 1,

            # Training parameters
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
        }

    def mutate(self, mutation_rate: float = 0.1) -> 'ArchitectureGene':
        """Create mutated copy of gene"""
        new_gene = copy.deepcopy(self)

        for key, value in new_gene.gene_dict.items():
            if random.random() < mutation_rate:
                new_gene.gene_dict[key] = self._mutate_gene(key, value)

        return new_gene

    def _mutate_gene(self, key: str, value):
        """Mutate individual gene"""
        if key == 'arch_type':
            return random.choice(['TRN', 'RSAN', 'Transformer', 'Hybrid'])

        elif key in ['num_layers', 'belief_head_layers', 'action_head_layers']:
            delta = random.choice([-1, 0, 1])
            return max(1, min(5, value + delta))

        elif key == 'hidden_dim':
            multiplier = random.choice([0.5, 1.0, 1.5, 2.0])
            return int(max(64, min(512, value * multiplier)))

        elif key == 'num_heads':
            return random.choice([2, 4, 8, 16])

        elif key == 'max_recursion':
            return random.randint(3, 7)

        elif key in ['use_layer_norm', 'use_dropout', 'use_update_gate', 'use_reset_gate']:
            return random.choice([True, False])

        elif key == 'dropout_rate':
            return random.uniform(0.0, 0.5)

        elif key == 'learning_rate':
            return random.uniform(0.0001, 0.01)

        elif key == 'weight_decay':
            return random.uniform(0.0, 0.001)

        return value

    def crossover(self, other: 'ArchitectureGene') -> Tuple['ArchitectureGene', 'ArchitectureGene']:
        """Crossover with another gene"""
        child1 = ArchitectureGene()
        child2 = ArchitectureGene()

        for key in self.gene_dict.keys():
            if random.random() < 0.5:
                child1.gene_dict[key] = self.gene_dict[key]
                child2.gene_dict[key] = other.gene_dict[key]
            else:
                child1.gene_dict[key] = other.gene_dict[key]
                child2.gene_dict[key] = self.gene_dict[key]

        return child1, child2


class WeightMutation:
    """Mutate network weights directly"""

    @staticmethod
    def gaussian_noise(model: nn.Module, noise_std: float = 0.01) -> nn.Module:
        """Add gaussian noise to weights"""
        mutated = copy.deepcopy(model)
        with torch.no_grad():
            for param in mutated.parameters():
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)
        return mutated

    @staticmethod
    def random_reset(model: nn.Module, reset_prob: float = 0.1) -> nn.Module:
        """Randomly reset some weights"""
        mutated = copy.deepcopy(model)
        with torch.no_grad():
            for param in mutated.parameters():
                mask = torch.rand_like(param) < reset_prob
                if mask.any():
                    param[mask] = torch.randn_like(param[mask]) * 0.02
        return mutated

    @staticmethod
    def layer_shuffle(model: nn.Module) -> nn.Module:
        """Shuffle some layers (experimental)"""
        mutated = copy.deepcopy(model)
        # This is a placeholder - full implementation would identify
        # and shuffle compatible layers
        return mutated


class ArchitectureCrossover:
    """Crossover operations for network architectures"""

    @staticmethod
    def weight_averaging(parent1: nn.Module, parent2: nn.Module,
                        alpha: float = 0.5) -> nn.Module:
        """Average weights of two networks"""
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for p1, p2, pc in zip(parent1.parameters(),
                                 parent2.parameters(),
                                 child.parameters()):
                if p1.shape == p2.shape:
                    pc.data = alpha * p1.data + (1 - alpha) * p2.data
        return child

    @staticmethod
    def layer_swap(parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        """Swap random layers between parents"""
        # Placeholder for layer swapping
        # Full implementation would identify compatible layers
        # and swap them between networks
        child = copy.deepcopy(parent1)
        return child


class PopulationOperators:
    """High-level operators for managing populations"""

    @staticmethod
    def tournament_selection(population: List[Tuple[nn.Module, float]],
                           tournament_size: int = 3) -> nn.Module:
        """Select individual via tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        winner = max(tournament, key=lambda x: x[1])  # x[1] is fitness
        return copy.deepcopy(winner[0])

    @staticmethod
    def elitism_selection(population: List[Tuple[nn.Module, float]],
                         elite_size: int = 2) -> List[nn.Module]:
        """Select top performers"""
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
        return [copy.deepcopy(ind[0]) for ind in sorted_pop[:elite_size]]

    @staticmethod
    def fitness_proportional_selection(population: List[Tuple[nn.Module, float]]) -> nn.Module:
        """Roulette wheel selection based on fitness"""
        fitnesses = [ind[1] for ind in population]
        total_fitness = sum(max(0, f) for f in fitnesses)

        if total_fitness == 0:
            return copy.deepcopy(random.choice(population)[0])

        # Normalize to probabilities
        probabilities = [max(0, f) / total_fitness for f in fitnesses]

        selected_idx = np.random.choice(len(population), p=probabilities)
        return copy.deepcopy(population[selected_idx][0])


class AdaptiveMutation:
    """Adaptive mutation rates based on population diversity"""

    def __init__(self, initial_rate: float = 0.1):
        self.base_rate = initial_rate
        self.current_rate = initial_rate
        self.diversity_history = []

    def update_rate(self, population_diversity: float):
        """Adjust mutation rate based on diversity"""
        self.diversity_history.append(population_diversity)

        # Increase mutation if diversity is low
        if population_diversity < 0.3:
            self.current_rate = min(0.5, self.base_rate * 1.5)
        elif population_diversity > 0.7:
            self.current_rate = max(0.01, self.base_rate * 0.5)
        else:
            self.current_rate = self.base_rate

    def get_rate(self) -> float:
        return self.current_rate


class SpeciesManager:
    """Manage species/niches for diversity preservation.

    Species are defined by architecture type (TRN, RSAN, Transformer, Hybrid).
    This ensures true coevolution between different architecture families.
    """

    def __init__(self, compatibility_threshold: float = 0.3):
        self.compatibility_threshold = compatibility_threshold
        # Species organized by architecture type
        self.species: Dict[str, List] = {
            'TRN': [],
            'RSAN': [],
            'Transformer': [],
            'Hybrid': []
        }

    def speciate(self, population: List[Tuple[nn.Module, ArchitectureGene, float]]):
        """Divide population into species BY ARCHITECTURE TYPE.

        This is the key coevolutionary mechanism - different architectures
        compete within their own species and cooperate across species.
        """
        # Clear existing species
        for arch_type in self.species:
            self.species[arch_type] = []

        # Assign each individual to species by arch_type
        for individual, gene, fitness in population:
            arch_type = gene.gene_dict.get('arch_type', 'TRN')
            if arch_type not in self.species:
                self.species[arch_type] = []
            self.species[arch_type].append((individual, gene, fitness))

    def _genes_compatible(self, gene1: ArchitectureGene,
                         gene2: ArchitectureGene) -> bool:
        """Check if two genes are compatible (same architecture type)."""
        return gene1.gene_dict.get('arch_type') == gene2.gene_dict.get('arch_type')

    def get_species_count(self) -> int:
        """Return count of non-empty species."""
        return sum(1 for pop in self.species.values() if len(pop) > 0)

    def get_species_sizes(self) -> List[int]:
        """Return size of each non-empty species."""
        return [len(pop) for pop in self.species.values() if len(pop) > 0]

    def get_species_stats(self) -> Dict[str, Dict]:
        """Return detailed stats per species."""
        stats = {}
        for arch_type, population in self.species.items():
            if population:
                fitnesses = [f for _, _, f in population if f is not None]
                stats[arch_type] = {
                    'count': len(population),
                    'best_fitness': max(fitnesses) if fitnesses else 0.0,
                    'avg_fitness': np.mean(fitnesses) if fitnesses else 0.0
                }
        return stats


class CoevolutionOperator:
    """Manages coevolution between architectures, tasks, and evaluation.

    Implements curriculum learning for ToM tasks:
    - Starts with simple 1st-order ToM tasks
    - Gradually introduces higher-order reasoning
    - Adapts zombie detection difficulty
    - Manages task diversity for robust learning
    """

    def __init__(self):
        self.task_difficulty = 1.0
        self.evaluation_strictness = 1.0
        self.generation = 0

        # Task curriculum state
        self.current_tom_order = 1  # Start with 1st-order ToM
        self.max_tom_order = 5
        self.zombie_detection_active = False
        self.multi_agent_active = False

        # Task weights (evolve over time)
        self.task_weights = {
            'sally_anne_basic': 1.0,
            'sally_anne_advanced': 0.0,
            'higher_order_tom': 0.0,
            'zombie_detection': 0.0,
            'cooperation': 0.5,
            'communication': 0.3,
            'resource_sharing': 0.2,
        }

        # Performance history for adaptive curriculum
        self.performance_history = []
        self.task_mastery = defaultdict(list)

        # Evaluation parameters
        self.episode_length_base = 20
        self.num_episodes_base = 3

    def adapt_tasks(self, population_performance: List[float]):
        """Adapt task curriculum based on population performance"""
        avg_performance = np.mean(population_performance)
        self.performance_history.append(avg_performance)
        self.generation += 1

        # Determine if population has mastered current tasks
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-3:])

            # Curriculum progression logic
            if recent_avg > 0.75:
                self._advance_curriculum()
            elif recent_avg < 0.3:
                self._simplify_curriculum()

        # Adjust task difficulty multiplier
        if avg_performance > 0.8:
            self.task_difficulty = min(2.0, self.task_difficulty * 1.1)
        elif avg_performance < 0.4:
            self.task_difficulty = max(0.5, self.task_difficulty * 0.9)

    def _advance_curriculum(self):
        """Progress to more challenging tasks"""
        # Increase ToM order if not at max
        if self.current_tom_order < self.max_tom_order:
            self.current_tom_order += 1
            self.task_weights['higher_order_tom'] = min(1.0,
                self.task_weights['higher_order_tom'] + 0.2)

        # Activate zombie detection after basic ToM mastery
        if self.current_tom_order >= 2 and not self.zombie_detection_active:
            self.zombie_detection_active = True
            self.task_weights['zombie_detection'] = 0.3

        # Activate advanced Sally-Anne tests
        if self.current_tom_order >= 3:
            self.task_weights['sally_anne_advanced'] = min(1.0,
                self.task_weights['sally_anne_advanced'] + 0.2)

        # Activate multi-agent scenarios
        if self.current_tom_order >= 4 and not self.multi_agent_active:
            self.multi_agent_active = True
            self.task_weights['cooperation'] = 1.0
            self.task_weights['communication'] = 0.8

    def _simplify_curriculum(self):
        """Reduce task difficulty if population is struggling"""
        # Don't reduce below baseline
        if self.current_tom_order > 1:
            self.current_tom_order -= 1

        # Reduce advanced task weights
        self.task_weights['higher_order_tom'] = max(0.0,
            self.task_weights['higher_order_tom'] - 0.1)
        self.task_weights['sally_anne_advanced'] = max(0.0,
            self.task_weights['sally_anne_advanced'] - 0.1)

    def adapt_evaluation(self, population_variance: float):
        """Adjust evaluation parameters based on population diversity"""
        if population_variance < 0.1:
            # Population converging - increase strictness to differentiate
            self.evaluation_strictness = min(2.0, self.evaluation_strictness * 1.1)
            # Also increase evaluation thoroughness
            self.num_episodes_base = min(10, self.num_episodes_base + 1)
        elif population_variance > 0.3:
            # Population diverse - be more lenient for exploration
            self.evaluation_strictness = max(0.5, self.evaluation_strictness * 0.9)
        else:
            # Gradually return to baseline
            self.evaluation_strictness = 0.9 * self.evaluation_strictness + 0.1 * 1.0

    def get_adjusted_fitness(self, raw_fitness: float) -> float:
        """Apply coevolutionary adjustments to fitness"""
        return raw_fitness * self.evaluation_strictness / self.task_difficulty

    def get_task_config(self) -> Dict:
        """Get current task configuration for fitness evaluation"""
        return {
            'tom_order': self.current_tom_order,
            'task_weights': self.task_weights.copy(),
            'zombie_detection_active': self.zombie_detection_active,
            'multi_agent_active': self.multi_agent_active,
            'episode_length': int(self.episode_length_base * self.task_difficulty),
            'num_episodes': self.num_episodes_base,
            'difficulty': self.task_difficulty,
            'strictness': self.evaluation_strictness,
        }

    def generate_proxy_task(self) -> Dict:
        """Generate a proxy task for fast evaluation (subset of full evaluation)"""
        # Proxy tasks are simplified versions for quick fitness approximation
        return {
            'type': 'proxy',
            'tom_order': min(self.current_tom_order, 2),  # Max 2nd order for speed
            'episode_length': max(5, self.episode_length_base // 2),
            'num_episodes': 1,
            'tasks': ['sally_anne_basic', 'cooperation'],
        }

    def record_task_performance(self, task_name: str, scores: List[float]):
        """Record performance on specific tasks for curriculum tracking"""
        self.task_mastery[task_name].extend(scores)
        # Keep only recent history
        if len(self.task_mastery[task_name]) > 100:
            self.task_mastery[task_name] = self.task_mastery[task_name][-100:]

    def get_curriculum_summary(self) -> Dict:
        """Get summary of current curriculum state"""
        return {
            'generation': self.generation,
            'tom_order': self.current_tom_order,
            'task_difficulty': self.task_difficulty,
            'evaluation_strictness': self.evaluation_strictness,
            'zombie_detection_active': self.zombie_detection_active,
            'multi_agent_active': self.multi_agent_active,
            'task_weights': self.task_weights.copy(),
            'recent_performance': np.mean(self.performance_history[-5:]) if self.performance_history else 0.0,
        }
