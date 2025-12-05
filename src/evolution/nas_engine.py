"""
Neural Architecture Search Engine for ToM-NAS
Main evolutionary algorithm coordinating architecture evolution
"""
import torch
import torch.nn as nn
import copy
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

from .operators import (
    ArchitectureGene, WeightMutation, ArchitectureCrossover,
    PopulationOperators, AdaptiveMutation, SpeciesManager, CoevolutionOperator
)
from .fitness import CompositeFitnessFunction
from ..agents.architectures import (
    TransparentRNN, RecursiveSelfAttention, TransformerToMAgent, HybridArchitecture
)


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary process"""
    population_size: int = 20
    num_generations: int = 100
    elite_size: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    weight_mutation_prob: float = 0.3
    use_speciation: bool = True
    use_coevolution: bool = True
    fitness_episodes: int = 5
    device: str = 'cpu'
    input_dim: int = 191
    output_dim: int = 181
    checkpoint_interval: int = 10
    # Convergence detection parameters
    convergence_window: int = 10  # Generations to check for improvement
    convergence_threshold: float = 0.001  # Minimum improvement to continue
    enable_early_stopping: bool = True  # Whether to use convergence-based stopping


class Individual:
    """Represents one individual in the population"""

    def __init__(self, model: nn.Module, gene: ArchitectureGene,
                 fitness: Optional[float] = None, generation: int = 0):
        self.model = model
        self.gene = gene
        self.fitness = fitness
        self.generation = generation
        self.age = 0
        self.parent_ids = []

    def __repr__(self):
        return (f"Individual(arch={self.gene.gene_dict['arch_type']}, "
                f"fitness={self.fitness:.4f if self.fitness else 'None'}, "
                f"gen={self.generation})")


class NASEngine:
    """Main Neural Architecture Search Engine"""

    def __init__(self, config: EvolutionConfig, world, belief_network):
        self.config = config
        self.world = world
        self.belief_network = belief_network

        # Fitness evaluator
        self.fitness_fn = CompositeFitnessFunction(
            world, belief_network, config.device
        )

        # Evolution operators
        self.population_ops = PopulationOperators()
        self.adaptive_mutation = AdaptiveMutation(config.mutation_rate)
        self.species_manager = SpeciesManager() if config.use_speciation else None
        self.coevolution = CoevolutionOperator() if config.use_coevolution else None

        # Population
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None

        # History tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'species_count': [],
            'best_genes': []
        }

        # Convergence tracking
        self._stagnant_generations = 0
        self._converged = False

    def initialize_population(self):
        """Create initial population with diverse architectures"""
        print(f"Initializing population of {self.config.population_size} individuals...")

        for i in range(self.config.population_size):
            gene = self._create_random_gene()
            model = self._gene_to_model(gene)
            individual = Individual(model, gene, generation=0)
            self.population.append(individual)

        print(f"  Created {len(self.population)} individuals")

    def _create_random_gene(self) -> ArchitectureGene:
        """Create random architecture gene"""
        gene = ArchitectureGene()

        gene.gene_dict['arch_type'] = random.choice(['TRN', 'RSAN', 'Transformer'])
        gene.gene_dict['num_layers'] = random.randint(1, 4)
        gene.gene_dict['hidden_dim'] = random.choice([64, 128, 256])
        gene.gene_dict['num_heads'] = random.choice([2, 4, 8])
        gene.gene_dict['max_recursion'] = random.randint(3, 7)
        gene.gene_dict['dropout_rate'] = random.uniform(0.0, 0.3)
        gene.gene_dict['learning_rate'] = random.uniform(0.0001, 0.01)

        return gene

    def _gene_to_model(self, gene: ArchitectureGene) -> nn.Module:
        """Convert gene to actual neural network"""
        arch_type = gene.gene_dict['arch_type']
        hidden_dim = gene.gene_dict['hidden_dim']
        num_layers = gene.gene_dict['num_layers']

        if arch_type == 'TRN':
            return TransparentRNN(
                self.config.input_dim,
                hidden_dim,
                self.config.output_dim,
                num_layers=num_layers
            )
        elif arch_type == 'RSAN':
            return RecursiveSelfAttention(
                self.config.input_dim,
                hidden_dim,
                self.config.output_dim,
                num_heads=gene.gene_dict['num_heads'],
                max_recursion=gene.gene_dict['max_recursion']
            )
        elif arch_type == 'Transformer':
            return TransformerToMAgent(
                self.config.input_dim,
                hidden_dim,
                self.config.output_dim,
                num_layers=num_layers,
                num_heads=gene.gene_dict['num_heads']
            )
        elif arch_type == 'Hybrid':
            return HybridArchitecture(
                self.config.input_dim,
                hidden_dim,
                self.config.output_dim,
                architecture_genes=gene.gene_dict
            )

        # Default to TRN
        return TransparentRNN(
            self.config.input_dim, hidden_dim, self.config.output_dim
        )

    def evaluate_population(self):
        """Evaluate fitness of all individuals"""
        print(f"\nEvaluating generation {self.generation}...")

        for i, individual in enumerate(self.population):
            if individual.fitness is None:
                fitness_results = self.fitness_fn.evaluate(
                    individual.model,
                    num_episodes=self.config.fitness_episodes
                )
                individual.fitness = fitness_results['total_fitness']

                if i % 5 == 0:
                    print(f"  Evaluated {i+1}/{len(self.population)} individuals")

        # Update best individual
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
            print(f"  New best! Fitness: {self.best_individual.fitness:.4f}")

    def evolve_generation(self):
        """Evolve population for one generation"""
        print(f"\nGeneration {self.generation}")
        print("="*60)

        # Evaluate current population
        self.evaluate_population()

        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = np.mean(fitnesses)
        diversity = self._calculate_diversity()

        print(f"  Best fitness:    {max(fitnesses):.4f}")
        print(f"  Average fitness: {avg_fitness:.4f}")
        print(f"  Diversity:       {diversity:.4f}")

        # Record history
        self.history['best_fitness'].append(max(fitnesses))
        self.history['avg_fitness'].append(avg_fitness)
        self.history['diversity'].append(diversity)
        self.history['best_genes'].append(copy.deepcopy(self.best_individual.gene.gene_dict))

        # Speciation
        if self.config.use_speciation and self.species_manager:
            pop_with_genes = [
                (ind.model, ind.gene, ind.fitness) for ind in self.population
            ]
            self.species_manager.speciate(pop_with_genes)
            num_species = self.species_manager.get_species_count()
            self.history['species_count'].append(num_species)
            print(f"  Species count:   {num_species}")

        # Coevolution adaptations
        if self.config.use_coevolution and self.coevolution:
            self.coevolution.adapt_tasks(fitnesses)
            self.coevolution.adapt_evaluation(np.var(fitnesses))

        # Adaptive mutation
        self.adaptive_mutation.update_rate(diversity)
        current_mutation_rate = self.adaptive_mutation.get_rate()
        print(f"  Mutation rate:   {current_mutation_rate:.4f}")

        # Check for convergence
        if self.config.enable_early_stopping:
            self._check_convergence()

        # Create next generation
        new_population = self._create_next_generation(current_mutation_rate)
        self.population = new_population
        self.generation += 1

    def _check_convergence(self) -> None:
        """
        Check if evolution has converged based on fitness improvement.

        Convergence is detected when the best fitness hasn't improved by more
        than convergence_threshold over the last convergence_window generations.
        """
        window = self.config.convergence_window
        threshold = self.config.convergence_threshold

        if len(self.history['best_fitness']) < window:
            return

        recent_best = self.history['best_fitness'][-window:]
        improvement = max(recent_best) - min(recent_best)

        if improvement < threshold:
            self._stagnant_generations += 1
            print(f"  Stagnant generations: {self._stagnant_generations}")
        else:
            self._stagnant_generations = 0

        # Check if converged (stagnant for entire window)
        if self._stagnant_generations >= window:
            self._converged = True
            print(f"\n  *** CONVERGENCE DETECTED ***")
            print(f"  No improvement > {threshold} in last {window} generations")

    def has_converged(self) -> bool:
        """Check if evolution has converged."""
        return self._converged

    def _create_next_generation(self, mutation_rate: float) -> List[Individual]:
        """Create next generation through selection, crossover, mutation"""
        new_population = []

        # Build model-to-individual lookup map for O(1) access
        model_to_individual = {id(ind.model): ind for ind in self.population}

        # Pre-build fitness list for selection (avoid recreating each time)
        fitness_list = [(ind.model, ind.fitness) for ind in self.population]

        # Elitism - keep best individuals
        elite = self.population_ops.elitism_selection(
            fitness_list,
            self.config.elite_size
        )

        for model in elite:
            # O(1) lookup instead of O(N) linear search
            ind = model_to_individual.get(id(model))
            if ind is not None:
                new_ind = Individual(
                    copy.deepcopy(model),
                    copy.deepcopy(ind.gene),
                    ind.fitness,
                    self.generation + 1
                )
                new_population.append(new_ind)

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection (reuse fitness_list)
            parent1_model = self.population_ops.tournament_selection(
                fitness_list,
                self.config.tournament_size
            )
            parent2_model = self.population_ops.tournament_selection(
                fitness_list,
                self.config.tournament_size
            )

            # O(1) gene lookup
            parent1_ind = model_to_individual.get(id(parent1_model))
            parent2_ind = model_to_individual.get(id(parent2_model))

            parent1_gene = parent1_ind.gene if parent1_ind else self._create_random_gene()
            parent2_gene = parent2_ind.gene if parent2_ind else self._create_random_gene()

            # Crossover genes
            if random.random() < self.config.crossover_rate:
                child_gene, _ = parent1_gene.crossover(parent2_gene)
            else:
                child_gene = copy.deepcopy(parent1_gene)

            # Mutation
            if random.random() < mutation_rate:
                child_gene = child_gene.mutate(mutation_rate)

            # Create model from gene
            child_model = self._gene_to_model(child_gene)

            # Weight mutation
            if random.random() < self.config.weight_mutation_prob:
                child_model = WeightMutation.gaussian_noise(child_model, noise_std=0.01)

            child = Individual(child_model, child_gene, generation=self.generation + 1)
            new_population.append(child)

        return new_population

    def _calculate_diversity(self) -> float:
        """
        Calculate population genetic diversity using random sampling.

        Uses O(N) sampling instead of O(N²) pairwise comparison for efficiency.
        For populations < 20, uses full pairwise (still fast).
        For larger populations, samples sqrt(N*(N-1)/2) pairs.
        """
        n = len(self.population)
        if n < 2:
            return 0.0

        # For small populations, use full pairwise (still O(N²) but N is small)
        max_full_pairwise = 20
        if n <= max_full_pairwise:
            total_distance = 0.0
            comparisons = 0
            for i in range(n):
                for j in range(i + 1, n):
                    distance = self._gene_distance(
                        self.population[i].gene,
                        self.population[j].gene
                    )
                    total_distance += distance
                    comparisons += 1
            return total_distance / comparisons if comparisons > 0 else 0.0

        # For larger populations, use random sampling
        # Sample approximately sqrt(N*(N-1)/2) pairs for good estimate
        num_pairs = n * (n - 1) // 2
        num_samples = min(int(np.sqrt(num_pairs) * 2), 50)  # Cap at 50 samples

        total_distance = 0.0
        for _ in range(num_samples):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)

            distance = self._gene_distance(
                self.population[i].gene,
                self.population[j].gene
            )
            total_distance += distance

        return total_distance / num_samples

    def _gene_distance(self, gene1: ArchitectureGene, gene2: ArchitectureGene) -> float:
        """Calculate distance between two genes"""
        differences = 0
        total = 0

        for key in gene1.gene_dict.keys():
            val1 = gene1.gene_dict[key]
            val2 = gene2.gene_dict[key]
            total += 1

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2), 1.0)
                differences += abs(val1 - val2) / max_val
            elif val1 != val2:
                differences += 1

        return differences / total if total > 0 else 0.0

    def run(self, num_generations: Optional[int] = None):
        """Run evolution for specified number of generations"""
        if num_generations is None:
            num_generations = self.config.num_generations

        print("\n" + "="*60)
        print("ToM-NAS Evolution Starting")
        print("="*60)
        print(f"Population size: {self.config.population_size}")
        print(f"Generations:     {num_generations}")
        print(f"Elite size:      {self.config.elite_size}")
        print(f"Mutation rate:   {self.config.mutation_rate}")
        print(f"Crossover rate:  {self.config.crossover_rate}")
        print("="*60)

        # Initialize if needed
        if not self.population:
            self.initialize_population()

        # Evolution loop
        for gen in range(num_generations):
            self.evolve_generation()

            # Checkpoint
            if (gen + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_gen_{gen+1}.pt")

            # Early stopping on convergence
            if self.config.enable_early_stopping and self.has_converged():
                print(f"\n  Early stopping at generation {self.generation}")
                print(f"  Evolution converged - no significant improvement detected")
                break

        print("\n" + "="*60)
        print("Evolution Complete!")
        print("="*60)
        print(f"Best fitness achieved: {self.best_individual.fitness:.4f}")
        print(f"Best architecture: {self.best_individual.gene.gene_dict['arch_type']}")
        print("="*60)

        return self.best_individual

    def save_checkpoint(self, filepath: str):
        """Save evolution checkpoint"""
        checkpoint = {
            'generation': self.generation,
            'best_individual': {
                'model_state': self.best_individual.model.state_dict(),
                'gene': self.best_individual.gene.gene_dict,
                'fitness': self.best_individual.fitness
            },
            'population': [
                {
                    'gene': ind.gene.gene_dict,
                    'fitness': ind.fitness,
                    'generation': ind.generation
                }
                for ind in self.population
            ],
            'history': self.history,
            'config': {
                'population_size': self.config.population_size,
                'num_generations': self.config.num_generations,
                'elite_size': self.config.elite_size,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate
            }
        }

        torch.save(checkpoint, filepath)
        print(f"  Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load evolution checkpoint safely."""
        # Use weights_only=True for security (prevents arbitrary code execution)
        # map_location ensures compatibility across devices
        checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)
        # Note: weights_only=False is required here because we save full checkpoint dicts
        # In a production environment, consider using safetensors or separate files

        self.generation = checkpoint['generation']

        # Restore best individual
        best_gene = ArchitectureGene()
        best_gene.gene_dict = checkpoint['best_individual']['gene']
        best_model = self._gene_to_model(best_gene)
        best_model.load_state_dict(checkpoint['best_individual']['model_state'])

        self.best_individual = Individual(
            best_model,
            best_gene,
            checkpoint['best_individual']['fitness'],
            self.generation
        )

        self.history = checkpoint['history']

        print(f"Checkpoint loaded from generation {self.generation}")
        print(f"Best fitness: {self.best_individual.fitness:.4f}")

    def get_best_model(self) -> nn.Module:
        """Get the best evolved model"""
        return self.best_individual.model if self.best_individual else None

    def get_evolution_summary(self) -> Dict:
        """Get summary of evolution process"""
        return {
            'total_generations': self.generation,
            'best_fitness': self.best_individual.fitness if self.best_individual else 0.0,
            'best_architecture': self.best_individual.gene.gene_dict if self.best_individual else {},
            'fitness_history': self.history['best_fitness'],
            'avg_fitness_history': self.history['avg_fitness'],
            'diversity_history': self.history['diversity'],
            'final_population_size': len(self.population)
        }
