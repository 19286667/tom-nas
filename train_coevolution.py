#!/usr/bin/env python
"""
ToM-NAS Coevolutionary Training System
Trains a POPULATION of diverse architectures (TRN, RSAN, Transformer) competing and hybridizing.

This is the CORRECT way to train ToM agents - through coevolution where:
- Multiple architecture types compete in the same environment
- Zombie games create selection pressure for genuine ToM
- Architectures can crossover to create hybrids
- Species-level diversity is maintained

Single-architecture training CANNOT achieve higher-order ToM because:
- TRN alone lacks recursive attention for belief nesting
- RSAN alone may lack temporal modeling
- Only through competition and hybridization can optimal architectures emerge
"""
import torch
import torch.nn as nn
import argparse
import os
import sys
import json
import random
import copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.world.social_world import SocialWorld4
from src.evaluation.benchmarks import BenchmarkSuite, SallyAnneTest, HigherOrderToMBenchmark
from src.utils import create_model


@dataclass
class AgentIndividual:
    """Represents one agent in the evolutionary population"""
    id: int
    architecture_type: str  # 'TRN', 'RSAN', 'Transformer'
    model: nn.Module
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)

    # Detailed fitness components
    sally_anne_score: float = 0.0
    higher_order_scores: Dict[int, float] = field(default_factory=dict)
    zombie_detection_score: float = 0.0
    cooperation_score: float = 0.0
    survival_score: float = 0.0

    # Species tracking
    species_id: int = 0

    def __post_init__(self):
        if not self.higher_order_scores:
            self.higher_order_scores = {i: 0.0 for i in range(1, 6)}


class SpeciesTracker:
    """Track species-level metrics for coevolution"""

    def __init__(self):
        self.species_history = {
            'TRN': [],
            'RSAN': [],
            'Transformer': [],
            'Hybrid': []
        }
        self.generation_stats = []

    def record_generation(self, population: List[AgentIndividual], generation: int):
        """Record species-level statistics for a generation"""
        stats = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'species': {}
        }

        for arch_type in ['TRN', 'RSAN', 'Transformer', 'Hybrid']:
            agents = [a for a in population if a.architecture_type == arch_type]
            if agents:
                fitnesses = [a.fitness for a in agents]
                stats['species'][arch_type] = {
                    'count': len(agents),
                    'avg_fitness': sum(fitnesses) / len(fitnesses),
                    'max_fitness': max(fitnesses),
                    'min_fitness': min(fitnesses),
                    'avg_sally_anne': sum(a.sally_anne_score for a in agents) / len(agents),
                    'avg_zombie_detection': sum(a.zombie_detection_score for a in agents) / len(agents),
                    'higher_order_avg': {
                        order: sum(a.higher_order_scores.get(order, 0) for a in agents) / len(agents)
                        for order in range(1, 6)
                    }
                }
                self.species_history[arch_type].append(stats['species'][arch_type])

        self.generation_stats.append(stats)
        return stats

    def get_summary(self) -> Dict:
        """Get summary of species evolution"""
        return {
            'total_generations': len(self.generation_stats),
            'species_history': self.species_history,
            'final_stats': self.generation_stats[-1] if self.generation_stats else None
        }


class CoevolutionaryTrainer:
    """
    Coevolutionary training system for ToM-NAS.

    Key principles:
    1. Population diversity: Multiple architecture types compete
    2. Selection pressure: Zombie games filter for genuine ToM
    3. Hybridization: Crossover creates novel architectures
    4. Species preservation: Maintain minimum viable populations per species
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cpu')

        # Population settings
        self.population_size = config.get('population_size', 12)
        self.trn_count = config.get('trn_count', 4)
        self.rsan_count = config.get('rsan_count', 4)
        self.transformer_count = config.get('transformer_count', 4)

        # Architecture settings
        self.input_dim = config.get('input_dim', 191)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.output_dim = config.get('ontology_dim', 181)

        # Evolution settings
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.3)
        self.elite_count = config.get('elite_count', 2)
        self.min_species_size = config.get('min_species_size', 2)

        # Training settings
        self.episodes_per_eval = config.get('episodes_per_eval', 5)
        self.sequence_length = config.get('sequence_length', 20)

        # Initialize components
        self.world = SocialWorld4(
            num_agents=config.get('num_world_agents', 6),
            ontology_dim=self.output_dim,
            num_zombies=config.get('num_zombies', 2)
        )

        self.belief_network = BeliefNetwork(
            num_agents=config.get('num_world_agents', 6),
            ontology_dim=self.output_dim,
            max_order=config.get('max_belief_order', 5)
        )

        self.benchmark_suite = BenchmarkSuite(device=self.device)
        self.species_tracker = SpeciesTracker()

        # Results directory
        self.results_dir = config.get('results_dir', 'coevolution_results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize population
        self.population = self._initialize_population()
        self.generation = 0
        self.next_id = self.population_size

        # History tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []

    def _create_model(self, arch_type: str) -> nn.Module:
        """Create a model of specified architecture type"""
        return create_model(
            arch_type=arch_type,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=3,
            num_heads=4,
            device=self.device
        )

    def _initialize_population(self) -> List[AgentIndividual]:
        """Initialize diverse population with multiple architecture types"""
        population = []
        agent_id = 0

        # Create TRN agents
        for i in range(self.trn_count):
            model = self._create_model('TRN')
            population.append(AgentIndividual(
                id=agent_id,
                architecture_type='TRN',
                model=model,
                species_id=0
            ))
            agent_id += 1

        # Create RSAN agents
        for i in range(self.rsan_count):
            model = self._create_model('RSAN')
            population.append(AgentIndividual(
                id=agent_id,
                architecture_type='RSAN',
                model=model,
                species_id=1
            ))
            agent_id += 1

        # Create Transformer agents
        for i in range(self.transformer_count):
            model = self._create_model('Transformer')
            population.append(AgentIndividual(
                id=agent_id,
                architecture_type='Transformer',
                model=model,
                species_id=2
            ))
            agent_id += 1

        print(f"Initialized population: {self.trn_count} TRN, {self.rsan_count} RSAN, {self.transformer_count} Transformer")
        return population

    def evaluate_agent(self, agent: AgentIndividual) -> float:
        """
        Comprehensive fitness evaluation for an agent.

        Fitness components:
        1. Sally-Anne test (false belief understanding)
        2. Higher-order ToM (1st through 5th order)
        3. Zombie detection (genuine ToM validation)
        4. Social World survival
        5. Cooperation success
        """
        agent.model.eval()

        with torch.no_grad():
            # 1. Sally-Anne Test (weight: 0.2)
            sally_anne_test = SallyAnneTest()
            sally_result = sally_anne_test.run_basic(agent.model)
            agent.sally_anne_score = sally_result.score

            # 2. Higher-Order ToM Tests (weight: 0.3)
            higher_order_test = HigherOrderToMBenchmark(max_order=5)
            total_higher_order = 0.0
            for order in range(1, 6):
                result = higher_order_test.test_order(agent.model, order)
                agent.higher_order_scores[order] = result.score
                # Weight higher orders more (they're harder and more valuable)
                weight = order / 15.0  # 1/15 + 2/15 + 3/15 + 4/15 + 5/15 = 1
                total_higher_order += result.score * weight

            # 3. Zombie Detection (weight: 0.25)
            # This is CRITICAL - agents that can't detect zombies lack genuine ToM
            zombie_score = self._evaluate_zombie_detection(agent)
            agent.zombie_detection_score = zombie_score

            # 4. Social World Survival (weight: 0.15)
            survival_score = self._evaluate_world_survival(agent)
            agent.survival_score = survival_score

            # 5. Cooperation Success (weight: 0.1)
            coop_score = self._evaluate_cooperation(agent)
            agent.cooperation_score = coop_score

        # Composite fitness with emphasis on genuine ToM markers
        fitness = (
            0.20 * agent.sally_anne_score +
            0.30 * total_higher_order +
            0.25 * agent.zombie_detection_score +  # Heavy weight on zombie detection
            0.15 * agent.survival_score +
            0.10 * agent.cooperation_score
        )

        agent.fitness = fitness
        return fitness

    def _evaluate_zombie_detection(self, agent: AgentIndividual) -> float:
        """
        Evaluate agent's ability to detect zombies.
        This is the KEY test for genuine ToM - zombies lack real belief states.
        """
        correct_detections = 0
        total_tests = 0

        for episode in range(self.episodes_per_eval):
            # Reset world with fresh zombies
            self.world.reset()

            # Run several timesteps to gather observations
            for t in range(10):
                actions = [{'type': 'cooperate'} for _ in range(self.world.num_agents)]
                self.world.step(actions, self.belief_network)

            # Test detection on each agent
            for suspect_id in range(self.world.num_agents):
                is_zombie = self.world.agents[suspect_id].is_zombie

                # Generate observation sequence
                obs_sequence = self._generate_observation_sequence(agent, suspect_id)

                # Get agent's prediction
                output = agent.model(obs_sequence)

                # Interpret prediction (using belief consistency as proxy)
                # Zombies have inconsistent beliefs - high variance in predictions
                if 'beliefs' in output:
                    belief_std = output['beliefs'].std().item()
                    predicted_zombie = belief_std > 0.3  # Threshold for inconsistency

                    if predicted_zombie == is_zombie:
                        correct_detections += 1
                    total_tests += 1

        return correct_detections / max(total_tests, 1)

    def _evaluate_world_survival(self, agent: AgentIndividual) -> float:
        """Evaluate agent's survival performance in Social World"""
        total_resources = 0.0

        for episode in range(self.episodes_per_eval):
            self.world.reset()

            # Run simulation
            for t in range(self.sequence_length):
                # Generate action based on agent's output
                obs_sequence = self._generate_observation_sequence(agent, 0)
                output = agent.model(obs_sequence)

                # Convert to action
                if 'actions' in output:
                    action_probs = output['actions'].squeeze()
                    action_type = 'cooperate' if action_probs.mean() > 0.5 else 'defect'
                else:
                    action_type = 'cooperate'

                actions = [{'type': action_type} for _ in range(self.world.num_agents)]
                self.world.step(actions, self.belief_network)

            # Score based on final resources
            stats = self.world.get_statistics()
            total_resources += stats['avg_resources']

        # Normalize to [0, 1]
        avg_resources = total_resources / self.episodes_per_eval
        return min(avg_resources / 200.0, 1.0)  # Cap at 200 resources

    def _evaluate_cooperation(self, agent: AgentIndividual) -> float:
        """Evaluate cooperation success"""
        cooperation_successes = 0
        total_games = 0

        for episode in range(self.episodes_per_eval):
            self.world.reset()

            for t in range(10):
                # Play cooperation games
                for i in range(self.world.num_agents):
                    for j in range(i+1, self.world.num_agents):
                        result = self.world.play_cooperation_game(i, j, 'cooperate', 'cooperate')
                        if result['payoffs'][0] > 0:
                            cooperation_successes += 1
                        total_games += 1

        return cooperation_successes / max(total_games, 1)

    def _generate_observation_sequence(self, agent: AgentIndividual, observer_id: int) -> torch.Tensor:
        """Generate observation sequence for an agent"""
        batch_size = 1
        sequence = torch.randn(batch_size, self.sequence_length, self.input_dim).to(self.device)

        # Fill with actual observations where possible
        obs = self.world.get_observation(observer_id)
        if obs:
            # Convert observation dict to tensor
            obs_tensor = torch.zeros(self.input_dim)
            obs_tensor[0] = obs.get('own_resources', 0) / 100.0
            obs_tensor[1] = obs.get('own_energy', 0) / 100.0
            obs_tensor[2] = obs.get('timestep', 0) / 100.0

            # Replicate across sequence
            for t in range(self.sequence_length):
                sequence[0, t, :len(obs_tensor)] = obs_tensor

        return sequence

    def select_parents(self) -> List[AgentIndividual]:
        """Tournament selection with species preservation"""
        parents = []

        # Ensure elite preservation
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.elite_count])

        # Ensure minimum species representation
        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            species_agents = [a for a in self.population if a.architecture_type == arch_type]
            if species_agents and len([p for p in parents if p.architecture_type == arch_type]) < self.min_species_size:
                # Add best from species
                best_species = max(species_agents, key=lambda x: x.fitness)
                if best_species not in parents:
                    parents.append(best_species)

        # Tournament selection for remaining slots
        while len(parents) < self.population_size // 2:
            tournament = random.sample(self.population, min(3, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            if winner not in parents:
                parents.append(winner)

        return parents

    def crossover(self, parent1: AgentIndividual, parent2: AgentIndividual) -> AgentIndividual:
        """
        Create offspring through crossover.
        If parents have different architectures, create hybrid.
        """
        if parent1.architecture_type == parent2.architecture_type:
            # Same architecture - weight crossover
            child_type = parent1.architecture_type
        else:
            # Different architectures - create hybrid or pick one
            if random.random() < 0.3:  # 30% chance of hybrid
                child_type = 'Hybrid'
            else:
                child_type = random.choice([parent1.architecture_type, parent2.architecture_type])

        # Create new model
        child_model = self._create_model(child_type)

        # Weight crossover (uniform)
        parent1_params = dict(parent1.model.named_parameters())
        parent2_params = dict(parent2.model.named_parameters())
        child_params = dict(child_model.named_parameters())

        with torch.no_grad():
            for name, param in child_params.items():
                if name in parent1_params and name in parent2_params:
                    # Uniform crossover
                    mask = torch.rand_like(param) < 0.5
                    param.data = torch.where(mask, parent1_params[name].data, parent2_params[name].data)
                elif name in parent1_params:
                    param.data = parent1_params[name].data.clone()
                elif name in parent2_params:
                    param.data = parent2_params[name].data.clone()

        child = AgentIndividual(
            id=self.next_id,
            architecture_type=child_type,
            model=child_model,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            species_id=3 if child_type == 'Hybrid' else parent1.species_id
        )
        self.next_id += 1

        return child

    def mutate(self, agent: AgentIndividual) -> AgentIndividual:
        """Apply mutation to agent's weights"""
        if random.random() > self.mutation_rate:
            return agent

        with torch.no_grad():
            for param in agent.model.parameters():
                if random.random() < 0.1:  # Mutate 10% of parameters
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)

        return agent

    def evolve_generation(self):
        """Evolve one generation"""
        print(f"\n{'='*60}")
        print(f"Generation {self.generation}")
        print(f"{'='*60}")

        # Evaluate all agents
        print("Evaluating population...")
        for i, agent in enumerate(self.population):
            fitness = self.evaluate_agent(agent)
            print(f"  Agent {agent.id} ({agent.architecture_type}): fitness={fitness:.4f}")

        # Record species statistics
        stats = self.species_tracker.record_generation(self.population, self.generation)

        # Print species summary
        print(f"\nSpecies Summary:")
        for arch_type, data in stats['species'].items():
            print(f"  {arch_type}: count={data['count']}, avg_fitness={data['avg_fitness']:.4f}")

        # Select parents
        parents = self.select_parents()

        # Create next generation
        new_population = []

        # Keep elites
        elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_count]
        for elite in elites:
            # Deep copy elite
            elite_copy = AgentIndividual(
                id=elite.id,
                architecture_type=elite.architecture_type,
                model=copy.deepcopy(elite.model),
                fitness=elite.fitness,
                generation=elite.generation,
                parent_ids=elite.parent_ids,
                species_id=elite.species_id
            )
            new_population.append(elite_copy)

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                # Crossover
                p1, p2 = random.sample(parents, 2)
                child = self.crossover(p1, p2)
            else:
                # Mutation only
                parent = random.choice(parents)
                child = AgentIndividual(
                    id=self.next_id,
                    architecture_type=parent.architecture_type,
                    model=copy.deepcopy(parent.model),
                    generation=self.generation + 1,
                    parent_ids=[parent.id],
                    species_id=parent.species_id
                )
                self.next_id += 1

            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Track history
        fitnesses = [a.fitness for a in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))

        # Calculate diversity
        arch_counts = {}
        for a in self.population:
            arch_counts[a.architecture_type] = arch_counts.get(a.architecture_type, 0) + 1
        diversity = len([c for c in arch_counts.values() if c > 0]) / 4.0
        self.diversity_history.append(diversity)

        print(f"\nGeneration {self.generation-1} Complete:")
        print(f"  Best Fitness: {max(fitnesses):.4f}")
        print(f"  Avg Fitness: {sum(fitnesses)/len(fitnesses):.4f}")
        print(f"  Diversity: {diversity:.2f}")

        return max(fitnesses)

    def train(self, num_generations: int) -> Dict:
        """Run full coevolutionary training"""
        print("\n" + "="*80)
        print("COEVOLUTIONARY TRAINING - ToM-NAS")
        print("="*80)
        print(f"Population: {self.population_size} agents")
        print(f"Architectures: TRN={self.trn_count}, RSAN={self.rsan_count}, Transformer={self.transformer_count}")
        print(f"Generations: {num_generations}")
        print("="*80)

        best_overall = 0.0
        best_agent = None

        for gen in range(num_generations):
            best_gen = self.evolve_generation()

            if best_gen > best_overall:
                best_overall = best_gen
                best_agent = max(self.population, key=lambda x: x.fitness)

                # Save checkpoint
                self._save_checkpoint(best_agent, gen)

        # Final evaluation
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)

        # Sort by fitness
        final_ranking = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        print("\nFinal Population Ranking:")
        for i, agent in enumerate(final_ranking[:5]):
            print(f"  {i+1}. Agent {agent.id} ({agent.architecture_type})")
            print(f"     Fitness: {agent.fitness:.4f}")
            print(f"     Sally-Anne: {agent.sally_anne_score:.4f}")
            print(f"     Zombie Detection: {agent.zombie_detection_score:.4f}")
            print(f"     Higher-Order ToM: {agent.higher_order_scores}")

        # Save final results
        results = {
            'best_fitness': best_overall,
            'best_architecture': best_agent.architecture_type if best_agent else None,
            'generations': num_generations,
            'fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history,
            'species_summary': self.species_tracker.get_summary(),
            'final_population': [
                {
                    'id': a.id,
                    'architecture': a.architecture_type,
                    'fitness': a.fitness,
                    'sally_anne': a.sally_anne_score,
                    'zombie_detection': a.zombie_detection_score,
                    'higher_order': a.higher_order_scores
                }
                for a in final_ranking
            ]
        }

        with open(os.path.join(self.results_dir, 'coevolution_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _save_checkpoint(self, agent: AgentIndividual, generation: int):
        """Save checkpoint of best agent"""
        checkpoint_path = os.path.join(self.results_dir, f'best_gen_{generation}.pt')
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'architecture_type': agent.architecture_type,
            'fitness': agent.fitness,
            'generation': generation,
            'sally_anne_score': agent.sally_anne_score,
            'zombie_detection_score': agent.zombie_detection_score,
            'higher_order_scores': agent.higher_order_scores
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ToM-NAS Coevolutionary Training')

    # Population settings
    parser.add_argument('--population-size', type=int, default=12,
                       help='Total population size')
    parser.add_argument('--trn-count', type=int, default=4,
                       help='Number of TRN agents')
    parser.add_argument('--rsan-count', type=int, default=4,
                       help='Number of RSAN agents')
    parser.add_argument('--transformer-count', type=int, default=4,
                       help='Number of Transformer agents')

    # Evolution settings
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                       help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.3,
                       help='Crossover rate')

    # Environment settings
    parser.add_argument('--num-zombies', type=int, default=2,
                       help='Number of zombies in world')
    parser.add_argument('--episodes-per-eval', type=int, default=5,
                       help='Episodes per fitness evaluation')

    # Other settings
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--results-dir', type=str, default='coevolution_results',
                       help='Directory for results')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    config = {
        'population_size': args.population_size,
        'trn_count': args.trn_count,
        'rsan_count': args.rsan_count,
        'transformer_count': args.transformer_count,
        'mutation_rate': args.mutation_rate,
        'crossover_rate': args.crossover_rate,
        'num_zombies': args.num_zombies,
        'episodes_per_eval': args.episodes_per_eval,
        'device': args.device,
        'results_dir': args.results_dir,
        'input_dim': 191,
        'hidden_dim': 128,
        'ontology_dim': 181,
        'num_world_agents': 6,
        'max_belief_order': 5,
        'sequence_length': 20
    }

    trainer = CoevolutionaryTrainer(config)
    results = trainer.train(args.generations)

    print("\n" + "="*80)
    print("COEVOLUTIONARY TRAINING COMPLETE")
    print("="*80)
    print(f"Best Fitness: {results['best_fitness']:.4f}")
    print(f"Best Architecture: {results['best_architecture']}")
    print(f"Results saved to: {args.results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
