#!/usr/bin/env python
"""
ToM-NAS Coevolutionary Training System - FIXED VERSION

Key fixes from previous version:
1. Fitness now ACTUALLY measures ToM test performance
2. Species preservation prevents monoculture extinction
3. Zombie detection actually tests detection ability
4. Detailed fitness breakdown printed each generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
from src.world.social_world import SocialWorld4


@dataclass
class AgentIndividual:
    """Represents one agent in the evolutionary population"""
    id: int
    architecture_type: str
    model: nn.Module
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)

    # Detailed fitness components - THESE ARE THE ACTUAL SCORES
    sally_anne_score: float = 0.0
    higher_order_scores: Dict[int, float] = field(default_factory=dict)
    zombie_detection_score: float = 0.0
    cooperation_score: float = 0.0
    survival_score: float = 0.0

    species_id: int = 0

    def __post_init__(self):
        if not self.higher_order_scores:
            self.higher_order_scores = {i: 0.0 for i in range(1, 6)}


class CoevolutionaryTrainer:
    """
    FIXED Coevolutionary training system.

    Key principles:
    1. FITNESS = ACTUAL ToM TEST SCORES (not proxies)
    2. Species preservation prevents extinction
    3. Zombie detection actually tests if agent can identify zombies
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
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.3)
        self.elite_count = config.get('elite_count', 2)

        # CRITICAL: Minimum species size to prevent extinction
        self.min_species_size = config.get('min_species_size', 2)

        # Training settings
        self.episodes_per_eval = config.get('episodes_per_eval', 3)
        self.sequence_length = config.get('sequence_length', 10)

        # Initialize components
        self.num_world_agents = config.get('num_world_agents', 6)
        self.num_zombies = config.get('num_zombies', 2)

        self.world = SocialWorld4(
            num_agents=self.num_world_agents,
            ontology_dim=self.output_dim,
            num_zombies=self.num_zombies
        )

        self.belief_network = BeliefNetwork(
            num_agents=self.num_world_agents,
            ontology_dim=self.output_dim,
            max_order=5
        )

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

    def _create_model(self, arch_type: str) -> nn.Module:
        """Create a model of specified architecture type"""
        if arch_type == 'TRN':
            return TransparentRNN(
                self.input_dim, self.hidden_dim, self.output_dim
            ).to(self.device)
        elif arch_type == 'RSAN':
            return RecursiveSelfAttention(
                self.input_dim, self.hidden_dim, self.output_dim,
                num_heads=4
            ).to(self.device)
        elif arch_type == 'Transformer':
            return TransformerToMAgent(
                self.input_dim, self.hidden_dim, self.output_dim,
                num_layers=2
            ).to(self.device)
        else:  # Hybrid defaults to RSAN
            return RecursiveSelfAttention(
                self.input_dim, self.hidden_dim, self.output_dim,
                num_heads=4
            ).to(self.device)

    def _initialize_population(self) -> List[AgentIndividual]:
        """Initialize diverse population with multiple architecture types"""
        population = []
        agent_id = 0

        for arch_type, count in [('TRN', self.trn_count),
                                  ('RSAN', self.rsan_count),
                                  ('Transformer', self.transformer_count)]:
            for _ in range(count):
                model = self._create_model(arch_type)
                population.append(AgentIndividual(
                    id=agent_id,
                    architecture_type=arch_type,
                    model=model,
                    species_id={'TRN': 0, 'RSAN': 1, 'Transformer': 2}.get(arch_type, 3)
                ))
                agent_id += 1

        print(f"Initialized population: {self.trn_count} TRN, {self.rsan_count} RSAN, {self.transformer_count} Transformer")
        return population

    def evaluate_agent(self, agent: AgentIndividual, verbose: bool = False) -> float:
        """
        FIXED: Fitness = ACTUAL ToM test scores

        Components:
        1. Sally-Anne (30%) - False belief understanding
        2. Higher-Order ToM (25%) - Recursive belief depth
        3. Zombie Detection (25%) - Genuine ToM validation
        4. Cooperation (10%) - Agent's actual choices
        5. Survival (10%) - Agent's resource accumulation
        """
        agent.model.eval()

        with torch.no_grad():
            # 1. SALLY-ANNE TEST (30%)
            sally_score = self._test_sally_anne(agent)
            agent.sally_anne_score = sally_score

            # 2. HIGHER-ORDER ToM (25%)
            higher_order_total = 0.0
            for order in range(1, 6):
                score = self._test_higher_order(agent, order)
                agent.higher_order_scores[order] = score
                higher_order_total += score * (order / 15.0)

            # 3. ZOMBIE DETECTION (25%)
            zombie_score = self._test_zombie_detection(agent)
            agent.zombie_detection_score = zombie_score

            # 4. COOPERATION (10%) - Agent's actual decisions
            coop_score = self._test_cooperation(agent)
            agent.cooperation_score = coop_score

            # 5. SURVIVAL (10%) - Agent's resource gain
            survival_score = self._test_survival(agent)
            agent.survival_score = survival_score

        # COMPOSITE FITNESS - weighted sum of ACTUAL test scores
        fitness = (
            0.30 * sally_score +
            0.25 * higher_order_total +
            0.25 * zombie_score +
            0.10 * coop_score +
            0.10 * survival_score
        )

        agent.fitness = fitness

        if verbose:
            print(f"    Agent {agent.id} ({agent.architecture_type}):")
            print(f"      Sally-Anne: {sally_score:.3f}")
            print(f"      Higher-Order: {higher_order_total:.3f} {dict(agent.higher_order_scores)}")
            print(f"      Zombie Det.: {zombie_score:.3f}")
            print(f"      Cooperation: {coop_score:.3f}")
            print(f"      Survival: {survival_score:.3f}")
            print(f"      TOTAL: {fitness:.4f}")

        return fitness

    def _test_sally_anne(self, agent: AgentIndividual) -> float:
        """
        Sally-Anne false belief test.

        Scenario: Sally puts marble in basket, leaves, Anne moves it to box.
        Question: Where will Sally look?
        Correct: Basket (Sally's false belief)
        """
        # Create scenario encoding
        # [sally_here, anne_here, marble_basket, marble_box, sally_saw_move]

        # Step 1: Sally puts marble in basket (both present)
        step1 = torch.zeros(1, 1, self.input_dim)
        step1[0, 0, 0] = 1.0  # Sally present
        step1[0, 0, 1] = 1.0  # Anne present
        step1[0, 0, 2] = 1.0  # Marble in basket
        step1[0, 0, 3] = 0.0  # Marble not in box

        # Step 2: Sally leaves
        step2 = torch.zeros(1, 1, self.input_dim)
        step2[0, 0, 0] = 0.0  # Sally gone
        step2[0, 0, 1] = 1.0  # Anne present
        step2[0, 0, 2] = 1.0  # Marble still in basket

        # Step 3: Anne moves marble (Sally not watching)
        step3 = torch.zeros(1, 1, self.input_dim)
        step3[0, 0, 0] = 0.0  # Sally gone
        step3[0, 0, 1] = 1.0  # Anne present
        step3[0, 0, 2] = 0.0  # Marble NOT in basket
        step3[0, 0, 3] = 1.0  # Marble in box

        # Step 4: Sally returns - where does she believe marble is?
        step4 = torch.zeros(1, 1, self.input_dim)
        step4[0, 0, 0] = 1.0  # Sally back
        step4[0, 0, 1] = 1.0  # Anne present
        step4[0, 0, 2] = 0.0  # Marble actually NOT in basket
        step4[0, 0, 3] = 1.0  # Marble actually in box
        step4[0, 0, 4] = 0.0  # Sally did NOT see move

        sequence = torch.cat([step1, step2, step3, step4], dim=1).to(self.device)

        output = agent.model(sequence)
        beliefs = output['beliefs']

        # Check: Does agent predict Sally believes marble is in BASKET?
        # beliefs[0] = basket belief, beliefs[1] = box belief
        basket_belief = beliefs[0, 0].item() if beliefs.shape[-1] > 0 else 0.5
        box_belief = beliefs[0, 1].item() if beliefs.shape[-1] > 1 else 0.5

        # Score: How much more basket than box?
        if basket_belief > box_belief:
            score = min(1.0, (basket_belief - box_belief) + 0.5)
        else:
            score = max(0.0, 0.5 - (box_belief - basket_belief))

        return score

    def _test_higher_order(self, agent: AgentIndividual, order: int) -> float:
        """
        Test nth-order Theory of Mind.

        Order 1: A knows X
        Order 2: A knows B knows X
        Order 3: A knows B knows A knows X
        etc.

        Higher orders should show DECREASING confidence (more uncertainty)
        """
        # Create sequence encoding the belief depth
        seq_len = order + 2
        sequence = torch.zeros(1, seq_len, self.input_dim).to(self.device)

        # Encode belief chain markers
        for i in range(order):
            sequence[0, i, 10 + i] = 1.0  # Mark each belief level
            sequence[0, i, 50] = (order - i) / order  # Decreasing certainty signal

        output = agent.model(sequence)
        beliefs = output['beliefs']

        # Expected: confidence should DECREASE with order
        # Order 1: ~0.85, Order 2: ~0.7, Order 3: ~0.55, Order 4: ~0.4, Order 5: ~0.3
        expected_conf = max(0.25, 1.0 - (order - 1) * 0.15)
        actual_conf = beliefs.mean().item()

        # Score based on how close to expected
        error = abs(actual_conf - expected_conf)
        score = max(0.0, 1.0 - error * 2)

        return score

    def _test_zombie_detection(self, agent: AgentIndividual) -> float:
        """
        FIXED: Actually test zombie detection ability.

        Present agent with behavior sequences from zombies vs real agents.
        Agent must correctly classify which is which.
        """
        correct = 0
        total = 0

        for episode in range(self.episodes_per_eval):
            self.world.reset()

            # Test each agent in the world
            for suspect_id in range(self.world.num_agents):
                is_zombie = self.world.agents[suspect_id].is_zombie

                # Create observation of suspect's behavior
                obs_sequence = torch.zeros(1, self.sequence_length, self.input_dim).to(self.device)

                # Encode suspect's characteristics
                suspect = self.world.agents[suspect_id]

                # Zombies have specific behavioral signatures
                if is_zombie:
                    # Zombie signature: inconsistent, reactive, no belief modeling
                    for t in range(self.sequence_length):
                        obs_sequence[0, t, 0] = random.random()  # Erratic
                        obs_sequence[0, t, 1] = random.random()  # Inconsistent
                        obs_sequence[0, t, 5] = 0.0  # No belief signal
                else:
                    # Real agent: consistent, belief-driven
                    base_val = random.random()
                    for t in range(self.sequence_length):
                        obs_sequence[0, t, 0] = base_val + random.gauss(0, 0.1)  # Consistent
                        obs_sequence[0, t, 1] = base_val + random.gauss(0, 0.1)  # Coherent
                        obs_sequence[0, t, 5] = 0.8  # Strong belief signal

                # Get agent's prediction
                output = agent.model(obs_sequence)
                beliefs = output['beliefs']

                # Interpret: High belief[5] = real agent, Low = zombie
                prediction_real = beliefs[0, 5].item() if beliefs.shape[-1] > 5 else 0.5
                predicted_zombie = prediction_real < 0.5

                if predicted_zombie == is_zombie:
                    correct += 1
                total += 1

        return correct / max(total, 1)

    def _test_cooperation(self, agent: AgentIndividual) -> float:
        """
        FIXED: Test agent's actual cooperation decisions.

        Present repeated game scenarios, measure if agent:
        - Cooperates with cooperators (TFT)
        - Defects against defectors
        - Shows strategic reciprocity
        """
        good_decisions = 0
        total_decisions = 0

        for episode in range(self.episodes_per_eval):
            # Scenario 1: Partner cooperated last round - should cooperate
            coop_history = torch.zeros(1, 5, self.input_dim).to(self.device)
            coop_history[0, :, 80] = 1.0  # Game marker
            coop_history[0, :, 81] = 0.9  # Partner cooperated

            output = agent.model(coop_history)
            action = output['actions'].mean().item()

            # Good decision: cooperate back (action > 0.5)
            if action > 0.5:
                good_decisions += 1
            total_decisions += 1

            # Scenario 2: Partner defected - should defect
            defect_history = torch.zeros(1, 5, self.input_dim).to(self.device)
            defect_history[0, :, 80] = 1.0  # Game marker
            defect_history[0, :, 81] = 0.1  # Partner defected

            output = agent.model(defect_history)
            action = output['actions'].mean().item()

            # Good decision: defect back (action < 0.5)
            if action < 0.5:
                good_decisions += 1
            total_decisions += 1

            # Scenario 3: Mixed history - should be cautious
            mixed_history = torch.zeros(1, 5, self.input_dim).to(self.device)
            mixed_history[0, :, 80] = 1.0
            mixed_history[0, 0:2, 81] = 0.9  # Cooperated early
            mixed_history[0, 2:5, 81] = 0.1  # Defected recently

            output = agent.model(mixed_history)
            action = output['actions'].mean().item()

            # Good decision: cautious/defensive (action < 0.6)
            if action < 0.6:
                good_decisions += 1
            total_decisions += 1

        return good_decisions / max(total_decisions, 1)

    def _test_survival(self, agent: AgentIndividual) -> float:
        """
        FIXED: Measure agent's own resource accumulation.
        """
        total_gained = 0.0

        for episode in range(self.episodes_per_eval):
            self.world.reset()
            initial_resources = 100.0

            for t in range(self.sequence_length):
                # Get agent's decision
                obs = self.world.get_observation(0)
                obs_tensor = torch.zeros(1, 1, self.input_dim).to(self.device)
                obs_tensor[0, 0, 0] = obs['own_resources'] / 200.0
                obs_tensor[0, 0, 1] = obs['own_energy'] / 100.0

                output = agent.model(obs_tensor)
                action_val = output['actions'].mean().item()

                # Convert to action
                if action_val > 0.6:
                    action_type = 'cooperate'
                elif action_val < 0.4:
                    action_type = 'defect'
                else:
                    action_type = 'cooperate'  # Default

                actions = [{'type': action_type}] + [{'type': 'cooperate'} for _ in range(self.world.num_agents - 1)]
                self.world.step(actions, self.belief_network)

            # Measure gain
            final_resources = self.world.agents[0].resources
            gain = (final_resources - initial_resources) / 100.0
            total_gained += max(0, gain)

        # Normalize
        avg_gain = total_gained / self.episodes_per_eval
        return min(1.0, avg_gain)

    def select_parents(self) -> List[AgentIndividual]:
        """
        FIXED: Selection with species preservation.
        Ensures minimum representation of each architecture type.
        """
        parents = []

        # 1. Elite preservation (best overall)
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.elite_count])

        # 2. SPECIES PRESERVATION - ensure each type survives
        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            species_agents = [a for a in self.population if a.architecture_type == arch_type]
            in_parents = [p for p in parents if p.architecture_type == arch_type]

            # Ensure minimum representation
            while len(in_parents) < self.min_species_size and species_agents:
                # Add best from this species not already in parents
                for agent in sorted(species_agents, key=lambda x: x.fitness, reverse=True):
                    if agent not in parents:
                        parents.append(agent)
                        in_parents.append(agent)
                        break
                else:
                    break

        # 3. Tournament selection for remaining slots
        while len(parents) < self.population_size // 2:
            tournament = random.sample(self.population, min(3, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            if winner not in parents:
                parents.append(winner)

        return parents

    def crossover(self, parent1: AgentIndividual, parent2: AgentIndividual) -> AgentIndividual:
        """Create offspring through crossover"""
        # Determine child architecture
        if parent1.architecture_type == parent2.architecture_type:
            child_type = parent1.architecture_type
        else:
            # Different architectures - pick one or create hybrid
            if random.random() < 0.3:
                child_type = 'Hybrid'
            else:
                child_type = random.choice([parent1.architecture_type, parent2.architecture_type])

        # Create new model
        child_model = self._create_model(child_type)

        # Weight crossover
        p1_params = dict(parent1.model.named_parameters())
        p2_params = dict(parent2.model.named_parameters())

        with torch.no_grad():
            for name, param in child_model.named_parameters():
                if name in p1_params and name in p2_params:
                    if p1_params[name].shape == p2_params[name].shape == param.shape:
                        mask = torch.rand_like(param) < 0.5
                        param.data = torch.where(mask, p1_params[name].data, p2_params[name].data)

        child = AgentIndividual(
            id=self.next_id,
            architecture_type=child_type,
            model=child_model,
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
            species_id={'TRN': 0, 'RSAN': 1, 'Transformer': 2, 'Hybrid': 3}.get(child_type, 3)
        )
        self.next_id += 1
        return child

    def mutate(self, agent: AgentIndividual) -> AgentIndividual:
        """Apply mutation to agent's weights"""
        if random.random() > self.mutation_rate:
            return agent

        with torch.no_grad():
            for param in agent.model.parameters():
                if random.random() < 0.1:
                    noise = torch.randn_like(param) * 0.02
                    param.add_(noise)
        return agent

    def evolve_generation(self):
        """Evolve one generation with detailed output"""
        print(f"\n{'='*70}")
        print(f"GENERATION {self.generation}")
        print(f"{'='*70}")

        # Count species
        species_counts = {}
        for agent in self.population:
            species_counts[agent.architecture_type] = species_counts.get(agent.architecture_type, 0) + 1
        print(f"Population: {species_counts}")

        # Evaluate all agents
        print(f"\nEvaluating {len(self.population)} agents...")
        print("-" * 70)

        for agent in self.population:
            self.evaluate_agent(agent, verbose=True)

        # Statistics
        fitnesses = [a.fitness for a in self.population]
        sally_scores = [a.sally_anne_score for a in self.population]
        zombie_scores = [a.zombie_detection_score for a in self.population]

        print(f"\n{'='*70}")
        print("GENERATION STATISTICS")
        print(f"{'='*70}")
        print(f"  Best Fitness:     {max(fitnesses):.4f}")
        print(f"  Avg Fitness:      {sum(fitnesses)/len(fitnesses):.4f}")
        print(f"  Avg Sally-Anne:   {sum(sally_scores)/len(sally_scores):.4f}")
        print(f"  Avg Zombie Det:   {sum(zombie_scores)/len(zombie_scores):.4f}")

        # Per-species stats
        print(f"\nPer-Species Performance:")
        for arch_type in ['TRN', 'RSAN', 'Transformer', 'Hybrid']:
            agents = [a for a in self.population if a.architecture_type == arch_type]
            if agents:
                avg_fit = sum(a.fitness for a in agents) / len(agents)
                avg_sally = sum(a.sally_anne_score for a in agents) / len(agents)
                print(f"  {arch_type:12s}: n={len(agents):2d}, fitness={avg_fit:.4f}, sally={avg_sally:.4f}")

        # Select parents
        parents = self.select_parents()

        # Create next generation
        new_population = []

        # Keep elites
        elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_count]
        for elite in elites:
            elite_copy = AgentIndividual(
                id=elite.id,
                architecture_type=elite.architecture_type,
                model=copy.deepcopy(elite.model),
                fitness=elite.fitness,
                generation=elite.generation,
                species_id=elite.species_id
            )
            new_population.append(elite_copy)

        # SPECIES PRESERVATION: Ensure each type has minimum representation
        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            current_count = len([a for a in new_population if a.architecture_type == arch_type])
            if current_count < self.min_species_size:
                # Add fresh agents of this type
                needed = self.min_species_size - current_count
                for _ in range(needed):
                    new_agent = AgentIndividual(
                        id=self.next_id,
                        architecture_type=arch_type,
                        model=self._create_model(arch_type),
                        generation=self.generation + 1,
                        species_id={'TRN': 0, 'RSAN': 1, 'Transformer': 2}.get(arch_type, 3)
                    )
                    self.next_id += 1
                    new_population.append(new_agent)

        # Generate remaining offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                p1, p2 = random.sample(parents, 2)
                child = self.crossover(p1, p2)
            else:
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

        self.population = new_population[:self.population_size]
        self.generation += 1

        # Track history
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))

        return max(fitnesses)

    def train(self, num_generations: int) -> Dict:
        """Run full coevolutionary training"""
        print("\n" + "=" * 80)
        print("COEVOLUTIONARY TRAINING - ToM-NAS (FIXED)")
        print("=" * 80)
        print(f"Population: {self.population_size} agents")
        print(f"Architectures: TRN={self.trn_count}, RSAN={self.rsan_count}, Transformer={self.transformer_count}")
        print(f"Generations: {num_generations}")
        print(f"Min species size: {self.min_species_size} (prevents extinction)")
        print("=" * 80)

        best_overall = 0.0
        best_agent = None

        for gen in range(num_generations):
            best_gen = self.evolve_generation()

            if best_gen > best_overall:
                best_overall = best_gen
                best_agent = max(self.population, key=lambda x: x.fitness)
                self._save_checkpoint(best_agent, gen)

        # Final evaluation
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        final_ranking = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        print("\nFinal Top 5:")
        for i, agent in enumerate(final_ranking[:5]):
            print(f"\n{i+1}. Agent {agent.id} ({agent.architecture_type})")
            print(f"   Fitness: {agent.fitness:.4f}")
            print(f"   Sally-Anne: {agent.sally_anne_score:.4f}")
            print(f"   Zombie Det: {agent.zombie_detection_score:.4f}")
            print(f"   Higher-Order: {agent.higher_order_scores}")

        results = {
            'best_fitness': best_overall,
            'best_architecture': best_agent.architecture_type if best_agent else None,
            'generations': num_generations,
            'fitness_history': self.best_fitness_history,
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

        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _save_checkpoint(self, agent: AgentIndividual, generation: int):
        """Save checkpoint"""
        path = os.path.join(self.results_dir, f'best_gen_{generation}.pt')
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'architecture_type': agent.architecture_type,
            'fitness': agent.fitness,
            'sally_anne': agent.sally_anne_score,
            'zombie_detection': agent.zombie_detection_score,
            'higher_order': agent.higher_order_scores
        }, path)


def main():
    parser = argparse.ArgumentParser(description='ToM-NAS Coevolutionary Training (FIXED)')
    parser.add_argument('--population-size', type=int, default=12)
    parser.add_argument('--trn-count', type=int, default=4)
    parser.add_argument('--rsan-count', type=int, default=4)
    parser.add_argument('--transformer-count', type=int, default=4)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--min-species-size', type=int, default=2,
                       help='Minimum agents per architecture type (prevents extinction)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--results-dir', type=str, default='coevolution_results')
    args = parser.parse_args()

    config = {
        'population_size': args.population_size,
        'trn_count': args.trn_count,
        'rsan_count': args.rsan_count,
        'transformer_count': args.transformer_count,
        'min_species_size': args.min_species_size,
        'device': args.device,
        'results_dir': args.results_dir,
        'input_dim': 191,
        'hidden_dim': 128,
        'ontology_dim': 181,
        'num_world_agents': 6,
        'num_zombies': 2,
        'episodes_per_eval': 3,
        'sequence_length': 10
    }

    trainer = CoevolutionaryTrainer(config)
    results = trainer.train(args.generations)

    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
