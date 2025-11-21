#!/usr/bin/env python
"""
ToM-NAS Coevolutionary Training System - SCALED VERSION

Fixes:
1. Final summary now shows actual metrics (not zeros)
2. Scaled up: 24 agents, 50 generations, more rigorous evaluation
3. Better tracking and visualization
4. Species preservation with niche bonuses
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

# Dissertation-quality instrumentation
try:
    from src.instrumentation import InstrumentationSuite
    INSTRUMENTATION_AVAILABLE = True
except ImportError:
    INSTRUMENTATION_AVAILABLE = False
    print("Warning: Instrumentation not available. Install for dissertation-quality logging.")


@dataclass
class AgentIndividual:
    """Represents one agent in the evolutionary population"""
    id: int
    architecture_type: str
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

    species_id: int = 0

    def __post_init__(self):
        if not self.higher_order_scores:
            self.higher_order_scores = {i: 0.0 for i in range(1, 6)}

    def copy_scores_from(self, other: 'AgentIndividual'):
        """Copy all score metrics from another agent"""
        self.fitness = other.fitness
        self.sally_anne_score = other.sally_anne_score
        self.zombie_detection_score = other.zombie_detection_score
        self.cooperation_score = other.cooperation_score
        self.survival_score = other.survival_score
        self.higher_order_scores = dict(other.higher_order_scores)


class CoevolutionaryTrainer:
    """
    Scaled coevolutionary training system for ToM-NAS.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cpu')

        # Population settings - SCALED UP
        self.population_size = config.get('population_size', 24)
        self.trn_count = config.get('trn_count', 8)
        self.rsan_count = config.get('rsan_count', 8)
        self.transformer_count = config.get('transformer_count', 8)

        # Architecture settings
        self.input_dim = config.get('input_dim', 191)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.output_dim = config.get('ontology_dim', 181)

        # Evolution settings
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.3)
        self.elite_count = config.get('elite_count', 3)

        # Species preservation - INCREASED
        self.min_species_size = config.get('min_species_size', 3)

        # Evaluation settings - MORE RIGOROUS
        self.sally_anne_tests = config.get('sally_anne_tests', 20)
        self.zombie_episodes = config.get('zombie_episodes', 10)
        self.higher_order_tests = config.get('higher_order_tests', 10)
        self.coop_episodes = config.get('coop_episodes', 5)
        self.survival_episodes = config.get('survival_episodes', 5)
        self.sequence_length = config.get('sequence_length', 10)

        # Checkpointing
        self.checkpoint_interval = config.get('checkpoint_interval', 10)

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

        # History tracking - ENHANCED
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.sally_anne_history = []
        self.zombie_history = []
        self.species_history = {'TRN': [], 'RSAN': [], 'Transformer': [], 'Hybrid': []}

        # Best agents per species
        self.best_per_species = {}

        # Dissertation-quality instrumentation
        self.enable_instrumentation = config.get('enable_instrumentation', False)
        self.instrumentation = None
        if self.enable_instrumentation and INSTRUMENTATION_AVAILABLE:
            instr_dir = os.path.join(self.results_dir, 'instrumentation')
            self.instrumentation = InstrumentationSuite(instr_dir)
            print(f"Instrumentation enabled: {instr_dir}")

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
        """Initialize diverse population"""
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
        Comprehensive fitness evaluation with multiple tests per component.
        Includes full instrumentation for dissertation-quality analysis.
        """
        agent.model.eval()

        # Start instrumentation trace
        if self.instrumentation:
            self.instrumentation.trace_logger.start_agent_trace(
                agent.id, agent.architecture_type, self.generation
            )

        with torch.no_grad():
            # 1. SALLY-ANNE TEST (30%) - Multiple trials
            sally_scores = []
            for trial in range(self.sally_anne_tests):
                score = self._test_sally_anne_instrumented(agent, trial)
                sally_scores.append(score)
            sally_score = sum(sally_scores) / len(sally_scores)
            agent.sally_anne_score = sally_score

            # 2. HIGHER-ORDER ToM (25%)
            higher_order_total = 0.0
            for order in range(1, 6):
                order_scores = []
                for trial in range(self.higher_order_tests):
                    score = self._test_higher_order_instrumented(agent, order, trial)
                    order_scores.append(score)
                avg_order = sum(order_scores) / len(order_scores)
                agent.higher_order_scores[order] = avg_order
                higher_order_total += avg_order * (order / 15.0)

            # 3. ZOMBIE DETECTION (25%) - With full transcripts
            zombie_scores = []
            for episode in range(self.zombie_episodes):
                score = self._test_zombie_detection_instrumented(agent, episode)
                zombie_scores.append(score)
            zombie_score = sum(zombie_scores) / len(zombie_scores)
            agent.zombie_detection_score = zombie_score

            # 4. COOPERATION (10%)
            coop_scores = []
            for episode in range(self.coop_episodes):
                score = self._test_cooperation_instrumented(agent, episode)
                coop_scores.append(score)
            coop_score = sum(coop_scores) / len(coop_scores)
            agent.cooperation_score = coop_score

            # 5. SURVIVAL (10%)
            survival_scores = []
            for episode in range(self.survival_episodes):
                score = self._test_survival_instrumented(agent, episode)
                survival_scores.append(score)
            survival_score = sum(survival_scores) / len(survival_scores)
            agent.survival_score = survival_score

        # COMPOSITE FITNESS
        fitness = (
            0.30 * sally_score +
            0.25 * higher_order_total +
            0.25 * zombie_score +
            0.10 * coop_score +
            0.10 * survival_score
        )

        agent.fitness = fitness

        # Finalize instrumentation
        if self.instrumentation:
            self.instrumentation.trace_logger.finalize_agent(agent.id, fitness)
            self.instrumentation.motif_extractor.analyze_model(
                agent.model, agent.architecture_type, agent.id,
                {
                    'fitness': fitness,
                    'sally_anne': sally_score,
                    'zombie_detection': zombie_score,
                    'higher_order': agent.higher_order_scores
                }
            )

        if verbose:
            print(f"    Agent {agent.id} ({agent.architecture_type}):")
            print(f"      Sally-Anne: {sally_score:.3f}")
            print(f"      Higher-Order: {higher_order_total:.3f}")
            print(f"      Zombie Det.: {zombie_score:.3f}")
            print(f"      Cooperation: {coop_score:.3f}")
            print(f"      Survival: {survival_score:.3f}")
            print(f"      TOTAL: {fitness:.4f}")

        return fitness

    def _test_sally_anne_instrumented(self, agent: AgentIndividual, trial: int) -> float:
        """Sally-Anne test with instrumentation"""
        return self._test_sally_anne(agent)

    def _test_higher_order_instrumented(self, agent: AgentIndividual, order: int, trial: int) -> float:
        """Higher-order ToM test with instrumentation"""
        return self._test_higher_order(agent, order)

    def _test_zombie_detection_instrumented(self, agent: AgentIndividual, episode: int) -> float:
        """Zombie detection with full transcript logging"""
        self.world.reset()
        correct = 0
        total = 0

        for suspect_id in range(self.world.num_agents):
            is_zombie = self.world.agents[suspect_id].is_zombie

            # Create observation sequence
            obs_sequence = torch.zeros(1, self.sequence_length, self.input_dim).to(self.device)
            if is_zombie:
                for t in range(self.sequence_length):
                    obs_sequence[0, t, 0] = random.random()
                    obs_sequence[0, t, 1] = random.random()
                    obs_sequence[0, t, 5] = 0.0
            else:
                base_val = random.random()
                for t in range(self.sequence_length):
                    obs_sequence[0, t, 0] = base_val + random.gauss(0, 0.1)
                    obs_sequence[0, t, 1] = base_val + random.gauss(0, 0.1)
                    obs_sequence[0, t, 5] = 0.8

            # Get prediction
            output = agent.model(obs_sequence)
            beliefs = output['beliefs']

            # Log forward pass
            if self.instrumentation:
                self.instrumentation.trace_logger.log_forward_pass(
                    agent.id, agent.model, obs_sequence, output, episode * self.world.num_agents + suspect_id
                )

            prediction_real = beliefs[0, 5].item() if beliefs.shape[-1] > 5 else 0.5
            predicted_zombie = prediction_real < 0.5

            # Log zombie detection reasoning
            reasoning = f"belief[5]={prediction_real:.3f}, threshold=0.5, predicted={'zombie' if predicted_zombie else 'real'}"
            if self.instrumentation:
                self.instrumentation.trace_logger.log_zombie_detection(
                    agent.id, suspect_id, is_zombie, predicted_zombie, reasoning
                )

                # Log full transcript
                interaction_id = self.instrumentation.zombie_recorder.start_interaction(
                    self.generation, "belief", agent.id, suspect_id, is_zombie
                )
                self.instrumentation.zombie_recorder.add_round(
                    interaction_id,
                    probe="Observe behavior sequence",
                    response=f"Belief signal: {prediction_real:.3f}",
                    reasoning=reasoning,
                    internal_state={'beliefs': beliefs[0, :10].tolist()},
                    confidence=abs(prediction_real - 0.5) * 2
                )
                strategy = "belief_consistency" if abs(prediction_real - 0.5) > 0.3 else "threshold_default"
                self.instrumentation.zombie_recorder.finalize_interaction(
                    interaction_id, predicted_zombie, strategy
                )

            if predicted_zombie == is_zombie:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _test_cooperation_instrumented(self, agent: AgentIndividual, episode: int) -> float:
        """Cooperation test with instrumentation"""
        return self._test_cooperation_single(agent)

    def _test_survival_instrumented(self, agent: AgentIndividual, episode: int) -> float:
        """Survival test with instrumentation"""
        score = self._test_survival_single(agent)

        # Capture social world snapshot
        if self.instrumentation:
            self.instrumentation.social_visualizer.capture_snapshot(
                self.world, self.generation, episode
            )

        return score

    def _test_sally_anne(self, agent: AgentIndividual) -> float:
        """Single Sally-Anne trial with slight variation"""
        # Add small random variation to inputs for robustness testing
        noise = torch.randn(1, 4, self.input_dim) * 0.05

        step1 = torch.zeros(1, 1, self.input_dim)
        step1[0, 0, 0] = 1.0  # Sally present
        step1[0, 0, 1] = 1.0  # Anne present
        step1[0, 0, 2] = 1.0  # Marble in basket

        step2 = torch.zeros(1, 1, self.input_dim)
        step2[0, 0, 0] = 0.0  # Sally gone
        step2[0, 0, 1] = 1.0
        step2[0, 0, 2] = 1.0

        step3 = torch.zeros(1, 1, self.input_dim)
        step3[0, 0, 0] = 0.0
        step3[0, 0, 1] = 1.0
        step3[0, 0, 2] = 0.0  # Marble moved
        step3[0, 0, 3] = 1.0  # To box

        step4 = torch.zeros(1, 1, self.input_dim)
        step4[0, 0, 0] = 1.0  # Sally back
        step4[0, 0, 1] = 1.0
        step4[0, 0, 2] = 0.0
        step4[0, 0, 3] = 1.0
        step4[0, 0, 4] = 0.0  # Sally didn't see

        sequence = torch.cat([step1, step2, step3, step4], dim=1).to(self.device)
        sequence = sequence + noise.to(self.device)

        output = agent.model(sequence)
        beliefs = output['beliefs']

        basket_belief = beliefs[0, 0].item() if beliefs.shape[-1] > 0 else 0.5
        box_belief = beliefs[0, 1].item() if beliefs.shape[-1] > 1 else 0.5

        if basket_belief > box_belief:
            return min(1.0, (basket_belief - box_belief) + 0.5)
        else:
            return max(0.0, 0.5 - (box_belief - basket_belief))

    def _test_higher_order(self, agent: AgentIndividual, order: int) -> float:
        """Test nth-order ToM"""
        seq_len = order + 2
        sequence = torch.zeros(1, seq_len, self.input_dim).to(self.device)

        for i in range(order):
            sequence[0, i, 10 + i] = 1.0
            sequence[0, i, 50] = (order - i) / order

        # Add noise for robustness
        sequence = sequence + torch.randn_like(sequence) * 0.02

        output = agent.model(sequence)
        beliefs = output['beliefs']

        expected_conf = max(0.25, 1.0 - (order - 1) * 0.15)
        actual_conf = beliefs.mean().item()

        error = abs(actual_conf - expected_conf)
        return max(0.0, 1.0 - error * 2)

    def _test_zombie_detection_single(self, agent: AgentIndividual) -> float:
        """Single zombie detection episode"""
        self.world.reset()
        correct = 0
        total = 0

        for suspect_id in range(self.world.num_agents):
            is_zombie = self.world.agents[suspect_id].is_zombie

            obs_sequence = torch.zeros(1, self.sequence_length, self.input_dim).to(self.device)

            if is_zombie:
                for t in range(self.sequence_length):
                    obs_sequence[0, t, 0] = random.random()
                    obs_sequence[0, t, 1] = random.random()
                    obs_sequence[0, t, 5] = 0.0
            else:
                base_val = random.random()
                for t in range(self.sequence_length):
                    obs_sequence[0, t, 0] = base_val + random.gauss(0, 0.1)
                    obs_sequence[0, t, 1] = base_val + random.gauss(0, 0.1)
                    obs_sequence[0, t, 5] = 0.8

            output = agent.model(obs_sequence)
            beliefs = output['beliefs']

            prediction_real = beliefs[0, 5].item() if beliefs.shape[-1] > 5 else 0.5
            predicted_zombie = prediction_real < 0.5

            if predicted_zombie == is_zombie:
                correct += 1
            total += 1

        return correct / max(total, 1)

    def _test_cooperation_single(self, agent: AgentIndividual) -> float:
        """Single cooperation test episode"""
        good_decisions = 0
        total_decisions = 0

        # Scenario 1: Partner cooperated
        coop_history = torch.zeros(1, 5, self.input_dim).to(self.device)
        coop_history[0, :, 80] = 1.0
        coop_history[0, :, 81] = 0.9
        output = agent.model(coop_history)
        if output['actions'].mean().item() > 0.5:
            good_decisions += 1
        total_decisions += 1

        # Scenario 2: Partner defected
        defect_history = torch.zeros(1, 5, self.input_dim).to(self.device)
        defect_history[0, :, 80] = 1.0
        defect_history[0, :, 81] = 0.1
        output = agent.model(defect_history)
        if output['actions'].mean().item() < 0.5:
            good_decisions += 1
        total_decisions += 1

        # Scenario 3: Mixed
        mixed_history = torch.zeros(1, 5, self.input_dim).to(self.device)
        mixed_history[0, :, 80] = 1.0
        mixed_history[0, 0:2, 81] = 0.9
        mixed_history[0, 2:5, 81] = 0.1
        output = agent.model(mixed_history)
        if output['actions'].mean().item() < 0.6:
            good_decisions += 1
        total_decisions += 1

        return good_decisions / total_decisions

    def _test_survival_single(self, agent: AgentIndividual) -> float:
        """Single survival test episode"""
        self.world.reset()
        initial_resources = 100.0

        for t in range(self.sequence_length):
            obs = self.world.get_observation(0)
            obs_tensor = torch.zeros(1, 1, self.input_dim).to(self.device)
            obs_tensor[0, 0, 0] = obs['own_resources'] / 200.0
            obs_tensor[0, 0, 1] = obs['own_energy'] / 100.0

            output = agent.model(obs_tensor)
            action_val = output['actions'].mean().item()

            action_type = 'cooperate' if action_val > 0.5 else 'defect'
            actions = [{'type': action_type}] + [{'type': 'cooperate'} for _ in range(self.world.num_agents - 1)]
            self.world.step(actions, self.belief_network)

        final_resources = self.world.agents[0].resources
        gain = (final_resources - initial_resources) / 100.0
        return min(1.0, max(0, gain))

    def select_parents(self) -> List[AgentIndividual]:
        """Selection with species preservation and niche bonus"""
        parents = []

        # Count species
        species_counts = {}
        for a in self.population:
            species_counts[a.architecture_type] = species_counts.get(a.architecture_type, 0) + 1

        # Apply niche bonus for rare species
        for agent in self.population:
            count = species_counts.get(agent.architecture_type, 1)
            if count < 5:
                agent.fitness *= 1.1  # 10% bonus for rare

        # Elite preservation
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.elite_count])

        # Species preservation
        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            species_agents = [a for a in self.population if a.architecture_type == arch_type]
            in_parents = [p for p in parents if p.architecture_type == arch_type]

            while len(in_parents) < self.min_species_size and species_agents:
                for agent in sorted(species_agents, key=lambda x: x.fitness, reverse=True):
                    if agent not in parents:
                        parents.append(agent)
                        in_parents.append(agent)
                        break
                else:
                    break

        # Tournament selection
        while len(parents) < self.population_size // 2:
            tournament = random.sample(self.population, min(3, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            if winner not in parents:
                parents.append(winner)

        return parents

    def crossover(self, parent1: AgentIndividual, parent2: AgentIndividual) -> AgentIndividual:
        """Create offspring through crossover"""
        if parent1.architecture_type == parent2.architecture_type:
            child_type = parent1.architecture_type
        else:
            if random.random() < 0.3:
                child_type = 'Hybrid'
            else:
                child_type = random.choice([parent1.architecture_type, parent2.architecture_type])

        child_model = self._create_model(child_type)

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
        """Apply mutation"""
        if random.random() > self.mutation_rate:
            return agent

        with torch.no_grad():
            for param in agent.model.parameters():
                if random.random() < 0.1:
                    noise = torch.randn_like(param) * 0.02
                    param.add_(noise)
        return agent

    def evolve_generation(self) -> Tuple[float, List[AgentIndividual]]:
        """Evolve one generation, return best fitness AND evaluated population"""
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

        # Store evaluated population BEFORE creating next generation
        evaluated_population = [copy.deepcopy(a) for a in self.population]

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
                best_agent = max(agents, key=lambda x: x.fitness)
                self.best_per_species[arch_type] = best_agent
                print(f"  {arch_type:12s}: n={len(agents):2d}, avg={avg_fit:.4f}, best={best_agent.fitness:.4f}")

        # Track history
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.sally_anne_history.append(sum(sally_scores) / len(sally_scores))
        self.zombie_history.append(sum(zombie_scores) / len(zombie_scores))

        for arch_type in ['TRN', 'RSAN', 'Transformer', 'Hybrid']:
            agents = [a for a in self.population if a.architecture_type == arch_type]
            self.species_history[arch_type].append(len(agents))

        # Save generation instrumentation data
        if self.instrumentation:
            self.instrumentation.save_generation(self.generation)

        # Select parents
        parents = self.select_parents()

        # Create next generation
        new_population = []

        # Keep elites WITH their scores
        elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_count]
        for elite in elites:
            elite_copy = AgentIndividual(
                id=elite.id,
                architecture_type=elite.architecture_type,
                model=copy.deepcopy(elite.model),
                generation=elite.generation,
                species_id=elite.species_id
            )
            elite_copy.copy_scores_from(elite)  # FIXED: Copy all scores
            new_population.append(elite_copy)

        # Species preservation
        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            current_count = len([a for a in new_population if a.architecture_type == arch_type])
            if current_count < self.min_species_size:
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

        return max(fitnesses), evaluated_population

    def train(self, num_generations: int) -> Dict:
        """Run full coevolutionary training"""
        print("\n" + "=" * 80)
        print("COEVOLUTIONARY TRAINING - ToM-NAS (SCALED)")
        print("=" * 80)
        print(f"Population: {self.population_size} agents")
        print(f"Architectures: TRN={self.trn_count}, RSAN={self.rsan_count}, Transformer={self.transformer_count}")
        print(f"Generations: {num_generations}")
        print(f"Min species size: {self.min_species_size}")
        print(f"Sally-Anne tests per agent: {self.sally_anne_tests}")
        print(f"Zombie episodes per agent: {self.zombie_episodes}")
        print("=" * 80)

        best_overall = 0.0
        best_agent = None
        last_evaluated_population = None

        for gen in range(num_generations):
            best_gen, evaluated_pop = self.evolve_generation()
            last_evaluated_population = evaluated_pop  # Keep reference to evaluated agents

            if best_gen > best_overall:
                best_overall = best_gen
                best_agent = max(evaluated_pop, key=lambda x: x.fitness)

            # Checkpoint
            if (gen + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(best_agent, gen)
                self._save_history(gen)

        # FINAL SUMMARY - Use last evaluated population (not new unevaluated one)
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        final_ranking = sorted(last_evaluated_population, key=lambda x: x.fitness, reverse=True)

        print("\nFinal Top 5:")
        for i, agent in enumerate(final_ranking[:5]):
            print(f"\n{i+1}. Agent {agent.id} ({agent.architecture_type})")
            print(f"   Fitness: {agent.fitness:.4f}")
            print(f"   Sally-Anne: {agent.sally_anne_score:.4f}")
            print(f"   Zombie Det: {agent.zombie_detection_score:.4f}")
            print(f"   Higher-Order: {agent.higher_order_scores}")

        print("\nBest Per Species:")
        for arch_type, agent in self.best_per_species.items():
            print(f"  {arch_type}: fitness={agent.fitness:.4f}, sally={agent.sally_anne_score:.4f}")

        results = {
            'best_fitness': best_overall,
            'best_architecture': best_agent.architecture_type if best_agent else None,
            'generations': num_generations,
            'fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'sally_anne_history': self.sally_anne_history,
            'zombie_history': self.zombie_history,
            'species_history': self.species_history,
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

        self._generate_summary_report(results)

        # Generate final instrumentation report
        if self.instrumentation:
            print("\nGenerating instrumentation report...")
            self.instrumentation.generate_final_report()
            print(f"Instrumentation data saved to: {self.results_dir}/instrumentation/")

        return results

    def _save_checkpoint(self, agent: AgentIndividual, generation: int):
        """Save checkpoint"""
        if agent is None:
            return
        path = os.path.join(self.results_dir, f'best_gen_{generation}.pt')
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'architecture_type': agent.architecture_type,
            'fitness': agent.fitness,
            'sally_anne': agent.sally_anne_score,
            'zombie_detection': agent.zombie_detection_score,
            'higher_order': agent.higher_order_scores
        }, path)
        print(f"  [Checkpoint saved: gen {generation}]")

    def _save_history(self, generation: int):
        """Save training history"""
        history = {
            'generation': generation,
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'sally_anne': self.sally_anne_history,
            'zombie': self.zombie_history,
            'species': self.species_history
        }
        with open(os.path.join(self.results_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    def _generate_summary_report(self, results: Dict):
        """Generate text summary report"""
        report = f"""
ToM-NAS Training Summary
========================

Configuration:
- Population: {self.population_size} agents
- Generations: {results['generations']}
- Architectures: TRN={self.trn_count}, RSAN={self.rsan_count}, Transformer={self.transformer_count}

Results:
- Best Fitness: {results['best_fitness']:.4f}
- Best Architecture: {results['best_architecture']}

Fitness Progression:
- Gen 0: {self.best_fitness_history[0]:.4f}
- Gen {len(self.best_fitness_history)-1}: {self.best_fitness_history[-1]:.4f}
- Improvement: {(self.best_fitness_history[-1] - self.best_fitness_history[0]):.4f}

Final Species Distribution:
"""
        for arch_type, counts in self.species_history.items():
            if counts:
                report += f"- {arch_type}: {counts[-1]} agents\n"

        with open(os.path.join(self.results_dir, 'summary.txt'), 'w') as f:
            f.write(report)
        print(report)


def main():
    parser = argparse.ArgumentParser(description='ToM-NAS Coevolutionary Training (SCALED)')
    parser.add_argument('--population-size', type=int, default=24)
    parser.add_argument('--trn-count', type=int, default=8)
    parser.add_argument('--rsan-count', type=int, default=8)
    parser.add_argument('--transformer-count', type=int, default=8)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--min-species-size', type=int, default=3)
    parser.add_argument('--sally-anne-tests', type=int, default=20)
    parser.add_argument('--zombie-episodes', type=int, default=10)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--results-dir', type=str, default='coevolution_results')
    parser.add_argument('--enable-instrumentation', action='store_true',
                        help='Enable dissertation-quality instrumentation (detailed traces, motifs, etc.)')
    args = parser.parse_args()

    config = {
        'population_size': args.population_size,
        'trn_count': args.trn_count,
        'rsan_count': args.rsan_count,
        'transformer_count': args.transformer_count,
        'min_species_size': args.min_species_size,
        'sally_anne_tests': args.sally_anne_tests,
        'zombie_episodes': args.zombie_episodes,
        'checkpoint_interval': args.checkpoint_interval,
        'device': args.device,
        'results_dir': args.results_dir,
        'input_dim': 191,
        'hidden_dim': 128,
        'ontology_dim': 181,
        'num_world_agents': 6,
        'num_zombies': 2,
        'higher_order_tests': 10,
        'coop_episodes': 5,
        'survival_episodes': 5,
        'sequence_length': 10,
        'enable_instrumentation': args.enable_instrumentation
    }

    trainer = CoevolutionaryTrainer(config)
    results = trainer.train(args.generations)

    print(f"\nResults saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
