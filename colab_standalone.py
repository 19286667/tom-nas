#!/usr/bin/env python
"""
ToM-NAS Complete Standalone Script for Google Colab
Just paste this entire file into a Colab cell and run!

No git clone, no setup - everything included.
"""

# ============================================================================
# CELL 1: Run this to install dependencies and execute training
# ============================================================================

print("ToM-NAS Coevolutionary Training System")
print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import random
import copy
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np

# ============================================================================
# SOUL MAP ONTOLOGY
# ============================================================================

class SoulMapOntology:
    """181-dimensional ontology for Theory of Mind representation"""

    def __init__(self, dim: int = 181):
        self.dim = dim
        self.categories = {
            'perception': (0, 30),      # What agent perceives
            'beliefs': (30, 70),        # What agent believes
            'desires': (70, 100),       # What agent wants
            'intentions': (100, 130),   # What agent plans to do
            'emotions': (130, 160),     # Emotional states
            'social': (160, 181)        # Social relationships
        }

    def get_category_slice(self, category: str) -> slice:
        start, end = self.categories[category]
        return slice(start, end)

    def encode_belief(self, belief_type: str, target: int, confidence: float) -> torch.Tensor:
        encoding = torch.zeros(self.dim)
        start, end = self.categories['beliefs']
        idx = start + (hash(belief_type) % (end - start))
        encoding[idx] = confidence
        return encoding


# ============================================================================
# BELIEF NETWORK
# ============================================================================

class BeliefNetwork:
    """Recursive belief tracking up to 5th order"""

    def __init__(self, num_agents: int, ontology_dim: int, max_order: int = 5):
        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.max_order = max_order
        self.beliefs = {}
        self._init_beliefs()

    def _init_beliefs(self):
        for order in range(1, self.max_order + 1):
            self.beliefs[order] = torch.zeros(self.num_agents, self.num_agents, self.ontology_dim)

    def update_belief(self, observer: int, target: int, order: int, belief: torch.Tensor):
        if order <= self.max_order:
            self.beliefs[order][observer, target] = belief

    def get_belief(self, observer: int, target: int, order: int) -> torch.Tensor:
        if order <= self.max_order:
            return self.beliefs[order][observer, target]
        return torch.zeros(self.ontology_dim)

    def reset(self):
        self._init_beliefs()


# ============================================================================
# AGENT ARCHITECTURES
# ============================================================================

class TransparentRNN(nn.Module):
    """Transparent RNN with interpretable belief updates"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict:
        batch_size = x.shape[0]

        projected = F.relu(self.input_proj(x))

        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        output, hidden_out = self.gru(projected, hidden)

        final_hidden = output[:, -1, :]
        beliefs = torch.sigmoid(self.belief_head(final_hidden))
        actions = torch.sigmoid(self.action_head(final_hidden))

        return {
            'beliefs': beliefs,
            'actions': actions,
            'hidden': hidden_out,
            'output': output
        }


class RecursiveSelfAttention(nn.Module):
    """Self-attention with recursive belief modeling"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.recursive_layer = nn.Linear(hidden_dim, hidden_dim)
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict:
        projected = F.relu(self.input_proj(x))
        attended, attn_weights = self.attention(projected, projected, projected)
        recursive = F.relu(self.recursive_layer(attended))

        final = recursive[:, -1, :]
        beliefs = torch.sigmoid(self.belief_head(final))
        actions = torch.sigmoid(self.action_head(final))

        return {
            'beliefs': beliefs,
            'actions': actions,
            'attention_weights': attn_weights,
            'recursive_output': recursive
        }


class TransformerToMAgent(nn.Module):
    """Transformer-based ToM agent"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.belief_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict:
        projected = F.relu(self.input_proj(x))
        transformed = self.transformer(projected)

        final = transformed[:, -1, :]
        beliefs = torch.sigmoid(self.belief_head(final))
        actions = torch.sigmoid(self.action_head(final))

        return {
            'beliefs': beliefs,
            'actions': actions,
            'transformer_output': transformed
        }


# ============================================================================
# SOCIAL WORLD
# ============================================================================

@dataclass
class WorldAgent:
    """Agent in the social world"""
    id: int
    resources: float = 100.0
    energy: float = 100.0
    is_zombie: bool = False
    beliefs: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.beliefs:
            self.beliefs = {}


class SocialWorld4:
    """Social world environment with zombies"""

    def __init__(self, num_agents: int, ontology_dim: int, num_zombies: int = 2):
        self.num_agents = num_agents
        self.ontology_dim = ontology_dim
        self.num_zombies = num_zombies
        self.timestep = 0
        self.history = []
        self.coalitions = {}
        self.next_coalition_id = 0
        self.agents = []
        self._init_agents()

    def _init_agents(self):
        self.agents = []
        zombie_ids = random.sample(range(self.num_agents), min(self.num_zombies, self.num_agents))

        for i in range(self.num_agents):
            self.agents.append(WorldAgent(
                id=i,
                resources=100.0,
                energy=100.0,
                is_zombie=(i in zombie_ids)
            ))

    def reset(self):
        """Reset world to initial state"""
        self.timestep = 0
        self.history = []
        self.coalitions = {}
        self.next_coalition_id = 0
        self._init_agents()

    def get_observation(self, agent_id: int) -> Dict:
        agent = self.agents[agent_id]
        return {
            'own_resources': agent.resources,
            'own_energy': agent.energy,
            'timestep': self.timestep,
            'num_agents': self.num_agents
        }

    def step(self, actions: List[Dict], belief_network: BeliefNetwork):
        self.timestep += 1

        for i, action in enumerate(actions):
            if i < len(self.agents):
                agent = self.agents[i]
                action_type = action.get('type', 'cooperate')

                if action_type == 'cooperate':
                    agent.resources += random.uniform(5, 15)
                    agent.energy -= 5
                else:
                    agent.resources += random.uniform(-5, 25)
                    agent.energy -= 10

                agent.energy = max(0, min(100, agent.energy))
                agent.resources = max(0, agent.resources)

        self.history.append({'timestep': self.timestep, 'actions': actions})


# ============================================================================
# INSTRUMENTATION (Simplified for Colab)
# ============================================================================

class SimpleInstrumentation:
    """Simplified instrumentation for Colab"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.traces = []
        self.generation_data = []

    def log_agent(self, agent_id: int, arch: str, fitness: float, scores: Dict):
        self.traces.append({
            'agent_id': agent_id,
            'architecture': arch,
            'fitness': fitness,
            'scores': scores
        })

    def save_generation(self, generation: int):
        data = {
            'generation': generation,
            'agents': self.traces.copy()
        }
        self.generation_data.append(data)
        self.traces = []

    def generate_final_report(self):
        report = {
            'total_generations': len(self.generation_data),
            'generations': self.generation_data
        }
        with open(os.path.join(self.output_dir, 'final_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {self.output_dir}/final_report.json")


# ============================================================================
# AGENT INDIVIDUAL
# ============================================================================

@dataclass
class AgentIndividual:
    """Represents one agent in the evolutionary population"""
    id: int
    architecture_type: str
    model: nn.Module
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
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
        self.fitness = other.fitness
        self.sally_anne_score = other.sally_anne_score
        self.zombie_detection_score = other.zombie_detection_score
        self.cooperation_score = other.cooperation_score
        self.survival_score = other.survival_score
        self.higher_order_scores = dict(other.higher_order_scores)


# ============================================================================
# COEVOLUTIONARY TRAINER
# ============================================================================

class CoevolutionaryTrainer:
    """Coevolutionary training system for ToM-NAS"""

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
        self.min_species_size = config.get('min_species_size', 2)

        # Evaluation settings
        self.sally_anne_tests = config.get('sally_anne_tests', 10)
        self.zombie_episodes = config.get('zombie_episodes', 5)
        self.higher_order_tests = config.get('higher_order_tests', 5)
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
        self.sally_anne_history = []
        self.zombie_history = []
        self.species_history = {'TRN': [], 'RSAN': [], 'Transformer': [], 'Hybrid': []}
        self.best_per_species = {}

        # Instrumentation
        self.enable_instrumentation = config.get('enable_instrumentation', False)
        self.instrumentation = None
        if self.enable_instrumentation:
            instr_dir = os.path.join(self.results_dir, 'instrumentation')
            self.instrumentation = SimpleInstrumentation(instr_dir)
            print(f"Instrumentation enabled: {instr_dir}")

    def _create_model(self, arch_type: str) -> nn.Module:
        if arch_type == 'TRN':
            return TransparentRNN(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        elif arch_type == 'RSAN':
            return RecursiveSelfAttention(self.input_dim, self.hidden_dim, self.output_dim, num_heads=4).to(self.device)
        elif arch_type == 'Transformer':
            return TransformerToMAgent(self.input_dim, self.hidden_dim, self.output_dim, num_layers=2).to(self.device)
        else:
            return RecursiveSelfAttention(self.input_dim, self.hidden_dim, self.output_dim, num_heads=4).to(self.device)

    def _initialize_population(self) -> List[AgentIndividual]:
        population = []
        agent_id = 0

        for arch_type, count in [('TRN', self.trn_count), ('RSAN', self.rsan_count), ('Transformer', self.transformer_count)]:
            for _ in range(count):
                model = self._create_model(arch_type)
                population.append(AgentIndividual(
                    id=agent_id,
                    architecture_type=arch_type,
                    model=model,
                    species_id={'TRN': 0, 'RSAN': 1, 'Transformer': 2}.get(arch_type, 3)
                ))
                agent_id += 1

        print(f"Initialized: {self.trn_count} TRN, {self.rsan_count} RSAN, {self.transformer_count} Transformer")
        return population

    def evaluate_agent(self, agent: AgentIndividual, verbose: bool = False) -> float:
        agent.model.eval()

        with torch.no_grad():
            # Sally-Anne test
            sally_scores = [self._test_sally_anne(agent) for _ in range(self.sally_anne_tests)]
            sally_score = sum(sally_scores) / len(sally_scores)
            agent.sally_anne_score = sally_score

            # Higher-order ToM
            higher_order_total = 0.0
            for order in range(1, 6):
                order_scores = [self._test_higher_order(agent, order) for _ in range(self.higher_order_tests)]
                avg_order = sum(order_scores) / len(order_scores)
                agent.higher_order_scores[order] = avg_order
                higher_order_total += avg_order * (order / 15.0)

            # Zombie detection
            zombie_scores = [self._test_zombie_detection(agent) for _ in range(self.zombie_episodes)]
            zombie_score = sum(zombie_scores) / len(zombie_scores)
            agent.zombie_detection_score = zombie_score

            # Cooperation
            coop_score = self._test_cooperation(agent)
            agent.cooperation_score = coop_score

            # Survival
            survival_score = self._test_survival(agent)
            agent.survival_score = survival_score

        # Composite fitness
        fitness = (0.30 * sally_score + 0.25 * higher_order_total +
                   0.25 * zombie_score + 0.10 * coop_score + 0.10 * survival_score)
        agent.fitness = fitness

        if self.instrumentation:
            self.instrumentation.log_agent(agent.id, agent.architecture_type, fitness, {
                'sally_anne': sally_score,
                'zombie': zombie_score,
                'higher_order': higher_order_total
            })

        if verbose:
            print(f"  Agent {agent.id} ({agent.architecture_type}): SA={sally_score:.3f} ZD={zombie_score:.3f} FIT={fitness:.4f}")

        return fitness

    def _test_sally_anne(self, agent: AgentIndividual) -> float:
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
        seq_len = order + 2
        sequence = torch.zeros(1, seq_len, self.input_dim).to(self.device)

        for i in range(order):
            sequence[0, i, 10 + i] = 1.0
            sequence[0, i, 50] = (order - i) / order

        sequence = sequence + torch.randn_like(sequence) * 0.02

        output = agent.model(sequence)
        beliefs = output['beliefs']

        expected_conf = max(0.25, 1.0 - (order - 1) * 0.15)
        actual_conf = beliefs.mean().item()

        error = abs(actual_conf - expected_conf)
        return max(0.0, 1.0 - error * 2)

    def _test_zombie_detection(self, agent: AgentIndividual) -> float:
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

    def _test_cooperation(self, agent: AgentIndividual) -> float:
        good_decisions = 0
        total_decisions = 0

        # Partner cooperated
        coop_history = torch.zeros(1, 5, self.input_dim).to(self.device)
        coop_history[0, :, 80] = 1.0
        coop_history[0, :, 81] = 0.9
        output = agent.model(coop_history)
        if output['actions'].mean().item() > 0.5:
            good_decisions += 1
        total_decisions += 1

        # Partner defected
        defect_history = torch.zeros(1, 5, self.input_dim).to(self.device)
        defect_history[0, :, 80] = 1.0
        defect_history[0, :, 81] = 0.1
        output = agent.model(defect_history)
        if output['actions'].mean().item() < 0.5:
            good_decisions += 1
        total_decisions += 1

        return good_decisions / total_decisions

    def _test_survival(self, agent: AgentIndividual) -> float:
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
        parents = []

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
        if parent1.architecture_type == parent2.architecture_type:
            child_type = parent1.architecture_type
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
        if random.random() > self.mutation_rate:
            return agent

        with torch.no_grad():
            for param in agent.model.parameters():
                if random.random() < 0.1:
                    noise = torch.randn_like(param) * 0.02
                    param.add_(noise)
        return agent

    def evolve_generation(self) -> Tuple[float, List[AgentIndividual]]:
        print(f"\n{'='*60}")
        print(f"GENERATION {self.generation}")
        print(f"{'='*60}")

        # Count species
        species_counts = {}
        for agent in self.population:
            species_counts[agent.architecture_type] = species_counts.get(agent.architecture_type, 0) + 1
        print(f"Population: {species_counts}")

        # Evaluate all agents
        print(f"\nEvaluating {len(self.population)} agents...")
        for agent in self.population:
            self.evaluate_agent(agent, verbose=True)

        # Store evaluated population
        evaluated_population = [copy.deepcopy(a) for a in self.population]

        # Statistics
        fitnesses = [a.fitness for a in self.population]
        sally_scores = [a.sally_anne_score for a in self.population]
        zombie_scores = [a.zombie_detection_score for a in self.population]

        print(f"\n{'='*60}")
        print(f"Best: {max(fitnesses):.4f} | Avg: {sum(fitnesses)/len(fitnesses):.4f}")
        print(f"Sally-Anne: {sum(sally_scores)/len(sally_scores):.4f} | Zombie: {sum(zombie_scores)/len(zombie_scores):.4f}")

        # Track history
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.sally_anne_history.append(sum(sally_scores) / len(sally_scores))
        self.zombie_history.append(sum(zombie_scores) / len(zombie_scores))

        for arch_type in ['TRN', 'RSAN', 'Transformer', 'Hybrid']:
            agents = [a for a in self.population if a.architecture_type == arch_type]
            self.species_history[arch_type].append(len(agents))
            if agents:
                best = max(agents, key=lambda x: x.fitness)
                self.best_per_species[arch_type] = best

        # Save instrumentation
        if self.instrumentation:
            self.instrumentation.save_generation(self.generation)

        # Select parents and create next generation
        parents = self.select_parents()
        new_population = []

        # Keep elites
        elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elite_count]
        for elite in elites:
            elite_copy = AgentIndividual(
                id=elite.id,
                architecture_type=elite.architecture_type,
                model=copy.deepcopy(elite.model),
                generation=elite.generation,
                species_id=elite.species_id
            )
            elite_copy.copy_scores_from(elite)
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

        # Generate offspring
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
        print("\n" + "=" * 60)
        print("ToM-NAS COEVOLUTIONARY TRAINING")
        print("=" * 60)
        print(f"Population: {self.population_size} | Generations: {num_generations}")
        print("=" * 60)

        best_overall = 0.0
        best_agent = None
        last_evaluated_population = None

        for gen in range(num_generations):
            best_gen, evaluated_pop = self.evolve_generation()
            last_evaluated_population = evaluated_pop

            if best_gen > best_overall:
                best_overall = best_gen
                best_agent = max(evaluated_pop, key=lambda x: x.fitness)

        # Final summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        final_ranking = sorted(last_evaluated_population, key=lambda x: x.fitness, reverse=True)

        print("\nTop 5 Agents:")
        for i, agent in enumerate(final_ranking[:5]):
            print(f"{i+1}. Agent {agent.id} ({agent.architecture_type}) - Fitness: {agent.fitness:.4f}")
            print(f"   Sally-Anne: {agent.sally_anne_score:.4f} | Zombie: {agent.zombie_detection_score:.4f}")

        # Save results
        results = {
            'best_fitness': best_overall,
            'best_architecture': best_agent.architecture_type if best_agent else None,
            'generations': num_generations,
            'fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'sally_anne_history': self.sally_anne_history,
            'zombie_history': self.zombie_history
        }

        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        if self.instrumentation:
            self.instrumentation.generate_final_report()

        return results


# ============================================================================
# MAIN - RUN THIS
# ============================================================================

def run_training(generations=10, population_size=12, enable_instrumentation=True):
    """Run ToM-NAS training with specified parameters"""

    config = {
        'population_size': population_size,
        'trn_count': population_size // 3,
        'rsan_count': population_size // 3,
        'transformer_count': population_size // 3,
        'min_species_size': 2,
        'sally_anne_tests': 10,
        'zombie_episodes': 5,
        'higher_order_tests': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'results_dir': 'coevolution_results',
        'input_dim': 191,
        'hidden_dim': 128,
        'ontology_dim': 181,
        'num_world_agents': 6,
        'num_zombies': 2,
        'sequence_length': 10,
        'enable_instrumentation': enable_instrumentation
    }

    print(f"Using device: {config['device']}")

    trainer = CoevolutionaryTrainer(config)
    results = trainer.train(generations)

    print(f"\nResults saved to: {config['results_dir']}/")
    return results


# Run with default settings
if __name__ == "__main__":
    results = run_training(generations=10, population_size=12, enable_instrumentation=True)
