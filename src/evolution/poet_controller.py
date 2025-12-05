"""
POET Controller - Paired Open-Ended Trailblazer with Sociological Genotypes

Implements POET co-evolution where both agents and environments evolve together.
Unlike standard POET, our environments are not just physical obstacles but
**sociological structures** - institutions with norms, power differentials,
and aesthetic qualities that create evolutionary pressure for Theory of Mind.

The Environment as Sociological Crucible:
- Genotype: Vector defining position in 80-Dimension Taxonomy
- The "Red Queen" Dynamics: Environments get harder, agents get smarter
- Institutional Friction: High friction forces deep recursive ToM

Key Innovation:
Standard POET evolves terrain difficulty. We evolve INSTITUTIONAL difficulty -
the complexity of social norms, power structures, and deceptive dynamics
that agents must navigate.

Theoretical Foundation:
- POET (Wang et al., 2019)
- Niche Construction (Odling-Smee)
- Institutional Economics (North)
- Red Queen Hypothesis (Van Valen)

Author: ToM-NAS Project
"""

import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .fitness import CompositeFitnessFunction
from .nas_engine import EvolutionConfig, Individual, NASEngine
from .operators import ArchitectureGene


class EnvironmentType(Enum):
    """Types of sociological environments."""

    THE_HOLLOW = auto()  # Minimal institutional structure
    THE_MARKET = auto()  # Economic competition, deception
    THE_MINISTRY = auto()  # Bureaucratic complexity, hierarchy
    THE_COURT = auto()  # Adversarial, high stakes
    THE_TEMPLE = auto()  # Ritual, orthodoxy, sacred norms
    THE_ACADEMY = auto()  # Knowledge asymmetry, credentialism
    THE_FAMILY = auto()  # Kinship obligations, loyalty
    CUSTOM = auto()  # Custom taxonomy position


@dataclass
class EnvironmentGenotype:
    """
    Genotype for a sociological environment.

    The environment is positioned in the 80-Dimension Taxonomy space,
    defining its institutional, mundane, and aesthetic qualities.
    """

    env_id: str
    env_type: EnvironmentType

    # 80-dimensional taxonomy position
    taxonomy_position: np.ndarray = field(default_factory=lambda: np.random.uniform(0, 1, 80))

    # Institutional parameters (derived from taxonomy or set directly)
    institutional_friction: float = 0.5  # How complex the norms are
    power_differential: float = 0.5  # Inequality between agents
    deception_pressure: float = 0.5  # Incentive to deceive
    norm_rigidity: float = 0.5  # How strict norm enforcement is
    information_asymmetry: float = 0.5  # Knowledge gaps between agents
    coalition_dynamics: float = 0.5  # Importance of alliances
    temporal_pressure: float = 0.5  # Time constraints on decisions
    stakes_level: float = 0.5  # Consequences of failure

    # Aesthetic qualities (affect perception and signaling)
    aesthetic_complexity: float = 0.5
    symbolic_density: float = 0.5

    # Evolution metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)

    # Agents that have solved this environment
    solved_by: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.env_id == "":
            self.env_id = f"env_{datetime.now().timestamp():.0f}"

    def extract_institutional_params(self):
        """Extract institutional parameters from taxonomy position."""
        # Institutional dimensions are 28-54 in the taxonomy
        inst_dims = self.taxonomy_position[27:54]

        self.institutional_friction = float(np.mean(inst_dims))
        self.power_differential = float(inst_dims[6])  # Economic_Hierarchy
        self.norm_rigidity = float(inst_dims[9])  # Enforcement_Intensity

        # Derive other parameters
        self.deception_pressure = float(1.0 - inst_dims[3])  # Inverse of Transparency
        self.information_asymmetry = float(1.0 - inst_dims[16])  # Inverse of Knowledge_Access

    def mutate(self, mutation_rate: float = 0.1) -> "EnvironmentGenotype":
        """Create mutated copy of environment genotype."""
        child = deepcopy(self)
        child.env_id = f"env_{datetime.now().timestamp():.0f}"
        child.parent_ids = [self.env_id]
        child.generation = self.generation + 1
        child.fitness_history = []
        child.solved_by = []

        # Mutate taxonomy position
        for i in range(80):
            if random.random() < mutation_rate:
                # Gaussian mutation
                child.taxonomy_position[i] += np.random.normal(0, 0.1)
                child.taxonomy_position[i] = np.clip(child.taxonomy_position[i], 0, 1)

        # Re-extract institutional parameters
        child.extract_institutional_params()

        return child

    def crossover(self, other: "EnvironmentGenotype") -> "EnvironmentGenotype":
        """Create child environment from two parents."""
        child = EnvironmentGenotype(
            env_id=f"env_{datetime.now().timestamp():.0f}",
            env_type=EnvironmentType.CUSTOM,
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.env_id, other.env_id],
        )

        # Uniform crossover on taxonomy dimensions
        for i in range(80):
            if random.random() < 0.5:
                child.taxonomy_position[i] = self.taxonomy_position[i]
            else:
                child.taxonomy_position[i] = other.taxonomy_position[i]

        child.extract_institutional_params()
        return child

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "env_id": self.env_id,
            "env_type": self.env_type.name,
            "taxonomy_position": self.taxonomy_position.tolist(),
            "institutional_friction": self.institutional_friction,
            "power_differential": self.power_differential,
            "deception_pressure": self.deception_pressure,
            "norm_rigidity": self.norm_rigidity,
            "generation": self.generation,
            "solved_by": self.solved_by,
        }


# Preset environments for bootstrapping evolution
PRESET_ENVIRONMENTS = {
    EnvironmentType.THE_HOLLOW: {
        # Minimal institutional structure - baseline
        28: 0.2,  # Authority_Structure: Decentralized
        29: 0.3,  # Legitimacy_Basis: Informal
        30: 0.3,  # Participation_Level: Low
        37: 0.2,  # Enforcement_Intensity: Lax
        34: 0.3,  # Economic_Hierarchy: Egalitarian
    },
    EnvironmentType.THE_MARKET: {
        # Economic competition, strategic deception
        31: 0.3,  # Transparency: Low
        32: 0.9,  # Market_Integration: High
        34: 0.7,  # Economic_Hierarchy: Stratified
        35: 0.9,  # Exchange_Mode: Transactional
        25: 0.8,  # Transaction_Complexity: High
    },
    EnvironmentType.THE_MINISTRY: {
        # Bureaucratic complexity, hierarchy
        28: 0.9,  # Authority_Structure: Centralized
        29: 0.9,  # Legitimacy_Basis: Rational-Legal
        36: 0.9,  # Legal_Formality: Codified
        37: 0.9,  # Enforcement_Intensity: Strict
        47: 0.8,  # Canon_Rigidity: Fixed
    },
    EnvironmentType.THE_COURT: {
        # Adversarial, high stakes
        28: 0.9,  # Authority_Structure: Centralized (Judge)
        36: 0.95,  # Legal_Formality: Highly codified
        37: 0.9,  # Enforcement_Intensity: Strict
        38: 0.5,  # Justice_Orientation: Mixed
        68: 0.9,  # Tension_Level: High
    },
    EnvironmentType.THE_TEMPLE: {
        # Sacred, orthodox, ritual-dense
        40: 0.95,  # Sacred_Presence: High
        41: 0.9,  # Ritual_Density: Elaborate
        42: 0.9,  # Orthodoxy: Strict
        43: 0.9,  # Transcendence_Orientation: High
        62: 0.9,  # Meaning_Saturation: Symbolic
    },
    EnvironmentType.THE_ACADEMY: {
        # Knowledge asymmetry, credentialism
        44: 0.7,  # Knowledge_Access: Moderate
        45: 0.7,  # Pedagogy_Mode: Critical
        46: 0.9,  # Credential_Importance: High
        63: 0.8,  # Reference_Density: Allusive
        78: 0.8,  # Taste_Marker: Highbrow
    },
    EnvironmentType.THE_FAMILY: {
        # Kinship, loyalty, obligation
        48: 0.8,  # Kinship_Structure: Extended
        49: 0.6,  # Authority_Pattern: Mixed
        50: 0.7,  # Marital_Norms: Moderate
        51: 0.9,  # Intergenerational_Obligation: Strong
        21: 0.9,  # Relationship_Depth: Intimate
    },
}


def create_preset_environment(env_type: EnvironmentType) -> EnvironmentGenotype:
    """Create an environment from a preset."""
    env = EnvironmentGenotype(
        env_id=f"preset_{env_type.name.lower()}",
        env_type=env_type,
    )

    # Apply preset taxonomy positions
    if env_type in PRESET_ENVIRONMENTS:
        for dim_id, value in PRESET_ENVIRONMENTS[env_type].items():
            env.taxonomy_position[dim_id - 1] = value

    env.extract_institutional_params()
    return env


@dataclass
class AgentEnvironmentPair:
    """A paired agent-environment for POET."""

    agent: Individual
    environment: EnvironmentGenotype
    fitness: float = 0.0
    tom_depth_achieved: int = 1
    transfer_history: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class POETConfig:
    """Configuration for POET evolution."""

    # Population sizes
    num_environments: int = 5
    num_agents_per_env: int = 4

    # Evolution parameters
    generations: int = 100
    env_mutation_rate: float = 0.2
    agent_mutation_rate: float = 0.1
    transfer_threshold: float = 0.7  # Min fitness for transfer
    novelty_threshold: float = 0.3  # Min novelty for new env

    # POET specific
    enable_transfer: bool = True  # Cross-environment transfer
    enable_env_evolution: bool = True  # Environment evolution

    # Evaluation
    eval_episodes: int = 10
    tom_depth_bonus: float = 0.1  # Bonus per ToM level

    # Computational limits
    max_parallel_evals: int = 4


class POETController:
    """
    POET Controller for co-evolving agents and sociological environments.

    Implements the paired open-ended evolution where:
    1. Agents evolve to solve increasingly complex social environments
    2. Environments evolve to challenge agents at the edge of their ability
    3. Transfer allows agents to jump between compatible environments
    """

    def __init__(self, config: POETConfig, nas_engine: NASEngine, world, belief_network):
        """
        Initialize POET controller.

        Args:
            config: POET configuration
            nas_engine: NAS engine for agent evolution
            world: Simulation world
            belief_network: Belief network for ToM tracking
        """
        self.config = config
        self.nas_engine = nas_engine
        self.world = world
        self.belief_network = belief_network

        # Environment population
        self.environments: List[EnvironmentGenotype] = []

        # Agent-environment pairs
        self.pairs: Dict[str, AgentEnvironmentPair] = {}

        # Archive of solved environments
        self.solved_environments: List[EnvironmentGenotype] = []

        # Evolution history
        self.generation = 0
        self.history = {
            "best_fitness_per_env": [],
            "env_complexity_progression": [],
            "tom_depth_progression": [],
            "transfer_events": [],
            "novelty_scores": [],
        }

        # Fitness evaluator (will be adapted per environment)
        self.base_fitness_fn = nas_engine.fitness_fn

    def initialize(self):
        """Initialize POET with bootstrap environments and agents."""
        print("\n" + "=" * 60)
        print("POET Initialization")
        print("=" * 60)

        # Create initial environments from presets
        print(f"Creating {self.config.num_environments} initial environments...")

        preset_types = list(EnvironmentType)[:-1]  # Exclude CUSTOM
        for i in range(self.config.num_environments):
            env_type = preset_types[i % len(preset_types)]
            env = create_preset_environment(env_type)
            env.generation = 0
            self.environments.append(env)
            print(f"  Created: {env.env_id} ({env_type.name})")

        # Initialize NAS population if needed
        if not self.nas_engine.population:
            self.nas_engine.initialize_population()

        # Create initial agent-environment pairs
        print(f"\nPairing agents with environments...")
        agents_per_env = min(self.config.num_agents_per_env, len(self.nas_engine.population) // len(self.environments))

        agent_idx = 0
        for env in self.environments:
            for _ in range(agents_per_env):
                if agent_idx < len(self.nas_engine.population):
                    agent = self.nas_engine.population[agent_idx]
                    pair = AgentEnvironmentPair(agent=agent, environment=env)
                    pair_id = f"{agent.gene.gene_dict['arch_type']}_{env.env_id}"
                    self.pairs[pair_id] = pair
                    agent_idx += 1

        print(f"  Created {len(self.pairs)} agent-environment pairs")
        print("=" * 60)

    def evaluate_pair(self, pair: AgentEnvironmentPair) -> Tuple[float, int]:
        """
        Evaluate an agent-environment pair.

        Returns:
            Tuple of (fitness, tom_depth_achieved)
        """
        # Adapt fitness function to environment
        env = pair.environment
        agent = pair.agent

        # Environment-adjusted fitness weights
        fitness_weights = self._compute_env_fitness_weights(env)

        # Run evaluation
        results = self.base_fitness_fn.evaluate(agent.model, num_episodes=self.config.eval_episodes)

        base_fitness = results["total_fitness"]

        # Apply environment modifiers
        modified_fitness = base_fitness

        # Penalty for low ToM in high-friction environments
        if env.institutional_friction > 0.7:
            tom_penalty = max(0, 0.7 - results.get("belief_accuracy", 0.5))
            modified_fitness -= tom_penalty * 0.2

        # Bonus for handling power differentials
        if env.power_differential > 0.6:
            power_bonus = results.get("cooperation_score", 0.5) * 0.1
            modified_fitness += power_bonus

        # Estimate ToM depth achieved
        tom_depth = self._estimate_tom_depth(agent.model, env)

        # ToM depth bonus
        modified_fitness += tom_depth * self.config.tom_depth_bonus

        return modified_fitness, tom_depth

    def _compute_env_fitness_weights(self, env: EnvironmentGenotype) -> Dict[str, float]:
        """Compute fitness component weights based on environment."""
        weights = {
            "cooperation_score": 0.2,
            "belief_accuracy": 0.3,
            "zombie_detection": 0.2,
            "communication_quality": 0.15,
            "resource_efficiency": 0.1,
            "behavioral_consistency": 0.05,
        }

        # High deception pressure -> emphasize belief accuracy
        if env.deception_pressure > 0.6:
            weights["belief_accuracy"] += 0.1
            weights["cooperation_score"] -= 0.1

        # High norm rigidity -> emphasize consistency
        if env.norm_rigidity > 0.7:
            weights["behavioral_consistency"] += 0.1
            weights["resource_efficiency"] -= 0.1

        return weights

    def _estimate_tom_depth(self, model: nn.Module, env: EnvironmentGenotype) -> int:
        """Estimate the ToM depth an agent achieves in an environment."""
        # In full implementation, would analyze belief recursion traces
        # For now, use heuristic based on environment complexity and model architecture

        base_depth = 1

        # Higher friction environments require deeper ToM
        if env.institutional_friction > 0.7:
            base_depth += 1
        if env.deception_pressure > 0.7:
            base_depth += 1

        # Cap based on model recursion capability
        # (Would check model's max_recursion parameter)

        return min(base_depth, 5)

    def evolve_environments(self):
        """Evolve the environment population."""
        if not self.config.enable_env_evolution:
            return

        print("\n  Evolving environments...")

        # Sort environments by how many agents solved them
        env_difficulty = []
        for env in self.environments:
            num_solved = len(env.solved_by)
            difficulty = 1.0 / (num_solved + 1)  # Higher = fewer solvers = harder
            env_difficulty.append((env, difficulty))

        env_difficulty.sort(key=lambda x: x[1], reverse=True)

        # Keep hardest environments (they're still challenging)
        # Replace easy environments with mutations of harder ones
        num_to_replace = max(1, len(self.environments) // 3)

        for i in range(num_to_replace):
            if i < len(env_difficulty) - num_to_replace:
                # Mutate a hard environment
                parent_env = env_difficulty[i][0]
                child_env = parent_env.mutate(self.config.env_mutation_rate)

                # Increase difficulty slightly
                child_env.institutional_friction = min(1.0, parent_env.institutional_friction + 0.1)

                # Check novelty
                novelty = self._compute_env_novelty(child_env)
                if novelty > self.config.novelty_threshold:
                    # Replace an easy environment
                    easy_idx = -(i + 1)
                    self.environments[easy_idx] = child_env
                    print(f"    New environment: {child_env.env_id} (novelty: {novelty:.3f})")

    def _compute_env_novelty(self, env: EnvironmentGenotype) -> float:
        """Compute novelty of an environment compared to existing ones."""
        if not self.environments:
            return 1.0

        min_distance = float("inf")
        for existing in self.environments:
            distance = np.linalg.norm(env.taxonomy_position - existing.taxonomy_position)
            min_distance = min(min_distance, distance)

        # Normalize by maximum possible distance
        max_distance = np.sqrt(80)  # 80 dimensions, each 0-1
        novelty = min_distance / max_distance

        return novelty

    def attempt_transfers(self):
        """Attempt to transfer successful agents between environments."""
        if not self.config.enable_transfer:
            return

        print("\n  Attempting transfers...")
        transfers = []

        # Find high-performing agents
        successful_pairs = [
            (pair_id, pair) for pair_id, pair in self.pairs.items() if pair.fitness > self.config.transfer_threshold
        ]

        for pair_id, source_pair in successful_pairs:
            # Try transferring to other environments
            for target_env in self.environments:
                if target_env.env_id == source_pair.environment.env_id:
                    continue

                # Evaluate agent in target environment
                transfer_pair = AgentEnvironmentPair(agent=source_pair.agent, environment=target_env)
                transfer_fitness, tom_depth = self.evaluate_pair(transfer_pair)

                # Check if transfer is beneficial
                if transfer_fitness > source_pair.fitness * 0.9:
                    transfers.append(
                        {
                            "agent": pair_id,
                            "from_env": source_pair.environment.env_id,
                            "to_env": target_env.env_id,
                            "original_fitness": source_pair.fitness,
                            "transfer_fitness": transfer_fitness,
                        }
                    )

                    print(f"    Transfer: {pair_id} -> {target_env.env_id} " f"(fitness: {transfer_fitness:.3f})")

        self.history["transfer_events"].append(transfers)
        return transfers

    def run_generation(self):
        """Run one generation of POET evolution."""
        print(f"\n{'='*60}")
        print(f"POET Generation {self.generation}")
        print("=" * 60)

        # 1. Evaluate all pairs
        print("\n  Evaluating pairs...")
        all_fitnesses = []
        all_tom_depths = []

        for pair_id, pair in self.pairs.items():
            fitness, tom_depth = self.evaluate_pair(pair)
            pair.fitness = fitness
            pair.tom_depth_achieved = tom_depth
            all_fitnesses.append(fitness)
            all_tom_depths.append(tom_depth)

            if fitness > self.config.transfer_threshold:
                pair.environment.solved_by.append(pair_id)

        print(f"    Average fitness: {np.mean(all_fitnesses):.4f}")
        print(f"    Average ToM depth: {np.mean(all_tom_depths):.2f}")

        # Record history
        self.history["best_fitness_per_env"].append(max(all_fitnesses))
        self.history["tom_depth_progression"].append(np.mean(all_tom_depths))

        # 2. Evolve agents (using NAS engine)
        print("\n  Evolving agents...")
        self.nas_engine.evolve_generation()

        # Update pairs with evolved agents
        for i, agent in enumerate(self.nas_engine.population):
            for pair_id, pair in self.pairs.items():
                if pair.agent.gene.gene_dict == agent.gene.gene_dict:
                    pair.agent = agent
                    break

        # 3. Evolve environments
        self.evolve_environments()

        # 4. Attempt transfers
        self.attempt_transfers()

        # 5. Archive solved environments
        for env in self.environments:
            if len(env.solved_by) > self.config.num_agents_per_env // 2:
                if env not in self.solved_environments:
                    self.solved_environments.append(env)
                    print(f"    Environment solved: {env.env_id}")

        # Record environment complexity
        avg_friction = np.mean([e.institutional_friction for e in self.environments])
        self.history["env_complexity_progression"].append(avg_friction)

        self.generation += 1

    def run(self, num_generations: Optional[int] = None):
        """Run POET evolution."""
        if num_generations is None:
            num_generations = self.config.generations

        print("\n" + "=" * 60)
        print("POET Evolution Starting")
        print("=" * 60)
        print(f"Environments:     {self.config.num_environments}")
        print(f"Agents per env:   {self.config.num_agents_per_env}")
        print(f"Generations:      {num_generations}")
        print(f"Transfer enabled: {self.config.enable_transfer}")
        print("=" * 60)

        # Initialize if needed
        if not self.environments:
            self.initialize()

        # Evolution loop
        for gen in range(num_generations):
            self.run_generation()

            # Report every 10 generations
            if (gen + 1) % 10 == 0:
                self._print_progress_report()

        print("\n" + "=" * 60)
        print("POET Evolution Complete!")
        print("=" * 60)
        self._print_final_report()

    def _print_progress_report(self):
        """Print progress report."""
        print(f"\n  --- Progress Report (Gen {self.generation}) ---")
        print(f"  Best fitness:        {max(self.history['best_fitness_per_env'][-10:]):.4f}")
        print(f"  Avg ToM depth:       {np.mean(self.history['tom_depth_progression'][-10:]):.2f}")
        print(f"  Env complexity:      {self.history['env_complexity_progression'][-1]:.3f}")
        print(f"  Environments solved: {len(self.solved_environments)}")

    def _print_final_report(self):
        """Print final evolution report."""
        print(f"Total generations:     {self.generation}")
        print(f"Best fitness achieved: {max(self.history['best_fitness_per_env']):.4f}")
        print(f"Max ToM depth:         {max(self.history['tom_depth_progression']):.0f}")
        print(f"Environments solved:   {len(self.solved_environments)}")
        print(f"Total transfers:       {sum(len(t) for t in self.history['transfer_events'])}")

        # Best agent
        best_pair = max(self.pairs.values(), key=lambda p: p.fitness)
        print(f"\nBest agent:")
        print(f"  Architecture: {best_pair.agent.gene.gene_dict['arch_type']}")
        print(f"  Fitness:      {best_pair.fitness:.4f}")
        print(f"  ToM depth:    {best_pair.tom_depth_achieved}")
        print(f"  Environment:  {best_pair.environment.env_id}")

    def get_best_agent(self) -> Optional[Individual]:
        """Get the best performing agent."""
        if not self.pairs:
            return None
        best_pair = max(self.pairs.values(), key=lambda p: p.fitness)
        return best_pair.agent

    def get_hardest_environment(self) -> Optional[EnvironmentGenotype]:
        """Get the most difficult environment (fewest solvers)."""
        if not self.environments:
            return None
        return min(self.environments, key=lambda e: len(e.solved_by))

    def save_state(self, filepath: str):
        """Save POET state to file."""
        state = {
            "generation": self.generation,
            "environments": [e.to_dict() for e in self.environments],
            "solved_environments": [e.to_dict() for e in self.solved_environments],
            "history": self.history,
            "config": {
                "num_environments": self.config.num_environments,
                "num_agents_per_env": self.config.num_agents_per_env,
                "generations": self.config.generations,
            },
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        print(f"POET state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load POET state from file."""
        with open(filepath, "r") as f:
            state = json.load(f)

        self.generation = state["generation"]
        self.history = state["history"]

        # Reconstruct environments
        self.environments = []
        for env_dict in state["environments"]:
            env = EnvironmentGenotype(
                env_id=env_dict["env_id"],
                env_type=EnvironmentType[env_dict["env_type"]],
                taxonomy_position=np.array(env_dict["taxonomy_position"]),
                generation=env_dict["generation"],
            )
            env.institutional_friction = env_dict["institutional_friction"]
            env.power_differential = env_dict["power_differential"]
            env.deception_pressure = env_dict["deception_pressure"]
            env.norm_rigidity = env_dict["norm_rigidity"]
            self.environments.append(env)

        print(f"POET state loaded from {filepath}")
        print(f"Restored generation {self.generation} with {len(self.environments)} environments")
