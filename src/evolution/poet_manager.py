"""
POET Manager: Paired Open-Ended Trailblazer for ToM-NAS
=======================================================

Implements POET co-evolution where:
- Agent Population: Neural architectures optimized for ToM
- Environment Population: Institutional configurations with varying friction

Key Insight: The environment is not terrain—it is SOCIAL STRUCTURE.
As agents master low-friction institutions (Family), they migrate to
higher-friction institutions (Workplace → Political → Adversarial).

Author: ToM-NAS Project
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import logging
import json
import uuid
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import copy

from ..simulation_config import (
    SimulationConfig,
    InstitutionGenotype,
    InstitutionType,
    POETConfig,
    NASConfig,
)
from ..core.beliefs import BeliefNetwork
from ..core.metamind import (
    MetaMindPipeline,
    BeliefNest,
    Observation,
    InstitutionalContext,
    ActionCandidate,
    create_metamind_pipeline,
)
from ..evaluation.situated_evaluator import SimulationState, AgentGroundTruth

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT GENOTYPE
# =============================================================================

@dataclass
class AgentGenotype:
    """
    Genetic representation of an agent's architecture.

    This encodes the neural architecture choices for ToM modules,
    optimized via NAS within the POET framework.
    """
    genotype_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Architecture parameters
    hidden_dim: int = 128
    num_attention_heads: int = 4
    num_tom_layers: int = 3
    max_tom_depth: int = 5

    # Module configuration
    intent_module_size: int = 64
    belief_module_size: int = 128
    emotion_module_size: int = 32
    norm_module_size: int = 64
    prediction_module_size: int = 64

    # Cross-module connectivity
    intent_belief_connection: float = 0.5
    belief_emotion_connection: float = 0.3
    norm_response_connection: float = 0.7

    # Learning parameters
    learning_rate: float = 0.001
    dropout_rate: float = 0.1

    # Fitness tracking
    fitness_history: List[float] = field(default_factory=list)
    environments_solved: List[str] = field(default_factory=list)
    generation: int = 0

    def mutate(self, mutation_rate: float = 0.1) -> 'AgentGenotype':
        """Create a mutated copy of this genotype."""
        new = copy.deepcopy(self)
        new.genotype_id = str(uuid.uuid4())[:8]
        new.generation = self.generation + 1
        new.fitness_history = []
        new.environments_solved = []

        # Mutate architecture parameters
        if np.random.random() < mutation_rate:
            new.hidden_dim = max(32, new.hidden_dim + np.random.randint(-32, 33))
        if np.random.random() < mutation_rate:
            new.num_attention_heads = max(1, min(16, new.num_attention_heads + np.random.randint(-2, 3)))
        if np.random.random() < mutation_rate:
            new.num_tom_layers = max(1, min(8, new.num_tom_layers + np.random.randint(-1, 2)))

        # Mutate module sizes
        if np.random.random() < mutation_rate:
            new.intent_module_size = max(16, new.intent_module_size + np.random.randint(-16, 17))
        if np.random.random() < mutation_rate:
            new.belief_module_size = max(32, new.belief_module_size + np.random.randint(-32, 33))
        if np.random.random() < mutation_rate:
            new.emotion_module_size = max(16, new.emotion_module_size + np.random.randint(-8, 9))

        # Mutate connections
        if np.random.random() < mutation_rate:
            new.intent_belief_connection = np.clip(
                new.intent_belief_connection + np.random.randn() * 0.1, 0, 1
            )
        if np.random.random() < mutation_rate:
            new.belief_emotion_connection = np.clip(
                new.belief_emotion_connection + np.random.randn() * 0.1, 0, 1
            )

        # Mutate learning parameters
        if np.random.random() < mutation_rate:
            new.learning_rate = np.clip(
                new.learning_rate * np.exp(np.random.randn() * 0.2), 1e-5, 0.1
            )

        return new

    def crossover(self, other: 'AgentGenotype') -> 'AgentGenotype':
        """Create offspring from two parent genotypes."""
        child = AgentGenotype(
            genotype_id=str(uuid.uuid4())[:8],
            generation=max(self.generation, other.generation) + 1,
        )

        # Uniform crossover for each parameter
        child.hidden_dim = self.hidden_dim if np.random.random() < 0.5 else other.hidden_dim
        child.num_attention_heads = self.num_attention_heads if np.random.random() < 0.5 else other.num_attention_heads
        child.num_tom_layers = self.num_tom_layers if np.random.random() < 0.5 else other.num_tom_layers
        child.max_tom_depth = self.max_tom_depth if np.random.random() < 0.5 else other.max_tom_depth

        child.intent_module_size = self.intent_module_size if np.random.random() < 0.5 else other.intent_module_size
        child.belief_module_size = self.belief_module_size if np.random.random() < 0.5 else other.belief_module_size
        child.emotion_module_size = self.emotion_module_size if np.random.random() < 0.5 else other.emotion_module_size

        # Blend continuous parameters
        alpha = np.random.random()
        child.intent_belief_connection = alpha * self.intent_belief_connection + (1 - alpha) * other.intent_belief_connection
        child.belief_emotion_connection = alpha * self.belief_emotion_connection + (1 - alpha) * other.belief_emotion_connection
        child.learning_rate = np.exp(alpha * np.log(self.learning_rate) + (1 - alpha) * np.log(other.learning_rate))

        return child

    def get_parameter_count(self) -> int:
        """Estimate total parameter count."""
        total = 0
        total += self.hidden_dim * self.hidden_dim * self.num_tom_layers
        total += self.intent_module_size * self.hidden_dim
        total += self.belief_module_size * self.hidden_dim * 2  # Larger for beliefs
        total += self.emotion_module_size * self.hidden_dim
        total += self.norm_module_size * self.hidden_dim
        total += self.prediction_module_size * self.hidden_dim
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "genotype_id": self.genotype_id,
            "hidden_dim": self.hidden_dim,
            "num_attention_heads": self.num_attention_heads,
            "num_tom_layers": self.num_tom_layers,
            "max_tom_depth": self.max_tom_depth,
            "intent_module_size": self.intent_module_size,
            "belief_module_size": self.belief_module_size,
            "emotion_module_size": self.emotion_module_size,
            "learning_rate": self.learning_rate,
            "generation": self.generation,
            "parameter_count": self.get_parameter_count(),
        }


# =============================================================================
# AGENT-ENVIRONMENT PAIR
# =============================================================================

@dataclass
class AgentEnvironmentPair:
    """
    A pairing of an agent genotype with an environment genotype.
    This is the fundamental unit of POET evolution.
    """
    pair_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: AgentGenotype = field(default_factory=AgentGenotype)
    environment: InstitutionGenotype = field(default_factory=lambda: InstitutionGenotype(
        institution_type=InstitutionType.FAMILY,
        complexity_level=0.2,
        information_asymmetry=0.1,
        deception_prevalence=0.05,
        role_hierarchy_depth=1,
        role_power_differential=0.1,
    ))

    # Performance tracking
    current_fitness: float = 0.0
    best_fitness: float = 0.0
    evaluations: int = 0
    stagnation_counter: int = 0

    # History
    fitness_history: List[Tuple[int, float]] = field(default_factory=list)

    def update_fitness(self, fitness: float, generation: int):
        """Update fitness tracking."""
        self.current_fitness = fitness
        self.best_fitness = max(self.best_fitness, fitness)
        self.evaluations += 1
        self.fitness_history.append((generation, fitness))

        # Track stagnation
        if len(self.fitness_history) > 10:
            recent = [f for _, f in self.fitness_history[-10:]]
            if max(recent) - min(recent) < 0.01:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0


# =============================================================================
# NOVELTY ARCHIVE
# =============================================================================

class NoveltyArchive:
    """
    Archive for tracking behavioral novelty.
    Prevents convergence to local optima by rewarding diverse behaviors.
    """

    def __init__(self, max_size: int = 500, k_nearest: int = 15):
        self.max_size = max_size
        self.k_nearest = k_nearest
        self.archive: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, behavior: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a behavior to the archive."""
        if len(self.archive) >= self.max_size:
            # Remove least novel (oldest) entry
            self.archive.pop(0)
            self.metadata.pop(0)

        self.archive.append(behavior)
        self.metadata.append(metadata or {})

    def compute_novelty(self, behavior: np.ndarray) -> float:
        """Compute novelty score for a behavior."""
        if len(self.archive) == 0:
            return 1.0

        # Compute distances to all archived behaviors
        distances = []
        for archived in self.archive:
            dist = np.linalg.norm(behavior - archived)
            distances.append(dist)

        # Average distance to k-nearest neighbors
        distances.sort()
        k = min(self.k_nearest, len(distances))
        novelty = np.mean(distances[:k])

        return novelty

    def get_archive_diversity(self) -> float:
        """Compute overall archive diversity."""
        if len(self.archive) < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i, a in enumerate(self.archive):
            for b in self.archive[i + 1:]:
                total_dist += np.linalg.norm(a - b)
                count += 1

        return total_dist / count if count > 0 else 0.0


# =============================================================================
# POET MANAGER
# =============================================================================

class POETManager:
    """
    Paired Open-Ended Trailblazer Manager.

    Orchestrates co-evolution of agents and institutional environments.

    Key Operations:
    1. EVALUATE: Run agents in environments, compute fitness
    2. TRANSFER: Move successful agents to harder environments
    3. MUTATE: Create variations of agents and environments
    4. SELECT: Keep fit agents, cull underperformers
    """

    def __init__(
        self,
        config: SimulationConfig,
        evaluator: 'SituatedEvaluator' = None,
        checkpoint_dir: str = None
    ):
        self.config = config
        self.poet_config = config.poet
        self.evaluator = evaluator

        # Populations
        self.agent_population: List[AgentGenotype] = []
        self.environment_population: List[InstitutionGenotype] = []
        self.pairs: List[AgentEnvironmentPair] = []

        # Novelty archives
        self.agent_novelty_archive = NoveltyArchive(
            max_size=self.poet_config.archive_size
        )
        self.environment_novelty_archive = NoveltyArchive(
            max_size=self.poet_config.archive_size
        )

        # Evolution state
        self.current_generation: int = 0
        self.total_evaluations: int = 0

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.checkpoint_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats_history: List[Dict[str, Any]] = []

        # Initialize shared belief network for simulations
        self.belief_network = BeliefNetwork(
            num_agents=config.poet.agent_population_size,
            ontology_dim=64,
            max_order=config.belief_nest.max_nesting_depth,
        )

        # Context manager for norms (lazy import to avoid circular)
        self._context_manager = None

        # Initialize populations
        self._initialize_populations()

        logger.info(f"POETManager initialized with {len(self.agent_population)} agents, "
                   f"{len(self.environment_population)} environments")

    def _initialize_populations(self):
        """Initialize agent and environment populations."""
        # Create initial agents with diverse architectures
        for i in range(self.poet_config.agent_population_size):
            agent = AgentGenotype(
                hidden_dim=np.random.choice([64, 128, 256]),
                num_attention_heads=np.random.choice([2, 4, 8]),
                num_tom_layers=np.random.choice([2, 3, 4]),
                max_tom_depth=self.config.belief_nest.max_nesting_depth,
            )
            # Add random variation
            agent = agent.mutate(mutation_rate=0.5)
            self.agent_population.append(agent)

        # Create initial environments (start simple)
        institution_progression = [
            InstitutionType.FAMILY,
            InstitutionType.FRIENDSHIP,
            InstitutionType.EDUCATION,
            InstitutionType.WORKPLACE,
        ]

        for i in range(self.poet_config.environment_population_size):
            # Start with simpler institutions
            inst_idx = min(i // 5, len(institution_progression) - 1)
            inst_type = institution_progression[inst_idx]

            env = InstitutionGenotype(
                institution_type=inst_type,
                complexity_level=0.1 + (i / self.poet_config.environment_population_size) * 0.3,
                information_asymmetry=0.1 + np.random.random() * 0.2,
                deception_prevalence=0.05 + np.random.random() * 0.1,
                role_hierarchy_depth=1 + i // 10,
                role_power_differential=np.random.random() * 0.3,
            )
            self.environment_population.append(env)

        # Create initial pairings
        self._create_initial_pairs()

    def _create_initial_pairs(self):
        """Create initial agent-environment pairs."""
        # Pair each agent with a compatible environment
        for i, agent in enumerate(self.agent_population):
            # Start with simpler environments
            env_idx = i % len(self.environment_population)
            env = self.environment_population[env_idx]

            pair = AgentEnvironmentPair(agent=agent, environment=env)
            self.pairs.append(pair)

    def evolve_generation(self) -> Dict[str, Any]:
        """
        Run one generation of POET evolution.

        Returns statistics about the generation.
        """
        self.current_generation += 1
        logger.info(f"Starting generation {self.current_generation}")

        stats = {
            "generation": self.current_generation,
            "timestamp": datetime.now().isoformat(),
            "num_agents": len(self.agent_population),
            "num_environments": len(self.environment_population),
            "num_pairs": len(self.pairs),
        }

        # Step 1: Evaluate all pairs
        fitness_scores = self._evaluate_all_pairs()
        stats["mean_fitness"] = np.mean(fitness_scores)
        stats["max_fitness"] = np.max(fitness_scores)
        stats["min_fitness"] = np.min(fitness_scores)

        # Step 2: Transfer successful agents to harder environments
        transfers = self._attempt_transfers()
        stats["transfers"] = transfers

        # Step 3: Mutate populations
        new_agents, new_environments = self._mutate_populations()
        stats["new_agents"] = new_agents
        stats["new_environments"] = new_environments

        # Step 4: Selection - remove underperformers
        culled = self._selection()
        stats["culled_agents"] = culled["agents"]
        stats["culled_environments"] = culled["environments"]

        # Step 5: Update novelty archives
        self._update_novelty_archives()

        # Track statistics
        self.stats_history.append(stats)

        # Checkpoint if needed
        if self.current_generation % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        logger.info(f"Generation {self.current_generation} complete: "
                   f"mean_fitness={stats['mean_fitness']:.3f}, "
                   f"transfers={transfers}, culled={culled}")

        return stats

    def _get_context_manager(self):
        """Lazy load context manager to avoid circular imports."""
        if self._context_manager is None:
            from ..core.context_manager import ContextManager
            self._context_manager = ContextManager()
        return self._context_manager

    def _run_episode(
        self,
        agent: AgentGenotype,
        environment: InstitutionGenotype,
        num_steps: int = 20
    ) -> SimulationState:
        """
        Run a simulation episode for fitness evaluation.

        Args:
            agent: Agent genotype to evaluate
            environment: Environment to evaluate in
            num_steps: Number of simulation steps

        Returns:
            SimulationState with ground truth for evaluation
        """
        episode_id = str(uuid.uuid4())[:8]
        state = SimulationState(
            episode_id=episode_id,
            institution=environment.institution_type.value,
            timestamp=0.0,
        )

        # Create a MetaMind pipeline for this agent
        agent_idx = 0  # Use first slot for evaluation
        pipeline = create_metamind_pipeline(
            self.belief_network,
            agent_idx,
            max_hypotheses=5,
            norm_weight=self.config.metamind.norm_weight,
            social_cost_weight=self.config.metamind.social_cost_weight,
        )

        # Create ground truth for target agents (simulated other agents)
        num_other_agents = 3
        for i in range(1, num_other_agents + 1):
            state.agent_states[i] = AgentGroundTruth(
                agent_id=i,
                timestamp=0.0,
                beliefs={"environment": "accurate"},
                goals=["social_goal", "task_goal"],
                emotional_state={"valence": 0.5, "arousal": 0.3},
            )

        context_manager = self._get_context_manager()

        # Run simulation steps
        for step in range(num_steps):
            state.timestamp = float(step)

            # Create observation of another agent
            target_agent = (step % num_other_agents) + 1
            obs = Observation(
                observer_id=agent_idx,
                timestamp=float(step),
                observed_entity_id=target_agent,
                observed_entity_type="agent",
                observed_entity_name=f"Agent_{target_agent}",
                position=(np.random.randn() * 5, 0, np.random.randn() * 5),
                velocity=(np.random.randn() * 0.5, 0, np.random.randn() * 0.5),
                location_type=self._get_location_for_institution(environment.institution_type),
                institution_context=environment.institution_type.value,
            )

            # Get institutional context
            norms = context_manager.get_norms(
                location=obs.location_type,
                institution=environment.institution_type.value,
            )

            context = InstitutionalContext(
                institution_type=environment.institution_type.value,
                location=obs.location_type,
                agent_role="member",
                explicit_norms=[n.name for n in norms],
                information_asymmetry=environment.information_asymmetry,
                power_differential=environment.role_power_differential,
            )

            # Create available actions
            actions = [
                ActionCandidate(
                    action_id="observe",
                    action_type="observe",
                    expected_goal_progress=0.2,
                    expected_social_cost=0.0,
                ),
                ActionCandidate(
                    action_id="interact",
                    action_type="interact",
                    target_entity_id=target_agent,
                    expected_goal_progress=0.5,
                    expected_social_cost=0.2,
                ),
                ActionCandidate(
                    action_id="communicate",
                    action_type="speak",
                    target_entity_id=target_agent,
                    expected_goal_progress=0.3,
                    expected_social_cost=0.1,
                ),
                ActionCandidate(
                    action_id="wait",
                    action_type="wait",
                    expected_goal_progress=0.1,
                    expected_social_cost=0.0,
                ),
            ]

            # Agent makes decision
            decision = pipeline.reason(obs, "achieve_social_goal", context, actions)

            # Record event
            state.events.append({
                "step": step,
                "agent": agent_idx,
                "action": decision.selected_action.action_type,
                "tom_depth": decision.tom_depth_used,
                "confidence": decision.confidence,
                "hypotheses_count": len(decision.hypotheses_considered),
            })

            # Simulate action outcome with some stochasticity based on environment
            action_success = np.random.random() > environment.complexity_level * 0.3

            # Update ground truth based on action
            if action_success and decision.selected_action.action_type == "interact":
                # Successful interaction reveals information
                gt = state.agent_states[target_agent]
                gt.known_facts.append(f"interacted_step_{step}")

        return state

    def _get_location_for_institution(self, institution: InstitutionType) -> str:
        """Get appropriate location type for an institution."""
        location_map = {
            InstitutionType.FAMILY: "home",
            InstitutionType.FRIENDSHIP: "public",
            InstitutionType.EDUCATION: "classroom",
            InstitutionType.WORKPLACE: "office",
            InstitutionType.HEALTHCARE: "clinic",
            InstitutionType.RELIGIOUS: "church",
            InstitutionType.LEGAL: "court",
            InstitutionType.POLITICAL: "parliament",
            InstitutionType.ECONOMIC_MARKET: "store",
            InstitutionType.MILITARY: "base",
            InstitutionType.SPORTS: "field",
            InstitutionType.MEDIA: "studio",
        }
        return location_map.get(institution, "public")

    def _evaluate_all_pairs(self) -> List[float]:
        """Evaluate all agent-environment pairs."""
        fitness_scores = []

        for pair in self.pairs:
            # Run actual episode simulation
            simulation_state = self._run_episode(pair.agent, pair.environment)

            # Compute fitness using evaluator or fallback
            if self.evaluator:
                fitness = self.evaluator.evaluate(
                    agent=pair.agent,
                    environment=pair.environment,
                    belief_network=self.belief_network,
                    simulation_state=simulation_state,
                    agent_idx=0,
                )
            else:
                # Fallback: compute fitness from simulation state directly
                fitness = self._compute_fitness_from_state(pair, simulation_state)

            pair.update_fitness(fitness, self.current_generation)
            fitness_scores.append(fitness)
            self.total_evaluations += 1

            # Update agent's fitness history
            pair.agent.fitness_history.append(fitness)

        return fitness_scores

    def _compute_fitness_from_state(
        self,
        pair: AgentEnvironmentPair,
        state: SimulationState
    ) -> float:
        """
        Compute fitness from simulation state when no evaluator is provided.

        Uses the same composite weights as SituatedEvaluator.
        """
        agent = pair.agent
        env = pair.environment

        # Extract metrics from simulation events
        events = state.events
        if not events:
            return 0.3 + np.random.randn() * 0.05

        # Belief accuracy proxy: higher ToM depth usage suggests active modeling
        tom_depths = [e.get("tom_depth", 0) for e in events]
        mean_tom_depth = np.mean(tom_depths) if tom_depths else 0
        belief_score = min(mean_tom_depth / agent.max_tom_depth, 1.0)

        # Action success proxy: variety and confidence
        confidences = [e.get("confidence", 0.5) for e in events]
        action_score = np.mean(confidences) if confidences else 0.5

        # Social cost proxy: based on environment friction
        social_score = 1.0 - (env.friction_coefficient * 0.3)

        # Efficiency: appropriate architecture size
        param_count = agent.get_parameter_count()
        efficiency_score = 1.0 - min(param_count / 1_000_000, 0.5)

        # Composite (matching EvaluationConfig defaults)
        fitness = (
            0.4 * belief_score +
            0.3 * action_score +
            0.2 * social_score +
            0.1 * efficiency_score
        )

        # Add small noise
        fitness += np.random.randn() * 0.02
        return np.clip(fitness, 0, 1)

    def _attempt_transfers(self) -> int:
        """
        Attempt to transfer successful agents to harder environments.

        Returns number of successful transfers.
        """
        transfers = 0

        # Sort pairs by fitness
        sorted_pairs = sorted(self.pairs, key=lambda p: p.current_fitness, reverse=True)

        # Top performers try harder environments
        for pair in sorted_pairs[:len(sorted_pairs) // 4]:
            if pair.current_fitness > self.poet_config.migration_threshold:
                # Find a harder environment
                harder_env = self._find_harder_environment(pair.environment)
                if harder_env:
                    # Create new pair with harder environment
                    new_pair = AgentEnvironmentPair(
                        agent=pair.agent,
                        environment=harder_env,
                    )
                    self.pairs.append(new_pair)
                    pair.agent.environments_solved.append(pair.environment.institution_type.value)
                    transfers += 1

        return transfers

    def _find_harder_environment(
        self,
        current_env: InstitutionGenotype
    ) -> Optional[InstitutionGenotype]:
        """Find a harder environment than the current one."""
        # Institution difficulty ordering
        difficulty_order = [
            InstitutionType.FAMILY,
            InstitutionType.FRIENDSHIP,
            InstitutionType.EDUCATION,
            InstitutionType.WORKPLACE,
            InstitutionType.HEALTHCARE,
            InstitutionType.RELIGIOUS,
            InstitutionType.LEGAL,
            InstitutionType.POLITICAL,
            InstitutionType.ECONOMIC_MARKET,
            InstitutionType.MILITARY,
        ]

        current_idx = difficulty_order.index(current_env.institution_type) \
            if current_env.institution_type in difficulty_order else 0

        # Try next difficulty level
        if current_idx < len(difficulty_order) - 1:
            next_type = difficulty_order[current_idx + 1]

            # Find or create environment of this type
            for env in self.environment_population:
                if env.institution_type == next_type:
                    return env

            # Create new harder environment
            return InstitutionGenotype(
                institution_type=next_type,
                complexity_level=current_env.complexity_level + 0.1,
                information_asymmetry=min(1.0, current_env.information_asymmetry + 0.1),
                deception_prevalence=min(1.0, current_env.deception_prevalence + 0.1),
                role_hierarchy_depth=current_env.role_hierarchy_depth + 1,
                role_power_differential=min(1.0, current_env.role_power_differential + 0.1),
            )

        return None

    def _mutate_populations(self) -> Tuple[int, int]:
        """
        Mutate agent and environment populations.

        Returns (new_agents, new_environments).
        """
        new_agents = 0
        new_environments = 0

        # Mutate top-performing agents
        sorted_agents = sorted(
            self.agent_population,
            key=lambda a: np.mean(a.fitness_history[-10:]) if a.fitness_history else 0,
            reverse=True
        )

        for agent in sorted_agents[:len(sorted_agents) // 4]:
            if np.random.random() < self.poet_config.agent_mutation_rate:
                mutant = agent.mutate()
                self.agent_population.append(mutant)
                new_agents += 1

        # Crossover between successful agents
        if len(sorted_agents) >= 2:
            for _ in range(len(sorted_agents) // 8):
                parent1, parent2 = np.random.choice(sorted_agents[:10], 2, replace=False)
                child = parent1.crossover(parent2)
                self.agent_population.append(child)
                new_agents += 1

        # Mutate environments (less frequently)
        for env in self.environment_population:
            if np.random.random() < self.poet_config.environment_mutation_rate:
                mutant = env.mutate()
                self.environment_population.append(mutant)
                new_environments += 1

        return new_agents, new_environments

    def _selection(self) -> Dict[str, int]:
        """
        Selection: Remove underperforming agents and environments.

        Returns counts of culled agents and environments.
        """
        culled = {"agents": 0, "environments": 0}

        # Cull agents below extinction threshold
        initial_agents = len(self.agent_population)
        self.agent_population = [
            a for a in self.agent_population
            if not a.fitness_history or
            np.mean(a.fitness_history[-10:]) > self.poet_config.extinction_threshold
        ]
        culled["agents"] = initial_agents - len(self.agent_population)

        # Ensure minimum population
        while len(self.agent_population) < self.poet_config.agent_population_size // 2:
            # Replenish with mutations of survivors
            if self.agent_population:
                parent = np.random.choice(self.agent_population)
                self.agent_population.append(parent.mutate())

        # Remove stagnant pairs
        self.pairs = [p for p in self.pairs if p.stagnation_counter < 20]

        # Recreate pairs for orphaned agents
        paired_agents = {p.agent.genotype_id for p in self.pairs}
        for agent in self.agent_population:
            if agent.genotype_id not in paired_agents:
                env = np.random.choice(self.environment_population)
                self.pairs.append(AgentEnvironmentPair(agent=agent, environment=env))

        return culled

    def _update_novelty_archives(self):
        """Update novelty archives with current population behaviors."""
        # Extract behavioral descriptors for agents
        for agent in self.agent_population:
            behavior = np.array([
                agent.hidden_dim / 256,
                agent.num_attention_heads / 8,
                agent.num_tom_layers / 5,
                agent.belief_module_size / 256,
                np.mean(agent.fitness_history[-10:]) if agent.fitness_history else 0,
            ])
            self.agent_novelty_archive.add(behavior, {"id": agent.genotype_id})

        # Extract behavioral descriptors for environments
        for env in self.environment_population:
            behavior = np.array([
                env.complexity_level,
                env.information_asymmetry,
                env.deception_prevalence,
                env.role_hierarchy_depth / 5,
                env.role_power_differential,
            ])
            self.environment_novelty_archive.add(behavior)

    def _save_checkpoint(self):
        """Save current state to checkpoint."""
        checkpoint = {
            "generation": self.current_generation,
            "total_evaluations": self.total_evaluations,
            "agents": [a.to_dict() for a in self.agent_population],
            "stats_history": self.stats_history[-100:],  # Last 100 generations
        }

        path = self.checkpoint_dir / f"checkpoint_gen{self.current_generation}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")

    def get_best_agents(self, n: int = 5) -> List[AgentGenotype]:
        """Get the n best-performing agents."""
        sorted_agents = sorted(
            self.agent_population,
            key=lambda a: np.mean(a.fitness_history[-10:]) if a.fitness_history else 0,
            reverse=True
        )
        return sorted_agents[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get current POET statistics."""
        agent_fitness = [
            np.mean(a.fitness_history[-10:]) if a.fitness_history else 0
            for a in self.agent_population
        ]

        return {
            "generation": self.current_generation,
            "total_evaluations": self.total_evaluations,
            "num_agents": len(self.agent_population),
            "num_environments": len(self.environment_population),
            "num_pairs": len(self.pairs),
            "mean_agent_fitness": np.mean(agent_fitness) if agent_fitness else 0,
            "max_agent_fitness": np.max(agent_fitness) if agent_fitness else 0,
            "agent_diversity": self.agent_novelty_archive.get_archive_diversity(),
            "environment_diversity": self.environment_novelty_archive.get_archive_diversity(),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AgentGenotype",
    "AgentEnvironmentPair",
    "NoveltyArchive",
    "POETManager",
]
