"""
ToM-NAS System: Complete Integration
====================================

This module wires together all components defined in the Constitution:
- SimulationConfig (master configuration)
- MetaMind (3-stage cognitive pipeline)
- BeliefNest (nested belief representation)
- ContextManager (sociological database)
- POETManager (co-evolutionary optimization)
- SituatedEvaluator (belief-accuracy-based fitness)
- EnhancedGodotServer (physical grounding)

This is the main entry point for running the complete system.

Author: ToM-NAS Project
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

# Configuration (The Constitution)
from .simulation_config import (
    SimulationConfig,
    InstitutionGenotype,
    InstitutionType,
    create_minimal_config,
    create_family_scenario_config,
    create_workplace_scenario_config,
    create_adversarial_scenario_config,
    create_full_poet_config,
)

# Core cognitive architecture
from .core.beliefs import BeliefNetwork
from .core.metamind import (
    MetaMindPipeline,
    BeliefNest,
    Observation,
    InstitutionalContext,
    ActionCandidate,
    create_metamind_pipeline,
)
from .core.context_manager import ContextManager

# Evolution
from .evolution.poet_manager import POETManager, AgentGenotype

# Evaluation
from .evaluation.situated_evaluator import SituatedEvaluator, SimulationState

# Godot integration
from .godot_bridge import (
    EnhancedGodotServer,
    EnhancedServerConfig,
    create_enhanced_server,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COGNITIVE AGENT
# =============================================================================

class CognitiveAgent:
    """
    A complete cognitive agent with MetaMind reasoning.

    This class integrates:
    - BeliefNest for belief representation
    - MetaMind for decision-making
    - ContextManager for norm lookup
    """

    def __init__(
        self,
        agent_id: int,
        name: str,
        belief_network: BeliefNetwork,
        context_manager: ContextManager,
        config: SimulationConfig,
    ):
        self.agent_id = agent_id
        self.name = name
        self.config = config

        # Initialize MetaMind pipeline
        self.pipeline = create_metamind_pipeline(
            belief_network,
            agent_id,
            max_hypotheses=config.metamind.max_hypotheses,
            norm_weight=config.metamind.norm_weight,
            social_cost_weight=config.metamind.social_cost_weight,
        )

        # Context manager for norm lookup
        self.context_manager = context_manager

        # State
        self.current_institution: str = "family"
        self.current_role: str = "member"
        self.goals: List[str] = []

        # Tracking
        self.decisions_made: int = 0
        self.tom_depth_used: List[int] = []

    def perceive(self, observation: Observation):
        """Process a perception."""
        # Update beliefs based on observation
        self.pipeline.belief_nest.add_belief(
            subject=observation.observed_entity_name,
            predicate="observed_at",
            obj=str(observation.position),
            nesting_level=0,
            confidence=1.0,
            source="direct_perception",
        )

    def decide(
        self,
        observation: Observation,
        goal: str,
        available_actions: List[ActionCandidate],
    ):
        """Make a decision using MetaMind pipeline."""
        # Get institutional context
        norms = self.context_manager.get_norms(
            location=observation.location_type,
            role=self.current_role,
            institution=self.current_institution,
        )

        context = InstitutionalContext(
            institution_type=self.current_institution,
            location=observation.location_type,
            agent_role=self.current_role,
            explicit_norms=[n.name for n in norms],
        )

        # Run MetaMind pipeline
        decision = self.pipeline.reason(
            observation=observation,
            goal=goal,
            context=context,
            available_actions=available_actions,
        )

        # Track
        self.decisions_made += 1
        self.tom_depth_used.append(decision.tom_depth_used)

        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "decisions_made": self.decisions_made,
            "mean_tom_depth": sum(self.tom_depth_used) / len(self.tom_depth_used) if self.tom_depth_used else 0,
            "max_tom_depth": max(self.tom_depth_used) if self.tom_depth_used else 0,
            "pipeline_stats": self.pipeline.get_statistics(),
        }


# =============================================================================
# TOM-NAS SYSTEM
# =============================================================================

class ToMNASSystem:
    """
    The complete ToM-NAS system.

    This is the main entry point that orchestrates:
    - Multiple cognitive agents
    - POET co-evolution
    - Situated evaluation
    - Godot integration (optional)
    """

    def __init__(
        self,
        config: SimulationConfig = None,
        enable_godot: bool = False,
    ):
        self.config = config or SimulationConfig()

        # Initialize belief network (shared)
        self.belief_network = BeliefNetwork(
            num_agents=self.config.poet.agent_population_size,
            ontology_dim=64,
            max_order=self.config.belief_nest.max_nesting_depth,
        )

        # Initialize context manager
        self.context_manager = ContextManager(
            db_path=self.config.taxonomy_db_path
        )

        # Initialize evaluator
        self.evaluator = SituatedEvaluator(self.config.evaluation)

        # Initialize POET manager
        self.poet_manager = POETManager(
            config=self.config,
            evaluator=self.evaluator,
            checkpoint_dir=self.config.checkpoint_path,
        )

        # Initialize agents
        self.agents: Dict[int, CognitiveAgent] = {}

        # Godot server (optional)
        self.godot_server: Optional[EnhancedGodotServer] = None
        if enable_godot:
            self._init_godot_server()

        logger.info(f"ToM-NAS System initialized: {self.config.experiment_name}")

    def _init_godot_server(self):
        """Initialize Godot server for physical grounding."""
        server_config = EnhancedServerConfig(
            host="localhost",
            port=self.config.godot_port,
            num_agents=self.config.poet.agent_population_size,
            max_tom_order=self.config.belief_nest.max_nesting_depth,
        )

        self.godot_server = EnhancedGodotServer(
            config=server_config,
            belief_network=self.belief_network,
        )

        # Wire belief updates to agents
        self.godot_server.on_belief_update(self._on_belief_update)

        logger.info(f"Godot server initialized on port {self.config.godot_port}")

    def _on_belief_update(self, event):
        """Handle belief updates from Godot."""
        # Propagate to relevant agents
        if event.observer_id in self.agents:
            agent = self.agents[event.observer_id]
            # Agent receives belief update notification
            logger.debug(f"Agent {agent.name} received belief update about {event.target_id}")

    def create_agent(self, agent_id: int, name: str) -> CognitiveAgent:
        """Create a new cognitive agent."""
        agent = CognitiveAgent(
            agent_id=agent_id,
            name=name,
            belief_network=self.belief_network,
            context_manager=self.context_manager,
            config=self.config,
        )

        self.agents[agent_id] = agent
        logger.info(f"Created agent: {name} (id={agent_id})")

        return agent

    def run_evolution(self, num_generations: int = 100) -> Dict[str, Any]:
        """
        Run POET co-evolution for specified generations.

        Returns final statistics.
        """
        logger.info(f"Starting evolution: {num_generations} generations")

        all_stats = []
        for gen in range(num_generations):
            stats = self.poet_manager.evolve_generation()
            all_stats.append(stats)

            if gen % 10 == 0:
                logger.info(f"Generation {gen}: mean_fitness={stats['mean_fitness']:.3f}")

        # Get best agents
        best_agents = self.poet_manager.get_best_agents(5)

        return {
            "generations": num_generations,
            "final_stats": self.poet_manager.get_statistics(),
            "best_agents": [a.to_dict() for a in best_agents],
            "history": all_stats,
        }

    def run_episode(
        self,
        institution: InstitutionType,
        num_steps: int = 100,
    ) -> SimulationState:
        """
        Run a single simulation episode.

        Returns simulation state for evaluation.
        """
        import uuid

        state = SimulationState(
            episode_id=str(uuid.uuid4())[:8],
            institution=institution.value,
            timestamp=0.0,
        )

        # Create environment genotype
        env = InstitutionGenotype(
            institution_type=institution,
            complexity_level=0.5,
            information_asymmetry=0.3,
            deception_prevalence=0.2,
            role_hierarchy_depth=2,
            role_power_differential=0.4,
        )

        # Run simulation steps
        for step in range(num_steps):
            state.timestamp = float(step)

            # Each agent takes an action
            for agent in self.agents.values():
                # Create synthetic observation
                obs = Observation(
                    observer_id=agent.agent_id,
                    timestamp=float(step),
                    observed_entity_id=0,
                    observed_entity_type="environment",
                    observed_entity_name="environment",
                    position=(0, 0, 0),
                    velocity=(0, 0, 0),
                    location_type="office" if institution == InstitutionType.WORKPLACE else "home",
                    institution_context=institution.value,
                )

                # Create synthetic actions
                actions = [
                    ActionCandidate(
                        action_id="wait",
                        action_type="wait",
                        expected_goal_progress=0.1,
                        expected_social_cost=0.0,
                    ),
                    ActionCandidate(
                        action_id="interact",
                        action_type="interact",
                        expected_goal_progress=0.5,
                        expected_social_cost=0.2,
                    ),
                    ActionCandidate(
                        action_id="speak",
                        action_type="speak",
                        expected_goal_progress=0.3,
                        expected_social_cost=0.1,
                    ),
                ]

                # Agent decides
                decision = agent.decide(obs, "achieve_goal", actions)

                # Record event
                state.events.append({
                    "step": step,
                    "agent": agent.agent_id,
                    "action": decision.selected_action.action_type,
                    "tom_depth": decision.tom_depth_used,
                })

        return state

    def start_godot_server(self, blocking: bool = False):
        """Start the Godot server."""
        if self.godot_server is None:
            self._init_godot_server()

        self.godot_server.start(blocking=blocking)
        logger.info("Godot server started")

    def stop_godot_server(self):
        """Stop the Godot server."""
        if self.godot_server:
            self.godot_server.stop()
            logger.info("Godot server stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get complete system statistics."""
        agent_stats = {
            agent_id: agent.get_statistics()
            for agent_id, agent in self.agents.items()
        }

        return {
            "config": self.config.experiment_name,
            "num_agents": len(self.agents),
            "poet_stats": self.poet_manager.get_statistics(),
            "evaluator_stats": self.evaluator.get_statistics(),
            "agent_stats": agent_stats,
        }

    def save_state(self, path: str):
        """Save system state to file."""
        state = {
            "config": self.config.to_dict(),
            "statistics": self.get_statistics(),
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {path}")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_tom_nas_system(
    scenario: str = "family",
    enable_godot: bool = False,
) -> ToMNASSystem:
    """
    Create a ToM-NAS system with a preset configuration.

    Args:
        scenario: One of "minimal", "family", "workplace", "adversarial", "full"
        enable_godot: Whether to enable Godot integration

    Returns:
        Configured ToMNASSystem instance
    """
    config_map = {
        "minimal": create_minimal_config,
        "family": create_family_scenario_config,
        "workplace": create_workplace_scenario_config,
        "adversarial": create_adversarial_scenario_config,
        "full": create_full_poet_config,
    }

    config_fn = config_map.get(scenario, create_family_scenario_config)
    config = config_fn()

    return ToMNASSystem(config=config, enable_godot=enable_godot)


# =============================================================================
# DEMO/TEST
# =============================================================================

def run_demo():
    """Run a demonstration of the ToM-NAS system."""
    logging.basicConfig(level=logging.INFO)

    # Create system
    system = create_tom_nas_system(scenario="family")

    # Create some agents
    for i in range(3):
        system.create_agent(i, f"Agent_{i}")

    # Run a short episode
    state = system.run_episode(InstitutionType.FAMILY, num_steps=10)

    print(f"\nEpisode completed: {len(state.events)} events")

    # Run a few generations of evolution
    results = system.run_evolution(num_generations=5)

    print(f"\nEvolution completed:")
    print(f"  Mean fitness: {results['final_stats']['mean_agent_fitness']:.3f}")
    print(f"  Best agents: {len(results['best_agents'])}")

    # Print statistics
    stats = system.get_statistics()
    print(f"\nSystem statistics:")
    print(f"  Agents: {stats['num_agents']}")
    print(f"  POET generations: {stats['poet_stats']['generation']}")

    return system


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CognitiveAgent",
    "ToMNASSystem",
    "create_tom_nas_system",
    "run_demo",
]


if __name__ == "__main__":
    run_demo()
