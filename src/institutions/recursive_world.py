"""
Recursive World: Simulations Containing Simulations

This module implements the core recursive capability: agents can create
simulations that contain other agents, who can create their own simulations.

This recursive structure is not just a theoretical curiosity - it creates
genuine selective pressure for Theory of Mind:

1. To predict Agent B's behavior, Agent A must model B's reasoning
2. If B creates simulations, A must model B's modeling of simulated agents
3. If those simulated agents create simulations... we get true recursion

The depth of accurate recursive modeling becomes a key fitness criterion,
naturally selecting for agents with genuine Theory of Mind capabilities.

Emergent Dimensionality:
    The representational capacity of inner worlds can exceed the outer world.
    Just as human minds create richer models than raw sensory input,
    simulated agents can develop abstractions and compressions that
    make their representations "larger" than the containing simulation.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import uuid

from src.config import get_logger
from src.config.constants import SOUL_MAP_DIMS, MAX_BELIEF_ORDER

if TYPE_CHECKING:
    from src.institutions.researcher_agent import ResearcherAgent

logger = get_logger(__name__)


# Maximum recursion depth to prevent infinite loops
MAX_SIMULATION_DEPTH = 5


class SimulationPurpose(Enum):
    """Why an agent creates a simulation."""
    HYPOTHESIS_TESTING = "hypothesis_testing"
    COUNTERFACTUAL_REASONING = "counterfactual"
    PREDICTION = "prediction"
    THEORY_OF_MIND = "theory_of_mind"
    EXPLORATION = "exploration"
    META_RESEARCH = "meta_research"


@dataclass
class SimulationConfig:
    """Configuration for creating a new simulation."""
    # World parameters
    num_agents: int = 5
    timesteps: int = 100
    environment_complexity: float = 0.5

    # Agent parameters
    agent_capabilities: List[str] = field(default_factory=lambda: ["basic_reasoning"])
    allow_nested_simulation: bool = True

    # Resource limits
    max_compute_per_step: float = 10.0
    max_memory_mb: float = 128.0

    # Scientific parameters
    purpose: SimulationPurpose = SimulationPurpose.HYPOTHESIS_TESTING
    hypothesis: Optional[str] = None
    variables_to_track: List[str] = field(default_factory=list)


@dataclass
class EmergentDimensionality:
    """
    Tracks the emergent representational capacity of a simulation.

    The key insight: inner simulations can develop representations
    RICHER than the containing simulation, through abstraction and
    compression. This is not a violation of information theory -
    it's the same phenomenon that allows human minds to create
    models more detailed than their sensory input.
    """
    # Raw dimensionality (state space size)
    state_dimensions: int = 0

    # Effective dimensionality (after compression/abstraction)
    effective_dimensions: int = 0

    # Representational depth (layers of abstraction)
    abstraction_depth: int = 0

    # Recursive capacity (ability to model other modelers)
    recursive_modeling_depth: int = 0

    # Compression ratio (effective / raw)
    @property
    def compression_ratio(self) -> float:
        if self.state_dimensions == 0:
            return 0.0
        return self.effective_dimensions / self.state_dimensions

    # Emergent dimensionality score
    @property
    def emergence_score(self) -> float:
        """
        Score > 1.0 indicates emergent dimensionality exceeding the base.

        This happens when:
        - Agents develop efficient representations (compression_ratio > 1)
        - Deep abstraction hierarchies form
        - Recursive modeling creates new representational axes
        """
        base = self.compression_ratio
        abstraction_bonus = min(1.0, self.abstraction_depth * 0.1)
        recursion_bonus = min(1.0, self.recursive_modeling_depth * 0.2)
        return base + abstraction_bonus + recursion_bonus


class RecursiveSimulation:
    """
    A simulation that can contain agents who create simulations.

    This is the core of the recursive capability. Each RecursiveSimulation:
    1. Maintains its own physics/dynamics
    2. Contains agents with their own beliefs and goals
    3. Allows agents to create nested simulations
    4. Tracks emergent dimensionality metrics
    5. Provides interfaces for the parent agent to observe
    """

    def __init__(
        self,
        parent_agent: 'ResearcherAgent' = None,
        depth: int = 0,
        config: Dict[str, Any] = None,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.parent_agent = parent_agent
        self.parent_agent_id = parent_agent.id if parent_agent else None
        self.depth = depth
        self.config = SimulationConfig(**(config or {}))

        # State
        self.timestep: int = 0
        self.agents: List['SimulatedAgent'] = []
        self.nested_simulations: List['RecursiveSimulation'] = []

        # Metrics
        self.emergence = EmergentDimensionality(
            state_dimensions=SOUL_MAP_DIMS * self.config.num_agents,
        )
        self.history: List[Dict[str, Any]] = []

        # Validate depth
        if depth > MAX_SIMULATION_DEPTH:
            raise ValueError(f"Simulation depth {depth} exceeds maximum {MAX_SIMULATION_DEPTH}")

        # Initialize agents
        self._initialize_agents()

        logger.info(f"Created RecursiveSimulation {self.id} at depth {depth} with {len(self.agents)} agents")

    def _initialize_agents(self):
        """Create the initial population of simulated agents."""
        from src.institutions.researcher_agent import ResearcherAgent, ResearchDomain

        for i in range(self.config.num_agents):
            # Create agent with reduced capabilities at deeper levels
            agent = SimulatedAgent(
                agent_id=f"{self.id}_agent_{i}",
                name=f"Agent_{i}",
                simulation_depth=self.depth,
                can_create_simulations=self.config.allow_nested_simulation and self.depth < MAX_SIMULATION_DEPTH,
            )
            self.agents.append(agent)

    def step(self) -> Dict[str, Any]:
        """
        Advance the simulation by one timestep.

        Returns metrics about the simulation state.
        """
        step_results = {
            "timestep": self.timestep,
            "num_agents": len(self.agents),
            "nested_simulations": len(self.nested_simulations),
            "agent_states": [],
            "events": [],
        }

        # Each agent takes an action
        for agent in self.agents:
            # Agent observes environment
            observation = self._get_observation_for(agent)

            # Agent decides and acts
            action, new_simulation = agent.step(observation)

            # Handle nested simulation creation
            if new_simulation is not None:
                self.nested_simulations.append(new_simulation)
                step_results["events"].append({
                    "type": "simulation_created",
                    "agent": agent.id,
                    "simulation_id": new_simulation.id,
                    "depth": new_simulation.depth,
                })

            step_results["agent_states"].append({
                "agent_id": agent.id,
                "action": action,
            })

        # Step nested simulations
        for sim in self.nested_simulations:
            nested_results = sim.step()
            step_results["events"].append({
                "type": "nested_simulation_step",
                "simulation_id": sim.id,
                "results": nested_results,
            })

        # Update emergence metrics
        self._update_emergence_metrics()

        # Store history
        self.history.append(step_results)
        self.timestep += 1

        return step_results

    def _get_observation_for(self, agent: 'SimulatedAgent') -> torch.Tensor:
        """
        Generate an observation tensor for an agent.

        In a full implementation, this would include:
        - Other agents' visible states
        - Environmental features
        - Results from nested simulations
        """
        observation = torch.zeros(SOUL_MAP_DIMS + 10)  # State + context

        # Include information about other agents
        for i, other in enumerate(self.agents[:5]):  # Limit for efficiency
            if other.id != agent.id:
                observation[i] = 0.5  # Simplified: just mark presence

        # Include simulation depth context
        observation[-1] = self.depth / MAX_SIMULATION_DEPTH

        return observation

    def _update_emergence_metrics(self):
        """
        Calculate emergent dimensionality metrics.

        This measures whether the simulation has developed
        representational capacity beyond its raw state space.
        """
        # Count total abstraction depth across agents
        total_abstraction = sum(
            agent.abstraction_depth for agent in self.agents
        )

        # Count recursive modeling depth
        max_recursive_depth = 0
        for agent in self.agents:
            if agent.simulations_created:
                max_recursive_depth = max(max_recursive_depth, len(agent.simulations_created))

        # Include nested simulations
        for sim in self.nested_simulations:
            max_recursive_depth = max(max_recursive_depth, sim.depth + 1)

        # Update metrics
        self.emergence.abstraction_depth = total_abstraction // max(1, len(self.agents))
        self.emergence.recursive_modeling_depth = max_recursive_depth
        self.emergence.effective_dimensions = int(
            self.emergence.state_dimensions * self.emergence.emergence_score
        )

    def run(self, max_steps: int = None) -> Dict[str, Any]:
        """
        Run the simulation for multiple steps.

        Returns aggregated results and metrics.
        """
        max_steps = max_steps or self.config.timesteps
        results = {
            "simulation_id": self.id,
            "depth": self.depth,
            "total_steps": 0,
            "final_emergence": None,
            "agent_fitness_scores": {},
            "nested_simulation_count": 0,
        }

        for _ in range(max_steps):
            step_result = self.step()
            results["total_steps"] += 1

            # Check for early termination
            if self._should_terminate():
                break

        # Calculate final metrics
        results["final_emergence"] = {
            "state_dimensions": self.emergence.state_dimensions,
            "effective_dimensions": self.emergence.effective_dimensions,
            "emergence_score": self.emergence.emergence_score,
            "recursive_depth": self.emergence.recursive_modeling_depth,
        }

        results["nested_simulation_count"] = len(self.nested_simulations)

        # Agent fitness
        for agent in self.agents:
            results["agent_fitness_scores"][agent.id] = agent.get_fitness()

        logger.info(f"Simulation {self.id} completed: {results['total_steps']} steps, "
                    f"emergence={self.emergence.emergence_score:.2f}")

        return results

    def _should_terminate(self) -> bool:
        """Check if simulation should end early."""
        # Terminate if all agents have converged
        # (Implementation depends on specific convergence criteria)
        return False

    def get_state_for_parent(self) -> Dict[str, Any]:
        """
        Package simulation state for the parent agent to observe.

        This is how parent agents learn from their simulations.
        """
        return {
            "timestep": self.timestep,
            "emergence_score": self.emergence.emergence_score,
            "num_agents": len(self.agents),
            "nested_simulations": len(self.nested_simulations),
            "average_agent_fitness": sum(a.get_fitness() for a in self.agents) / max(1, len(self.agents)),
        }


class SimulatedAgent:
    """
    A lightweight agent that exists within a simulation.

    Simulated agents have reduced capabilities compared to full
    ResearcherAgents, but can still:
    - Process observations
    - Form beliefs
    - Take actions
    - Create nested simulations (if allowed)
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        simulation_depth: int,
        can_create_simulations: bool = True,
    ):
        self.id = agent_id
        self.name = name
        self.simulation_depth = simulation_depth
        self.can_create_simulations = can_create_simulations

        # Simplified state
        self.beliefs = torch.zeros(SOUL_MAP_DIMS)
        self.abstraction_depth = 0
        self.simulations_created: List[str] = []

        # Fitness tracking
        self.actions_taken = 0
        self.successful_predictions = 0

    def step(self, observation: torch.Tensor) -> tuple[str, Optional[RecursiveSimulation]]:
        """
        Process observation and return action + optional new simulation.
        """
        # Update beliefs (simplified)
        self.beliefs = 0.9 * self.beliefs + 0.1 * observation[:SOUL_MAP_DIMS]

        # Decide action
        action = self._decide_action(observation)
        self.actions_taken += 1

        # Possibly create simulation
        new_simulation = None
        if self.can_create_simulations and self._should_create_simulation():
            new_simulation = self._create_simulation()

        return action, new_simulation

    def _decide_action(self, observation: torch.Tensor) -> str:
        """Decide what action to take."""
        # Simplified: random action selection
        import random
        actions = ["observe", "communicate", "experiment", "simulate"]
        return random.choice(actions)

    def _should_create_simulation(self) -> bool:
        """Decide whether to create a nested simulation."""
        import random
        # Low probability, decreasing with depth
        probability = 0.01 / (self.simulation_depth + 1)
        return random.random() < probability

    def _create_simulation(self) -> RecursiveSimulation:
        """Create a nested simulation."""
        sim = RecursiveSimulation(
            parent_agent=None,  # Simulated agents don't have full identity
            depth=self.simulation_depth + 1,
            config={
                "num_agents": 3,  # Smaller nested simulations
                "timesteps": 50,
            }
        )
        self.simulations_created.append(sim.id)
        self.abstraction_depth += 1
        return sim

    def get_fitness(self) -> float:
        """Calculate agent fitness."""
        base = self.actions_taken * 0.01
        abstraction_bonus = self.abstraction_depth * 0.1
        simulation_bonus = len(self.simulations_created) * 0.2
        return min(1.0, base + abstraction_bonus + simulation_bonus)


class WorldFactory:
    """
    Factory for creating simulations with various configurations.

    Provides preset configurations for common experimental scenarios.
    """

    @staticmethod
    def create_tom_experiment(depth: int = 0) -> RecursiveSimulation:
        """Create a simulation designed to test Theory of Mind."""
        return RecursiveSimulation(
            depth=depth,
            config={
                "num_agents": 5,
                "timesteps": 200,
                "environment_complexity": 0.7,
                "allow_nested_simulation": True,
                "purpose": "theory_of_mind",
            }
        )

    @staticmethod
    def create_minimal_simulation(depth: int = 0) -> RecursiveSimulation:
        """Create a minimal simulation for testing."""
        return RecursiveSimulation(
            depth=depth,
            config={
                "num_agents": 2,
                "timesteps": 10,
                "allow_nested_simulation": False,
            }
        )

    @staticmethod
    def create_meta_research_simulation(depth: int = 0) -> RecursiveSimulation:
        """Create a simulation studying how agents do research."""
        return RecursiveSimulation(
            depth=depth,
            config={
                "num_agents": 10,
                "timesteps": 500,
                "environment_complexity": 0.9,
                "allow_nested_simulation": True,
                "purpose": "meta_research",
            }
        )
