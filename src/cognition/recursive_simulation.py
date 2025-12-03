"""
Recursive Simulation Node - The Fractal Sandbox

Implements the Recursive Self-Compression (RSC) engine where agents
can spin up sandboxed sub-simulations to predict other agents' behavior.

The Process:
1. Snapshot: Agent A takes its perception and compresses it
2. Instantiation: Agent A spins up a lightweight internal simulation
3. Population: Agent A places models of itself and others in the simulation
4. Execution: Agent A runs the simulation forward N steps
5. Recursion: Simulated agents can spin up their own sub-simulations

This is the computational mechanism for deep Theory of Mind:
If I want to predict what you will do, I simulate you.
If you might be predicting me, I simulate you simulating me.

Theoretical Foundation:
- Simulation Theory of Mind (Goldman)
- Counterfactual Reasoning (Pearl)
- Mental Models (Johnson-Laird)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from copy import deepcopy
import logging

from .mentalese import (
    CognitiveBlock,
    BeliefBlock,
    IntentBlock,
    SimulationState,
    MemoryBlock,
    BlockType,
)


logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for recursive simulation."""
    max_recursion_depth: int = 3         # Maximum nesting depth
    max_steps_per_simulation: int = 10   # Steps to run each simulation
    confidence_decay_per_depth: float = 0.7  # Confidence decay per level
    pruning_threshold: float = 0.1       # Prune low-confidence branches

    # Computational limits
    max_parallel_simulations: int = 4    # Limit concurrent simulations
    timeout_ms: float = 1000.0           # Timeout per simulation

    # Approximation settings
    use_approximation: bool = True       # Use TRM approximation
    approximation_threshold: int = 2     # Depth at which to switch to approximation


@dataclass
class WorldModel:
    """
    A compressed model of the world state for simulation.

    This is not the full Godot world - it's a lightweight
    representation sufficient for cognitive simulation.
    """
    # Entities in the world
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # entity_id -> {position, properties, affordances}

    # Agents in the world
    agents: Dict[str, 'AgentModel'] = field(default_factory=dict)
    # agent_id -> AgentModel

    # World state variables
    state_variables: Dict[str, Any] = field(default_factory=dict)

    # Timestep
    timestep: int = 0

    def clone(self) -> 'WorldModel':
        """Create a deep copy for sandboxed simulation."""
        return WorldModel(
            entities=deepcopy(self.entities),
            agents={k: v.clone() for k, v in self.agents.items()},
            state_variables=deepcopy(self.state_variables),
            timestep=self.timestep,
        )

    def get_observable_state(self, observer_id: str) -> Dict[str, Any]:
        """Get world state observable by a specific agent."""
        # In full implementation, would filter by visibility/knowledge
        return {
            'entities': self.entities,
            'other_agents': {
                k: v.get_observable_state()
                for k, v in self.agents.items()
                if k != observer_id
            },
            'timestep': self.timestep,
        }


@dataclass
class AgentModel:
    """
    A model of an agent for use in simulation.

    This is what Agent A believes about Agent B's cognitive state.
    It's an approximation that enables ToM reasoning.
    """
    agent_id: str

    # Believed mental state
    beliefs: List[BeliefBlock] = field(default_factory=list)
    intents: List[IntentBlock] = field(default_factory=list)
    memories: List[MemoryBlock] = field(default_factory=list)

    # Behavioral model
    behavior_policy: Optional[Callable] = None  # How this agent acts

    # Uncertainty about this model
    model_confidence: float = 0.5        # How confident in this model

    # Historical observations of this agent
    observation_history: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> 'AgentModel':
        """Create a copy for simulation."""
        return AgentModel(
            agent_id=self.agent_id,
            beliefs=[deepcopy(b) for b in self.beliefs],
            intents=[deepcopy(i) for i in self.intents],
            memories=[deepcopy(m) for m in self.memories],
            behavior_policy=self.behavior_policy,
            model_confidence=self.model_confidence,
            observation_history=list(self.observation_history),
        )

    def get_observable_state(self) -> Dict[str, Any]:
        """Get externally observable aspects of this agent."""
        return {
            'agent_id': self.agent_id,
            'apparent_intents': [i.to_natural_language() for i in self.intents[:3]],
        }

    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        """Update model based on new observation of this agent."""
        self.observation_history.append(observation)

        # Simple belief update - in full implementation would use Bayesian inference
        if 'action' in observation:
            # Infer intent from action
            inferred_intent = IntentBlock(
                goal=f"achieve_{observation['action']}",
                action_type=observation['action'],
                confidence=0.6,
            )
            self.intents.append(inferred_intent)

        if 'utterance' in observation:
            # Infer belief from utterance
            inferred_belief = BeliefBlock(
                proposition=observation['utterance'],
                source_type="testimony",
                confidence=0.5,
            )
            self.beliefs.append(inferred_belief)


@dataclass
class SimulationResult:
    """Result of running a recursive simulation."""
    # Simulation identity
    simulating_agent: str
    simulated_agent: str
    recursion_depth: int

    # Prediction
    predicted_action: Optional[str] = None
    predicted_intent: Optional[IntentBlock] = None
    action_probabilities: Dict[str, float] = field(default_factory=dict)

    # Confidence
    prediction_confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)

    # Nested results (if recursive)
    nested_results: List['SimulationResult'] = field(default_factory=list)

    # Metadata
    steps_simulated: int = 0
    computation_time_ms: float = 0.0
    was_approximated: bool = False

    def to_belief(self) -> BeliefBlock:
        """Convert simulation result to a belief about the simulated agent."""
        return BeliefBlock(
            proposition=f"{self.simulated_agent} will {self.predicted_action}",
            subject=self.simulated_agent,
            predicate=f"will_{self.predicted_action}",
            about_agent=self.simulated_agent,
            confidence=self.prediction_confidence,
            belief_order=self.recursion_depth + 1,
            source_type="simulation",
        )


class RecursiveSimulationNode:
    """
    A node in the recursive simulation tree.

    When Agent A needs to predict Agent B:
    1. A creates a RecursiveSimulationNode
    2. The node contains a WorldModel and AgentModels
    3. The node can run forward simulation
    4. If B needs to think about A, a child node is created

    This implements the "fractal sandbox" - simulations within simulations.
    """

    def __init__(
        self,
        simulating_agent: str,
        world_model: WorldModel,
        config: SimulationConfig,
        parent_node: Optional['RecursiveSimulationNode'] = None,
        current_depth: int = 0
    ):
        """
        Initialize a simulation node.

        Args:
            simulating_agent: Agent running this simulation
            world_model: The world state to simulate
            config: Simulation configuration
            parent_node: Parent node if this is a nested simulation
            current_depth: Current recursion depth
        """
        self.simulating_agent = simulating_agent
        self.world_model = world_model.clone()  # Always work on a copy
        self.config = config
        self.parent_node = parent_node
        self.current_depth = current_depth

        # Child simulations (nested)
        self.child_nodes: List[RecursiveSimulationNode] = []

        # Simulation state
        self.current_step = 0
        self.is_running = False
        self.is_complete = False

        # Results
        self.results: Dict[str, SimulationResult] = {}  # agent_id -> result

        # Logging
        self.trace: List[str] = []

    def can_recurse(self) -> bool:
        """Check if we can create a deeper nested simulation."""
        return self.current_depth < self.config.max_recursion_depth

    def should_approximate(self) -> bool:
        """Check if we should use TRM approximation instead of full simulation."""
        return (
            self.config.use_approximation and
            self.current_depth >= self.config.approximation_threshold
        )

    def run_simulation(
        self,
        target_agent: str,
        num_steps: Optional[int] = None
    ) -> SimulationResult:
        """
        Run simulation to predict a target agent's behavior.

        Args:
            target_agent: The agent to predict
            num_steps: Number of steps to simulate (default from config)

        Returns:
            SimulationResult with prediction
        """
        start_time = datetime.now()
        steps = num_steps or self.config.max_steps_per_simulation

        self.trace.append(
            f"[Depth {self.current_depth}] {self.simulating_agent} simulating {target_agent}"
        )

        # Check if we should approximate
        if self.should_approximate():
            return self._approximate_simulation(target_agent)

        # Get target agent model
        if target_agent not in self.world_model.agents:
            return SimulationResult(
                simulating_agent=self.simulating_agent,
                simulated_agent=target_agent,
                recursion_depth=self.current_depth,
                prediction_confidence=0.0,
                reasoning_trace=[f"No model of {target_agent} available"],
            )

        agent_model = self.world_model.agents[target_agent]
        self.is_running = True

        # Run simulation steps
        action_counts: Dict[str, int] = {}
        all_actions: List[str] = []

        for step in range(steps):
            self.current_step = step

            # Get agent's observable world
            observable = self.world_model.get_observable_state(target_agent)

            # Check if target needs to simulate us (recursive!)
            if self._agent_needs_tom(agent_model, observable):
                if self.can_recurse():
                    nested_result = self._create_nested_simulation(
                        target_agent,
                        self.simulating_agent  # Target simulates us
                    )
                    # Incorporate nested result into agent's beliefs
                    agent_model.beliefs.append(nested_result.to_belief())

            # Predict agent's action
            predicted_action = self._predict_action(agent_model, observable)
            all_actions.append(predicted_action)
            action_counts[predicted_action] = action_counts.get(predicted_action, 0) + 1

            # Update world state
            self._apply_action(target_agent, predicted_action)

            self.trace.append(
                f"  Step {step}: {target_agent} -> {predicted_action}"
            )

        self.is_running = False
        self.is_complete = True

        # Compute prediction
        if action_counts:
            most_common = max(action_counts, key=action_counts.get)
            confidence = action_counts[most_common] / len(all_actions)
            confidence *= (self.config.confidence_decay_per_depth ** self.current_depth)
        else:
            most_common = "no_action"
            confidence = 0.0

        # Build action probabilities
        action_probs = {
            action: count / len(all_actions)
            for action, count in action_counts.items()
        }

        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds() * 1000

        result = SimulationResult(
            simulating_agent=self.simulating_agent,
            simulated_agent=target_agent,
            recursion_depth=self.current_depth,
            predicted_action=most_common,
            action_probabilities=action_probs,
            prediction_confidence=confidence,
            steps_simulated=steps,
            computation_time_ms=computation_time,
            reasoning_trace=self.trace.copy(),
            nested_results=[
                self.results.get(child.simulating_agent)
                for child in self.child_nodes
                if child.simulating_agent in self.results
            ],
        )

        self.results[target_agent] = result
        return result

    def _agent_needs_tom(
        self,
        agent_model: AgentModel,
        observable: Dict[str, Any]
    ) -> bool:
        """
        Determine if an agent needs to use ToM in this situation.

        Agents need ToM when:
        - Interacting with other agents
        - Stakes are high
        - Deception might be involved
        """
        # Simple heuristic: need ToM if other agents present
        if 'other_agents' in observable and observable['other_agents']:
            # Check if any intent involves other agents
            for intent in agent_model.intents:
                if intent.target_agent:
                    return True

        return False

    def _create_nested_simulation(
        self,
        simulating_agent: str,
        target_agent: str
    ) -> SimulationResult:
        """Create a nested simulation (agent simulating another agent)."""
        self.trace.append(
            f"  [Creating nested simulation: {simulating_agent} -> {target_agent}]"
        )

        # Create child node
        child = RecursiveSimulationNode(
            simulating_agent=simulating_agent,
            world_model=self.world_model,
            config=self.config,
            parent_node=self,
            current_depth=self.current_depth + 1,
        )

        self.child_nodes.append(child)

        # Run the nested simulation (with reduced steps)
        nested_steps = max(1, self.config.max_steps_per_simulation // 2)
        result = child.run_simulation(target_agent, num_steps=nested_steps)

        return result

    def _predict_action(
        self,
        agent_model: AgentModel,
        observable: Dict[str, Any]
    ) -> str:
        """
        Predict what action the agent will take.

        Uses the agent's behavior policy if available,
        otherwise uses intent-based prediction.
        """
        # If we have a behavior policy, use it
        if agent_model.behavior_policy:
            try:
                return agent_model.behavior_policy(agent_model, observable)
            except Exception:
                pass

        # Intent-based prediction
        if agent_model.intents:
            # Find highest urgency intent
            sorted_intents = sorted(
                agent_model.intents,
                key=lambda i: i.urgency * i.confidence,
                reverse=True
            )
            top_intent = sorted_intents[0]
            return top_intent.action_type or "pursue_goal"

        # Default action
        return "wait"

    def _apply_action(self, agent_id: str, action: str) -> None:
        """Apply an action to the simulated world."""
        # Simple state update - in full implementation would use action effects
        self.world_model.timestep += 1
        self.world_model.state_variables[f"{agent_id}_last_action"] = action

    def _approximate_simulation(self, target_agent: str) -> SimulationResult:
        """
        Use TRM approximation instead of full simulation.

        At deeper recursion levels, we use a learned approximation
        to avoid exponential computation costs.
        """
        self.trace.append(
            f"  [Using TRM approximation at depth {self.current_depth}]"
        )

        # Get agent model
        agent_model = self.world_model.agents.get(target_agent)

        if agent_model and agent_model.intents:
            # Simple approximation: predict based on top intent
            top_intent = max(
                agent_model.intents,
                key=lambda i: i.urgency * i.confidence
            )
            predicted = top_intent.action_type or "wait"
            confidence = top_intent.confidence * 0.5  # Lower confidence for approximation
        else:
            predicted = "wait"
            confidence = 0.2

        return SimulationResult(
            simulating_agent=self.simulating_agent,
            simulated_agent=target_agent,
            recursion_depth=self.current_depth,
            predicted_action=predicted,
            prediction_confidence=confidence,
            reasoning_trace=self.trace.copy(),
            was_approximated=True,
        )

    def get_simulation_tree_depth(self) -> int:
        """Get the maximum depth of the simulation tree."""
        if not self.child_nodes:
            return self.current_depth

        return max(
            child.get_simulation_tree_depth()
            for child in self.child_nodes
        )

    def get_total_simulations_run(self) -> int:
        """Get total number of simulations in the tree."""
        count = len(self.results)
        for child in self.child_nodes:
            count += child.get_total_simulations_run()
        return count


class SimulationManager:
    """
    Manages recursive simulations for an agent.

    Provides a high-level interface for ToM-based prediction,
    handling the complexity of nested simulation trees.
    """

    def __init__(self, agent_id: str, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation manager.

        Args:
            agent_id: The agent this manager belongs to
            config: Simulation configuration
        """
        self.agent_id = agent_id
        self.config = config or SimulationConfig()

        # Agent models (what we believe about others)
        self.agent_models: Dict[str, AgentModel] = {}

        # Simulation history
        self.simulation_history: List[SimulationResult] = []

    def update_agent_model(
        self,
        target_agent: str,
        observation: Dict[str, Any]
    ) -> None:
        """
        Update our model of another agent based on observation.

        Args:
            target_agent: Agent we observed
            observation: What we observed
        """
        if target_agent not in self.agent_models:
            self.agent_models[target_agent] = AgentModel(agent_id=target_agent)

        self.agent_models[target_agent].update_from_observation(observation)

    def predict_agent(
        self,
        target_agent: str,
        world_model: WorldModel,
        num_steps: int = 5
    ) -> SimulationResult:
        """
        Predict what a target agent will do.

        Args:
            target_agent: Agent to predict
            world_model: Current world state
            num_steps: Steps to simulate

        Returns:
            SimulationResult with prediction
        """
        # Ensure we have a model of the target
        if target_agent not in self.agent_models:
            return SimulationResult(
                simulating_agent=self.agent_id,
                simulated_agent=target_agent,
                recursion_depth=0,
                prediction_confidence=0.0,
                reasoning_trace=["No model of target agent"],
            )

        # Add our agent models to the world model
        world_with_agents = world_model.clone()
        for agent_id, model in self.agent_models.items():
            world_with_agents.agents[agent_id] = model

        # Create root simulation node
        root = RecursiveSimulationNode(
            simulating_agent=self.agent_id,
            world_model=world_with_agents,
            config=self.config,
            current_depth=0,
        )

        # Run simulation
        result = root.run_simulation(target_agent, num_steps)

        # Store in history
        self.simulation_history.append(result)

        return result

    def predict_what_agent_thinks_i_will_do(
        self,
        target_agent: str,
        world_model: WorldModel
    ) -> SimulationResult:
        """
        Predict what another agent thinks I will do.

        This is second-order ToM: I simulate you simulating me.

        Args:
            target_agent: Agent whose prediction we're predicting
            world_model: Current world state

        Returns:
            SimulationResult representing target's prediction of us
        """
        # Add a model of ourselves
        world_with_agents = world_model.clone()
        world_with_agents.agents[self.agent_id] = AgentModel(
            agent_id=self.agent_id,
            model_confidence=0.3,  # They don't know us perfectly
        )

        for agent_id, model in self.agent_models.items():
            world_with_agents.agents[agent_id] = model

        # Create simulation where target simulates us
        config = SimulationConfig(
            max_recursion_depth=self.config.max_recursion_depth - 1,
            max_steps_per_simulation=self.config.max_steps_per_simulation // 2,
        )

        root = RecursiveSimulationNode(
            simulating_agent=target_agent,  # Target is simulating
            world_model=world_with_agents,
            config=config,
            current_depth=1,  # Start at depth 1 since this is nested
        )

        # Run simulation of target predicting us
        result = root.run_simulation(self.agent_id)

        return result

    def get_tom_depth_for_agent(self, target_agent: str) -> int:
        """
        Get the ToM depth we've used for a target agent.

        Returns the deepest recursion level from simulation history.
        """
        max_depth = 0
        for result in self.simulation_history:
            if result.simulated_agent == target_agent:
                max_depth = max(max_depth, result.recursion_depth)
        return max_depth
