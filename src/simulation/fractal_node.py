"""
Fractal Simulation Node - The Heart of Recursive Self-Compression

This module implements the SimulationNode class - the fundamental container
for the fractal simulation structure. Each SimulationNode represents a
distinct "world" that can contain agents running their own nested simulations.

Key Concepts:
1. The simulation is a TREE, not a singleton
2. Agents can spawn sub-simulations to predict futures
3. Information density threshold triggers "The Nothing" (dissolution)
4. Recursive stepping propagates through the entire tree

Reference: Master Prompt - Fractal Semiotic Engine Architecture
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum, auto
import uuid
import numpy as np
import torch
from abc import ABC, abstractmethod


class SimulationStatus(Enum):
    """Status of a simulation node."""
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    DISSOLVING = auto()  # Entropy too low, becoming "The Nothing"
    TERMINATED = auto()


@dataclass
class SimulationConfig:
    """Configuration for a SimulationNode."""
    max_recursive_depth: int = 5  # Prevent infinite recursion
    entropy_threshold: float = 0.1  # Below this, simulation dissolves
    max_ticks_per_step: int = 100  # Prevent runaway simulations
    enable_visualization: bool = False
    godot_sync_enabled: bool = True
    vector_db_dim: int = 256  # Dimension of world state vectors


@dataclass
class WorldStateVector:
    """
    A localized Vector Database entry representing world state.
    
    Each simulation has its own "pocket reality" stored as vectors.
    """
    vector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Optional[torch.Tensor] = None
    entity_id: Optional[str] = None
    entity_type: str = "generic"
    semantic_links: List[str] = field(default_factory=list)
    last_updated_tick: int = 0
    confidence: float = 1.0
    
    def similarity(self, other: 'WorldStateVector') -> float:
        """Calculate cosine similarity between vectors."""
        if self.content is None or other.content is None:
            return 0.0
        v1 = self.content.flatten()
        v2 = other.content.flatten()
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(torch.dot(v1, v2) / (norm1 * norm2))


class RSCAgent(ABC):
    """
    Abstract base class for agents using Recursive Self-Compression.
    
    RSC agents:
    1. Receive attention streams (Worldsim protocol)
    2. Compress into CognitiveBlocks via TRM
    3. Can act OR spawn nested simulations
    """
    
    def __init__(self, agent_id: str, tom_depth: int = 3):
        self.agent_id = agent_id
        self.tom_depth = tom_depth
        self.current_simulation_id: Optional[str] = None
        self.compression_ratio: float = 1.0
        self.surprise_threshold: float = 0.7  # Triggers architecture search
        
    @abstractmethod
    def perceive(self, attention_stream: Any) -> Any:
        """Process incoming perceptions via TRM."""
        pass
    
    @abstractmethod
    def decide_action(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to:
        - ACT: Return physical action for Godot
        - SIMULATE: Return simulation request
        """
        pass
    
    @abstractmethod
    def compress(self, data: Any) -> Any:
        """Apply TRM compression to create CognitiveBlock."""
        pass
    
    def should_search_architecture(self, surprise: float) -> bool:
        """Check if compression failure warrants architecture search."""
        return surprise > self.surprise_threshold


class SimulationNode:
    """
    A Fractal Container for the simulation tree.
    
    The simulation is not a singleton; it is a TREE. Each node:
    - Holds a localized Vector Database (world state)
    - Contains active RSC_Agent instances
    - Maintains registry of child (nested) simulations
    - Enforces information density threshold ("The Nothing")
    
    When step() is called on the Root Node, it recursively steps
    all active child nodes, creating a cascading update through
    the entire fractal structure.
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        parent_node: Optional['SimulationNode'] = None,
        config: Optional[SimulationConfig] = None,
        spawning_agent_id: Optional[str] = None,
    ):
        self.node_id = node_id or str(uuid.uuid4())
        self.parent_node = parent_node
        self.config = config or SimulationConfig()
        self.spawning_agent_id = spawning_agent_id  # Agent that created this simulation
        
        # Calculate recursive depth
        self.recursive_depth = 0
        if parent_node:
            self.recursive_depth = parent_node.recursive_depth + 1
        
        # State
        self.status = SimulationStatus.INITIALIZING
        self.current_tick = 0
        
        # World state (localized Vector Database)
        self.world_state: Dict[str, WorldStateVector] = {}
        
        # Agents in this simulation
        self.agents: Dict[str, RSCAgent] = {}
        
        # Child simulations (nested predictions)
        self.child_nodes: Dict[str, 'SimulationNode'] = {}
        
        # Entropy tracking
        self.semantic_entropy = 1.0  # High = complex, Low = dissolving
        self.information_density = 1.0
        self.entropy_history: List[float] = []
        
        # Event log
        self.event_log: List[Dict[str, Any]] = []
        
        # Godot sync queue
        self.godot_action_queue: List[Dict[str, Any]] = []
        
    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the simulation with starting state.
        
        For the root node, this comes from Godot.
        For child nodes, this comes from the parent's world state.
        """
        if self.recursive_depth > self.config.max_recursive_depth:
            self.status = SimulationStatus.TERMINATED
            self._log_event("init_failed", "Max recursive depth exceeded")
            return False
        
        if initial_state:
            for entity_id, data in initial_state.items():
                self._add_world_state(entity_id, data)
        
        self.status = SimulationStatus.ACTIVE
        self._log_event("initialized", f"Depth={self.recursive_depth}")
        return True
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one tick of the simulation.
        
        This recursively steps all child nodes, creating a cascading
        update through the entire fractal tree.
        
        Returns:
            Dict with tick results including any Godot actions
        """
        if self.status != SimulationStatus.ACTIVE:
            return {"status": self.status.name, "tick": self.current_tick}
        
        self.current_tick += 1
        results = {
            "node_id": self.node_id,
            "tick": self.current_tick,
            "depth": self.recursive_depth,
            "agent_actions": [],
            "child_results": [],
            "godot_actions": [],
        }
        
        # 1. Check entropy threshold (The Nothing)
        self._calculate_entropy()
        if self._should_dissolve():
            self._dissolve()
            results["status"] = "dissolved"
            return results
        
        # 2. Collect perceptions for agents
        world_snapshot = self._get_world_snapshot()
        
        # 3. Let each agent process and decide
        for agent_id, agent in self.agents.items():
            agent_result = self._process_agent_tick(agent, world_snapshot)
            results["agent_actions"].append(agent_result)
            
            # If agent wants to simulate, spawn child node
            if agent_result.get("action_type") == "simulate":
                child_result = self._spawn_child_simulation(agent, agent_result)
                results["child_results"].append(child_result)
        
        # 4. Recursively step all child simulations
        for child_id, child_node in list(self.child_nodes.items()):
            if child_node.status == SimulationStatus.ACTIVE:
                child_tick_result = child_node.step()
                results["child_results"].append(child_tick_result)
                
                # Clean up terminated children
                if child_node.status in [SimulationStatus.TERMINATED, SimulationStatus.DISSOLVING]:
                    self._cleanup_child(child_id)
        
        # 5. Collect Godot actions for root to dispatch
        results["godot_actions"] = self.godot_action_queue.copy()
        self.godot_action_queue.clear()
        
        results["entropy"] = self.semantic_entropy
        results["status"] = "active"
        return results
    
    def _process_agent_tick(
        self,
        agent: RSCAgent,
        world_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single tick for one agent."""
        # Agent perceives world
        perception = agent.perceive(world_snapshot)
        
        # Agent decides action
        action = agent.decide_action(world_snapshot)
        
        result = {
            "agent_id": agent.agent_id,
            "action_type": action.get("type", "none"),
            "action_data": action,
        }
        
        # Handle different action types
        if action.get("type") == "physical":
            # Queue for Godot execution
            self.godot_action_queue.append({
                "agent_id": agent.agent_id,
                "command": action.get("godot_command"),
                "target": action.get("target_entity"),
                "params": action.get("parameters", {}),
            })
            
        elif action.get("type") == "simulate":
            # Will be handled by _spawn_child_simulation
            result["simulation_seed"] = action.get("seed_state")
            result["simulation_horizon"] = action.get("horizon", 10)
        
        return result
    
    def _spawn_child_simulation(
        self,
        agent: RSCAgent,
        simulation_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a nested simulation for agent prediction.
        
        The agent feeds a CognitiveBlock (seed) into a new SimulationNode
        to predict future states.
        """
        child_id = f"{self.node_id}_child_{len(self.child_nodes)}_{agent.agent_id}"
        
        child = SimulationNode(
            node_id=child_id,
            parent_node=self,
            config=self.config,
            spawning_agent_id=agent.agent_id,
        )
        
        # Initialize with seed state from agent's compressed representation
        seed_state = simulation_request.get("seed_state", {})
        if child.initialize(seed_state):
            self.child_nodes[child_id] = child
            self._log_event("child_spawned", f"Agent {agent.agent_id} spawned {child_id}")
            
            # Run simulation for requested horizon with early termination
            horizon = simulation_request.get("horizon", 10)
            prediction_results = []
            convergence_window = 3  # Check convergence over last N ticks
            convergence_threshold = 0.01  # Entropy change threshold
            
            for i in range(min(horizon, self.config.max_ticks_per_step)):
                tick_result = child.step()
                prediction_results.append(tick_result)
                
                # Early termination conditions
                if child.status != SimulationStatus.ACTIVE:
                    break
                
                # Check for convergence (entropy stable)
                if len(child.entropy_history) >= convergence_window:
                    recent = child.entropy_history[-convergence_window:]
                    entropy_change = max(recent) - min(recent)
                    if entropy_change < convergence_threshold:
                        self._log_event("simulation_converged", f"{child_id} converged at tick {i}")
                        break
            
            return {
                "child_id": child_id,
                "status": "completed",
                "ticks_run": len(prediction_results),
                "final_state": child._get_world_snapshot(),
                "entropy_trajectory": child.entropy_history,
            }
        else:
            return {
                "child_id": child_id,
                "status": "failed",
                "reason": "Initialization failed (max depth?)",
            }
    
    def _cleanup_child(self, child_id: str) -> None:
        """Clean up a terminated child simulation."""
        if child_id in self.child_nodes:
            child = self.child_nodes[child_id]
            # Optionally extract useful information before deletion
            self._log_event("child_cleaned", f"Removed {child_id}")
            del self.child_nodes[child_id]
    
    def _calculate_entropy(self) -> float:
        """
        Calculate semantic entropy of the simulation.
        
        Entropy measures the complexity/richness of the simulation.
        Low entropy triggers "The Nothing" - dissolution.
        """
        if not self.world_state:
            self.semantic_entropy = 0.0
            return 0.0
        
        # Count unique semantic concepts
        all_concepts: Set[str] = set()
        for state_vec in self.world_state.values():
            all_concepts.update(state_vec.semantic_links)
        
        concept_diversity = len(all_concepts) / max(len(self.world_state), 1)
        
        # Activity level (events this tick)
        recent_events = [e for e in self.event_log if e.get("tick", 0) == self.current_tick]
        activity_level = len(recent_events) / 10.0
        
        # Agent complexity
        agent_complexity = sum(a.tom_depth for a in self.agents.values()) / max(len(self.agents), 1) / 5.0
        
        # Combined entropy
        self.semantic_entropy = min(1.0, (concept_diversity + activity_level + agent_complexity) / 3)
        self.entropy_history.append(self.semantic_entropy)
        
        # Information density based on world state richness
        vector_norms = [
            torch.norm(v.content).item() if v.content is not None else 0.0
            for v in self.world_state.values()
        ]
        self.information_density = np.mean(vector_norms) if vector_norms else 0.0
        
        return self.semantic_entropy
    
    def _should_dissolve(self) -> bool:
        """
        Check if simulation should dissolve into "The Nothing".
        
        Enforces information density threshold - simulations that
        lack semantic complexity are terminated.
        """
        # Check entropy threshold
        if self.semantic_entropy < self.config.entropy_threshold:
            return True
        
        # Check for sustained low entropy
        if len(self.entropy_history) > 10:
            recent = self.entropy_history[-10:]
            if all(e < self.config.entropy_threshold * 1.5 for e in recent):
                return True
        
        return False
    
    def _dissolve(self) -> None:
        """
        Dissolve simulation into "The Nothing".
        
        This represents the collapse of a prediction/simulation
        that failed to maintain semantic coherence.
        """
        self.status = SimulationStatus.DISSOLVING
        self._log_event("dissolving", f"Entropy={self.semantic_entropy:.3f}")
        
        # Terminate all child simulations
        for child_id, child in list(self.child_nodes.items()):
            child._dissolve()
            self._cleanup_child(child_id)
        
        # Clear agents
        self.agents.clear()
        
        # Clear world state
        self.world_state.clear()
        
        self.status = SimulationStatus.TERMINATED
        self._log_event("dissolved", "Became The Nothing")
    
    def _add_world_state(
        self,
        entity_id: str,
        data: Any,
        entity_type: str = "generic"
    ) -> WorldStateVector:
        """Add or update world state for an entity."""
        if isinstance(data, torch.Tensor):
            content = data
        elif isinstance(data, np.ndarray):
            content = torch.from_numpy(data).float()
        elif isinstance(data, dict):
            # Convert dict to tensor (simplified)
            content = torch.zeros(self.config.vector_db_dim)
        else:
            content = torch.zeros(self.config.vector_db_dim)
        
        state_vec = WorldStateVector(
            entity_id=entity_id,
            entity_type=entity_type,
            content=content,
            last_updated_tick=self.current_tick,
        )
        self.world_state[entity_id] = state_vec
        return state_vec
    
    def _get_world_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current world state."""
        return {
            "tick": self.current_tick,
            "depth": self.recursive_depth,
            "entropy": self.semantic_entropy,
            "entities": {
                eid: {
                    "type": vec.entity_type,
                    "content": vec.content,
                    "links": vec.semantic_links,
                }
                for eid, vec in self.world_state.items()
            },
            "agents": list(self.agents.keys()),
            "children": list(self.child_nodes.keys()),
        }
    
    def _log_event(self, event_type: str, details: str = "") -> None:
        """Log an event in this simulation."""
        self.event_log.append({
            "tick": self.current_tick,
            "type": event_type,
            "details": details,
            "node_id": self.node_id,
            "depth": self.recursive_depth,
        })
    
    # Public interface methods
    
    def add_agent(self, agent: RSCAgent) -> None:
        """Add an agent to this simulation."""
        agent.current_simulation_id = self.node_id
        self.agents[agent.agent_id] = agent
        self._log_event("agent_added", agent.agent_id)
    
    def remove_agent(self, agent_id: str) -> Optional[RSCAgent]:
        """Remove and return an agent from this simulation."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            agent.current_simulation_id = None
            self._log_event("agent_removed", agent_id)
            return agent
        return None
    
    def query_world_state(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5
    ) -> List[WorldStateVector]:
        """
        Query world state using vector similarity.
        
        This is the Vector Database query for semantic retrieval.
        """
        query_wsv = WorldStateVector(content=query_vector)
        
        similarities = [
            (eid, query_wsv.similarity(wsv))
            for eid, wsv in self.world_state.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [self.world_state[eid] for eid, _ in similarities[:top_k]]
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """Get the full tree structure from this node down."""
        return {
            "node_id": self.node_id,
            "depth": self.recursive_depth,
            "status": self.status.name,
            "tick": self.current_tick,
            "entropy": self.semantic_entropy,
            "num_agents": len(self.agents),
            "num_entities": len(self.world_state),
            "children": {
                child_id: child.get_tree_structure()
                for child_id, child in self.child_nodes.items()
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize simulation state to dictionary."""
        return {
            "node_id": self.node_id,
            "recursive_depth": self.recursive_depth,
            "status": self.status.name,
            "current_tick": self.current_tick,
            "semantic_entropy": self.semantic_entropy,
            "information_density": self.information_density,
            "spawning_agent_id": self.spawning_agent_id,
            "agents": list(self.agents.keys()),
            "world_state_count": len(self.world_state),
            "child_count": len(self.child_nodes),
            "event_log_length": len(self.event_log),
        }


class RootSimulationNode(SimulationNode):
    """
    The root node of the simulation tree.
    
    Special properties:
    - Connects to Godot via WebSocket
    - Has no parent
    - Recursive depth = 0 (reality)
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        godot_bridge: Optional[Any] = None,
    ):
        super().__init__(
            node_id="root",
            parent_node=None,
            config=config or SimulationConfig(),
        )
        self.godot_bridge = godot_bridge
        
    def sync_with_godot(self, godot_state: Dict[str, Any]) -> None:
        """
        Sync world state from Godot physics engine.
        
        This is called every frame to update the simulation
        with physical reality.
        """
        for entity_id, entity_data in godot_state.get("entities", {}).items():
            self._add_world_state(
                entity_id,
                entity_data.get("state", {}),
                entity_data.get("type", "godot_entity"),
            )
    
    def dispatch_actions_to_godot(self) -> List[Dict[str, Any]]:
        """
        Get all queued Godot actions from entire tree.
        
        Walks the tree and collects all physical actions
        to send to Godot.
        """
        all_actions = []
        all_actions.extend(self.godot_action_queue)
        
        def collect_child_actions(node: SimulationNode):
            for child in node.child_nodes.values():
                all_actions.extend(child.godot_action_queue)
                child.godot_action_queue.clear()
                collect_child_actions(child)
        
        collect_child_actions(self)
        self.godot_action_queue.clear()
        
        return all_actions


# Factory function for creating a standard simulation
def create_simulation(
    initial_state: Optional[Dict[str, Any]] = None,
    config: Optional[SimulationConfig] = None,
    is_root: bool = True,
) -> SimulationNode:
    """
    Create a new simulation node.
    
    Args:
        initial_state: Initial world state dictionary
        config: Simulation configuration
        is_root: Whether this is the root node
    
    Returns:
        Initialized SimulationNode
    """
    if is_root:
        node = RootSimulationNode(config=config)
    else:
        node = SimulationNode(config=config)
    
    node.initialize(initial_state)
    return node


# Concrete RSCAgent Implementations

class NeuralRSCAgent(RSCAgent):
    """
    Neural network-based RSC agent using learned perception and decision-making.

    Uses a simple neural network for perception compression and
    action selection based on world state.
    """

    def __init__(
        self,
        agent_id: str,
        tom_depth: int = 3,
        hidden_dim: int = 128,
        input_dim: int = 256,
    ):
        super().__init__(agent_id, tom_depth)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Perception network (compresses attention stream)
        self.perception_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Decision network (selects action from world state)
        self.decision_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2 + input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 4),  # 4 action types
        )

        # Internal state (compressed representation)
        self.internal_state = torch.zeros(hidden_dim // 2)

    def perceive(self, attention_stream: Any) -> torch.Tensor:
        """Process incoming perceptions via neural compression."""
        # Convert attention stream to tensor
        if isinstance(attention_stream, dict):
            # Extract entity tensors and aggregate
            entity_tensors = []
            for entity_data in attention_stream.get('entities', {}).values():
                content = entity_data.get('content')
                if content is not None:
                    entity_tensors.append(content.flatten()[:self.input_dim])

            if entity_tensors:
                # Mean pooling over entities
                input_tensor = torch.stack([
                    torch.nn.functional.pad(t, (0, max(0, self.input_dim - len(t))))[:self.input_dim]
                    for t in entity_tensors
                ]).mean(dim=0)
            else:
                input_tensor = torch.zeros(self.input_dim)
        elif isinstance(attention_stream, torch.Tensor):
            input_tensor = attention_stream.flatten()[:self.input_dim]
            if len(input_tensor) < self.input_dim:
                input_tensor = torch.nn.functional.pad(
                    input_tensor, (0, self.input_dim - len(input_tensor))
                )
        else:
            input_tensor = torch.zeros(self.input_dim)

        # Compress through perception network
        with torch.no_grad():
            self.internal_state = self.perception_net(input_tensor)

        return self.internal_state

    def decide_action(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide action based on world state and internal state.

        Returns action dict with type: 'physical', 'simulate', or 'none'
        """
        # Build input from world state
        world_features = torch.zeros(self.input_dim)
        world_features[0] = world_state.get('tick', 0) / 1000.0
        world_features[1] = world_state.get('entropy', 0.5)
        world_features[2] = len(world_state.get('agents', [])) / 10.0
        world_features[3] = len(world_state.get('entities', {})) / 100.0

        # Combine with internal state
        combined = torch.cat([self.internal_state, world_features])

        # Get action logits
        with torch.no_grad():
            action_logits = self.decision_net(combined)
            action_probs = torch.softmax(action_logits, dim=0)
            action_idx = torch.argmax(action_probs).item()

        # Map to action types
        action_types = ['none', 'physical', 'simulate', 'communicate']
        action_type = action_types[action_idx]

        if action_type == 'physical':
            return {
                'type': 'physical',
                'godot_command': 'move',
                'target_entity': None,
                'parameters': {'direction': [1, 0, 0]},
            }
        elif action_type == 'simulate':
            return {
                'type': 'simulate',
                'seed_state': {'prediction_seed': self.internal_state.tolist()},
                'horizon': self.tom_depth * 3,
            }
        elif action_type == 'communicate':
            return {
                'type': 'physical',
                'godot_command': 'communicate',
                'target_entity': None,
                'parameters': {'message': 'greeting'},
            }
        else:
            return {'type': 'none'}

    def compress(self, data: Any) -> torch.Tensor:
        """Apply compression to create CognitiveBlock representation."""
        if isinstance(data, torch.Tensor):
            return self.perception_net(data.flatten()[:self.input_dim])
        elif isinstance(data, dict):
            # Convert dict to tensor representation
            tensor = torch.zeros(self.input_dim)
            for i, (k, v) in enumerate(list(data.items())[:self.input_dim]):
                if isinstance(v, (int, float)):
                    tensor[i] = float(v)
            return self.perception_net(tensor)
        else:
            return self.internal_state


class RuleBasedRSCAgent(RSCAgent):
    """
    Rule-based RSC agent for testing and baseline comparisons.

    Uses simple heuristics for perception and decision-making.
    """

    def __init__(self, agent_id: str, tom_depth: int = 3):
        super().__init__(agent_id, tom_depth)
        self.memory: List[Dict[str, Any]] = []
        self.action_history: List[str] = []

    def perceive(self, attention_stream: Any) -> Dict[str, Any]:
        """Extract key features from attention stream."""
        perception = {
            'tick': 0,
            'entity_count': 0,
            'agent_count': 0,
            'entropy': 1.0,
        }

        if isinstance(attention_stream, dict):
            perception['tick'] = attention_stream.get('tick', 0)
            perception['entity_count'] = len(attention_stream.get('entities', {}))
            perception['agent_count'] = len(attention_stream.get('agents', []))
            perception['entropy'] = attention_stream.get('entropy', 1.0)

        # Store in memory (limited)
        self.memory.append(perception)
        if len(self.memory) > 100:
            self.memory = self.memory[-50:]

        return perception

    def decide_action(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Use heuristic rules for action selection."""
        entropy = world_state.get('entropy', 1.0)
        tick = world_state.get('tick', 0)

        # Rule 1: If entropy is low, simulate to predict future
        if entropy < 0.3:
            self.action_history.append('simulate')
            return {
                'type': 'simulate',
                'seed_state': {'entropy_crisis': True},
                'horizon': 5,
            }

        # Rule 2: Every 10 ticks, take a physical action
        if tick % 10 == 0:
            self.action_history.append('physical')
            return {
                'type': 'physical',
                'godot_command': 'explore',
                'target_entity': None,
                'parameters': {},
            }

        # Rule 3: If many other agents, simulate their behavior
        if world_state.get('agents', []) and len(world_state.get('agents', [])) > 2:
            if tick % 5 == 0:
                self.action_history.append('simulate')
                return {
                    'type': 'simulate',
                    'seed_state': {'social_prediction': True},
                    'horizon': self.tom_depth * 2,
                }

        # Default: no action
        self.action_history.append('none')
        return {'type': 'none'}

    def compress(self, data: Any) -> Dict[str, Any]:
        """Simple compression by extracting key features."""
        if isinstance(data, dict):
            return {k: v for k, v in list(data.items())[:10]}
        return {'raw': str(data)[:100]}


# Export
__all__ = [
    'SimulationStatus',
    'SimulationConfig',
    'WorldStateVector',
    'RSCAgent',
    'NeuralRSCAgent',
    'RuleBasedRSCAgent',
    'SimulationNode',
    'RootSimulationNode',
    'create_simulation',
]
