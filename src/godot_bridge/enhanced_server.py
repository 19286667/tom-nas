"""
Enhanced Godot Server - BeliefNetwork Integration

This module provides an enhanced WebSocket server that bridges the Godot physics
simulation with the Theory of Mind belief network system. It extends the base
GodotBridge with:

1. Real-time BeliefNetwork updates from agent perceptions
2. Integration with the Liminal SoulMap for agent psychology
3. Theory of Mind inference from observed agent behavior
4. Recursive belief state tracking (up to 5th order)

The enhanced server enables agents in Godot to develop genuine Theory of Mind
by grounding their beliefs in physical observations and social interactions.

Author: ToM-NAS Project
"""

import asyncio
import json
import logging
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from queue import Queue
import threading
import uuid

from .bridge import GodotBridge, BridgeConfig, ConnectionState
from .protocol import (
    GodotMessage, MessageType, EntityUpdate, AgentPerception,
    WorldState, AgentCommand, Vector3
)
from .symbol_grounding import SymbolGrounder, GroundingContext
from .perception import PerceptionProcessor, PerceptualField

# Import core belief system
from ..core.beliefs import BeliefNetwork, RecursiveBeliefState, Belief

# Import liminal systems for psychological modeling
try:
    from ..liminal import SoulMap, SoulMapCluster
    from ..liminal.psychosocial_coevolution import SocialNetwork, BeliefPropagationEngine
    LIMINAL_AVAILABLE = True
except ImportError:
    LIMINAL_AVAILABLE = False
    SoulMap = None
    SoulMapCluster = None


logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """
    Complete state of an agent including beliefs and psychology.

    This bridges the Godot entity with the cognitive architecture.
    """
    # Identity
    godot_id: int
    agent_name: str
    agent_index: int  # Index in BeliefNetwork

    # Current physical state
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    orientation: Vector3 = field(default_factory=Vector3)

    # Perception state
    last_perception: Optional[PerceptualField] = None
    perception_history: List[PerceptualField] = field(default_factory=list)
    max_perception_history: int = 100

    # Psychological state (Liminal)
    soul_map: Optional[Any] = None  # SoulMap if available

    # Social tracking
    observed_agents: Dict[int, float] = field(default_factory=dict)  # id -> last_seen
    social_interactions: List[Dict] = field(default_factory=list)

    # Action tracking
    current_action: Optional[str] = None
    action_history: List[Dict] = field(default_factory=list)

    def add_perception(self, perception: PerceptualField):
        """Add perception to history, maintaining max size."""
        self.last_perception = perception
        self.perception_history.append(perception)
        if len(self.perception_history) > self.max_perception_history:
            self.perception_history.pop(0)


@dataclass
class EnhancedServerConfig(BridgeConfig):
    """Configuration for the enhanced server."""
    # Belief network configuration
    num_agents: int = 10
    ontology_dim: int = 64
    max_tom_order: int = 5

    # ToM inference configuration
    belief_update_interval_ms: float = 100.0  # How often to update beliefs
    enable_tom_inference: bool = True
    tom_confidence_threshold: float = 0.3

    # Psychology integration
    enable_soul_maps: bool = True
    psychology_update_interval_ms: float = 500.0

    # Logging
    log_belief_updates: bool = False
    log_tom_inferences: bool = False


class BeliefUpdateEvent:
    """Event representing a belief update from observation."""

    def __init__(
        self,
        observer_id: int,
        target_id: int,
        belief_order: int,
        content: torch.Tensor,
        confidence: float,
        source: str,
        evidence: List[Any] = None
    ):
        self.observer_id = observer_id
        self.target_id = target_id
        self.belief_order = belief_order
        self.content = content
        self.confidence = confidence
        self.source = source
        self.evidence = evidence or []
        self.timestamp = datetime.now()


class EnhancedGodotServer(GodotBridge):
    """
    Enhanced Godot server with BeliefNetwork integration.

    This server extends the base bridge to:
    - Maintain BeliefNetwork for all agents
    - Update beliefs from perceptions
    - Infer Theory of Mind from observations
    - Track agent psychology via SoulMaps
    """

    def __init__(
        self,
        config: EnhancedServerConfig,
        belief_network: Optional[BeliefNetwork] = None,
        knowledge_base=None
    ):
        """
        Initialize the enhanced server.

        Args:
            config: Enhanced server configuration
            belief_network: Optional pre-configured belief network
            knowledge_base: Indra's Net knowledge graph
        """
        # Initialize base bridge
        super().__init__(config, knowledge_base=knowledge_base)
        self.enhanced_config = config

        # Initialize or use provided belief network
        self.belief_network = belief_network or BeliefNetwork(
            num_agents=config.num_agents,
            ontology_dim=config.ontology_dim,
            max_order=config.max_tom_order
        )

        # Agent state tracking
        self.agent_states: Dict[int, AgentState] = {}
        self.godot_id_to_index: Dict[int, int] = {}  # Map Godot IDs to network indices
        self.next_agent_index = 0

        # Social network for belief propagation
        self.social_network: Optional[SocialNetwork] = None
        self.belief_propagation: Optional[BeliefPropagationEngine] = None

        if LIMINAL_AVAILABLE:
            try:
                self.social_network = SocialNetwork()
                self.belief_propagation = BeliefPropagationEngine(self.social_network)
            except Exception as e:
                logger.warning(f"Failed to initialize social network: {e}")

        # Belief update queue
        self.belief_updates: Queue = Queue()

        # ToM inference cache
        self.tom_inferences: Dict[Tuple[int, int], Dict] = {}

        # Event callbacks
        self._belief_callbacks: List[Callable[[BeliefUpdateEvent], None]] = []
        self._tom_callbacks: List[Callable[[int, int, int, float], None]] = []

        # Register enhanced handlers
        self._register_enhanced_handlers()

        logger.info(f"EnhancedGodotServer initialized with {config.num_agents} agent capacity")

    def _register_enhanced_handlers(self):
        """Register enhanced message handlers for belief integration."""

        @self.on(MessageType.AGENT_PERCEPTION)
        def handle_perception_for_beliefs(msg: GodotMessage):
            """Process perception for belief updates."""
            perception = self._parse_agent_perception(msg.payload)
            self._process_perception_for_beliefs(perception)

        @self.on(MessageType.INTERACTION_EVENT)
        def handle_interaction_for_beliefs(msg: GodotMessage):
            """Process interactions for social belief updates."""
            self._process_interaction_for_beliefs(msg.payload)

        @self.on(MessageType.UTTERANCE_EVENT)
        def handle_utterance_for_beliefs(msg: GodotMessage):
            """Process utterances for communication-based beliefs."""
            self._process_utterance_for_beliefs(msg.payload)

    def _parse_agent_perception(self, payload: Dict) -> AgentPerception:
        """Parse perception payload."""
        visible = [
            EntityUpdate.from_dict(e)
            for e in payload.get('visible_entities', [])
        ]

        return AgentPerception(
            agent_godot_id=payload.get('agent_godot_id', 0),
            agent_name=payload.get('agent_name', ''),
            visible_entities=visible,
            occluded_entities=payload.get('occluded_entities', []),
            heard_utterances=payload.get('heard_utterances', []),
            own_position=Vector3(**payload.get('own_position', {})),
            own_velocity=Vector3(**payload.get('own_velocity', {})),
            own_orientation=Vector3(**payload.get('own_orientation', {})),
            energy_level=payload.get('energy_level', 1.0),
            held_object=payload.get('held_object'),
            current_institution=payload.get('current_institution'),
            timestamp=payload.get('timestamp', 0.0),
        )

    def register_agent(
        self,
        godot_id: int,
        agent_name: str,
        soul_map: Optional[Any] = None
    ) -> AgentState:
        """
        Register a new agent with the belief system.

        Args:
            godot_id: Godot node ID for the agent
            agent_name: Human-readable name
            soul_map: Optional SoulMap for psychology

        Returns:
            AgentState for the registered agent
        """
        if godot_id in self.agent_states:
            return self.agent_states[godot_id]

        # Assign index in belief network
        agent_index = self.next_agent_index
        if agent_index >= self.enhanced_config.num_agents:
            logger.warning(f"Max agents reached, cannot register {agent_name}")
            # Expand network if needed
            self._expand_belief_network()
            agent_index = self.next_agent_index

        self.next_agent_index += 1
        self.godot_id_to_index[godot_id] = agent_index

        # Create agent state
        state = AgentState(
            godot_id=godot_id,
            agent_name=agent_name,
            agent_index=agent_index,
            soul_map=soul_map
        )

        # Create SoulMap if enabled and available
        if (LIMINAL_AVAILABLE and
            self.enhanced_config.enable_soul_maps and
            soul_map is None):
            try:
                state.soul_map = SoulMap.generate_random()
            except Exception as e:
                logger.debug(f"Could not create SoulMap: {e}")

        self.agent_states[godot_id] = state

        logger.info(f"Registered agent {agent_name} (godot_id={godot_id}, index={agent_index})")
        return state

    def _expand_belief_network(self):
        """Expand belief network capacity."""
        new_capacity = self.belief_network.num_agents * 2
        logger.info(f"Expanding belief network from {self.belief_network.num_agents} to {new_capacity}")

        # Create new larger network
        new_network = BeliefNetwork(
            num_agents=new_capacity,
            ontology_dim=self.belief_network.ontology_dim,
            max_order=self.belief_network.max_order
        )

        # Copy existing beliefs
        for i, old_state in enumerate(self.belief_network.agent_beliefs):
            new_network.agent_beliefs[i] = old_state

        self.belief_network = new_network
        self.enhanced_config.num_agents = new_capacity

    def _process_perception_for_beliefs(self, perception: AgentPerception):
        """Process an agent's perception to update beliefs."""
        godot_id = perception.agent_godot_id

        # Ensure agent is registered
        if godot_id not in self.agent_states:
            self.register_agent(godot_id, perception.agent_name)

        state = self.agent_states[godot_id]
        agent_index = state.agent_index

        # Update physical state
        state.position = perception.own_position
        state.velocity = perception.own_velocity
        state.orientation = perception.own_orientation

        # Track observed agents
        current_time = perception.timestamp

        for entity in perception.visible_entities:
            if entity.entity_type == 'agent':
                observed_id = entity.godot_id
                state.observed_agents[observed_id] = current_time

                # Update beliefs about observed agent
                self._update_beliefs_from_observation(
                    observer_id=godot_id,
                    observed_entity=entity,
                    context=perception
                )

        # Process for higher-order ToM
        if self.enhanced_config.enable_tom_inference:
            self._infer_tom_from_perception(state, perception)

    def _update_beliefs_from_observation(
        self,
        observer_id: int,
        observed_entity: EntityUpdate,
        context: AgentPerception
    ):
        """
        Update observer's beliefs based on observing an entity.

        This implements 0th and 1st order belief updates:
        - 0th order: What the observer directly sees
        - 1st order: What the observer believes about the observed
        """
        if observer_id not in self.agent_states:
            return

        observer_index = self.godot_id_to_index.get(observer_id)
        if observer_index is None:
            return

        # Get or register observed agent
        observed_id = observed_entity.godot_id
        if observed_id not in self.agent_states:
            self.register_agent(observed_id, observed_entity.name)

        observed_index = self.godot_id_to_index.get(observed_id)
        if observed_index is None:
            return

        # Create belief content tensor
        belief_content = self._encode_entity_to_belief(observed_entity)

        # 0th order: Direct observation
        self.belief_network.update_agent_belief(
            agent_id=observer_index,
            order=0,
            target=observed_index,
            content=belief_content,
            confidence=1.0,  # Direct observation is high confidence
            source="perception"
        )

        # 1st order: Belief about observed agent's state
        inferred_state = self._infer_agent_state(observed_entity, context)
        if inferred_state is not None:
            self.belief_network.update_agent_belief(
                agent_id=observer_index,
                order=1,
                target=observed_index,
                content=inferred_state,
                confidence=0.7,  # Inference has lower confidence
                source="inference"
            )

        # Notify callbacks
        event = BeliefUpdateEvent(
            observer_id=observer_id,
            target_id=observed_id,
            belief_order=1,
            content=belief_content,
            confidence=0.7,
            source="observation"
        )
        self._notify_belief_update(event)

        if self.enhanced_config.log_belief_updates:
            logger.debug(
                f"Belief update: Agent {observer_id} updated beliefs about {observed_id}"
            )

    def _encode_entity_to_belief(self, entity: EntityUpdate) -> torch.Tensor:
        """Encode an entity update to a belief content tensor."""
        ontology_dim = self.belief_network.ontology_dim

        # Create feature vector
        features = torch.zeros(ontology_dim)

        # Position features (normalized)
        features[0] = entity.position.x / 100.0
        features[1] = entity.position.y / 100.0
        features[2] = entity.position.z / 100.0

        # Velocity features
        features[3] = entity.velocity.x / 10.0
        features[4] = entity.velocity.y / 10.0
        features[5] = entity.velocity.z / 10.0

        # State features
        features[6] = 1.0 if entity.visible else 0.0
        features[7] = 1.0 if entity.is_interactable else 0.0
        features[8] = 1.0 if entity.is_being_held else 0.0

        # Semantic tag encoding (simple hash-based)
        for i, tag in enumerate(entity.semantic_tags[:5]):
            tag_hash = hash(tag) % 1000 / 1000.0
            features[10 + i] = tag_hash

        # Affordance encoding
        for i, affordance in enumerate(entity.affordances[:5]):
            aff_hash = hash(affordance) % 1000 / 1000.0
            features[20 + i] = aff_hash

        return features

    def _infer_agent_state(
        self,
        entity: EntityUpdate,
        context: AgentPerception
    ) -> Optional[torch.Tensor]:
        """
        Infer an agent's internal state from external observation.

        This uses physical cues to infer psychological state.
        """
        if entity.entity_type != 'agent':
            return None

        ontology_dim = self.belief_network.ontology_dim
        inferred = torch.zeros(ontology_dim)

        # Infer from velocity (high velocity might indicate urgency)
        speed = (entity.velocity.x**2 + entity.velocity.y**2 + entity.velocity.z**2) ** 0.5
        inferred[0] = min(speed / 5.0, 1.0)  # Normalized urgency

        # Infer from position relative to observer
        distance = entity.position.distance_to(context.own_position)
        inferred[1] = max(0, 1.0 - distance / 50.0)  # Proximity

        # Infer from held object
        if entity.is_being_held:
            inferred[2] = 1.0  # Has object

        # Infer social orientation from facing direction
        # (Would need rotation data for full implementation)
        inferred[3] = 0.5  # Default neutral orientation

        return inferred

    def _infer_tom_from_perception(
        self,
        state: AgentState,
        perception: AgentPerception
    ):
        """
        Perform Theory of Mind inference from perception.

        This implements higher-order belief reasoning:
        - 2nd order: I believe you believe X
        - 3rd+ order: I believe you believe they believe X
        """
        observer_index = state.agent_index

        for entity in perception.visible_entities:
            if entity.entity_type != 'agent':
                continue

            observed_id = entity.godot_id
            if observed_id not in self.godot_id_to_index:
                continue

            observed_index = self.godot_id_to_index[observed_id]

            # Get observed agent's last known perception
            if observed_id in self.agent_states:
                observed_state = self.agent_states[observed_id]

                # 2nd order: What does observed agent believe?
                # Use their visibility to infer what they might believe
                self._infer_second_order_beliefs(
                    observer_index=observer_index,
                    observed_index=observed_index,
                    observed_state=observed_state,
                    current_context=perception
                )

                # 3rd+ order: Recursive inference (limited depth)
                if self.enhanced_config.max_tom_order >= 3:
                    self._infer_higher_order_beliefs(
                        observer_index=observer_index,
                        chain=[observed_index],
                        max_depth=min(self.enhanced_config.max_tom_order, 5),
                        context=perception
                    )

    def _infer_second_order_beliefs(
        self,
        observer_index: int,
        observed_index: int,
        observed_state: AgentState,
        current_context: AgentPerception
    ):
        """Infer second-order beliefs: I believe you believe X."""

        # Get what the observed agent can see
        if observed_state.last_perception is None:
            return

        # For each entity the observed agent saw
        # We don't have direct access to their perception, so we estimate
        # based on their position and the world state

        # Create estimated belief about what they believe about us
        # (the observer)
        observer_id = current_context.agent_godot_id

        # Estimate: Can observed agent see observer?
        distance = current_context.own_position.distance_to(observed_state.position)
        can_see_observer = distance < 30.0  # Visibility range

        if can_see_observer:
            # Observed agent probably has a belief about observer
            estimated_belief = self._encode_self_to_belief(current_context)

            # Decay confidence with ToM order
            confidence = 0.5 * (0.7 ** 2)  # 2nd order decay

            if confidence > self.enhanced_config.tom_confidence_threshold:
                self.belief_network.update_agent_belief(
                    agent_id=observer_index,
                    order=2,
                    target=observed_index,
                    content=estimated_belief,
                    confidence=confidence,
                    source="tom_inference"
                )

                if self.enhanced_config.log_tom_inferences:
                    logger.debug(
                        f"2nd order ToM: Agent {observer_index} infers "
                        f"agent {observed_index}'s beliefs"
                    )

    def _encode_self_to_belief(self, perception: AgentPerception) -> torch.Tensor:
        """Encode self-state to belief tensor."""
        ontology_dim = self.belief_network.ontology_dim
        features = torch.zeros(ontology_dim)

        features[0] = perception.own_position.x / 100.0
        features[1] = perception.own_position.y / 100.0
        features[2] = perception.own_position.z / 100.0
        features[3] = perception.energy_level
        features[4] = 1.0 if perception.held_object else 0.0

        return features

    def _infer_higher_order_beliefs(
        self,
        observer_index: int,
        chain: List[int],
        max_depth: int,
        context: AgentPerception
    ):
        """
        Recursively infer higher-order beliefs.

        Chain represents the belief chain: [A, B] means "A's beliefs about B"
        """
        current_order = len(chain) + 1
        if current_order > max_depth:
            return

        # Get last agent in chain
        last_index = chain[-1]

        # Find agents that the last agent might have beliefs about
        # (Based on proximity and visibility)
        potential_targets = self._get_potential_belief_targets(last_index)

        for target_index in potential_targets:
            if target_index in chain:
                continue  # Avoid cycles

            # Create belief content with decayed confidence
            confidence = 0.5 * (0.7 ** current_order)

            if confidence < self.enhanced_config.tom_confidence_threshold:
                continue

            # Estimate belief content
            belief_content = self._estimate_belief_content(
                chain + [target_index],
                context
            )

            self.belief_network.update_agent_belief(
                agent_id=observer_index,
                order=current_order,
                target=target_index,
                content=belief_content,
                confidence=confidence,
                source=f"tom_order_{current_order}"
            )

            # Recurse
            self._infer_higher_order_beliefs(
                observer_index=observer_index,
                chain=chain + [target_index],
                max_depth=max_depth,
                context=context
            )

    def _get_potential_belief_targets(self, agent_index: int) -> List[int]:
        """Get agents that a given agent might have beliefs about."""
        # Find the godot_id for this agent
        godot_id = None
        for gid, idx in self.godot_id_to_index.items():
            if idx == agent_index:
                godot_id = gid
                break

        if godot_id is None or godot_id not in self.agent_states:
            return []

        state = self.agent_states[godot_id]

        # Return recently observed agents
        return [
            self.godot_id_to_index.get(obs_id)
            for obs_id in state.observed_agents.keys()
            if obs_id in self.godot_id_to_index
        ]

    def _estimate_belief_content(
        self,
        chain: List[int],
        context: AgentPerception
    ) -> torch.Tensor:
        """Estimate belief content for a belief chain."""
        # Simplified: Return a noisy version of direct observation
        ontology_dim = self.belief_network.ontology_dim

        # Start with prior or random
        content = torch.randn(ontology_dim) * 0.1

        # Add noise proportional to chain length
        noise_scale = 0.1 * len(chain)
        content += torch.randn(ontology_dim) * noise_scale

        return content

    def _process_interaction_for_beliefs(self, payload: Dict):
        """Process an interaction event for belief updates."""
        agent_id = payload.get('agent_godot_id')
        target_id = payload.get('target_godot_id')
        interaction_type = payload.get('interaction_type')

        if agent_id is None or target_id is None:
            return

        # Update social interaction history
        if agent_id in self.agent_states:
            self.agent_states[agent_id].social_interactions.append({
                'target': target_id,
                'type': interaction_type,
                'timestamp': datetime.now().isoformat(),
                'payload': payload
            })

        # Update beliefs about target based on interaction
        if target_id in self.godot_id_to_index:
            agent_index = self.godot_id_to_index.get(agent_id)
            target_index = self.godot_id_to_index.get(target_id)

            if agent_index is not None and target_index is not None:
                # Create interaction-based belief update
                belief_content = self._encode_interaction_belief(
                    interaction_type,
                    payload.get('success', True)
                )

                self.belief_network.update_agent_belief(
                    agent_id=agent_index,
                    order=1,
                    target=target_index,
                    content=belief_content,
                    confidence=0.8,
                    source=f"interaction_{interaction_type}"
                )

    def _encode_interaction_belief(
        self,
        interaction_type: str,
        success: bool
    ) -> torch.Tensor:
        """Encode an interaction to belief content."""
        ontology_dim = self.belief_network.ontology_dim
        content = torch.zeros(ontology_dim)

        # Encode interaction type
        type_hash = hash(interaction_type) % (ontology_dim // 2)
        content[type_hash] = 1.0

        # Encode success
        content[ontology_dim - 1] = 1.0 if success else 0.0

        return content

    def _process_utterance_for_beliefs(self, payload: Dict):
        """Process an utterance event for belief updates."""
        speaker_id = payload.get('speaker_godot_id')
        text = payload.get('text', '')
        hearers = payload.get('hearers', [])

        if speaker_id is None:
            return

        speaker_index = self.godot_id_to_index.get(speaker_id)
        if speaker_index is None:
            return

        # All hearers update their beliefs about the speaker
        for hearer_id in hearers:
            hearer_index = self.godot_id_to_index.get(hearer_id)
            if hearer_index is not None:
                # Create utterance-based belief
                belief_content = self._encode_utterance_belief(text)

                self.belief_network.update_agent_belief(
                    agent_id=hearer_index,
                    order=1,
                    target=speaker_index,
                    content=belief_content,
                    confidence=0.9,  # Verbal communication is reliable
                    source="utterance"
                )

    def _encode_utterance_belief(self, text: str) -> torch.Tensor:
        """Encode an utterance to belief content."""
        ontology_dim = self.belief_network.ontology_dim
        content = torch.zeros(ontology_dim)

        # Simple encoding: hash-based word features
        words = text.lower().split()
        for i, word in enumerate(words[:10]):
            idx = hash(word) % ontology_dim
            content[idx] += 1.0

        # Normalize
        if content.norm() > 0:
            content = content / content.norm()

        return content

    def on_belief_update(
        self,
        callback: Callable[[BeliefUpdateEvent], None]
    ):
        """Register a callback for belief updates."""
        self._belief_callbacks.append(callback)

    def on_tom_inference(
        self,
        callback: Callable[[int, int, int, float], None]
    ):
        """Register a callback for ToM inferences (observer, target, order, confidence)."""
        self._tom_callbacks.append(callback)

    def _notify_belief_update(self, event: BeliefUpdateEvent):
        """Notify all belief update callbacks."""
        for callback in self._belief_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Belief callback error: {e}")

    def get_agent_belief(
        self,
        observer_godot_id: int,
        target_godot_id: int,
        order: int = 1
    ) -> Optional[Belief]:
        """
        Get a specific belief.

        Args:
            observer_godot_id: Godot ID of observing agent
            target_godot_id: Godot ID of target agent
            order: Belief order (1 = I believe X, 2 = I believe you believe X, etc.)

        Returns:
            Belief if exists, None otherwise
        """
        observer_index = self.godot_id_to_index.get(observer_godot_id)
        target_index = self.godot_id_to_index.get(target_godot_id)

        if observer_index is None or target_index is None:
            return None

        belief_state = self.belief_network.get_agent_belief_state(observer_index)
        if belief_state is None:
            return None

        return belief_state.get_belief(order, target_index)

    def get_all_beliefs_about(
        self,
        target_godot_id: int
    ) -> Dict[int, Dict[int, Belief]]:
        """
        Get all beliefs that all agents have about a target.

        Returns:
            Dict mapping observer_godot_id -> {order -> Belief}
        """
        target_index = self.godot_id_to_index.get(target_godot_id)
        if target_index is None:
            return {}

        result = {}
        for godot_id, state in self.agent_states.items():
            observer_index = state.agent_index
            beliefs = {}

            belief_state = self.belief_network.get_agent_belief_state(observer_index)
            if belief_state:
                for order in range(self.enhanced_config.max_tom_order + 1):
                    belief = belief_state.get_belief(order, target_index)
                    if belief is not None:
                        beliefs[order] = belief

            if beliefs:
                result[godot_id] = beliefs

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including belief system stats."""
        base_stats = super().get_statistics()

        # Belief system statistics
        total_beliefs = 0
        beliefs_by_order = {i: 0 for i in range(self.enhanced_config.max_tom_order + 1)}

        for agent_id in range(self.belief_network.num_agents):
            belief_state = self.belief_network.get_agent_belief_state(agent_id)
            if belief_state:
                for order in range(self.enhanced_config.max_tom_order + 1):
                    for target in range(self.belief_network.num_agents):
                        if belief_state.get_belief(order, target) is not None:
                            total_beliefs += 1
                            beliefs_by_order[order] += 1

        base_stats['belief_system'] = {
            'registered_agents': len(self.agent_states),
            'total_beliefs': total_beliefs,
            'beliefs_by_order': beliefs_by_order,
            'max_tom_order': self.enhanced_config.max_tom_order,
            'ontology_dim': self.belief_network.ontology_dim,
        }

        # Agent state statistics
        agent_stats = {}
        for godot_id, state in self.agent_states.items():
            agent_stats[godot_id] = {
                'name': state.agent_name,
                'observations': len(state.observed_agents),
                'interactions': len(state.social_interactions),
                'perception_history_size': len(state.perception_history),
            }

        base_stats['agent_states'] = agent_stats

        return base_stats

    def reset_beliefs(self):
        """Reset all beliefs to initial state."""
        self.belief_network = BeliefNetwork(
            num_agents=self.enhanced_config.num_agents,
            ontology_dim=self.enhanced_config.ontology_dim,
            max_order=self.enhanced_config.max_tom_order
        )

        # Clear agent observation histories
        for state in self.agent_states.values():
            state.observed_agents.clear()
            state.social_interactions.clear()
            state.perception_history.clear()

        logger.info("Belief system reset")

    def export_beliefs_to_json(self) -> str:
        """Export all beliefs to JSON format for debugging/visualization."""
        export = {
            'agents': {},
            'timestamp': datetime.now().isoformat(),
        }

        for godot_id, state in self.agent_states.items():
            agent_beliefs = {
                'name': state.agent_name,
                'index': state.agent_index,
                'beliefs': {}
            }

            belief_state = self.belief_network.get_agent_belief_state(state.agent_index)
            if belief_state:
                for order in range(self.enhanced_config.max_tom_order + 1):
                    order_beliefs = {}
                    for target_id, target_state in self.agent_states.items():
                        belief = belief_state.get_belief(order, target_state.agent_index)
                        if belief is not None:
                            order_beliefs[target_id] = {
                                'confidence': belief.confidence,
                                'timestamp': belief.timestamp,
                                'source': belief.source,
                            }
                    if order_beliefs:
                        agent_beliefs['beliefs'][f'order_{order}'] = order_beliefs

            export['agents'][godot_id] = agent_beliefs

        return json.dumps(export, indent=2)


def create_enhanced_server(
    host: str = "localhost",
    port: int = 9080,
    num_agents: int = 10,
    max_tom_order: int = 5,
    **kwargs
) -> EnhancedGodotServer:
    """
    Factory function to create an enhanced Godot server.

    Args:
        host: Server host
        port: Server port
        num_agents: Maximum number of agents
        max_tom_order: Maximum Theory of Mind order
        **kwargs: Additional configuration options

    Returns:
        Configured EnhancedGodotServer instance
    """
    config = EnhancedServerConfig(
        host=host,
        port=port,
        num_agents=num_agents,
        max_tom_order=max_tom_order,
        **kwargs
    )

    return EnhancedGodotServer(config)
