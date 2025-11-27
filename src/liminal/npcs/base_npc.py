"""
Base NPC Class for Liminal Architectures

Every NPC in the game - from main cast to random pedestrians - is an instance
of this class, containing:
- Soul Map: 60-dimensional psychological state
- Current state: active goal, emotional state, awareness
- Behaviors: How they respond to stimuli
- Beliefs: What they know/believe about others (ToM integration)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING
from enum import Enum, auto
import uuid
import random

if TYPE_CHECKING:
    from ..soul_map import SoulMap, SoulMapDelta
    from ..realms import Realm, RealmType


class NPCState(Enum):
    """Current activity state of an NPC."""
    IDLE = "idle"
    WALKING = "walking"
    WORKING = "working"
    TALKING = "talking"
    OBSERVING = "observing"
    FLEEING = "fleeing"
    HIDING = "hiding"
    INVESTIGATING = "investigating"
    LOOPING = "looping"  # For Spleen Towns NPCs stuck in time loops
    GLITCHING = "glitching"  # For Nothing NPCs
    CONSUMING = "consuming"  # For Hollow Reaches hive-mind
    FILING = "filing"  # For Ministry bureaucrats


class NPCBehavior(Enum):
    """Behavioral patterns NPCs can exhibit."""
    PATROL = "patrol"
    WANDER = "wander"
    STATIONARY = "stationary"
    FOLLOW = "follow"
    AVOID = "avoid"
    APPROACH = "approach"
    CONVERSE = "converse"
    LOOP_ACTIVITY = "loop_activity"
    ENFORCE = "enforce"  # For Parameter Enforcers
    ADAPT = "adapt"  # For Adaptives


@dataclass
class NPCBelief:
    """A belief an NPC holds about another entity."""
    target_id: str  # Who/what the belief is about
    belief_type: str  # e.g., "location", "intention", "state"
    content: Any  # The actual belief content
    confidence: float  # 0-1 confidence in this belief
    timestamp: int  # When this belief was formed/updated
    source: str  # How they know this: "observation", "inference", "told"


@dataclass
class NPCMemory:
    """Short-term memory for NPCs."""
    observations: List[Dict[str, Any]] = field(default_factory=list)
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    max_observations: int = 20
    max_interactions: int = 10

    def add_observation(self, observation: Dict[str, Any]) -> None:
        """Add an observation, maintaining max size."""
        self.observations.append(observation)
        if len(self.observations) > self.max_observations:
            self.observations.pop(0)

    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add an interaction record."""
        self.interactions.append(interaction)
        if len(self.interactions) > self.max_interactions:
            self.interactions.pop(0)


@dataclass
class BaseNPC:
    """
    Core NPC class for Liminal Architectures.

    Contains the Soul Map, behavioral systems, and integration points
    for the NAS backend to control NPC behavior.
    """

    # Identity
    npc_id: str
    name: str
    archetype: str

    # Psychological state
    soul_map: "SoulMap"

    # Current state
    current_state: NPCState = NPCState.IDLE
    active_goal: str = ""
    emotional_state: str = "neutral"

    # Position and realm
    position: Tuple[float, float] = (0.0, 0.0)
    current_realm: Optional["RealmType"] = None

    # Awareness of player
    awareness_of_player: float = 0.0  # 0-1: how aware they are of player

    # Theory of Mind
    tom_depth: int = 3  # Max order of belief reasoning

    # Behavioral components
    current_behavior: NPCBehavior = NPCBehavior.WANDER
    behavior_target: Optional[str] = None

    # Beliefs about others
    beliefs: Dict[str, List[NPCBelief]] = field(default_factory=dict)

    # Memory
    memory: NPCMemory = field(default_factory=NPCMemory)

    # Social
    reputation_with_player: float = 0.5  # -1 to 1
    relationships: Dict[str, float] = field(default_factory=dict)  # NPC ID -> relationship value

    # Is this NPC a "zombie" (scripted, not genuine ToM)?
    is_zombie: bool = False
    zombie_type: Optional[str] = None

    # Energy/resources (for game mechanics)
    energy: float = 1.0

    # Whether this NPC is hero (hand-crafted) or systemic (procedural)
    is_hero: bool = False

    # Dialogue and quests
    dialogue_tree: Optional[Dict[str, Any]] = None
    quests_available: List[str] = field(default_factory=list)

    # NAS model reference (set during integration)
    _nas_model: Optional[Any] = None

    def __post_init__(self):
        """Initialize derived values."""
        if self.npc_id is None:
            self.npc_id = str(uuid.uuid4())[:8]

        # Update ToM depth from Soul Map
        if hasattr(self.soul_map, 'get_tom_depth_int'):
            self.tom_depth = self.soul_map.get_tom_depth_int()

    @classmethod
    def create(cls, name: str, archetype: str, soul_map: "SoulMap", **kwargs) -> "BaseNPC":
        """Factory method to create an NPC."""
        npc_id = kwargs.pop('npc_id', f"{archetype}_{str(uuid.uuid4())[:6]}")
        return cls(
            npc_id=npc_id,
            name=name,
            archetype=archetype,
            soul_map=soul_map,
            **kwargs
        )

    def to_json(self) -> Dict[str, Any]:
        """Convert NPC to JSON format (matching MDD spec)."""
        return {
            "npc_id": self.npc_id,
            "name": self.name,
            "archetype": self.archetype,
            "tom_depth": self.tom_depth,
            "current_state": {
                "active_goal": self.active_goal,
                "emotional_state": self.emotional_state,
                "awareness_of_player": self.awareness_of_player,
                "state": self.current_state.value,
            },
            "soul_map": self.soul_map.to_json(),
            "position": self.position,
            "current_realm": self.current_realm.value if self.current_realm else None,
            "is_hero": self.is_hero,
            "is_zombie": self.is_zombie,
        }

    def to_observation_tensor(self) -> torch.Tensor:
        """
        Convert NPC state to tensor for NAS model input.

        Returns a tensor combining Soul Map (65 dims) + context (additional dims).
        """
        # Soul Map tensor (65 dims)
        soul_tensor = self.soul_map.to_tensor()

        # Context tensor
        context = torch.tensor([
            self.awareness_of_player,
            self.energy,
            self.reputation_with_player,
            float(self.current_state.value == NPCState.FLEEING.value),
            float(self.is_zombie),
        ], dtype=torch.float32)

        # Combine
        return torch.cat([soul_tensor, context])

    def update_belief(self, target_id: str, belief_type: str, content: Any,
                      confidence: float, timestamp: int, source: str = "observation") -> None:
        """Update or add a belief about another entity."""
        if target_id not in self.beliefs:
            self.beliefs[target_id] = []

        # Check if belief of this type exists
        for belief in self.beliefs[target_id]:
            if belief.belief_type == belief_type:
                belief.content = content
                belief.confidence = confidence
                belief.timestamp = timestamp
                belief.source = source
                return

        # Add new belief
        self.beliefs[target_id].append(NPCBelief(
            target_id=target_id,
            belief_type=belief_type,
            content=content,
            confidence=confidence,
            timestamp=timestamp,
            source=source
        ))

    def get_belief(self, target_id: str, belief_type: str) -> Optional[NPCBelief]:
        """Get a specific belief about a target."""
        if target_id not in self.beliefs:
            return None

        for belief in self.beliefs[target_id]:
            if belief.belief_type == belief_type:
                return belief
        return None

    def perceive(self, observation: Dict[str, Any], timestamp: int) -> None:
        """Process an observation from the environment."""
        self.memory.add_observation({**observation, "timestamp": timestamp})

        # Update awareness of player if player is in observation
        if "player" in observation:
            player_data = observation["player"]
            distance = player_data.get("distance", 100.0)

            # Awareness decreases with distance
            new_awareness = max(0.0, 1.0 - distance / 50.0)

            # Modulate by social monitoring
            new_awareness *= (0.5 + 0.5 * self.soul_map.social["social_monitoring"])

            self.awareness_of_player = max(self.awareness_of_player * 0.9, new_awareness)

    def decide_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide on an action based on current state and context.

        This is where the NAS model would be called for genuine ToM reasoning.
        For non-NAS NPCs, uses simple rule-based behavior.
        """
        # If NAS model is attached, use it
        if self._nas_model is not None:
            return self._nas_action(context)

        # Simple rule-based fallback
        return self._rule_based_action(context)

    def _nas_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from NAS model."""
        # Convert state to tensor
        obs_tensor = self.to_observation_tensor()

        # Get context tensor (environment, nearby NPCs, etc.)
        context_tensor = self._encode_context(context)

        # Combine and forward through NAS model
        input_tensor = torch.cat([obs_tensor, context_tensor]).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self._nas_model(input_tensor)

        # Interpret output
        beliefs = output.get('beliefs', torch.zeros(1, 60))
        actions = output.get('actions', torch.zeros(1))

        return {
            'action_type': self._interpret_action(actions[0].item()),
            'beliefs_update': beliefs[0],
            'confidence': output.get('confidence', 0.5),
        }

    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode environmental context as tensor."""
        # Encode relevant context features
        features = [
            context.get('threat_level', 0.0),
            context.get('social_density', 0.0),
            context.get('realm_vibe_match', 0.5),
            context.get('time_of_day', 0.5),
        ]

        # Pad to consistent size
        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32)

    def _interpret_action(self, action_value: float) -> str:
        """Interpret action value as action type."""
        if action_value < 0.2:
            return "flee"
        elif action_value < 0.4:
            return "avoid"
        elif action_value < 0.6:
            return "observe"
        elif action_value < 0.8:
            return "approach"
        else:
            return "interact"

    def _rule_based_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based action selection (for non-NAS NPCs)."""
        # High anxiety + threat = flee
        if (self.soul_map.emotional["anxiety_baseline"] > 0.7 and
            context.get('threat_level', 0.0) > 0.5):
            return {'action_type': 'flee', 'confidence': 0.8}

        # Low trust + stranger = avoid
        if (self.soul_map.social["trust_default"] < 0.3 and
            self.awareness_of_player > 0.5):
            return {'action_type': 'avoid', 'confidence': 0.6}

        # High novelty + unknown = investigate
        if (self.soul_map.motivational["novelty_drive"] > 0.7 and
            context.get('novel_stimulus', False)):
            return {'action_type': 'investigate', 'confidence': 0.7}

        # Default: continue current behavior
        return {'action_type': 'continue', 'confidence': 0.5}

    def apply_intervention(self, delta: "SoulMapDelta", source: str = "player") -> Dict[str, Any]:
        """
        Apply a psychological intervention (cognitive hazard) to this NPC.

        Returns a report of what changed.
        """
        # Get pre-intervention state
        pre_stability = self.soul_map.compute_stability()
        pre_threat = self.soul_map.compute_threat_response()

        # Apply the delta
        delta.apply_to(self.soul_map)

        # Get post-intervention state
        post_stability = self.soul_map.compute_stability()
        post_threat = self.soul_map.compute_threat_response()

        # Record this intervention
        self.memory.add_interaction({
            'type': 'intervention',
            'source': source,
            'stability_change': post_stability - pre_stability,
            'threat_change': post_threat - pre_threat,
        })

        # Update emotional state based on changes
        if post_threat > pre_threat + 0.2:
            self.emotional_state = "alarmed"
        elif post_stability < pre_stability - 0.2:
            self.emotional_state = "destabilized"
        elif post_stability > pre_stability + 0.1:
            self.emotional_state = "calmed"

        return {
            'stability_change': post_stability - pre_stability,
            'threat_change': post_threat - pre_threat,
            'new_emotional_state': self.emotional_state,
        }

    def update_from_realm(self, realm: "Realm", duration: float = 1.0) -> Dict[str, float]:
        """Apply realm ambient effects to this NPC."""
        changes = realm.apply_ambient_effects(self.soul_map, duration)

        # Update realm-specific modifier
        if realm.realm_type.value == "ministry":
            # Corporeal Certainty decays if not filing
            if self.current_state != NPCState.FILING:
                current = self.soul_map.realm_modifiers.get("corporeal_certainty", 1.0)
                self.soul_map.realm_modifiers["corporeal_certainty"] = max(0.0, current - 0.01 * duration)

        elif realm.realm_type.value == "hollow":
            # Corruption increases
            current = self.soul_map.realm_modifiers.get("corruption", 0.0)
            resistance = self.soul_map.motivational["survival_drive"]
            corruption_rate = realm.realm_variables.get("corruption_rate", 0.02)
            self.soul_map.realm_modifiers["corruption"] = min(1.0, current + corruption_rate * (1 - resistance) * duration)

        return changes

    def get_analysis_display(self) -> Dict[str, Any]:
        """
        Get Soul Map data for HUD display (Soul Scanner).

        Returns organized data for visualization.
        """
        return {
            "name": self.name,
            "archetype": self.archetype,
            "tom_depth": self.tom_depth,
            "current_state": self.current_state.value,
            "emotional_state": self.emotional_state,
            "stability": self.soul_map.compute_stability(),
            "threat_response": self.soul_map.compute_threat_response(),
            "social_openness": self.soul_map.compute_social_openness(),
            "dominant_motivation": self.soul_map.get_dominant_motivation(),
            "clusters": {
                "cognitive": dict(self.soul_map.cognitive),
                "emotional": dict(self.soul_map.emotional),
                "motivational": dict(self.soul_map.motivational),
                "social": dict(self.soul_map.social),
                "self": dict(self.soul_map.self),
            },
            "realm_modifiers": dict(self.soul_map.realm_modifiers),
        }

    def __repr__(self) -> str:
        return (f"NPC({self.name}, {self.archetype}, "
                f"state={self.current_state.value}, "
                f"emotional={self.emotional_state})")


# Export
__all__ = [
    'BaseNPC',
    'NPCState',
    'NPCBehavior',
    'NPCBelief',
    'NPCMemory',
]
