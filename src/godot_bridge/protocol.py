"""
Godot Communication Protocol

Defines the message format for bidirectional communication between
the Python cognitive controller and the Godot physics simulation.

Message Flow:
1. Godot -> Python: Entity updates, sensory input, world state
2. Python -> Godot: Agent commands, spawn requests, world modifications

All messages are JSON-encoded with type headers for dispatch.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Union
import json


class MessageType(Enum):
    """Types of messages in the Godot-Python protocol."""

    # Godot -> Python
    ENTITY_UPDATE = auto()  # Entity position/state changed
    AGENT_PERCEPTION = auto()  # What an agent perceives
    WORLD_STATE = auto()  # Full world state snapshot
    COLLISION_EVENT = auto()  # Collision occurred
    INTERACTION_EVENT = auto()  # Agent interacted with object
    UTTERANCE_EVENT = auto()  # Agent spoke

    # Python -> Godot
    AGENT_COMMAND = auto()  # Command for agent to execute
    SPAWN_ENTITY = auto()  # Spawn new entity
    MODIFY_ENTITY = auto()  # Modify existing entity
    WORLD_COMMAND = auto()  # Global world command

    # Bidirectional
    HEARTBEAT = auto()  # Connection keepalive
    ACK = auto()  # Acknowledgment
    ERROR = auto()  # Error message

    # Simulation control
    PAUSE = auto()
    RESUME = auto()
    RESET = auto()
    STEP = auto()  # Single step (for debugging)


@dataclass
class Vector3:
    """3D vector representation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vector3":
        return cls(x=t[0], y=t[1], z=t[2])

    def distance_to(self, other: "Vector3") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5


@dataclass
class EntityUpdate:
    """
    Update for a single entity in the simulation.

    Sent from Godot when entity state changes.
    """

    # Identity
    godot_id: int  # Unique Godot node ID
    entity_type: str  # "object", "agent", "location"
    name: str  # Human-readable name

    # Transform
    position: Vector3 = field(default_factory=Vector3)
    rotation: Vector3 = field(default_factory=Vector3)  # Euler angles
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    # Physics state
    velocity: Vector3 = field(default_factory=Vector3)
    is_static: bool = True

    # Visual properties
    visible: bool = True
    material_id: Optional[str] = None
    color: Optional[Tuple[float, float, float, float]] = None  # RGBA

    # Semantic hints (from Godot metadata)
    semantic_tags: List[str] = field(default_factory=list)
    affordances: List[str] = field(default_factory=list)

    # Interaction state
    is_interactable: bool = False
    is_being_held: bool = False
    held_by: Optional[int] = None  # Agent godot_id

    # Timestamp
    timestamp: float = 0.0  # Simulation time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "godot_id": self.godot_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "position": asdict(self.position),
            "rotation": asdict(self.rotation),
            "velocity": asdict(self.velocity),
            "is_static": self.is_static,
            "visible": self.visible,
            "semantic_tags": self.semantic_tags,
            "affordances": self.affordances,
            "is_interactable": self.is_interactable,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityUpdate":
        return cls(
            godot_id=d["godot_id"],
            entity_type=d["entity_type"],
            name=d["name"],
            position=Vector3(**d.get("position", {})),
            rotation=Vector3(**d.get("rotation", {})),
            velocity=Vector3(**d.get("velocity", {})),
            is_static=d.get("is_static", True),
            visible=d.get("visible", True),
            semantic_tags=d.get("semantic_tags", []),
            affordances=d.get("affordances", []),
            is_interactable=d.get("is_interactable", False),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class AgentPerception:
    """
    What an agent perceives at a given moment.

    Sent from Godot with filtered/occluded vision.
    """

    # Perceiving agent
    agent_godot_id: int
    agent_name: str

    # Visual perception
    visible_entities: List[EntityUpdate] = field(default_factory=list)
    occluded_entities: List[int] = field(default_factory=list)  # IDs of known but not visible

    # Auditory perception
    heard_utterances: List[Dict[str, Any]] = field(default_factory=list)
    ambient_sounds: List[str] = field(default_factory=list)

    # Proprioception
    own_position: Vector3 = field(default_factory=Vector3)
    own_velocity: Vector3 = field(default_factory=Vector3)
    own_orientation: Vector3 = field(default_factory=Vector3)

    # Internal state
    energy_level: float = 1.0
    held_object: Optional[int] = None  # What agent is holding

    # Context
    current_location_id: Optional[int] = None
    current_institution: Optional[str] = None  # Detected from location

    # Timestamp
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_godot_id": self.agent_godot_id,
            "agent_name": self.agent_name,
            "visible_entities": [e.to_dict() for e in self.visible_entities],
            "occluded_entities": self.occluded_entities,
            "heard_utterances": self.heard_utterances,
            "own_position": asdict(self.own_position),
            "own_velocity": asdict(self.own_velocity),
            "energy_level": self.energy_level,
            "held_object": self.held_object,
            "current_institution": self.current_institution,
            "timestamp": self.timestamp,
        }


@dataclass
class WorldState:
    """
    Complete snapshot of the simulation world.

    Sent periodically or on request for synchronization.
    """

    # All entities
    entities: List[EntityUpdate] = field(default_factory=list)

    # Agents specifically
    agents: List[Dict[str, Any]] = field(default_factory=list)

    # Locations/Zones
    locations: List[Dict[str, Any]] = field(default_factory=list)

    # Global state
    simulation_time: float = 0.0
    timestep: int = 0
    is_paused: bool = False

    # Environment conditions
    time_of_day: float = 12.0  # 0-24 hours
    weather: str = "clear"

    # Active institution (global context)
    active_institution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "agents": self.agents,
            "locations": self.locations,
            "simulation_time": self.simulation_time,
            "timestep": self.timestep,
            "is_paused": self.is_paused,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "active_institution": self.active_institution,
        }


@dataclass
class AgentCommand:
    """
    Command for an agent to execute in Godot.

    Sent from Python to control agent behavior.
    """

    # Target agent
    agent_godot_id: int

    # Command type
    command_type: str  # "move", "interact", "speak", "look", etc.

    # Command parameters
    target_position: Optional[Vector3] = None
    target_entity_id: Optional[int] = None
    utterance_text: Optional[str] = None
    animation_name: Optional[str] = None
    speed: float = 1.0

    # Execution parameters
    priority: int = 0  # Higher = more urgent
    interruptible: bool = True
    timeout_seconds: float = 10.0

    # Metadata
    command_id: str = ""  # For tracking/acknowledgment
    reason: str = ""  # Why this command (for logging)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "agent_godot_id": self.agent_godot_id,
            "command_type": self.command_type,
            "priority": self.priority,
            "interruptible": self.interruptible,
            "timeout_seconds": self.timeout_seconds,
            "command_id": self.command_id,
            "reason": self.reason,
        }
        if self.target_position:
            d["target_position"] = asdict(self.target_position)
        if self.target_entity_id is not None:
            d["target_entity_id"] = self.target_entity_id
        if self.utterance_text:
            d["utterance_text"] = self.utterance_text
        if self.animation_name:
            d["animation_name"] = self.animation_name
        return d


@dataclass
class InteractionEvent:
    """
    Event when an agent interacts with an entity.
    """

    agent_godot_id: int
    target_godot_id: int
    interaction_type: str  # "pick_up", "put_down", "use", "examine", etc.
    success: bool = True
    result_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class UtteranceEvent:
    """
    Event when an agent speaks.
    """

    speaker_godot_id: int
    text: str
    volume: float = 1.0  # Affects who can hear
    target_agent_id: Optional[int] = None  # Directed speech
    hearers: List[int] = field(default_factory=list)  # Who heard it
    timestamp: float = 0.0


@dataclass
class GodotMessage:
    """
    Wrapper for all Godot-Python messages.

    Provides consistent serialization and type dispatch.
    """

    message_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    sequence_id: int = 0  # For ordering/ack

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "type": self.message_type.name,
                "payload": self.payload,
                "timestamp": self.timestamp,
                "sequence_id": self.sequence_id,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "GodotMessage":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(
            message_type=MessageType[d["type"]],
            payload=d.get("payload", {}),
            timestamp=d.get("timestamp", 0.0),
            sequence_id=d.get("sequence_id", 0),
        )

    @classmethod
    def create_entity_update(cls, update: EntityUpdate) -> "GodotMessage":
        """Create entity update message."""
        return cls(
            message_type=MessageType.ENTITY_UPDATE,
            payload=update.to_dict(),
        )

    @classmethod
    def create_agent_command(cls, command: AgentCommand) -> "GodotMessage":
        """Create agent command message."""
        return cls(
            message_type=MessageType.AGENT_COMMAND,
            payload=command.to_dict(),
        )

    @classmethod
    def create_world_state(cls, state: WorldState) -> "GodotMessage":
        """Create world state message."""
        return cls(
            message_type=MessageType.WORLD_STATE,
            payload=state.to_dict(),
        )

    @classmethod
    def create_heartbeat(cls) -> "GodotMessage":
        """Create heartbeat message."""
        return cls(
            message_type=MessageType.HEARTBEAT,
            payload={"ping": True},
        )

    @classmethod
    def create_error(cls, error_msg: str, code: int = 0) -> "GodotMessage":
        """Create error message."""
        return cls(
            message_type=MessageType.ERROR,
            payload={"error": error_msg, "code": code},
        )
