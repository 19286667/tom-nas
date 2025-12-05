extends Node
## Protocol Handler - Message Serialization and Validation
##
## Provides type-safe message creation and parsing for the ToM-NAS protocol.
## Ensures all messages conform to the expected format.

class_name ProtocolHandlerClass

# Message types enum (mirrors Python MessageType)
enum MessageType {
	# Godot -> Python
	ENTITY_UPDATE,
	AGENT_PERCEPTION,
	WORLD_STATE,
	COLLISION_EVENT,
	INTERACTION_EVENT,
	UTTERANCE_EVENT,

	# Python -> Godot
	AGENT_COMMAND,
	SPAWN_ENTITY,
	MODIFY_ENTITY,
	WORLD_COMMAND,

	# Bidirectional
	HEARTBEAT,
	ACK,
	ERROR,

	# Simulation control
	PAUSE,
	RESUME,
	RESET,
	STEP
}

# Command types for agents
enum CommandType {
	MOVE,
	TURN,
	FOLLOW,
	FLEE,
	PICK_UP,
	PUT_DOWN,
	USE,
	EXAMINE,
	GIVE,
	SPEAK,
	GESTURE,
	LOOK_AT,
	WAIT,
	CANCEL
}

# Entity types
enum EntityType {
	OBJECT,
	AGENT,
	LOCATION,
	TRIGGER,
	ITEM
}

# Message type string names
const MESSAGE_TYPE_NAMES: Dictionary = {
	MessageType.ENTITY_UPDATE: "ENTITY_UPDATE",
	MessageType.AGENT_PERCEPTION: "AGENT_PERCEPTION",
	MessageType.WORLD_STATE: "WORLD_STATE",
	MessageType.COLLISION_EVENT: "COLLISION_EVENT",
	MessageType.INTERACTION_EVENT: "INTERACTION_EVENT",
	MessageType.UTTERANCE_EVENT: "UTTERANCE_EVENT",
	MessageType.AGENT_COMMAND: "AGENT_COMMAND",
	MessageType.SPAWN_ENTITY: "SPAWN_ENTITY",
	MessageType.MODIFY_ENTITY: "MODIFY_ENTITY",
	MessageType.WORLD_COMMAND: "WORLD_COMMAND",
	MessageType.HEARTBEAT: "HEARTBEAT",
	MessageType.ACK: "ACK",
	MessageType.ERROR: "ERROR",
	MessageType.PAUSE: "PAUSE",
	MessageType.RESUME: "RESUME",
	MessageType.RESET: "RESET",
	MessageType.STEP: "STEP"
}

const COMMAND_TYPE_NAMES: Dictionary = {
	CommandType.MOVE: "move",
	CommandType.TURN: "turn",
	CommandType.FOLLOW: "follow",
	CommandType.FLEE: "flee",
	CommandType.PICK_UP: "pick_up",
	CommandType.PUT_DOWN: "put_down",
	CommandType.USE: "use",
	CommandType.EXAMINE: "examine",
	CommandType.GIVE: "give",
	CommandType.SPEAK: "speak",
	CommandType.GESTURE: "gesture",
	CommandType.LOOK_AT: "look_at",
	CommandType.WAIT: "wait",
	CommandType.CANCEL: "cancel"
}

# Create entity update payload
static func create_entity_update(
	godot_id: int,
	entity_type: String,
	name: String,
	position: Vector3,
	rotation: Vector3 = Vector3.ZERO,
	scale: Vector3 = Vector3.ONE,
	velocity: Vector3 = Vector3.ZERO,
	is_static: bool = true,
	visible: bool = true,
	semantic_tags: Array = [],
	affordances: Array = [],
	is_interactable: bool = false,
	timestamp: float = 0.0
) -> Dictionary:
	return {
		"godot_id": godot_id,
		"entity_type": entity_type,
		"name": name,
		"position": vector3_to_dict(position),
		"rotation": vector3_to_dict(rotation),
		"scale": vector3_to_dict(scale),
		"velocity": vector3_to_dict(velocity),
		"is_static": is_static,
		"visible": visible,
		"semantic_tags": semantic_tags,
		"affordances": affordances,
		"is_interactable": is_interactable,
		"timestamp": timestamp
	}

# Create agent perception payload
static func create_agent_perception(
	agent_godot_id: int,
	agent_name: String,
	visible_entities: Array = [],
	occluded_entities: Array = [],
	heard_utterances: Array = [],
	own_position: Vector3 = Vector3.ZERO,
	own_velocity: Vector3 = Vector3.ZERO,
	own_orientation: Vector3 = Vector3.ZERO,
	energy_level: float = 1.0,
	held_object: int = -1,
	current_institution: String = "",
	timestamp: float = 0.0
) -> Dictionary:
	return {
		"agent_godot_id": agent_godot_id,
		"agent_name": agent_name,
		"visible_entities": visible_entities,
		"occluded_entities": occluded_entities,
		"heard_utterances": heard_utterances,
		"own_position": vector3_to_dict(own_position),
		"own_velocity": vector3_to_dict(own_velocity),
		"own_orientation": vector3_to_dict(own_orientation),
		"energy_level": energy_level,
		"held_object": held_object if held_object >= 0 else null,
		"current_institution": current_institution if current_institution else null,
		"timestamp": timestamp
	}

# Create world state payload
static func create_world_state(
	entities: Array = [],
	agents: Array = [],
	locations: Array = [],
	simulation_time: float = 0.0,
	timestep: int = 0,
	is_paused: bool = false,
	time_of_day: float = 12.0,
	weather: String = "clear",
	active_institution: String = ""
) -> Dictionary:
	return {
		"entities": entities,
		"agents": agents,
		"locations": locations,
		"simulation_time": simulation_time,
		"timestep": timestep,
		"is_paused": is_paused,
		"time_of_day": time_of_day,
		"weather": weather,
		"active_institution": active_institution if active_institution else null
	}

# Create interaction event payload
static func create_interaction_event(
	agent_godot_id: int,
	target_godot_id: int,
	interaction_type: String,
	success: bool = true,
	result_data: Dictionary = {},
	timestamp: float = 0.0
) -> Dictionary:
	return {
		"agent_godot_id": agent_godot_id,
		"target_godot_id": target_godot_id,
		"interaction_type": interaction_type,
		"success": success,
		"result_data": result_data,
		"timestamp": timestamp
	}

# Create utterance event payload
static func create_utterance_event(
	speaker_godot_id: int,
	text: String,
	volume: float = 1.0,
	target_agent_id: int = -1,
	hearers: Array = [],
	timestamp: float = 0.0
) -> Dictionary:
	return {
		"speaker_godot_id": speaker_godot_id,
		"text": text,
		"volume": volume,
		"target_agent_id": target_agent_id if target_agent_id >= 0 else null,
		"hearers": hearers,
		"timestamp": timestamp
	}

# Create collision event payload
static func create_collision_event(
	agent_godot_id: int,
	collider_godot_id: int,
	collision_point: Vector3 = Vector3.ZERO,
	collision_normal: Vector3 = Vector3.UP,
	timestamp: float = 0.0
) -> Dictionary:
	return {
		"agent_godot_id": agent_godot_id,
		"collider_godot_id": collider_godot_id,
		"collision_point": vector3_to_dict(collision_point),
		"collision_normal": vector3_to_dict(collision_normal),
		"timestamp": timestamp
	}

# Parse agent command from payload
static func parse_agent_command(payload: Dictionary) -> Dictionary:
	"""Parse and validate an agent command payload."""
	return {
		"agent_godot_id": payload.get("agent_godot_id", -1),
		"command_type": payload.get("command_type", "wait"),
		"target_position": dict_to_vector3(payload.get("target_position", {})) if payload.has("target_position") else null,
		"target_entity_id": payload.get("target_entity_id"),
		"utterance_text": payload.get("utterance_text"),
		"animation_name": payload.get("animation_name"),
		"speed": payload.get("speed", 1.0),
		"priority": payload.get("priority", 0),
		"interruptible": payload.get("interruptible", true),
		"timeout_seconds": payload.get("timeout_seconds", 10.0),
		"command_id": payload.get("command_id", ""),
		"reason": payload.get("reason", "")
	}

# Parse entity update from payload
static func parse_entity_update(payload: Dictionary) -> Dictionary:
	"""Parse and validate an entity update payload."""
	return {
		"godot_id": payload.get("godot_id", -1),
		"entity_type": payload.get("entity_type", "object"),
		"name": payload.get("name", ""),
		"position": dict_to_vector3(payload.get("position", {})),
		"rotation": dict_to_vector3(payload.get("rotation", {})),
		"scale": dict_to_vector3(payload.get("scale", {"x": 1, "y": 1, "z": 1})),
		"velocity": dict_to_vector3(payload.get("velocity", {})),
		"is_static": payload.get("is_static", true),
		"visible": payload.get("visible", true),
		"semantic_tags": payload.get("semantic_tags", []),
		"affordances": payload.get("affordances", []),
		"is_interactable": payload.get("is_interactable", false),
		"timestamp": payload.get("timestamp", 0.0)
	}

# Utility functions

static func vector3_to_dict(v: Vector3) -> Dictionary:
	"""Convert Vector3 to dictionary."""
	return {"x": v.x, "y": v.y, "z": v.z}

static func dict_to_vector3(d: Dictionary) -> Vector3:
	"""Convert dictionary to Vector3."""
	if d.is_empty():
		return Vector3.ZERO
	return Vector3(
		float(d.get("x", 0.0)),
		float(d.get("y", 0.0)),
		float(d.get("z", 0.0))
	)

static func color_to_array(c: Color) -> Array:
	"""Convert Color to RGBA array."""
	return [c.r, c.g, c.b, c.a]

static func array_to_color(a: Array) -> Color:
	"""Convert RGBA array to Color."""
	if a.size() < 4:
		return Color.WHITE
	return Color(a[0], a[1], a[2], a[3])

static func get_message_type_name(type: MessageType) -> String:
	"""Get string name for message type."""
	return MESSAGE_TYPE_NAMES.get(type, "UNKNOWN")

static func get_command_type_name(type: CommandType) -> String:
	"""Get string name for command type."""
	return COMMAND_TYPE_NAMES.get(type, "wait")

static func command_name_to_type(name: String) -> CommandType:
	"""Convert command string to CommandType enum."""
	for key in COMMAND_TYPE_NAMES:
		if COMMAND_TYPE_NAMES[key] == name:
			return key
	return CommandType.WAIT

# Validation functions

static func validate_entity_update(payload: Dictionary) -> bool:
	"""Validate entity update payload has required fields."""
	return (
		payload.has("godot_id") and
		payload.has("entity_type") and
		payload.has("name")
	)

static func validate_agent_command(payload: Dictionary) -> bool:
	"""Validate agent command payload has required fields."""
	return (
		payload.has("agent_godot_id") and
		payload.has("command_type")
	)

static func validate_perception(payload: Dictionary) -> bool:
	"""Validate perception payload has required fields."""
	return (
		payload.has("agent_godot_id") and
		payload.has("agent_name")
	)
