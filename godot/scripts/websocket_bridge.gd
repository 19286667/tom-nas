extends Node
## WebSocket Bridge for ToM-NAS Python Integration
##
## This autoload singleton manages the WebSocket connection to the Python
## cognitive controller (enhanced_server.py). It handles:
## - Connection lifecycle
## - Message serialization/deserialization
## - Perception sending
## - Command receiving
##
## The bridge implements the protocol defined in godot_bridge/protocol.py

class_name WebSocketBridgeClass

# Signals
signal connected()
signal disconnected()
signal connection_error(message: String)
signal message_received(msg_type: String, payload: Dictionary)
signal perception_requested(agent_id: int)
signal command_received(agent_id: int, command: Dictionary)

# Configuration
@export var server_host: String = "localhost"
@export var server_port: int = 9080
@export var reconnect_delay: float = 5.0
@export var heartbeat_interval: float = 1.0
@export var log_messages: bool = false

# Connection state
enum ConnectionState { DISCONNECTED, CONNECTING, CONNECTED, ERROR }
var state: ConnectionState = ConnectionState.DISCONNECTED
var _socket: WebSocketPeer = null
var _reconnect_timer: float = 0.0
var _heartbeat_timer: float = 0.0
var _sequence_id: int = 0

# Statistics
var messages_sent: int = 0
var messages_received: int = 0
var bytes_sent: int = 0
var bytes_received: int = 0
var last_heartbeat_time: float = 0.0

# Message types (matching protocol.py)
const MessageType = {
	# Godot -> Python
	"ENTITY_UPDATE": "ENTITY_UPDATE",
	"AGENT_PERCEPTION": "AGENT_PERCEPTION",
	"WORLD_STATE": "WORLD_STATE",
	"COLLISION_EVENT": "COLLISION_EVENT",
	"INTERACTION_EVENT": "INTERACTION_EVENT",
	"UTTERANCE_EVENT": "UTTERANCE_EVENT",
	# Python -> Godot
	"AGENT_COMMAND": "AGENT_COMMAND",
	"SPAWN_ENTITY": "SPAWN_ENTITY",
	"MODIFY_ENTITY": "MODIFY_ENTITY",
	"WORLD_COMMAND": "WORLD_COMMAND",
	# Bidirectional
	"HEARTBEAT": "HEARTBEAT",
	"ACK": "ACK",
	"ERROR": "ERROR",
	# Control
	"PAUSE": "PAUSE",
	"RESUME": "RESUME",
	"RESET": "RESET",
	"STEP": "STEP",
}


func _ready() -> void:
	_log("WebSocketBridge initializing...")
	connect_to_server()


func _process(delta: float) -> void:
	match state:
		ConnectionState.CONNECTING, ConnectionState.CONNECTED:
			_poll_socket()
		ConnectionState.DISCONNECTED, ConnectionState.ERROR:
			_handle_reconnect(delta)

	# Heartbeat
	if state == ConnectionState.CONNECTED:
		_heartbeat_timer += delta
		if _heartbeat_timer >= heartbeat_interval:
			_send_heartbeat()
			_heartbeat_timer = 0.0


func connect_to_server() -> void:
	if _socket != null:
		_socket.close()

	_socket = WebSocketPeer.new()
	var url = "ws://%s:%d" % [server_host, server_port]

	_log("Connecting to Python server at %s" % url)
	state = ConnectionState.CONNECTING

	var err = _socket.connect_to_url(url)
	if err != OK:
		_log("Failed to initiate connection: %s" % error_string(err))
		state = ConnectionState.ERROR
		connection_error.emit("Failed to connect: %s" % error_string(err))


func disconnect_from_server() -> void:
	if _socket != null:
		_socket.close()
		_socket = null
	state = ConnectionState.DISCONNECTED
	disconnected.emit()
	_log("Disconnected from server")


func is_connected_to_server() -> bool:
	return state == ConnectionState.CONNECTED


func _poll_socket() -> void:
	if _socket == null:
		return

	_socket.poll()
	var socket_state = _socket.get_ready_state()

	match socket_state:
		WebSocketPeer.STATE_OPEN:
			if state != ConnectionState.CONNECTED:
				state = ConnectionState.CONNECTED
				_log("Connected to Python server!")
				connected.emit()

			# Process incoming messages
			while _socket.get_available_packet_count() > 0:
				var packet = _socket.get_packet()
				bytes_received += packet.size()
				_handle_message(packet.get_string_from_utf8())

		WebSocketPeer.STATE_CONNECTING:
			pass  # Still connecting

		WebSocketPeer.STATE_CLOSING:
			pass  # Wait for close

		WebSocketPeer.STATE_CLOSED:
			var code = _socket.get_close_code()
			var reason = _socket.get_close_reason()
			_log("Connection closed: %d - %s" % [code, reason])
			state = ConnectionState.DISCONNECTED
			_socket = null
			disconnected.emit()


func _handle_reconnect(delta: float) -> void:
	_reconnect_timer += delta
	if _reconnect_timer >= reconnect_delay:
		_reconnect_timer = 0.0
		connect_to_server()


func _handle_message(json_str: String) -> void:
	messages_received += 1

	var json = JSON.new()
	var error = json.parse(json_str)
	if error != OK:
		_log("Failed to parse message: %s" % json.get_error_message())
		return

	var data = json.get_data()
	if not data is Dictionary:
		_log("Invalid message format")
		return

	var msg_type = data.get("type", "")
	var payload = data.get("payload", {})

	if log_messages:
		_log("Received: %s" % msg_type)

	# Handle specific message types
	match msg_type:
		MessageType.AGENT_COMMAND:
			var agent_id = payload.get("agent_godot_id", 0)
			command_received.emit(agent_id, payload)

		MessageType.HEARTBEAT:
			last_heartbeat_time = Time.get_ticks_msec() / 1000.0
			_send_ack(data.get("sequence_id", 0))

		MessageType.PAUSE:
			get_tree().paused = true

		MessageType.RESUME:
			get_tree().paused = false

		MessageType.RESET:
			WorldState.reset_world()

		MessageType.STEP:
			# Single step mode - pause after next physics tick
			get_tree().paused = false
			await get_tree().physics_frame
			get_tree().paused = true

		MessageType.SPAWN_ENTITY:
			WorldState.spawn_entity(payload)

		MessageType.MODIFY_ENTITY:
			WorldState.modify_entity(payload)

		MessageType.WORLD_COMMAND:
			_handle_world_command(payload)

	# Emit generic signal
	message_received.emit(msg_type, payload)


func _handle_world_command(payload: Dictionary) -> void:
	var command = payload.get("command", "")
	match command:
		"get_world_state":
			send_world_state()
		"set_time_of_day":
			WorldState.set_time_of_day(payload.get("time", 12.0))
		"set_weather":
			WorldState.set_weather(payload.get("weather", "clear"))


# --- Sending Messages ---

func send_message(msg_type: String, payload: Dictionary) -> void:
	if state != ConnectionState.CONNECTED or _socket == null:
		return

	_sequence_id += 1
	var message = {
		"type": msg_type,
		"payload": payload,
		"timestamp": Time.get_unix_time_from_system(),
		"sequence_id": _sequence_id,
	}

	var json_str = JSON.stringify(message)
	var packet = json_str.to_utf8_buffer()

	_socket.send(packet)
	messages_sent += 1
	bytes_sent += packet.size()

	if log_messages:
		_log("Sent: %s" % msg_type)


func send_entity_update(entity: Node3D) -> void:
	var payload = _entity_to_dict(entity)
	send_message(MessageType.ENTITY_UPDATE, payload)


func send_agent_perception(agent: Node3D, perception: Dictionary) -> void:
	var payload = {
		"agent_godot_id": agent.get_instance_id(),
		"agent_name": agent.name,
		"visible_entities": perception.get("visible_entities", []),
		"occluded_entities": perception.get("occluded_entities", []),
		"heard_utterances": perception.get("heard_utterances", []),
		"own_position": _vector3_to_dict(agent.global_position),
		"own_velocity": _vector3_to_dict(perception.get("velocity", Vector3.ZERO)),
		"own_orientation": _vector3_to_dict(agent.global_rotation),
		"energy_level": perception.get("energy_level", 1.0),
		"held_object": perception.get("held_object"),
		"current_institution": perception.get("current_institution"),
		"timestamp": Time.get_ticks_msec() / 1000.0,
	}
	send_message(MessageType.AGENT_PERCEPTION, payload)


func send_world_state() -> void:
	var payload = WorldState.get_world_state_dict()
	send_message(MessageType.WORLD_STATE, payload)


func send_collision_event(body_a: Node3D, body_b: Node3D, impact_velocity: float) -> void:
	var payload = {
		"body_a_id": body_a.get_instance_id(),
		"body_b_id": body_b.get_instance_id(),
		"body_a_name": body_a.name,
		"body_b_name": body_b.name,
		"impact_velocity": impact_velocity,
		"timestamp": Time.get_ticks_msec() / 1000.0,
	}
	send_message(MessageType.COLLISION_EVENT, payload)


func send_interaction_event(
	agent: Node3D,
	target: Node3D,
	interaction_type: String,
	success: bool,
	result_data: Dictionary = {}
) -> void:
	var payload = {
		"agent_godot_id": agent.get_instance_id(),
		"target_godot_id": target.get_instance_id(),
		"interaction_type": interaction_type,
		"success": success,
		"result_data": result_data,
		"timestamp": Time.get_ticks_msec() / 1000.0,
	}
	send_message(MessageType.INTERACTION_EVENT, payload)


func send_utterance_event(
	speaker: Node3D,
	text: String,
	volume: float = 1.0,
	target_agent: Node3D = null,
	hearers: Array = []
) -> void:
	var hearer_ids = []
	for hearer in hearers:
		if hearer is Node3D:
			hearer_ids.append(hearer.get_instance_id())

	var payload = {
		"speaker_godot_id": speaker.get_instance_id(),
		"text": text,
		"volume": volume,
		"target_agent_id": target_agent.get_instance_id() if target_agent else null,
		"hearers": hearer_ids,
		"timestamp": Time.get_ticks_msec() / 1000.0,
	}
	send_message(MessageType.UTTERANCE_EVENT, payload)


func _send_heartbeat() -> void:
	send_message(MessageType.HEARTBEAT, {"ping": true})


func _send_ack(sequence_id: int) -> void:
	send_message(MessageType.ACK, {"sequence_id": sequence_id})


# --- Utility Functions ---

func _entity_to_dict(entity: Node3D) -> Dictionary:
	var semantic_tags: Array[String] = []
	var affordances: Array[String] = []

	# Get tags from metadata or groups
	if entity.has_meta("semantic_tags"):
		semantic_tags = entity.get_meta("semantic_tags")
	for group in entity.get_groups():
		if group.begins_with("tag_"):
			semantic_tags.append(group.substr(4))

	if entity.has_meta("affordances"):
		affordances = entity.get_meta("affordances")

	var entity_type = "object"
	if entity.is_in_group("agents"):
		entity_type = "agent"
	elif entity.is_in_group("locations"):
		entity_type = "location"

	var velocity = Vector3.ZERO
	if entity is RigidBody3D:
		velocity = entity.linear_velocity
	elif entity is CharacterBody3D:
		velocity = entity.velocity

	return {
		"godot_id": entity.get_instance_id(),
		"entity_type": entity_type,
		"name": entity.name,
		"position": _vector3_to_dict(entity.global_position),
		"rotation": _vector3_to_dict(entity.global_rotation),
		"scale": _vector3_to_dict(entity.scale),
		"velocity": _vector3_to_dict(velocity),
		"is_static": entity is StaticBody3D,
		"visible": entity.visible,
		"semantic_tags": semantic_tags,
		"affordances": affordances,
		"is_interactable": entity.is_in_group("interactable"),
		"is_being_held": entity.has_meta("held_by"),
		"held_by": entity.get_meta("held_by") if entity.has_meta("held_by") else null,
		"timestamp": Time.get_ticks_msec() / 1000.0,
	}


func _vector3_to_dict(v: Vector3) -> Dictionary:
	return {"x": v.x, "y": v.y, "z": v.z}


func _log(message: String) -> void:
	print("[WebSocketBridge] %s" % message)


# --- Statistics ---

func get_statistics() -> Dictionary:
	return {
		"state": ConnectionState.keys()[state],
		"messages_sent": messages_sent,
		"messages_received": messages_received,
		"bytes_sent": bytes_sent,
		"bytes_received": bytes_received,
		"last_heartbeat": last_heartbeat_time,
	}
