extends Node
## ToM-NAS Bridge - WebSocket Communication with Python Cognitive Controller
##
## This autoload manages the bidirectional WebSocket connection between
## the Godot physical simulation and the Python cognitive system.
##
## Communication Protocol:
## - All messages are JSON with type, payload, timestamp, sequence_id
## - Godot sends: entity updates, perceptions, world state, events
## - Python sends: agent commands, spawn requests, simulation control

class_name TomBridgeClass

# Connection settings
const DEFAULT_HOST: String = "localhost"
const DEFAULT_PORT: int = 9080
const HEARTBEAT_INTERVAL: float = 1.0
const RECONNECT_DELAY: float = 5.0
const MESSAGE_QUEUE_SIZE: int = 1000

# Connection state
enum ConnectionState {
	DISCONNECTED,
	CONNECTING,
	CONNECTED,
	ERROR
}

# Signals
signal connection_state_changed(state: ConnectionState)
signal message_received(message_type: String, payload: Dictionary)
signal command_received(agent_id: int, command: Dictionary)
signal simulation_control(control_type: String)
signal error_occurred(error_message: String)

# State
var connection_state: ConnectionState = ConnectionState.DISCONNECTED
var _socket: WebSocketPeer = null
var _host: String = DEFAULT_HOST
var _port: int = DEFAULT_PORT
var _sequence_id: int = 0
var _heartbeat_timer: float = 0.0
var _reconnect_timer: float = 0.0
var _should_reconnect: bool = true

# Message queues
var _outgoing_queue: Array[Dictionary] = []
var _pending_acks: Dictionary = {}  # command_id -> callback

# Statistics
var stats: Dictionary = {
	"messages_sent": 0,
	"messages_received": 0,
	"bytes_sent": 0,
	"bytes_received": 0,
	"errors": 0,
	"connection_time": 0.0,
	"last_heartbeat": 0.0
}

func _ready() -> void:
	# Initialize socket
	_socket = WebSocketPeer.new()

	# Connect to Python bridge on startup
	call_deferred("connect_to_bridge")

func _process(delta: float) -> void:
	if _socket == null:
		return

	_socket.poll()

	var state = _socket.get_ready_state()

	match state:
		WebSocketPeer.STATE_OPEN:
			if connection_state != ConnectionState.CONNECTED:
				_on_connected()
			_process_messages()
			_process_outgoing_queue()
			_handle_heartbeat(delta)

		WebSocketPeer.STATE_CONNECTING:
			if connection_state != ConnectionState.CONNECTING:
				connection_state = ConnectionState.CONNECTING
				connection_state_changed.emit(connection_state)

		WebSocketPeer.STATE_CLOSING:
			pass  # Wait for close

		WebSocketPeer.STATE_CLOSED:
			if connection_state != ConnectionState.DISCONNECTED:
				_on_disconnected()
			_handle_reconnect(delta)

func connect_to_bridge(host: String = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Error:
	"""Connect to the Python WebSocket server."""
	_host = host
	_port = port

	var url = "ws://%s:%d" % [host, port]
	print("[TomBridge] Connecting to %s..." % url)

	var error = _socket.connect_to_url(url)
	if error != OK:
		push_error("[TomBridge] Failed to connect: %s" % error_string(error))
		connection_state = ConnectionState.ERROR
		connection_state_changed.emit(connection_state)
		error_occurred.emit("Connection failed: %s" % error_string(error))
		return error

	connection_state = ConnectionState.CONNECTING
	connection_state_changed.emit(connection_state)
	return OK

func disconnect_from_bridge() -> void:
	"""Disconnect from the Python bridge."""
	_should_reconnect = false
	if _socket:
		_socket.close()
	connection_state = ConnectionState.DISCONNECTED
	connection_state_changed.emit(connection_state)

func _on_connected() -> void:
	"""Handle successful connection."""
	print("[TomBridge] Connected to Python bridge!")
	connection_state = ConnectionState.CONNECTED
	stats.connection_time = Time.get_unix_time_from_system()
	connection_state_changed.emit(connection_state)

	# Send initial world state
	call_deferred("_send_world_state")

func _on_disconnected() -> void:
	"""Handle disconnection."""
	print("[TomBridge] Disconnected from Python bridge")
	connection_state = ConnectionState.DISCONNECTED
	connection_state_changed.emit(connection_state)
	_reconnect_timer = 0.0

func _process_messages() -> void:
	"""Process incoming messages from Python."""
	while _socket.get_available_packet_count() > 0:
		var packet = _socket.get_packet()
		var json_str = packet.get_string_from_utf8()
		stats.bytes_received += len(json_str)
		stats.messages_received += 1

		var json = JSON.new()
		var error = json.parse(json_str)
		if error != OK:
			push_error("[TomBridge] JSON parse error: %s" % json.get_error_message())
			stats.errors += 1
			continue

		var message = json.get_data()
		_handle_message(message)

func _handle_message(message: Dictionary) -> void:
	"""Handle a received message based on type."""
	var msg_type = message.get("type", "")
	var payload = message.get("payload", {})
	var seq_id = message.get("sequence_id", 0)

	# Emit generic signal
	message_received.emit(msg_type, payload)

	match msg_type:
		"AGENT_COMMAND":
			_handle_agent_command(payload)

		"SPAWN_ENTITY":
			_handle_spawn_entity(payload)

		"MODIFY_ENTITY":
			_handle_modify_entity(payload)

		"WORLD_COMMAND":
			_handle_world_command(payload)

		"HEARTBEAT":
			stats.last_heartbeat = Time.get_unix_time_from_system()
			_send_heartbeat_response()

		"PAUSE":
			simulation_control.emit("pause")
			get_tree().paused = true

		"RESUME":
			simulation_control.emit("resume")
			get_tree().paused = false

		"RESET":
			simulation_control.emit("reset")
			_reset_world()

		"STEP":
			simulation_control.emit("step")
			_step_simulation()

		"ACK":
			_handle_ack(payload)

		"ERROR":
			push_error("[TomBridge] Error from Python: %s" % payload.get("error", "Unknown"))
			stats.errors += 1
			error_occurred.emit(payload.get("error", "Unknown error"))

func _handle_agent_command(payload: Dictionary) -> void:
	"""Handle a command for an agent."""
	var agent_id = payload.get("agent_godot_id", -1)
	if agent_id < 0:
		return

	# Find agent and dispatch command
	var agent = WorldManager.get_entity(agent_id)
	if agent and agent.has_method("execute_command"):
		agent.execute_command(payload)

	command_received.emit(agent_id, payload)

func _handle_spawn_entity(payload: Dictionary) -> void:
	"""Handle entity spawn request from Python."""
	var entity_type = payload.get("entity_type", "object")
	var name = payload.get("name", "entity")
	var position = _dict_to_vector3(payload.get("position", {}))
	var properties = payload.get("properties", {})

	WorldManager.spawn_entity(entity_type, name, position, properties)

func _handle_modify_entity(payload: Dictionary) -> void:
	"""Handle entity modification request."""
	var godot_id = payload.get("godot_id", -1)
	var modifications = payload.get("modifications", {})

	WorldManager.modify_entity(godot_id, modifications)

func _handle_world_command(payload: Dictionary) -> void:
	"""Handle global world commands."""
	var command = payload.get("command", "")

	match command:
		"get_world_state":
			_send_world_state()
		"set_time_of_day":
			WorldManager.set_time_of_day(payload.get("time", 12.0))
		"set_weather":
			WorldManager.set_weather(payload.get("weather", "clear"))

func _handle_ack(payload: Dictionary) -> void:
	"""Handle acknowledgment from Python."""
	var command_id = payload.get("command_id", "")
	if command_id in _pending_acks:
		var callback = _pending_acks[command_id]
		if callback:
			callback.call(payload)
		_pending_acks.erase(command_id)

func _process_outgoing_queue() -> void:
	"""Send queued messages."""
	while _outgoing_queue.size() > 0 and connection_state == ConnectionState.CONNECTED:
		var message = _outgoing_queue.pop_front()
		_send_message_internal(message)

func _send_message_internal(message: Dictionary) -> void:
	"""Actually send a message over the socket."""
	var json_str = JSON.stringify(message)
	var error = _socket.send_text(json_str)
	if error != OK:
		push_error("[TomBridge] Send error: %s" % error_string(error))
		stats.errors += 1
	else:
		stats.bytes_sent += len(json_str)
		stats.messages_sent += 1

func send_message(msg_type: String, payload: Dictionary, callback: Callable = Callable()) -> void:
	"""Queue a message to send to Python."""
	_sequence_id += 1

	var message = {
		"type": msg_type,
		"payload": payload,
		"timestamp": Time.get_unix_time_from_system(),
		"sequence_id": _sequence_id
	}

	# Store callback if provided
	if callback.is_valid() and payload.has("command_id"):
		_pending_acks[payload.command_id] = callback

	if connection_state == ConnectionState.CONNECTED:
		_send_message_internal(message)
	else:
		# Queue for later
		if _outgoing_queue.size() < MESSAGE_QUEUE_SIZE:
			_outgoing_queue.append(message)

# Convenience methods for specific message types

func send_entity_update(entity: Node3D) -> void:
	"""Send entity update to Python."""
	var payload = _create_entity_payload(entity)
	send_message("ENTITY_UPDATE", payload)

func send_agent_perception(agent: Node3D, perception_data: Dictionary) -> void:
	"""Send agent perception to Python."""
	var payload = {
		"agent_godot_id": agent.get_instance_id(),
		"agent_name": agent.name,
		"visible_entities": perception_data.get("visible", []),
		"occluded_entities": perception_data.get("occluded", []),
		"heard_utterances": perception_data.get("utterances", []),
		"own_position": _vector3_to_dict(agent.global_position),
		"own_velocity": _vector3_to_dict(agent.get("velocity", Vector3.ZERO)),
		"own_orientation": _vector3_to_dict(agent.global_rotation),
		"energy_level": agent.get("energy", 1.0),
		"held_object": agent.get("held_object_id", null),
		"current_institution": InstitutionManager.get_agent_institution(agent),
		"timestamp": WorldManager.simulation_time
	}
	send_message("AGENT_PERCEPTION", payload)

func send_collision_event(agent: Node3D, collider: Node3D, collision_info: Dictionary) -> void:
	"""Send collision event to Python."""
	var payload = {
		"agent_godot_id": agent.get_instance_id(),
		"collider_godot_id": collider.get_instance_id(),
		"collision_point": _vector3_to_dict(collision_info.get("point", Vector3.ZERO)),
		"collision_normal": _vector3_to_dict(collision_info.get("normal", Vector3.UP)),
		"timestamp": WorldManager.simulation_time
	}
	send_message("COLLISION_EVENT", payload)

func send_interaction_event(agent: Node3D, target: Node3D, interaction_type: String, success: bool, result_data: Dictionary = {}) -> void:
	"""Send interaction event to Python."""
	var payload = {
		"agent_godot_id": agent.get_instance_id(),
		"target_godot_id": target.get_instance_id(),
		"interaction_type": interaction_type,
		"success": success,
		"result_data": result_data,
		"timestamp": WorldManager.simulation_time
	}
	send_message("INTERACTION_EVENT", payload)

func send_utterance_event(speaker: Node3D, text: String, volume: float = 1.0, target: Node3D = null, hearers: Array = []) -> void:
	"""Send utterance event to Python."""
	var hearer_ids = []
	for h in hearers:
		hearer_ids.append(h.get_instance_id())

	var payload = {
		"speaker_godot_id": speaker.get_instance_id(),
		"text": text,
		"volume": volume,
		"target_agent_id": target.get_instance_id() if target else null,
		"hearers": hearer_ids,
		"timestamp": WorldManager.simulation_time
	}
	send_message("UTTERANCE_EVENT", payload)

func send_ack(command_id: String, success: bool, state_changes: Dictionary = {}, error: String = "") -> void:
	"""Send acknowledgment to Python."""
	var payload = {
		"command_id": command_id,
		"success": success,
		"state_changes": state_changes
	}
	if error:
		payload["error"] = error
	send_message("ACK", payload)

func _send_world_state() -> void:
	"""Send complete world state to Python."""
	var payload = WorldManager.get_world_state_payload()
	send_message("WORLD_STATE", payload)

func _send_heartbeat_response() -> void:
	"""Send heartbeat response."""
	send_message("HEARTBEAT", {"pong": true})

func _handle_heartbeat(delta: float) -> void:
	"""Send periodic heartbeats."""
	_heartbeat_timer += delta
	if _heartbeat_timer >= HEARTBEAT_INTERVAL:
		_heartbeat_timer = 0.0
		send_message("HEARTBEAT", {"ping": true})

func _handle_reconnect(delta: float) -> void:
	"""Handle reconnection attempts."""
	if not _should_reconnect:
		return

	_reconnect_timer += delta
	if _reconnect_timer >= RECONNECT_DELAY:
		_reconnect_timer = 0.0
		print("[TomBridge] Attempting reconnection...")
		connect_to_bridge(_host, _port)

func _reset_world() -> void:
	"""Reset the world state."""
	WorldManager.reset_world()

func _step_simulation() -> void:
	"""Step simulation by one frame."""
	# Unpause for one frame then pause again
	get_tree().paused = false
	await get_tree().process_frame
	get_tree().paused = true

# Helper functions

func _create_entity_payload(entity: Node3D) -> Dictionary:
	"""Create payload dictionary for an entity."""
	var payload = {
		"godot_id": entity.get_instance_id(),
		"entity_type": entity.get("entity_type", "object"),
		"name": entity.name,
		"position": _vector3_to_dict(entity.global_position),
		"rotation": _vector3_to_dict(entity.global_rotation),
		"scale": _vector3_to_dict(entity.scale),
		"velocity": _vector3_to_dict(entity.get("velocity", Vector3.ZERO)),
		"is_static": entity.get("is_static", true),
		"visible": entity.visible,
		"semantic_tags": entity.get("semantic_tags", []),
		"affordances": entity.get("affordances", []),
		"is_interactable": entity.get("is_interactable", false),
		"timestamp": WorldManager.simulation_time
	}

	if entity.get("is_being_held", false):
		payload["is_being_held"] = true
		payload["held_by"] = entity.get("held_by_id", null)

	return payload

func _vector3_to_dict(v: Vector3) -> Dictionary:
	"""Convert Vector3 to dictionary."""
	return {"x": v.x, "y": v.y, "z": v.z}

func _dict_to_vector3(d: Dictionary) -> Vector3:
	"""Convert dictionary to Vector3."""
	return Vector3(d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))

func is_connected_to_bridge() -> bool:
	"""Check if connected to Python bridge."""
	return connection_state == ConnectionState.CONNECTED

func get_statistics() -> Dictionary:
	"""Get bridge statistics."""
	return stats.duplicate()
