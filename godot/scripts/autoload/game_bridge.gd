extends Node
## GameBridge: WebSocket connection to Python ToM backend
## Handles all communication between Godot and the NAS inference server

signal connected
signal disconnected
signal connection_failed
signal message_received(msg_type: String, payload: Dictionary)

# Connection state
enum State { DISCONNECTED, CONNECTING, CONNECTED }
var state: State = State.DISCONNECTED

# WebSocket
var _socket: WebSocketPeer = WebSocketPeer.new()
var _url: String = "ws://localhost:9080"

# Message queue for when we're waiting on responses
var _pending_queries: Dictionary = {}  # query_id -> callback
var _query_counter: int = 0

# Reconnection
var _reconnect_timer: float = 0.0
var _reconnect_delay: float = 2.0
var _max_reconnect_attempts: int = 5
var _reconnect_attempts: int = 0


func _ready() -> void:
	set_process(true)
	# Auto-connect on start - comment out if you want manual control
	connect_to_backend()


func _process(delta: float) -> void:
	match state:
		State.CONNECTING, State.CONNECTED:
			_socket.poll()
			var socket_state = _socket.get_ready_state()
			
			match socket_state:
				WebSocketPeer.STATE_OPEN:
					if state == State.CONNECTING:
						state = State.CONNECTED
						_reconnect_attempts = 0
						print("[GameBridge] Connected to Python backend")
						connected.emit()
					
					# Process incoming messages
					while _socket.get_available_packet_count() > 0:
						var packet = _socket.get_packet()
						_handle_packet(packet)
				
				WebSocketPeer.STATE_CLOSING:
					pass  # Wait for close
				
				WebSocketPeer.STATE_CLOSED:
					var code = _socket.get_close_code()
					var reason = _socket.get_close_reason()
					print("[GameBridge] Connection closed: %d - %s" % [code, reason])
					state = State.DISCONNECTED
					disconnected.emit()
					_attempt_reconnect()
		
		State.DISCONNECTED:
			if _reconnect_timer > 0:
				_reconnect_timer -= delta
				if _reconnect_timer <= 0:
					connect_to_backend()


func connect_to_backend(url: String = "") -> void:
	if url != "":
		_url = url
	
	if state != State.DISCONNECTED:
		push_warning("[GameBridge] Already connecting or connected")
		return
	
	print("[GameBridge] Connecting to %s..." % _url)
	state = State.CONNECTING
	
	var err = _socket.connect_to_url(_url)
	if err != OK:
		push_error("[GameBridge] Failed to initiate connection: %d" % err)
		state = State.DISCONNECTED
		connection_failed.emit()
		_attempt_reconnect()


func disconnect_from_backend() -> void:
	if state == State.DISCONNECTED:
		return
	_socket.close()
	state = State.DISCONNECTED
	_reconnect_attempts = _max_reconnect_attempts  # Prevent auto-reconnect


func _attempt_reconnect() -> void:
	if _reconnect_attempts >= _max_reconnect_attempts:
		push_error("[GameBridge] Max reconnection attempts reached")
		return
	
	_reconnect_attempts += 1
	_reconnect_timer = _reconnect_delay * _reconnect_attempts
	print("[GameBridge] Reconnecting in %.1f seconds (attempt %d/%d)" % [
		_reconnect_timer, _reconnect_attempts, _max_reconnect_attempts
	])


func is_connected_to_backend() -> bool:
	return state == State.CONNECTED


# =============================================================================
# MESSAGE HANDLING
# =============================================================================

func _handle_packet(packet: PackedByteArray) -> void:
	var json_str = packet.get_string_from_utf8()
	var json = JSON.new()
	var err = json.parse(json_str)
	
	if err != OK:
		push_error("[GameBridge] Failed to parse message: %s" % json.get_error_message())
		return
	
	var data = json.get_data()
	if not data is Dictionary:
		push_error("[GameBridge] Message is not a dictionary")
		return
	
	var msg_type: String = data.get("type", "unknown")
	var payload: Dictionary = data.get("payload", {})
	var query_id: String = data.get("query_id", "")
	
	# Check if this is a response to a pending query
	if query_id != "" and _pending_queries.has(query_id):
		var callback = _pending_queries[query_id]
		_pending_queries.erase(query_id)
		if callback.is_valid():
			callback.call(payload)
		return
	
	# Otherwise emit as a general message
	message_received.emit(msg_type, payload)
	
	# Handle specific message types
	match msg_type:
		"UPDATE_SOUL_MAP":
			_handle_soul_map_update(payload)
		"SPAWN_NPC":
			_handle_spawn_npc(payload)
		"NARRATIVE_BEAT":
			_handle_narrative_beat(payload)
		"DIALOGUE_RESPONSE":
			_handle_dialogue_response(payload)


func _handle_soul_map_update(payload: Dictionary) -> void:
	var npc_id: String = payload.get("npc_id", "")
	var soul_map: Dictionary = payload.get("soul_map", {})
	if npc_id != "":
		SoulMapManager.update_soul_map(npc_id, soul_map)


func _handle_spawn_npc(payload: Dictionary) -> void:
	EventBus.npc_spawn_requested.emit(payload)


func _handle_narrative_beat(payload: Dictionary) -> void:
	EventBus.narrative_beat.emit(payload)


func _handle_dialogue_response(payload: Dictionary) -> void:
	EventBus.dialogue_received.emit(payload)


# =============================================================================
# OUTGOING MESSAGES
# =============================================================================

func _send_message(msg_type: String, payload: Dictionary, query_id: String = "") -> void:
	if state != State.CONNECTED:
		push_warning("[GameBridge] Cannot send message: not connected")
		return
	
	var message = {
		"type": msg_type,
		"payload": payload,
		"timestamp": Time.get_unix_time_from_system()
	}
	
	if query_id != "":
		message["query_id"] = query_id
	
	var json_str = JSON.stringify(message)
	_socket.send_text(json_str)


func _generate_query_id() -> String:
	_query_counter += 1
	return "q_%d_%d" % [Time.get_ticks_msec(), _query_counter]


# --- Player Actions ---

func report_player_action(action_type: String, target_id: String = "", context: Dictionary = {}) -> void:
	"""Report player behavior to Python for belief updates"""
	_send_message("PLAYER_ACTION", {
		"action_type": action_type,
		"target_id": target_id,
		"context": context,
		"player_position": _get_player_position()
	})


func _get_player_position() -> Dictionary:
	var player = get_tree().get_first_node_in_group("player")
	if player:
		return {"x": player.global_position.x, "y": player.global_position.y, "z": player.global_position.z}
	return {"x": 0, "y": 0, "z": 0}


# --- Tier 2: Strategic Queries ---

func query_strategic_decision(npc_id: String, situation: Dictionary, callback: Callable) -> void:
	"""Ask Python for a strategic decision (Tier 2)
	
	Example situation:
	{
		"type": "trust_decision",
		"target": "player",
		"context": {"recent_actions": [...], "relationship_history": [...]}
	}
	"""
	var query_id = _generate_query_id()
	_pending_queries[query_id] = callback
	
	_send_message("QUERY_STRATEGIC", {
		"npc_id": npc_id,
		"situation": situation,
		"soul_map": SoulMapManager.get_soul_map(npc_id)
	}, query_id)


# --- Tier 3: Deep ToM Queries ---

func query_deep_tom(npc_id: String, target_id: String, depth: int, query_type: String, callback: Callable) -> void:
	"""Ask Python for deep recursive ToM reasoning (Tier 3)
	
	query_type examples:
	- "belief_state": What does NPC think target believes?
	- "deception_detection": Is target trying to deceive?
	- "intention_inference": What is target's goal?
	"""
	var query_id = _generate_query_id()
	_pending_queries[query_id] = callback
	
	_send_message("QUERY_DEEP_TOM", {
		"npc_id": npc_id,
		"target_id": target_id,
		"depth": depth,
		"query_type": query_type,
		"soul_map": SoulMapManager.get_soul_map(npc_id),
		"local_beliefs": SoulMapManager.get_beliefs(npc_id)
	}, query_id)


# --- Dialogue ---

func request_dialogue(npc_id: String, context: Dictionary, callback: Callable) -> void:
	"""Request dialogue generation from Python"""
	var query_id = _generate_query_id()
	_pending_queries[query_id] = callback
	
	_send_message("DIALOGUE_REQUEST", {
		"npc_id": npc_id,
		"context": context,
		"soul_map": SoulMapManager.get_soul_map(npc_id),
		"conversation_history": context.get("history", [])
	}, query_id)


# --- Perception Events ---

func report_perception(npc_id: String, perceived_entities: Array) -> void:
	"""Report what an NPC perceives to Python for processing"""
	_send_message("PERCEPTION_EVENT", {
		"npc_id": npc_id,
		"entities": perceived_entities,
		"timestamp": Time.get_unix_time_from_system()
	})


# --- World State Sync ---

func sync_world_state(entities: Array, relationships: Dictionary) -> void:
	"""Send full world state snapshot to Python"""
	_send_message("WORLD_STATE", {
		"entities": entities,
		"relationships": relationships,
		"game_time": Time.get_ticks_msec() / 1000.0
	})
