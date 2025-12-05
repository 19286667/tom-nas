extends Node3D
## Main Scene Controller for ToM-NAS Validation
##
## This script manages the main scene, handling:
## - Connection status display
## - Statistics updates
## - Debug controls

@onready var connection_label: Label = $UI/ConnectionStatus
@onready var stats_label: Label = $UI/StatsLabel


func _ready() -> void:
	# Connect to WebSocket signals
	WebSocketBridge.connected.connect(_on_connected)
	WebSocketBridge.disconnected.connect(_on_disconnected)
	WebSocketBridge.connection_error.connect(_on_connection_error)

	# Register all agents
	for agent in $Agents.get_children():
		WorldState.register_entity(agent)

	# Register all objects
	for obj in $Objects.get_children():
		WorldState.register_entity(obj)

	_update_connection_label()


func _process(_delta: float) -> void:
	_update_stats_label()


func _on_connected() -> void:
	_update_connection_label()
	print("Connected to Python cognitive controller!")

	# Send initial world state
	WebSocketBridge.send_world_state()


func _on_disconnected() -> void:
	_update_connection_label()
	print("Disconnected from Python server")


func _on_connection_error(message: String) -> void:
	_update_connection_label()
	print("Connection error: %s" % message)


func _update_connection_label() -> void:
	if connection_label == null:
		return

	match WebSocketBridge.state:
		WebSocketBridge.ConnectionState.CONNECTED:
			connection_label.text = "Python Server: Connected"
			connection_label.modulate = Color.GREEN
		WebSocketBridge.ConnectionState.CONNECTING:
			connection_label.text = "Python Server: Connecting..."
			connection_label.modulate = Color.YELLOW
		WebSocketBridge.ConnectionState.DISCONNECTED:
			connection_label.text = "Python Server: Disconnected"
			connection_label.modulate = Color.RED
		WebSocketBridge.ConnectionState.ERROR:
			connection_label.text = "Python Server: Error"
			connection_label.modulate = Color.RED


func _update_stats_label() -> void:
	if stats_label == null:
		return

	var stats = WebSocketBridge.get_statistics()
	var agent_count = WorldState.agents.size()

	stats_label.text = """Agents: %d
Messages: %d sent / %d recv
Time: %.1fs
Timestep: %d""" % [
		agent_count,
		stats.get("messages_sent", 0),
		stats.get("messages_received", 0),
		WorldState.simulation_time,
		WorldState.timestep,
	]


func _input(event: InputEvent) -> void:
	# Debug controls
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_R:  # Reset
				WorldState.reset_world()
			KEY_P:  # Pause/Resume
				get_tree().paused = not get_tree().paused
			KEY_W:  # Send world state
				WebSocketBridge.send_world_state()
