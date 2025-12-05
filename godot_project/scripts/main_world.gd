extends Node3D
## Main World - Top-level scene controller
##
## Manages the main simulation world, including setup,
## debug UI updates, and scenario loading.

class_name MainWorld

# Node references
@onready var agents_container: Node3D = $Agents
@onready var objects_container: Node3D = $Objects
@onready var locations_container: Node3D = $Locations
@onready var camera: Camera3D = $Camera3D

# Debug UI
@onready var connection_label: Label = $UI/DebugPanel/VBoxContainer/ConnectionStatus
@onready var entity_label: Label = $UI/DebugPanel/VBoxContainer/EntityCount
@onready var sim_time_label: Label = $UI/DebugPanel/VBoxContainer/SimTime
@onready var time_of_day_label: Label = $UI/DebugPanel/VBoxContainer/TimeOfDay
@onready var fps_label: Label = $UI/DebugPanel/VBoxContainer/FPS

# Scene references
const AgentScene = preload("res://scenes/entities/agent.tscn")
const ObjectScene = preload("res://scenes/entities/interactable_object.tscn")
const ItemScene = preload("res://scenes/entities/item.tscn")

# Configuration
@export var auto_spawn_agents: bool = true
@export var initial_agent_count: int = 4
@export var spawn_test_objects: bool = true

# Camera control
var camera_zoom: float = 30.0
var camera_angle: float = 45.0
var camera_rotation_y: float = 0.0

func _ready() -> void:
	# Connect to bridge signals
	if TomBridge:
		TomBridge.connection_state_changed.connect(_on_connection_state_changed)
		TomBridge.simulation_control.connect(_on_simulation_control)

	# Spawn initial entities
	if auto_spawn_agents:
		call_deferred("_spawn_initial_agents")

	if spawn_test_objects:
		call_deferred("_spawn_test_objects")

	EventBus.game_started.emit()

func _process(delta: float) -> void:
	_update_debug_ui()
	_handle_camera_input(delta)

func _input(event: InputEvent) -> void:
	# Camera zoom with scroll wheel
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_zoom = max(10.0, camera_zoom - 2.0)
			_update_camera()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_zoom = min(100.0, camera_zoom + 2.0)
			_update_camera()

	# Pause toggle
	if event.is_action_pressed("pause_simulation"):
		get_tree().paused = not get_tree().paused
		if get_tree().paused:
			EventBus.game_paused.emit()
		else:
			EventBus.game_resumed.emit()

func _handle_camera_input(delta: float) -> void:
	var move = Vector3.ZERO

	if Input.is_action_pressed("move_forward"):
		move.z -= 1
	if Input.is_action_pressed("move_back"):
		move.z += 1
	if Input.is_action_pressed("move_left"):
		move.x -= 1
	if Input.is_action_pressed("move_right"):
		move.x += 1

	if move.length() > 0:
		move = move.normalized() * 20.0 * delta
		move = move.rotated(Vector3.UP, camera_rotation_y)
		camera.global_position += move

func _update_camera() -> void:
	var target_pos = camera.global_position
	target_pos.y = camera_zoom * sin(deg_to_rad(camera_angle))
	var horizontal_dist = camera_zoom * cos(deg_to_rad(camera_angle))

	camera.global_position = target_pos
	camera.look_at(target_pos - Vector3(0, camera.global_position.y, horizontal_dist), Vector3.UP)

func _update_debug_ui() -> void:
	# Connection status
	if TomBridge:
		var state = "Connected" if TomBridge.is_connected_to_bridge() else "Disconnected"
		connection_label.text = "Connection: %s" % state

	# Entity count
	if WorldManager:
		var info = WorldManager.get_debug_info()
		entity_label.text = "Entities: %d (Agents: %d)" % [info.total_entities, info.agents]
		sim_time_label.text = "Sim Time: %.1f" % info.simulation_time

		var hour = int(info.time_of_day)
		var minute = int((info.time_of_day - hour) * 60)
		time_of_day_label.text = "Hour: %02d:%02d" % [hour, minute]

	# FPS
	fps_label.text = "FPS: %d" % Engine.get_frames_per_second()

func _on_connection_state_changed(state) -> void:
	print("[MainWorld] Connection state changed: %s" % state)

func _on_simulation_control(control: String) -> void:
	match control:
		"reset":
			_reset_world()

# Spawning

func _spawn_initial_agents() -> void:
	"""Spawn initial test agents."""
	var spawn_positions = [
		Vector3(0, 0, 0),
		Vector3(5, 0, 5),
		Vector3(-5, 0, 5),
		Vector3(0, 0, -5)
	]

	var agent_names = ["Alice", "Bob", "Carol", "Dave"]

	for i in range(min(initial_agent_count, spawn_positions.size())):
		spawn_agent(agent_names[i], spawn_positions[i])

func _spawn_test_objects() -> void:
	"""Spawn test objects for interaction testing."""
	# Spawn some items in the market
	var market_items = [
		{"name": "Apple", "position": Vector3(12, 0.5, 0), "type": "food", "value": 5.0},
		{"name": "Gold Coin", "position": Vector3(15, 0.5, 3), "type": "valuable", "value": 100.0},
		{"name": "Hammer", "position": Vector3(18, 0.5, -2), "type": "tool", "value": 25.0}
	]

	for item_data in market_items:
		spawn_item(item_data.name, item_data.position, item_data.type, item_data.value)

	# Spawn objects in ministry
	spawn_object("Document", Vector3(-15, 0.5, 0), ["document", "paper", "official"])

	# Spawn objects in temple
	spawn_object("Candle", Vector3(0, 0.5, -20), ["sacred", "light", "ritual"])

func spawn_agent(agent_name: String, position: Vector3) -> TomAgent:
	"""Spawn an agent at a position."""
	var agent = AgentScene.instantiate()
	agent.agent_name = agent_name
	agent.global_position = position
	agents_container.add_child(agent)
	print("[MainWorld] Spawned agent: %s at %s" % [agent_name, position])
	return agent

func spawn_object(object_name: String, position: Vector3, tags: Array = []) -> InteractableObject:
	"""Spawn an interactable object."""
	var obj = ObjectScene.instantiate()
	obj.object_name = object_name
	obj.global_position = position
	for tag in tags:
		if tag not in obj.semantic_tags:
			obj.semantic_tags.append(tag)
	objects_container.add_child(obj)
	return obj

func spawn_item(item_name: String, position: Vector3, item_type: String = "generic", value: float = 1.0) -> TomItem:
	"""Spawn an item."""
	var item = ItemScene.instantiate()
	item.object_name = item_name
	item.item_type = item_type
	item.base_value = value
	item.global_position = position
	objects_container.add_child(item)
	return item

func _reset_world() -> void:
	"""Reset the world to initial state."""
	# Clear agents
	for child in agents_container.get_children():
		child.queue_free()

	# Clear objects
	for child in objects_container.get_children():
		child.queue_free()

	# Respawn
	await get_tree().process_frame
	_spawn_initial_agents()
	_spawn_test_objects()

# Scenario loading

func load_scenario(scenario_name: String) -> bool:
	"""Load a predefined scenario."""
	match scenario_name:
		"sally_anne":
			return _setup_sally_anne_scenario()
		"market_exchange":
			return _setup_market_exchange_scenario()
		"zombie_detection":
			return _setup_zombie_detection_scenario()
		_:
			push_error("Unknown scenario: %s" % scenario_name)
			return false

func _setup_sally_anne_scenario() -> bool:
	"""Set up the Sally-Anne false belief test scenario."""
	_reset_world()
	await get_tree().process_frame

	# Spawn Sally and Anne
	var sally = spawn_agent("Sally", Vector3(0, 0, 0))
	var anne = spawn_agent("Anne", Vector3(3, 0, 0))

	# Spawn the basket and box
	var basket = spawn_object("Basket", Vector3(-2, 0.5, 0), ["container", "basket"])
	var box = spawn_object("Box", Vector3(2, 0.5, 0), ["container", "box"])

	# Spawn the marble
	var marble = spawn_item("Marble", Vector3(-2, 1, 0), "toy", 1.0)

	EventBus.scenario_loaded.emit("sally_anne")
	return true

func _setup_market_exchange_scenario() -> bool:
	"""Set up a market exchange scenario."""
	_reset_world()
	await get_tree().process_frame

	# Spawn buyer and seller agents
	var buyer = spawn_agent("Buyer", Vector3(10, 0, 0))
	var seller = spawn_agent("Seller", Vector3(18, 0, 0))

	# Spawn trade goods
	spawn_item("Apple", Vector3(15, 0.5, 0), "food", 5.0)
	spawn_item("Gold", Vector3(12, 0.5, 0), "currency", 50.0)

	EventBus.scenario_loaded.emit("market_exchange")
	return true

func _setup_zombie_detection_scenario() -> bool:
	"""Set up a zombie detection scenario."""
	_reset_world()
	await get_tree().process_frame

	# Spawn detector agent
	var detector = spawn_agent("Detector", Vector3(0, 0, 0))

	# Spawn mix of normal and "zombie" agents
	spawn_agent("Normal1", Vector3(5, 0, 0))
	spawn_agent("Normal2", Vector3(-5, 0, 0))

	# Zombie agents would have modified behavior
	var zombie = spawn_agent("Suspect", Vector3(0, 0, 5))
	# Mark as potential zombie for testing
	zombie.set_meta("is_zombie", true)

	EventBus.scenario_loaded.emit("zombie_detection")
	return true
