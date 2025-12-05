extends CharacterBody3D
## Agent Controller for ToM-NAS
##
## This script controls an AI agent in the Godot simulation.
## It handles:
## - Movement and navigation
## - Perception (what the agent can see/hear)
## - Command execution from Python
## - Interaction with objects and other agents
##
## The agent sends perceptions to Python and receives commands back.

class_name AgentController

# Signals
signal perception_updated(perception: Dictionary)
signal command_completed(command_id: String, success: bool)
signal interaction_performed(target: Node3D, interaction_type: String)

# Configuration
@export var agent_name: String = "Agent"
@export var move_speed: float = 5.0
@export var turn_speed: float = 3.0
@export var perception_range: float = 20.0
@export var perception_fov: float = 120.0  # degrees
@export var hearing_range: float = 30.0
@export var interaction_range: float = 2.0
@export var perception_interval: float = 0.1  # seconds between perception updates

# State
var energy_level: float = 1.0
var held_object: Node3D = null
var current_institution: String = ""
var is_speaking: bool = false

# Current command
var current_command: Dictionary = {}
var command_target_position: Vector3 = Vector3.ZERO
var command_target_entity: Node3D = null

# Perception
var _perception_timer: float = 0.0
var last_perception: Dictionary = {}

# Movement
var _target_position: Vector3 = Vector3.ZERO
var _is_moving: bool = false

# Raycast for vision
@onready var vision_raycast: RayCast3D = $VisionRaycast if has_node("VisionRaycast") else null


func _ready() -> void:
	add_to_group("agents")

	# Create vision raycast if not exists
	if vision_raycast == null:
		vision_raycast = RayCast3D.new()
		vision_raycast.name = "VisionRaycast"
		vision_raycast.target_position = Vector3(0, 0, -perception_range)
		add_child(vision_raycast)

	# Register with world state
	WorldState.register_entity(self)

	# Connect to WebSocket for commands
	WebSocketBridge.command_received.connect(_on_command_received)


func _physics_process(delta: float) -> void:
	# Update perception
	_perception_timer += delta
	if _perception_timer >= perception_interval:
		_perception_timer = 0.0
		_update_and_send_perception()

	# Process current command
	_process_command(delta)

	# Apply gravity
	if not is_on_floor():
		velocity.y -= 9.8 * delta

	move_and_slide()


func _update_and_send_perception() -> Dictionary:
	var perception = _gather_perception()
	last_perception = perception
	perception_updated.emit(perception)

	# Send to Python
	WebSocketBridge.send_agent_perception(self, perception)

	return perception


func _gather_perception() -> Dictionary:
	var visible_entities: Array = []
	var occluded_entities: Array = []
	var heard_utterances: Array = []

	# Get all nearby entities
	var nearby = WorldState.get_entities_in_radius(global_position, perception_range)

	for entity in nearby:
		if entity == self:
			continue

		var to_entity = entity.global_position - global_position
		var distance = to_entity.length()

		# Check if in field of view
		var forward = -global_transform.basis.z
		var angle = rad_to_deg(forward.angle_to(to_entity.normalized()))

		if angle <= perception_fov / 2.0:
			# Check line of sight
			if _can_see(entity):
				visible_entities.append(WebSocketBridge._entity_to_dict(entity))
			else:
				occluded_entities.append(entity.get_instance_id())

	# Gather heard utterances (simplified - would use actual audio system)
	# For now, check for speaking agents within hearing range
	for agent in WorldState.get_all_agents():
		if agent != self and agent.has_meta("current_utterance"):
			var distance = global_position.distance_to(agent.global_position)
			if distance <= hearing_range:
				var volume = agent.get_meta("utterance_volume", 1.0)
				if distance <= hearing_range * volume:
					heard_utterances.append({
						"speaker_id": agent.get_instance_id(),
						"text": agent.get_meta("current_utterance"),
						"volume": volume,
					})

	return {
		"visible_entities": visible_entities,
		"occluded_entities": occluded_entities,
		"heard_utterances": heard_utterances,
		"velocity": velocity,
		"energy_level": energy_level,
		"held_object": held_object.get_instance_id() if held_object else null,
		"current_institution": current_institution,
	}


func _can_see(target: Node3D) -> bool:
	if vision_raycast == null:
		return true  # Assume visible if no raycast

	# Point raycast at target
	var direction = (target.global_position - global_position).normalized()
	vision_raycast.target_position = direction * perception_range
	vision_raycast.force_raycast_update()

	if vision_raycast.is_colliding():
		var collider = vision_raycast.get_collider()
		return collider == target or not (collider is Node3D)

	return true


func _on_command_received(agent_id: int, command: Dictionary) -> void:
	if agent_id != get_instance_id():
		return

	current_command = command
	_execute_command(command)


func _execute_command(command: Dictionary) -> void:
	var command_type = command.get("command_type", "")
	var command_id = command.get("command_id", "")

	match command_type:
		"move":
			_start_move_command(command)
		"interact":
			_execute_interact_command(command)
		"speak":
			_execute_speak_command(command)
		"look":
			_execute_look_command(command)
		"pick_up":
			_execute_pickup_command(command)
		"put_down":
			_execute_putdown_command(command)
		"stop":
			_stop_movement()
			_complete_command(command_id, true)
		_:
			print("Unknown command type: %s" % command_type)
			_complete_command(command_id, false, "Unknown command type")


func _start_move_command(command: Dictionary) -> void:
	if command.has("target_position"):
		var pos_dict = command.get("target_position")
		_target_position = Vector3(
			pos_dict.get("x", 0),
			pos_dict.get("y", 0),
			pos_dict.get("z", 0)
		)
		_is_moving = true
	elif command.has("target_entity_id"):
		var target_id = command.get("target_entity_id")
		command_target_entity = WorldState.find_entity_by_name(str(target_id))
		if command_target_entity:
			_target_position = command_target_entity.global_position
			_is_moving = true


func _process_command(delta: float) -> void:
	if not _is_moving:
		return

	# Update target if following entity
	if command_target_entity and is_instance_valid(command_target_entity):
		_target_position = command_target_entity.global_position

	# Calculate direction to target
	var direction = (_target_position - global_position)
	direction.y = 0  # Stay on ground plane
	var distance = direction.length()

	if distance < 0.5:
		# Reached target
		_stop_movement()
		_complete_command(current_command.get("command_id", ""), true)
		return

	# Move towards target
	direction = direction.normalized()
	velocity.x = direction.x * move_speed
	velocity.z = direction.z * move_speed

	# Rotate to face movement direction
	var target_rotation = atan2(direction.x, direction.z)
	rotation.y = lerp_angle(rotation.y, target_rotation, turn_speed * delta)


func _stop_movement() -> void:
	_is_moving = false
	velocity.x = 0
	velocity.z = 0
	command_target_entity = null


func _execute_interact_command(command: Dictionary) -> void:
	var target_id = command.get("target_entity_id")
	var target: Node3D = null

	# Find target
	for entity in WorldState.entities.values():
		if entity.get_instance_id() == target_id:
			target = entity
			break

	if target == null:
		_complete_command(command.get("command_id", ""), false, "Target not found")
		return

	# Check range
	if global_position.distance_to(target.global_position) > interaction_range:
		_complete_command(command.get("command_id", ""), false, "Target out of range")
		return

	# Perform interaction
	var interaction_type = command.get("interaction_type", "use")

	if target.has_method("interact"):
		var result = target.interact(self, interaction_type)
		WebSocketBridge.send_interaction_event(self, target, interaction_type, result.get("success", true), result)
		interaction_performed.emit(target, interaction_type)
		_complete_command(command.get("command_id", ""), result.get("success", true))
	else:
		WebSocketBridge.send_interaction_event(self, target, interaction_type, true, {})
		interaction_performed.emit(target, interaction_type)
		_complete_command(command.get("command_id", ""), true)


func _execute_speak_command(command: Dictionary) -> void:
	var text = command.get("utterance_text", "")
	var volume = command.get("volume", 1.0)

	if text.is_empty():
		_complete_command(command.get("command_id", ""), false, "No text provided")
		return

	# Set speaking state
	is_speaking = true
	set_meta("current_utterance", text)
	set_meta("utterance_volume", volume)

	# Find hearers
	var hearers: Array = []
	for agent in WorldState.get_all_agents():
		if agent != self:
			var distance = global_position.distance_to(agent.global_position)
			if distance <= hearing_range * volume:
				hearers.append(agent)

	# Send utterance event
	WebSocketBridge.send_utterance_event(self, text, volume, null, hearers)

	# Clear after a delay (speech duration)
	var duration = max(text.length() * 0.05, 1.0)  # Rough estimate
	await get_tree().create_timer(duration).timeout

	is_speaking = false
	remove_meta("current_utterance")
	remove_meta("utterance_volume")

	_complete_command(command.get("command_id", ""), true)


func _execute_look_command(command: Dictionary) -> void:
	var target_position: Vector3

	if command.has("target_position"):
		var pos_dict = command.get("target_position")
		target_position = Vector3(
			pos_dict.get("x", 0),
			pos_dict.get("y", 0),
			pos_dict.get("z", 0)
		)
	elif command.has("target_entity_id"):
		var target_id = command.get("target_entity_id")
		var target = instance_from_id(target_id)
		if target:
			target_position = target.global_position
		else:
			_complete_command(command.get("command_id", ""), false, "Target not found")
			return
	else:
		_complete_command(command.get("command_id", ""), false, "No target specified")
		return

	# Rotate to face target
	var direction = (target_position - global_position).normalized()
	var target_rotation = atan2(direction.x, direction.z)
	rotation.y = target_rotation

	_complete_command(command.get("command_id", ""), true)


func _execute_pickup_command(command: Dictionary) -> void:
	var target_id = command.get("target_entity_id")
	var target: Node3D = null

	for entity in WorldState.entities.values():
		if entity.get_instance_id() == target_id:
			target = entity
			break

	if target == null:
		_complete_command(command.get("command_id", ""), false, "Target not found")
		return

	if global_position.distance_to(target.global_position) > interaction_range:
		_complete_command(command.get("command_id", ""), false, "Target out of range")
		return

	if held_object != null:
		_complete_command(command.get("command_id", ""), false, "Already holding object")
		return

	# Pick up object
	held_object = target
	target.set_meta("held_by", get_instance_id())
	target.visible = false  # Or reparent to hand

	WebSocketBridge.send_interaction_event(self, target, "pick_up", true, {})
	_complete_command(command.get("command_id", ""), true)


func _execute_putdown_command(command: Dictionary) -> void:
	if held_object == null:
		_complete_command(command.get("command_id", ""), false, "Not holding anything")
		return

	# Put down object
	var target_pos = global_position + (-global_transform.basis.z * 1.0)
	target_pos.y = 0  # Ground level

	held_object.global_position = target_pos
	held_object.visible = true
	held_object.remove_meta("held_by")

	WebSocketBridge.send_interaction_event(self, held_object, "put_down", true, {})

	held_object = null
	_complete_command(command.get("command_id", ""), true)


func _complete_command(command_id: String, success: bool, error: String = "") -> void:
	if command_id.is_empty():
		return

	var payload = {
		"command_id": command_id,
		"success": success,
	}
	if not error.is_empty():
		payload["error"] = error

	WebSocketBridge.send_message(WebSocketBridge.MessageType.ACK, payload)
	command_completed.emit(command_id, success)
	current_command = {}


## Get agent state for serialization
func get_agent_state() -> Dictionary:
	return {
		"energy_level": energy_level,
		"held_object": held_object.get_instance_id() if held_object else null,
		"current_institution": current_institution,
		"is_speaking": is_speaking,
		"is_moving": _is_moving,
	}
