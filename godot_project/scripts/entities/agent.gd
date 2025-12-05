extends CharacterBody3D
## Agent - Cognitive Agent in ToM-NAS Simulation
##
## Represents an agent that can move, interact, perceive, and communicate.
## Controlled by the Python cognitive system via the WebSocket bridge.

class_name TomAgent

# Configuration
@export var agent_name: String = "Agent"
@export var move_speed: float = 5.0
@export var run_speed: float = 10.0
@export var rotation_speed: float = 5.0
@export var interaction_range: float = 2.0

# Perception configuration
@export var view_distance: float = 20.0
@export var view_angle: float = 120.0
@export var hearing_distance: float = 15.0

# Entity metadata
var entity_type: String = "agent"
var semantic_tags: Array = ["agent", "animate", "social"]
var affordances: Array = ["can_speak", "can_move", "can_interact"]
var is_interactable: bool = true
var is_static: bool = false

# State
var energy: float = 1.0
var held_object: Node3D = null
var held_object_id: int = -1
var current_action: String = "idle"
var is_speaking: bool = false

# Movement state
var target_position: Vector3 = Vector3.ZERO
var has_target: bool = false
var is_moving: bool = false
var movement_speed: float = 5.0

# Command execution
var _current_command: Dictionary = {}
var _command_queue: Array = []
var _command_timeout: float = 0.0

# Signals
signal action_started(action: String, target: Node3D)
signal action_completed(action: String, success: bool)
signal energy_changed(old_energy: float, new_energy: float)
signal spoke(text: String)
signal picked_up_object(object: Node3D)
signal dropped_object(object: Node3D)

# Node references
@onready var collision_shape: CollisionShape3D = $CollisionShape3D
@onready var interaction_area: Area3D = $InteractionArea
@onready var speech_label: Label3D = $SpeechLabel if has_node("SpeechLabel") else null
@onready var navigation_agent: NavigationAgent3D = $NavigationAgent3D if has_node("NavigationAgent3D") else null

func _ready() -> void:
	# Set name from export
	name = agent_name

	# Register with world manager
	call_deferred("_register_with_world")

	# Connect interaction area signals
	if interaction_area:
		interaction_area.body_entered.connect(_on_body_entered_interaction)
		interaction_area.body_exited.connect(_on_body_exited_interaction)

func _register_with_world() -> void:
	if WorldManager:
		WorldManager.register_entity(self)

func _exit_tree() -> void:
	if WorldManager:
		WorldManager.unregister_entity(self)

func _physics_process(delta: float) -> void:
	# Handle movement
	if has_target and is_moving:
		_process_movement(delta)

	# Handle command timeout
	if _current_command.size() > 0:
		_command_timeout -= delta
		if _command_timeout <= 0:
			_complete_command(false, "timeout")

	# Apply gravity
	if not is_on_floor():
		velocity.y -= 9.8 * delta

	move_and_slide()

	# Energy decay
	if is_moving:
		_modify_energy(-0.001 * delta)

func _process_movement(delta: float) -> void:
	"""Process movement towards target."""
	var direction = (target_position - global_position)
	direction.y = 0  # Keep on ground

	var distance = direction.length()

	if distance < 0.5:
		# Reached target
		has_target = false
		is_moving = false
		velocity = Vector3.ZERO
		_complete_command(true)
		return

	# Move towards target
	direction = direction.normalized()
	velocity.x = direction.x * movement_speed
	velocity.z = direction.z * movement_speed

	# Rotate to face direction
	var target_rotation = atan2(direction.x, direction.z)
	rotation.y = lerp_angle(rotation.y, target_rotation, rotation_speed * delta)

# Command Execution

func execute_command(command: Dictionary) -> void:
	"""Execute a command from the Python cognitive system."""
	var command_type = command.get("command_type", "wait")
	var command_id = command.get("command_id", "")
	var timeout = command.get("timeout_seconds", 10.0)

	# Store current command
	_current_command = command
	_command_timeout = timeout

	print("[Agent %s] Executing command: %s" % [name, command_type])

	match command_type:
		"move":
			_execute_move(command)
		"turn":
			_execute_turn(command)
		"follow":
			_execute_follow(command)
		"flee":
			_execute_flee(command)
		"pick_up":
			_execute_pick_up(command)
		"put_down":
			_execute_put_down(command)
		"use":
			_execute_use(command)
		"examine":
			_execute_examine(command)
		"give":
			_execute_give(command)
		"speak":
			_execute_speak(command)
		"gesture":
			_execute_gesture(command)
		"look_at":
			_execute_look_at(command)
		"wait":
			_execute_wait(command)
		"cancel":
			_cancel_current_action()
		_:
			_complete_command(false, "unknown_command")

func _execute_move(command: Dictionary) -> void:
	"""Move to a position."""
	var target = _get_target_position(command)
	if target == Vector3.ZERO:
		_complete_command(false, "no_target")
		return

	target_position = target
	has_target = true
	is_moving = true
	movement_speed = command.get("speed", 1.0) * move_speed
	current_action = "moving"
	action_started.emit("move", null)

func _execute_turn(command: Dictionary) -> void:
	"""Turn to face a direction or entity."""
	var target_entity_id = command.get("target_entity_id")
	if target_entity_id:
		var target = WorldManager.get_entity(target_entity_id)
		if target:
			var direction = (target.global_position - global_position).normalized()
			var target_rotation = atan2(direction.x, direction.z)
			rotation.y = target_rotation

	_complete_command(true)

func _execute_follow(command: Dictionary) -> void:
	"""Follow another agent."""
	var target_id = command.get("target_entity_id")
	if not target_id:
		_complete_command(false, "no_target")
		return

	var target = WorldManager.get_entity(target_id)
	if not target:
		_complete_command(false, "target_not_found")
		return

	# Set target position behind the followed entity
	var follow_distance = 2.0
	var behind = -target.global_transform.basis.z * follow_distance
	target_position = target.global_position + behind
	has_target = true
	is_moving = true
	current_action = "following"
	action_started.emit("follow", target)

func _execute_flee(command: Dictionary) -> void:
	"""Flee from an entity."""
	var target_id = command.get("target_entity_id")
	if not target_id:
		_complete_command(false, "no_target")
		return

	var threat = WorldManager.get_entity(target_id)
	if not threat:
		_complete_command(false, "target_not_found")
		return

	# Move in opposite direction
	var away = (global_position - threat.global_position).normalized()
	target_position = global_position + away * 10.0
	has_target = true
	is_moving = true
	movement_speed = run_speed
	current_action = "fleeing"
	action_started.emit("flee", threat)

func _execute_pick_up(command: Dictionary) -> void:
	"""Pick up an object."""
	if held_object:
		_complete_command(false, "already_holding")
		return

	var target_id = command.get("target_entity_id")
	if not target_id:
		_complete_command(false, "no_target")
		return

	var target = WorldManager.get_entity(target_id)
	if not target:
		_complete_command(false, "target_not_found")
		return

	# Check distance
	var distance = global_position.distance_to(target.global_position)
	if distance > interaction_range:
		_complete_command(false, "too_far")
		return

	# Check if object can be picked up
	if not target.get("is_interactable", false):
		_complete_command(false, "not_interactable")
		return

	if not "can_pick_up" in target.get("affordances", []):
		_complete_command(false, "cannot_pick_up")
		return

	# Pick up the object
	held_object = target
	held_object_id = target.get_instance_id()
	target.set("is_being_held", true)
	target.set("held_by_id", get_instance_id())

	# Reparent to agent
	var old_parent = target.get_parent()
	old_parent.remove_child(target)
	add_child(target)
	target.position = Vector3(0, 1.5, 0.5)  # Hold position

	current_action = "idle"
	picked_up_object.emit(target)
	action_started.emit("pick_up", target)

	# Send interaction event
	if TomBridge:
		TomBridge.send_interaction_event(self, target, "pick_up", true)

	_complete_command(true)

func _execute_put_down(command: Dictionary) -> void:
	"""Put down held object."""
	if not held_object:
		_complete_command(false, "not_holding")
		return

	var drop_position = global_position + global_transform.basis.z * 1.0
	drop_position.y = 0.5  # Slightly above ground

	# Reparent to world
	var obj = held_object
	remove_child(obj)
	get_tree().current_scene.add_child(obj)
	obj.global_position = drop_position
	obj.set("is_being_held", false)
	obj.set("held_by_id", null)

	dropped_object.emit(obj)

	if TomBridge:
		TomBridge.send_interaction_event(self, obj, "put_down", true)

	held_object = null
	held_object_id = -1

	_complete_command(true)

func _execute_use(command: Dictionary) -> void:
	"""Use an object or held object."""
	var target_id = command.get("target_entity_id")
	var target = null

	if target_id:
		target = WorldManager.get_entity(target_id)
	elif held_object:
		target = held_object

	if not target:
		_complete_command(false, "no_target")
		return

	if target.has_method("use"):
		var result = target.use(self)
		if TomBridge:
			TomBridge.send_interaction_event(self, target, "use", result.success, result)
		_complete_command(result.success, result.get("error", ""))
	else:
		_complete_command(false, "cannot_use")

func _execute_examine(command: Dictionary) -> void:
	"""Examine an object."""
	var target_id = command.get("target_entity_id")
	if not target_id:
		_complete_command(false, "no_target")
		return

	var target = WorldManager.get_entity(target_id)
	if not target:
		_complete_command(false, "target_not_found")
		return

	# Turn to face target
	var direction = (target.global_position - global_position).normalized()
	var target_rotation = atan2(direction.x, direction.z)
	rotation.y = target_rotation

	current_action = "examining"
	action_started.emit("examine", target)

	if TomBridge:
		TomBridge.send_interaction_event(self, target, "examine", true)

	_complete_command(true)

func _execute_give(command: Dictionary) -> void:
	"""Give held object to another agent."""
	if not held_object:
		_complete_command(false, "not_holding")
		return

	var target_id = command.get("target_entity_id")
	if not target_id:
		_complete_command(false, "no_target")
		return

	var target = WorldManager.get_entity(target_id)
	if not target or target.get("entity_type") != "agent":
		_complete_command(false, "invalid_target")
		return

	# Check distance
	var distance = global_position.distance_to(target.global_position)
	if distance > interaction_range:
		_complete_command(false, "too_far")
		return

	# Check if target can receive
	if target.get("held_object"):
		_complete_command(false, "target_hands_full")
		return

	# Transfer object
	var obj = held_object
	remove_child(obj)
	target.add_child(obj)
	obj.position = Vector3(0, 1.5, 0.5)
	obj.set("held_by_id", target.get_instance_id())

	target.set("held_object", obj)
	target.set("held_object_id", obj.get_instance_id())

	held_object = null
	held_object_id = -1

	EventBus.object_given.emit(self, target, obj)

	if TomBridge:
		TomBridge.send_interaction_event(self, target, "give", true, {"object_id": obj.get_instance_id()})

	_complete_command(true)

func _execute_speak(command: Dictionary) -> void:
	"""Speak an utterance."""
	var text = command.get("utterance_text", "")
	if text.is_empty():
		_complete_command(false, "no_text")
		return

	var volume = command.get("speed", 1.0)  # Use speed as volume
	var target_id = command.get("target_entity_id")
	var target = WorldManager.get_entity(target_id) if target_id else null

	# Display speech
	if speech_label:
		speech_label.text = text
		speech_label.visible = true
		await get_tree().create_timer(3.0).timeout
		speech_label.visible = false

	is_speaking = true
	spoke.emit(text)
	current_action = "speaking"

	# Broadcast to perception system
	if PerceptionSystem:
		PerceptionSystem.broadcast_utterance(self, text, volume, target)

	EventBus.emit_utterance(self, text)

	is_speaking = false
	current_action = "idle"
	_complete_command(true)

func _execute_gesture(command: Dictionary) -> void:
	"""Perform a gesture."""
	var animation = command.get("animation_name", "wave")
	current_action = "gesturing"
	action_started.emit("gesture", null)

	# Play animation if available
	if has_node("AnimationPlayer"):
		var anim_player = get_node("AnimationPlayer")
		if anim_player.has_animation(animation):
			anim_player.play(animation)
			await anim_player.animation_finished

	current_action = "idle"
	_complete_command(true)

func _execute_look_at(command: Dictionary) -> void:
	"""Look at a target."""
	var target_id = command.get("target_entity_id")
	var target_pos = _get_target_position(command)

	if target_id:
		var target = WorldManager.get_entity(target_id)
		if target:
			target_pos = target.global_position

	if target_pos != Vector3.ZERO:
		var direction = (target_pos - global_position).normalized()
		var target_rotation = atan2(direction.x, direction.z)
		rotation.y = target_rotation

	_complete_command(true)

func _execute_wait(command: Dictionary) -> void:
	"""Wait for a duration."""
	var duration = command.get("timeout_seconds", 1.0)
	current_action = "waiting"

	await get_tree().create_timer(duration).timeout

	current_action = "idle"
	_complete_command(true)

func _cancel_current_action() -> void:
	"""Cancel current action."""
	has_target = false
	is_moving = false
	velocity = Vector3.ZERO
	current_action = "idle"

	if _current_command.size() > 0:
		_complete_command(false, "cancelled")

func _complete_command(success: bool, error: String = "") -> void:
	"""Mark command as completed and notify Python."""
	if _current_command.size() == 0:
		return

	var command_id = _current_command.get("command_id", "")

	if TomBridge and command_id:
		TomBridge.send_ack(command_id, success, {}, error)

	action_completed.emit(_current_command.get("command_type", ""), success)
	_current_command = {}

func _get_target_position(command: Dictionary) -> Vector3:
	"""Get target position from command."""
	if command.has("target_position"):
		return ProtocolHandler.dict_to_vector3(command.target_position)

	if command.has("target_entity_id"):
		var target = WorldManager.get_entity(command.target_entity_id)
		if target:
			return target.global_position

	return Vector3.ZERO

# Energy management

func _modify_energy(delta: float) -> void:
	var old_energy = energy
	energy = clamp(energy + delta, 0.0, 1.0)
	if energy != old_energy:
		energy_changed.emit(old_energy, energy)

func restore_energy(amount: float) -> void:
	_modify_energy(amount)

func consume_energy(amount: float) -> bool:
	if energy < amount:
		return false
	_modify_energy(-amount)
	return true

# Perception config

func get_perception_config() -> Dictionary:
	return {
		"view_distance": view_distance,
		"view_angle": view_angle,
		"hearing_distance": hearing_distance
	}

# Interaction detection

func _on_body_entered_interaction(body: Node3D) -> void:
	if body != self and body.get("is_interactable", false):
		# Can interact with this object
		pass

func _on_body_exited_interaction(body: Node3D) -> void:
	pass

# Debug

func get_state_dict() -> Dictionary:
	return {
		"name": name,
		"position": global_position,
		"rotation": rotation,
		"velocity": velocity,
		"energy": energy,
		"current_action": current_action,
		"is_moving": is_moving,
		"has_target": has_target,
		"held_object_id": held_object_id
	}
