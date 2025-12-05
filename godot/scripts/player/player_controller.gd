extends CharacterBody3D
class_name PlayerController
## Simple first-person player controller for testing NPC interactions

@export var move_speed: float = 5.0
@export var mouse_sensitivity: float = 0.002
@export var interaction_range: float = 3.0

@onready var camera: Camera3D = $Camera3D
@onready var interaction_raycast: RayCast3D = $Camera3D/InteractionRay

var camera_rotation: Vector2 = Vector2.ZERO
var looking_at: Node3D = null


func _ready() -> void:
	add_to_group("player")
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	
	# Setup interaction raycast
	if interaction_raycast:
		interaction_raycast.target_position = Vector3(0, 0, -interaction_range)
		interaction_raycast.enabled = true


func _input(event: InputEvent) -> void:
	# Mouse look
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		camera_rotation.x -= event.relative.y * mouse_sensitivity
		camera_rotation.y -= event.relative.x * mouse_sensitivity
		camera_rotation.x = clamp(camera_rotation.x, -PI/2, PI/2)
		
		rotation.y = camera_rotation.y
		camera.rotation.x = camera_rotation.x
	
	# Toggle mouse capture
	if event.is_action_pressed("ui_cancel"):
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	
	# Interaction
	if event.is_action_pressed("interact"):
		_try_interact()
	
	# Soul Scanner (inspect)
	if event.is_action_pressed("inspect"):
		_try_inspect()
	
	# Debug toggle
	if event.is_action_pressed("debug_toggle"):
		EventBus.debug_panel_toggled.emit(true)


func _physics_process(_delta: float) -> void:
	# Get input direction
	var input_dir = Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var direction = (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	
	# Apply movement
	if direction:
		velocity.x = direction.x * move_speed
		velocity.z = direction.z * move_speed
	else:
		velocity.x = move_toward(velocity.x, 0, move_speed)
		velocity.z = move_toward(velocity.z, 0, move_speed)
	
	# Gravity
	if not is_on_floor():
		velocity.y -= 9.8 * _delta
	
	move_and_slide()
	
	# Update what we're looking at
	_update_looking_at()


func _update_looking_at() -> void:
	"""Check what entity we're looking at"""
	if not interaction_raycast:
		return
	
	if interaction_raycast.is_colliding():
		var collider = interaction_raycast.get_collider()
		if collider != looking_at:
			looking_at = collider
			# Could emit signal for UI crosshair changes
	else:
		looking_at = null


func _try_interact() -> void:
	"""Attempt to interact with whatever we're looking at"""
	if not looking_at:
		return
	
	if looking_at is NPCController:
		var npc = looking_at as NPCController
		print("[Player] Interacting with NPC: %s" % npc.npc_id)
		EventBus.player_interacted.emit(npc.npc_id, "talk")
		
		# Report to Python
		GameBridge.report_player_action("interact", npc.npc_id, {
			"interaction_type": "talk"
		})
		
		# Request dialogue
		if GameBridge.is_connected_to_backend():
			GameBridge.request_dialogue(npc.npc_id, {
				"player_position": global_position,
				"relationship": SoulMapManager.get_relationship(npc.npc_id, "player"),
				"history": []
			}, _on_dialogue_received)


func _try_inspect() -> void:
	"""Use Soul Scanner on target"""
	if not looking_at:
		return
	
	if looking_at is NPCController:
		var npc = looking_at as NPCController
		print("[Player] Inspecting NPC: %s" % npc.npc_id)
		EventBus.show_soul_scanner.emit(npc.npc_id)


func _on_dialogue_received(response: Dictionary) -> void:
	"""Handle dialogue response from Python"""
	var npc_id = response.get("npc_id", "")
	var text = response.get("text", "...")
	var choices = response.get("choices", [])
	
	EventBus.show_dialogue_ui.emit(npc_id, text, choices)
