extends RigidBody3D
## Interactable Object - Objects that agents can interact with
##
## Base class for all interactable objects in the ToM-NAS simulation.
## Supports pick up, use, examine, and other interactions.

class_name InteractableObject

# Configuration
@export var object_name: String = "Object"
@export var object_description: String = "An object"
@export var can_be_picked_up: bool = true
@export var can_be_used: bool = false
@export var use_energy_cost: float = 0.1
@export var use_cooldown: float = 1.0

# Semantic properties
@export var semantic_tags: Array[String] = ["object", "inanimate"]
@export_enum("mundane", "valuable", "tool", "food", "container") var category: String = "mundane"

# Entity metadata
var entity_type: String = "object"
var is_interactable: bool = true
var is_static: bool = false

# State
var is_being_held: bool = false
var held_by_id: int = -1
var owner_id: int = -1
var last_use_time: float = 0.0

# Affordances (computed)
var affordances: Array = []

# Signals
signal picked_up(by_agent: Node3D)
signal dropped(at_position: Vector3)
signal used(by_agent: Node3D, result: Dictionary)
signal examined(by_agent: Node3D)

func _ready() -> void:
	name = object_name
	_compute_affordances()

	# Register with world manager
	call_deferred("_register_with_world")

func _register_with_world() -> void:
	if WorldManager:
		WorldManager.register_entity(self)

func _exit_tree() -> void:
	if WorldManager:
		WorldManager.unregister_entity(self)

func _compute_affordances() -> void:
	"""Compute available affordances based on configuration."""
	affordances.clear()
	affordances.append("can_examine")

	if can_be_picked_up:
		affordances.append("can_pick_up")

	if can_be_used:
		affordances.append("can_use")

	# Category-based affordances
	match category:
		"food":
			affordances.append("can_eat")
		"container":
			affordances.append("can_open")
			affordances.append("can_store")
		"tool":
			affordances.append("can_use_on")

# Interaction methods

func pick_up(agent: Node3D) -> Dictionary:
	"""Called when agent picks up this object."""
	if not can_be_picked_up:
		return {"success": false, "error": "cannot_pick_up"}

	if is_being_held:
		return {"success": false, "error": "already_held"}

	is_being_held = true
	held_by_id = agent.get_instance_id()

	# Disable physics while held
	freeze = true

	picked_up.emit(agent)
	return {"success": true}

func drop(position: Vector3) -> Dictionary:
	"""Called when agent drops this object."""
	is_being_held = false
	held_by_id = -1

	# Re-enable physics
	freeze = false
	global_position = position

	dropped.emit(position)
	return {"success": true}

func use(agent: Node3D) -> Dictionary:
	"""Called when agent uses this object."""
	if not can_be_used:
		return {"success": false, "error": "cannot_use"}

	# Check cooldown
	var current_time = WorldManager.simulation_time if WorldManager else 0.0
	if current_time - last_use_time < use_cooldown:
		return {"success": false, "error": "on_cooldown"}

	# Check agent energy
	if agent.has_method("consume_energy"):
		if not agent.consume_energy(use_energy_cost):
			return {"success": false, "error": "insufficient_energy"}

	last_use_time = current_time

	# Override in subclasses for specific behavior
	var result = _on_use(agent)
	used.emit(agent, result)
	return result

func _on_use(agent: Node3D) -> Dictionary:
	"""Override in subclasses for specific use behavior."""
	return {"success": true, "effect": "none"}

func examine(agent: Node3D) -> Dictionary:
	"""Called when agent examines this object."""
	examined.emit(agent)
	return {
		"success": true,
		"name": object_name,
		"description": object_description,
		"category": category,
		"semantic_tags": semantic_tags,
		"affordances": affordances
	}

# Property setters

func set_owner(agent: Node3D) -> void:
	"""Set the owner of this object."""
	owner_id = agent.get_instance_id() if agent else -1

func clear_owner() -> void:
	"""Clear ownership."""
	owner_id = -1

func is_owned() -> bool:
	"""Check if object has an owner."""
	return owner_id >= 0

func get_owner() -> Node3D:
	"""Get the owner agent."""
	if owner_id < 0:
		return null
	return WorldManager.get_entity(owner_id) if WorldManager else null

# State

func get_state_dict() -> Dictionary:
	return {
		"name": object_name,
		"description": object_description,
		"category": category,
		"position": global_position,
		"is_being_held": is_being_held,
		"held_by_id": held_by_id,
		"owner_id": owner_id,
		"semantic_tags": semantic_tags,
		"affordances": affordances
	}
