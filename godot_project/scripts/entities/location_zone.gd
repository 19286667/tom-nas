extends Area3D
## Location Zone - Spatial regions with institutional context
##
## Defines areas in the world that have specific institutional contexts
## and affect agent behavior through norms and roles.

class_name LocationZone

# Configuration
@export var location_name: String = "Location"
@export var location_description: String = ""
@export_enum("NONE", "HOLLOW", "MARKET", "MINISTRY", "COURT", "TEMPLE") var institution_type: String = "NONE"
@export var bounds_size: Vector3 = Vector3(10, 5, 10)

# Semantic properties
@export var semantic_tags: Array[String] = ["location", "zone"]
@export var default_norms: Array[String] = []
@export var allowed_actions: Array[String] = []
@export var forbidden_actions: Array[String] = []

# Entity metadata
var entity_type: String = "location"
var is_interactable: bool = false
var is_static: bool = true
var affordances: Array = ["can_enter", "can_exit"]

# State
var agents_inside: Array = []
var objects_inside: Array = []

# Signals
signal agent_entered(agent: Node3D)
signal agent_exited(agent: Node3D)
signal object_entered(object: Node3D)
signal object_exited(object: Node3D)

func _ready() -> void:
	name = location_name

	# Register with world manager
	call_deferred("_register_with_world")

	# Connect signals
	body_entered.connect(_on_body_entered)
	body_exited.connect(_on_body_exited)

	# Register institution zone
	call_deferred("_register_institution")

func _register_with_world() -> void:
	if WorldManager:
		WorldManager.register_entity(self)

func _register_institution() -> void:
	if InstitutionManager and institution_type != "NONE":
		var inst = InstitutionManager.institution_from_string(institution_type)
		InstitutionManager.register_institution_zone(self, inst)

func _exit_tree() -> void:
	if WorldManager:
		WorldManager.unregister_entity(self)

	if InstitutionManager:
		InstitutionManager.unregister_institution_zone(self)

func _on_body_entered(body: Node3D) -> void:
	"""Handle body entering the zone."""
	var entity_type_str = body.get("entity_type", "")

	if entity_type_str == "agent":
		if body not in agents_inside:
			agents_inside.append(body)

		# Set agent institution
		if InstitutionManager and institution_type != "NONE":
			var inst = InstitutionManager.institution_from_string(institution_type)
			InstitutionManager.set_agent_institution(body, inst)

		agent_entered.emit(body)
		EventBus.location_entered.emit(body, self)

	else:
		if body not in objects_inside:
			objects_inside.append(body)
		object_entered.emit(body)

func _on_body_exited(body: Node3D) -> void:
	"""Handle body exiting the zone."""
	var entity_type_str = body.get("entity_type", "")

	if entity_type_str == "agent":
		agents_inside.erase(body)

		# Clear agent institution
		if InstitutionManager:
			InstitutionManager.set_agent_institution(body, InstitutionManager.InstitutionType.NONE)

		agent_exited.emit(body)
		EventBus.location_exited.emit(body, self)

	else:
		objects_inside.erase(body)
		object_exited.emit(body)

# Query methods

func get_agents_inside() -> Array:
	"""Get all agents currently in this zone."""
	# Clean up invalid references
	agents_inside = agents_inside.filter(func(a): return is_instance_valid(a))
	return agents_inside

func get_objects_inside() -> Array:
	"""Get all objects currently in this zone."""
	objects_inside = objects_inside.filter(func(o): return is_instance_valid(o))
	return objects_inside

func is_agent_inside(agent: Node3D) -> bool:
	"""Check if a specific agent is in this zone."""
	return agent in agents_inside

func get_agent_count() -> int:
	"""Get number of agents in this zone."""
	return get_agents_inside().size()

# Action validation

func is_action_allowed(action: String) -> bool:
	"""Check if an action is allowed in this zone."""
	if action in forbidden_actions:
		return false

	if allowed_actions.size() > 0:
		return action in allowed_actions

	return true  # No restrictions

func get_institution_data() -> Dictionary:
	"""Get institution data for this zone."""
	if InstitutionManager and institution_type != "NONE":
		var inst = InstitutionManager.institution_from_string(institution_type)
		return InstitutionManager.get_institution_data(inst)
	return {}

# State

func get_state_dict() -> Dictionary:
	return {
		"name": location_name,
		"description": location_description,
		"institution": institution_type,
		"position": global_position,
		"bounds_size": bounds_size,
		"agent_count": get_agent_count(),
		"semantic_tags": semantic_tags,
		"norms": default_norms
	}
