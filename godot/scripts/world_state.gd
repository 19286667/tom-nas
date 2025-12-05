extends Node
## World State Manager for ToM-NAS
##
## This autoload singleton tracks the complete state of the simulation world.
## It provides methods for:
## - Entity registration and tracking
## - World state serialization for Python
## - Environment control (time, weather)
## - Entity spawning and modification

class_name WorldStateClass

# Signals
signal entity_registered(entity: Node3D)
signal entity_unregistered(entity: Node3D)
signal world_reset()
signal time_of_day_changed(time: float)
signal weather_changed(weather: String)

# World state
var simulation_time: float = 0.0
var timestep: int = 0
var is_paused: bool = false
var time_of_day: float = 12.0  # 0-24 hours
var weather: String = "clear"
var active_institution: String = ""

# Entity tracking
var entities: Dictionary = {}  # instance_id -> Node3D
var agents: Dictionary = {}    # instance_id -> Node3D
var locations: Dictionary = {} # instance_id -> Node3D

# Scene references
var spawn_container: Node3D = null


func _ready() -> void:
	# Create spawn container if not exists
	spawn_container = Node3D.new()
	spawn_container.name = "SpawnedEntities"
	add_child(spawn_container)


func _physics_process(delta: float) -> void:
	if not get_tree().paused:
		simulation_time += delta
		timestep += 1

		# Update time of day (1 game hour = 1 real minute by default)
		time_of_day = fmod(time_of_day + delta / 60.0, 24.0)


## Register an entity with the world state tracker
func register_entity(entity: Node3D) -> void:
	var id = entity.get_instance_id()

	if entity.is_in_group("agents"):
		agents[id] = entity
	elif entity.is_in_group("locations"):
		locations[id] = entity
	else:
		entities[id] = entity

	entity_registered.emit(entity)


## Unregister an entity from tracking
func unregister_entity(entity: Node3D) -> void:
	var id = entity.get_instance_id()

	entities.erase(id)
	agents.erase(id)
	locations.erase(id)

	entity_unregistered.emit(entity)


## Get the complete world state as a dictionary for Python
func get_world_state_dict() -> Dictionary:
	var entity_list: Array = []
	var agent_list: Array = []
	var location_list: Array = []

	# Serialize entities
	for entity in entities.values():
		if is_instance_valid(entity):
			entity_list.append(_entity_to_dict(entity))

	# Serialize agents
	for agent in agents.values():
		if is_instance_valid(agent):
			var agent_dict = _entity_to_dict(agent)
			# Add agent-specific data
			if agent.has_method("get_agent_state"):
				agent_dict.merge(agent.get_agent_state())
			agent_list.append(agent_dict)

	# Serialize locations
	for location in locations.values():
		if is_instance_valid(location):
			var loc_dict = {
				"godot_id": location.get_instance_id(),
				"name": location.name,
				"position": _vector3_to_dict(location.global_position),
			}
			if location.has_meta("institution"):
				loc_dict["institution"] = location.get_meta("institution")
			if location.has_method("get_bounds"):
				loc_dict["bounds"] = location.get_bounds()
			location_list.append(loc_dict)

	return {
		"entities": entity_list,
		"agents": agent_list,
		"locations": location_list,
		"simulation_time": simulation_time,
		"timestep": timestep,
		"is_paused": is_paused,
		"time_of_day": time_of_day,
		"weather": weather,
		"active_institution": active_institution,
	}


## Reset the world to initial state
func reset_world() -> void:
	# Clear spawned entities
	for child in spawn_container.get_children():
		child.queue_free()

	# Reset state
	simulation_time = 0.0
	timestep = 0
	time_of_day = 12.0
	weather = "clear"
	active_institution = ""

	# Notify
	world_reset.emit()

	# Reload current scene (optional - can be configured)
	# get_tree().reload_current_scene()


## Spawn a new entity from Python command
func spawn_entity(payload: Dictionary) -> Node3D:
	var entity_type = payload.get("entity_type", "object")
	var entity_name = payload.get("name", "SpawnedEntity")
	var position = _dict_to_vector3(payload.get("position", {}))
	var rotation = _dict_to_vector3(payload.get("rotation", {}))
	var scale_vec = _dict_to_vector3(payload.get("scale", {"x": 1, "y": 1, "z": 1}))

	var entity: Node3D = null

	match entity_type:
		"agent":
			entity = _spawn_agent(payload)
		"object":
			entity = _spawn_object(payload)
		"location":
			entity = _spawn_location(payload)
		_:
			entity = _spawn_object(payload)

	if entity:
		entity.name = entity_name
		entity.global_position = position
		entity.global_rotation = rotation
		entity.scale = scale_vec

		spawn_container.add_child(entity)
		register_entity(entity)

	return entity


func _spawn_agent(payload: Dictionary) -> Node3D:
	# Create a basic CharacterBody3D agent
	var agent = CharacterBody3D.new()
	agent.add_to_group("agents")

	# Add collision shape
	var collision = CollisionShape3D.new()
	var capsule = CapsuleShape3D.new()
	capsule.radius = 0.3
	capsule.height = 1.8
	collision.shape = capsule
	agent.add_child(collision)

	# Add visual mesh
	var mesh_instance = MeshInstance3D.new()
	var capsule_mesh = CapsuleMesh.new()
	capsule_mesh.radius = 0.3
	capsule_mesh.height = 1.8
	mesh_instance.mesh = capsule_mesh
	agent.add_child(mesh_instance)

	# Store metadata
	if payload.has("soul_map"):
		agent.set_meta("soul_map", payload.get("soul_map"))

	return agent


func _spawn_object(payload: Dictionary) -> Node3D:
	var is_static = payload.get("is_static", true)
	var body: Node3D

	if is_static:
		body = StaticBody3D.new()
	else:
		body = RigidBody3D.new()

	# Add collision shape (default box)
	var collision = CollisionShape3D.new()
	var box = BoxShape3D.new()
	box.size = Vector3.ONE
	collision.shape = box
	body.add_child(collision)

	# Add visual mesh
	var mesh_instance = MeshInstance3D.new()
	var box_mesh = BoxMesh.new()
	mesh_instance.mesh = box_mesh
	body.add_child(mesh_instance)

	# Set tags and affordances
	if payload.has("semantic_tags"):
		body.set_meta("semantic_tags", payload.get("semantic_tags"))
		for tag in payload.get("semantic_tags"):
			body.add_to_group("tag_" + tag)

	if payload.has("affordances"):
		body.set_meta("affordances", payload.get("affordances"))

	if payload.get("is_interactable", false):
		body.add_to_group("interactable")

	return body


func _spawn_location(payload: Dictionary) -> Node3D:
	var location = Area3D.new()
	location.add_to_group("locations")

	# Add collision shape for detection
	var collision = CollisionShape3D.new()
	var size = _dict_to_vector3(payload.get("size", {"x": 10, "y": 5, "z": 10}))
	var box = BoxShape3D.new()
	box.size = size
	collision.shape = box
	location.add_child(collision)

	# Store institution info
	if payload.has("institution"):
		location.set_meta("institution", payload.get("institution"))

	return location


## Modify an existing entity from Python command
func modify_entity(payload: Dictionary) -> void:
	var godot_id = payload.get("godot_id", 0)
	var entity: Node3D = null

	# Find entity by ID
	if entities.has(godot_id):
		entity = entities[godot_id]
	elif agents.has(godot_id):
		entity = agents[godot_id]
	elif locations.has(godot_id):
		entity = locations[godot_id]

	if not is_instance_valid(entity):
		return

	# Apply modifications
	if payload.has("position"):
		entity.global_position = _dict_to_vector3(payload.get("position"))

	if payload.has("rotation"):
		entity.global_rotation = _dict_to_vector3(payload.get("rotation"))

	if payload.has("scale"):
		entity.scale = _dict_to_vector3(payload.get("scale"))

	if payload.has("visible"):
		entity.visible = payload.get("visible")

	if payload.has("semantic_tags"):
		entity.set_meta("semantic_tags", payload.get("semantic_tags"))

	if payload.has("destroy") and payload.get("destroy"):
		unregister_entity(entity)
		entity.queue_free()


## Set time of day (0-24)
func set_time_of_day(time: float) -> void:
	time_of_day = clamp(time, 0.0, 24.0)
	time_of_day_changed.emit(time_of_day)


## Set weather condition
func set_weather(new_weather: String) -> void:
	weather = new_weather
	weather_changed.emit(weather)


## Get all entities within a radius
func get_entities_in_radius(center: Vector3, radius: float) -> Array:
	var result: Array = []

	for entity in entities.values():
		if is_instance_valid(entity):
			if entity.global_position.distance_to(center) <= radius:
				result.append(entity)

	for agent in agents.values():
		if is_instance_valid(agent):
			if agent.global_position.distance_to(center) <= radius:
				result.append(agent)

	return result


## Get all agents
func get_all_agents() -> Array:
	var result: Array = []
	for agent in agents.values():
		if is_instance_valid(agent):
			result.append(agent)
	return result


## Find entity by name
func find_entity_by_name(entity_name: String) -> Node3D:
	for entity in entities.values():
		if is_instance_valid(entity) and entity.name == entity_name:
			return entity

	for agent in agents.values():
		if is_instance_valid(agent) and agent.name == entity_name:
			return agent

	return null


# --- Utility Functions ---

func _entity_to_dict(entity: Node3D) -> Dictionary:
	var semantic_tags: Array = []
	var affordances: Array = []

	if entity.has_meta("semantic_tags"):
		semantic_tags = entity.get_meta("semantic_tags")

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
		"timestamp": simulation_time,
	}


func _vector3_to_dict(v: Vector3) -> Dictionary:
	return {"x": v.x, "y": v.y, "z": v.z}


func _dict_to_vector3(d: Dictionary) -> Vector3:
	return Vector3(
		d.get("x", 0.0),
		d.get("y", 0.0),
		d.get("z", 0.0)
	)
