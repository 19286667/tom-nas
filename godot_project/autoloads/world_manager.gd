extends Node
## World Manager - Entity Registry and World State
##
## Manages all entities in the simulation, tracks world state,
## and provides queries for the Python cognitive system.

class_name WorldManagerClass

# Signals
signal entity_spawned(entity: Node3D)
signal entity_removed(entity_id: int)
signal entity_updated(entity: Node3D)
signal world_reset()
signal time_changed(time_of_day: float)
signal weather_changed(weather: String)

# World state
var simulation_time: float = 0.0
var timestep: int = 0
var time_of_day: float = 12.0  # 0-24 hours
var weather: String = "clear"
var is_paused: bool = false

# Entity registries
var _entities: Dictionary = {}  # instance_id -> Node3D
var _agents: Dictionary = {}    # instance_id -> AgentNode
var _locations: Dictionary = {} # instance_id -> LocationNode
var _objects: Dictionary = {}   # instance_id -> ObjectNode

# Spatial index for efficient queries
var _spatial_hash: Dictionary = {}  # grid_cell -> Array[entity_id]
var _spatial_cell_size: float = 5.0

# Prefab paths
const PREFABS = {
	"agent": "res://scenes/entities/agent.tscn",
	"object": "res://scenes/entities/interactable_object.tscn",
	"location": "res://scenes/entities/location_zone.tscn",
	"item": "res://scenes/entities/item.tscn"
}

# Update frequency settings
var _entity_update_interval: float = 0.1  # Send updates every 100ms
var _entity_update_timer: float = 0.0
var _changed_entities: Array[int] = []

func _ready() -> void:
	# Connect to simulation control
	if TomBridge:
		TomBridge.simulation_control.connect(_on_simulation_control)

func _process(delta: float) -> void:
	if is_paused:
		return

	simulation_time += delta
	timestep += 1

	# Update time of day (1 game hour = 1 real minute)
	time_of_day += delta / 60.0
	if time_of_day >= 24.0:
		time_of_day -= 24.0

	# Send periodic entity updates
	_entity_update_timer += delta
	if _entity_update_timer >= _entity_update_interval:
		_entity_update_timer = 0.0
		_send_changed_entities()

func _on_simulation_control(control: String) -> void:
	match control:
		"pause":
			is_paused = true
		"resume":
			is_paused = false
		"reset":
			reset_world()

# Entity Management

func register_entity(entity: Node3D) -> void:
	"""Register an entity with the world manager."""
	var entity_id = entity.get_instance_id()
	_entities[entity_id] = entity

	# Categorize by type
	var entity_type = entity.get("entity_type", "object")
	match entity_type:
		"agent":
			_agents[entity_id] = entity
		"location":
			_locations[entity_id] = entity
		_:
			_objects[entity_id] = entity

	# Add to spatial hash
	_add_to_spatial_hash(entity)

	# Connect signals
	if entity.has_signal("transform_changed"):
		entity.transform_changed.connect(_on_entity_transform_changed.bind(entity_id))

	entity_spawned.emit(entity)
	print("[WorldManager] Registered entity: %s (ID: %d)" % [entity.name, entity_id])

func unregister_entity(entity: Node3D) -> void:
	"""Unregister an entity from the world manager."""
	var entity_id = entity.get_instance_id()

	_remove_from_spatial_hash(entity)

	_entities.erase(entity_id)
	_agents.erase(entity_id)
	_locations.erase(entity_id)
	_objects.erase(entity_id)

	entity_removed.emit(entity_id)

func get_entity(entity_id: int) -> Node3D:
	"""Get entity by instance ID."""
	return _entities.get(entity_id)

func get_all_entities() -> Array:
	"""Get all registered entities."""
	return _entities.values()

func get_all_agents() -> Array:
	"""Get all registered agents."""
	return _agents.values()

func get_all_locations() -> Array:
	"""Get all registered locations."""
	return _locations.values()

func get_all_objects() -> Array:
	"""Get all registered objects."""
	return _objects.values()

func spawn_entity(entity_type: String, entity_name: String, position: Vector3, properties: Dictionary = {}) -> Node3D:
	"""Spawn a new entity from prefab."""
	var prefab_path = PREFABS.get(entity_type.to_lower(), PREFABS.object)

	if not ResourceLoader.exists(prefab_path):
		push_error("[WorldManager] Prefab not found: %s" % prefab_path)
		return null

	var scene = load(prefab_path)
	var entity = scene.instantiate()

	entity.name = entity_name
	entity.global_position = position

	# Apply properties
	for key in properties:
		if entity.has_method("set_" + key):
			entity.call("set_" + key, properties[key])
		elif key in entity:
			entity.set(key, properties[key])

	# Add to scene tree
	get_tree().current_scene.add_child(entity)

	# Register
	register_entity(entity)

	return entity

func modify_entity(entity_id: int, modifications: Dictionary) -> bool:
	"""Modify an existing entity's properties."""
	var entity = get_entity(entity_id)
	if not entity:
		return false

	for key in modifications:
		match key:
			"position":
				entity.global_position = ProtocolHandler.dict_to_vector3(modifications[key])
			"rotation":
				entity.global_rotation = ProtocolHandler.dict_to_vector3(modifications[key])
			"scale":
				entity.scale = ProtocolHandler.dict_to_vector3(modifications[key])
			"visible":
				entity.visible = modifications[key]
			_:
				if entity.has_method("set_" + key):
					entity.call("set_" + key, modifications[key])
				elif key in entity:
					entity.set(key, modifications[key])

	_mark_entity_changed(entity_id)
	entity_updated.emit(entity)
	return true

func remove_entity(entity_id: int) -> bool:
	"""Remove an entity from the world."""
	var entity = get_entity(entity_id)
	if not entity:
		return false

	unregister_entity(entity)
	entity.queue_free()
	return true

# Spatial Queries

func get_entities_in_radius(center: Vector3, radius: float, filter_type: String = "") -> Array:
	"""Get all entities within a radius of a point."""
	var result = []
	var radius_sq = radius * radius

	for entity in _entities.values():
		if filter_type and entity.get("entity_type", "") != filter_type:
			continue

		var dist_sq = entity.global_position.distance_squared_to(center)
		if dist_sq <= radius_sq:
			result.append(entity)

	return result

func get_nearest_entity(from: Vector3, filter_type: String = "", exclude: Array = []) -> Node3D:
	"""Get the nearest entity to a position."""
	var nearest: Node3D = null
	var nearest_dist_sq = INF

	for entity in _entities.values():
		if entity in exclude:
			continue
		if filter_type and entity.get("entity_type", "") != filter_type:
			continue

		var dist_sq = entity.global_position.distance_squared_to(from)
		if dist_sq < nearest_dist_sq:
			nearest_dist_sq = dist_sq
			nearest = entity

	return nearest

func get_entities_in_area(aabb: AABB, filter_type: String = "") -> Array:
	"""Get all entities within an AABB."""
	var result = []

	for entity in _entities.values():
		if filter_type and entity.get("entity_type", "") != filter_type:
			continue

		if aabb.has_point(entity.global_position):
			result.append(entity)

	return result

func raycast_entities(from: Vector3, to: Vector3, exclude: Array = []) -> Dictionary:
	"""Raycast and return first entity hit."""
	var space_state = get_tree().current_scene.get_world_3d().direct_space_state

	var query = PhysicsRayQueryParameters3D.create(from, to)
	query.exclude = []
	for e in exclude:
		if e is Node3D and e.has_node("CollisionShape3D"):
			var body = e.get_node_or_null(".")
			if body is PhysicsBody3D:
				query.exclude.append(body.get_rid())

	var result = space_state.intersect_ray(query)

	if result:
		return {
			"hit": true,
			"position": result.position,
			"normal": result.normal,
			"collider": result.collider,
			"entity": _find_entity_from_collider(result.collider)
		}

	return {"hit": false}

func _find_entity_from_collider(collider: Node) -> Node3D:
	"""Find the entity node from a collider."""
	var node = collider
	while node:
		if node.get_instance_id() in _entities:
			return node
		node = node.get_parent()
	return null

# Spatial Hash

func _add_to_spatial_hash(entity: Node3D) -> void:
	var cell = _get_spatial_cell(entity.global_position)
	if cell not in _spatial_hash:
		_spatial_hash[cell] = []
	_spatial_hash[cell].append(entity.get_instance_id())

func _remove_from_spatial_hash(entity: Node3D) -> void:
	var cell = _get_spatial_cell(entity.global_position)
	if cell in _spatial_hash:
		_spatial_hash[cell].erase(entity.get_instance_id())

func _update_spatial_hash(entity: Node3D, old_pos: Vector3) -> void:
	var old_cell = _get_spatial_cell(old_pos)
	var new_cell = _get_spatial_cell(entity.global_position)

	if old_cell != new_cell:
		if old_cell in _spatial_hash:
			_spatial_hash[old_cell].erase(entity.get_instance_id())
		if new_cell not in _spatial_hash:
			_spatial_hash[new_cell] = []
		_spatial_hash[new_cell].append(entity.get_instance_id())

func _get_spatial_cell(pos: Vector3) -> Vector3i:
	return Vector3i(
		int(floor(pos.x / _spatial_cell_size)),
		int(floor(pos.y / _spatial_cell_size)),
		int(floor(pos.z / _spatial_cell_size))
	)

# World State

func get_world_state_payload() -> Dictionary:
	"""Create world state payload for Python."""
	var entities_data = []
	for entity in _entities.values():
		entities_data.append(_create_entity_data(entity))

	var agents_data = []
	for agent in _agents.values():
		agents_data.append(_create_agent_data(agent))

	var locations_data = []
	for location in _locations.values():
		locations_data.append(_create_location_data(location))

	return ProtocolHandler.create_world_state(
		entities_data,
		agents_data,
		locations_data,
		simulation_time,
		timestep,
		is_paused,
		time_of_day,
		weather,
		InstitutionManager.get_active_institution() if InstitutionManager else ""
	)

func _create_entity_data(entity: Node3D) -> Dictionary:
	return ProtocolHandler.create_entity_update(
		entity.get_instance_id(),
		entity.get("entity_type", "object"),
		entity.name,
		entity.global_position,
		entity.global_rotation,
		entity.scale,
		entity.get("velocity", Vector3.ZERO),
		entity.get("is_static", true),
		entity.visible,
		entity.get("semantic_tags", []),
		entity.get("affordances", []),
		entity.get("is_interactable", false),
		simulation_time
	)

func _create_agent_data(agent: Node3D) -> Dictionary:
	var data = _create_entity_data(agent)
	data["energy_level"] = agent.get("energy", 1.0)
	data["held_object"] = agent.get("held_object_id")
	data["current_action"] = agent.get("current_action", "idle")
	return data

func _create_location_data(location: Node3D) -> Dictionary:
	return {
		"godot_id": location.get_instance_id(),
		"name": location.name,
		"position": ProtocolHandler.vector3_to_dict(location.global_position),
		"bounds": {
			"size": ProtocolHandler.vector3_to_dict(location.get("bounds_size", Vector3(10, 5, 10)))
		},
		"institution": location.get("institution", ""),
		"semantic_tags": location.get("semantic_tags", [])
	}

# Change tracking

func _mark_entity_changed(entity_id: int) -> void:
	if entity_id not in _changed_entities:
		_changed_entities.append(entity_id)

func _on_entity_transform_changed(entity_id: int) -> void:
	_mark_entity_changed(entity_id)

func _send_changed_entities() -> void:
	"""Send updates for changed entities to Python."""
	if not TomBridge or not TomBridge.is_connected_to_bridge():
		_changed_entities.clear()
		return

	for entity_id in _changed_entities:
		var entity = get_entity(entity_id)
		if entity:
			TomBridge.send_entity_update(entity)

	_changed_entities.clear()

# World Control

func set_time_of_day(time: float) -> void:
	time_of_day = clamp(time, 0.0, 24.0)
	time_changed.emit(time_of_day)

func set_weather(new_weather: String) -> void:
	weather = new_weather
	weather_changed.emit(weather)

func reset_world() -> void:
	"""Reset the world to initial state."""
	# Clear all entities
	for entity in _entities.values():
		if is_instance_valid(entity):
			entity.queue_free()

	_entities.clear()
	_agents.clear()
	_locations.clear()
	_objects.clear()
	_spatial_hash.clear()
	_changed_entities.clear()

	# Reset state
	simulation_time = 0.0
	timestep = 0
	time_of_day = 12.0
	weather = "clear"
	is_paused = false

	world_reset.emit()

	# Reload main scene
	get_tree().reload_current_scene()

# Debug

func get_debug_info() -> Dictionary:
	return {
		"total_entities": _entities.size(),
		"agents": _agents.size(),
		"locations": _locations.size(),
		"objects": _objects.size(),
		"simulation_time": simulation_time,
		"timestep": timestep,
		"time_of_day": time_of_day,
		"weather": weather,
		"is_paused": is_paused
	}
