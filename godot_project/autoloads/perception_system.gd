extends Node
## Perception System - Agent Sensory Processing
##
## Manages visual, auditory, and other perceptions for agents.
## Provides raycasting-based visibility with occlusion detection.

class_name PerceptionSystemClass

# Configuration
const DEFAULT_VIEW_DISTANCE: float = 20.0
const DEFAULT_VIEW_ANGLE: float = 120.0  # degrees
const DEFAULT_HEARING_DISTANCE: float = 15.0
const PERCEPTION_UPDATE_INTERVAL: float = 0.1  # 100ms

# Signals
signal perception_updated(agent: Node3D, perception: Dictionary)
signal entity_seen(observer: Node3D, entity: Node3D)
signal entity_lost(observer: Node3D, entity: Node3D)
signal utterance_heard(listener: Node3D, speaker: Node3D, text: String)

# State
var _perception_timer: float = 0.0
var _agent_perceptions: Dictionary = {}  # agent_id -> perception data
var _previous_visible: Dictionary = {}  # agent_id -> Set of visible entity ids

# Utterance queue (recent utterances that can be heard)
var _active_utterances: Array = []
const UTTERANCE_LIFETIME: float = 2.0

func _process(delta: float) -> void:
	# Clean up old utterances
	_clean_old_utterances(delta)

	# Update perceptions periodically
	_perception_timer += delta
	if _perception_timer >= PERCEPTION_UPDATE_INTERVAL:
		_perception_timer = 0.0
		_update_all_perceptions()

func _update_all_perceptions() -> void:
	"""Update perceptions for all agents."""
	for agent in WorldManager.get_all_agents():
		if agent.has_method("get_perception_config"):
			var perception = calculate_perception(agent)
			_agent_perceptions[agent.get_instance_id()] = perception

			# Send to Python bridge
			if TomBridge and TomBridge.is_connected_to_bridge():
				TomBridge.send_agent_perception(agent, perception)

			perception_updated.emit(agent, perception)

func calculate_perception(agent: Node3D) -> Dictionary:
	"""Calculate full perception for an agent."""
	var config = _get_perception_config(agent)

	var perception = {
		"visible": [],
		"occluded": [],
		"utterances": [],
		"nearby_sounds": []
	}

	# Get visual perception
	var visual = _calculate_visual_perception(agent, config)
	perception.visible = visual.visible
	perception.occluded = visual.occluded

	# Track visibility changes
	_track_visibility_changes(agent, visual.visible_ids)

	# Get auditory perception
	var auditory = _calculate_auditory_perception(agent, config)
	perception.utterances = auditory.utterances
	perception.nearby_sounds = auditory.sounds

	return perception

func _get_perception_config(agent: Node3D) -> Dictionary:
	"""Get perception configuration for an agent."""
	if agent.has_method("get_perception_config"):
		return agent.get_perception_config()

	return {
		"view_distance": DEFAULT_VIEW_DISTANCE,
		"view_angle": DEFAULT_VIEW_ANGLE,
		"hearing_distance": DEFAULT_HEARING_DISTANCE
	}

func _calculate_visual_perception(agent: Node3D, config: Dictionary) -> Dictionary:
	"""Calculate what an agent can see."""
	var visible_entities = []
	var occluded_entities = []
	var visible_ids = []

	var view_distance = config.get("view_distance", DEFAULT_VIEW_DISTANCE)
	var view_angle = config.get("view_angle", DEFAULT_VIEW_ANGLE)
	var half_angle = deg_to_rad(view_angle / 2.0)

	var agent_pos = agent.global_position + Vector3(0, 1.6, 0)  # Eye height
	var agent_forward = -agent.global_transform.basis.z

	# Get entities in range
	var candidates = WorldManager.get_entities_in_radius(agent.global_position, view_distance)

	for entity in candidates:
		if entity == agent:
			continue

		var entity_pos = entity.global_position + Vector3(0, 0.5, 0)  # Center height
		var to_entity = entity_pos - agent_pos
		var distance = to_entity.length()

		if distance < 0.1:
			continue

		# Check field of view
		var direction = to_entity.normalized()
		var angle = agent_forward.angle_to(direction)

		if angle > half_angle:
			continue  # Outside FOV

		# Check occlusion with raycast
		var is_visible = _check_line_of_sight(agent_pos, entity_pos, agent, entity)

		var entity_data = _create_entity_perception_data(entity, distance)

		if is_visible:
			visible_entities.append(entity_data)
			visible_ids.append(entity.get_instance_id())
		else:
			occluded_entities.append(entity.get_instance_id())

	return {
		"visible": visible_entities,
		"occluded": occluded_entities,
		"visible_ids": visible_ids
	}

func _check_line_of_sight(from: Vector3, to: Vector3, exclude_from: Node3D, exclude_to: Node3D) -> bool:
	"""Check if there's a clear line of sight between two points."""
	var space_state = get_tree().current_scene.get_world_3d().direct_space_state

	var query = PhysicsRayQueryParameters3D.create(from, to)
	query.collision_mask = 1  # World layer only for occlusion

	# Exclude the observer and target from raycast
	var excludes = []
	if exclude_from is CollisionObject3D:
		excludes.append(exclude_from.get_rid())
	if exclude_to is CollisionObject3D:
		excludes.append(exclude_to.get_rid())
	query.exclude = excludes

	var result = space_state.intersect_ray(query)

	if result.is_empty():
		return true  # No obstruction

	# Check if we hit the target or something else
	var hit_point = result.position
	var to_hit = (hit_point - from).length()
	var to_target = (to - from).length()

	# If hit point is close to target, we can see it
	return to_hit >= to_target - 0.5

func _create_entity_perception_data(entity: Node3D, distance: float) -> Dictionary:
	"""Create perception data for a single entity."""
	return {
		"godot_id": entity.get_instance_id(),
		"entity_type": entity.get("entity_type", "object"),
		"name": entity.name,
		"position": ProtocolHandler.vector3_to_dict(entity.global_position),
		"rotation": ProtocolHandler.vector3_to_dict(entity.global_rotation),
		"velocity": ProtocolHandler.vector3_to_dict(entity.get("velocity", Vector3.ZERO)),
		"distance": distance,
		"is_static": entity.get("is_static", true),
		"visible": entity.visible,
		"semantic_tags": entity.get("semantic_tags", []),
		"affordances": entity.get("affordances", []),
		"is_interactable": entity.get("is_interactable", false),
		"timestamp": WorldManager.simulation_time
	}

func _track_visibility_changes(agent: Node3D, current_visible: Array) -> void:
	"""Track which entities became visible or lost."""
	var agent_id = agent.get_instance_id()
	var previous = _previous_visible.get(agent_id, [])

	var current_set = {}
	for id in current_visible:
		current_set[id] = true

	var previous_set = {}
	for id in previous:
		previous_set[id] = true

	# Check for newly seen entities
	for id in current_visible:
		if id not in previous_set:
			var entity = WorldManager.get_entity(id)
			if entity:
				entity_seen.emit(agent, entity)

	# Check for lost entities
	for id in previous:
		if id not in current_set:
			var entity = WorldManager.get_entity(id)
			if entity:
				entity_lost.emit(agent, entity)

	_previous_visible[agent_id] = current_visible

func _calculate_auditory_perception(agent: Node3D, config: Dictionary) -> Dictionary:
	"""Calculate what an agent can hear."""
	var heard_utterances = []
	var nearby_sounds = []

	var hearing_distance = config.get("hearing_distance", DEFAULT_HEARING_DISTANCE)
	var agent_pos = agent.global_position

	# Check active utterances
	for utterance in _active_utterances:
		var speaker_pos = utterance.position
		var distance = agent_pos.distance_to(speaker_pos)

		# Calculate effective hearing range based on volume
		var effective_range = hearing_distance * utterance.volume

		if distance <= effective_range:
			# Agent can hear this utterance
			heard_utterances.append({
				"speaker_id": utterance.speaker_id,
				"speaker_name": utterance.speaker_name,
				"text": utterance.text,
				"volume": utterance.volume,
				"distance": distance,
				"timestamp": utterance.timestamp
			})

			utterance_heard.emit(agent, utterance.speaker, utterance.text)

	return {
		"utterances": heard_utterances,
		"sounds": nearby_sounds
	}

# Utterance management

func broadcast_utterance(speaker: Node3D, text: String, volume: float = 1.0, target: Node3D = null) -> void:
	"""Broadcast an utterance that can be heard by nearby agents."""
	var utterance = {
		"speaker": speaker,
		"speaker_id": speaker.get_instance_id(),
		"speaker_name": speaker.name,
		"text": text,
		"volume": volume,
		"position": speaker.global_position,
		"target": target,
		"target_id": target.get_instance_id() if target else null,
		"timestamp": WorldManager.simulation_time,
		"age": 0.0
	}

	_active_utterances.append(utterance)

	# Find hearers and send to Python bridge
	var hearers = _find_utterance_hearers(utterance)
	if TomBridge and TomBridge.is_connected_to_bridge():
		TomBridge.send_utterance_event(speaker, text, volume, target, hearers)

func _find_utterance_hearers(utterance: Dictionary) -> Array:
	"""Find all agents who can hear an utterance."""
	var hearers = []
	var speaker_pos = utterance.position
	var base_range = DEFAULT_HEARING_DISTANCE * utterance.volume

	for agent in WorldManager.get_all_agents():
		if agent == utterance.speaker:
			continue

		var distance = agent.global_position.distance_to(speaker_pos)
		if distance <= base_range:
			hearers.append(agent)

	return hearers

func _clean_old_utterances(delta: float) -> void:
	"""Remove old utterances that have expired."""
	var i = _active_utterances.size() - 1
	while i >= 0:
		_active_utterances[i].age += delta
		if _active_utterances[i].age > UTTERANCE_LIFETIME:
			_active_utterances.remove_at(i)
		i -= 1

# Query functions

func get_agent_perception(agent: Node3D) -> Dictionary:
	"""Get the last calculated perception for an agent."""
	return _agent_perceptions.get(agent.get_instance_id(), {})

func can_agent_see(observer: Node3D, target: Node3D) -> bool:
	"""Check if an observer can currently see a target."""
	var perception = get_agent_perception(observer)
	var visible = perception.get("visible", [])

	var target_id = target.get_instance_id()
	for entity in visible:
		if entity.godot_id == target_id:
			return true

	return false

func get_visible_agents(observer: Node3D) -> Array:
	"""Get all agents visible to an observer."""
	var perception = get_agent_perception(observer)
	var visible = perception.get("visible", [])
	var agents = []

	for entity_data in visible:
		if entity_data.entity_type == "agent":
			var agent = WorldManager.get_entity(entity_data.godot_id)
			if agent:
				agents.append(agent)

	return agents

func get_visible_objects(observer: Node3D) -> Array:
	"""Get all objects visible to an observer."""
	var perception = get_agent_perception(observer)
	var visible = perception.get("visible", [])
	var objects = []

	for entity_data in visible:
		if entity_data.entity_type != "agent":
			var obj = WorldManager.get_entity(entity_data.godot_id)
			if obj:
				objects.append(obj)

	return objects

# Direct perception checks (for immediate queries)

func check_visibility_now(observer: Node3D, target: Node3D) -> bool:
	"""Immediately check if observer can see target (not using cached data)."""
	var observer_pos = observer.global_position + Vector3(0, 1.6, 0)
	var target_pos = target.global_position + Vector3(0, 0.5, 0)

	# Check distance
	var config = _get_perception_config(observer)
	var distance = observer_pos.distance_to(target_pos)
	if distance > config.view_distance:
		return false

	# Check angle
	var agent_forward = -observer.global_transform.basis.z
	var to_target = (target_pos - observer_pos).normalized()
	var angle = agent_forward.angle_to(to_target)
	var half_angle = deg_to_rad(config.view_angle / 2.0)

	if angle > half_angle:
		return false

	# Check line of sight
	return _check_line_of_sight(observer_pos, target_pos, observer, target)

# Debug

func get_debug_info() -> Dictionary:
	return {
		"tracked_agents": _agent_perceptions.size(),
		"active_utterances": _active_utterances.size()
	}

func draw_perception_cone(agent: Node3D, color: Color = Color.GREEN) -> void:
	"""Debug: Draw perception cone for an agent."""
	# This would need to be implemented with ImmediateMesh or similar
	pass
