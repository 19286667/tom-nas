extends CharacterBody3D
class_name NPCController
## NPCController: Agent with tiered cognitive processing
## Tier 0: Reactive (immediate, in Godot)
## Tier 1: Heuristic (fast, in Godot using soul map)
## Tier 2: Strategic (queries Python)
## Tier 3: Deep ToM (queries Python for recursive reasoning)

signal state_changed(old_state: String, new_state: String)
signal perception_updated(perceived_entities: Array)
signal decision_made(decision: Dictionary)

# Identity
@export var npc_id: String = ""
@export var display_name: String = "Unknown"
@export var archetype: String = "neutral"

# Soul map reference (synced from SoulMapManager)
var soul_map: Dictionary = {}

# Current state
enum State { IDLE, PATROL, APPROACH, FLEE, ATTACK, CONVERSE, INVESTIGATE }
var current_state: State = State.IDLE
var previous_state: State = State.IDLE

# Perception
@export var perception_radius: float = 10.0
var perceived_entities: Dictionary = {}  # entity_id -> data
var attention_target: Node3D = null

# Navigation
@export var move_speed: float = 3.0
@export var run_speed: float = 6.0
var nav_agent: NavigationAgent3D
var current_destination: Vector3 = Vector3.ZERO

# Decision making
var pending_strategic_query: bool = false
var last_decision_time: float = 0.0
var decision_cooldown: float = 0.5  # Minimum time between decisions

# Visual
@onready var mesh: MeshInstance3D = $Mesh
@onready var perception_area: Area3D = $PerceptionArea


func _ready() -> void:
	# Generate ID if not set
	if npc_id == "":
		npc_id = "npc_%d" % get_instance_id()
	
	# Add to group for easy lookup
	add_to_group("npcs")
	
	# Register with SoulMapManager
	var initial_soul_map = SoulMapManager.generate_random_soul_map(archetype)
	SoulMapManager.register_npc(npc_id, initial_soul_map)
	soul_map = SoulMapManager.get_soul_map(npc_id)
	
	# Setup navigation
	if has_node("NavigationAgent3D"):
		nav_agent = $NavigationAgent3D
	
	# Setup perception
	if perception_area:
		perception_area.body_entered.connect(_on_body_entered_perception)
		perception_area.body_exited.connect(_on_body_exited_perception)
		_update_perception_radius()
	
	# Connect to events
	SoulMapManager.soul_map_updated.connect(_on_soul_map_updated)
	GameBridge.connected.connect(_on_bridge_connected)
	
	# Visual indication of archetype
	_update_visual()
	
	# Notify spawn
	EventBus.npc_spawned.emit(npc_id, self)
	print("[NPC %s] Spawned with archetype: %s" % [npc_id, archetype])


func _physics_process(delta: float) -> void:
	# Update perception
	_process_perception()
	
	# Tier 0: Immediate reactive responses
	_process_tier0()
	
	# Tier 1: Heuristic decisions based on soul map
	if not pending_strategic_query:
		_process_tier1(delta)
	
	# Movement
	_process_movement(delta)


# =============================================================================
# TIER 0: REACTIVE (Immediate responses, no deliberation)
# =============================================================================

func _process_tier0() -> void:
	"""Immediate reactive responses - runs every frame"""
	
	# Threat response based on soul map
	var threat = _get_immediate_threat()
	if threat:
		var threat_response = _calculate_threat_response(threat)
		if threat_response == "flee":
			_transition_to(State.FLEE)
			attention_target = threat
		elif threat_response == "attack":
			_transition_to(State.ATTACK)
			attention_target = threat


func _get_immediate_threat() -> Node3D:
	"""Check for immediate threats in perception"""
	for entity_id in perceived_entities:
		var data = perceived_entities[entity_id]
		if data.get("is_threat", false):
			return data.get("node")
	return null


func _calculate_threat_response(threat: Node3D) -> String:
	"""Quick response calculation using soul map thresholds"""
	var distance = global_position.distance_to(threat.global_position)
	
	# Soul map factors
	var harm_avoidance = soul_map.get("harm_avoidance", 0.5)
	var risk_tolerance = soul_map.get("risk_tolerance", 0.5)
	var impulsivity = soul_map.get("impulsivity", 0.5)
	
	# Very close + high harm avoidance = flee
	if distance < 3.0 and harm_avoidance > 0.7:
		return "flee"
	
	# High risk tolerance + impulsive = attack
	if risk_tolerance > 0.6 and impulsivity > 0.5:
		return "attack"
	
	# Default: flee if scared, otherwise hold
	if harm_avoidance > risk_tolerance:
		return "flee"
	
	return "hold"


# =============================================================================
# TIER 1: HEURISTIC (Fast decisions using soul map)
# =============================================================================

func _process_tier1(delta: float) -> void:
	"""Heuristic decisions - runs when not waiting on Python"""
	
	# Cooldown check
	if Time.get_ticks_msec() / 1000.0 - last_decision_time < decision_cooldown:
		return
	
	match current_state:
		State.IDLE:
			_tier1_idle_behavior()
		State.PATROL:
			_tier1_patrol_behavior()
		State.APPROACH:
			_tier1_approach_behavior()
		State.FLEE:
			_tier1_flee_behavior()
		State.INVESTIGATE:
			_tier1_investigate_behavior()


func _tier1_idle_behavior() -> void:
	"""What to do when idle"""
	var novelty_seeking = soul_map.get("novelty_seeking", 0.5)
	var affiliation_need = soul_map.get("affiliation_need", 0.5)
	
	# High novelty seeking -> patrol/explore
	if novelty_seeking > 0.6 and randf() < 0.01:
		_transition_to(State.PATROL)
		return
	
	# See player and high affiliation -> approach
	if perceived_entities.has("player"):
		var player_data = perceived_entities["player"]
		var trust = SoulMapManager.get_belief_about(npc_id, "player", "trust")
		if trust == null:
			trust = soul_map.get("trust_propensity", 0.5)
		
		if affiliation_need > 0.5 and trust > 0.4:
			_transition_to(State.APPROACH)
			attention_target = player_data.get("node")


func _tier1_patrol_behavior() -> void:
	"""Patrol behavior"""
	if nav_agent and nav_agent.is_navigation_finished():
		# Pick new random destination
		var random_offset = Vector3(
			randf_range(-10, 10),
			0,
			randf_range(-10, 10)
		)
		current_destination = global_position + random_offset
		nav_agent.target_position = current_destination
	
	# Might get bored and stop
	if randf() < 0.005:
		_transition_to(State.IDLE)


func _tier1_approach_behavior() -> void:
	"""Approaching a target"""
	if not attention_target:
		_transition_to(State.IDLE)
		return
	
	var distance = global_position.distance_to(attention_target.global_position)
	
	# Close enough to interact
	if distance < 2.0:
		if attention_target.is_in_group("player"):
			# Trigger strategic decision about what to say/do
			_request_strategic_decision("approach_player", {
				"distance": distance,
				"relationship": SoulMapManager.get_relationship(npc_id, "player")
			})
		_transition_to(State.IDLE)
		return
	
	# Move toward target
	if nav_agent:
		nav_agent.target_position = attention_target.global_position


func _tier1_flee_behavior() -> void:
	"""Fleeing from threat"""
	if not attention_target:
		_transition_to(State.IDLE)
		return
	
	var distance = global_position.distance_to(attention_target.global_position)
	
	# Far enough, stop fleeing
	if distance > perception_radius * 1.5:
		_transition_to(State.IDLE)
		attention_target = null
		return
	
	# Run away
	if nav_agent:
		var flee_direction = (global_position - attention_target.global_position).normalized()
		var flee_target = global_position + flee_direction * 10.0
		nav_agent.target_position = flee_target


func _tier1_investigate_behavior() -> void:
	"""Investigating something interesting"""
	if not attention_target:
		_transition_to(State.IDLE)
		return
	
	var distance = global_position.distance_to(attention_target.global_position)
	
	if distance < 2.0:
		# Close enough - might trigger deeper analysis
		var curiosity = soul_map.get("novelty_seeking", 0.5)
		if curiosity > 0.7:
			# High curiosity NPCs query Python for deeper understanding
			_request_deep_tom_analysis(attention_target)
		_transition_to(State.IDLE)
		return
	
	if nav_agent:
		nav_agent.target_position = attention_target.global_position


# =============================================================================
# TIER 2: STRATEGIC (Query Python for considered decisions)
# =============================================================================

func _request_strategic_decision(situation_type: String, context: Dictionary) -> void:
	"""Request a strategic decision from Python"""
	if not GameBridge.is_connected_to_backend():
		print("[NPC %s] Cannot query Python - not connected" % npc_id)
		return
	
	pending_strategic_query = true
	last_decision_time = Time.get_ticks_msec() / 1000.0
	
	GameBridge.query_strategic_decision(npc_id, {
		"type": situation_type,
		"context": context,
		"current_state": State.keys()[current_state],
		"perceived_entities": _serialize_perceived_entities()
	}, _on_strategic_response)


func _on_strategic_response(response: Dictionary) -> void:
	"""Handle strategic decision from Python"""
	pending_strategic_query = false
	
	var decision = response.get("decision", {})
	var action = decision.get("action", "none")
	var target = decision.get("target", "")
	var parameters = decision.get("parameters", {})
	
	print("[NPC %s] Strategic decision: %s" % [npc_id, action])
	decision_made.emit(decision)
	
	match action:
		"speak":
			_initiate_dialogue(target, parameters.get("opening", ""))
		"attack":
			attention_target = _find_entity_node(target)
			_transition_to(State.ATTACK)
		"flee":
			attention_target = _find_entity_node(target)
			_transition_to(State.FLEE)
		"cooperate":
			SoulMapManager.update_belief_about(npc_id, target, "trust", 0.1)
		"betray":
			SoulMapManager.update_belief_about(npc_id, target, "trust", -0.3)


# =============================================================================
# TIER 3: DEEP TOM (Recursive reasoning about beliefs)
# =============================================================================

func _request_deep_tom_analysis(target: Node3D) -> void:
	"""Request deep Theory of Mind analysis from Python"""
	if not GameBridge.is_connected_to_backend():
		return
	
	var target_id = _get_entity_id(target)
	var tom_depth = int(soul_map.get("theory_of_mind_depth", 0.5) * 4) + 1  # 1-5 depth
	
	GameBridge.query_deep_tom(npc_id, target_id, tom_depth, "belief_state", _on_deep_tom_response)


func _on_deep_tom_response(response: Dictionary) -> void:
	"""Handle deep ToM analysis from Python"""
	var target_id = response.get("target_id", "")
	var belief_state = response.get("belief_state", {})
	var inferred_intentions = response.get("intentions", [])
	var deception_probability = response.get("deception_probability", 0.0)
	
	print("[NPC %s] Deep ToM on %s: deception=%.2f" % [npc_id, target_id, deception_probability])
	
	# Update beliefs based on analysis
	if deception_probability > 0.6:
		SoulMapManager.update_belief_about(npc_id, target_id, "trust", -0.2)
		SoulMapManager.update_belief_about(npc_id, target_id, "is_deceptive", true)
		EventBus.betrayal_detected.emit(npc_id, target_id)
	
	# Store inferred beliefs for later use
	for intention in inferred_intentions:
		SoulMapManager.set_belief_about(npc_id, target_id, "intention_%s" % intention.get("type"), intention.get("confidence"))


# =============================================================================
# PERCEPTION
# =============================================================================

func _process_perception() -> void:
	"""Update perceived entity data"""
	for entity_id in perceived_entities.keys():
		var data = perceived_entities[entity_id]
		var node = data.get("node") as Node3D
		
		if not is_instance_valid(node):
			perceived_entities.erase(entity_id)
			EventBus.entity_lost.emit(npc_id, entity_id)
			continue
		
		# Update position and distance
		data["position"] = node.global_position
		data["distance"] = global_position.distance_to(node.global_position)
		
		# Check if still in range
		if data["distance"] > perception_radius:
			perceived_entities.erase(entity_id)
			EventBus.entity_lost.emit(npc_id, entity_id)


func _on_body_entered_perception(body: Node3D) -> void:
	"""Something entered perception range"""
	var entity_id = _get_entity_id(body)
	
	var entity_data = {
		"node": body,
		"entity_id": entity_id,
		"position": body.global_position,
		"distance": global_position.distance_to(body.global_position),
		"is_player": body.is_in_group("player"),
		"is_npc": body.is_in_group("npcs"),
		"is_threat": _assess_threat(body),
		"first_seen": Time.get_ticks_msec()
	}
	
	perceived_entities[entity_id] = entity_data
	EventBus.entity_perceived.emit(npc_id, entity_id, entity_data)
	
	# Report to Python for processing
	if GameBridge.is_connected_to_backend():
		GameBridge.report_perception(npc_id, [entity_data])


func _on_body_exited_perception(body: Node3D) -> void:
	"""Something left perception range"""
	var entity_id = _get_entity_id(body)
	if perceived_entities.has(entity_id):
		perceived_entities.erase(entity_id)
		EventBus.entity_lost.emit(npc_id, entity_id)


func _assess_threat(body: Node3D) -> bool:
	"""Quick threat assessment"""
	# For now, simple logic - expand based on game design
	if body.is_in_group("enemies"):
		return true
	if body.is_in_group("player"):
		var trust = SoulMapManager.get_belief_about(npc_id, "player", "trust")
		if trust != null and trust < 0.2:
			return true
	return false


func _update_perception_radius() -> void:
	"""Adjust perception area based on soul map"""
	if perception_area and perception_area.has_node("CollisionShape3D"):
		var shape = perception_area.get_node("CollisionShape3D")
		if shape.shape is SphereShape3D:
			shape.shape.radius = perception_radius


# =============================================================================
# MOVEMENT
# =============================================================================

func _process_movement(delta: float) -> void:
	"""Handle movement based on current state"""
	if not nav_agent:
		return
	
	if nav_agent.is_navigation_finished():
		return
	
	var current_speed = move_speed
	if current_state == State.FLEE:
		current_speed = run_speed
	
	var next_pos = nav_agent.get_next_path_position()
	var direction = (next_pos - global_position).normalized()
	
	velocity = direction * current_speed
	move_and_slide()
	
	# Face movement direction
	if velocity.length() > 0.1:
		look_at(global_position + velocity, Vector3.UP)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

func _transition_to(new_state: State) -> void:
	"""Transition to a new state"""
	if new_state == current_state:
		return
	
	previous_state = current_state
	current_state = new_state
	
	var old_name = State.keys()[previous_state]
	var new_name = State.keys()[new_state]
	
	state_changed.emit(old_name, new_name)
	EventBus.npc_state_changed.emit(npc_id, old_name, new_name)
	
	# Visual feedback
	_update_visual()


func _update_visual() -> void:
	"""Update visual representation based on state/archetype"""
	if not mesh:
		return
	
	var material = mesh.get_active_material(0)
	if not material:
		return
	
	# Color by state
	var color: Color
	match current_state:
		State.IDLE:
			color = Color.WHITE
		State.PATROL:
			color = Color.CYAN
		State.APPROACH:
			color = Color.GREEN
		State.FLEE:
			color = Color.YELLOW
		State.ATTACK:
			color = Color.RED
		State.CONVERSE:
			color = Color.PURPLE
		State.INVESTIGATE:
			color = Color.ORANGE
	
	if material is StandardMaterial3D:
		material.albedo_color = color


# =============================================================================
# DIALOGUE
# =============================================================================

func _initiate_dialogue(target_id: String, opening: String) -> void:
	"""Start a dialogue with a target"""
	_transition_to(State.CONVERSE)
	EventBus.dialogue_started.emit(npc_id)
	
	if opening != "":
		EventBus.show_dialogue_ui.emit(npc_id, opening, [])


# =============================================================================
# UTILITIES
# =============================================================================

func _get_entity_id(node: Node3D) -> String:
	"""Get consistent ID for an entity"""
	if node.is_in_group("player"):
		return "player"
	if node.has_method("get_npc_id"):
		return node.get_npc_id()
	if node is NPCController:
		return node.npc_id
	return "entity_%d" % node.get_instance_id()


func _find_entity_node(entity_id: String) -> Node3D:
	"""Find a node by entity ID"""
	if entity_id == "player":
		return get_tree().get_first_node_in_group("player")
	
	for npc in get_tree().get_nodes_in_group("npcs"):
		if npc is NPCController and npc.npc_id == entity_id:
			return npc
	
	return null


func _serialize_perceived_entities() -> Array:
	"""Serialize perceived entities for Python"""
	var result: Array = []
	for entity_id in perceived_entities:
		var data = perceived_entities[entity_id]
		result.append({
			"entity_id": entity_id,
			"position": {"x": data["position"].x, "y": data["position"].y, "z": data["position"].z},
			"distance": data["distance"],
			"is_player": data["is_player"],
			"is_npc": data["is_npc"],
			"is_threat": data["is_threat"]
		})
	return result


func get_npc_id() -> String:
	return npc_id


func _on_soul_map_updated(updated_npc_id: String, new_soul_map: Dictionary) -> void:
	if updated_npc_id == npc_id:
		soul_map = new_soul_map


func _on_bridge_connected() -> void:
	# Sync soul map to Python on connect
	pass
