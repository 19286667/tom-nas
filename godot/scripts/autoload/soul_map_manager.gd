extends Node
## SoulMapManager: Local storage and access for NPC psychological states
## The Soul Map is the contract between Godot and Python - 65 dimensions of psychology

signal soul_map_updated(npc_id: String, soul_map: Dictionary)
signal beliefs_updated(npc_id: String, beliefs: Dictionary)

# Storage
var _soul_maps: Dictionary = {}  # npc_id -> SoulMap
var _beliefs: Dictionary = {}    # npc_id -> belief state
var _relationships: Dictionary = {}  # npc_id -> {other_id -> relationship}


# =============================================================================
# SOUL MAP STRUCTURE
# The 65 dimensions organized by domain
# =============================================================================

const SOUL_MAP_TEMPLATE := {
	# --- Cognitive Architecture (15 dims) ---
	"working_memory_capacity": 0.5,
	"attention_stability": 0.5,
	"cognitive_flexibility": 0.5,
	"processing_speed": 0.5,
	"pattern_recognition": 0.5,
	"abstraction_capacity": 0.5,
	"metacognitive_accuracy": 0.5,
	"learning_rate": 0.5,
	"inference_depth": 0.5,
	"uncertainty_tolerance": 0.5,
	"confirmation_bias": 0.5,
	"anchoring_strength": 0.5,
	"availability_bias": 0.5,
	"theory_of_mind_depth": 0.5,
	"recursive_modeling_limit": 0.5,
	
	# --- Emotional Dynamics (12 dims) ---
	"emotional_intensity": 0.5,
	"emotional_stability": 0.5,
	"positive_affectivity": 0.5,
	"negative_affectivity": 0.5,
	"emotion_regulation": 0.5,
	"empathic_accuracy": 0.5,
	"emotional_contagion": 0.5,
	"mood_persistence": 0.5,
	"stress_reactivity": 0.5,
	"recovery_rate": 0.5,
	"alexithymia": 0.5,
	"emotional_granularity": 0.5,
	
	# --- Motivational Landscape (10 dims) ---
	"approach_motivation": 0.5,
	"avoidance_motivation": 0.5,
	"intrinsic_motivation": 0.5,
	"extrinsic_motivation": 0.5,
	"achievement_drive": 0.5,
	"affiliation_need": 0.5,
	"power_need": 0.5,
	"autonomy_need": 0.5,
	"competence_need": 0.5,
	"meaning_seeking": 0.5,
	
	# --- Social Cognition (12 dims) ---
	"trust_propensity": 0.5,
	"cooperation_tendency": 0.5,
	"competition_tendency": 0.5,
	"social_dominance": 0.5,
	"social_vigilance": 0.5,
	"reputation_concern": 0.5,
	"reciprocity_tracking": 0.5,
	"coalition_sensitivity": 0.5,
	"ingroup_favoritism": 0.5,
	"fairness_sensitivity": 0.5,
	"deception_propensity": 0.5,
	"deception_detection": 0.5,
	
	# --- Identity & Self (8 dims) ---
	"self_esteem": 0.5,
	"self_efficacy": 0.5,
	"identity_clarity": 0.5,
	"self_consistency": 0.5,
	"narrative_coherence": 0.5,
	"authenticity": 0.5,
	"self_monitoring": 0.5,
	"impression_management": 0.5,
	
	# --- Behavioral Tendencies (8 dims) ---
	"impulsivity": 0.5,
	"risk_tolerance": 0.5,
	"novelty_seeking": 0.5,
	"persistence": 0.5,
	"harm_avoidance": 0.5,
	"reward_dependence": 0.5,
	"self_directedness": 0.5,
	"cooperativeness": 0.5
}

# Quick access to dimension names by category
const COGNITIVE_DIMS := [
	"working_memory_capacity", "attention_stability", "cognitive_flexibility",
	"processing_speed", "pattern_recognition", "abstraction_capacity",
	"metacognitive_accuracy", "learning_rate", "inference_depth",
	"uncertainty_tolerance", "confirmation_bias", "anchoring_strength",
	"availability_bias", "theory_of_mind_depth", "recursive_modeling_limit"
]

const EMOTIONAL_DIMS := [
	"emotional_intensity", "emotional_stability", "positive_affectivity",
	"negative_affectivity", "emotion_regulation", "empathic_accuracy",
	"emotional_contagion", "mood_persistence", "stress_reactivity",
	"recovery_rate", "alexithymia", "emotional_granularity"
]

const MOTIVATIONAL_DIMS := [
	"approach_motivation", "avoidance_motivation", "intrinsic_motivation",
	"extrinsic_motivation", "achievement_drive", "affiliation_need",
	"power_need", "autonomy_need", "competence_need", "meaning_seeking"
]

const SOCIAL_DIMS := [
	"trust_propensity", "cooperation_tendency", "competition_tendency",
	"social_dominance", "social_vigilance", "reputation_concern",
	"reciprocity_tracking", "coalition_sensitivity", "ingroup_favoritism",
	"fairness_sensitivity", "deception_propensity", "deception_detection"
]

const IDENTITY_DIMS := [
	"self_esteem", "self_efficacy", "identity_clarity", "self_consistency",
	"narrative_coherence", "authenticity", "self_monitoring", "impression_management"
]

const BEHAVIORAL_DIMS := [
	"impulsivity", "risk_tolerance", "novelty_seeking", "persistence",
	"harm_avoidance", "reward_dependence", "self_directedness", "cooperativeness"
]


# =============================================================================
# SOUL MAP MANAGEMENT
# =============================================================================

func register_npc(npc_id: String, initial_soul_map: Dictionary = {}) -> void:
	"""Register an NPC with their soul map"""
	var soul_map = SOUL_MAP_TEMPLATE.duplicate(true)
	
	# Override with provided values
	for key in initial_soul_map:
		if soul_map.has(key):
			soul_map[key] = clampf(initial_soul_map[key], 0.0, 1.0)
	
	_soul_maps[npc_id] = soul_map
	_beliefs[npc_id] = {}
	_relationships[npc_id] = {}
	
	print("[SoulMapManager] Registered NPC: %s" % npc_id)


func unregister_npc(npc_id: String) -> void:
	"""Remove an NPC from management"""
	_soul_maps.erase(npc_id)
	_beliefs.erase(npc_id)
	_relationships.erase(npc_id)


func get_soul_map(npc_id: String) -> Dictionary:
	"""Get full soul map for an NPC"""
	return _soul_maps.get(npc_id, SOUL_MAP_TEMPLATE.duplicate(true))


func get_dimension(npc_id: String, dimension: String) -> float:
	"""Get a single soul map dimension"""
	var soul_map = get_soul_map(npc_id)
	return soul_map.get(dimension, 0.5)


func set_dimension(npc_id: String, dimension: String, value: float) -> void:
	"""Set a single soul map dimension locally"""
	if not _soul_maps.has(npc_id):
		push_warning("[SoulMapManager] NPC not registered: %s" % npc_id)
		return
	
	_soul_maps[npc_id][dimension] = clampf(value, 0.0, 1.0)
	soul_map_updated.emit(npc_id, _soul_maps[npc_id])


func update_soul_map(npc_id: String, updates: Dictionary) -> void:
	"""Update multiple dimensions (typically from Python)"""
	if not _soul_maps.has(npc_id):
		register_npc(npc_id, updates)
		return
	
	for key in updates:
		if _soul_maps[npc_id].has(key):
			_soul_maps[npc_id][key] = clampf(updates[key], 0.0, 1.0)
	
	soul_map_updated.emit(npc_id, _soul_maps[npc_id])


func get_soul_map_vector(npc_id: String) -> PackedFloat32Array:
	"""Get soul map as a flat vector (for neural network input)"""
	var soul_map = get_soul_map(npc_id)
	var vector = PackedFloat32Array()
	
	# Consistent ordering across all dimensions
	for dim in COGNITIVE_DIMS:
		vector.append(soul_map.get(dim, 0.5))
	for dim in EMOTIONAL_DIMS:
		vector.append(soul_map.get(dim, 0.5))
	for dim in MOTIVATIONAL_DIMS:
		vector.append(soul_map.get(dim, 0.5))
	for dim in SOCIAL_DIMS:
		vector.append(soul_map.get(dim, 0.5))
	for dim in IDENTITY_DIMS:
		vector.append(soul_map.get(dim, 0.5))
	for dim in BEHAVIORAL_DIMS:
		vector.append(soul_map.get(dim, 0.5))
	
	return vector


# =============================================================================
# BELIEF MANAGEMENT (First-order, local)
# =============================================================================

func get_beliefs(npc_id: String) -> Dictionary:
	"""Get all first-order beliefs for an NPC"""
	return _beliefs.get(npc_id, {})


func get_belief_about(npc_id: String, target_id: String, belief_type: String) -> Variant:
	"""Get a specific belief about a target
	
	Example: get_belief_about("npc_1", "player", "trust") -> 0.7
	"""
	var npc_beliefs = _beliefs.get(npc_id, {})
	var target_beliefs = npc_beliefs.get(target_id, {})
	return target_beliefs.get(belief_type, null)


func set_belief_about(npc_id: String, target_id: String, belief_type: String, value: Variant) -> void:
	"""Set a first-order belief about a target"""
	if not _beliefs.has(npc_id):
		_beliefs[npc_id] = {}
	if not _beliefs[npc_id].has(target_id):
		_beliefs[npc_id][target_id] = {}
	
	_beliefs[npc_id][target_id][belief_type] = value
	beliefs_updated.emit(npc_id, _beliefs[npc_id])


func update_belief_about(npc_id: String, target_id: String, belief_type: String, delta: float) -> void:
	"""Incrementally update a belief (with clamping for 0-1 values)"""
	var current = get_belief_about(npc_id, target_id, belief_type)
	if current == null:
		current = 0.5
	
	var new_value = clampf(current + delta, 0.0, 1.0)
	set_belief_about(npc_id, target_id, belief_type, new_value)


# =============================================================================
# RELATIONSHIP MANAGEMENT
# =============================================================================

func get_relationship(npc_id: String, other_id: String) -> Dictionary:
	"""Get relationship state between two agents"""
	var npc_rels = _relationships.get(npc_id, {})
	return npc_rels.get(other_id, {
		"trust": 0.5,
		"affect": 0.0,  # -1 to 1
		"familiarity": 0.0,
		"relationship_type": "stranger"
	})


func update_relationship(npc_id: String, other_id: String, updates: Dictionary) -> void:
	"""Update relationship values"""
	if not _relationships.has(npc_id):
		_relationships[npc_id] = {}
	if not _relationships[npc_id].has(other_id):
		_relationships[npc_id][other_id] = {
			"trust": 0.5,
			"affect": 0.0,
			"familiarity": 0.0,
			"relationship_type": "stranger"
		}
	
	for key in updates:
		_relationships[npc_id][other_id][key] = updates[key]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

func generate_random_soul_map(archetype: String = "neutral") -> Dictionary:
	"""Generate a random soul map, optionally biased by archetype"""
	var soul_map = SOUL_MAP_TEMPLATE.duplicate(true)
	
	# Base randomization
	for key in soul_map:
		soul_map[key] = randf_range(0.2, 0.8)
	
	# Archetype biases
	match archetype:
		"paranoid":
			soul_map["trust_propensity"] = randf_range(0.1, 0.3)
			soul_map["social_vigilance"] = randf_range(0.7, 0.95)
			soul_map["deception_detection"] = randf_range(0.6, 0.9)
			soul_map["harm_avoidance"] = randf_range(0.7, 0.9)
		"naive":
			soul_map["trust_propensity"] = randf_range(0.7, 0.95)
			soul_map["social_vigilance"] = randf_range(0.1, 0.3)
			soul_map["deception_detection"] = randf_range(0.1, 0.4)
		"manipulative":
			soul_map["deception_propensity"] = randf_range(0.7, 0.95)
			soul_map["theory_of_mind_depth"] = randf_range(0.7, 0.9)
			soul_map["impression_management"] = randf_range(0.7, 0.9)
			soul_map["empathic_accuracy"] = randf_range(0.6, 0.85)
		"impulsive":
			soul_map["impulsivity"] = randf_range(0.7, 0.95)
			soul_map["emotion_regulation"] = randf_range(0.1, 0.4)
			soul_map["risk_tolerance"] = randf_range(0.6, 0.9)
		"stoic":
			soul_map["emotional_intensity"] = randf_range(0.1, 0.3)
			soul_map["emotion_regulation"] = randf_range(0.8, 0.95)
			soul_map["stress_reactivity"] = randf_range(0.1, 0.3)
		"leader":
			soul_map["social_dominance"] = randf_range(0.7, 0.9)
			soul_map["self_efficacy"] = randf_range(0.7, 0.9)
			soul_map["power_need"] = randf_range(0.6, 0.85)
			soul_map["theory_of_mind_depth"] = randf_range(0.6, 0.85)
	
	return soul_map


func get_all_npc_ids() -> Array:
	"""Get list of all registered NPC IDs"""
	return _soul_maps.keys()


func get_dominant_traits(npc_id: String, top_n: int = 5) -> Array:
	"""Get the most extreme traits for an NPC (furthest from 0.5)"""
	var soul_map = get_soul_map(npc_id)
	var deviations: Array = []
	
	for key in soul_map:
		var deviation = absf(soul_map[key] - 0.5)
		deviations.append({"trait": key, "value": soul_map[key], "deviation": deviation})
	
	deviations.sort_custom(func(a, b): return a["deviation"] > b["deviation"])
	return deviations.slice(0, top_n)
