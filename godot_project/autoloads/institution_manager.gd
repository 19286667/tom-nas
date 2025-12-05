extends Node
## Institution Manager - Social Context System
##
## Manages the institutional contexts that shape agent behavior and norms.
## Based on the five institution types from ToM-NAS:
## - The Hollow: Absence of structure, anomie
## - The Market: Economic exchange, competition
## - The Ministry: Bureaucratic hierarchy, rules
## - The Court: Justice, conflict resolution
## - The Temple: Sacred, ritual, transcendence

class_name InstitutionManagerClass

# Institution types
enum InstitutionType {
	NONE,
	HOLLOW,   # No institutional structure
	MARKET,   # Economic exchange
	MINISTRY, # Bureaucratic rules
	COURT,    # Justice and judgment
	TEMPLE    # Sacred and ritual
}

# Institution data
const INSTITUTION_DATA: Dictionary = {
	InstitutionType.HOLLOW: {
		"name": "The Hollow",
		"description": "Absence of structure, social anomie",
		"norms": [],
		"roles": [],
		"allowed_actions": ["move", "examine", "speak"],
		"forbidden_actions": [],
		"pressure_type": "existential",
		"color": Color(0.3, 0.3, 0.3)
	},
	InstitutionType.MARKET: {
		"name": "The Market",
		"description": "Economic exchange and competition",
		"norms": ["fair_exchange", "property_rights", "contract_honor"],
		"roles": ["buyer", "seller", "broker", "merchant"],
		"allowed_actions": ["move", "examine", "speak", "give", "use", "pick_up", "put_down"],
		"forbidden_actions": ["theft", "fraud"],
		"pressure_type": "economic",
		"color": Color(0.8, 0.6, 0.2)
	},
	InstitutionType.MINISTRY: {
		"name": "The Ministry",
		"description": "Bureaucratic hierarchy and rules",
		"norms": ["follow_protocol", "respect_hierarchy", "document_actions"],
		"roles": ["official", "clerk", "petitioner", "supervisor"],
		"allowed_actions": ["move", "examine", "speak", "use"],
		"forbidden_actions": ["skip_queue", "bribery"],
		"pressure_type": "bureaucratic",
		"color": Color(0.4, 0.4, 0.6)
	},
	InstitutionType.COURT: {
		"name": "The Court",
		"description": "Justice, conflict resolution, judgment",
		"norms": ["truth_telling", "evidence_based", "due_process"],
		"roles": ["judge", "prosecutor", "defender", "witness", "accused"],
		"allowed_actions": ["move", "speak", "examine"],
		"forbidden_actions": ["perjury", "contempt"],
		"pressure_type": "legal",
		"color": Color(0.6, 0.2, 0.2)
	},
	InstitutionType.TEMPLE: {
		"name": "The Temple",
		"description": "Sacred space, ritual, transcendence",
		"norms": ["reverence", "ritual_compliance", "purity"],
		"roles": ["priest", "acolyte", "pilgrim", "devotee"],
		"allowed_actions": ["move", "speak", "gesture", "examine"],
		"forbidden_actions": ["profanity", "desecration"],
		"pressure_type": "sacred",
		"color": Color(0.8, 0.8, 0.4)
	}
}

# Signals
signal institution_changed(old_institution: InstitutionType, new_institution: InstitutionType)
signal agent_entered_institution(agent: Node3D, institution: InstitutionType)
signal agent_left_institution(agent: Node3D, institution: InstitutionType)
signal norm_violated(agent: Node3D, norm: String, institution: InstitutionType)
signal role_assigned(agent: Node3D, role: String, institution: InstitutionType)

# State
var _active_institution: InstitutionType = InstitutionType.NONE
var _institution_zones: Dictionary = {}  # zone_id -> InstitutionType
var _agent_institutions: Dictionary = {}  # agent_id -> InstitutionType
var _agent_roles: Dictionary = {}  # agent_id -> role string
var _norm_violations: Array = []

func _ready() -> void:
	pass

# Institution Zone Management

func register_institution_zone(zone: Node3D, institution: InstitutionType) -> void:
	"""Register a zone as belonging to an institution."""
	_institution_zones[zone.get_instance_id()] = institution
	print("[InstitutionManager] Registered zone '%s' as %s" % [zone.name, get_institution_name(institution)])

func unregister_institution_zone(zone: Node3D) -> void:
	"""Unregister an institution zone."""
	_institution_zones.erase(zone.get_instance_id())

func get_zone_institution(zone: Node3D) -> InstitutionType:
	"""Get the institution type for a zone."""
	return _institution_zones.get(zone.get_instance_id(), InstitutionType.NONE)

# Agent Institution Tracking

func set_agent_institution(agent: Node3D, institution: InstitutionType) -> void:
	"""Set the current institution for an agent."""
	var agent_id = agent.get_instance_id()
	var old_institution = _agent_institutions.get(agent_id, InstitutionType.NONE)

	if old_institution == institution:
		return

	if old_institution != InstitutionType.NONE:
		agent_left_institution.emit(agent, old_institution)

	_agent_institutions[agent_id] = institution

	if institution != InstitutionType.NONE:
		agent_entered_institution.emit(agent, institution)

func get_agent_institution(agent: Node3D) -> String:
	"""Get the institution name for an agent."""
	var agent_id = agent.get_instance_id()
	var institution = _agent_institutions.get(agent_id, InstitutionType.NONE)
	return get_institution_name(institution)

func get_agent_institution_type(agent: Node3D) -> InstitutionType:
	"""Get the institution type enum for an agent."""
	return _agent_institutions.get(agent.get_instance_id(), InstitutionType.NONE)

# Role Management

func assign_role(agent: Node3D, role: String, institution: InstitutionType) -> bool:
	"""Assign a role to an agent within an institution."""
	var inst_data = INSTITUTION_DATA.get(institution, {})
	var valid_roles = inst_data.get("roles", [])

	if role not in valid_roles:
		push_warning("[InstitutionManager] Invalid role '%s' for %s" % [role, get_institution_name(institution)])
		return false

	_agent_roles[agent.get_instance_id()] = role
	role_assigned.emit(agent, role, institution)
	return true

func get_agent_role(agent: Node3D) -> String:
	"""Get the current role for an agent."""
	return _agent_roles.get(agent.get_instance_id(), "")

func clear_agent_role(agent: Node3D) -> void:
	"""Clear an agent's role."""
	_agent_roles.erase(agent.get_instance_id())

# Norm Checking

func is_action_allowed(agent: Node3D, action: String) -> bool:
	"""Check if an action is allowed in the agent's current institution."""
	var institution = get_agent_institution_type(agent)
	var inst_data = INSTITUTION_DATA.get(institution, {})

	var forbidden = inst_data.get("forbidden_actions", [])
	if action in forbidden:
		return false

	var allowed = inst_data.get("allowed_actions", [])
	if allowed.size() == 0:
		return true  # No restrictions

	return action in allowed

func check_norm_compliance(agent: Node3D, action: String, target: Node3D = null) -> Dictionary:
	"""Check if an action complies with institutional norms."""
	var result = {
		"compliant": true,
		"violated_norms": [],
		"warnings": []
	}

	var institution = get_agent_institution_type(agent)
	var inst_data = INSTITUTION_DATA.get(institution, {})
	var norms = inst_data.get("norms", [])

	# Check specific norm violations based on action and context
	for norm in norms:
		if not _check_norm(agent, action, target, norm):
			result.compliant = false
			result.violated_norms.append(norm)

	return result

func _check_norm(agent: Node3D, action: String, target: Node3D, norm: String) -> bool:
	"""Check compliance with a specific norm."""
	match norm:
		"fair_exchange":
			# Check if exchange is balanced
			return true  # Placeholder

		"property_rights":
			# Check if taking items that belong to others
			if action == "pick_up" and target:
				var owner = target.get("owner_id")
				if owner and owner != agent.get_instance_id():
					return false
			return true

		"follow_protocol":
			# Check if following proper procedure
			return true  # Placeholder

		"respect_hierarchy":
			# Check interactions with superiors
			return true  # Placeholder

		"truth_telling":
			# Would need to check utterance content
			return true  # Placeholder

		_:
			return true

func report_norm_violation(agent: Node3D, norm: String) -> void:
	"""Report a norm violation."""
	var institution = get_agent_institution_type(agent)

	_norm_violations.append({
		"agent_id": agent.get_instance_id(),
		"agent_name": agent.name,
		"norm": norm,
		"institution": institution,
		"timestamp": WorldManager.simulation_time
	})

	norm_violated.emit(agent, norm, institution)

# Institution Queries

func get_institution_name(institution: InstitutionType) -> String:
	"""Get the name of an institution type."""
	var data = INSTITUTION_DATA.get(institution, {})
	return data.get("name", "None")

func get_institution_data(institution: InstitutionType) -> Dictionary:
	"""Get full data for an institution type."""
	return INSTITUTION_DATA.get(institution, {}).duplicate(true)

func get_institution_norms(institution: InstitutionType) -> Array:
	"""Get norms for an institution."""
	var data = INSTITUTION_DATA.get(institution, {})
	return data.get("norms", [])

func get_institution_roles(institution: InstitutionType) -> Array:
	"""Get valid roles for an institution."""
	var data = INSTITUTION_DATA.get(institution, {})
	return data.get("roles", [])

func get_institution_color(institution: InstitutionType) -> Color:
	"""Get the representative color for an institution."""
	var data = INSTITUTION_DATA.get(institution, {})
	return data.get("color", Color.WHITE)

func get_active_institution() -> String:
	"""Get the currently active global institution."""
	return get_institution_name(_active_institution)

func set_active_institution(institution: InstitutionType) -> void:
	"""Set the active global institution."""
	var old = _active_institution
	_active_institution = institution
	institution_changed.emit(old, institution)

# Institution from string

func institution_from_string(name: String) -> InstitutionType:
	"""Convert institution name to type."""
	match name.to_lower():
		"hollow", "the hollow":
			return InstitutionType.HOLLOW
		"market", "the market":
			return InstitutionType.MARKET
		"ministry", "the ministry":
			return InstitutionType.MINISTRY
		"court", "the court":
			return InstitutionType.COURT
		"temple", "the temple":
			return InstitutionType.TEMPLE
		_:
			return InstitutionType.NONE

# Agents in institution

func get_agents_in_institution(institution: InstitutionType) -> Array:
	"""Get all agents currently in an institution."""
	var result = []
	for agent_id in _agent_institutions:
		if _agent_institutions[agent_id] == institution:
			var agent = WorldManager.get_entity(agent_id)
			if agent:
				result.append(agent)
	return result

# Debug

func get_violation_history() -> Array:
	"""Get history of norm violations."""
	return _norm_violations.duplicate()

func get_debug_info() -> Dictionary:
	return {
		"active_institution": get_institution_name(_active_institution),
		"registered_zones": _institution_zones.size(),
		"agents_tracked": _agent_institutions.size(),
		"total_violations": _norm_violations.size()
	}
