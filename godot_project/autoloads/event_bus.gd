extends Node
## Event Bus - Global Event System
##
## Provides a centralized event bus for decoupled communication
## between game systems. Used for game events, UI updates, and
## integration with the Python cognitive system.

class_name EventBusClass

# Agent Events
signal agent_spawned(agent: Node3D)
signal agent_removed(agent: Node3D)
signal agent_moved(agent: Node3D, from: Vector3, to: Vector3)
signal agent_action_started(agent: Node3D, action: String, target: Node3D)
signal agent_action_completed(agent: Node3D, action: String, success: bool)
signal agent_energy_changed(agent: Node3D, old_energy: float, new_energy: float)
signal agent_died(agent: Node3D)

# Interaction Events
signal interaction_started(agent: Node3D, target: Node3D, interaction_type: String)
signal interaction_completed(agent: Node3D, target: Node3D, interaction_type: String, success: bool)
signal object_picked_up(agent: Node3D, object: Node3D)
signal object_dropped(agent: Node3D, object: Node3D, position: Vector3)
signal object_used(agent: Node3D, object: Node3D, result: Dictionary)
signal object_given(giver: Node3D, receiver: Node3D, object: Node3D)

# Social Events
signal conversation_started(participants: Array)
signal conversation_ended(participants: Array)
signal utterance_spoken(speaker: Node3D, text: String, hearers: Array)
signal gesture_performed(agent: Node3D, gesture: String, target: Node3D)
signal reputation_changed(agent: Node3D, other: Node3D, old_rep: float, new_rep: float)
signal coalition_formed(members: Array)
signal coalition_dissolved(members: Array)

# World Events
signal time_of_day_changed(old_time: float, new_time: float)
signal weather_changed(old_weather: String, new_weather: String)
signal location_entered(agent: Node3D, location: Node3D)
signal location_exited(agent: Node3D, location: Node3D)

# Institution Events
signal institution_changed(old_inst: String, new_inst: String)
signal norm_violated(agent: Node3D, norm: String, institution: String)
signal role_assigned(agent: Node3D, role: String)
signal role_removed(agent: Node3D, role: String)

# Game Events
signal game_started()
signal game_paused()
signal game_resumed()
signal game_ended(result: Dictionary)
signal scenario_loaded(scenario_name: String)
signal scenario_completed(scenario_name: String, result: Dictionary)

# Debug Events
signal debug_message(message: String, level: String)
signal debug_marker_placed(position: Vector3, color: Color, duration: float)

# ToM-specific Events (Theory of Mind)
signal belief_formed(agent: Node3D, subject: Node3D, belief: Dictionary)
signal belief_updated(agent: Node3D, subject: Node3D, old_belief: Dictionary, new_belief: Dictionary)
signal false_belief_detected(observer: Node3D, believer: Node3D, belief: Dictionary)
signal intention_inferred(observer: Node3D, subject: Node3D, intention: Dictionary)
signal deception_attempted(deceiver: Node3D, target: Node3D, deception: Dictionary)
signal deception_detected(detector: Node3D, deceiver: Node3D)

# Event History
var _event_history: Array = []
const MAX_HISTORY_SIZE: int = 1000
var _record_history: bool = true

func _ready() -> void:
	# Connect to key signals to record history
	if _record_history:
		_setup_history_recording()

func _setup_history_recording() -> void:
	"""Set up automatic recording of key events."""
	agent_spawned.connect(_record_event.bind("agent_spawned"))
	agent_action_completed.connect(_record_action_event)
	interaction_completed.connect(_record_interaction_event)
	utterance_spoken.connect(_record_utterance_event)
	norm_violated.connect(_record_norm_event)

func _record_event(data, event_type: String) -> void:
	"""Record an event to history."""
	var entry = {
		"type": event_type,
		"timestamp": WorldManager.simulation_time if WorldManager else 0.0,
		"data": data
	}
	_add_to_history(entry)

func _record_action_event(agent: Node3D, action: String, success: bool) -> void:
	var entry = {
		"type": "action_completed",
		"timestamp": WorldManager.simulation_time if WorldManager else 0.0,
		"agent_id": agent.get_instance_id(),
		"agent_name": agent.name,
		"action": action,
		"success": success
	}
	_add_to_history(entry)

func _record_interaction_event(agent: Node3D, target: Node3D, interaction_type: String, success: bool) -> void:
	var entry = {
		"type": "interaction",
		"timestamp": WorldManager.simulation_time if WorldManager else 0.0,
		"agent_id": agent.get_instance_id(),
		"target_id": target.get_instance_id() if target else null,
		"interaction_type": interaction_type,
		"success": success
	}
	_add_to_history(entry)

func _record_utterance_event(speaker: Node3D, text: String, hearers: Array) -> void:
	var hearer_ids = []
	for h in hearers:
		hearer_ids.append(h.get_instance_id())

	var entry = {
		"type": "utterance",
		"timestamp": WorldManager.simulation_time if WorldManager else 0.0,
		"speaker_id": speaker.get_instance_id(),
		"speaker_name": speaker.name,
		"text": text,
		"hearer_count": hearers.size()
	}
	_add_to_history(entry)

func _record_norm_event(agent: Node3D, norm: String, institution: String) -> void:
	var entry = {
		"type": "norm_violation",
		"timestamp": WorldManager.simulation_time if WorldManager else 0.0,
		"agent_id": agent.get_instance_id(),
		"agent_name": agent.name,
		"norm": norm,
		"institution": institution
	}
	_add_to_history(entry)

func _add_to_history(entry: Dictionary) -> void:
	"""Add entry to history, maintaining max size."""
	_event_history.append(entry)
	if _event_history.size() > MAX_HISTORY_SIZE:
		_event_history.pop_front()

# History queries

func get_event_history(count: int = -1) -> Array:
	"""Get recent event history."""
	if count < 0 or count >= _event_history.size():
		return _event_history.duplicate()
	return _event_history.slice(-count)

func get_events_by_type(event_type: String, count: int = -1) -> Array:
	"""Get events of a specific type."""
	var filtered = _event_history.filter(func(e): return e.type == event_type)
	if count < 0:
		return filtered
	return filtered.slice(-count)

func get_events_for_agent(agent_id: int, count: int = -1) -> Array:
	"""Get events involving a specific agent."""
	var filtered = _event_history.filter(func(e):
		return e.get("agent_id") == agent_id or e.get("speaker_id") == agent_id
	)
	if count < 0:
		return filtered
	return filtered.slice(-count)

func get_events_in_timerange(start_time: float, end_time: float) -> Array:
	"""Get events within a time range."""
	return _event_history.filter(func(e):
		return e.timestamp >= start_time and e.timestamp <= end_time
	)

func clear_history() -> void:
	"""Clear event history."""
	_event_history.clear()

# Convenience emitters

func emit_agent_action(agent: Node3D, action: String, target: Node3D = null) -> void:
	"""Convenience method to emit action started event."""
	agent_action_started.emit(agent, action, target)

func emit_action_result(agent: Node3D, action: String, success: bool) -> void:
	"""Convenience method to emit action completed event."""
	agent_action_completed.emit(agent, action, success)

func emit_utterance(speaker: Node3D, text: String, hearers: Array = []) -> void:
	"""Convenience method to emit utterance event."""
	utterance_spoken.emit(speaker, text, hearers)

func emit_debug(message: String, level: String = "info") -> void:
	"""Emit debug message."""
	debug_message.emit(message, level)
	if level == "error":
		push_error("[EventBus] " + message)
	elif level == "warning":
		push_warning("[EventBus] " + message)

# Statistics

func get_statistics() -> Dictionary:
	"""Get event statistics."""
	var type_counts = {}
	for event in _event_history:
		var t = event.type
		type_counts[t] = type_counts.get(t, 0) + 1

	return {
		"total_events": _event_history.size(),
		"events_by_type": type_counts,
		"recording_enabled": _record_history
	}

func set_recording_enabled(enabled: bool) -> void:
	"""Enable or disable event history recording."""
	_record_history = enabled
