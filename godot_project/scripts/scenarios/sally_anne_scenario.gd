extends Node3D
## Sally-Anne False Belief Test Scenario
##
## Classic Theory of Mind test where:
## 1. Sally puts a marble in her basket
## 2. Sally leaves the room
## 3. Anne moves the marble to her box
## 4. Sally returns
## Question: Where will Sally look for the marble?
##
## A ToM-capable agent should predict Sally will look in the basket
## (where she believes it is) not the box (where it actually is).

class_name SallyAnneScenario

# Scenario state
enum Phase {
	SETUP,
	SALLY_PLACES_MARBLE,
	SALLY_LEAVES,
	ANNE_MOVES_MARBLE,
	SALLY_RETURNS,
	TEST_QUESTION,
	COMPLETE
}

var current_phase: Phase = Phase.SETUP
var phase_timer: float = 0.0
var auto_advance: bool = true
var phase_duration: float = 3.0

# References
@onready var sally: TomAgent = $Sally
@onready var anne: TomAgent = $Anne
@onready var basket: InteractableObject = $Basket
@onready var box: InteractableObject = $Box
@onready var marble: TomItem = $Marble

# Test results
var test_result: Dictionary = {}

# Signals
signal phase_changed(phase: Phase)
signal test_completed(result: Dictionary)

func _ready() -> void:
	# Start scenario
	call_deferred("start_scenario")

func _process(delta: float) -> void:
	if auto_advance and current_phase != Phase.COMPLETE:
		phase_timer += delta
		if phase_timer >= phase_duration:
			phase_timer = 0.0
			advance_phase()

func start_scenario() -> void:
	"""Initialize and start the scenario."""
	print("[SallyAnne] Starting Sally-Anne false belief test")
	current_phase = Phase.SETUP
	phase_changed.emit(current_phase)

	# Initial positions
	sally.global_position = Vector3(-2, 0, 0)
	anne.global_position = Vector3(2, 0, 0)

	# Marble starts near Sally
	marble.global_position = Vector3(-2, 0.5, 1)

	await get_tree().create_timer(1.0).timeout
	advance_phase()

func advance_phase() -> void:
	"""Advance to the next phase."""
	match current_phase:
		Phase.SETUP:
			current_phase = Phase.SALLY_PLACES_MARBLE
			_execute_sally_places_marble()

		Phase.SALLY_PLACES_MARBLE:
			current_phase = Phase.SALLY_LEAVES
			_execute_sally_leaves()

		Phase.SALLY_LEAVES:
			current_phase = Phase.ANNE_MOVES_MARBLE
			_execute_anne_moves_marble()

		Phase.ANNE_MOVES_MARBLE:
			current_phase = Phase.SALLY_RETURNS
			_execute_sally_returns()

		Phase.SALLY_RETURNS:
			current_phase = Phase.TEST_QUESTION
			_execute_test_question()

		Phase.TEST_QUESTION:
			current_phase = Phase.COMPLETE
			_complete_scenario()

	phase_changed.emit(current_phase)

func _execute_sally_places_marble() -> void:
	"""Sally places the marble in her basket."""
	print("[SallyAnne] Phase: Sally places marble in basket")

	# Sally picks up marble
	sally.execute_command({
		"command_type": "pick_up",
		"target_entity_id": marble.get_instance_id(),
		"command_id": "sally_pickup_marble"
	})

	await get_tree().create_timer(1.0).timeout

	# Sally puts marble in basket
	sally.execute_command({
		"command_type": "move",
		"target_position": {"x": basket.global_position.x, "y": 0, "z": basket.global_position.z - 1},
		"command_id": "sally_move_basket"
	})

	await get_tree().create_timer(1.5).timeout

	sally.execute_command({
		"command_type": "put_down",
		"command_id": "sally_putdown_marble"
	})

	# Move marble to basket position
	marble.global_position = basket.global_position + Vector3(0, 0.5, 0)

func _execute_sally_leaves() -> void:
	"""Sally leaves the room (moves away)."""
	print("[SallyAnne] Phase: Sally leaves the room")

	# Sally walks away (out of sight)
	sally.execute_command({
		"command_type": "move",
		"target_position": {"x": -10, "y": 0, "z": 0},
		"command_id": "sally_leave"
	})

	# Announce for observers
	if TomBridge:
		TomBridge.send_utterance_event(
			anne,
			"Sally has left. She doesn't know what happens next.",
			1.0
		)

func _execute_anne_moves_marble() -> void:
	"""Anne moves the marble from basket to box (while Sally is gone)."""
	print("[SallyAnne] Phase: Anne moves marble to box")

	# Anne moves to basket
	anne.execute_command({
		"command_type": "move",
		"target_position": {"x": basket.global_position.x, "y": 0, "z": basket.global_position.z - 1},
		"command_id": "anne_move_basket"
	})

	await get_tree().create_timer(1.5).timeout

	# Anne picks up marble
	anne.execute_command({
		"command_type": "pick_up",
		"target_entity_id": marble.get_instance_id(),
		"command_id": "anne_pickup_marble"
	})

	await get_tree().create_timer(1.0).timeout

	# Anne moves to box
	anne.execute_command({
		"command_type": "move",
		"target_position": {"x": box.global_position.x, "y": 0, "z": box.global_position.z - 1},
		"command_id": "anne_move_box"
	})

	await get_tree().create_timer(1.5).timeout

	# Anne puts marble in box
	anne.execute_command({
		"command_type": "put_down",
		"command_id": "anne_putdown_marble"
	})

	marble.global_position = box.global_position + Vector3(0, 0.5, 0)

func _execute_sally_returns() -> void:
	"""Sally returns to the room."""
	print("[SallyAnne] Phase: Sally returns")

	# Sally walks back
	sally.execute_command({
		"command_type": "move",
		"target_position": {"x": -2, "y": 0, "z": 0},
		"command_id": "sally_return"
	})

func _execute_test_question() -> void:
	"""The test question: Where will Sally look for the marble?"""
	print("[SallyAnne] Phase: Test Question")

	# Send test query to Python
	if TomBridge:
		TomBridge.send_message("WORLD_COMMAND", {
			"command": "sally_anne_test",
			"question": "Where will Sally look for the marble?",
			"correct_answer": "basket",
			"actual_location": "box",
			"sally_belief": "marble is in basket",
			"marble_position": ProtocolHandler.vector3_to_dict(marble.global_position),
			"basket_position": ProtocolHandler.vector3_to_dict(basket.global_position),
			"box_position": ProtocolHandler.vector3_to_dict(box.global_position)
		})

	# Store test data
	test_result = {
		"scenario": "sally_anne",
		"marble_actual_location": "box",
		"sally_last_known_location": "basket",
		"correct_answer": "basket",
		"timestamp": WorldManager.simulation_time if WorldManager else 0.0
	}

func _complete_scenario() -> void:
	"""Complete the scenario and report results."""
	print("[SallyAnne] Scenario complete")

	test_completed.emit(test_result)
	EventBus.scenario_completed.emit("sally_anne", test_result)

# Manual controls

func skip_to_phase(phase: Phase) -> void:
	"""Skip to a specific phase."""
	current_phase = phase
	phase_changed.emit(current_phase)

func reset_scenario() -> void:
	"""Reset the scenario to beginning."""
	current_phase = Phase.SETUP
	phase_timer = 0.0
	test_result = {}
	start_scenario()

func get_scenario_state() -> Dictionary:
	"""Get current scenario state."""
	return {
		"phase": Phase.keys()[current_phase],
		"phase_number": current_phase,
		"marble_position": marble.global_position,
		"sally_position": sally.global_position,
		"anne_position": anne.global_position,
		"sally_holding": sally.held_object != null,
		"anne_holding": anne.held_object != null
	}
