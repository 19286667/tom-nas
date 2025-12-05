extends CanvasLayer
## DebugPanel: Real-time monitoring of game state and Python connection

@onready var panel: Panel = $Panel
@onready var connection_label: Label = $Panel/VBox/ConnectionStatus
@onready var npc_list: VBoxContainer = $Panel/VBox/ScrollContainer/NPCList
@onready var selected_npc_info: RichTextLabel = $Panel/VBox/SelectedInfo

var visible_panel: bool = false
var selected_npc_id: String = ""
var update_timer: float = 0.0


func _ready() -> void:
	# Start hidden
	panel.visible = false
	
	# Connect to events
	EventBus.debug_panel_toggled.connect(_on_toggle)
	EventBus.npc_spawned.connect(_on_npc_spawned)
	EventBus.npc_state_changed.connect(_on_npc_state_changed)
	GameBridge.connected.connect(_on_bridge_connected)
	GameBridge.disconnected.connect(_on_bridge_disconnected)
	
	_update_connection_status()


func _process(delta: float) -> void:
	if not visible_panel:
		return
	
	update_timer += delta
	if update_timer >= 0.25:  # Update 4x per second
		update_timer = 0.0
		_update_display()


func _on_toggle(_visible: bool = true) -> void:
	visible_panel = not visible_panel
	panel.visible = visible_panel
	
	if visible_panel:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		_refresh_npc_list()
	else:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _update_connection_status() -> void:
	if not connection_label:
		return
	
	if GameBridge.is_connected_to_backend():
		connection_label.text = "🟢 Python Backend: Connected"
		connection_label.modulate = Color.GREEN
	else:
		connection_label.text = "🔴 Python Backend: Disconnected"
		connection_label.modulate = Color.RED


func _on_bridge_connected() -> void:
	_update_connection_status()


func _on_bridge_disconnected() -> void:
	_update_connection_status()


func _refresh_npc_list() -> void:
	"""Rebuild the NPC list"""
	if not npc_list:
		return
	
	# Clear existing
	for child in npc_list.get_children():
		child.queue_free()
	
	# Add all NPCs
	for npc_id in SoulMapManager.get_all_npc_ids():
		var button = Button.new()
		button.text = npc_id
		button.pressed.connect(_on_npc_selected.bind(npc_id))
		npc_list.add_child(button)


func _on_npc_spawned(npc_id: String, _node: Node) -> void:
	if visible_panel:
		_refresh_npc_list()


func _on_npc_state_changed(npc_id: String, _old: String, _new: String) -> void:
	if npc_id == selected_npc_id:
		_update_selected_info()


func _on_npc_selected(npc_id: String) -> void:
	selected_npc_id = npc_id
	_update_selected_info()


func _update_display() -> void:
	_update_connection_status()
	if selected_npc_id != "":
		_update_selected_info()


func _update_selected_info() -> void:
	"""Update the detailed info panel for selected NPC"""
	if not selected_npc_info or selected_npc_id == "":
		return
	
	var soul_map = SoulMapManager.get_soul_map(selected_npc_id)
	var beliefs = SoulMapManager.get_beliefs(selected_npc_id)
	var dominant = SoulMapManager.get_dominant_traits(selected_npc_id, 5)
	
	# Find NPC node for state info
	var npc_node: NPCController = null
	for npc in get_tree().get_nodes_in_group("npcs"):
		if npc is NPCController and npc.npc_id == selected_npc_id:
			npc_node = npc
			break
	
	var text = "[b]%s[/b]\n\n" % selected_npc_id
	
	# State
	if npc_node:
		text += "[color=cyan]State:[/color] %s\n" % NPCController.State.keys()[npc_node.current_state]
		text += "[color=cyan]Perceived:[/color] %d entities\n\n" % npc_node.perceived_entities.size()
	
	# Dominant traits
	text += "[b]Dominant Traits:[/b]\n"
	for trait_info in dominant:
		var value = trait_info["value"]
		var bar = _make_bar(value)
		var color = "green" if value > 0.5 else "red"
		text += "[color=%s]%s[/color]: %s %.2f\n" % [color, trait_info["trait"], bar, value]
	
	# Key social dimensions
	text += "\n[b]Social Cognition:[/b]\n"
	text += "Trust propensity: %s %.2f\n" % [_make_bar(soul_map.get("trust_propensity", 0.5)), soul_map.get("trust_propensity", 0.5)]
	text += "Deception detection: %s %.2f\n" % [_make_bar(soul_map.get("deception_detection", 0.5)), soul_map.get("deception_detection", 0.5)]
	text += "ToM depth: %s %.2f\n" % [_make_bar(soul_map.get("theory_of_mind_depth", 0.5)), soul_map.get("theory_of_mind_depth", 0.5)]
	
	# Beliefs about player
	if beliefs.has("player"):
		text += "\n[b]Beliefs about Player:[/b]\n"
		var player_beliefs = beliefs["player"]
		for key in player_beliefs:
			text += "%s: %s\n" % [key, player_beliefs[key]]
	
	selected_npc_info.bbcode_enabled = true
	selected_npc_info.text = text


func _make_bar(value: float, width: int = 10) -> String:
	"""Create a simple text-based progress bar"""
	var filled = int(value * width)
	var empty = width - filled
	return "[" + "█".repeat(filled) + "░".repeat(empty) + "]"
