extends InteractableObject
## Item - Specialized interactable objects
##
## Items are objects that can be picked up, traded, used, and have value.
## Used in social scenarios like market exchanges.

class_name TomItem

# Item properties
@export var base_value: float = 1.0
@export var item_type: String = "generic"
@export var quantity: int = 1
@export var max_stack: int = 1

# Trade properties
@export var is_tradeable: bool = true
@export var is_valuable: bool = false

# Use effects
@export var use_effect: String = ""  # "heal", "damage", "buff", etc.
@export var effect_magnitude: float = 0.1

# State
var current_value: float = 1.0

func _ready() -> void:
	super._ready()

	current_value = base_value
	entity_type = "item"

	# Add item-specific tags
	semantic_tags.append("item")
	if is_valuable:
		semantic_tags.append("valuable")
	if is_tradeable:
		semantic_tags.append("tradeable")

	# Update affordances
	_compute_affordances()

func _compute_affordances() -> void:
	super._compute_affordances()

	if is_tradeable:
		affordances.append("can_trade")

	if use_effect:
		affordances.append("has_effect")

func _on_use(agent: Node3D) -> Dictionary:
	"""Apply item use effect."""
	if use_effect.is_empty():
		return {"success": true, "effect": "none"}

	var result = {"success": true, "effect": use_effect, "magnitude": effect_magnitude}

	match use_effect:
		"heal":
			if agent.has_method("restore_energy"):
				agent.restore_energy(effect_magnitude)
				result["restored"] = effect_magnitude

		"damage":
			if agent.has_method("consume_energy"):
				agent.consume_energy(effect_magnitude)
				result["damage"] = effect_magnitude

		"buff":
			# Would apply a buff effect
			result["buff_applied"] = true

		_:
			result["effect"] = "custom"

	# Consume item after use (if single use)
	if quantity <= 1:
		result["consumed"] = true
		queue_free()
	else:
		quantity -= 1

	return result

# Trading methods

func get_value() -> float:
	"""Get current trade value."""
	return current_value

func set_value(value: float) -> void:
	"""Set trade value."""
	current_value = max(0.0, value)

func can_trade() -> bool:
	"""Check if item can be traded."""
	return is_tradeable and not is_being_held

func trade_to(new_owner: Node3D) -> bool:
	"""Transfer ownership through trade."""
	if not can_trade():
		return false

	set_owner(new_owner)
	return true

# Stacking

func can_stack_with(other: TomItem) -> bool:
	"""Check if this item can stack with another."""
	if other.item_type != item_type:
		return false
	if quantity + other.quantity > max_stack:
		return false
	return true

func stack_with(other: TomItem) -> bool:
	"""Stack another item into this one."""
	if not can_stack_with(other):
		return false

	quantity += other.quantity
	other.queue_free()
	return true

func split(amount: int) -> TomItem:
	"""Split items from stack."""
	if amount >= quantity or amount <= 0:
		return null

	quantity -= amount

	# Create new item with split amount
	var new_item = duplicate()
	new_item.quantity = amount
	get_parent().add_child(new_item)
	new_item.global_position = global_position

	return new_item

func get_state_dict() -> Dictionary:
	var state = super.get_state_dict()
	state["item_type"] = item_type
	state["value"] = current_value
	state["quantity"] = quantity
	state["is_tradeable"] = is_tradeable
	state["use_effect"] = use_effect
	return state
