extends Node
## EventBus: Central signal hub for decoupled communication
## Allows systems to communicate without direct references

# --- NPC Events ---
signal npc_spawn_requested(payload: Dictionary)
signal npc_spawned(npc_id: String, npc_node: Node)
signal npc_died(npc_id: String)
signal npc_state_changed(npc_id: String, old_state: String, new_state: String)

# --- Player Events ---
signal player_entered_area(area_name: String)
signal player_interacted(target_id: String, interaction_type: String)
signal player_attacked(target_id: String)
signal player_spoke(dialogue_choice: String)

# --- Dialogue Events ---
signal dialogue_started(npc_id: String)
signal dialogue_ended(npc_id: String)
signal dialogue_received(payload: Dictionary)
signal dialogue_choice_made(npc_id: String, choice_index: int)

# --- Perception Events ---
signal entity_perceived(perceiver_id: String, entity_id: String, entity_data: Dictionary)
signal entity_lost(perceiver_id: String, entity_id: String)
signal threat_detected(perceiver_id: String, threat_id: String, threat_level: float)

# --- Social Events ---
signal relationship_changed(npc_id: String, target_id: String, relationship: Dictionary)
signal trust_updated(npc_id: String, target_id: String, old_trust: float, new_trust: float)
signal betrayal_detected(detector_id: String, betrayer_id: String)
signal alliance_formed(members: Array)
signal alliance_broken(members: Array)

# --- Narrative Events ---
signal narrative_beat(payload: Dictionary)
signal quest_started(quest_id: String)
signal quest_completed(quest_id: String)
signal quest_failed(quest_id: String)
signal realm_entered(realm_name: String)
signal realm_exited(realm_name: String)

# --- Combat Events ---
signal combat_started(participants: Array)
signal combat_ended(outcome: Dictionary)
signal damage_dealt(attacker_id: String, target_id: String, amount: float)
signal entity_defeated(entity_id: String, by_id: String)

# --- UI Events ---
signal show_dialogue_ui(npc_id: String, text: String, choices: Array)
signal hide_dialogue_ui()
signal show_soul_scanner(npc_id: String)
signal hide_soul_scanner()
signal show_notification(text: String, duration: float)
signal debug_panel_toggled(visible: bool)

# --- System Events ---
signal game_paused()
signal game_resumed()
signal save_requested()
signal load_requested(save_id: String)
