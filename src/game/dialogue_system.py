"""
ToM-Driven Dialogue System

Dialogue that responds to psychological states and demonstrates theory of mind.
NPCs reason about player beliefs, intentions, and emotions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork

# ============================================================================
# Data Structures
# ============================================================================

class DialogueIntent(Enum):
    """Player's possible intents in dialogue"""
    GREET = "greet"
    QUESTION = "question"
    REQUEST_HELP = "request_help"
    THREATEN = "threaten"
    PERSUADE = "persuade"
    DECEIVE = "deceive"
    EMPATHIZE = "empathize"
    FAREWELL = "farewell"

@dataclass
class DialogueNode:
    """A single dialogue node with ToM reasoning"""
    node_id: str
    speaker: str  # "player" or NPC name
    text: str

    # ToM reasoning (NPC only)
    observations: Optional[Dict[str, Any]] = None  # What NPC observes
    first_order_belief: Optional[str] = None  # What player feels/wants
    second_order_belief: Optional[str] = None  # What player thinks NPC believes
    response_rationale: Optional[str] = None  # Why NPC responds this way

    # Player options
    player_options: Optional[List['DialogueOption']] = None

    # Effects
    soul_map_delta: Optional[Dict[str, float]] = None  # Changes to soul map
    relationship_delta: float = 0.0  # Change in relationship

    # Flow control
    next_node: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None

@dataclass
class DialogueOption:
    """A player dialogue choice"""
    option_id: str
    text: str
    intent: DialogueIntent

    # Requirements
    required_soul_map: Optional[Dict[str, Tuple[float, float]]] = None  # Min/max ranges
    required_relationship: Optional[float] = None

    # Effects
    soul_map_delta: Dict[str, float] = None
    relationship_delta: float = 0.0

    # ToM interpretation (how NPC understands this choice)
    tom_interpretation: str = ""

    # Next node
    next_node: str = ""

# ============================================================================
# Dialogue Manager
# ============================================================================

class DialogueManager:
    """
    Manages ToM-driven dialogue trees.

    Features:
    - Dynamic response generation based on Soul Maps
    - Theory of Mind reasoning (1st through 5th order)
    - Deception detection
    - Emotional state tracking
    - Relationship evolution
    """

    def __init__(self, ontology: SoulMapOntology):
        self.ontology = ontology
        self.conversations: Dict[str, 'Conversation'] = {}

        # Template responses for different emotional states
        self.response_templates = self._build_response_templates()

    def start_conversation(
        self,
        conversation_id: str,
        npc_name: str,
        npc_soul_map: torch.Tensor,
        player_soul_map: torch.Tensor,
        context: str = ""
    ) -> 'Conversation':
        """Start a new conversation"""

        conversation = Conversation(
            conversation_id=conversation_id,
            npc_name=npc_name,
            npc_soul_map=npc_soul_map,
            player_soul_map=player_soul_map,
            context=context,
            ontology=self.ontology
        )

        self.conversations[conversation_id] = conversation

        # Generate opening dialogue
        opening = conversation.generate_npc_greeting()

        return conversation

    def process_player_choice(
        self,
        conversation_id: str,
        option_id: str
    ) -> DialogueNode:
        """Process player's dialogue choice and generate NPC response"""

        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]

        # Apply player choice effects
        conversation.apply_player_choice(option_id)

        # Generate NPC response
        response = conversation.generate_npc_response()

        return response

    def _build_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Build response templates for different states"""

        return {
            'greeting': {
                'friendly': [
                    "Welcome, friend! How can I help you?",
                    "It's good to see you again.",
                    "Hello! You look well today."
                ],
                'neutral': [
                    "Greetings, traveler.",
                    "Yes? What is it?",
                    "You have business with me?"
                ],
                'hostile': [
                    "What do YOU want?",
                    "I have nothing to say to you.",
                    "You have nerve showing your face here."
                ]
            },
            'response_to_distress': {
                'empathetic': [
                    "I can see something is troubling you. Would you like to talk about it?",
                    "You seem burdened. I'm here to listen if you need.",
                    "Take a moment. There's no rush."
                ],
                'indifferent': [
                    "That's... unfortunate.",
                    "Well, we all have our problems.",
                    "I see."
                ],
                'exploitative': [
                    "You look vulnerable. Perhaps I can help... for a price.",
                    "Desperation makes people careless.",
                    "In your state, you'd agree to anything, wouldn't you?"
                ]
            },
            'response_to_deception': {
                'call_out': [
                    "That's not the truth, and we both know it.",
                    "You're lying to me. Why?",
                    "I can see right through that facade."
                ],
                'play_along': [
                    "Is that so? Interesting...",
                    "Of course. I believe you.",
                    "Mm-hmm. Go on."
                ],
                'ignore': [
                    "If you say so.",
                    "Whatever you claim.",
                    "I won't press the matter."
                ]
            }
        }

# ============================================================================
# Conversation
# ============================================================================

class Conversation:
    """A single conversation between player and NPC"""

    def __init__(
        self,
        conversation_id: str,
        npc_name: str,
        npc_soul_map: torch.Tensor,
        player_soul_map: torch.Tensor,
        context: str,
        ontology: SoulMapOntology
    ):
        self.conversation_id = conversation_id
        self.npc_name = npc_name
        self.npc_soul_map = npc_soul_map.clone()
        self.player_soul_map = player_soul_map.clone()
        self.context = context
        self.ontology = ontology

        # Conversation state
        self.history: List[DialogueNode] = []
        self.current_node: Optional[DialogueNode] = None
        self.relationship = 0.5  # 0 = hostile, 1 = friendly
        self.trust = 0.5  # NPC's trust in player
        self.deception_detected = False

        # ToM beliefs
        self.belief_network = BeliefNetwork(ontology.total_dims)

    def generate_npc_greeting(self) -> DialogueNode:
        """Generate NPC's opening line based on psychological states"""

        # Analyze player's state
        player_state = self._analyze_soul_map(self.player_soul_map)
        npc_state = self._analyze_soul_map(self.npc_soul_map)

        # Determine NPC's stance
        stance = self._determine_stance()

        # First-order ToM: What is the player feeling?
        first_order = self._first_order_reasoning(player_state)

        # Second-order ToM: What does the player expect from me?
        second_order = self._second_order_reasoning(player_state, npc_state)

        # Generate greeting text
        if player_state['distressed'] and npc_state['empathetic']:
            greeting_text = "You seem troubled, traveler. What burdens you?"
        elif player_state['confident'] and npc_state['analytical']:
            greeting_text = "You carry yourself with purpose. State your business."
        elif stance == 'hostile':
            greeting_text = "What do you want?"
        elif stance == 'friendly':
            greeting_text = "Welcome, friend! How may I assist you?"
        else:
            greeting_text = "Greetings."

        # Create dialogue node
        node = DialogueNode(
            node_id="greeting_01",
            speaker=self.npc_name,
            text=greeting_text,
            observations={
                'player_emotional_state': player_state,
                'stance': stance
            },
            first_order_belief=first_order,
            second_order_belief=second_order,
            response_rationale=f"NPC is {stance} and responds to player's {self._describe_state(player_state)}"
        )

        # Generate player options
        node.player_options = self._generate_player_options(player_state, npc_state)

        self.history.append(node)
        self.current_node = node

        return node

    def generate_npc_response(self) -> DialogueNode:
        """Generate NPC response to player's last choice"""

        if not self.history:
            return self.generate_npc_greeting()

        last_player_choice = self.history[-1]

        # Analyze what just happened
        player_state = self._analyze_soul_map(self.player_soul_map)
        npc_state = self._analyze_soul_map(self.npc_soul_map)

        # ToM reasoning about player's choice
        first_order = self._first_order_reasoning(player_state)
        second_order = self._second_order_reasoning(player_state, npc_state)

        # Detect deception if applicable
        if last_player_choice.soul_map_delta:
            self._check_deception(last_player_choice)

        # Generate response based on ToM reasoning
        response_text = self._generate_response_text(
            player_state,
            npc_state,
            last_player_choice
        )

        # Create node
        node = DialogueNode(
            node_id=f"response_{len(self.history)}",
            speaker=self.npc_name,
            text=response_text,
            observations={'player_state': player_state},
            first_order_belief=first_order,
            second_order_belief=second_order,
            response_rationale=f"Responding to player's {last_player_choice.text}"
        )

        # Generate new options
        node.player_options = self._generate_player_options(player_state, npc_state)

        self.history.append(node)
        self.current_node = node

        return node

    def apply_player_choice(self, option_id: str):
        """Apply effects of player's dialogue choice"""

        if not self.current_node or not self.current_node.player_options:
            return

        # Find chosen option
        chosen_option = None
        for option in self.current_node.player_options:
            if option.option_id == option_id:
                chosen_option = option
                break

        if not chosen_option:
            raise ValueError(f"Option {option_id} not found")

        # Apply soul map changes
        if chosen_option.soul_map_delta:
            for dim_name, delta in chosen_option.soul_map_delta.items():
                if dim_name in self.ontology.name_to_idx:
                    idx = self.ontology.name_to_idx[dim_name]
                    self.player_soul_map[idx] += delta
                    self.player_soul_map[idx] = torch.clamp(
                        self.player_soul_map[idx], 0, 1
                    )

        # Apply relationship change
        self.relationship += chosen_option.relationship_delta
        self.relationship = np.clip(self.relationship, 0, 1)

        # Record in history
        player_node = DialogueNode(
            node_id=f"player_{len(self.history)}",
            speaker="player",
            text=chosen_option.text,
            soul_map_delta=chosen_option.soul_map_delta,
            relationship_delta=chosen_option.relationship_delta
        )

        self.history.append(player_node)

    def _analyze_soul_map(self, soul_map: torch.Tensor) -> Dict[str, Any]:
        """Analyze a soul map and extract key psychological states"""

        # Extract key dimensions
        valence = float(soul_map[1]) if len(soul_map) > 1 else 0.5
        arousal = float(soul_map[2]) if len(soul_map) > 2 else 0.5
        dominance = float(soul_map[3]) if len(soul_map) > 3 else 0.5

        return {
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'distressed': valence < 0.3 and arousal > 0.7,
            'confident': dominance > 0.7 and valence > 0.5,
            'fearful': valence < 0.3 and dominance < 0.3,
            'calm': arousal < 0.3 and valence > 0.5,
            'empathetic': True,  # Placeholder - would check social dimensions
            'analytical': True,  # Placeholder - would check cognitive dimensions
        }

    def _determine_stance(self) -> str:
        """Determine NPC's overall stance toward player"""

        if self.relationship > 0.7:
            return 'friendly'
        elif self.relationship < 0.3:
            return 'hostile'
        else:
            return 'neutral'

    def _first_order_reasoning(self, player_state: Dict[str, Any]) -> str:
        """First-order ToM: What is the player feeling/wanting?"""

        if player_state['distressed']:
            return "The player is emotionally distressed and likely needs support."
        elif player_state['confident']:
            return "The player is confident and has clear intentions."
        elif player_state['fearful']:
            return "The player is afraid and may be defensive."
        else:
            return "The player's emotional state is neutral."

    def _second_order_reasoning(
        self,
        player_state: Dict[str, Any],
        npc_state: Dict[str, Any]
    ) -> str:
        """Second-order ToM: What does the player think I believe?"""

        if self.relationship > 0.7:
            return "The player trusts me and expects empathy."
        elif self.relationship < 0.3:
            return "The player sees me as hostile and expects conflict."
        else:
            return "The player is uncertain about my intentions."

    def _describe_state(self, state: Dict[str, Any]) -> str:
        """Describe a psychological state in words"""

        if state['distressed']:
            return "distressed state"
        elif state['confident']:
            return "confident demeanor"
        elif state['fearful']:
            return "fearful disposition"
        else:
            return "neutral state"

    def _check_deception(self, player_node: DialogueNode):
        """Detect if player is being deceptive"""

        # Simple heuristic: inconsistency between words and soul map
        # In full version, use more sophisticated analysis

        if player_node.soul_map_delta:
            # Check for defensive behavior
            defensiveness = player_node.soul_map_delta.get('affect.defensiveness', 0)
            if defensiveness > 0.2:
                self.deception_detected = True
                self.trust -= 0.1

    def _generate_response_text(
        self,
        player_state: Dict[str, Any],
        npc_state: Dict[str, Any],
        last_player_choice: DialogueNode
    ) -> str:
        """Generate NPC's response text"""

        # Template-based for now - in full version, use LLM

        if self.deception_detected:
            return "I sense you're not being entirely truthful with me..."

        if player_state['distressed'] and npc_state['empathetic']:
            return "I understand. These are difficult matters."

        if self.relationship > 0.7:
            return "Of course. I'm happy to help."
        elif self.relationship < 0.3:
            return "Fine. But don't expect much from me."
        else:
            return "I see. Go on."

    def _generate_player_options(
        self,
        player_state: Dict[str, Any],
        npc_state: Dict[str, Any]
    ) -> List[DialogueOption]:
        """Generate player dialogue options"""

        options = []

        # Always available: basic interaction
        options.append(DialogueOption(
            option_id="ask_about_npc",
            text="Tell me about yourself.",
            intent=DialogueIntent.QUESTION,
            soul_map_delta={'social.openness': 0.05},
            relationship_delta=0.05,
            tom_interpretation="Player shows interest in learning about me",
            next_node="npc_background"
        ))

        # Conditional: if player is distressed
        if player_state['distressed']:
            options.append(DialogueOption(
                option_id="request_help",
                text="I need your help. I'm struggling.",
                intent=DialogueIntent.REQUEST_HELP,
                soul_map_delta={'affect.vulnerability': 0.1, 'social.trust': 0.15},
                relationship_delta=0.1,
                tom_interpretation="Player is vulnerable and trusts me with their distress",
                next_node="offer_help"
            ))

        # Conditional: if relationship is high
        if self.relationship > 0.6:
            options.append(DialogueOption(
                option_id="empathize",
                text="I understand how you feel.",
                intent=DialogueIntent.EMPATHIZE,
                soul_map_delta={'social.empathy': 0.1},
                relationship_delta=0.15,
                tom_interpretation="Player demonstrates empathy and emotional intelligence",
                next_node="deepen_bond"
            ))

        # Always available: farewell
        options.append(DialogueOption(
            option_id="farewell",
            text="I should go now.",
            intent=DialogueIntent.FAREWELL,
            soul_map_delta={},
            relationship_delta=0.0,
            tom_interpretation="Player wishes to end conversation",
            next_node="end"
        ))

        return options

# ============================================================================
# Dialogue Tree Builder
# ============================================================================

class DialogueTreeBuilder:
    """Build complex dialogue trees with ToM reasoning"""

    def __init__(self, ontology: SoulMapOntology):
        self.ontology = ontology
        self.trees: Dict[str, Dict[str, DialogueNode]] = {}

    def create_tree(self, tree_id: str) -> Dict[str, DialogueNode]:
        """Create a new dialogue tree"""
        self.trees[tree_id] = {}
        return self.trees[tree_id]

    def add_node(
        self,
        tree_id: str,
        node: DialogueNode
    ):
        """Add a node to a tree"""
        if tree_id not in self.trees:
            self.create_tree(tree_id)

        self.trees[tree_id][node.node_id] = node

    def save_tree(self, tree_id: str, filepath: str):
        """Save dialogue tree to JSON"""

        if tree_id not in self.trees:
            raise ValueError(f"Tree {tree_id} not found")

        # Convert to serializable format
        tree_data = {}
        for node_id, node in self.trees[tree_id].items():
            tree_data[node_id] = self._node_to_dict(node)

        with open(filepath, 'w') as f:
            json.dump(tree_data, f, indent=2)

    def _node_to_dict(self, node: DialogueNode) -> Dict:
        """Convert node to dictionary"""
        # Implementation would serialize all fields
        return {
            'node_id': node.node_id,
            'speaker': node.speaker,
            'text': node.text,
            # ... other fields
        }

# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ToM Dialogue System Demo")
    print("=" * 60)

    # Create ontology
    ontology = SoulMapOntology()

    # Create soul maps
    # Player: distressed, low valence, high arousal
    player_soul_map = ontology.get_default_state()
    player_soul_map[1] = 0.2  # Low valence
    player_soul_map[2] = 0.8  # High arousal

    # NPC: empathetic, calm
    npc_soul_map = ontology.get_default_state()
    npc_soul_map[1] = 0.7  # Positive valence
    npc_soul_map[2] = 0.3  # Low arousal (calm)

    # Start conversation
    manager = DialogueManager(ontology)
    conversation = manager.start_conversation(
        conversation_id="test_001",
        npc_name="Peregrine",
        npc_soul_map=npc_soul_map,
        player_soul_map=player_soul_map,
        context="Player approaches Peregrine's cottage seeking guidance"
    )

    # Get greeting
    greeting = conversation.generate_npc_greeting()

    print(f"\n{greeting.speaker}: {greeting.text}")
    print(f"\nToM Reasoning:")
    print(f"  First-order: {greeting.first_order_belief}")
    print(f"  Second-order: {greeting.second_order_belief}")
    print(f"  Rationale: {greeting.response_rationale}")

    print(f"\nPlayer Options:")
    for i, option in enumerate(greeting.player_options, 1):
        print(f"  {i}. {option.text}")
        print(f"     -> NPC interprets as: {option.tom_interpretation}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
