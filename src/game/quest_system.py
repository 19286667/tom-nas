"""
ToM-Driven Quest System

Quests that require theory of mind reasoning to complete.
Players must understand mental states, predict behavior, and reason recursively.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork

# ============================================================================
# Quest Types and Enums
# ============================================================================

class QuestType(Enum):
    """Type of ToM quest"""
    TUTORIAL = "tutorial"  # Learn basic ToM
    EMPATHY = "empathy"  # Understand emotions
    DECEPTION = "deception"  # Detect lies
    RECURSIVE = "recursive"  # Multi-level ToM
    COALITION = "coalition"  # Group dynamics
    ZOMBIE_DETECTION = "zombie_detection"  # Find non-conscious NPCs
    MORAL_DILEMMA = "moral_dilemma"  # Ethical reasoning

class QuestStatus(Enum):
    """Status of a quest"""
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

# ============================================================================
# Quest Objectives
# ============================================================================

@dataclass
class QuestObjective:
    """A single objective within a quest"""
    objective_id: str
    description: str
    required_tom_order: int = 1

    # Completion criteria
    completion_type: str = "custom"  # "custom", "dialogue", "observation", "action"
    completion_data: Dict[str, Any] = None

    # State
    is_complete: bool = False
    progress: float = 0.0  # 0.0 to 1.0

    # Rewards
    soul_map_reward: Optional[Dict[str, float]] = None

@dataclass
class Quest:
    """A complete quest with ToM requirements"""
    quest_id: str
    name: str
    description: str
    quest_type: QuestType
    required_tom_order: int  # Minimum ToM order to complete

    # Objectives
    objectives: List[QuestObjective] = None

    # NPCs involved
    involved_npcs: List[str] = None

    # Prerequisites
    required_quests: List[str] = None
    required_soul_map: Optional[Dict[str, Tuple[float, float]]] = None

    # State
    status: QuestStatus = QuestStatus.NOT_STARTED
    current_objective: int = 0

    # Rewards
    soul_map_rewards: Optional[Dict[str, float]] = None
    special_rewards: Optional[List[str]] = None

    # Story
    intro_text: str = ""
    completion_text: str = ""

# ============================================================================
# Quest Manager
# ============================================================================

class QuestManager:
    """
    Manages quests and tracks player progress.

    Features:
    - ToM-based quest progression
    - Dynamic objective generation
    - Recursive ToM challenges
    - Zombie detection quests
    - Coalition dynamics
    """

    def __init__(self, ontology: SoulMapOntology):
        self.ontology = ontology
        self.quests: Dict[str, Quest] = {}
        self.active_quests: List[str] = []
        self.completed_quests: List[str] = []

        # Create default quests
        self._create_tutorial_quests()
        self._create_advanced_quests()

    def start_quest(self, quest_id: str) -> Optional[Quest]:
        """Start a quest"""

        if quest_id not in self.quests:
            return None

        quest = self.quests[quest_id]

        # Check prerequisites
        if quest.required_quests:
            for req_id in quest.required_quests:
                if req_id not in self.completed_quests:
                    return None  # Prerequisites not met

        quest.status = QuestStatus.ACTIVE
        self.active_quests.append(quest_id)

        return quest

    def complete_objective(
        self,
        quest_id: str,
        objective_id: str,
        player_soul_map: torch.Tensor
    ) -> Dict[str, Any]:
        """Mark an objective as complete"""

        if quest_id not in self.quests:
            return {'success': False, 'error': 'Quest not found'}

        quest = self.quests[quest_id]

        if quest.status != QuestStatus.ACTIVE:
            return {'success': False, 'error': 'Quest not active'}

        # Find objective
        objective = None
        for obj in quest.objectives:
            if obj.objective_id == objective_id:
                objective = obj
                break

        if not objective:
            return {'success': False, 'error': 'Objective not found'}

        # Mark complete
        objective.is_complete = True
        objective.progress = 1.0

        # Apply rewards
        if objective.soul_map_reward:
            for dim_name, delta in objective.soul_map_reward.items():
                if dim_name in self.ontology.name_to_idx:
                    idx = self.ontology.name_to_idx[dim_name]
                    player_soul_map[idx] += delta
                    player_soul_map[idx] = torch.clamp(player_soul_map[idx], 0, 1)

        # Check if quest is complete
        all_complete = all(obj.is_complete for obj in quest.objectives)

        if all_complete:
            quest.status = QuestStatus.COMPLETED
            self.active_quests.remove(quest_id)
            self.completed_quests.append(quest_id)

            return {
                'success': True,
                'objective_complete': True,
                'quest_complete': True,
                'completion_text': quest.completion_text
            }

        return {
            'success': True,
            'objective_complete': True,
            'quest_complete': False
        }

    def check_tom_reasoning(
        self,
        quest_id: str,
        reasoning_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if player's ToM reasoning is correct.

        Used for quests that require explicit ToM demonstration.
        """

        if quest_id not in self.quests:
            return {'success': False, 'error': 'Quest not found'}

        quest = self.quests[quest_id]

        # Validate ToM order
        required_order = quest.required_tom_order
        demonstrated_order = reasoning_data.get('tom_order', 0)

        if demonstrated_order < required_order:
            return {
                'success': False,
                'feedback': f"This quest requires {required_order}-order ToM reasoning."
            }

        # Quest-specific validation
        if quest.quest_type == QuestType.DECEPTION:
            return self._check_deception_reasoning(reasoning_data)
        elif quest.quest_type == QuestType.RECURSIVE:
            return self._check_recursive_reasoning(reasoning_data)
        elif quest.quest_type == QuestType.ZOMBIE_DETECTION:
            return self._check_zombie_detection(reasoning_data)

        return {'success': True}

    def _check_deception_reasoning(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate deception detection reasoning"""

        # Check if player identified:
        # 1. Inconsistency in NPC's statements
        # 2. Mismatch between words and soul map
        # 3. Correct conclusion about deception

        identified_inconsistencies = data.get('inconsistencies', [])
        conclusion = data.get('conclusion', '')

        if len(identified_inconsistencies) >= 2 and 'lying' in conclusion.lower():
            return {
                'success': True,
                'feedback': "Excellent deception detection!"
            }

        return {
            'success': False,
            'feedback': "Look more carefully for inconsistencies."
        }

    def _check_recursive_reasoning(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate recursive ToM reasoning"""

        # Check depth of reasoning
        reasoning_levels = data.get('reasoning_levels', [])

        required_depth = 3  # Example: "A thinks B thinks C wants X"

        if len(reasoning_levels) >= required_depth:
            return {
                'success': True,
                'feedback': "Impressive recursive reasoning!"
            }

        return {
            'success': False,
            'feedback': f"This quest requires {required_depth} levels of reasoning."
        }

    def _check_zombie_detection(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate zombie (non-conscious NPC) detection"""

        # Check if player correctly identified behavioral markers
        markers_identified = data.get('markers', [])

        required_markers = ['belief_inconsistency', 'no_tom', 'scripted_responses']

        correct = sum(1 for m in markers_identified if m in required_markers)

        if correct >= 2:
            return {
                'success': True,
                'feedback': "You've identified the hollow one!"
            }

        return {
            'success': False,
            'feedback': "Look for signs of genuine theory of mind."
        }

    # ========================================================================
    # Quest Creation
    # ========================================================================

    def _create_tutorial_quests(self):
        """Create tutorial quests for learning ToM"""

        # Quest 1: The Cottage Test (1st order ToM)
        self.quests['cottage_test'] = Quest(
            quest_id='cottage_test',
            name="The Cottage Test",
            description="Learn to read emotions and basic mental states.",
            quest_type=QuestType.TUTORIAL,
            required_tom_order=1,
            objectives=[
                QuestObjective(
                    objective_id='meet_peregrine',
                    description="Meet Peregrine at the cottage",
                    required_tom_order=0,
                    completion_type='dialogue'
                ),
                QuestObjective(
                    objective_id='read_emotion',
                    description="Correctly identify Peregrine's emotional state",
                    required_tom_order=1,
                    completion_type='observation'
                ),
                QuestObjective(
                    objective_id='empathetic_response',
                    description="Respond with empathy",
                    required_tom_order=1,
                    completion_type='dialogue',
                    soul_map_reward={'social.empathy': 0.1}
                )
            ],
            intro_text=(
                "You arrive at a small cottage on the edge of the woods. "
                "An old figure sits by the window, seemingly lost in thought. "
                "This is Peregrine, the mentor. Your first lesson begins here."
            ),
            completion_text=(
                "Peregrine nods approvingly. 'You are learning to see beyond "
                "words, to understand the hearts of others. This is the first "
                "step on a long journey.'"
            ),
            soul_map_rewards={'wisdom.self_awareness': 0.15}
        )

        # Quest 2: The Lying Merchant (Deception detection)
        self.quests['lying_merchant'] = Quest(
            quest_id='lying_merchant',
            name="The Lying Merchant",
            description="Detect when an NPC is being deceptive.",
            quest_type=QuestType.DECEPTION,
            required_tom_order=2,
            required_quests=['cottage_test'],
            objectives=[
                QuestObjective(
                    objective_id='converse_merchant',
                    description="Talk to the merchant about their wares",
                    required_tom_order=1
                ),
                QuestObjective(
                    objective_id='detect_lie',
                    description="Identify inconsistencies in the merchant's story",
                    required_tom_order=2,
                    completion_type='custom'
                ),
                QuestObjective(
                    objective_id='confront_truth',
                    description="Confront the merchant with evidence",
                    required_tom_order=2,
                    soul_map_reward={'cognitive.analytical': 0.1, 'social.assertiveness': 0.05}
                )
            ],
            intro_text=(
                "A merchant in the town square offers you a 'magical' amulet "
                "for an exorbitant price. Something about their manner seems off..."
            ),
            completion_text=(
                "The merchant's facade crumbles. 'You... you saw through me. "
                "Few can do that. Perhaps I underestimated you.'"
            )
        )

    def _create_advanced_quests(self):
        """Create advanced ToM quests"""

        # The Infinite Jest (3rd order ToM, recursive reasoning)
        self.quests['infinite_jest'] = Quest(
            quest_id='infinite_jest',
            name="The Infinite Jest",
            description="Navigate the complex social dynamics of the mysterious tavern.",
            quest_type=QuestType.RECURSIVE,
            required_tom_order=3,
            required_quests=['cottage_test', 'lying_merchant'],
            objectives=[
                QuestObjective(
                    objective_id='enter_jest',
                    description="Gain entry to the Infinite Jest",
                    required_tom_order=1
                ),
                QuestObjective(
                    objective_id='understand_barkeep',
                    description="Understand what the Barkeep wants from you",
                    required_tom_order=2
                ),
                QuestObjective(
                    objective_id='navigate_coalitions',
                    description="Navigate the three factions without offending anyone",
                    required_tom_order=3,
                    completion_type='custom'
                ),
                QuestObjective(
                    objective_id='reveal_stranger',
                    description="Discover the Stranger's true intentions",
                    required_tom_order=3
                ),
                QuestObjective(
                    objective_id='recursive_deception',
                    description="'A knows that B knows that C believes D is lying about E'",
                    required_tom_order=4,
                    soul_map_reward={'cognitive.strategic_thinking': 0.2}
                )
            ],
            involved_npcs=['The Barkeep', 'The Stranger', 'Coalition Leaders'],
            intro_text=(
                "The Infinite Jest is not like other taverns. Here, every "
                "conversation is a game, every word a move on an invisible "
                "board. The Barkeep watches with knowing eyes as you enter..."
            ),
            completion_text=(
                "The Barkeep pours you a drink. 'Impressive. Most never make "
                "it past the third layer. You see not just minds, but minds "
                "within minds. Welcome to the deeper game.'"
            ),
            soul_map_rewards={
                'wisdom.self_awareness': 0.3,
                'cognitive.strategic_thinking': 0.2,
                'social.political_acumen': 0.25
            }
        )

        # The Hollow Ones (Zombie detection)
        self.quests['hollow_ones'] = Quest(
            quest_id='hollow_ones',
            name="The Hollow Ones",
            description="Identify NPCs without genuine consciousness.",
            quest_type=QuestType.ZOMBIE_DETECTION,
            required_tom_order=2,
            objectives=[
                QuestObjective(
                    objective_id='learn_about_hollows',
                    description="Learn the signs of a hollow one from Peregrine",
                    required_tom_order=1
                ),
                QuestObjective(
                    objective_id='test_npcs',
                    description="Test 5 NPCs for signs of genuine ToM",
                    required_tom_order=2,
                    completion_type='custom'
                ),
                QuestObjective(
                    objective_id='identify_zombie',
                    description="Correctly identify the zombie NPC",
                    required_tom_order=2,
                    soul_map_reward={'cognitive.analytical': 0.15}
                ),
                QuestObjective(
                    objective_id='philosophical_reflection',
                    description="Reflect on the nature of consciousness",
                    required_tom_order=3,
                    soul_map_reward={'existential.awareness': 0.2}
                )
            ],
            intro_text=(
                "Peregrine speaks gravely: 'Not all who walk and talk are "
                "truly present. Some are hollow shells, going through "
                "motions without understanding. Can you tell the difference?'"
            ),
            completion_text=(
                "'You have seen the truth,' Peregrine says. 'The hollow ones "
                "lack the spark that makes us real. But knowing this raises "
                "deeper questions: what makes YOU real?'"
            )
        )

# ============================================================================
# Quest Event System
# ============================================================================

class QuestEventManager:
    """Manages quest-related events and triggers"""

    def __init__(self, quest_manager: QuestManager):
        self.quest_manager = quest_manager
        self.event_handlers: Dict[str, List] = {}

    def trigger_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Trigger a quest event"""

        # Check all active quests for relevant objectives
        for quest_id in self.quest_manager.active_quests:
            quest = self.quest_manager.quests[quest_id]

            for objective in quest.objectives:
                if objective.is_complete:
                    continue

                # Check if this event completes the objective
                if objective.completion_type == event_type:
                    # Validate completion criteria
                    if self._validate_completion(objective, event_data):
                        objective.is_complete = True
                        objective.progress = 1.0

    def _validate_completion(
        self,
        objective: QuestObjective,
        event_data: Dict[str, Any]
    ) -> bool:
        """Validate if event satisfies objective"""

        # Custom validation logic per completion type
        if objective.completion_type == 'dialogue':
            required_npc = objective.completion_data.get('npc_id') if objective.completion_data else None
            actual_npc = event_data.get('npc_id')
            return required_npc == actual_npc or required_npc is None

        elif objective.completion_type == 'observation':
            # Check if observation was correct
            return event_data.get('correct', False)

        return True

# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ToM Quest System Demo")
    print("=" * 60)

    # Create ontology and quest manager
    ontology = SoulMapOntology()
    quest_manager = QuestManager(ontology)

    # Player soul map
    player_soul_map = ontology.get_default_state()

    print("\n📜 Available Quests:")
    for quest_id, quest in quest_manager.quests.items():
        print(f"\n{quest.name} ({quest_id})")
        print(f"  Type: {quest.quest_type.value}")
        print(f"  Required ToM Order: {quest.required_tom_order}")
        print(f"  Objectives: {len(quest.objectives)}")
        print(f"  Description: {quest.description}")

    # Start tutorial quest
    print("\n" + "=" * 60)
    print("Starting: The Cottage Test")
    print("=" * 60)

    quest = quest_manager.start_quest('cottage_test')

    if quest:
        print(f"\n{quest.intro_text}")

        print(f"\nObjectives:")
        for i, obj in enumerate(quest.objectives, 1):
            status = "✓" if obj.is_complete else "○"
            print(f"  {status} {i}. {obj.description}")

        # Simulate completing first objective
        print(f"\n--- Completing Objective: Meet Peregrine ---")
        result = quest_manager.complete_objective(
            'cottage_test',
            'meet_peregrine',
            player_soul_map
        )

        print(f"Result: {result}")

        # Show updated objectives
        print(f"\nUpdated Objectives:")
        for i, obj in enumerate(quest.objectives, 1):
            status = "✓" if obj.is_complete else "○"
            print(f"  {status} {i}. {obj.description}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
