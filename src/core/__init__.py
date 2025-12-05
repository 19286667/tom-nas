"""
Core module for ToM-NAS.

Contains the foundational components:
- SoulMapOntology: 181-dimensional psychological state space
- RecursiveBeliefState: 5th-order belief modeling
- BeliefNetwork: Multi-agent belief management
- Event system with observation tracking for information asymmetry
"""

from .ontology import SoulMapOntology, OntologyDimension
from .beliefs import BeliefNode, RecursiveBeliefState, BeliefNetwork
from .events import (
    Event, ActionType, AgentBeliefs, Question, QuestionType,
    EventEncoder, ScenarioEncoder, AnswerDecoder,
    compute_ground_truth, create_sally_anne_scenario, verify_information_asymmetry
)

__all__ = [
    'SoulMapOntology',
    'OntologyDimension',
    'BeliefNode',
    'RecursiveBeliefState',
    'BeliefNetwork',
    # Event system
    'Event',
    'ActionType',
    'AgentBeliefs',
    'Question',
    'QuestionType',
    'EventEncoder',
    'ScenarioEncoder',
    'AnswerDecoder',
    'compute_ground_truth',
    'create_sally_anne_scenario',
    'verify_information_asymmetry',
]
