"""
Core module for ToM-NAS.

Contains the foundational components:
- SoulMapOntology: 181-dimensional psychological state space
- RecursiveBeliefState: 5th-order belief modeling
- BeliefNetwork: Multi-agent belief management
- Events: Information asymmetry tracking for ToM scenarios
"""

from .ontology import SoulMapOntology, OntologyDimension
from .beliefs import Belief, RecursiveBeliefState, BeliefNetwork
from .events import (
    Event,
    EventType,
    AgentBeliefState,
    WorldState,
    InformationAsymmetryTracker,
    create_sally_anne_scenario,
    verify_information_asymmetry,
)

__all__ = [
    'SoulMapOntology',
    'OntologyDimension',
    'Belief',
    'RecursiveBeliefState',
    'BeliefNetwork',
    'Event',
    'EventType',
    'AgentBeliefState',
    'WorldState',
    'InformationAsymmetryTracker',
    'create_sally_anne_scenario',
    'verify_information_asymmetry',
]
