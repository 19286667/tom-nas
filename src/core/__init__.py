"""
Core module for ToM-NAS.

Contains the foundational components:
- SoulMapOntology: 181-dimensional psychological state space
- RecursiveBeliefState: 5th-order belief modeling
- BeliefNetwork: Multi-agent belief management
"""

from .ontology import SoulMapOntology, OntologyDimension
from .beliefs import Belief, RecursiveBeliefState, BeliefNetwork

__all__ = [
    'SoulMapOntology',
    'OntologyDimension',
    'Belief',
    'RecursiveBeliefState',
    'BeliefNetwork',
]
