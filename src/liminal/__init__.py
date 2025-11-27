"""
LIMINAL ARCHITECTURES: Game Environment for ToM-NAS

An open-world psychological action-RPG environment where NPCs have 60-dimensional
Soul Maps representing their psychological states. This serves as the training
and evaluation environment for Neural Architecture Search agents developing
Theory of Mind capabilities.

Core Systems:
- Soul Map: 60-dimensional psychological ontology (cognitive, emotional, motivational, social, self)
- Realms: 5 distinct world zones with unique mechanics
- NPCs: Hero characters + procedurally generated systemic citizens
- Mechanics: Analysis loop, Intervention loop, Cognitive Hazards

Genre: "Grand Theft Ontology" - Explore, Observe, Predict, Alter
"""

from .soul_map import SoulMap, SoulMapCluster, DIMENSION_RANGES
from .realms import Realm, RealmType, REALMS
from .game_environment import LiminalEnvironment, GameState

__all__ = [
    'SoulMap',
    'SoulMapCluster',
    'DIMENSION_RANGES',
    'Realm',
    'RealmType',
    'REALMS',
    'LiminalEnvironment',
    'GameState',
]

__version__ = '1.0.0'
