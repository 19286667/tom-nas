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
- Psychosocial Co-Evolution: Bidirectional evolution of agents and environment
- Narrative Emergence: Meaningful stories from emergent dynamics

Genre: "Grand Theft Ontology" - Explore, Observe, Predict, Alter

Scientific Grounding:
The system draws from established cognitive science and social psychology:
- Dunbar's Social Brain Hypothesis (relationship limits)
- Heider's Balance Theory (triadic dynamics)
- Axelrod's Evolution of Cooperation (reputation/reciprocity)
- Frith & Frith's ToM neuroscience (nested belief modeling)
"""

from .game_environment import GameState, LiminalEnvironment

# Narrative Emergence System
from .narrative_emergence import (
    EmergentNarrative,
    NarrativeArchetype,
    NarrativeDetector,
    NarrativeEmergenceSystem,
)

# Psychosocial Co-Evolution System
from .psychosocial_coevolution import (
    BeliefPropagationEngine,
    EnvironmentEvolutionStrategy,
    PsychosocialCoevolutionEngine,
    RelationshipType,
    SocialEdge,
    SocialNetwork,
    TheoreticalConstants,
)
from .realms import REALMS, Realm, RealmType
from .soul_map import DIMENSION_RANGES, SoulMap, SoulMapCluster

__all__ = [
    # Core Systems
    "SoulMap",
    "SoulMapCluster",
    "DIMENSION_RANGES",
    "Realm",
    "RealmType",
    "REALMS",
    "LiminalEnvironment",
    "GameState",
    # Psychosocial Co-Evolution
    "PsychosocialCoevolutionEngine",
    "EnvironmentEvolutionStrategy",
    "SocialNetwork",
    "SocialEdge",
    "RelationshipType",
    "BeliefPropagationEngine",
    "TheoreticalConstants",
    # Narrative Emergence
    "NarrativeEmergenceSystem",
    "NarrativeArchetype",
    "EmergentNarrative",
    "NarrativeDetector",
]

__version__ = "2.0.0"  # Major version bump for co-evolution addition
