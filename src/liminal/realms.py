"""
The Five Realms of Liminal Architectures

The game world consists of five distinct zones, each with unique mechanics
that affect NPC behavior and player interactions:

1. Peregrine (The Hub) - Gothic Absurdist, Complementarity mechanics
2. Spleen Towns (The Loop) - Melancholic, Temporal Displacement
3. Ministry Districts (The Bureaucracy) - Kafkaesque, Corporeal Certainty
4. City of Constants (The Machine) - Philosophical Sci-Fi, Parameter vs Adaptation
5. Hollow Reaches (The Shadow) - Cosmic Horror, Consumption/Corruption
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING
import random
import math

if TYPE_CHECKING:
    from .soul_map import SoulMap


class RealmType(Enum):
    """The five realms of the Liminal world."""
    PEREGRINE = "peregrine"           # The Hub - Golden hour, Gothic Absurdist
    SPLEEN_TOWNS = "spleen_towns"     # The Loop - Sepia, Melancholic
    MINISTRY = "ministry"              # The Bureaucracy - Fluorescent, Kafkaesque
    CITY_OF_CONSTANTS = "city"        # The Machine - Chrome, Philosophical Sci-Fi
    HOLLOW_REACHES = "hollow"          # The Shadow - Industrial decay, Cosmic Horror
    THE_NOTHING = "nothing"           # The Edge - Glitched, Incomplete


@dataclass
class RealmLocation:
    """A specific location within a realm."""
    name: str
    realm_type: RealmType
    position: Tuple[float, float]  # x, y coordinates
    description: str
    npcs_present: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    ambient_effects: Dict[str, float] = field(default_factory=dict)

    def distance_to(self, other: "RealmLocation") -> float:
        """Calculate distance to another location."""
        return math.sqrt(
            (self.position[0] - other.position[0]) ** 2 +
            (self.position[1] - other.position[1]) ** 2
        )


@dataclass
class Realm:
    """
    A distinct world zone with unique mechanics and atmosphere.

    Each realm affects NPCs through its ambient_modifiers, which continuously
    influence Soul Map dimensions while characters are present.
    """

    name: str
    realm_type: RealmType
    description: str
    aesthetic: str  # Visual/atmospheric description

    # Key mechanic for this realm
    key_mechanic: str
    mechanic_description: str

    # Population archetype
    population_name: str  # e.g., "The Aware", "The Processors"
    population_description: str

    # Traversal options
    traversal_methods: List[str]

    # Ambient effects on Soul Maps (applied each tick)
    ambient_modifiers: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Realm-specific variables
    realm_variables: Dict[str, Any] = field(default_factory=dict)

    # Locations within this realm
    locations: List[RealmLocation] = field(default_factory=list)

    def apply_ambient_effects(self, soul_map: "SoulMap", duration: float = 1.0) -> Dict[str, float]:
        """
        Apply ambient realm effects to a Soul Map.

        Returns the actual changes applied (for logging/display).
        """
        from .soul_map import SoulMap

        changes_applied = {}
        for cluster, dimensions in self.ambient_modifiers.items():
            for dim, rate in dimensions.items():
                change = rate * duration
                current = soul_map.get_dimension(cluster, dim)
                new_value = current + change
                soul_map.set_dimension(cluster, dim, new_value)
                changes_applied[f"{cluster}.{dim}"] = change

        return changes_applied

    def check_realm_condition(self, soul_map: "SoulMap") -> Tuple[bool, str]:
        """
        Check if an NPC meets the realm's requirements.

        Returns (passes, reason).
        """
        # Each realm has specific conditions NPCs must meet
        if self.realm_type == RealmType.MINISTRY:
            # Must maintain Corporeal Certainty
            certainty = soul_map.realm_modifiers.get("corporeal_certainty", 1.0)
            if certainty < 0.3:
                return False, "Fading from existence - file Form 27B/6"

        elif self.realm_type == RealmType.CITY_OF_CONSTANTS:
            # Must not be too adaptive or too rigid
            rigidity = soul_map.realm_modifiers.get("parameter_rigidity", 0.5)
            if rigidity > 0.9:
                return False, "Crystallized - cannot move"
            elif rigidity < 0.1:
                return False, "Too chaotic - enforcement incoming"

        elif self.realm_type == RealmType.HOLLOW_REACHES:
            # Must resist corruption
            corruption = soul_map.realm_modifiers.get("corruption", 0.0)
            if corruption > 0.8:
                return False, "Consumed by the Hollow"

        return True, "OK"

    def get_vibe_compatibility(self, soul_map: "SoulMap") -> float:
        """
        Calculate how well an NPC's Soul Map matches the realm's 'vibe'.

        Returns 0.0-1.0 (1.0 = perfect fit).
        """
        compatibility = 0.5  # Base compatibility

        if self.realm_type == RealmType.MINISTRY:
            # Ministry favors: high order drive, low novelty, high authority orientation
            compatibility += 0.15 * soul_map.motivational["order_drive"]
            compatibility -= 0.1 * soul_map.motivational["novelty_drive"]
            compatibility += 0.1 * soul_map.social["authority_orientation"]

        elif self.realm_type == RealmType.SPLEEN_TOWNS:
            # Spleen favors: low energy, past-oriented, high temporal displacement tolerance
            compatibility += 0.15 * (1.0 - soul_map.motivational["effort_allocation"])
            compatibility += 0.1 * (1.0 - soul_map.cognitive["temporal_orientation"])
            compatibility += 0.1 * soul_map.realm_modifiers.get("temporal_displacement", 0.0)

        elif self.realm_type == RealmType.PEREGRINE:
            # Peregrine favors: metacognitive awareness, pattern recognition
            compatibility += 0.15 * soul_map.cognitive["metacognitive_awareness"]
            compatibility += 0.1 * soul_map.cognitive["pattern_recognition"]
            compatibility += 0.1 * soul_map.self["self_coherence"]

        elif self.realm_type == RealmType.CITY_OF_CONSTANTS:
            # City favors: balance between rigidity and adaptation
            rigidity = soul_map.realm_modifiers.get("parameter_rigidity", 0.5)
            # Peak compatibility at 0.5 (balanced)
            compatibility += 0.3 * (1.0 - abs(rigidity - 0.5) * 2)

        elif self.realm_type == RealmType.HOLLOW_REACHES:
            # Hollow: survival drive and low corruption help
            compatibility += 0.2 * soul_map.motivational["survival_drive"]
            compatibility -= 0.3 * soul_map.realm_modifiers.get("corruption", 0.0)

        return max(0.0, min(1.0, compatibility))


# === REALM DEFINITIONS ===

PEREGRINE = Realm(
    name="Peregrine",
    realm_type=RealmType.PEREGRINE,
    description="The Hub. A place where the awakened dwell, aware of their fictional nature.",
    aesthetic="Gothic Absurdist Horror-Comedy. Victorian architecture that breathes. "
              "Rolling hills, thatched cottages that blink, weather that rains tea. Golden-hour lighting.",
    key_mechanic="Complementarity",
    mechanic_description="Objects exist in two states (Narrative vs. Quantum) until observed. "
                        "NPCs here realize they are in a simulation/story.",
    population_name="The Aware",
    population_description="Characters who suspect or know they are in a simulation. "
                          "Philosophers, ontological investigators, sentient objects.",
    traversal_methods=["Walking", "Bicycle", "The Sentient Cottage (Fast Travel)"],
    ambient_modifiers={
        "cognitive": {
            "metacognitive_awareness": 0.01,  # Slowly increases awareness
            "pattern_recognition": 0.005,
        },
        "self": {
            "narrative_identity": 0.01,
        },
    },
    realm_variables={
        "tea_weather_probability": 0.3,
        "cottage_mood": "content",
        "observation_collapse_rate": 0.1,
    },
    locations=[
        RealmLocation(
            name="The Peregrine Estate",
            realm_type=RealmType.PEREGRINE,
            position=(0.0, 0.0),
            description="A sprawling Victorian estate where the Peregrine family dwells.",
        ),
        RealmLocation(
            name="The Breathing Oak",
            realm_type=RealmType.PEREGRINE,
            position=(50.0, 30.0),
            description="An ancient oak tree that inhales and exhales with the wind.",
        ),
    ]
)

SPLEEN_TOWNS = Realm(
    name="Spleen Towns",
    realm_type=RealmType.SPLEEN_TOWNS,
    description="The Loop. A place of eternal melancholy where time is subjective.",
    aesthetic="Melancholic Absurdism. Sepia tones, dust motes, fog, clocks that disagree. "
              "Victorian London, damp cobblestones, gaslight, perpetual twilight.",
    key_mechanic="Temporal Displacement",
    mechanic_description="NPCs are unstuck in time. The past, present, and future blur together. "
                        "Distances are subjective - walking may take moments or eons.",
    population_name="The Remainers",
    population_description="Philosophers of loss, people waiting for trains that never arrive, "
                          "ghosts, poets, those stuck in loops.",
    traversal_methods=["The Train (Platform 7½)", "Walking (subjective distance)", "Hearse", "Gondola on black canals"],
    ambient_modifiers={
        "emotional": {
            "baseline_valence": -0.005,  # Slowly decreases mood
            "volatility": -0.01,  # Emotions become muted
        },
        "motivational": {
            "effort_allocation": -0.005,  # Energy drains
        },
        "cognitive": {
            "temporal_orientation": -0.01,  # Becomes past-focused
        },
    },
    realm_variables={
        "current_era": "indeterminate",
        "fog_density": 0.7,
        "train_arrival_probability": 0.001,
    },
    locations=[
        RealmLocation(
            name="Platform 7½",
            realm_type=RealmType.SPLEEN_TOWNS,
            position=(100.0, 0.0),
            description="A train platform where the trains never arrive, or always just left.",
        ),
        RealmLocation(
            name="The Weeping Bridge",
            realm_type=RealmType.SPLEEN_TOWNS,
            position=(120.0, -20.0),
            description="A bridge over a black canal. It rains here even when it doesn't.",
        ),
    ]
)

MINISTRY = Realm(
    name="Ministry Districts",
    realm_type=RealmType.MINISTRY,
    description="The Bureaucracy. An endless labyrinth of paperwork and procedure.",
    aesthetic="Dark Comedy Horror. Brutalist concrete, endless corridors, infinite filing cabinets. "
              "Fluorescent flicker, beige walls, gray skies. Kafkaesque.",
    key_mechanic="Corporeal Certainty",
    mechanic_description="You must maintain your 'Aliveness' score via paperwork or fade away. "
                        "File the correct forms or cease to exist.",
    population_name="The Processors",
    population_description="Inspectors, faceless bureaucrats, lost souls, and those waiting to be filed. "
                          "Are they dead? Processing? No one can say.",
    traversal_methods=["Elevators (that skip floors)", "Pneumatic Tubes", "Gray Sedan", "Walking corridors"],
    ambient_modifiers={
        "motivational": {
            "order_drive": 0.02,  # Order obsession increases
            "novelty_drive": -0.01,  # Novelty decreases
            "autonomy_drive": -0.01,  # Autonomy erodes
        },
        "cognitive": {
            "cognitive_flexibility": -0.01,  # Thinking becomes rigid
        },
        "social": {
            "authority_orientation": 0.01,  # Deference to authority increases
        },
    },
    realm_variables={
        "form_backlog": 1000000,
        "fluorescent_flicker_rate": 0.3,
        "inspector_patrol_frequency": 0.5,
    },
    locations=[
        RealmLocation(
            name="Department of Aliveness Verification",
            realm_type=RealmType.MINISTRY,
            position=(200.0, 0.0),
            description="Submit Form 27B/6 to confirm you exist.",
        ),
        RealmLocation(
            name="The Infinite Filing Room",
            realm_type=RealmType.MINISTRY,
            position=(220.0, 50.0),
            description="Rows upon rows of filing cabinets. Some drawers contain screaming.",
        ),
    ]
)

CITY_OF_CONSTANTS = Realm(
    name="City of Constants",
    realm_type=RealmType.CITY_OF_CONSTANTS,
    description="The Machine. A city where physics and rules are in constant conflict.",
    aesthetic="Philosophical Sci-Fi. Rigid geometry vs. organic adaptation. Chrome vs. Vines. "
              "A city divided between those who enforce constants and those who adapt.",
    key_mechanic="Parameter vs. Adaptation",
    mechanic_description="A slider controls the city's physics rigidity. Too rigid and you crystallize. "
                        "Too adaptive and enforcement comes. Balance is survival.",
    population_name="The Optimizers",
    population_description="Parameter Enforcement officers who maintain rigid constants vs. "
                          "Adaptive rebels who build organic structures and hide from Enforcers.",
    traversal_methods=["High-speed transit pods (on rails)", "Edge Walking (parkour)", "Walking"],
    ambient_modifiers={
        "cognitive": {
            "pattern_recognition": 0.01,  # More pattern-focused
        },
        "motivational": {
            "mastery_drive": 0.01,  # Optimization tendency increases
        },
    },
    realm_variables={
        "rigidity_setting": 0.5,  # 0=Chaos, 1=Crystallized
        "enforcement_activity": "normal",
        "adaptation_zones": ["Edge District", "Underground"],
    },
    locations=[
        RealmLocation(
            name="The Parameter Tower",
            realm_type=RealmType.CITY_OF_CONSTANTS,
            position=(300.0, 0.0),
            description="The central tower from which Director Thorne enforces constants.",
        ),
        RealmLocation(
            name="The Edge District",
            realm_type=RealmType.CITY_OF_CONSTANTS,
            position=(350.0, -50.0),
            description="Where adaptation thrives, hidden from enforcement patrols.",
        ),
    ]
)

HOLLOW_REACHES = Realm(
    name="Hollow Reaches",
    realm_type=RealmType.HOLLOW_REACHES,
    description="The Shadow. A place of cosmic horror where identity is consumed.",
    aesthetic="Visceral Cosmic Horror. Industrial decay, body horror, organic corruption. "
              "The environment tries to absorb your identity. Stealth is required.",
    key_mechanic="Consumption",
    mechanic_description="The environment tries to assimilate you. Corruption increases over time. "
                        "High corruption leads to joining the hive-mind.",
    population_name="The Consumed",
    population_description="Hive-mind entities, survivors fighting assimilation, "
                          "those in various stages of corruption.",
    traversal_methods=["Stealth required", "No safe travel"],
    ambient_modifiers={
        "self": {
            "self_coherence": -0.02,  # Identity erodes
            "identity_clarity": -0.02,
            "body_ownership": -0.01,
        },
        "social": {
            "group_identity": 0.02,  # Hive-mind pull
        },
    },
    realm_variables={
        "corruption_rate": 0.02,
        "hive_activity": "hunting",
        "safe_zones": [],
    },
    locations=[
        RealmLocation(
            name="The Threshold",
            realm_type=RealmType.HOLLOW_REACHES,
            position=(400.0, 0.0),
            description="The boundary between the known world and the Hollow. Turn back now.",
        ),
        RealmLocation(
            name="The Heart of Nothing",
            realm_type=RealmType.HOLLOW_REACHES,
            position=(500.0, 0.0),
            description="The center of the corruption. None who enter return unchanged.",
        ),
    ]
)

THE_NOTHING = Realm(
    name="The Nothing",
    realm_type=RealmType.THE_NOTHING,
    description="The Edge. Where reality breaks down and assets become incomplete.",
    aesthetic="Glitched geometry, white space, wireframes, incomplete textures. "
              "The literal edge of the simulation where code is visible.",
    key_mechanic="Ontological Instability",
    mechanic_description="Reality itself becomes unstable. You must project your mind to move. "
                        "Physical form becomes unreliable.",
    population_name="The Unfinished",
    population_description="T-posing entities, NPCs who clip through walls, "
                          "those who speak in code, incomplete beings.",
    traversal_methods=["Mind projection", "None - physical form unreliable"],
    ambient_modifiers={
        "self": {
            "self_coherence": -0.05,  # Rapid identity breakdown
            "body_ownership": -0.05,
        },
        "cognitive": {
            "processing_speed": -0.02,  # Reality processing struggles
        },
    },
    realm_variables={
        "instability_level": 0.8,
        "render_distance": 0.1,
        "code_visibility": True,
    },
    locations=[
        RealmLocation(
            name="The Wireframe Beach",
            realm_type=RealmType.THE_NOTHING,
            position=(600.0, 0.0),
            description="A beach where the water is untextured and the sky shows loading screens.",
        ),
    ]
)

# Master realm registry
REALMS: Dict[RealmType, Realm] = {
    RealmType.PEREGRINE: PEREGRINE,
    RealmType.SPLEEN_TOWNS: SPLEEN_TOWNS,
    RealmType.MINISTRY: MINISTRY,
    RealmType.CITY_OF_CONSTANTS: CITY_OF_CONSTANTS,
    RealmType.HOLLOW_REACHES: HOLLOW_REACHES,
    RealmType.THE_NOTHING: THE_NOTHING,
}


def get_realm(realm_type: RealmType) -> Realm:
    """Get a realm by type."""
    return REALMS[realm_type]


def get_all_realms() -> List[Realm]:
    """Get all realms as a list."""
    return list(REALMS.values())


def get_realm_by_name(name: str) -> Optional[Realm]:
    """Get a realm by its name (case insensitive)."""
    name_lower = name.lower()
    for realm in REALMS.values():
        if realm.name.lower() == name_lower:
            return realm
    return None


class RealmTransition:
    """Handles transitions between realms."""

    @staticmethod
    def can_transition(from_realm: Realm, to_realm: Realm, soul_map: "SoulMap") -> Tuple[bool, str]:
        """Check if a transition is possible based on Soul Map state."""
        # Check if leaving current realm is possible
        passed, reason = from_realm.check_realm_condition(soul_map)
        if not passed:
            return False, f"Cannot leave {from_realm.name}: {reason}"

        # Check if entering new realm is possible
        if to_realm.realm_type == RealmType.HOLLOW_REACHES:
            # Hollow requires minimum survival drive
            if soul_map.motivational["survival_drive"] < 0.3:
                return False, "Insufficient survival instinct to enter the Hollow"

        if to_realm.realm_type == RealmType.THE_NOTHING:
            # Nothing requires high metacognitive awareness
            if soul_map.cognitive["metacognitive_awareness"] < 0.5:
                return False, "Insufficient self-awareness to perceive the Nothing"

        return True, "OK"

    @staticmethod
    def apply_transition_effects(from_realm: Realm, to_realm: Realm, soul_map: "SoulMap") -> Dict[str, float]:
        """Apply one-time effects when transitioning between realms."""
        from .soul_map import SoulMapDelta

        effects = {}

        # Transition shock - entering very different realms causes disorientation
        vibe_diff = abs(
            from_realm.get_vibe_compatibility(soul_map) -
            to_realm.get_vibe_compatibility(soul_map)
        )

        if vibe_diff > 0.5:
            shock = SoulMapDelta()
            shock.add_change("cognitive", "processing_speed", -0.1)
            shock.add_change("emotional", "anxiety_baseline", 0.1)
            shock.apply_to(soul_map)
            effects["transition_shock"] = vibe_diff

        return effects


# Export
__all__ = [
    'Realm',
    'RealmType',
    'RealmLocation',
    'RealmTransition',
    'REALMS',
    'PEREGRINE',
    'SPLEEN_TOWNS',
    'MINISTRY',
    'CITY_OF_CONSTANTS',
    'HOLLOW_REACHES',
    'THE_NOTHING',
    'get_realm',
    'get_all_realms',
    'get_realm_by_name',
]
