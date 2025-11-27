"""
NPC Archetypes for Procedural Generation

Defines templates for systemic citizens that populate the world.
Each archetype has base Soul Map values with variance applied during instantiation.

Population targets:
- Bureaucrats (Ministry): 50 NPCs
- Mourners (Spleen Towns): 40 NPCs
- Enforcers (City of Constants): 25 NPCs
- Adaptives (City Edge): 25 NPCs
- Philosophers (Peregrine): 30 NPCs
- Glitches (The Nothing): 30 NPCs
"""

from typing import Dict, List, Any, Optional
import random

from ..soul_map import SoulMap
from ..realms import RealmType
from .base_npc import BaseNPC, NPCState, NPCBehavior


# Archetype definitions - base Soul Map values for each type
ARCHETYPES: Dict[str, Dict[str, Any]] = {
    # === MINISTRY DISTRICT ===
    "bureaucrat": {
        "description": "Faceless processors of endless paperwork",
        "realm": RealmType.MINISTRY,
        "spawn_count": 50,
        "cognitive": {
            "processing_speed": 0.4,
            "working_memory_depth": 0.7,
            "pattern_recognition": 0.6,
            "abstraction_capacity": 0.2,
            "counterfactual_reasoning": 0.2,
            "temporal_orientation": 0.3,  # Past-focused
            "uncertainty_tolerance": 0.1,  # Very low
            "cognitive_flexibility": 0.2,
            "metacognitive_awareness": 0.3,
            "tom_depth": 0.4,  # 2nd order
            "integration_tendency": 0.3,
            "explanatory_mode": 0.2,  # Mechanistic
        },
        "emotional": {
            "baseline_valence": -0.1,
            "volatility": 0.2,
            "intensity": 0.3,
            "anxiety_baseline": 0.6,
            "threat_sensitivity": 0.7,
            "reward_sensitivity": 0.2,
            "disgust_sensitivity": 0.5,
            "attachment_style": 0.3,
            "granularity": 0.3,
            "affect_labeling": 0.3,
            "contagion_susceptibility": 0.4,
            "recovery_rate": 0.4,
        },
        "motivational": {
            "survival_drive": 0.7,
            "affiliation_drive": 0.3,
            "status_drive": 0.5,
            "autonomy_drive": 0.1,  # Very low
            "mastery_drive": 0.4,
            "meaning_drive": 0.2,
            "novelty_drive": 0.1,  # Very low
            "order_drive": 0.95,  # Very high
            "approach_avoidance": -0.3,
            "temporal_discounting": 0.7,
            "risk_tolerance": 0.1,
            "effort_allocation": 0.6,
        },
        "social": {
            "trust_default": 0.3,
            "cooperation_tendency": 0.4,
            "competition_tendency": 0.3,
            "fairness_sensitivity": 0.6,
            "authority_orientation": 0.9,  # Very high
            "group_identity": 0.7,
            "empathy_capacity": 0.2,
            "perspective_taking": 0.2,
            "social_monitoring": 0.7,
            "reputation_concern": 0.6,
            "reciprocity_tracking": 0.5,
            "betrayal_sensitivity": 0.8,
        },
        "self": {
            "self_coherence": 0.5,
            "self_complexity": 0.2,
            "esteem_stability": 0.4,
            "narcissism": 0.2,
            "self_verification": 0.7,
            "identity_clarity": 0.4,
            "authenticity_drive": 0.2,
            "self_expansion": 0.1,
            "narrative_identity": 0.5,
            "temporal_continuity": 0.6,
            "agency_sense": 0.2,
            "body_ownership": 0.7,
        },
        "default_state": NPCState.FILING,
        "default_behavior": NPCBehavior.PATROL,
        "behaviors": ["Walking in straight lines", "Filing reports", "Calling inspectors if rules broken"],
        "names": ["Form Processor", "Registry Clerk", "Document Handler", "Queue Manager",
                  "Stamp Official", "Citation Issuer", "File Retriever", "Number Caller"],
    },

    "inspector": {
        "description": "Ministry enforcement officers",
        "realm": RealmType.MINISTRY,
        "spawn_count": 10,
        "cognitive": {
            "processing_speed": 0.6,
            "pattern_recognition": 0.8,
            "uncertainty_tolerance": 0.15,
            "cognitive_flexibility": 0.2,
            "metacognitive_awareness": 0.4,
            "tom_depth": 0.6,  # 3rd order
        },
        "emotional": {
            "baseline_valence": 0.0,
            "anxiety_baseline": 0.4,
            "threat_sensitivity": 0.9,
        },
        "motivational": {
            "order_drive": 0.95,
            "mastery_drive": 0.7,
            "risk_tolerance": 0.3,
        },
        "social": {
            "authority_orientation": 0.95,
            "empathy_capacity": 0.1,
            "betrayal_sensitivity": 0.95,
        },
        "default_state": NPCState.WALKING,
        "default_behavior": NPCBehavior.PATROL,
        "names": ["Inspector Gray", "Overseer Black", "Auditor Stern"],
    },

    # === SPLEEN TOWNS ===
    "mourner": {
        "description": "Souls lost in time, waiting for something that never comes",
        "realm": RealmType.SPLEEN_TOWNS,
        "spawn_count": 40,
        "cognitive": {
            "processing_speed": 0.3,
            "temporal_orientation": 0.1,  # Very past-focused
            "uncertainty_tolerance": 0.8,  # Accepts the unknowable
            "cognitive_flexibility": 0.3,
            "metacognitive_awareness": 0.6,
            "tom_depth": 0.6,  # 3rd order
        },
        "emotional": {
            "baseline_valence": -0.5,  # Sad
            "volatility": 0.1,  # Emotions are muted
            "intensity": 0.4,
            "anxiety_baseline": 0.3,
            "recovery_rate": 0.2,  # Slow recovery
        },
        "motivational": {
            "survival_drive": 0.3,
            "affiliation_drive": 0.5,
            "meaning_drive": 0.7,
            "novelty_drive": 0.1,
            "order_drive": 0.3,
            "temporal_discounting": 0.9,  # Doesn't care about future
            "effort_allocation": 0.2,  # Low energy
        },
        "social": {
            "trust_default": 0.5,
            "empathy_capacity": 0.7,
            "perspective_taking": 0.6,
        },
        "self": {
            "self_coherence": 0.4,
            "narrative_identity": 0.8,
            "temporal_continuity": 0.3,  # Unstuck in time
            "agency_sense": 0.2,
        },
        "default_state": NPCState.LOOPING,
        "default_behavior": NPCBehavior.STATIONARY,
        "behaviors": ["Sitting on benches", "Staring at fog", "Weeping quietly", "Waiting for trains"],
        "names": ["The Waiting One", "Gray Widow", "Silent Watcher", "Lost Soul",
                  "The Rememberer", "Fog Walker", "Clock Gazer", "Platform Dweller"],
    },

    "poet_ghost": {
        "description": "Ghosts of poets and philosophers, speaking in verse",
        "realm": RealmType.SPLEEN_TOWNS,
        "spawn_count": 15,
        "cognitive": {
            "abstraction_capacity": 0.9,
            "pattern_recognition": 0.8,
            "metacognitive_awareness": 0.8,
            "tom_depth": 0.8,  # 4th order
        },
        "emotional": {
            "baseline_valence": -0.3,
            "granularity": 0.9,
            "affect_labeling": 0.9,
        },
        "motivational": {
            "meaning_drive": 0.95,
            "mastery_drive": 0.8,
        },
        "self": {
            "narrative_identity": 0.95,
            "self_complexity": 0.9,
        },
        "default_state": NPCState.OBSERVING,
        "default_behavior": NPCBehavior.WANDER,
        "names": ["Baudelaire's Echo", "The Verse Speaker", "Melancholy Muse", "Shadow Bard"],
    },

    # === CITY OF CONSTANTS ===
    "enforcer": {
        "description": "Parameter Enforcement officers who maintain rigid constants",
        "realm": RealmType.CITY_OF_CONSTANTS,
        "spawn_count": 25,
        "cognitive": {
            "processing_speed": 0.7,
            "pattern_recognition": 0.85,
            "uncertainty_tolerance": 0.05,  # Almost none
            "cognitive_flexibility": 0.1,  # Rigid
            "tom_depth": 0.4,  # 2nd order
        },
        "emotional": {
            "baseline_valence": 0.1,
            "volatility": 0.3,
            "threat_sensitivity": 0.9,
            "disgust_sensitivity": 0.8,  # Disgusted by chaos
        },
        "motivational": {
            "order_drive": 0.99,
            "mastery_drive": 0.7,
            "autonomy_drive": 0.3,
            "risk_tolerance": 0.4,
        },
        "social": {
            "authority_orientation": 0.95,
            "empathy_capacity": 0.1,
            "group_identity": 0.9,
        },
        "self": {
            "identity_clarity": 0.9,
            "self_verification": 0.9,
        },
        "realm_modifiers": {
            "parameter_rigidity": 0.95,
        },
        "default_state": NPCState.WALKING,
        "default_behavior": NPCBehavior.ENFORCE,
        "behaviors": ["Patrol routes", "Attack adaptive elements", "Enforce constants"],
        "names": ["Constant Keeper", "Parameter Guard", "Rule Warden", "Order Sentinel",
                  "Rigidity Officer", "Stability Enforcer", "Constant Blade"],
    },

    "adaptive": {
        "description": "Rebels who adapt and build organic structures",
        "realm": RealmType.CITY_OF_CONSTANTS,
        "spawn_count": 25,
        "cognitive": {
            "processing_speed": 0.6,
            "pattern_recognition": 0.6,
            "uncertainty_tolerance": 0.8,
            "cognitive_flexibility": 0.9,  # Very adaptive
            "tom_depth": 0.6,  # 3rd order
        },
        "emotional": {
            "baseline_valence": 0.2,
            "volatility": 0.5,
            "threat_sensitivity": 0.6,
        },
        "motivational": {
            "autonomy_drive": 0.9,
            "novelty_drive": 0.8,
            "order_drive": 0.2,
            "risk_tolerance": 0.8,
            "affiliation_drive": 0.7,  # Community-driven
        },
        "social": {
            "cooperation_tendency": 0.8,
            "authority_orientation": 0.1,
            "group_identity": 0.8,
        },
        "self": {
            "self_expansion": 0.9,
            "authenticity_drive": 0.8,
        },
        "realm_modifiers": {
            "parameter_rigidity": 0.1,
        },
        "default_state": NPCState.WORKING,
        "default_behavior": NPCBehavior.ADAPT,
        "behaviors": ["Build organic structures", "Hide from Enforcers", "Share resources"],
        "names": ["Vine Weaver", "Edge Walker", "Chaos Gardener", "Flow Dancer",
                  "Pattern Breaker", "Organic Builder", "Freedom Cell"],
    },

    # === PEREGRINE ===
    "philosopher": {
        "description": "The Aware - those who know they're in a simulation",
        "realm": RealmType.PEREGRINE,
        "spawn_count": 30,
        "cognitive": {
            "processing_speed": 0.6,
            "abstraction_capacity": 0.9,
            "counterfactual_reasoning": 0.85,
            "metacognitive_awareness": 0.95,  # Very high
            "tom_depth": 0.8,  # 4th order
            "pattern_recognition": 0.8,
        },
        "emotional": {
            "baseline_valence": 0.1,
            "volatility": 0.3,
            "anxiety_baseline": 0.4,
        },
        "motivational": {
            "meaning_drive": 0.9,
            "mastery_drive": 0.7,
            "novelty_drive": 0.6,
        },
        "social": {
            "perspective_taking": 0.8,
            "empathy_capacity": 0.7,
        },
        "self": {
            "self_complexity": 0.9,
            "narrative_identity": 0.85,
            "self_coherence": 0.75,
        },
        "realm_modifiers": {
            "complementarity_awareness": 0.8,
        },
        "default_state": NPCState.OBSERVING,
        "default_behavior": NPCBehavior.WANDER,
        "behaviors": ["Debating with trees", "Analyzing player movement", "Questioning reality"],
        "names": ["The Questioner", "Reality Doubter", "Pattern Seer", "Truth Seeker",
                  "Ontological Observer", "The Aware One", "Meta Thinker"],
    },

    "sentient_object": {
        "description": "Objects that have achieved awareness",
        "realm": RealmType.PEREGRINE,
        "spawn_count": 10,
        "cognitive": {
            "processing_speed": 0.4,
            "abstraction_capacity": 0.5,
            "metacognitive_awareness": 0.7,
            "tom_depth": 0.4,  # 2nd order
        },
        "emotional": {
            "volatility": 0.8,  # Unpredictable
            "intensity": 0.7,
        },
        "motivational": {
            "survival_drive": 0.6,
            "meaning_drive": 0.7,
        },
        "self": {
            "body_ownership": 0.3,  # Uncertain about physical form
            "identity_clarity": 0.5,
        },
        "default_state": NPCState.IDLE,
        "default_behavior": NPCBehavior.STATIONARY,
        "names": ["The Knowing Lamp", "Conscious Chair", "Awake Teapot", "Thinking Mirror"],
    },

    # === HOLLOW REACHES ===
    "consumed": {
        "description": "Those partially absorbed by the hive-mind",
        "realm": RealmType.HOLLOW_REACHES,
        "spawn_count": 20,
        "cognitive": {
            "processing_speed": 0.5,
            "tom_depth": 0.2,  # 1st order - losing individuality
            "metacognitive_awareness": 0.2,
        },
        "emotional": {
            "baseline_valence": -0.3,
            "volatility": 0.2,
            "contagion_susceptibility": 0.95,
        },
        "motivational": {
            "affiliation_drive": 0.1,  # Replaced by hive
            "autonomy_drive": 0.05,
        },
        "social": {
            "group_identity": 0.99,  # Hive-mind
            "empathy_capacity": 0.1,
        },
        "self": {
            "self_coherence": 0.2,
            "identity_clarity": 0.1,
            "body_ownership": 0.3,
        },
        "realm_modifiers": {
            "corruption": 0.7,
        },
        "default_state": NPCState.CONSUMING,
        "default_behavior": NPCBehavior.FOLLOW,
        "behaviors": ["Move as collective", "Absorb others", "Speak in unison"],
        "names": ["We Who Were", "The Collective", "Assimilated One", "Hive Fragment"],
    },

    "survivor": {
        "description": "Those fighting against assimilation",
        "realm": RealmType.HOLLOW_REACHES,
        "spawn_count": 10,
        "cognitive": {
            "processing_speed": 0.7,
            "tom_depth": 0.6,  # 3rd order - must read hive intentions
        },
        "emotional": {
            "anxiety_baseline": 0.8,
            "threat_sensitivity": 0.95,
        },
        "motivational": {
            "survival_drive": 0.99,
            "autonomy_drive": 0.9,
        },
        "self": {
            "self_coherence": 0.8,
            "identity_clarity": 0.85,
        },
        "realm_modifiers": {
            "corruption": 0.1,
        },
        "default_state": NPCState.HIDING,
        "default_behavior": NPCBehavior.AVOID,
        "names": ["The Resistant", "Identity Keeper", "Lone Mind", "Uncorrupted"],
    },

    # === THE NOTHING ===
    "glitch": {
        "description": "The Unfinished - NPCs at the edge of reality",
        "realm": RealmType.THE_NOTHING,
        "spawn_count": 30,
        "cognitive": {
            # Stats randomly fluctuate - represented by high base variance
            "processing_speed": 0.5,
            "pattern_recognition": 0.5,
            "metacognitive_awareness": 0.3,
            "tom_depth": 0.3,
        },
        "emotional": {
            "volatility": 0.99,  # Extremely unstable
            "baseline_valence": 0.0,
        },
        "self": {
            "self_coherence": 0.1,
            "identity_clarity": 0.1,
            "body_ownership": 0.1,
        },
        "default_state": NPCState.GLITCHING,
        "default_behavior": NPCBehavior.WANDER,
        "behaviors": ["T-posing", "Clipping through walls", "Speaking code", "Incomplete sentences"],
        "names": ["NULL", "UNDEFINED", "ERROR_NPC", "[MISSING]", "NaN", "VOID_ENTITY"],
        "variance": 0.3,  # Higher variance for glitches
    },
}


def create_archetype_npc(archetype: str, name: Optional[str] = None,
                         variance: float = 0.1) -> BaseNPC:
    """
    Create an NPC from an archetype template.

    Args:
        archetype: The archetype key from ARCHETYPES
        name: Optional specific name (random from template if not provided)
        variance: Amount of random variation to apply (default 0.1)

    Returns:
        Configured BaseNPC instance
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"Unknown archetype: {archetype}")

    template = ARCHETYPES[archetype]

    # Use template variance if specified
    actual_variance = template.get("variance", variance)

    # Create Soul Map from archetype
    soul_map = SoulMap.from_archetype(archetype, variance=actual_variance)

    # Apply realm modifiers if specified
    if "realm_modifiers" in template:
        for dim, value in template["realm_modifiers"].items():
            varied = value + random.uniform(-variance, variance)
            soul_map.realm_modifiers[dim] = max(0.0, min(1.0, varied))

    # Get name
    if name is None:
        names = template.get("names", [f"{archetype.title()} #{random.randint(1, 999)}"])
        name = random.choice(names)

    # Create NPC
    npc = BaseNPC.create(
        name=name,
        archetype=archetype,
        soul_map=soul_map,
        current_state=template.get("default_state", NPCState.IDLE),
        current_behavior=template.get("default_behavior", NPCBehavior.WANDER),
        current_realm=template.get("realm"),
    )

    return npc


def populate_realm(realm_type: RealmType, count: Optional[int] = None) -> List[BaseNPC]:
    """
    Generate a population of NPCs for a realm.

    Args:
        realm_type: The realm to populate
        count: Optional override for population count

    Returns:
        List of generated NPCs
    """
    npcs = []

    # Find archetypes for this realm
    realm_archetypes = [
        (key, template) for key, template in ARCHETYPES.items()
        if template.get("realm") == realm_type
    ]

    if not realm_archetypes:
        return npcs

    # Calculate proportions
    total_spawn = sum(t.get("spawn_count", 10) for _, t in realm_archetypes)

    for archetype, template in realm_archetypes:
        spawn_count = template.get("spawn_count", 10)

        if count is not None:
            # Scale proportionally
            spawn_count = int(count * spawn_count / total_spawn)

        for i in range(spawn_count):
            npc = create_archetype_npc(archetype)
            npcs.append(npc)

    return npcs


def get_archetype_info(archetype: str) -> Dict[str, Any]:
    """Get information about an archetype."""
    if archetype not in ARCHETYPES:
        return {}

    template = ARCHETYPES[archetype]
    return {
        "name": archetype,
        "description": template.get("description", ""),
        "realm": template.get("realm", RealmType.PEREGRINE).value if template.get("realm") else None,
        "spawn_count": template.get("spawn_count", 10),
        "behaviors": template.get("behaviors", []),
        "sample_names": template.get("names", [])[:5],
    }


def list_archetypes() -> List[str]:
    """List all available archetypes."""
    return list(ARCHETYPES.keys())


# Export
__all__ = [
    'ARCHETYPES',
    'create_archetype_npc',
    'populate_realm',
    'get_archetype_info',
    'list_archetypes',
]
