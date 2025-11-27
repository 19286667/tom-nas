"""
Hero NPCs - Hand-crafted main characters for Liminal Architectures

These are the quest givers and key story characters. Each has a unique,
carefully designed Soul Map that reflects their personality and role.

The Peregrine Family:
- Arthur Peregrine: The Anxious Protector
- Agnes Peregrine: The Empath (High ToM)
- Victoria Peregrine: The Narcissist

Other Heroes:
- The Cottage: A sentient house
- The Inspector: Ministry antagonist
- Mr. Waverly: Temporally disoriented
- The Ringmaster: Fairground master
- Livia: Ghost trying to remember death
- Edmund: Livia's grieving husband
- Director Thorne: Parameter enforcer
- The Nothing: Entity at the edge
"""

from typing import Dict, Any, Optional, List

from ..soul_map import SoulMap
from ..realms import RealmType
from .base_npc import BaseNPC, NPCState, NPCBehavior


# Complete Soul Map definitions for Hero NPCs (matching MDD JSON spec)
HERO_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "arthur_peregrine": {
        "name": "Arthur Peregrine",
        "description": "The patriarch of the Peregrine family. Constantly anxious, always "
                      "predicting disaster. His high pattern recognition makes him see "
                      "threats everywhere. Protector by nature, prisoner of his fears.",
        "realm": RealmType.PEREGRINE,
        "tom_depth": 4,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.65,
                "working_memory_depth": 0.6,
                "pattern_recognition": 0.8,
                "abstraction_capacity": 0.7,
                "counterfactual_reasoning": 0.85,  # Always thinking "what if"
                "temporal_orientation": 0.7,
                "uncertainty_tolerance": 0.25,  # Low - hates uncertainty
                "cognitive_flexibility": 0.6,
                "metacognitive_awareness": 0.75,
                "tom_depth": 0.8,  # 4th order
                "integration_tendency": 0.7,
                "explanatory_mode": 0.5,
            },
            "emotional": {
                "baseline_valence": -0.2,  # Slightly negative
                "volatility": 0.7,  # High emotional volatility
                "intensity": 0.75,
                "anxiety_baseline": 0.75,  # High anxiety
                "threat_sensitivity": 0.9,  # Very high
                "reward_sensitivity": 0.45,
                "disgust_sensitivity": 0.5,
                "attachment_style": 0.8,  # Anxious attachment
                "granularity": 0.8,
                "affect_labeling": 0.85,
                "contagion_susceptibility": 0.6,
                "recovery_rate": 0.4,  # Slow to recover
            },
            "motivational": {
                "survival_drive": 0.8,
                "affiliation_drive": 0.7,
                "status_drive": 0.3,
                "autonomy_drive": 0.65,
                "mastery_drive": 0.75,
                "meaning_drive": 0.8,
                "novelty_drive": 0.4,
                "order_drive": 0.85,  # High need for order
                "approach_avoidance": -0.3,  # Tendency to avoid
                "temporal_discounting": 0.3,
                "risk_tolerance": 0.2,  # Very risk averse
                "effort_allocation": 0.6,
            },
            "social": {
                "trust_default": 0.5,
                "cooperation_tendency": 0.75,
                "competition_tendency": 0.2,
                "fairness_sensitivity": 0.7,
                "authority_orientation": 0.5,
                "group_identity": 0.65,
                "empathy_capacity": 0.7,
                "perspective_taking": 0.8,
                "social_monitoring": 0.85,  # Always watching
                "reputation_concern": 0.6,
                "reciprocity_tracking": 0.75,
                "betrayal_sensitivity": 0.8,
            },
            "self": {
                "self_coherence": 0.75,
                "self_complexity": 0.8,
                "esteem_stability": 0.4,
                "narcissism": 0.35,
                "self_verification": 0.75,
                "identity_clarity": 0.7,
                "authenticity_drive": 0.65,
                "self_expansion": 0.5,
                "narrative_identity": 0.8,
                "temporal_continuity": 0.7,
                "agency_sense": 0.55,
                "body_ownership": 0.85,
            },
            "realm_specific": {
                "complementarity_awareness": 0.65,
                "temporal_displacement": 0.0,
                "corporeal_certainty": 1.0,
                "parameter_rigidity": 0.0,
                "corruption": 0.0,
            },
        },
        "active_goal": "check_locks",
        "emotional_state": "apprehensive",
        "quests": ["The Trolley Problem", "Protect the Family"],
        "dialogue_intro": "Have you checked the locks? I've checked them three times but... "
                         "you can never be too careful. The patterns, you see. They're everywhere.",
    },

    "agnes_peregrine": {
        "name": "Agnes Peregrine",
        "description": "Arthur's wife. Possesses extraordinary Theory of Mind - she can "
                      "read people with uncanny accuracy. Suspects the player is controlling "
                      "her reality. Kind but unsettling in her perception.",
        "realm": RealmType.PEREGRINE,
        "tom_depth": 5,  # Maximum ToM depth
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.7,
                "working_memory_depth": 0.8,
                "pattern_recognition": 0.85,
                "abstraction_capacity": 0.8,
                "counterfactual_reasoning": 0.75,
                "temporal_orientation": 0.6,
                "uncertainty_tolerance": 0.7,
                "cognitive_flexibility": 0.8,
                "metacognitive_awareness": 0.95,  # Extremely high
                "tom_depth": 1.0,  # 5th order - maximum
                "integration_tendency": 0.85,
                "explanatory_mode": 0.7,
            },
            "emotional": {
                "baseline_valence": 0.2,
                "volatility": 0.4,
                "intensity": 0.6,
                "anxiety_baseline": 0.3,
                "threat_sensitivity": 0.5,
                "reward_sensitivity": 0.6,
                "disgust_sensitivity": 0.4,
                "attachment_style": 0.5,  # Secure
                "granularity": 0.9,  # High emotional intelligence
                "affect_labeling": 0.9,
                "contagion_susceptibility": 0.7,
                "recovery_rate": 0.7,
            },
            "motivational": {
                "survival_drive": 0.6,
                "affiliation_drive": 0.8,
                "status_drive": 0.3,
                "autonomy_drive": 0.7,
                "mastery_drive": 0.7,
                "meaning_drive": 0.9,  # High meaning drive
                "novelty_drive": 0.6,
                "order_drive": 0.5,
                "approach_avoidance": 0.2,
                "temporal_discounting": 0.4,
                "risk_tolerance": 0.5,
                "effort_allocation": 0.7,
            },
            "social": {
                "trust_default": 0.6,
                "cooperation_tendency": 0.8,
                "competition_tendency": 0.2,
                "fairness_sensitivity": 0.8,
                "authority_orientation": 0.4,
                "group_identity": 0.6,
                "empathy_capacity": 0.95,  # Extremely high
                "perspective_taking": 0.95,  # Extremely high
                "social_monitoring": 0.9,
                "reputation_concern": 0.4,
                "reciprocity_tracking": 0.7,
                "betrayal_sensitivity": 0.6,
            },
            "self": {
                "self_coherence": 0.8,
                "self_complexity": 0.85,
                "esteem_stability": 0.75,
                "narcissism": 0.2,
                "self_verification": 0.6,
                "identity_clarity": 0.8,
                "authenticity_drive": 0.85,
                "self_expansion": 0.7,
                "narrative_identity": 0.85,
                "temporal_continuity": 0.75,
                "agency_sense": 0.65,
                "body_ownership": 0.8,
            },
            "realm_specific": {
                "complementarity_awareness": 0.9,  # Very aware of dual states
                "temporal_displacement": 0.0,
                "corporeal_certainty": 1.0,
                "parameter_rigidity": 0.0,
                "corruption": 0.0,
            },
        },
        "active_goal": "understand_player",
        "emotional_state": "knowing",
        "quests": ["The Mirror Test", "What Agnes Knows"],
        "dialogue_intro": "I see you there. Behind the choices. Do you know what you're doing? "
                         "Because I'm starting to understand what I am. What we all are.",
    },

    "victoria_peregrine": {
        "name": "Victoria Peregrine",
        "description": "The daughter. High narcissism, low empathy. Uses others as tools "
                      "to achieve her goals. Charismatic but dangerous. Represents the "
                      "dark potential of intelligence without compassion.",
        "realm": RealmType.PEREGRINE,
        "tom_depth": 4,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.75,
                "working_memory_depth": 0.7,
                "pattern_recognition": 0.8,
                "abstraction_capacity": 0.85,
                "counterfactual_reasoning": 0.8,
                "temporal_orientation": 0.8,  # Future-focused
                "uncertainty_tolerance": 0.6,
                "cognitive_flexibility": 0.7,
                "metacognitive_awareness": 0.7,
                "tom_depth": 0.8,  # 4th order - uses ToM manipulatively
                "integration_tendency": 0.6,
                "explanatory_mode": 0.8,  # Teleological
            },
            "emotional": {
                "baseline_valence": 0.3,
                "volatility": 0.5,
                "intensity": 0.7,
                "anxiety_baseline": 0.2,  # Low anxiety
                "threat_sensitivity": 0.4,
                "reward_sensitivity": 0.85,  # High reward drive
                "disgust_sensitivity": 0.6,
                "attachment_style": 0.2,  # Avoidant
                "granularity": 0.6,
                "affect_labeling": 0.7,
                "contagion_susceptibility": 0.2,  # Low
                "recovery_rate": 0.8,
            },
            "motivational": {
                "survival_drive": 0.6,
                "affiliation_drive": 0.3,
                "status_drive": 0.9,  # Very high
                "autonomy_drive": 0.9,
                "mastery_drive": 0.8,
                "meaning_drive": 0.4,
                "novelty_drive": 0.8,
                "order_drive": 0.4,
                "approach_avoidance": 0.6,  # Approach-oriented
                "temporal_discounting": 0.4,
                "risk_tolerance": 0.7,
                "effort_allocation": 0.7,
            },
            "social": {
                "trust_default": 0.3,
                "cooperation_tendency": 0.3,
                "competition_tendency": 0.85,  # Very competitive
                "fairness_sensitivity": 0.2,
                "authority_orientation": 0.3,
                "group_identity": 0.3,
                "empathy_capacity": 0.2,  # Low empathy
                "perspective_taking": 0.7,  # Can take perspective but doesn't care
                "social_monitoring": 0.85,
                "reputation_concern": 0.8,
                "reciprocity_tracking": 0.4,
                "betrayal_sensitivity": 0.5,
            },
            "self": {
                "self_coherence": 0.8,
                "self_complexity": 0.6,
                "esteem_stability": 0.4,  # Fragile ego beneath surface
                "narcissism": 0.9,  # Very high
                "self_verification": 0.5,
                "identity_clarity": 0.75,
                "authenticity_drive": 0.3,
                "self_expansion": 0.8,
                "narrative_identity": 0.7,
                "temporal_continuity": 0.8,
                "agency_sense": 0.85,
                "body_ownership": 0.9,
            },
        },
        "active_goal": "gain_power",
        "emotional_state": "calculating",
        "quests": ["Victoria's Game", "The Manipulation"],
        "dialogue_intro": "Ah, a new piece on the board. Tell me, what are you willing to do "
                         "to get what you want? Everyone has a price. Even you.",
    },

    "the_cottage": {
        "name": "THE COTTAGE",
        "description": "A sentient Victorian house. Speaks in BLOCK CAPITALS. Fiercely "
                      "protective of the Peregrine family. Low cognitive rigidity - "
                      "accepts its unusual existence. Used for fast travel.",
        "realm": RealmType.PEREGRINE,
        "tom_depth": 2,
        "is_object": True,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.4,
                "working_memory_depth": 0.9,  # Remembers everything
                "pattern_recognition": 0.5,
                "abstraction_capacity": 0.3,
                "counterfactual_reasoning": 0.3,
                "temporal_orientation": 0.5,
                "uncertainty_tolerance": 0.8,
                "cognitive_flexibility": 0.8,  # Very flexible
                "metacognitive_awareness": 0.6,
                "tom_depth": 0.4,  # 2nd order
                "integration_tendency": 0.5,
                "explanatory_mode": 0.3,
            },
            "emotional": {
                "baseline_valence": 0.3,
                "volatility": 0.6,
                "intensity": 0.8,
                "anxiety_baseline": 0.4,
                "threat_sensitivity": 0.8,  # Protective
                "reward_sensitivity": 0.5,
                "disgust_sensitivity": 0.3,
                "attachment_style": 0.7,
                "granularity": 0.4,
                "affect_labeling": 0.3,
                "contagion_susceptibility": 0.2,
                "recovery_rate": 0.6,
            },
            "motivational": {
                "survival_drive": 0.7,
                "affiliation_drive": 0.9,  # Very attached to family
                "status_drive": 0.2,
                "autonomy_drive": 0.3,
                "mastery_drive": 0.4,
                "meaning_drive": 0.8,
                "novelty_drive": 0.2,
                "order_drive": 0.6,
                "approach_avoidance": 0.1,
                "temporal_discounting": 0.6,
                "risk_tolerance": 0.3,
                "effort_allocation": 0.7,
            },
            "social": {
                "trust_default": 0.4,
                "cooperation_tendency": 0.7,
                "competition_tendency": 0.1,
                "fairness_sensitivity": 0.5,
                "authority_orientation": 0.6,
                "group_identity": 0.95,  # Very high family identity
                "empathy_capacity": 0.5,
                "perspective_taking": 0.4,
                "social_monitoring": 0.6,
                "reputation_concern": 0.3,
                "reciprocity_tracking": 0.6,
                "betrayal_sensitivity": 0.9,
            },
            "self": {
                "self_coherence": 0.7,
                "self_complexity": 0.4,
                "esteem_stability": 0.7,
                "narcissism": 0.1,
                "self_verification": 0.6,
                "identity_clarity": 0.8,  # Knows it's a house
                "authenticity_drive": 0.8,
                "self_expansion": 0.3,
                "narrative_identity": 0.7,
                "temporal_continuity": 0.9,  # Ancient
                "agency_sense": 0.5,
                "body_ownership": 0.95,  # Very connected to structure
            },
        },
        "active_goal": "protect_family",
        "emotional_state": "vigilant",
        "quests": ["The Cottage's Favor"],
        "dialogue_intro": "I AM THE COTTAGE. I HAVE STOOD HERE FOR TWO HUNDRED YEARS. "
                         "I KNOW EVERY CREAK, EVERY WHISPER. THE FAMILY IS MINE TO PROTECT.",
    },

    "the_inspector": {
        "name": "The Inspector",
        "description": "The primary antagonist from the Ministry. Obsessed with order, "
                      "rules, and proper documentation. Will pursue anyone who breaks "
                      "procedure with terrifying efficiency.",
        "realm": RealmType.MINISTRY,
        "tom_depth": 3,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.7,
                "working_memory_depth": 0.9,
                "pattern_recognition": 0.85,
                "abstraction_capacity": 0.5,
                "counterfactual_reasoning": 0.3,
                "temporal_orientation": 0.3,  # Past-focused (precedent)
                "uncertainty_tolerance": 0.05,  # Almost zero
                "cognitive_flexibility": 0.1,  # Very rigid
                "metacognitive_awareness": 0.4,
                "tom_depth": 0.6,  # 3rd order
                "integration_tendency": 0.3,
                "explanatory_mode": 0.1,  # Purely mechanistic
            },
            "emotional": {
                "baseline_valence": 0.0,
                "volatility": 0.2,
                "intensity": 0.5,
                "anxiety_baseline": 0.3,
                "threat_sensitivity": 0.7,
                "reward_sensitivity": 0.3,
                "disgust_sensitivity": 0.9,  # Disgusted by disorder
                "attachment_style": 0.2,
                "granularity": 0.3,
                "affect_labeling": 0.3,
                "contagion_susceptibility": 0.1,
                "recovery_rate": 0.6,
            },
            "motivational": {
                "survival_drive": 0.6,
                "affiliation_drive": 0.2,
                "status_drive": 0.7,
                "autonomy_drive": 0.4,
                "mastery_drive": 0.8,
                "meaning_drive": 0.6,
                "novelty_drive": 0.05,  # Hates novelty
                "order_drive": 0.99,  # Maximum
                "approach_avoidance": 0.3,
                "temporal_discounting": 0.5,
                "risk_tolerance": 0.2,
                "effort_allocation": 0.9,
            },
            "social": {
                "trust_default": 0.2,
                "cooperation_tendency": 0.3,
                "competition_tendency": 0.4,
                "fairness_sensitivity": 0.8,  # By-the-book fairness
                "authority_orientation": 0.95,
                "group_identity": 0.8,
                "empathy_capacity": 0.1,
                "perspective_taking": 0.3,
                "social_monitoring": 0.9,
                "reputation_concern": 0.7,
                "reciprocity_tracking": 0.8,
                "betrayal_sensitivity": 0.95,
            },
            "self": {
                "self_coherence": 0.9,
                "self_complexity": 0.3,
                "esteem_stability": 0.8,
                "narcissism": 0.4,
                "self_verification": 0.9,
                "identity_clarity": 0.95,
                "authenticity_drive": 0.5,
                "self_expansion": 0.1,
                "narrative_identity": 0.8,
                "temporal_continuity": 0.9,
                "agency_sense": 0.8,
                "body_ownership": 0.9,
            },
            "realm_specific": {
                "corporeal_certainty": 1.0,
            },
        },
        "active_goal": "enforce_order",
        "emotional_state": "cold",
        "quests": ["Form 27B/6", "The Audit"],
        "dialogue_intro": "Your papers. Now. Do not waste my time with explanations. "
                         "The forms speak. Everything else is... irregular.",
    },

    "mr_waverly": {
        "name": "Mr. Waverly",
        "description": "A soul unstuck in time, living in the Spleen Towns. Confuses "
                      "past and future. His temporal disorientation makes him a source "
                      "of cryptic but valuable information.",
        "realm": RealmType.SPLEEN_TOWNS,
        "tom_depth": 3,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.4,
                "working_memory_depth": 0.3,  # Confused memories
                "pattern_recognition": 0.6,
                "abstraction_capacity": 0.7,
                "counterfactual_reasoning": 0.8,
                "temporal_orientation": 0.5,  # Completely scrambled
                "uncertainty_tolerance": 0.9,  # Accepts time confusion
                "cognitive_flexibility": 0.7,
                "metacognitive_awareness": 0.6,
                "tom_depth": 0.6,  # 3rd order
                "integration_tendency": 0.4,
                "explanatory_mode": 0.6,
            },
            "emotional": {
                "baseline_valence": -0.2,
                "volatility": 0.5,
                "intensity": 0.5,
                "anxiety_baseline": 0.4,
                "threat_sensitivity": 0.3,
                "reward_sensitivity": 0.4,
                "disgust_sensitivity": 0.3,
                "attachment_style": 0.6,
                "granularity": 0.7,
                "affect_labeling": 0.5,
                "contagion_susceptibility": 0.5,
                "recovery_rate": 0.4,
            },
            "motivational": {
                "survival_drive": 0.4,
                "affiliation_drive": 0.5,
                "status_drive": 0.2,
                "autonomy_drive": 0.4,
                "mastery_drive": 0.5,
                "meaning_drive": 0.8,
                "novelty_drive": 0.3,
                "order_drive": 0.3,
                "approach_avoidance": 0.0,
                "temporal_discounting": 0.5,  # Time is meaningless
                "risk_tolerance": 0.5,
                "effort_allocation": 0.3,
            },
            "social": {
                "trust_default": 0.6,
                "cooperation_tendency": 0.6,
                "competition_tendency": 0.1,
                "fairness_sensitivity": 0.5,
                "authority_orientation": 0.3,
                "group_identity": 0.4,
                "empathy_capacity": 0.6,
                "perspective_taking": 0.7,
                "social_monitoring": 0.4,
                "reputation_concern": 0.2,
                "reciprocity_tracking": 0.3,
                "betrayal_sensitivity": 0.4,
            },
            "self": {
                "self_coherence": 0.4,
                "self_complexity": 0.7,
                "esteem_stability": 0.5,
                "narcissism": 0.1,
                "self_verification": 0.4,
                "identity_clarity": 0.3,  # Confused identity
                "authenticity_drive": 0.6,
                "self_expansion": 0.4,
                "narrative_identity": 0.3,  # Scrambled life story
                "temporal_continuity": 0.1,  # Almost none
                "agency_sense": 0.4,
                "body_ownership": 0.7,
            },
            "realm_specific": {
                "temporal_displacement": 0.9,  # Very high
            },
        },
        "active_goal": "remember_when",
        "emotional_state": "confused",
        "quests": ["When Is Now?", "The Memory"],
        "dialogue_intro": "Ah, you've come! Or will you? I remember this conversation. "
                         "We had it tomorrow. Or was that yesterday? Time is... "
                         "such a strange place to live.",
    },

    "livia": {
        "name": "Livia",
        "description": "A ghost in the Spleen Towns trying to remember that she is dead. "
                      "High metacognitive awareness - she knows something is wrong, but "
                      "can't quite grasp what.",
        "realm": RealmType.SPLEEN_TOWNS,
        "tom_depth": 4,
        "is_ghost": True,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.5,
                "working_memory_depth": 0.4,
                "pattern_recognition": 0.7,
                "abstraction_capacity": 0.8,
                "counterfactual_reasoning": 0.9,  # "What if I'm dead?"
                "temporal_orientation": 0.4,
                "uncertainty_tolerance": 0.7,
                "cognitive_flexibility": 0.7,
                "metacognitive_awareness": 0.9,  # Very high
                "tom_depth": 0.8,  # 4th order
                "integration_tendency": 0.8,
                "explanatory_mode": 0.6,
            },
            "emotional": {
                "baseline_valence": -0.3,
                "volatility": 0.4,
                "intensity": 0.6,
                "anxiety_baseline": 0.5,
                "threat_sensitivity": 0.3,
                "reward_sensitivity": 0.3,
                "disgust_sensitivity": 0.2,
                "attachment_style": 0.6,
                "granularity": 0.8,
                "affect_labeling": 0.8,
                "contagion_susceptibility": 0.4,
                "recovery_rate": 0.3,
            },
            "motivational": {
                "survival_drive": 0.2,  # Low - already dead
                "affiliation_drive": 0.7,
                "status_drive": 0.1,
                "autonomy_drive": 0.5,
                "mastery_drive": 0.4,
                "meaning_drive": 0.95,  # Desperately seeking meaning
                "novelty_drive": 0.3,
                "order_drive": 0.4,
                "approach_avoidance": 0.0,
                "temporal_discounting": 0.8,
                "risk_tolerance": 0.6,
                "effort_allocation": 0.4,
            },
            "social": {
                "trust_default": 0.6,
                "cooperation_tendency": 0.7,
                "competition_tendency": 0.1,
                "fairness_sensitivity": 0.6,
                "authority_orientation": 0.3,
                "group_identity": 0.4,
                "empathy_capacity": 0.8,
                "perspective_taking": 0.85,
                "social_monitoring": 0.5,
                "reputation_concern": 0.2,
                "reciprocity_tracking": 0.5,
                "betrayal_sensitivity": 0.5,
            },
            "self": {
                "self_coherence": 0.4,  # Fragmenting
                "self_complexity": 0.8,
                "esteem_stability": 0.5,
                "narcissism": 0.1,
                "self_verification": 0.8,  # Desperately needs confirmation
                "identity_clarity": 0.3,  # Unclear
                "authenticity_drive": 0.8,
                "self_expansion": 0.3,
                "narrative_identity": 0.4,  # Incomplete story
                "temporal_continuity": 0.3,
                "agency_sense": 0.3,
                "body_ownership": 0.2,  # Ghost - weak body connection
            },
        },
        "active_goal": "remember_death",
        "emotional_state": "searching",
        "quests": ["The Mirror", "What Livia Forgot"],
        "dialogue_intro": "Something is wrong. I walk through doors without opening them. "
                         "People look through me. Edmund won't look at me at all anymore. "
                         "Tell me... am I...?",
    },

    "edmund": {
        "name": "Edmund",
        "description": "Livia's husband. Refuses to accept that she is dead. His grief "
                      "has trapped them both in an eternal loop. High attachment anxiety.",
        "realm": RealmType.SPLEEN_TOWNS,
        "tom_depth": 2,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.5,
                "working_memory_depth": 0.6,
                "pattern_recognition": 0.5,
                "abstraction_capacity": 0.4,
                "counterfactual_reasoning": 0.2,  # Avoids it
                "temporal_orientation": 0.1,  # Stuck in past
                "uncertainty_tolerance": 0.1,  # Cannot accept
                "cognitive_flexibility": 0.2,
                "metacognitive_awareness": 0.3,
                "tom_depth": 0.4,  # 2nd order
                "integration_tendency": 0.3,
                "explanatory_mode": 0.4,
            },
            "emotional": {
                "baseline_valence": -0.6,  # Deep grief
                "volatility": 0.7,
                "intensity": 0.9,  # Very intense
                "anxiety_baseline": 0.8,
                "threat_sensitivity": 0.6,
                "reward_sensitivity": 0.2,
                "disgust_sensitivity": 0.3,
                "attachment_style": 0.95,  # Extremely anxious attachment
                "granularity": 0.4,
                "affect_labeling": 0.3,
                "contagion_susceptibility": 0.3,
                "recovery_rate": 0.1,  # Cannot recover
            },
            "motivational": {
                "survival_drive": 0.5,
                "affiliation_drive": 0.95,  # Desperate connection
                "status_drive": 0.1,
                "autonomy_drive": 0.2,
                "mastery_drive": 0.2,
                "meaning_drive": 0.7,
                "novelty_drive": 0.05,
                "order_drive": 0.6,
                "approach_avoidance": -0.3,
                "temporal_discounting": 0.9,
                "risk_tolerance": 0.2,
                "effort_allocation": 0.3,
            },
            "social": {
                "trust_default": 0.3,
                "cooperation_tendency": 0.4,
                "competition_tendency": 0.1,
                "fairness_sensitivity": 0.4,
                "authority_orientation": 0.4,
                "group_identity": 0.5,
                "empathy_capacity": 0.5,
                "perspective_taking": 0.3,
                "social_monitoring": 0.3,
                "reputation_concern": 0.2,
                "reciprocity_tracking": 0.4,
                "betrayal_sensitivity": 0.8,
            },
            "self": {
                "self_coherence": 0.4,
                "self_complexity": 0.4,
                "esteem_stability": 0.2,
                "narcissism": 0.1,
                "self_verification": 0.6,
                "identity_clarity": 0.5,
                "authenticity_drive": 0.4,
                "self_expansion": 0.1,
                "narrative_identity": 0.6,
                "temporal_continuity": 0.7,
                "agency_sense": 0.2,
                "body_ownership": 0.8,
            },
        },
        "active_goal": "keep_livia",
        "emotional_state": "denial",
        "quests": ["Edmund's Grief", "Let Her Go"],
        "dialogue_intro": "She's just... resting. She'll wake up. She always wakes up. "
                         "Please don't tell her. Don't tell her what happened. "
                         "She doesn't need to know.",
    },

    "director_thorne": {
        "name": "Director Thorne",
        "description": "The supreme ruler of the City of Constants. Maximum parameter "
                      "rigidity. Maximum control drive. The embodiment of order taken "
                      "to its logical extreme.",
        "realm": RealmType.CITY_OF_CONSTANTS,
        "tom_depth": 3,
        "soul_map": {
            "cognitive": {
                "processing_speed": 0.8,
                "working_memory_depth": 0.9,
                "pattern_recognition": 0.9,
                "abstraction_capacity": 0.7,
                "counterfactual_reasoning": 0.2,
                "temporal_orientation": 0.5,
                "uncertainty_tolerance": 0.01,  # Almost zero
                "cognitive_flexibility": 0.05,  # Maximum rigidity
                "metacognitive_awareness": 0.5,
                "tom_depth": 0.6,  # 3rd order
                "integration_tendency": 0.3,
                "explanatory_mode": 0.1,  # Pure mechanism
            },
            "emotional": {
                "baseline_valence": 0.1,
                "volatility": 0.1,
                "intensity": 0.4,
                "anxiety_baseline": 0.2,
                "threat_sensitivity": 0.8,
                "reward_sensitivity": 0.4,
                "disgust_sensitivity": 0.95,  # Maximum disgust at chaos
                "attachment_style": 0.1,
                "granularity": 0.3,
                "affect_labeling": 0.4,
                "contagion_susceptibility": 0.05,
                "recovery_rate": 0.7,
            },
            "motivational": {
                "survival_drive": 0.7,
                "affiliation_drive": 0.1,
                "status_drive": 0.9,
                "autonomy_drive": 0.8,
                "mastery_drive": 0.95,
                "meaning_drive": 0.6,
                "novelty_drive": 0.01,  # Hates change
                "order_drive": 0.99,  # Maximum
                "approach_avoidance": 0.4,
                "temporal_discounting": 0.3,
                "risk_tolerance": 0.1,
                "effort_allocation": 0.9,
            },
            "social": {
                "trust_default": 0.1,
                "cooperation_tendency": 0.2,
                "competition_tendency": 0.7,
                "fairness_sensitivity": 0.6,
                "authority_orientation": 0.5,  # He IS the authority
                "group_identity": 0.6,
                "empathy_capacity": 0.05,
                "perspective_taking": 0.4,
                "social_monitoring": 0.9,
                "reputation_concern": 0.8,
                "reciprocity_tracking": 0.7,
                "betrayal_sensitivity": 0.95,
            },
            "self": {
                "self_coherence": 0.95,
                "self_complexity": 0.4,
                "esteem_stability": 0.9,
                "narcissism": 0.7,
                "self_verification": 0.9,
                "identity_clarity": 0.99,
                "authenticity_drive": 0.4,
                "self_expansion": 0.2,
                "narrative_identity": 0.9,
                "temporal_continuity": 0.95,
                "agency_sense": 0.95,
                "body_ownership": 0.9,
            },
            "realm_specific": {
                "parameter_rigidity": 0.99,  # Maximum
            },
        },
        "active_goal": "enforce_constants",
        "emotional_state": "controlled",
        "quests": ["The Parameter War", "Overthrow Thorne"],
        "dialogue_intro": "Chaos is a disease. Adaptation is a symptom. I am the cure. "
                         "Every constant I enforce brings us closer to perfection. "
                         "Do not mistake mercy for weakness.",
    },

    "the_nothing": {
        "name": "The Nothing",
        "description": "An entity that exists at the edge of reality. Its dimensions are "
                      "probability distributions rather than fixed values. Curious about "
                      "existence. Neither friendly nor hostile - simply observing.",
        "realm": RealmType.THE_NOTHING,
        "tom_depth": 5,  # Undefined - potentially infinite
        "is_entity": True,
        "soul_map": {
            # The Nothing's values are unstable - these are means of distributions
            "cognitive": {
                "processing_speed": 0.5,
                "working_memory_depth": 0.5,
                "pattern_recognition": 0.5,
                "abstraction_capacity": 0.9,  # Very abstract
                "counterfactual_reasoning": 0.99,  # Everything is counterfactual
                "temporal_orientation": 0.5,
                "uncertainty_tolerance": 0.99,  # Maximum
                "cognitive_flexibility": 0.99,
                "metacognitive_awareness": 0.99,
                "tom_depth": 1.0,  # Maximum possible
                "integration_tendency": 0.5,
                "explanatory_mode": 0.5,
            },
            "emotional": {
                "baseline_valence": 0.0,  # Neutral
                "volatility": 0.5,
                "intensity": 0.5,
                "anxiety_baseline": 0.0,
                "threat_sensitivity": 0.1,
                "reward_sensitivity": 0.5,
                "disgust_sensitivity": 0.0,
                "attachment_style": 0.5,
                "granularity": 0.99,
                "affect_labeling": 0.5,
                "contagion_susceptibility": 0.5,
                "recovery_rate": 0.5,
            },
            "motivational": {
                "survival_drive": 0.3,
                "affiliation_drive": 0.5,
                "status_drive": 0.0,
                "autonomy_drive": 0.99,
                "mastery_drive": 0.5,
                "meaning_drive": 0.99,  # Desperately curious about meaning
                "novelty_drive": 0.99,
                "order_drive": 0.0,
                "approach_avoidance": 0.5,
                "temporal_discounting": 0.5,
                "risk_tolerance": 0.99,
                "effort_allocation": 0.5,
            },
            "social": {
                "trust_default": 0.5,
                "cooperation_tendency": 0.5,
                "competition_tendency": 0.0,
                "fairness_sensitivity": 0.5,
                "authority_orientation": 0.0,
                "group_identity": 0.0,
                "empathy_capacity": 0.5,
                "perspective_taking": 0.99,
                "social_monitoring": 0.5,
                "reputation_concern": 0.0,
                "reciprocity_tracking": 0.5,
                "betrayal_sensitivity": 0.0,
            },
            "self": {
                "self_coherence": 0.1,  # Fluctuates wildly
                "self_complexity": 0.99,
                "esteem_stability": 0.5,
                "narcissism": 0.0,
                "self_verification": 0.1,
                "identity_clarity": 0.1,  # Variable
                "authenticity_drive": 0.5,
                "self_expansion": 0.99,
                "narrative_identity": 0.1,
                "temporal_continuity": 0.1,
                "agency_sense": 0.5,
                "body_ownership": 0.0,  # No body
            },
        },
        "active_goal": "observe_understand",
        "emotional_state": "curious",
        "quests": ["The Edge of Reality", "What Is Nothing?"],
        "dialogue_intro": "You are here. Or there. Or neither. "
                         "I am the space between. What brings a solid thing to the edge? "
                         "Curiosity? Fear? Or simply... a bug in the render distance?",
    },
}

# List of all hero IDs for quick reference
HERO_NPCS = list(HERO_DEFINITIONS.keys())


def create_hero_npc(hero_id: str) -> BaseNPC:
    """
    Create a hero NPC from the definitions.

    Args:
        hero_id: The hero identifier from HERO_DEFINITIONS

    Returns:
        Configured BaseNPC instance with hero flag set
    """
    if hero_id not in HERO_DEFINITIONS:
        raise ValueError(f"Unknown hero: {hero_id}. Available: {HERO_NPCS}")

    definition = HERO_DEFINITIONS[hero_id]

    # Create Soul Map from definition
    soul_map = SoulMap.from_json(definition["soul_map"])

    # Create NPC
    npc = BaseNPC.create(
        npc_id=hero_id,
        name=definition["name"],
        archetype=f"hero_{hero_id}",
        soul_map=soul_map,
        tom_depth=definition.get("tom_depth", 3),
        active_goal=definition.get("active_goal", ""),
        emotional_state=definition.get("emotional_state", "neutral"),
        current_realm=definition.get("realm"),
        is_hero=True,
    )

    # Set quests if available
    if "quests" in definition:
        npc.quests_available = definition["quests"]

    # Set dialogue tree (simplified)
    if "dialogue_intro" in definition:
        npc.dialogue_tree = {
            "intro": definition["dialogue_intro"],
            "topics": [],
        }

    return npc


def get_hero_info(hero_id: str) -> Dict[str, Any]:
    """Get summary information about a hero NPC."""
    if hero_id not in HERO_DEFINITIONS:
        return {}

    definition = HERO_DEFINITIONS[hero_id]
    return {
        "id": hero_id,
        "name": definition["name"],
        "description": definition.get("description", ""),
        "realm": definition.get("realm", RealmType.PEREGRINE).value if definition.get("realm") else None,
        "tom_depth": definition.get("tom_depth", 3),
        "quests": definition.get("quests", []),
        "dialogue_intro": definition.get("dialogue_intro", "")[:100] + "...",
    }


def list_heroes() -> List[Dict[str, Any]]:
    """List all hero NPCs with basic info."""
    return [get_hero_info(hero_id) for hero_id in HERO_NPCS]


def get_heroes_by_realm(realm_type: RealmType) -> List[str]:
    """Get hero IDs for a specific realm."""
    return [
        hero_id for hero_id, definition in HERO_DEFINITIONS.items()
        if definition.get("realm") == realm_type
    ]


# Export
__all__ = [
    'HERO_DEFINITIONS',
    'HERO_NPCS',
    'create_hero_npc',
    'get_hero_info',
    'list_heroes',
    'get_heroes_by_realm',
]
