"""
Cognitive Hazards - The Intervention System

Instead of traditional weapons, players in Liminal Architectures use
"Cognitive Hazards" - psychological interventions that modify NPC Soul Maps.

Types of interventions:
- Negative: Doubt, Fear, Paradox, Confusion
- Positive: Validation, Reassurance, Clarity, Empathy
- Neutral: Curiosity, Focus, Memory, Truth

Each hazard has:
- Target dimensions it affects
- Intensity (how much change)
- Duration (if temporary)
- Cost (player resource cost)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
import random

from ..soul_map import SoulMap, SoulMapDelta
from ..npcs.base_npc import BaseNPC


class HazardCategory(Enum):
    """Categories of cognitive hazards."""
    DESTABILIZING = "destabilizing"  # Reduces stability
    STABILIZING = "stabilizing"      # Increases stability
    EMOTIONAL = "emotional"          # Affects emotions
    COGNITIVE = "cognitive"          # Affects thinking
    SOCIAL = "social"               # Affects social behavior
    EXISTENTIAL = "existential"     # Affects sense of self


@dataclass
class CognitiveHazard:
    """
    A cognitive hazard that can be applied to NPCs.

    Hazards modify Soul Map dimensions to influence NPC behavior.
    """

    name: str
    description: str
    category: HazardCategory

    # The Soul Map changes this hazard causes
    delta: SoulMapDelta

    # Base intensity (1.0 = full effect)
    base_intensity: float = 1.0

    # Duration in game ticks (0 = permanent)
    duration: int = 0

    # Cost to use (0-1 scale, affects player resources)
    cost: float = 0.1

    # Minimum ToM depth required to use effectively
    required_tom_depth: int = 1

    # Does this hazard require line of sight?
    requires_los: bool = True

    # Maximum effective range
    max_range: float = 30.0

    # Cooldown in ticks
    cooldown: int = 10

    # Sound/visual effect identifier
    effect_id: str = "default_hazard"

    def apply(self, target: BaseNPC, intensity_modifier: float = 1.0) -> Dict[str, Any]:
        """
        Apply this hazard to a target NPC.

        Args:
            target: The NPC to affect
            intensity_modifier: Multiplier for effect strength (0-1)

        Returns:
            Report of changes made
        """
        # Calculate actual intensity
        actual_intensity = self.base_intensity * intensity_modifier

        # Create scaled delta
        scaled_delta = self.delta.scale(actual_intensity)

        # Apply to target
        result = target.apply_intervention(scaled_delta, source=f"hazard:{self.name}")

        # Record this in the result
        result["hazard_name"] = self.name
        result["intensity"] = actual_intensity
        result["duration"] = self.duration

        return result

    def can_affect(self, target: BaseNPC, player_position: Tuple[float, float],
                   player_tom: int) -> Tuple[bool, str]:
        """
        Check if this hazard can affect the target.

        Returns (can_affect, reason).
        """
        # Check ToM requirement
        if player_tom < self.required_tom_depth:
            return False, f"Requires ToM depth {self.required_tom_depth}"

        # Check range (simplified - would use actual distance in game)
        if self.requires_los:
            # Would check line of sight here
            pass

        # Some NPCs are resistant
        resistance = self._calculate_resistance(target)
        if resistance > 0.9:
            return False, "Target is highly resistant"

        return True, "OK"

    def _calculate_resistance(self, target: BaseNPC) -> float:
        """
        Calculate target's resistance to this hazard.

        Different hazards are resisted by different traits.
        """
        soul = target.soul_map

        if self.category == HazardCategory.DESTABILIZING:
            # Stability and self-coherence resist destabilization
            return (soul.self["self_coherence"] * 0.4 +
                   soul.compute_stability() * 0.4 +
                   soul.cognitive["metacognitive_awareness"] * 0.2)

        elif self.category == HazardCategory.EMOTIONAL:
            # Emotional regulation resists emotional manipulation
            return (soul.emotional["recovery_rate"] * 0.4 +
                   (1.0 - soul.emotional["contagion_susceptibility"]) * 0.4 +
                   soul.emotional["granularity"] * 0.2)

        elif self.category == HazardCategory.COGNITIVE:
            # Cognitive flexibility helps resist cognitive attacks
            return (soul.cognitive["cognitive_flexibility"] * 0.3 +
                   soul.cognitive["metacognitive_awareness"] * 0.3 +
                   soul.cognitive["uncertainty_tolerance"] * 0.2 +
                   soul.cognitive["processing_speed"] * 0.2)

        elif self.category == HazardCategory.SOCIAL:
            # Social intelligence resists social manipulation
            return (soul.social["social_monitoring"] * 0.4 +
                   soul.social["perspective_taking"] * 0.3 +
                   soul.social["betrayal_sensitivity"] * 0.3)

        elif self.category == HazardCategory.EXISTENTIAL:
            # Strong sense of self resists existential attacks
            return (soul.self["identity_clarity"] * 0.4 +
                   soul.self["self_coherence"] * 0.3 +
                   soul.self["narrative_identity"] * 0.3)

        return 0.5  # Default moderate resistance


# === HAZARD DEFINITIONS ===

def _create_doubt_delta() -> SoulMapDelta:
    """Create the Doubt hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("cognitive", "uncertainty_tolerance", -0.2)
    delta.add_change("self", "self_coherence", -0.15)
    delta.add_change("self", "agency_sense", -0.1)
    delta.add_change("emotional", "anxiety_baseline", 0.1)
    return delta


def _create_fear_delta() -> SoulMapDelta:
    """Create the Fear hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("emotional", "anxiety_baseline", 0.3)
    delta.add_change("emotional", "threat_sensitivity", 0.2)
    delta.add_change("motivational", "approach_avoidance", -0.3)
    delta.add_change("motivational", "survival_drive", 0.2)
    return delta


def _create_paradox_delta() -> SoulMapDelta:
    """Create the Paradox hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("cognitive", "cognitive_flexibility", -0.3)
    delta.add_change("cognitive", "processing_speed", -0.2)
    delta.add_change("self", "self_coherence", -0.2)
    delta.add_change("cognitive", "uncertainty_tolerance", -0.15)
    return delta


def _create_validation_delta() -> SoulMapDelta:
    """Create the Validation hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("self", "esteem_stability", 0.2)
    delta.add_change("emotional", "baseline_valence", 0.15)
    delta.add_change("emotional", "anxiety_baseline", -0.1)
    delta.add_change("social", "trust_default", 0.1)
    return delta


def _create_reassurance_delta() -> SoulMapDelta:
    """Create the Reassurance hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("emotional", "anxiety_baseline", -0.25)
    delta.add_change("emotional", "threat_sensitivity", -0.15)
    delta.add_change("social", "trust_default", 0.15)
    delta.add_change("self", "self_coherence", 0.1)
    return delta


def _create_curiosity_delta() -> SoulMapDelta:
    """Create the Curiosity hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("motivational", "novelty_drive", 0.25)
    delta.add_change("motivational", "approach_avoidance", 0.2)
    delta.add_change("cognitive", "uncertainty_tolerance", 0.15)
    delta.add_change("emotional", "anxiety_baseline", -0.05)
    return delta


def _create_confusion_delta() -> SoulMapDelta:
    """Create the Confusion hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("cognitive", "processing_speed", -0.25)
    delta.add_change("cognitive", "working_memory_depth", -0.2)
    delta.add_change("self", "identity_clarity", -0.15)
    delta.add_change("cognitive", "metacognitive_awareness", -0.1)
    return delta


def _create_clarity_delta() -> SoulMapDelta:
    """Create the Clarity hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("cognitive", "metacognitive_awareness", 0.2)
    delta.add_change("self", "identity_clarity", 0.2)
    delta.add_change("cognitive", "processing_speed", 0.1)
    delta.add_change("self", "self_coherence", 0.1)
    return delta


def _create_empathy_delta() -> SoulMapDelta:
    """Create the Empathy injection delta."""
    delta = SoulMapDelta()
    delta.add_change("social", "empathy_capacity", 0.25)
    delta.add_change("social", "perspective_taking", 0.2)
    delta.add_change("social", "cooperation_tendency", 0.15)
    delta.add_change("social", "trust_default", 0.1)
    return delta


def _create_truth_delta() -> SoulMapDelta:
    """Create the Truth hazard delta - reveals reality."""
    delta = SoulMapDelta()
    delta.add_change("cognitive", "metacognitive_awareness", 0.3)
    delta.add_change("realm_modifiers", "complementarity_awareness", 0.3)
    delta.add_change("self", "self_coherence", -0.1)  # Truth can destabilize
    return delta


def _create_obedience_delta() -> SoulMapDelta:
    """Create the Obedience hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("social", "authority_orientation", 0.3)
    delta.add_change("motivational", "autonomy_drive", -0.2)
    delta.add_change("self", "agency_sense", -0.15)
    return delta


def _create_rebellion_delta() -> SoulMapDelta:
    """Create the Rebellion hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("social", "authority_orientation", -0.3)
    delta.add_change("motivational", "autonomy_drive", 0.25)
    delta.add_change("self", "agency_sense", 0.2)
    delta.add_change("motivational", "risk_tolerance", 0.15)
    return delta


def _create_nostalgia_delta() -> SoulMapDelta:
    """Create the Nostalgia hazard delta."""
    delta = SoulMapDelta()
    delta.add_change("cognitive", "temporal_orientation", -0.3)  # Past-focused
    delta.add_change("emotional", "baseline_valence", -0.1)
    delta.add_change("self", "narrative_identity", 0.2)
    delta.add_change("realm_modifiers", "temporal_displacement", 0.2)
    return delta


def _create_corruption_delta() -> SoulMapDelta:
    """Create the Corruption hazard delta (Hollow Reaches special)."""
    delta = SoulMapDelta()
    delta.add_change("self", "self_coherence", -0.25)
    delta.add_change("self", "identity_clarity", -0.2)
    delta.add_change("social", "group_identity", 0.3)  # Hive pull
    delta.add_change("realm_modifiers", "corruption", 0.3)
    return delta


# === HAZARD REGISTRY ===

HAZARD_REGISTRY: Dict[str, CognitiveHazard] = {
    # Destabilizing hazards
    "doubt": CognitiveHazard(
        name="Doubt",
        description="Plants seeds of uncertainty. The target questions their own decisions.",
        category=HazardCategory.DESTABILIZING,
        delta=_create_doubt_delta(),
        base_intensity=1.0,
        cost=0.1,
        required_tom_depth=2,
        effect_id="doubt_ripple",
    ),
    "paradox": CognitiveHazard(
        name="Paradox",
        description="Confronts the target with a logical impossibility, causing cognitive freeze.",
        category=HazardCategory.COGNITIVE,
        delta=_create_paradox_delta(),
        base_intensity=1.0,
        cost=0.2,
        required_tom_depth=3,
        cooldown=20,
        effect_id="paradox_spiral",
    ),
    "confusion": CognitiveHazard(
        name="Confusion",
        description="Scrambles cognitive processing, making it hard to think clearly.",
        category=HazardCategory.COGNITIVE,
        delta=_create_confusion_delta(),
        base_intensity=1.0,
        duration=50,  # Temporary
        cost=0.15,
        required_tom_depth=2,
        effect_id="confusion_fog",
    ),

    # Emotional hazards
    "fear": CognitiveHazard(
        name="Fear",
        description="Induces acute fear response. Target likely to flee.",
        category=HazardCategory.EMOTIONAL,
        delta=_create_fear_delta(),
        base_intensity=1.0,
        duration=30,
        cost=0.15,
        required_tom_depth=1,
        effect_id="fear_shadow",
    ),
    "validation": CognitiveHazard(
        name="Validation",
        description="Affirms the target's sense of self. Builds trust.",
        category=HazardCategory.STABILIZING,
        delta=_create_validation_delta(),
        base_intensity=1.0,
        cost=0.1,
        required_tom_depth=2,
        effect_id="validation_glow",
    ),
    "reassurance": CognitiveHazard(
        name="Reassurance",
        description="Calms anxiety and reduces threat perception. Opens dialogue.",
        category=HazardCategory.STABILIZING,
        delta=_create_reassurance_delta(),
        base_intensity=1.0,
        cost=0.1,
        required_tom_depth=2,
        effect_id="reassurance_wave",
    ),

    # Motivational hazards
    "curiosity": CognitiveHazard(
        name="Curiosity",
        description="Sparks interest and desire to investigate. Makes NPCs approach.",
        category=HazardCategory.COGNITIVE,
        delta=_create_curiosity_delta(),
        base_intensity=1.0,
        cost=0.1,
        required_tom_depth=1,
        effect_id="curiosity_sparkle",
    ),

    # Self hazards
    "clarity": CognitiveHazard(
        name="Clarity",
        description="Enhances self-awareness and cognitive function. Can trigger awakening.",
        category=HazardCategory.STABILIZING,
        delta=_create_clarity_delta(),
        base_intensity=1.0,
        cost=0.15,
        required_tom_depth=3,
        effect_id="clarity_light",
    ),
    "truth": CognitiveHazard(
        name="Truth",
        description="Reveals the nature of reality. Dangerous to unstable minds.",
        category=HazardCategory.EXISTENTIAL,
        delta=_create_truth_delta(),
        base_intensity=1.0,
        cost=0.25,
        required_tom_depth=4,
        cooldown=50,
        effect_id="truth_reveal",
    ),

    # Social hazards
    "empathy": CognitiveHazard(
        name="Empathy Injection",
        description="Temporarily increases empathic capacity and cooperation.",
        category=HazardCategory.SOCIAL,
        delta=_create_empathy_delta(),
        base_intensity=1.0,
        duration=60,
        cost=0.2,
        required_tom_depth=3,
        effect_id="empathy_connection",
    ),
    "obedience": CognitiveHazard(
        name="Obedience",
        description="Increases deference to authority. Useful for bypassing enforcers.",
        category=HazardCategory.SOCIAL,
        delta=_create_obedience_delta(),
        base_intensity=1.0,
        duration=40,
        cost=0.15,
        required_tom_depth=2,
        effect_id="obedience_chain",
    ),
    "rebellion": CognitiveHazard(
        name="Rebellion",
        description="Sparks resistance to authority and rules. Useful against Ministry.",
        category=HazardCategory.SOCIAL,
        delta=_create_rebellion_delta(),
        base_intensity=1.0,
        duration=40,
        cost=0.15,
        required_tom_depth=2,
        effect_id="rebellion_fire",
    ),

    # Realm-specific hazards
    "nostalgia": CognitiveHazard(
        name="Nostalgia",
        description="Pulls the target's focus to the past. Strong in Spleen Towns.",
        category=HazardCategory.EMOTIONAL,
        delta=_create_nostalgia_delta(),
        base_intensity=1.0,
        cost=0.1,
        required_tom_depth=2,
        effect_id="nostalgia_sepia",
    ),
    "corruption": CognitiveHazard(
        name="Corruption",
        description="Accelerates the Hollow's influence. Dangerous weapon.",
        category=HazardCategory.EXISTENTIAL,
        delta=_create_corruption_delta(),
        base_intensity=1.0,
        cost=0.3,
        required_tom_depth=4,
        cooldown=100,
        effect_id="corruption_spread",
    ),
}


def apply_hazard(hazard_name: str, target: BaseNPC,
                 player_tom: int = 3,
                 intensity_modifier: float = 1.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to apply a hazard to a target.

    Args:
        hazard_name: Name of hazard from registry
        target: Target NPC
        player_tom: Player's ToM depth
        intensity_modifier: Intensity multiplier

    Returns:
        (success, result_dict)
    """
    if hazard_name not in HAZARD_REGISTRY:
        return False, {"error": f"Unknown hazard: {hazard_name}"}

    hazard = HAZARD_REGISTRY[hazard_name]

    # Check if can affect
    can_affect, reason = hazard.can_affect(target, (0, 0), player_tom)
    if not can_affect:
        return False, {"error": reason}

    # Calculate resistance
    resistance = hazard._calculate_resistance(target)

    # Apply with resistance factored in
    effective_intensity = intensity_modifier * (1.0 - resistance * 0.5)
    result = hazard.apply(target, effective_intensity)

    result["resistance"] = resistance
    result["effective_intensity"] = effective_intensity

    return True, result


def get_hazard(hazard_name: str) -> Optional[CognitiveHazard]:
    """Get a hazard by name."""
    return HAZARD_REGISTRY.get(hazard_name)


def list_hazards(category: Optional[HazardCategory] = None) -> List[str]:
    """List available hazards, optionally filtered by category."""
    if category is None:
        return list(HAZARD_REGISTRY.keys())

    return [
        name for name, hazard in HAZARD_REGISTRY.items()
        if hazard.category == category
    ]


# Export
__all__ = [
    'CognitiveHazard',
    'HazardCategory',
    'HAZARD_REGISTRY',
    'apply_hazard',
    'get_hazard',
    'list_hazards',
]
