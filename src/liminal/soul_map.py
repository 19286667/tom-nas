"""
Soul Map: 60-Dimensional Psychological Ontology for Liminal NPCs

The Soul Map is the core data structure representing an NPC's complete psychological
state. It consists of 5 clusters with 12 dimensions each:

1. Cognitive (12 dims): Processing, reasoning, metacognition
2. Emotional (12 dims): Affect, sensitivity, regulation
3. Motivational (12 dims): Drives, goals, risk orientation
4. Social (12 dims): Trust, cooperation, social intelligence
5. Self (12 dims): Identity, coherence, agency

Additionally, realm-specific modifiers (5 dims) track how the environment
affects the NPC's psychological state.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import random


class SoulMapCluster(Enum):
    """The five psychological clusters of the Soul Map."""

    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    MOTIVATIONAL = "motivational"
    SOCIAL = "social"
    SELF = "self"


# Dimension definitions for each cluster
COGNITIVE_DIMENSIONS = [
    "processing_speed",  # 0.0-1.0: How quickly they process information
    "working_memory_depth",  # 0.0-1.0: How much context they can hold
    "pattern_recognition",  # 0.0-1.0: Ability to detect patterns
    "abstraction_capacity",  # 0.0-1.0: Abstract thinking ability
    "counterfactual_reasoning",  # 0.0-1.0: "What if" thinking
    "temporal_orientation",  # 0.0-1.0: 0=Past-focused, 1=Future-focused
    "uncertainty_tolerance",  # 0.0-1.0: Comfort with ambiguity
    "cognitive_flexibility",  # 0.0-1.0: Ability to adapt thinking
    "metacognitive_awareness",  # 0.0-1.0: Thinking about thinking
    "tom_depth",  # 1-5 (normalized to 0.0-1.0): Theory of Mind depth
    "integration_tendency",  # 0.0-1.0: Tendency to synthesize information
    "explanatory_mode",  # 0.0-1.0: 0=Mechanistic, 1=Teleological
]

EMOTIONAL_DIMENSIONS = [
    "baseline_valence",  # -1.0 to 1.0: Default emotional state
    "volatility",  # 0.0-1.0: How quickly emotions change
    "intensity",  # 0.0-1.0: Strength of emotional responses
    "anxiety_baseline",  # 0.0-1.0: Default anxiety level
    "threat_sensitivity",  # 0.0-1.0: Sensitivity to threats
    "reward_sensitivity",  # 0.0-1.0: Sensitivity to rewards
    "disgust_sensitivity",  # 0.0-1.0: Sensitivity to disgust triggers
    "attachment_style",  # 0.0-1.0: 0=Avoidant, 1=Anxious
    "granularity",  # 0.0-1.0: Emotional differentiation
    "affect_labeling",  # 0.0-1.0: Ability to name emotions
    "contagion_susceptibility",  # 0.0-1.0: Emotional contagion
    "recovery_rate",  # 0.0-1.0: How quickly they recover
]

MOTIVATIONAL_DIMENSIONS = [
    "survival_drive",  # 0.0-1.0: Self-preservation instinct
    "affiliation_drive",  # 0.0-1.0: Need for social connection
    "status_drive",  # 0.0-1.0: Need for status/recognition
    "autonomy_drive",  # 0.0-1.0: Need for independence
    "mastery_drive",  # 0.0-1.0: Need for competence
    "meaning_drive",  # 0.0-1.0: Need for purpose
    "novelty_drive",  # 0.0-1.0: Need for new experiences
    "order_drive",  # 0.0-1.0: Need for structure
    "approach_avoidance",  # -1.0 to 1.0: -1=Avoid, 1=Approach
    "temporal_discounting",  # 0.0-1.0: Future vs immediate reward pref
    "risk_tolerance",  # 0.0-1.0: Comfort with risk
    "effort_allocation",  # 0.0-1.0: Willingness to expend effort
]

SOCIAL_DIMENSIONS = [
    "trust_default",  # 0.0-1.0: Default trust level
    "cooperation_tendency",  # 0.0-1.0: Tendency to cooperate
    "competition_tendency",  # 0.0-1.0: Tendency to compete
    "fairness_sensitivity",  # 0.0-1.0: Sensitivity to fairness
    "authority_orientation",  # 0.0-1.0: Deference to authority
    "group_identity",  # 0.0-1.0: Strength of group identification
    "empathy_capacity",  # 0.0-1.0: Ability to feel others' emotions
    "perspective_taking",  # 0.0-1.0: Ability to see others' viewpoints
    "social_monitoring",  # 0.0-1.0: Awareness of social dynamics
    "reputation_concern",  # 0.0-1.0: Care about reputation
    "reciprocity_tracking",  # 0.0-1.0: Memory for social exchanges
    "betrayal_sensitivity",  # 0.0-1.0: Sensitivity to betrayal
]

SELF_DIMENSIONS = [
    "self_coherence",  # 0.0-1.0: Stability of self-concept
    "self_complexity",  # 0.0-1.0: Richness of self-understanding
    "esteem_stability",  # 0.0-1.0: Stability of self-esteem
    "narcissism",  # 0.0-1.0: Self-focus/grandiosity
    "self_verification",  # 0.0-1.0: Need for self-consistency
    "identity_clarity",  # 0.0-1.0: Clarity of identity
    "authenticity_drive",  # 0.0-1.0: Need to be genuine
    "self_expansion",  # 0.0-1.0: Drive to grow/change
    "narrative_identity",  # 0.0-1.0: Coherence of life story
    "temporal_continuity",  # 0.0-1.0: Sense of continuity over time
    "agency_sense",  # 0.0-1.0: Sense of personal control
    "body_ownership",  # 0.0-1.0: Connection to physical form
]

# Realm-specific dimensions (modified by environment)
REALM_DIMENSIONS = [
    "complementarity_awareness",  # Peregrine: Awareness of dual states
    "temporal_displacement",  # Spleen Towns: Unstuck in time
    "corporeal_certainty",  # Ministry: Certainty of being alive
    "parameter_rigidity",  # City of Constants: Adherence to rules
    "corruption",  # Hollow Reaches: Degree of assimilation
]

# Dimension range specifications
DIMENSION_RANGES = {
    "baseline_valence": (-1.0, 1.0),
    "approach_avoidance": (-1.0, 1.0),
    "tom_depth": (0.2, 1.0),  # 1-5 normalized
}

# Default range for most dimensions
DEFAULT_RANGE = (0.0, 1.0)


@dataclass
class SoulMap:
    """
    60-dimensional psychological state representation for NPCs.

    The Soul Map encapsulates the complete psychological profile of an NPC,
    enabling rich behavioral simulation and Theory of Mind reasoning.
    """

    # Core psychological clusters (each 12 dimensions)
    cognitive: Dict[str, float] = field(default_factory=dict)
    emotional: Dict[str, float] = field(default_factory=dict)
    motivational: Dict[str, float] = field(default_factory=dict)
    social: Dict[str, float] = field(default_factory=dict)
    self: Dict[str, float] = field(default_factory=dict)

    # Realm-specific modifiers
    realm_modifiers: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with default values if not provided."""
        if not self.cognitive:
            self.cognitive = {dim: 0.5 for dim in COGNITIVE_DIMENSIONS}
        if not self.emotional:
            self.emotional = {dim: 0.5 for dim in EMOTIONAL_DIMENSIONS}
            self.emotional["baseline_valence"] = 0.0
        if not self.motivational:
            self.motivational = {dim: 0.5 for dim in MOTIVATIONAL_DIMENSIONS}
            self.motivational["approach_avoidance"] = 0.0
        if not self.social:
            self.social = {dim: 0.5 for dim in SOCIAL_DIMENSIONS}
        if not self.self:
            self.self = {dim: 0.5 for dim in SELF_DIMENSIONS}
        if not self.realm_modifiers:
            self.realm_modifiers = {dim: 0.0 for dim in REALM_DIMENSIONS}
            self.realm_modifiers["corporeal_certainty"] = 1.0

    @classmethod
    def from_archetype(cls, archetype: str, variance: float = 0.1) -> "SoulMap":
        """Create a Soul Map from a predefined archetype with random variance."""
        from .npcs.archetypes import ARCHETYPES

        if archetype not in ARCHETYPES:
            raise ValueError(f"Unknown archetype: {archetype}")

        base = ARCHETYPES[archetype]
        soul_map = cls()

        # Apply archetype values with variance
        for cluster_name in ["cognitive", "emotional", "motivational", "social", "self"]:
            cluster = getattr(soul_map, cluster_name)
            archetype_cluster = base.get(cluster_name, {})

            for dim, value in cluster.items():
                if dim in archetype_cluster:
                    base_value = archetype_cluster[dim]
                else:
                    base_value = value

                # Add random variance
                dim_range = DIMENSION_RANGES.get(dim, DEFAULT_RANGE)
                varied = base_value + random.uniform(-variance, variance)
                cluster[dim] = max(dim_range[0], min(dim_range[1], varied))

        return soul_map

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "SoulMap":
        """Create a Soul Map from JSON data (as per MDD spec)."""
        soul_map = cls()

        if "soul_map" in json_data:
            json_data = json_data["soul_map"]

        for cluster_name in ["cognitive", "emotional", "motivational", "social", "self"]:
            if cluster_name in json_data:
                cluster = getattr(soul_map, cluster_name)
                for dim, value in json_data[cluster_name].items():
                    if dim in cluster:
                        cluster[dim] = float(value)

        if "realm_specific" in json_data:
            for dim, value in json_data["realm_specific"].items():
                if dim in soul_map.realm_modifiers:
                    soul_map.realm_modifiers[dim] = float(value)

        return soul_map

    def to_tensor(self) -> torch.Tensor:
        """Convert Soul Map to a 65-dimensional tensor (60 + 5 realm mods)."""
        values = []

        for cluster in [self.cognitive, self.emotional, self.motivational, self.social, self.self]:
            for dim_name in self._get_dimension_list(cluster):
                values.append(cluster.get(dim_name, 0.5))

        for dim_name in REALM_DIMENSIONS:
            values.append(self.realm_modifiers.get(dim_name, 0.0))

        return torch.tensor(values, dtype=torch.float32)

    def _get_dimension_list(self, cluster: Dict) -> List[str]:
        """Get the ordered dimension list for a cluster."""
        if cluster is self.cognitive:
            return COGNITIVE_DIMENSIONS
        elif cluster is self.emotional:
            return EMOTIONAL_DIMENSIONS
        elif cluster is self.motivational:
            return MOTIVATIONAL_DIMENSIONS
        elif cluster is self.social:
            return SOCIAL_DIMENSIONS
        elif cluster is self.self:
            return SELF_DIMENSIONS
        return list(cluster.keys())

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "SoulMap":
        """Create Soul Map from a tensor representation."""
        if tensor.dim() > 1:
            tensor = tensor.squeeze()

        values = tensor.tolist()
        soul_map = cls()

        idx = 0
        for cluster_name, dim_list in [
            ("cognitive", COGNITIVE_DIMENSIONS),
            ("emotional", EMOTIONAL_DIMENSIONS),
            ("motivational", MOTIVATIONAL_DIMENSIONS),
            ("social", SOCIAL_DIMENSIONS),
            ("self", SELF_DIMENSIONS),
        ]:
            cluster = getattr(soul_map, cluster_name)
            for dim in dim_list:
                if idx < len(values):
                    cluster[dim] = values[idx]
                    idx += 1

        for dim in REALM_DIMENSIONS:
            if idx < len(values):
                soul_map.realm_modifiers[dim] = values[idx]
                idx += 1

        return soul_map

    def to_json(self) -> Dict[str, Any]:
        """Convert Soul Map to JSON format."""
        return {
            "cognitive": dict(self.cognitive),
            "emotional": dict(self.emotional),
            "motivational": dict(self.motivational),
            "social": dict(self.social),
            "self": dict(self.self),
            "realm_specific": dict(self.realm_modifiers),
        }

    def get_dimension(self, cluster: str, dimension: str) -> float:
        """Get a specific dimension value."""
        cluster_data = getattr(self, cluster, None)
        if cluster_data is None:
            cluster_data = self.realm_modifiers
        return cluster_data.get(dimension, 0.0)

    def set_dimension(self, cluster: str, dimension: str, value: float) -> None:
        """Set a specific dimension value (clamped to valid range)."""
        dim_range = DIMENSION_RANGES.get(dimension, DEFAULT_RANGE)
        clamped = max(dim_range[0], min(dim_range[1], value))

        if cluster == "realm_modifiers":
            self.realm_modifiers[dimension] = clamped
        else:
            cluster_data = getattr(self, cluster, None)
            if cluster_data is not None and dimension in cluster_data:
                cluster_data[dimension] = clamped

    def apply_delta(self, delta: Dict[str, Dict[str, float]]) -> None:
        """Apply a change delta to the Soul Map."""
        for cluster_name, changes in delta.items():
            for dim, change in changes.items():
                current = self.get_dimension(cluster_name, dim)
                self.set_dimension(cluster_name, dim, current + change)

    def get_cluster_vector(self, cluster: SoulMapCluster) -> torch.Tensor:
        """Get a specific cluster as a tensor."""
        cluster_data = getattr(self, cluster.value)
        dim_list = {
            SoulMapCluster.COGNITIVE: COGNITIVE_DIMENSIONS,
            SoulMapCluster.EMOTIONAL: EMOTIONAL_DIMENSIONS,
            SoulMapCluster.MOTIVATIONAL: MOTIVATIONAL_DIMENSIONS,
            SoulMapCluster.SOCIAL: SOCIAL_DIMENSIONS,
            SoulMapCluster.SELF: SELF_DIMENSIONS,
        }[cluster]

        return torch.tensor([cluster_data[dim] for dim in dim_list], dtype=torch.float32)

    def compute_stability(self) -> float:
        """Compute overall psychological stability (0-1)."""
        # Low volatility + high recovery + high self-coherence = stable
        stability_factors = [
            1.0 - self.emotional["volatility"],
            self.emotional["recovery_rate"],
            self.self["self_coherence"],
            self.self["esteem_stability"],
            self.cognitive["uncertainty_tolerance"],
        ]
        return sum(stability_factors) / len(stability_factors)

    def compute_threat_response(self) -> float:
        """Compute expected threat response intensity (0-1)."""
        return (
            self.emotional["threat_sensitivity"] * 0.4
            + self.emotional["anxiety_baseline"] * 0.3
            + (1.0 - self.cognitive["uncertainty_tolerance"]) * 0.2
            + (1.0 - self.motivational["risk_tolerance"]) * 0.1
        )

    def compute_social_openness(self) -> float:
        """Compute social openness/approachability (0-1)."""
        return (
            self.social["trust_default"] * 0.3
            + self.social["cooperation_tendency"] * 0.25
            + self.social["empathy_capacity"] * 0.25
            + self.motivational["affiliation_drive"] * 0.2
        )

    def get_dominant_motivation(self) -> Tuple[str, float]:
        """Get the strongest motivational drive."""
        drives = [
            ("survival", self.motivational["survival_drive"]),
            ("affiliation", self.motivational["affiliation_drive"]),
            ("status", self.motivational["status_drive"]),
            ("autonomy", self.motivational["autonomy_drive"]),
            ("mastery", self.motivational["mastery_drive"]),
            ("meaning", self.motivational["meaning_drive"]),
            ("novelty", self.motivational["novelty_drive"]),
            ("order", self.motivational["order_drive"]),
        ]
        return max(drives, key=lambda x: x[1])

    def get_tom_depth_int(self) -> int:
        """Get Theory of Mind depth as integer (1-5)."""
        # tom_depth is normalized 0.2-1.0 representing 1-5
        normalized = self.cognitive["tom_depth"]
        return int(1 + (normalized - 0.2) / 0.2)

    def distance_to(self, other: "SoulMap") -> float:
        """Compute psychological distance to another Soul Map."""
        self_tensor = self.to_tensor()
        other_tensor = other.to_tensor()
        return torch.norm(self_tensor - other_tensor).item()

    def blend_with(self, other: "SoulMap", alpha: float = 0.5) -> "SoulMap":
        """Create a blended Soul Map (for hybrid NPCs or influence effects)."""
        self_tensor = self.to_tensor()
        other_tensor = other.to_tensor()
        blended_tensor = alpha * self_tensor + (1 - alpha) * other_tensor
        return SoulMap.from_tensor(blended_tensor)

    def __repr__(self) -> str:
        stability = self.compute_stability()
        threat = self.compute_threat_response()
        social = self.compute_social_openness()
        dominant = self.get_dominant_motivation()
        return (
            f"SoulMap(stability={stability:.2f}, threat_response={threat:.2f}, "
            f"social_openness={social:.2f}, dominant_motivation={dominant[0]})"
        )


class SoulMapDelta:
    """Represents a change to a Soul Map, used for cognitive hazards and interventions."""

    def __init__(self, changes: Optional[Dict[str, Dict[str, float]]] = None):
        self.changes = changes or {}

    def add_change(self, cluster: str, dimension: str, delta: float) -> None:
        """Add a change to the delta."""
        if cluster not in self.changes:
            self.changes[cluster] = {}
        self.changes[cluster][dimension] = delta

    def apply_to(self, soul_map: SoulMap) -> None:
        """Apply this delta to a Soul Map."""
        soul_map.apply_delta(self.changes)

    def scale(self, factor: float) -> "SoulMapDelta":
        """Return a scaled version of this delta."""
        scaled = SoulMapDelta()
        for cluster, dims in self.changes.items():
            for dim, value in dims.items():
                scaled.add_change(cluster, dim, value * factor)
        return scaled

    @classmethod
    def doubt(cls, intensity: float = 0.2) -> "SoulMapDelta":
        """Create a 'Doubt' cognitive hazard delta."""
        delta = cls()
        delta.add_change("cognitive", "uncertainty_tolerance", -intensity)
        delta.add_change("self", "self_coherence", -intensity * 0.5)
        delta.add_change("self", "agency_sense", -intensity * 0.3)
        return delta

    @classmethod
    def validation(cls, intensity: float = 0.2) -> "SoulMapDelta":
        """Create a 'Validation' positive intervention delta."""
        delta = cls()
        delta.add_change("self", "esteem_stability", intensity)
        delta.add_change("emotional", "baseline_valence", intensity * 0.3)
        delta.add_change("emotional", "anxiety_baseline", -intensity * 0.2)
        return delta

    @classmethod
    def reassurance(cls, intensity: float = 0.2) -> "SoulMapDelta":
        """Create a 'Reassurance' intervention delta."""
        delta = cls()
        delta.add_change("emotional", "anxiety_baseline", -intensity)
        delta.add_change("emotional", "threat_sensitivity", -intensity * 0.5)
        delta.add_change("social", "trust_default", intensity * 0.3)
        return delta

    @classmethod
    def paradox(cls, intensity: float = 0.3) -> "SoulMapDelta":
        """Create a 'Paradox' cognitive hazard delta."""
        delta = cls()
        delta.add_change("cognitive", "cognitive_flexibility", -intensity)
        delta.add_change("cognitive", "processing_speed", -intensity * 0.5)
        delta.add_change("self", "self_coherence", -intensity * 0.4)
        return delta

    @classmethod
    def fear(cls, intensity: float = 0.3) -> "SoulMapDelta":
        """Create a 'Fear' emotional injection delta."""
        delta = cls()
        delta.add_change("emotional", "anxiety_baseline", intensity)
        delta.add_change("emotional", "threat_sensitivity", intensity * 0.5)
        delta.add_change("motivational", "approach_avoidance", -intensity)
        delta.add_change("motivational", "survival_drive", intensity * 0.3)
        return delta

    @classmethod
    def curiosity(cls, intensity: float = 0.2) -> "SoulMapDelta":
        """Create a 'Curiosity' motivational injection delta."""
        delta = cls()
        delta.add_change("motivational", "novelty_drive", intensity)
        delta.add_change("motivational", "approach_avoidance", intensity * 0.5)
        delta.add_change("cognitive", "uncertainty_tolerance", intensity * 0.3)
        return delta


# Export main classes and constants
__all__ = [
    "SoulMap",
    "SoulMapCluster",
    "SoulMapDelta",
    "COGNITIVE_DIMENSIONS",
    "EMOTIONAL_DIMENSIONS",
    "MOTIVATIONAL_DIMENSIONS",
    "SOCIAL_DIMENSIONS",
    "SELF_DIMENSIONS",
    "REALM_DIMENSIONS",
    "DIMENSION_RANGES",
    "DEFAULT_RANGE",
]
