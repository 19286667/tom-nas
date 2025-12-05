"""
Soul Scanner - The Analysis System

The Soul Scanner is the player's primary tool for understanding NPCs.
Instead of a minimap and ammo counter, this HUD visualizes the 60-dimensional
psychological state of any targeted NPC.

Features:
- Passive Aura Reading: Basic emotional state on hover
- Deep Analysis Mode: Full Soul Map radar chart overlay
- Cluster Highlighting: Focus on specific psychological dimensions
- Prediction Display: Show predicted behaviors based on state
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


from ..npcs.base_npc import BaseNPC
from ..soul_map import SoulMap


class AnalysisDepth(Enum):
    """Levels of analysis depth."""

    PASSIVE = "passive"  # Just hovering - basic info
    SHALLOW = "shallow"  # Quick scan - emotional state
    MODERATE = "moderate"  # Standard analysis - main clusters
    DEEP = "deep"  # Full analysis - all 60 dimensions
    PREDICTIVE = "predictive"  # Analysis + behavior prediction


@dataclass
class AnalysisResult:
    """Result of analyzing an NPC with the Soul Scanner."""

    # Target info
    npc_id: str
    npc_name: str
    archetype: str

    # Analysis depth achieved
    depth: AnalysisDepth

    # Basic info (always available)
    emotional_state: str
    current_activity: str
    threat_level: float  # How threatening they are to player

    # Passive level (hover)
    aura_color: str  # Visual representation of overall state
    aura_intensity: float

    # Shallow level
    dominant_emotion: Optional[str] = None
    emotional_valence: Optional[float] = None
    is_aware_of_player: bool = False

    # Moderate level - cluster summaries
    cluster_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Deep level - full Soul Map
    full_soul_map: Optional[Dict[str, Any]] = None

    # Predictive level
    predicted_behaviors: List[str] = field(default_factory=list)
    behavior_confidence: float = 0.0

    # Special indicators
    is_zombie: bool = False
    zombie_type: Optional[str] = None
    tom_depth: int = 0

    # Volatility indicators (dimensions that are spiking)
    volatile_dimensions: List[str] = field(default_factory=list)


class SoulScanner:
    """
    The player's primary analysis tool.

    Usage:
    1. Hover over NPC: Get passive aura reading
    2. Right-click (hold): Enter deep analysis mode
    3. Mouse over clusters: Get detailed breakdowns
    4. Left-click: Select for intervention targeting
    """

    def __init__(self, player_tom_depth: int = 3):
        """
        Initialize the Soul Scanner.

        Args:
            player_tom_depth: Player's Theory of Mind capability (affects analysis quality)
        """
        self.player_tom_depth = player_tom_depth
        self.current_target: Optional[BaseNPC] = None
        self.analysis_history: List[AnalysisResult] = []
        self.max_history = 50

    def passive_scan(self, npc: BaseNPC) -> AnalysisResult:
        """
        Quick passive scan (hover mode).

        Returns basic aura and emotional state.
        """
        soul = npc.soul_map

        # Calculate aura color based on dominant emotional state
        aura_color = self._calculate_aura_color(soul)
        aura_intensity = self._calculate_aura_intensity(soul)

        return AnalysisResult(
            npc_id=npc.npc_id,
            npc_name=npc.name,
            archetype=npc.archetype,
            depth=AnalysisDepth.PASSIVE,
            emotional_state=npc.emotional_state,
            current_activity=npc.current_state.value,
            threat_level=self._estimate_threat(npc),
            aura_color=aura_color,
            aura_intensity=aura_intensity,
        )

    def shallow_scan(self, npc: BaseNPC) -> AnalysisResult:
        """
        Quick analysis - dominant emotion and awareness.
        """
        result = self.passive_scan(npc)
        result.depth = AnalysisDepth.SHALLOW

        soul = npc.soul_map

        # Determine dominant emotion
        result.dominant_emotion = self._get_dominant_emotion(soul)
        result.emotional_valence = soul.emotional["baseline_valence"]
        result.is_aware_of_player = npc.awareness_of_player > 0.3
        result.tom_depth = npc.tom_depth

        return result

    def moderate_scan(self, npc: BaseNPC) -> AnalysisResult:
        """
        Standard analysis - cluster summaries.
        """
        result = self.shallow_scan(npc)
        result.depth = AnalysisDepth.MODERATE

        soul = npc.soul_map

        # Generate cluster summaries
        result.cluster_summaries = {
            "cognitive": self._summarize_cognitive(soul),
            "emotional": self._summarize_emotional(soul),
            "motivational": self._summarize_motivational(soul),
            "social": self._summarize_social(soul),
            "self": self._summarize_self(soul),
        }

        # Identify volatile dimensions
        result.volatile_dimensions = self._identify_volatile_dimensions(soul)

        # Check for zombie indicators
        result.is_zombie = npc.is_zombie
        result.zombie_type = npc.zombie_type

        return result

    def deep_scan(self, npc: BaseNPC) -> AnalysisResult:
        """
        Full analysis - complete Soul Map data.

        Quality depends on player's ToM depth vs NPC's ToM depth.
        """
        result = self.moderate_scan(npc)
        result.depth = AnalysisDepth.DEEP

        # Full Soul Map (accuracy depends on ToM match)
        accuracy = self._calculate_analysis_accuracy(npc)
        result.full_soul_map = self._get_soul_map_with_accuracy(npc.soul_map, accuracy)

        return result

    def predictive_scan(self, npc: BaseNPC, context: Dict[str, Any]) -> AnalysisResult:
        """
        Predictive analysis - behavior forecasting.

        This is where ToM reasoning really matters.
        """
        result = self.deep_scan(npc)
        result.depth = AnalysisDepth.PREDICTIVE

        # Predict behaviors based on Soul Map and context
        predictions = self._predict_behaviors(npc, context)
        result.predicted_behaviors = predictions
        result.behavior_confidence = self._calculate_prediction_confidence(npc)

        # Store in history
        self._add_to_history(result)

        return result

    def _calculate_aura_color(self, soul: SoulMap) -> str:
        """
        Calculate aura color based on emotional state.

        Colors:
        - Blue: Calm/stable
        - Red: Angry/threatened
        - Yellow: Anxious/alert
        - Purple: Complex/deep
        - Gray: Muted/depressed
        - Green: Balanced/healthy
        - Orange: Volatile/unpredictable
        """
        valence = soul.emotional["baseline_valence"]
        volatility = soul.emotional["volatility"]
        anxiety = soul.emotional["anxiety_baseline"]
        intensity = soul.emotional["intensity"]

        # High volatility = orange
        if volatility > 0.7:
            return "orange"

        # High anxiety = yellow
        if anxiety > 0.7:
            return "yellow"

        # Negative valence
        if valence < -0.3:
            if intensity > 0.6:
                return "red"  # Angry
            else:
                return "gray"  # Depressed

        # Positive valence
        if valence > 0.3:
            return "green"

        # Neutral but complex
        complexity = soul.self["self_complexity"]
        if complexity > 0.7:
            return "purple"

        return "blue"

    def _calculate_aura_intensity(self, soul: SoulMap) -> float:
        """Calculate aura brightness/intensity."""
        return (
            soul.emotional["intensity"] * 0.5
            + (1.0 - soul.emotional["recovery_rate"]) * 0.3
            + abs(soul.emotional["baseline_valence"]) * 0.2
        )

    def _estimate_threat(self, npc: BaseNPC) -> float:
        """Estimate how threatening this NPC is."""
        soul = npc.soul_map

        threat = 0.0

        # High competition tendency
        threat += soul.social["competition_tendency"] * 0.3

        # Low trust + aware of player
        if npc.awareness_of_player > 0.5:
            threat += (1.0 - soul.social["trust_default"]) * 0.3

        # High threat sensitivity (they might attack first)
        threat += soul.emotional["threat_sensitivity"] * 0.2

        # Low empathy
        threat += (1.0 - soul.social["empathy_capacity"]) * 0.2

        return min(1.0, threat)

    def _get_dominant_emotion(self, soul: SoulMap) -> str:
        """Determine the dominant emotional state."""
        valence = soul.emotional["baseline_valence"]
        anxiety = soul.emotional["anxiety_baseline"]
        intensity = soul.emotional["intensity"]

        if anxiety > 0.7:
            return "anxious"
        if valence < -0.5:
            if intensity > 0.6:
                return "angry"
            return "sad"
        if valence > 0.5:
            return "content"
        if intensity < 0.3:
            return "apathetic"

        return "neutral"

    def _summarize_cognitive(self, soul: SoulMap) -> Dict[str, Any]:
        """Summarize cognitive cluster."""
        return {
            "intelligence": (soul.cognitive["processing_speed"] + soul.cognitive["pattern_recognition"]) / 2,
            "flexibility": soul.cognitive["cognitive_flexibility"],
            "metacognition": soul.cognitive["metacognitive_awareness"],
            "tom_potential": soul.cognitive["tom_depth"],
            "key_trait": self._get_cognitive_key_trait(soul),
        }

    def _summarize_emotional(self, soul: SoulMap) -> Dict[str, Any]:
        """Summarize emotional cluster."""
        return {
            "mood": soul.emotional["baseline_valence"],
            "stability": 1.0 - soul.emotional["volatility"],
            "sensitivity": soul.emotional["threat_sensitivity"],
            "recovery": soul.emotional["recovery_rate"],
            "key_trait": self._get_emotional_key_trait(soul),
        }

    def _summarize_motivational(self, soul: SoulMap) -> Dict[str, Any]:
        """Summarize motivational cluster."""
        dominant = soul.get_dominant_motivation()
        return {
            "dominant_drive": dominant[0],
            "drive_strength": dominant[1],
            "approach_tendency": soul.motivational["approach_avoidance"],
            "risk_profile": soul.motivational["risk_tolerance"],
            "key_trait": f"driven by {dominant[0]}",
        }

    def _summarize_social(self, soul: SoulMap) -> Dict[str, Any]:
        """Summarize social cluster."""
        return {
            "trust_baseline": soul.social["trust_default"],
            "cooperation": soul.social["cooperation_tendency"],
            "empathy": soul.social["empathy_capacity"],
            "authority_deference": soul.social["authority_orientation"],
            "key_trait": self._get_social_key_trait(soul),
        }

    def _summarize_self(self, soul: SoulMap) -> Dict[str, Any]:
        """Summarize self cluster."""
        return {
            "coherence": soul.self["self_coherence"],
            "identity_clarity": soul.self["identity_clarity"],
            "agency": soul.self["agency_sense"],
            "narcissism": soul.self["narcissism"],
            "key_trait": self._get_self_key_trait(soul),
        }

    def _get_cognitive_key_trait(self, soul: SoulMap) -> str:
        """Get the defining cognitive trait."""
        traits = [
            ("analytical", soul.cognitive["pattern_recognition"]),
            ("flexible", soul.cognitive["cognitive_flexibility"]),
            ("self-aware", soul.cognitive["metacognitive_awareness"]),
            ("rigid", 1.0 - soul.cognitive["uncertainty_tolerance"]),
        ]
        return max(traits, key=lambda x: x[1])[0]

    def _get_emotional_key_trait(self, soul: SoulMap) -> str:
        """Get the defining emotional trait."""
        if soul.emotional["anxiety_baseline"] > 0.7:
            return "anxious"
        if soul.emotional["volatility"] > 0.7:
            return "volatile"
        if soul.emotional["recovery_rate"] > 0.7:
            return "resilient"
        if soul.emotional["baseline_valence"] < -0.3:
            return "melancholic"
        return "stable"

    def _get_social_key_trait(self, soul: SoulMap) -> str:
        """Get the defining social trait."""
        if soul.social["empathy_capacity"] > 0.7:
            return "empathetic"
        if soul.social["authority_orientation"] > 0.7:
            return "deferential"
        if soul.social["competition_tendency"] > 0.7:
            return "competitive"
        if soul.social["trust_default"] < 0.3:
            return "suspicious"
        return "balanced"

    def _get_self_key_trait(self, soul: SoulMap) -> str:
        """Get the defining self trait."""
        if soul.self["narcissism"] > 0.7:
            return "narcissistic"
        if soul.self["self_coherence"] < 0.3:
            return "fragmented"
        if soul.self["agency_sense"] > 0.7:
            return "agentic"
        if soul.self["authenticity_drive"] > 0.7:
            return "authentic"
        return "stable"

    def _identify_volatile_dimensions(self, soul: SoulMap) -> List[str]:
        """Identify dimensions that are at extreme values or changing."""
        volatile = []

        # Check for extreme values
        for cluster_name in ["cognitive", "emotional", "motivational", "social", "self"]:
            cluster = getattr(soul, cluster_name)
            for dim, value in cluster.items():
                if value > 0.85 or value < 0.15:
                    volatile.append(f"{cluster_name}.{dim}")

        return volatile[:10]  # Limit to top 10

    def _calculate_analysis_accuracy(self, npc: BaseNPC) -> float:
        """
        Calculate how accurate our analysis is.

        If our ToM depth >= NPC's ToM depth, we can fully understand them.
        Otherwise, accuracy decreases.
        """
        tom_gap = npc.tom_depth - self.player_tom_depth
        if tom_gap <= 0:
            return 1.0

        # Each level of ToM gap reduces accuracy by 15%
        return max(0.3, 1.0 - tom_gap * 0.15)

    def _get_soul_map_with_accuracy(self, soul: SoulMap, accuracy: float) -> Dict[str, Any]:
        """Get Soul Map data with accuracy-based noise for limited ToM."""
        import random

        data = soul.to_json()

        if accuracy >= 1.0:
            return data

        # Add noise based on inaccuracy
        noise_level = 1.0 - accuracy
        for cluster_name, cluster in data.items():
            if isinstance(cluster, dict):
                for dim, value in cluster.items():
                    if isinstance(value, (int, float)):
                        noise = random.uniform(-noise_level * 0.3, noise_level * 0.3)
                        cluster[dim] = max(0.0, min(1.0, value + noise))

        return data

    def _predict_behaviors(self, npc: BaseNPC, context: Dict[str, Any]) -> List[str]:
        """Predict likely behaviors based on Soul Map and context."""
        predictions = []
        soul = npc.soul_map

        # High anxiety + threat = flee
        if soul.emotional["anxiety_baseline"] > 0.6 and context.get("threat_present", False):
            predictions.append("likely to flee or hide")

        # Low trust + stranger = avoidance
        if soul.social["trust_default"] < 0.3 and context.get("stranger_present", False):
            predictions.append("will avoid interaction")

        # High order drive = rule-following
        if soul.motivational["order_drive"] > 0.7:
            predictions.append("will follow established rules")

        # High novelty + opportunity = investigation
        if soul.motivational["novelty_drive"] > 0.7:
            predictions.append("curious - will investigate anomalies")

        # High empathy + distressed nearby = help
        if soul.social["empathy_capacity"] > 0.7 and context.get("distressed_npc_nearby", False):
            predictions.append("will try to help others")

        # High competition = challenge
        if soul.social["competition_tendency"] > 0.7:
            predictions.append("may challenge or compete")

        # Realm-specific predictions
        if npc.current_realm:
            realm_predictions = self._predict_realm_behaviors(npc, context)
            predictions.extend(realm_predictions)

        return predictions[:5]  # Limit to top 5

    def _predict_realm_behaviors(self, npc: BaseNPC, context: Dict[str, Any]) -> List[str]:
        """Predict behaviors specific to current realm."""
        predictions = []
        soul = npc.soul_map

        if npc.current_realm and npc.current_realm.value == "ministry":
            certainty = soul.realm_modifiers.get("corporeal_certainty", 1.0)
            if certainty < 0.5:
                predictions.append("fading - desperate for paperwork")

        elif npc.current_realm and npc.current_realm.value == "hollow":
            corruption = soul.realm_modifiers.get("corruption", 0.0)
            if corruption > 0.5:
                predictions.append("partially consumed - may act for hive")

        return predictions

    def _calculate_prediction_confidence(self, npc: BaseNPC) -> float:
        """Calculate confidence in our predictions."""
        base = self._calculate_analysis_accuracy(npc)

        # More predictable NPCs = higher confidence
        predictability = (
            (1.0 - npc.soul_map.emotional["volatility"]) * 0.3
            + npc.soul_map.self["self_coherence"] * 0.3
            + (1.0 - npc.soul_map.cognitive["cognitive_flexibility"]) * 0.2
            + 0.2  # Base confidence
        )

        return min(1.0, base * predictability)

    def _add_to_history(self, result: AnalysisResult) -> None:
        """Add analysis to history."""
        self.analysis_history.append(result)
        if len(self.analysis_history) > self.max_history:
            self.analysis_history.pop(0)

    def get_history_for_npc(self, npc_id: str) -> List[AnalysisResult]:
        """Get analysis history for a specific NPC."""
        return [r for r in self.analysis_history if r.npc_id == npc_id]


# Export
__all__ = [
    "SoulScanner",
    "AnalysisResult",
    "AnalysisDepth",
]
