"""
Ontological Instability - The "Wanted" System

Instead of Police Stars, Liminal Architectures uses Ontological Instability.
If the player disrupts reality too much (through excessive interventions,
breaking realm rules, or causing too much psychological damage), The Nothing
begins to erase the assets around them.

Instability Sources:
- Using too many cognitive hazards in quick succession
- Causing NPCs to have psychological breaks
- Breaking realm-specific rules (e.g., not filing forms in Ministry)
- Entering restricted areas
- Revealing Truth to too many NPCs

Instability Effects:
- Level 1: Visual glitches, ambient warnings
- Level 2: NPCs become aware of wrongness, hostile reactions
- Level 3: Environment starts degrading, paths close
- Level 4: The Nothing manifests, active erasure
- Level 5: Total reality collapse (game over condition)
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


class InstabilityLevel(Enum):
    """Levels of ontological instability (like GTA wanted stars)."""

    STABLE = 0  # No instability
    RIPPLES = 1  # Minor disturbances
    DISTORTION = 2  # Noticeable reality warping
    FRACTURING = 3  # Active environmental damage
    MANIFESTATION = 4  # The Nothing is here
    COLLAPSE = 5  # Reality failure (game over)


@dataclass
class InstabilityEvent:
    """Record of an event that caused instability."""

    timestamp: int
    source: str
    amount: float
    description: str
    location: Optional[Tuple[float, float]] = None


@dataclass
class InstabilityEffect:
    """An effect triggered by instability level."""

    level: InstabilityLevel
    effect_type: str
    intensity: float
    description: str


class OntologicalInstability:
    """
    Manages the player's ontological instability level.

    This is the "wanted" system - disrupting reality too much brings
    consequences that escalate until reality itself collapses.
    """

    def __init__(self, decay_rate: float = 0.01):
        """
        Initialize the instability tracker.

        Args:
            decay_rate: How much instability naturally decays per tick
        """
        self.instability: float = 0.0  # 0-100 scale
        self.decay_rate = decay_rate
        self.current_level = InstabilityLevel.STABLE

        # History of instability events
        self.event_history: List[InstabilityEvent] = []
        self.max_history = 100

        # Thresholds for each level
        self.level_thresholds = {
            InstabilityLevel.STABLE: 0,
            InstabilityLevel.RIPPLES: 15,
            InstabilityLevel.DISTORTION: 35,
            InstabilityLevel.FRACTURING: 55,
            InstabilityLevel.MANIFESTATION: 75,
            InstabilityLevel.COLLAPSE: 100,
        }

        # Active effects at current level
        self.active_effects: List[InstabilityEffect] = []

        # Callbacks for level changes
        self._level_change_callbacks: List[Callable] = []

        # Is The Nothing currently manifested?
        self.nothing_manifested = False

        # How long until collapse at level 5
        self.collapse_timer: Optional[int] = None

    def add_instability(
        self,
        amount: float,
        source: str,
        description: str = "",
        timestamp: int = 0,
        location: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Add instability from an action.

        Args:
            amount: How much instability to add (0-20 typical)
            source: What caused it (e.g., "hazard:fear", "realm_violation")
            description: Human-readable description
            timestamp: Game tick when it occurred
            location: Where it happened
        """
        # Scale amount based on current level (higher levels = more sensitive)
        scale = 1.0 + (self.current_level.value * 0.1)
        actual_amount = amount * scale

        self.instability = min(100, self.instability + actual_amount)

        # Record event
        event = InstabilityEvent(
            timestamp=timestamp,
            source=source,
            amount=actual_amount,
            description=description,
            location=location,
        )
        self._add_event(event)

        # Check for level changes
        self._update_level()

    def tick(self, delta_time: float = 1.0) -> Dict[str, Any]:
        """
        Process one game tick.

        Args:
            delta_time: Time since last tick

        Returns:
            Dict with current state and any triggered effects
        """
        # Natural decay
        decay = self.decay_rate * delta_time
        # Decay is slower at higher levels
        decay *= 1.0 - self.current_level.value * 0.1
        self.instability = max(0, self.instability - decay)

        # Update level
        old_level = self.current_level
        self._update_level()

        # Process collapse timer if at max level
        if self.current_level == InstabilityLevel.COLLAPSE:
            if self.collapse_timer is None:
                self.collapse_timer = 300  # 5 seconds at 60 fps
            else:
                self.collapse_timer -= int(delta_time)

        elif self.collapse_timer is not None:
            self.collapse_timer = None

        # Generate current effects
        effects = self._generate_effects()

        return {
            "instability": self.instability,
            "level": self.current_level,
            "level_changed": old_level != self.current_level,
            "effects": effects,
            "collapse_timer": self.collapse_timer,
            "nothing_manifested": self.nothing_manifested,
        }

    def _update_level(self) -> None:
        """Update the current instability level based on instability value."""
        old_level = self.current_level

        # Find appropriate level
        for level in reversed(list(InstabilityLevel)):
            if self.instability >= self.level_thresholds[level]:
                self.current_level = level
                break

        # Level changed?
        if old_level != self.current_level:
            self._on_level_change(old_level, self.current_level)

    def _on_level_change(self, old: InstabilityLevel, new: InstabilityLevel) -> None:
        """Handle level change events."""
        # Manifest The Nothing at high levels
        if new.value >= InstabilityLevel.MANIFESTATION.value:
            self.nothing_manifested = True
        elif new.value < InstabilityLevel.MANIFESTATION.value:
            self.nothing_manifested = False

        # Call registered callbacks
        for callback in self._level_change_callbacks:
            callback(old, new)

    def _generate_effects(self) -> List[InstabilityEffect]:
        """Generate effects for current instability level."""
        effects = []
        level = self.current_level

        if level == InstabilityLevel.STABLE:
            return effects

        if level.value >= InstabilityLevel.RIPPLES.value:
            # Visual glitches
            effects.append(
                InstabilityEffect(
                    level=level,
                    effect_type="visual_glitch",
                    intensity=self.instability / 100,
                    description="Reality ripples at the edges of vision",
                )
            )

        if level.value >= InstabilityLevel.DISTORTION.value:
            # Audio distortion, NPC awareness
            effects.append(
                InstabilityEffect(
                    level=level,
                    effect_type="audio_distortion",
                    intensity=(self.instability - 35) / 65,
                    description="Sounds echo strangely, NPCs sense wrongness",
                )
            )
            effects.append(
                InstabilityEffect(
                    level=level,
                    effect_type="npc_awareness",
                    intensity=(self.instability - 35) / 65,
                    description="NPCs become uneasy, may become hostile",
                )
            )

        if level.value >= InstabilityLevel.FRACTURING.value:
            # Environmental damage
            effects.append(
                InstabilityEffect(
                    level=level,
                    effect_type="environment_damage",
                    intensity=(self.instability - 55) / 45,
                    description="The world cracks, paths may close",
                )
            )

        if level.value >= InstabilityLevel.MANIFESTATION.value:
            # The Nothing appears
            effects.append(
                InstabilityEffect(
                    level=level,
                    effect_type="nothing_manifestation",
                    intensity=(self.instability - 75) / 25,
                    description="The Nothing manifests, actively erasing reality",
                )
            )

        if level.value >= InstabilityLevel.COLLAPSE.value:
            # Collapse imminent
            effects.append(
                InstabilityEffect(
                    level=level,
                    effect_type="collapse_imminent",
                    intensity=1.0,
                    description="Reality is collapsing. Seek shelter in stable zones.",
                )
            )

        return effects

    def _add_event(self, event: InstabilityEvent) -> None:
        """Add event to history."""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

    def on_level_change(self, callback: Callable) -> None:
        """Register a callback for level changes."""
        self._level_change_callbacks.append(callback)

    def reduce_instability(self, amount: float, source: str = "natural") -> None:
        """Reduce instability (e.g., entering safe zones, meditation)."""
        self.instability = max(0, self.instability - amount)
        self._update_level()

    def get_safe_zone_requirement(self) -> float:
        """Get how much time in a safe zone is needed to fully stabilize."""
        return self.instability / (self.decay_rate * 10)  # In ticks

    def is_collapse_imminent(self) -> bool:
        """Check if reality collapse is about to occur."""
        return self.collapse_timer is not None and self.collapse_timer < 60

    def get_display_data(self) -> Dict[str, Any]:
        """Get data for HUD display."""
        return {
            "level": self.current_level.value,
            "level_name": self.current_level.name,
            "instability_percent": self.instability,
            "nothing_manifested": self.nothing_manifested,
            "collapse_timer": self.collapse_timer,
            "danger_color": self._get_danger_color(),
        }

    def _get_danger_color(self) -> str:
        """Get HUD color based on instability level."""
        colors = {
            InstabilityLevel.STABLE: "green",
            InstabilityLevel.RIPPLES: "yellow",
            InstabilityLevel.DISTORTION: "orange",
            InstabilityLevel.FRACTURING: "red",
            InstabilityLevel.MANIFESTATION: "purple",
            InstabilityLevel.COLLAPSE: "white",  # Everything is white/void
        }
        return colors.get(self.current_level, "gray")


# === INSTABILITY SOURCES ===

# Standard instability amounts for different actions
INSTABILITY_AMOUNTS = {
    # Cognitive hazards
    "hazard_destabilizing": 5.0,
    "hazard_emotional": 3.0,
    "hazard_cognitive": 4.0,
    "hazard_social": 3.0,
    "hazard_existential": 8.0,
    "hazard_stabilizing": -1.0,  # Actually reduces instability
    # NPC reactions
    "npc_psychological_break": 10.0,
    "npc_death": 15.0,
    "npc_awakening": 5.0,  # Revealing truth
    # Realm violations
    "ministry_no_paperwork": 3.0,
    "ministry_running": 2.0,
    "city_excessive_adaptation": 4.0,
    "city_excessive_rigidity": 4.0,
    "hollow_corruption_spread": 6.0,
    "nothing_prolonged_stay": 8.0,
    # Player actions
    "rapid_interventions": 2.0,  # Per intervention above threshold
    "area_violation": 5.0,
    "reality_manipulation": 7.0,
}


def calculate_hazard_instability(hazard_category: str) -> float:
    """Calculate instability for using a cognitive hazard."""
    category_map = {
        "destabilizing": INSTABILITY_AMOUNTS["hazard_destabilizing"],
        "emotional": INSTABILITY_AMOUNTS["hazard_emotional"],
        "cognitive": INSTABILITY_AMOUNTS["hazard_cognitive"],
        "social": INSTABILITY_AMOUNTS["hazard_social"],
        "existential": INSTABILITY_AMOUNTS["hazard_existential"],
        "stabilizing": INSTABILITY_AMOUNTS["hazard_stabilizing"],
    }
    return category_map.get(hazard_category, 3.0)


# Export
__all__ = [
    "OntologicalInstability",
    "InstabilityLevel",
    "InstabilityEvent",
    "InstabilityEffect",
    "INSTABILITY_AMOUNTS",
    "calculate_hazard_instability",
]
