"""
Context Manager: Sociological Database for ToM-NAS
==================================================

Provides RAG-based lookup for:
- Institutional norms and constraints
- Social prototypes and stereotypes
- Mundane activity expectations
- Role definitions and power dynamics

This is NOT hardcoded logic. The taxonomies are stored as queryable
data that the MetaMind pipeline retrieves at runtime.

Author: ToM-NAS Project
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import logging

from ..simulation_config import (
    InstitutionType,
    InstitutionalNorm,
    SocialPrototype,
    StereotypeDimension,
    MundaneCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TAXONOMY ENTRIES
# =============================================================================

@dataclass
class NormEntry:
    """An entry in the norms database."""
    norm_id: str
    institution: str
    name: str
    description: str
    violation_cost: float
    detection_probability: float
    applies_to_roles: List[str]
    context_modifiers: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class PrototypeEntry:
    """An entry in the social prototypes database."""
    prototype_id: str
    name: str
    warmth_prior: float
    competence_prior: float
    feature_triggers: List[str]
    institution_context: Optional[str] = None
    role_context: Optional[str] = None
    confidence: float = 0.5
    embedding: Optional[List[float]] = None


@dataclass
class MundaneEntry:
    """An entry in the mundane activities database."""
    activity_id: str
    category: str
    name: str
    description: str
    typical_duration_minutes: int
    interruptibility: float  # 0 = never interrupt, 1 = always ok
    time_windows: List[Tuple[float, float]]  # Hours of day
    location_types: List[str]
    cognitive_impact: float  # Effect on cognitive performance
    social_acceptability: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoleEntry:
    """An entry in the role definitions database."""
    role_id: str
    institution: str
    name: str
    description: str
    power_level: float  # 0-1 scale
    expected_competence: float
    expected_warmth: float
    typical_goals: List[str]
    normative_behaviors: List[str]
    taboo_behaviors: List[str]


# =============================================================================
# SOCIOLOGICAL DATABASE
# =============================================================================

class SociologicalDatabase:
    """
    In-memory database of sociological knowledge.

    In production, this would be backed by a vector database (ChromaDB, etc.)
    for semantic similarity search.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None

        # In-memory storage
        self.norms: Dict[str, NormEntry] = {}
        self.prototypes: Dict[str, PrototypeEntry] = {}
        self.mundane: Dict[str, MundaneEntry] = {}
        self.roles: Dict[str, RoleEntry] = {}

        # Index by institution
        self.norms_by_institution: Dict[str, List[str]] = {}
        self.roles_by_institution: Dict[str, List[str]] = {}

        # Load default data
        self._load_default_data()

        if self.data_path and self.data_path.exists():
            self._load_from_path()

    def _load_default_data(self):
        """Load default taxonomies."""
        self._load_default_norms()
        self._load_default_prototypes()
        self._load_default_mundane()
        self._load_default_roles()

    def _load_default_norms(self):
        """Load default institutional norms."""
        # Family norms
        family_norms = [
            NormEntry(
                norm_id="family_support",
                institution="family",
                name="Provide emotional support",
                description="Family members are expected to support each other emotionally",
                violation_cost=0.6,
                detection_probability=0.8,
                applies_to_roles=["parent", "sibling", "child"],
            ),
            NormEntry(
                norm_id="family_honesty",
                institution="family",
                name="Be honest with family",
                description="Deception within family is strongly discouraged",
                violation_cost=0.8,
                detection_probability=0.6,
                applies_to_roles=["all"],
            ),
            NormEntry(
                norm_id="family_meals",
                institution="family",
                name="Share meals together",
                description="Eating together reinforces family bonds",
                violation_cost=0.3,
                detection_probability=0.9,
                applies_to_roles=["all"],
            ),
        ]

        # Workplace norms
        workplace_norms = [
            NormEntry(
                norm_id="workplace_professionalism",
                institution="workplace",
                name="Maintain professional demeanor",
                description="Emotional displays should be controlled in workplace",
                violation_cost=0.5,
                detection_probability=0.7,
                applies_to_roles=["employee", "manager", "executive"],
            ),
            NormEntry(
                norm_id="workplace_hierarchy",
                institution="workplace",
                name="Respect organizational hierarchy",
                description="Defer to superiors in decision-making",
                violation_cost=0.6,
                detection_probability=0.8,
                applies_to_roles=["employee"],
            ),
            NormEntry(
                norm_id="workplace_confidentiality",
                institution="workplace",
                name="Maintain confidentiality",
                description="Do not share sensitive company information",
                violation_cost=0.9,
                detection_probability=0.4,
                applies_to_roles=["all"],
            ),
            NormEntry(
                norm_id="workplace_punctuality",
                institution="workplace",
                name="Be punctual",
                description="Arrive on time for meetings and work",
                violation_cost=0.4,
                detection_probability=0.9,
                applies_to_roles=["all"],
            ),
        ]

        # Political norms
        political_norms = [
            NormEntry(
                norm_id="political_coalition",
                institution="political",
                name="Maintain coalition loyalty",
                description="Support allies publicly even when privately disagreeing",
                violation_cost=0.7,
                detection_probability=0.5,
                applies_to_roles=["politician", "advisor"],
            ),
            NormEntry(
                norm_id="political_face",
                institution="political",
                name="Present unified front",
                description="Internal disagreements stay internal",
                violation_cost=0.8,
                detection_probability=0.6,
                applies_to_roles=["all"],
            ),
            NormEntry(
                norm_id="political_reciprocity",
                institution="political",
                name="Honor political favors",
                description="Favors create obligations that must be repaid",
                violation_cost=0.9,
                detection_probability=0.7,
                applies_to_roles=["all"],
            ),
        ]

        # Add all norms
        for norm in family_norms + workplace_norms + political_norms:
            self.norms[norm.norm_id] = norm
            if norm.institution not in self.norms_by_institution:
                self.norms_by_institution[norm.institution] = []
            self.norms_by_institution[norm.institution].append(norm.norm_id)

    def _load_default_prototypes(self):
        """Load default social prototypes."""
        prototypes = [
            # Status-based prototypes
            PrototypeEntry(
                prototype_id="high_status_professional",
                name="High-status professional",
                warmth_prior=0.3,
                competence_prior=0.8,
                feature_triggers=["wearing_suit", "expensive_watch", "confident_posture"],
                institution_context="workplace",
            ),
            PrototypeEntry(
                prototype_id="caring_parent",
                name="Caring parent",
                warmth_prior=0.9,
                competence_prior=0.5,
                feature_triggers=["with_child", "nurturing_behavior", "protective_stance"],
                institution_context="family",
            ),
            PrototypeEntry(
                prototype_id="authority_figure",
                name="Authority figure",
                warmth_prior=0.2,
                competence_prior=0.7,
                feature_triggers=["uniform", "badge", "commanding_voice"],
                institution_context=None,
            ),
            PrototypeEntry(
                prototype_id="helpful_stranger",
                name="Helpful stranger",
                warmth_prior=0.6,
                competence_prior=0.4,
                feature_triggers=["offering_help", "friendly_approach", "open_body_language"],
            ),
            PrototypeEntry(
                prototype_id="stressed_worker",
                name="Stressed worker",
                warmth_prior=0.3,
                competence_prior=0.5,
                feature_triggers=["rushed_movement", "checking_phone", "tense_posture"],
                institution_context="workplace",
            ),
        ]

        for proto in prototypes:
            self.prototypes[proto.prototype_id] = proto

    def _load_default_mundane(self):
        """Load default mundane activities."""
        activities = [
            MundaneEntry(
                activity_id="morning_coffee",
                category="morning_routine",
                name="Morning coffee",
                description="Consuming caffeinated beverage after waking",
                typical_duration_minutes=15,
                interruptibility=0.2,  # Don't interrupt pre-coffee people
                time_windows=[(6.0, 9.0)],
                location_types=["kitchen", "cafe", "office_kitchen"],
                cognitive_impact=-0.3,  # Performance penalty if skipped
            ),
            MundaneEntry(
                activity_id="lunch_break",
                category="eating",
                name="Lunch break",
                description="Midday meal consumption",
                typical_duration_minutes=45,
                interruptibility=0.4,
                time_windows=[(11.5, 14.0)],
                location_types=["cafeteria", "restaurant", "break_room"],
                cognitive_impact=-0.2,
            ),
            MundaneEntry(
                activity_id="commute",
                category="transition",
                name="Commute",
                description="Travel between home and work",
                typical_duration_minutes=30,
                interruptibility=0.1,  # Hard to interrupt in transit
                time_windows=[(7.0, 9.0), (17.0, 19.0)],
                location_types=["transit", "car", "walking"],
                cognitive_impact=0.0,
            ),
            MundaneEntry(
                activity_id="focused_work",
                category="work",
                name="Focused work",
                description="Deep concentration on task",
                typical_duration_minutes=90,
                interruptibility=0.1,  # Major social cost to interrupt
                time_windows=[(9.0, 12.0), (14.0, 17.0)],
                location_types=["office", "desk", "study"],
                cognitive_impact=0.2,  # Boosts cognitive state
                social_acceptability={"workplace": 0.9, "family": 0.4},
            ),
        ]

        for activity in activities:
            self.mundane[activity.activity_id] = activity

    def _load_default_roles(self):
        """Load default role definitions."""
        roles = [
            # Family roles
            RoleEntry(
                role_id="family_parent",
                institution="family",
                name="Parent",
                description="Primary caregiver and authority figure",
                power_level=0.8,
                expected_competence=0.6,
                expected_warmth=0.8,
                typical_goals=["protect_children", "provide_guidance", "maintain_household"],
                normative_behaviors=["nurturing", "teaching", "disciplining"],
                taboo_behaviors=["abandonment", "abuse", "neglect"],
            ),
            RoleEntry(
                role_id="family_child",
                institution="family",
                name="Child",
                description="Dependent family member",
                power_level=0.2,
                expected_competence=0.3,
                expected_warmth=0.7,
                typical_goals=["learn", "play", "seek_approval"],
                normative_behaviors=["obedience", "learning", "expressing_needs"],
                taboo_behaviors=["disrespect", "deception_to_parents"],
            ),

            # Workplace roles
            RoleEntry(
                role_id="workplace_manager",
                institution="workplace",
                name="Manager",
                description="Supervisory role with authority over team",
                power_level=0.7,
                expected_competence=0.8,
                expected_warmth=0.4,
                typical_goals=["team_performance", "project_delivery", "talent_development"],
                normative_behaviors=["decision_making", "delegation", "feedback"],
                taboo_behaviors=["favoritism", "emotional_outbursts", "micromanagement"],
            ),
            RoleEntry(
                role_id="workplace_employee",
                institution="workplace",
                name="Employee",
                description="Individual contributor",
                power_level=0.3,
                expected_competence=0.6,
                expected_warmth=0.5,
                typical_goals=["task_completion", "career_advancement", "skill_development"],
                normative_behaviors=["following_instructions", "collaboration", "reporting"],
                taboo_behaviors=["insubordination", "gossip", "time_theft"],
            ),

            # Political roles
            RoleEntry(
                role_id="political_leader",
                institution="political",
                name="Political leader",
                description="Elected or appointed authority",
                power_level=0.9,
                expected_competence=0.7,
                expected_warmth=0.3,
                typical_goals=["maintain_power", "advance_agenda", "build_coalition"],
                normative_behaviors=["public_speaking", "negotiation", "alliance_building"],
                taboo_behaviors=["public_weakness", "betraying_allies_openly"],
            ),
        ]

        for role in roles:
            self.roles[role.role_id] = role
            if role.institution not in self.roles_by_institution:
                self.roles_by_institution[role.institution] = []
            self.roles_by_institution[role.institution].append(role.role_id)

    def _load_from_path(self):
        """Load additional data from file path."""
        # Would load JSON/YAML files from data_path
        pass

    def query_norms(
        self,
        institution: str,
        role: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[InstitutionalNorm]:
        """
        Query norms applicable to a given context.

        Args:
            institution: The institution type
            role: Optional role filter
            location: Optional location filter

        Returns:
            List of applicable norms
        """
        results = []

        norm_ids = self.norms_by_institution.get(institution, [])
        for norm_id in norm_ids:
            norm_entry = self.norms.get(norm_id)
            if norm_entry is None:
                continue

            # Filter by role
            if role and "all" not in norm_entry.applies_to_roles:
                if role not in norm_entry.applies_to_roles:
                    continue

            # Convert to InstitutionalNorm
            results.append(InstitutionalNorm(
                name=norm_entry.name,
                description=norm_entry.description,
                violation_cost=norm_entry.violation_cost,
                detection_probability=norm_entry.detection_probability,
                applies_to_roles=norm_entry.applies_to_roles,
                context_modifiers=norm_entry.context_modifiers,
            ))

        return results

    def query_prototypes(
        self,
        features: List[str],
        institution: Optional[str] = None
    ) -> List[SocialPrototype]:
        """
        Query social prototypes matching observed features.

        Args:
            features: List of observed features
            institution: Optional institution context

        Returns:
            List of matching prototypes
        """
        results = []

        for proto_entry in self.prototypes.values():
            # Check feature match
            matching_features = set(features) & set(proto_entry.feature_triggers)
            if not matching_features:
                continue

            # Check institution context
            if institution and proto_entry.institution_context:
                if proto_entry.institution_context != institution:
                    continue

            # Compute confidence based on feature overlap
            confidence = len(matching_features) / len(proto_entry.feature_triggers)

            results.append(SocialPrototype(
                name=proto_entry.name,
                warmth_prior=proto_entry.warmth_prior,
                competence_prior=proto_entry.competence_prior,
                feature_triggers=proto_entry.feature_triggers,
                confidence=confidence * proto_entry.confidence,
            ))

        # Sort by confidence
        results.sort(key=lambda p: p.confidence, reverse=True)
        return results

    def query_mundane_expectations(
        self,
        time_of_day: float,
        location_type: str
    ) -> Dict[str, Any]:
        """
        Get expected mundane activity for a context.

        Args:
            time_of_day: Hour of day (0-24)
            location_type: Type of location

        Returns:
            Dictionary with expected activity, interruptibility, etc.
        """
        candidates = []

        for activity in self.mundane.values():
            # Check time window
            in_window = False
            for start, end in activity.time_windows:
                if start <= time_of_day <= end:
                    in_window = True
                    break

            if not in_window:
                continue

            # Check location
            if location_type not in activity.location_types:
                continue

            candidates.append(activity)

        if not candidates:
            return {
                "expected_activity": None,
                "interruptibility": 0.5,
                "cognitive_impact": 0.0,
            }

        # Return most likely activity
        best = candidates[0]
        return {
            "expected_activity": best.name,
            "activity_category": best.category,
            "interruptibility": best.interruptibility,
            "cognitive_impact": best.cognitive_impact,
            "typical_duration": best.typical_duration_minutes,
        }

    def query_role(
        self,
        institution: str,
        role_name: str
    ) -> Optional[RoleEntry]:
        """Get role definition."""
        role_ids = self.roles_by_institution.get(institution, [])
        for role_id in role_ids:
            role = self.roles.get(role_id)
            if role and role.name.lower() == role_name.lower():
                return role
        return None

    def get_power_differential(
        self,
        institution: str,
        role_a: str,
        role_b: str
    ) -> float:
        """
        Compute power differential between two roles.

        Returns: Positive if role_a has more power, negative if role_b.
        """
        role_a_entry = self.query_role(institution, role_a)
        role_b_entry = self.query_role(institution, role_b)

        if role_a_entry is None or role_b_entry is None:
            return 0.0

        return role_a_entry.power_level - role_b_entry.power_level


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class ContextManager:
    """
    High-level interface for sociological context queries.

    This is the API specified in the Architectural Manifest:
        context_manager.get_norms(location="Bank", role="Teller")
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db = SociologicalDatabase(db_path)

    def get_norms(
        self,
        location: str,
        role: Optional[str] = None,
        institution: Optional[str] = None
    ) -> List[InstitutionalNorm]:
        """
        Retrieve applicable norms for a given context.

        Example:
            norms = context_manager.get_norms(
                location="Bank",
                role="Teller",
                institution="economic"
            )
        """
        # Infer institution from location if not provided
        if institution is None:
            institution = self._infer_institution(location)

        return self.db.query_norms(institution, role, location)

    def get_prototypes(
        self,
        features: List[str],
        institution: Optional[str] = None
    ) -> List[SocialPrototype]:
        """
        Retrieve social prototypes matching observed features.

        Example:
            prototypes = context_manager.get_prototypes(
                features=["wearing_suit", "in_office", "confident_posture"]
            )
        """
        return self.db.query_prototypes(features, institution)

    def get_mundane_constraints(
        self,
        time_of_day: float,
        location_type: str
    ) -> Dict[str, Any]:
        """
        Get mundane activity expectations for context.

        Example:
            constraints = context_manager.get_mundane_constraints(
                time_of_day=7.5,  # 7:30 AM
                location_type="kitchen"
            )
            # Returns: {"expected_activity": "morning_routine", "interruptibility": 0.2}
        """
        return self.db.query_mundane_expectations(time_of_day, location_type)

    def get_role_expectations(
        self,
        institution: str,
        role: str
    ) -> Dict[str, Any]:
        """
        Get expectations for a role.

        Returns competence/warmth expectations, typical goals, etc.
        """
        role_entry = self.db.query_role(institution, role)
        if role_entry is None:
            return {}

        return {
            "power_level": role_entry.power_level,
            "expected_competence": role_entry.expected_competence,
            "expected_warmth": role_entry.expected_warmth,
            "typical_goals": role_entry.typical_goals,
            "normative_behaviors": role_entry.normative_behaviors,
            "taboo_behaviors": role_entry.taboo_behaviors,
        }

    def compute_social_cost(
        self,
        action_type: str,
        institution: str,
        agent_role: str,
        target_role: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> float:
        """
        Compute the social cost of an action in context.

        Considers norm violations, role expectations, and mundane constraints.
        """
        cost = 0.0

        # Get applicable norms
        norms = self.db.query_norms(institution, agent_role)

        # Check for norm violations
        for norm in norms:
            # This would do semantic matching in production
            if self._action_violates_norm(action_type, norm, context):
                expected_cost = norm.violation_cost * norm.detection_probability
                cost += expected_cost

        # Power differential costs
        if target_role:
            power_diff = self.db.get_power_differential(institution, agent_role, target_role)
            if power_diff < 0:  # Acting against higher power
                cost += abs(power_diff) * 0.2

        # Mundane interruption costs
        if context and "time_of_day" in context:
            mundane = self.get_mundane_constraints(
                context["time_of_day"],
                context.get("location_type", "unknown")
            )
            if action_type in ["interrupt", "request", "demand"]:
                cost += (1 - mundane.get("interruptibility", 0.5)) * 0.3

        return min(cost, 1.0)

    def _infer_institution(self, location: str) -> str:
        """Infer institution type from location."""
        location_lower = location.lower()

        mapping = {
            "home": "family",
            "house": "family",
            "kitchen": "family",
            "office": "workplace",
            "workplace": "workplace",
            "meeting_room": "workplace",
            "bank": "economic",
            "store": "economic",
            "school": "education",
            "classroom": "education",
            "hospital": "healthcare",
            "clinic": "healthcare",
            "church": "religious",
            "temple": "religious",
            "court": "legal",
            "police_station": "legal",
            "parliament": "political",
            "city_hall": "political",
        }

        for key, inst in mapping.items():
            if key in location_lower:
                return inst

        return "public"

    def _action_violates_norm(
        self,
        action_type: str,
        norm: InstitutionalNorm,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if an action violates a norm."""
        # Simplified heuristic matching
        violation_map = {
            "lie": ["honesty", "truthfulness", "transparency"],
            "interrupt": ["respect", "professionalism", "courtesy"],
            "emotional_outburst": ["professionalism", "composure", "demeanor"],
            "skip_greeting": ["courtesy", "respect"],
            "share_confidential": ["confidentiality", "privacy"],
        }

        if action_type in violation_map:
            for keyword in violation_map[action_type]:
                if keyword in norm.name.lower() or keyword in norm.description.lower():
                    return True

        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NormEntry",
    "PrototypeEntry",
    "MundaneEntry",
    "RoleEntry",
    "SociologicalDatabase",
    "ContextManager",
]
