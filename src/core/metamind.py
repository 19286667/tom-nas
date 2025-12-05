"""
MetaMind: 3-Stage Theory of Mind Reasoning Pipeline
====================================================

This module implements the MetaMind cognitive architecture that replaces
naive LLM calls with structured ToM reasoning. The pipeline has three stages:

Stage 1: ToM Agent (Hypothesis Generation)
    - Input: BeliefNest graph + current observation
    - Output: Set of hypotheses about other agents' mental states

Stage 2: Domain Agent (Institutional Filtering)
    - Input: Hypotheses + Current Institution + Role context
    - Output: Institutionally-filtered hypotheses

Stage 3: Response Agent (Action Selection)
    - Input: Filtered hypotheses + Goal + Social cost function
    - Output: Selected action

This is NOT a chatbot. This is a structured reasoning system.

Author: ToM-NAS Project
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging

from .beliefs import BeliefNetwork, RecursiveBeliefState, Belief

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Observation:
    """
    A grounded observation from the physical world (Godot).
    """
    observer_id: int
    timestamp: float

    # What was observed
    observed_entity_id: int
    observed_entity_type: str  # "agent", "object", "location"
    observed_entity_name: str

    # Physical state
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]

    # Semantic features (from Symbol Grounding)
    semantic_tags: List[str] = field(default_factory=list)
    affordances: List[str] = field(default_factory=list)

    # Context
    location_type: str = "unknown"
    institution_context: str = "none"
    mundane_context: str = "none"  # e.g., "morning_routine", "eating"

    # Additional features
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MentalStateHypothesis:
    """
    A hypothesis about another agent's mental state.
    """
    target_agent_id: int
    hypothesis_id: str

    # The hypothesis content
    state_type: str  # "belief", "goal", "emotion", "intention"
    content: str     # Natural language description
    content_vector: Optional[torch.Tensor] = None  # Encoded representation

    # Confidence
    probability: float = 0.5
    evidence: List[str] = field(default_factory=list)

    # ToM depth required
    tom_order: int = 1  # 1 = "I believe X", 2 = "I believe you believe X"

    # Competing hypotheses
    alternatives: List[str] = field(default_factory=list)


@dataclass
class InstitutionalContext:
    """
    The institutional context constraining behavior.
    """
    institution_type: str  # "family", "workplace", "legal", etc.
    location: str

    # Agent's role in this institution
    agent_role: str
    target_role: Optional[str] = None

    # Active norms
    explicit_norms: List[str] = field(default_factory=list)
    implicit_norms: List[str] = field(default_factory=list)

    # Power dynamics
    power_differential: float = 0.0  # -1 = agent subordinate, +1 = agent dominant

    # Information state
    information_asymmetry: float = 0.0  # How much private info exists


@dataclass
class ActionCandidate:
    """
    A candidate action to be evaluated.
    """
    action_id: str
    action_type: str  # "move", "speak", "interact", "observe", "wait"

    # Action parameters
    target_entity_id: Optional[int] = None
    target_position: Optional[Tuple[float, float, float]] = None
    utterance: Optional[str] = None
    interaction_type: Optional[str] = None

    # Expected outcomes
    expected_goal_progress: float = 0.0
    expected_social_cost: float = 0.0
    expected_information_gain: float = 0.0

    # Risk assessment
    norm_violation_risk: float = 0.0
    relationship_damage_risk: float = 0.0

    # Metadata
    reasoning: str = ""


@dataclass
class MetaMindDecision:
    """
    The output of the MetaMind pipeline.
    """
    selected_action: ActionCandidate
    hypotheses_considered: List[MentalStateHypothesis]
    norms_applied: List[str]
    tom_depth_used: int
    confidence: float
    mental_simulation_count: int
    reasoning_trace: List[str] = field(default_factory=list)


# =============================================================================
# BELIEF NEST WRAPPER
# =============================================================================

class BeliefNest:
    """
    Wrapper around BeliefNetwork providing the nested belief API.

    This provides the API specified in the Architectural Manifest:
        agent.belief_system.add_belief(
            subject="Bob",
            predicate="is_holding",
            object="Gun",
            nesting_level=2
        )
    """

    def __init__(self, belief_network: BeliefNetwork, agent_id: int):
        self.network = belief_network
        self.agent_id = agent_id

        # Name-to-index mapping
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}

        # Predicate encoding
        self._predicate_encoder = PredicateEncoder()

    def register_agent(self, name: str, idx: int):
        """Register an agent name-index mapping."""
        self._name_to_idx[name] = idx
        self._idx_to_name[idx] = name

    def add_belief(
        self,
        subject: str,
        predicate: str,
        obj: str,
        nesting_level: int = 1,
        confidence: float = 1.0,
        evidence: List[str] = None,
        source: str = "observation"
    ) -> bool:
        """
        Add a nested belief.

        Args:
            subject: Who/what the belief is about
            predicate: The relation (e.g., "is_holding", "believes", "wants")
            obj: The object of the belief
            nesting_level: ToM order (1 = direct belief, 2 = meta-belief)
            confidence: Certainty level
            evidence: Supporting observations
            source: Where this belief came from

        Returns:
            True if belief was added successfully
        """
        # Get target index
        target_idx = self._name_to_idx.get(subject)
        if target_idx is None:
            # Unknown agent - create placeholder
            target_idx = len(self._name_to_idx)
            self.register_agent(subject, target_idx)

        # Encode belief content
        content = self._predicate_encoder.encode(predicate, obj)

        # Add to network
        return self.network.update_agent_belief(
            agent_id=self.agent_id,
            order=nesting_level,
            target=target_idx,
            content=content,
            confidence=confidence,
            source=source
        )

    def query_belief(
        self,
        belief_path: List[str],
        predicate: Optional[str] = None
    ) -> Optional[Belief]:
        """
        Query a nested belief.

        Args:
            belief_path: Path of agents, e.g., ["me", "bob", "alice"]
                        means "my belief about bob's belief about alice"
            predicate: Optional filter for specific predicate

        Returns:
            Belief if found, None otherwise
        """
        if len(belief_path) < 2:
            return None

        nesting_level = len(belief_path) - 1
        target_name = belief_path[-1]
        target_idx = self._name_to_idx.get(target_name)

        if target_idx is None:
            return None

        belief_state = self.network.get_agent_belief_state(self.agent_id)
        if belief_state is None:
            return None

        return belief_state.get_belief(nesting_level, target_idx)

    def get_all_beliefs_about(self, target: str) -> Dict[int, Belief]:
        """Get all beliefs at all nesting levels about a target."""
        target_idx = self._name_to_idx.get(target)
        if target_idx is None:
            return {}

        belief_state = self.network.get_agent_belief_state(self.agent_id)
        if belief_state is None:
            return {}

        result = {}
        for order in range(belief_state.max_order + 1):
            belief = belief_state.get_belief(order, target_idx)
            if belief is not None:
                result[order] = belief

        return result

    def get_contradictions(self) -> List[Tuple[Belief, Belief]]:
        """Find contradictory beliefs."""
        contradictions = []
        belief_state = self.network.get_agent_belief_state(self.agent_id)
        if belief_state is None:
            return contradictions

        # Check for beliefs that contradict each other
        # This is a simplified implementation
        for order in range(belief_state.max_order + 1):
            beliefs_at_order = belief_state.beliefs[order]
            for target_a, belief_a in beliefs_at_order.items():
                for target_b, belief_b in beliefs_at_order.items():
                    if target_a != target_b and belief_a and belief_b:
                        if self._beliefs_contradict(belief_a, belief_b):
                            contradictions.append((belief_a, belief_b))

        return contradictions

    def _beliefs_contradict(self, a: Belief, b: Belief) -> bool:
        """Check if two beliefs contradict."""
        if a.content is None or b.content is None:
            return False
        # Simple: high cosine similarity with opposite sign indicates contradiction
        similarity = torch.cosine_similarity(
            a.content.unsqueeze(0),
            b.content.unsqueeze(0)
        ).item()
        return similarity < -0.5


class PredicateEncoder:
    """Encodes predicates and objects to tensor representations."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._predicate_embeddings: Dict[str, torch.Tensor] = {}
        self._object_embeddings: Dict[str, torch.Tensor] = {}

    def encode(self, predicate: str, obj: str) -> torch.Tensor:
        """Encode a predicate-object pair."""
        # Get or create predicate embedding
        if predicate not in self._predicate_embeddings:
            self._predicate_embeddings[predicate] = self._hash_embed(predicate)

        # Get or create object embedding
        if obj not in self._object_embeddings:
            self._object_embeddings[obj] = self._hash_embed(obj)

        # Combine
        pred_emb = self._predicate_embeddings[predicate]
        obj_emb = self._object_embeddings[obj]

        # Simple combination: concatenate and project
        combined = torch.cat([pred_emb, obj_emb])
        return combined[:self.dim]  # Truncate to dim

    def _hash_embed(self, text: str) -> torch.Tensor:
        """Create a deterministic embedding from text."""
        # Use hash for reproducibility
        h = hash(text)
        np.random.seed(h % (2**32))
        return torch.from_numpy(np.random.randn(self.dim).astype(np.float32))


# =============================================================================
# METAMIND STAGES
# =============================================================================

class ToMAgent:
    """
    Stage 1: Theory of Mind Agent (Hypothesis Generation)

    Generates hypotheses about other agents' mental states from observations.
    Uses contextual cues, physical observations, and institutional knowledge.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_hypotheses = self.config.get("max_hypotheses", 5)
        self.diversity_pressure = self.config.get("diversity_pressure", 0.3)

        # Context manager for prototype-based inference (lazy loaded)
        self._context_manager = None

    def _get_context_manager(self):
        """Lazy load context manager for norm and prototype lookup."""
        if self._context_manager is None:
            try:
                from .context_manager import ContextManager
                self._context_manager = ContextManager()
            except ImportError:
                logger.warning("ContextManager not available for hypothesis generation")
        return self._context_manager

    def generate_hypotheses(
        self,
        observation: Observation,
        belief_nest: BeliefNest,
        prior_hypotheses: List[MentalStateHypothesis] = None
    ) -> List[MentalStateHypothesis]:
        """
        Generate hypotheses about an observed agent's mental state.

        This implements the "Hypothetical Minds" approach:
        generate multiple competing explanations for observed behavior.
        """
        hypotheses = []

        if observation.observed_entity_type != "agent":
            return hypotheses

        target_id = observation.observed_entity_id

        # Generate belief hypotheses
        belief_hyps = self._generate_belief_hypotheses(
            observation, belief_nest, target_id
        )
        hypotheses.extend(belief_hyps)

        # Generate goal hypotheses
        goal_hyps = self._generate_goal_hypotheses(
            observation, belief_nest, target_id
        )
        hypotheses.extend(goal_hyps)

        # Generate emotion hypotheses
        emotion_hyps = self._generate_emotion_hypotheses(
            observation, belief_nest, target_id
        )
        hypotheses.extend(emotion_hyps)

        # Generate intention hypotheses
        intention_hyps = self._generate_intention_hypotheses(
            observation, belief_nest, target_id
        )
        hypotheses.extend(intention_hyps)

        # Enforce diversity
        hypotheses = self._enforce_diversity(hypotheses)

        # Limit count
        hypotheses = sorted(hypotheses, key=lambda h: h.probability, reverse=True)
        return hypotheses[:self.max_hypotheses]

    def _generate_belief_hypotheses(
        self,
        obs: Observation,
        belief_nest: BeliefNest,
        target_id: int
    ) -> List[MentalStateHypothesis]:
        """Generate hypotheses about what the target believes."""
        hyps = []

        # Hypothesis: Target has accurate beliefs about their environment
        hyps.append(MentalStateHypothesis(
            target_agent_id=target_id,
            hypothesis_id=f"belief_accurate_{target_id}",
            state_type="belief",
            content=f"Agent {target_id} has accurate environmental beliefs",
            probability=0.6,
            tom_order=2,
            evidence=[f"location: {obs.location_type}"],
        ))

        # Hypothesis: Target has false beliefs (for deception scenarios)
        hyps.append(MentalStateHypothesis(
            target_agent_id=target_id,
            hypothesis_id=f"belief_false_{target_id}",
            state_type="belief",
            content=f"Agent {target_id} may have false beliefs",
            probability=0.3,
            tom_order=2,
            evidence=["no direct observation of their perception"],
        ))

        return hyps

    def _generate_goal_hypotheses(
        self,
        obs: Observation,
        belief_nest: BeliefNest,
        target_id: int
    ) -> List[MentalStateHypothesis]:
        """Generate hypotheses about what the target wants."""
        hyps = []

        # Infer from velocity - moving agents have destinations
        velocity_magnitude = np.linalg.norm(obs.velocity)

        if velocity_magnitude > 0.5:
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"goal_destination_{target_id}",
                state_type="goal",
                content=f"Agent {target_id} is moving toward a destination",
                probability=0.7,
                tom_order=1,
                evidence=[f"velocity: {velocity_magnitude:.2f}"],
            ))

        # Context-based goal inference from mundane expectations
        ctx = self._get_context_manager()
        if ctx:
            # Get expected activities for this time/location
            mundane = ctx.get_mundane_constraints(
                time_of_day=obs.timestamp % 24,  # Convert to hour of day
                location_type=obs.location_type,
            )

            expected_activity = mundane.get("expected_activity")
            if expected_activity:
                hyps.append(MentalStateHypothesis(
                    target_agent_id=target_id,
                    hypothesis_id=f"goal_mundane_{target_id}",
                    state_type="goal",
                    content=f"Agent {target_id} wants to complete {expected_activity}",
                    probability=0.6,
                    tom_order=1,
                    evidence=[f"expected_activity: {expected_activity}", f"location: {obs.location_type}"],
                ))

        # Institution-specific goal inference
        institution_goals = {
            "family": ["maintain_relationships", "provide_support", "share_resources"],
            "workplace": ["complete_tasks", "advance_career", "maintain_reputation"],
            "education": ["learn_material", "demonstrate_competence", "social_belonging"],
            "political": ["build_coalition", "maintain_power", "advance_agenda"],
            "economic_market": ["complete_transaction", "maximize_value", "build_trust"],
        }

        if obs.institution_context in institution_goals:
            goals = institution_goals[obs.institution_context]
            # Pick most likely goal based on context
            primary_goal = goals[0]
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"goal_institutional_{target_id}",
                state_type="goal",
                content=f"Agent {target_id} wants to {primary_goal}",
                probability=0.65,
                tom_order=1,
                evidence=[f"institution: {obs.institution_context}"],
            ))

        # Mundane context-based goal
        if obs.mundane_context == "morning_routine":
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"goal_routine_{target_id}",
                state_type="goal",
                content=f"Agent {target_id} wants to complete morning routine",
                probability=0.6,
                tom_order=1,
                evidence=[f"context: {obs.mundane_context}"],
            ))

        return hyps

    def _generate_emotion_hypotheses(
        self,
        obs: Observation,
        belief_nest: BeliefNest,
        target_id: int
    ) -> List[MentalStateHypothesis]:
        """Generate hypotheses about what the target is feeling."""
        hyps = []

        # Infer emotional state from physical cues
        velocity_magnitude = np.linalg.norm(obs.velocity)

        # High velocity might indicate urgency/stress
        if velocity_magnitude > 2.0:
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"emotion_rushed_{target_id}",
                state_type="emotion",
                content=f"Agent {target_id} appears rushed or stressed",
                probability=0.6,
                tom_order=1,
                evidence=[f"high_velocity: {velocity_magnitude:.2f}"],
            ))
        elif velocity_magnitude < 0.1:
            # Very low velocity might indicate rest, contemplation, or waiting
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"emotion_calm_{target_id}",
                state_type="emotion",
                content=f"Agent {target_id} appears calm or at rest",
                probability=0.55,
                tom_order=1,
                evidence=["stationary_position"],
            ))
        else:
            # Default: neutral emotional state
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"emotion_neutral_{target_id}",
                state_type="emotion",
                content=f"Agent {target_id} is in a neutral emotional state",
                probability=0.5,
                tom_order=1,
                evidence=["default prior"],
            ))

        # Use prototypes from context manager for stereotype-based inference
        ctx = self._get_context_manager()
        if ctx and obs.semantic_tags:
            prototypes = ctx.get_prototypes(obs.semantic_tags, obs.institution_context)
            for proto in prototypes[:2]:  # Consider top 2 matching prototypes
                # High warmth prototypes suggest positive emotions
                if proto.warmth_prior > 0.7:
                    hyps.append(MentalStateHypothesis(
                        target_agent_id=target_id,
                        hypothesis_id=f"emotion_positive_{target_id}_{proto.name}",
                        state_type="emotion",
                        content=f"Agent {target_id} likely has positive affect (matches {proto.name} prototype)",
                        probability=proto.confidence * 0.6,
                        tom_order=1,
                        evidence=[f"prototype: {proto.name}", f"warmth: {proto.warmth_prior:.2f}"],
                    ))
                elif proto.warmth_prior < 0.3:
                    hyps.append(MentalStateHypothesis(
                        target_agent_id=target_id,
                        hypothesis_id=f"emotion_negative_{target_id}_{proto.name}",
                        state_type="emotion",
                        content=f"Agent {target_id} may have negative or guarded affect (matches {proto.name} prototype)",
                        probability=proto.confidence * 0.5,
                        tom_order=1,
                        evidence=[f"prototype: {proto.name}", f"warmth: {proto.warmth_prior:.2f}"],
                    ))

        # Context-based emotion inference from features
        features_str = str(obs.features).lower()
        if "dropped" in features_str:
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"emotion_embarrassed_{target_id}",
                state_type="emotion",
                content=f"Agent {target_id} may feel embarrassed",
                probability=0.6,
                tom_order=1,
                evidence=["observed dropping object"],
            ))

        if "laughing" in features_str or "smiling" in features_str:
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"emotion_happy_{target_id}",
                state_type="emotion",
                content=f"Agent {target_id} appears happy or amused",
                probability=0.75,
                tom_order=1,
                evidence=["positive facial expression"],
            ))

        if "frowning" in features_str or "crossed_arms" in features_str:
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"emotion_negative_{target_id}",
                state_type="emotion",
                content=f"Agent {target_id} may be displeased or defensive",
                probability=0.6,
                tom_order=1,
                evidence=["negative body language"],
            ))

        return hyps

    def _generate_intention_hypotheses(
        self,
        obs: Observation,
        belief_nest: BeliefNest,
        target_id: int
    ) -> List[MentalStateHypothesis]:
        """Generate hypotheses about what the target intends to do."""
        hyps = []

        velocity_magnitude = np.linalg.norm(obs.velocity)

        # Infer from movement patterns
        if velocity_magnitude > 0.5:
            # Moving - infer destination-oriented intention
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"intention_destination_{target_id}",
                state_type="intention",
                content=f"Agent {target_id} intends to reach a specific location",
                probability=0.7,
                tom_order=1,
                evidence=[f"moving: {velocity_magnitude:.2f}"],
            ))
        else:
            # Stationary - infer waiting or observing intention
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"intention_waiting_{target_id}",
                state_type="intention",
                content=f"Agent {target_id} intends to wait or observe",
                probability=0.55,
                tom_order=1,
                evidence=["stationary"],
            ))

        # Institution-specific intentions
        intention_by_institution = {
            "family": [
                ("communicate", "Agent intends to communicate with family members"),
                ("help", "Agent intends to help or support family members"),
            ],
            "workplace": [
                ("complete_task", "Agent intends to complete a work task"),
                ("collaborate", "Agent intends to collaborate with colleagues"),
                ("report", "Agent intends to report or communicate status"),
            ],
            "education": [
                ("learn", "Agent intends to learn or acquire information"),
                ("demonstrate", "Agent intends to demonstrate knowledge"),
            ],
            "political": [
                ("negotiate", "Agent intends to negotiate or persuade"),
                ("build_alliance", "Agent intends to build or maintain alliances"),
            ],
            "economic_market": [
                ("transact", "Agent intends to complete a transaction"),
                ("evaluate", "Agent intends to evaluate options"),
            ],
        }

        if obs.institution_context in intention_by_institution:
            intentions = intention_by_institution[obs.institution_context]
            for intent_type, description in intentions[:2]:  # Top 2
                hyps.append(MentalStateHypothesis(
                    target_agent_id=target_id,
                    hypothesis_id=f"intention_{intent_type}_{target_id}",
                    state_type="intention",
                    content=f"{description.replace('Agent', f'Agent {target_id}')}",
                    probability=0.55,
                    tom_order=1,
                    evidence=[f"institution: {obs.institution_context}"],
                ))

        # Default: continue current activity
        hyps.append(MentalStateHypothesis(
            target_agent_id=target_id,
            hypothesis_id=f"intention_continue_{target_id}",
            state_type="intention",
            content=f"Agent {target_id} intends to continue current activity",
            probability=0.5,
            tom_order=1,
            evidence=["behavior continuation prior"],
        ))

        # Social intention based on proximity
        position_magnitude = np.linalg.norm(obs.position)
        if position_magnitude < 3.0:  # Close proximity
            hyps.append(MentalStateHypothesis(
                target_agent_id=target_id,
                hypothesis_id=f"intention_interact_{target_id}",
                state_type="intention",
                content=f"Agent {target_id} intends to interact (close proximity)",
                probability=0.6,
                tom_order=1,
                evidence=[f"proximity: {position_magnitude:.2f}"],
            ))

        return hyps

    def _enforce_diversity(
        self,
        hypotheses: List[MentalStateHypothesis]
    ) -> List[MentalStateHypothesis]:
        """Ensure hypothesis set is diverse."""
        # Group by type
        by_type: Dict[str, List[MentalStateHypothesis]] = {}
        for h in hypotheses:
            if h.state_type not in by_type:
                by_type[h.state_type] = []
            by_type[h.state_type].append(h)

        # Take top from each type
        diverse = []
        for type_hyps in by_type.values():
            sorted_hyps = sorted(type_hyps, key=lambda h: h.probability, reverse=True)
            diverse.extend(sorted_hyps[:2])  # Max 2 per type

        return diverse


class DomainAgent:
    """
    Stage 2: Domain Agent (Institutional Filtering)

    Applies institutional norms and role constraints to filter hypotheses.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.norm_weight = self.config.get("norm_weight", 0.7)
        self.role_weight = self.config.get("role_weight", 0.5)

    def apply_institutional_filter(
        self,
        hypotheses: List[MentalStateHypothesis],
        context: InstitutionalContext
    ) -> List[MentalStateHypothesis]:
        """
        Filter and adjust hypotheses based on institutional context.

        Norms constrain what mental states are likely/acceptable.
        """
        filtered = []

        for hyp in hypotheses:
            adjusted = self._adjust_for_norms(hyp, context)
            adjusted = self._adjust_for_role(adjusted, context)
            filtered.append(adjusted)

        # Re-normalize probabilities
        total_prob = sum(h.probability for h in filtered)
        if total_prob > 0:
            for h in filtered:
                h.probability /= total_prob

        return filtered

    def _adjust_for_norms(
        self,
        hyp: MentalStateHypothesis,
        context: InstitutionalContext
    ) -> MentalStateHypothesis:
        """Adjust hypothesis probability based on norms."""

        # Example norm effects
        if context.institution_type == "workplace":
            # In workplace, emotional display is suppressed
            if hyp.state_type == "emotion" and "embarrassed" in hyp.content:
                # Managers expected to maintain composure
                if context.target_role == "manager":
                    hyp.probability *= 1.2  # More notable if violated
                    hyp.evidence.append("norm_violation: managers maintain composure")

        if context.institution_type == "family":
            # In family, emotional expression is expected
            if hyp.state_type == "emotion":
                hyp.probability *= 1.1  # Emotions more likely expressed

        return hyp

    def _adjust_for_role(
        self,
        hyp: MentalStateHypothesis,
        context: InstitutionalContext
    ) -> MentalStateHypothesis:
        """Adjust hypothesis based on role expectations."""

        # Power dynamics affect mental state likelihood
        if context.power_differential > 0.5:  # Agent is dominant
            if hyp.state_type == "emotion" and "fear" in hyp.content.lower():
                hyp.probability *= 0.5  # Subordinate fears more likely

        return hyp


class ResponseAgent:
    """
    Stage 3: Response Agent (Action Selection)

    Selects optimal action based on filtered hypotheses and social costs.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.temperature = self.config.get("temperature", 0.3)
        self.social_cost_weight = self.config.get("social_cost_weight", 0.6)
        self.simulation_budget = self.config.get("simulation_budget", 10)

    def select_action(
        self,
        hypotheses: List[MentalStateHypothesis],
        context: InstitutionalContext,
        goal: str,
        available_actions: List[ActionCandidate]
    ) -> Tuple[ActionCandidate, List[str]]:
        """
        Select the best action given hypotheses about others' mental states.

        Uses mental simulation to predict outcomes.
        """
        reasoning_trace = []

        # Score each action
        scored_actions = []
        for action in available_actions:
            score, trace = self._score_action(
                action, hypotheses, context, goal
            )
            scored_actions.append((score, action))
            reasoning_trace.extend(trace)

        # Select (with temperature for exploration)
        scores = np.array([s for s, _ in scored_actions])
        if self.temperature > 0:
            # Softmax selection
            exp_scores = np.exp(scores / self.temperature)
            probs = exp_scores / exp_scores.sum()
            idx = np.random.choice(len(scored_actions), p=probs)
        else:
            idx = np.argmax(scores)

        selected = scored_actions[idx][1]
        reasoning_trace.append(f"Selected: {selected.action_type} (score: {scores[idx]:.2f})")

        return selected, reasoning_trace

    def _score_action(
        self,
        action: ActionCandidate,
        hypotheses: List[MentalStateHypothesis],
        context: InstitutionalContext,
        goal: str
    ) -> Tuple[float, List[str]]:
        """Score an action based on expected outcomes."""
        trace = []

        # Base score from goal progress
        goal_score = action.expected_goal_progress
        trace.append(f"{action.action_type}: goal_score={goal_score:.2f}")

        # Social cost penalty
        social_penalty = action.expected_social_cost * self.social_cost_weight
        trace.append(f"{action.action_type}: social_penalty={social_penalty:.2f}")

        # Norm violation risk
        norm_penalty = action.norm_violation_risk * context.power_differential
        trace.append(f"{action.action_type}: norm_penalty={norm_penalty:.2f}")

        # Mental simulation: predict others' reactions
        reaction_score = self._simulate_reactions(action, hypotheses)
        trace.append(f"{action.action_type}: reaction_score={reaction_score:.2f}")

        total = goal_score - social_penalty - norm_penalty + reaction_score
        return total, trace

    def _simulate_reactions(
        self,
        action: ActionCandidate,
        hypotheses: List[MentalStateHypothesis]
    ) -> float:
        """Simulate how others might react to this action."""
        reaction_score = 0.0

        for hyp in hypotheses:
            # If they're embarrassed and we help, positive
            if "embarrassed" in hyp.content and action.action_type == "help":
                reaction_score += 0.3 * hyp.probability

            # If they want privacy and we intrude, negative
            if "privacy" in hyp.content and action.action_type == "interact":
                reaction_score -= 0.4 * hyp.probability

        return reaction_score


# =============================================================================
# METAMIND PIPELINE
# =============================================================================

class MetaMindPipeline:
    """
    The complete MetaMind 3-stage reasoning pipeline.

    This replaces direct LLM calls with structured ToM reasoning.
    """

    def __init__(
        self,
        belief_network: BeliefNetwork,
        agent_id: int,
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        self.agent_id = agent_id

        # Initialize BeliefNest wrapper
        self.belief_nest = BeliefNest(belief_network, agent_id)

        # Initialize stages
        self.tom_agent = ToMAgent(self.config.get("tom_agent", {}))
        self.domain_agent = DomainAgent(self.config.get("domain_agent", {}))
        self.response_agent = ResponseAgent(self.config.get("response_agent", {}))

        # Tracking
        self.decision_history: List[MetaMindDecision] = []
        self.max_tom_depth_used: int = 0

    def reason(
        self,
        observation: Observation,
        goal: str,
        context: InstitutionalContext,
        available_actions: List[ActionCandidate]
    ) -> MetaMindDecision:
        """
        Execute the full MetaMind pipeline.

        This is the main entry point for agent reasoning.
        """
        reasoning_trace = []
        reasoning_trace.append(f"Observation: {observation.observed_entity_name}")
        reasoning_trace.append(f"Goal: {goal}")
        reasoning_trace.append(f"Institution: {context.institution_type}")

        # Stage 1: Generate hypotheses
        hypotheses = self.tom_agent.generate_hypotheses(
            observation, self.belief_nest
        )
        reasoning_trace.append(f"Generated {len(hypotheses)} hypotheses")

        # Track ToM depth
        max_tom = max((h.tom_order for h in hypotheses), default=0)
        self.max_tom_depth_used = max(self.max_tom_depth_used, max_tom)

        # Stage 2: Apply institutional filter
        filtered_hypotheses = self.domain_agent.apply_institutional_filter(
            hypotheses, context
        )
        reasoning_trace.append(f"Filtered to {len(filtered_hypotheses)} hypotheses")

        # Stage 3: Select action
        selected_action, action_trace = self.response_agent.select_action(
            filtered_hypotheses, context, goal, available_actions
        )
        reasoning_trace.extend(action_trace)

        # Create decision record
        decision = MetaMindDecision(
            selected_action=selected_action,
            hypotheses_considered=filtered_hypotheses,
            norms_applied=context.explicit_norms + context.implicit_norms,
            tom_depth_used=max_tom,
            confidence=max((h.probability for h in filtered_hypotheses), default=0.5),
            mental_simulation_count=len(available_actions),
            reasoning_trace=reasoning_trace,
        )

        self.decision_history.append(decision)
        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "decisions_made": len(self.decision_history),
            "max_tom_depth_used": self.max_tom_depth_used,
            "avg_hypotheses_per_decision": np.mean([
                len(d.hypotheses_considered) for d in self.decision_history
            ]) if self.decision_history else 0,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_metamind_pipeline(
    belief_network: BeliefNetwork,
    agent_id: int,
    **kwargs
) -> MetaMindPipeline:
    """Factory function for creating MetaMind pipelines."""
    config = {
        "tom_agent": {
            "max_hypotheses": kwargs.get("max_hypotheses", 5),
            "diversity_pressure": kwargs.get("diversity_pressure", 0.3),
        },
        "domain_agent": {
            "norm_weight": kwargs.get("norm_weight", 0.7),
            "role_weight": kwargs.get("role_weight", 0.5),
        },
        "response_agent": {
            "temperature": kwargs.get("temperature", 0.3),
            "social_cost_weight": kwargs.get("social_cost_weight", 0.6),
            "simulation_budget": kwargs.get("simulation_budget", 10),
        },
    }
    return MetaMindPipeline(belief_network, agent_id, config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "Observation",
    "MentalStateHypothesis",
    "InstitutionalContext",
    "ActionCandidate",
    "MetaMindDecision",
    # Core classes
    "BeliefNest",
    "PredicateEncoder",
    "ToMAgent",
    "DomainAgent",
    "ResponseAgent",
    "MetaMindPipeline",
    # Factory
    "create_metamind_pipeline",
]
