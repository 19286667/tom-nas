"""
Psychosocial Co-Evolution Engine
================================

This module implements genuine bidirectional co-evolution between:
1. NAS-evolved agents developing Theory of Mind
2. Psychosocially complex environments that adapt to agent strategies

The key insight: environments that don't evolve become trivially solvable.
True ToM requires navigating genuinely unpredictable social dynamics.

THEORETICAL FOUNDATIONS
-----------------------
This implementation draws from established research:

- Axelrod's Evolution of Cooperation (1984): Reputation dynamics and reciprocity
- Dunbar's Social Brain Hypothesis (1998): Cognitive limits on social network size
- Frith & Frith's ToM neuroscience (2006): Nested belief modeling
- Heider's Balance Theory (1958): Triadic relationship dynamics
- Festinger's Social Comparison (1954): Identity and status emergence
- Sperber & Wilson's Relevance Theory (1986): Communication pragmatics

DESIGN PRINCIPLES
-----------------
1. BIDIRECTIONALITY: Agents shape environment; environment shapes agents
2. EMERGENCE: Complex patterns arise from simple, principled rules
3. SELF-DOCUMENTATION: Constants explain their theoretical origin
4. SCIENTIFIC VALIDITY: All dynamics grounded in empirical psychology
5. MEANINGFUL COMPLEXITY: Every mechanism serves narrative and science
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from enum import Enum, auto
from collections import defaultdict
import random
import math

from .soul_map import SoulMap, SoulMapDelta
from .npcs.base_npc import BaseNPC


# =============================================================================
# THEORETICAL CONSTANTS
# Each constant is derived from empirical research or principled design
# =============================================================================

class TheoreticalConstants:
    """
    Constants grounded in cognitive science and social psychology.

    Each constant includes its theoretical justification, making the
    system self-documenting and scientifically transparent.
    """

    # DUNBAR'S NUMBER: Cognitive limit on stable social relationships
    # Source: Dunbar, R. I. M. (1992). Neocortex size and group size in primates.
    # Typical human value is ~150; we use scaled version for computational tractability
    DUNBAR_NUMBER = 15  # Scaled for agent populations

    # BELIEF ORDER DECAY: Confidence drops exponentially with recursion depth
    # Source: Kinderman et al. (1998). ToM deficits across psychiatric conditions
    # Human limit is typically 4th-5th order; confidence at each level ~0.7-0.8
    BELIEF_CONFIDENCE_DECAY = 0.73  # Per recursion level

    # HEIDER BALANCE: Triads tend toward balance (friend/enemy consistency)
    # Source: Heider, F. (1958). The Psychology of Interpersonal Relations
    # Balanced triads: (+,+,+), (+,-,-); Imbalanced: (+,+,-), (-,-,-)
    BALANCE_PRESSURE = 0.15  # Per tick tendency toward balance

    # REPUTATION MEMORY: Exponential decay of reputation information
    # Source: Milinski et al. (2002). Reputation helps solve the tragedy of commons
    # Typical human memory half-life ~20-30 interactions
    REPUTATION_DECAY_RATE = 0.03  # Per tick decay

    # EMOTIONAL CONTAGION: Rate of affect spread through social proximity
    # Source: Hatfield et al. (1993). Emotional Contagion
    # Close proximity increases synchrony; scales with relationship strength
    CONTAGION_RATE = 0.08  # Base rate per interaction

    # BELIEF REVISION: How quickly beliefs update given disconfirmation
    # Source: Klayman & Ha (1987). Confirmation bias in hypothesis testing
    BELIEF_REVISION_RATE = 0.12  # Confirmatory evidence
    DISCONFIRMATION_RESISTANCE = 0.65  # Slower update on disconfirmation

    # COALITION STABILITY: Threshold for coalition formation/dissolution
    # Source: Harcourt & de Waal (1992). Coalitions and Alliances
    COALITION_FORMATION_THRESHOLD = 0.6  # Mutual reputation required
    COALITION_DISSOLUTION_THRESHOLD = 0.3  # Below this, coalition breaks

    # STATUS HIERARCHY: Rate of dominance relationship crystallization
    # Source: Sapolsky (2005). The Influence of Social Hierarchy on Primate Health
    HIERARCHY_CRYSTALLIZATION_RATE = 0.02  # Per conflict resolution

    # SOCIAL LEARNING: Rate of behavioral adaptation from observation
    # Source: Bandura (1977). Social Learning Theory
    OBSERVATIONAL_LEARNING_RATE = 0.05  # Per observation

    # COGNITIVE LOAD: Computational cost of deeper ToM reasoning
    # Source: Premack & Woodruff (1978). Does the chimpanzee have a ToM?
    TOM_PROCESSING_COST = [0.0, 0.1, 0.25, 0.45, 0.7, 1.0]  # By order


# =============================================================================
# SOCIAL NETWORK DYNAMICS
# =============================================================================

class RelationshipType(Enum):
    """Types of dyadic relationships between agents."""
    STRANGER = auto()      # No significant history
    ACQUAINTANCE = auto()  # Some interaction history
    ALLY = auto()          # Positive reciprocal relationship
    RIVAL = auto()         # Competitive relationship
    ENEMY = auto()         # Negative reciprocal relationship
    COALITION = auto()     # Formally allied (explicit agreement)


@dataclass
class SocialEdge:
    """
    An edge in the social network representing a relationship.

    Relationships are directional: A's view of B may differ from B's view of A.
    This captures asymmetric social perception, a core ToM challenge.
    """
    source_id: str
    target_id: str

    # Core relationship metrics
    trust: float = 0.5          # Expectation of cooperative behavior
    familiarity: float = 0.0    # Interaction history depth
    affect: float = 0.0         # Emotional valence (-1 to +1)

    # Belief about target's view of source (2nd order ToM)
    perceived_trust: float = 0.5
    perceived_affect: float = 0.0

    # Interaction history
    cooperation_history: List[bool] = field(default_factory=list)
    last_interaction_tick: int = 0

    def get_relationship_type(self) -> RelationshipType:
        """Classify relationship based on current metrics."""
        if self.familiarity < 0.1:
            return RelationshipType.STRANGER
        elif self.familiarity < 0.3:
            return RelationshipType.ACQUAINTANCE
        elif self.trust > TheoreticalConstants.COALITION_FORMATION_THRESHOLD and self.affect > 0.3:
            return RelationshipType.COALITION
        elif self.trust > 0.6 and self.affect > 0:
            return RelationshipType.ALLY
        elif self.trust < 0.4 and self.affect < -0.3:
            return RelationshipType.ENEMY
        elif self.affect < 0:
            return RelationshipType.RIVAL
        else:
            return RelationshipType.ACQUAINTANCE

    def update_from_interaction(self, cooperated: bool, tick: int):
        """
        Update relationship based on interaction outcome.

        Uses Bayesian-like updating with asymmetric learning rates:
        - Positive outcomes build trust slowly
        - Negative outcomes erode trust quickly (negativity bias)

        Source: Baumeister et al. (2001). Bad is stronger than good.
        """
        self.cooperation_history.append(cooperated)
        self.last_interaction_tick = tick

        # Familiarity always increases with interaction
        self.familiarity = min(1.0, self.familiarity + 0.05)

        if cooperated:
            # Slower trust building
            self.trust = min(1.0, self.trust + TheoreticalConstants.BELIEF_REVISION_RATE)
            self.affect = min(1.0, self.affect + 0.05)
        else:
            # Faster trust erosion (negativity bias factor ~2.5)
            self.trust = max(0.0, self.trust - TheoreticalConstants.BELIEF_REVISION_RATE * 2.5)
            self.affect = max(-1.0, self.affect - 0.1)

    def decay(self, current_tick: int):
        """Apply temporal decay to relationship."""
        ticks_since = current_tick - self.last_interaction_tick
        decay_factor = TheoreticalConstants.REPUTATION_DECAY_RATE * ticks_since

        # Trust regresses toward neutral (0.5)
        self.trust = self.trust + (0.5 - self.trust) * min(decay_factor, 0.5)

        # Affect regresses toward neutral (0)
        self.affect = self.affect * max(0.5, 1.0 - decay_factor)


class SocialNetwork:
    """
    Dynamic social network tracking all agent relationships.

    The network evolves through interactions, creating emergent
    social structures (coalitions, hierarchies, cliques) that
    agents must navigate using Theory of Mind.
    """

    def __init__(self):
        self.edges: Dict[Tuple[str, str], SocialEdge] = {}
        self.coalitions: Dict[str, Set[str]] = {}  # coalition_id -> member_ids
        self.hierarchy: Dict[str, float] = {}  # agent_id -> status score

    def get_or_create_edge(self, source: str, target: str) -> SocialEdge:
        """Get existing edge or create new one."""
        key = (source, target)
        if key not in self.edges:
            self.edges[key] = SocialEdge(source_id=source, target_id=target)
        return self.edges[key]

    def record_interaction(self, agent1: str, agent2: str,
                          outcome1_cooperated: bool, outcome2_cooperated: bool,
                          tick: int):
        """Record bilateral interaction between two agents."""
        edge1 = self.get_or_create_edge(agent1, agent2)
        edge2 = self.get_or_create_edge(agent2, agent1)

        edge1.update_from_interaction(outcome2_cooperated, tick)  # A's view based on B's action
        edge2.update_from_interaction(outcome1_cooperated, tick)  # B's view based on A's action

        # Update perceived trust (what A thinks B thinks of A)
        edge1.perceived_trust = edge2.trust
        edge2.perceived_trust = edge1.trust

    def apply_heider_balance(self, tick: int):
        """
        Apply Heider balance pressure to triads.

        Balanced triads are stable; imbalanced triads create pressure
        for relationship change. This creates realistic social dynamics
        where "the enemy of my enemy is my friend."
        """
        agents = list(set(e.source_id for e in self.edges.values()))

        if len(agents) < 3:
            return

        # Sample triads for efficiency
        num_triads = min(100, len(agents) * (len(agents) - 1) // 2)

        for _ in range(num_triads):
            if len(agents) < 3:
                break
            a, b, c = random.sample(agents, 3)

            # Get relationship signs
            ab = self._get_sign(a, b)
            bc = self._get_sign(b, c)
            ac = self._get_sign(a, c)

            # Check balance (product should be positive)
            balance = ab * bc * ac

            if balance < 0:  # Imbalanced triad
                # Apply pressure toward balance
                self._apply_balance_pressure(a, b, c, tick)

    def _get_sign(self, source: str, target: str) -> int:
        """Get relationship sign (+1, -1, or 0 for neutral)."""
        edge = self.edges.get((source, target))
        if edge is None:
            return 0

        if edge.affect > 0.2:
            return 1
        elif edge.affect < -0.2:
            return -1
        return 0

    def _apply_balance_pressure(self, a: str, b: str, c: str, tick: int):
        """Apply Heider balance pressure to an imbalanced triad."""
        # Find the weakest link and adjust it
        edges = [
            (a, b, self.edges.get((a, b))),
            (b, c, self.edges.get((b, c))),
            (a, c, self.edges.get((a, c)))
        ]

        # Find weakest relationship (lowest familiarity)
        edges = [(s, t, e) for s, t, e in edges if e is not None]
        if not edges:
            return

        weakest = min(edges, key=lambda x: x[2].familiarity)

        # Adjust toward balance
        adjustment = TheoreticalConstants.BALANCE_PRESSURE

        # Determine target sign for balance
        other_product = 1
        for s, t, e in edges:
            if (s, t) != (weakest[0], weakest[1]):
                other_product *= 1 if e.affect > 0 else -1

        target_positive = other_product > 0

        if target_positive and weakest[2].affect < 0:
            weakest[2].affect += adjustment
        elif not target_positive and weakest[2].affect > 0:
            weakest[2].affect -= adjustment

    def detect_coalitions(self) -> Dict[str, Set[str]]:
        """
        Detect emergent coalitions from relationship patterns.

        Uses connected component analysis on positive relationships
        that exceed the coalition threshold.
        """
        # Build adjacency for strong positive relationships
        adjacency: Dict[str, Set[str]] = defaultdict(set)

        for (source, target), edge in self.edges.items():
            if (edge.trust > TheoreticalConstants.COALITION_FORMATION_THRESHOLD and
                edge.affect > 0.2):
                # Check reciprocity
                reverse = self.edges.get((target, source))
                if reverse and reverse.trust > TheoreticalConstants.COALITION_FORMATION_THRESHOLD:
                    adjacency[source].add(target)
                    adjacency[target].add(source)

        # Find connected components
        visited = set()
        coalitions = {}
        coalition_id = 0

        for agent in adjacency:
            if agent not in visited:
                # BFS to find component
                component = set()
                queue = [agent]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    component.add(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)

                if len(component) >= 2:  # Minimum coalition size
                    coalitions[f"coalition_{coalition_id}"] = component
                    coalition_id += 1

        self.coalitions = coalitions
        return coalitions

    def update_hierarchy(self, winner: str, loser: str):
        """
        Update status hierarchy based on competitive outcome.

        Uses Elo-like rating system adapted for social status.
        """
        if winner not in self.hierarchy:
            self.hierarchy[winner] = 0.5
        if loser not in self.hierarchy:
            self.hierarchy[loser] = 0.5

        # Transfer status
        transfer = TheoreticalConstants.HIERARCHY_CRYSTALLIZATION_RATE
        expected_winner = 1 / (1 + 10 ** (self.hierarchy[loser] - self.hierarchy[winner]))

        # Update based on outcome vs expectation
        self.hierarchy[winner] += transfer * (1 - expected_winner)
        self.hierarchy[loser] -= transfer * expected_winner

        # Clamp to [0, 1]
        self.hierarchy[winner] = max(0, min(1, self.hierarchy[winner]))
        self.hierarchy[loser] = max(0, min(1, self.hierarchy[loser]))

    def tick(self, current_tick: int):
        """Advance the social network by one tick."""
        # Apply decay to all edges
        for edge in self.edges.values():
            edge.decay(current_tick)

        # Apply Heider balance periodically
        if current_tick % 10 == 0:
            self.apply_heider_balance(current_tick)

        # Detect coalitions periodically
        if current_tick % 50 == 0:
            self.detect_coalitions()


# =============================================================================
# BELIEF PROPAGATION SYSTEM
# =============================================================================

@dataclass
class PropagatingBelief:
    """
    A belief that can spread through the social network.

    Beliefs spread through:
    1. Direct communication (high fidelity, requires trust)
    2. Observation (medium fidelity, no trust required)
    3. Gossip (low fidelity, mediated by relationships)
    """
    belief_id: str
    content: Dict[str, Any]  # What the belief is about
    source_id: str           # Original source
    confidence: float        # Current confidence level
    timestamp: int           # When created

    # Spread tracking
    holders: Set[str] = field(default_factory=set)  # Who holds this belief
    transmission_count: int = 0


class BeliefPropagationEngine:
    """
    Manages the spread of beliefs through the social network.

    This creates emergent information asymmetry that ToM agents
    must navigate - what do others know? What do they think I know?
    """

    def __init__(self, social_network: SocialNetwork):
        self.network = social_network
        self.beliefs: Dict[str, PropagatingBelief] = {}
        self.agent_beliefs: Dict[str, Set[str]] = defaultdict(set)  # agent -> belief_ids

    def introduce_belief(self, belief_id: str, content: Dict[str, Any],
                        source_id: str, tick: int) -> PropagatingBelief:
        """Introduce a new belief into the system."""
        belief = PropagatingBelief(
            belief_id=belief_id,
            content=content,
            source_id=source_id,
            confidence=1.0,
            timestamp=tick,
            holders={source_id}
        )
        self.beliefs[belief_id] = belief
        self.agent_beliefs[source_id].add(belief_id)
        return belief

    def propagate(self, tick: int, interactions: List[Tuple[str, str]]):
        """
        Propagate beliefs based on interactions this tick.

        Transmission probability depends on:
        - Relationship trust (higher trust = more sharing)
        - Belief confidence (confident beliefs spread faster)
        - Social distance (coalition members share more)
        """
        for source, target in interactions:
            edge = self.network.edges.get((source, target))
            if edge is None:
                continue

            # Beliefs source might share
            for belief_id in list(self.agent_beliefs.get(source, set())):
                belief = self.beliefs.get(belief_id)
                if belief is None or target in belief.holders:
                    continue

                # Calculate transmission probability
                trust_factor = edge.trust
                confidence_factor = belief.confidence

                # Coalition bonus
                coalition_bonus = 0.2 if self._in_same_coalition(source, target) else 0.0

                transmission_prob = trust_factor * confidence_factor * 0.3 + coalition_bonus

                if random.random() < transmission_prob:
                    self._transmit_belief(belief_id, source, target, edge.trust)

    def _transmit_belief(self, belief_id: str, source: str, target: str, trust: float):
        """Transmit a belief from source to target."""
        belief = self.beliefs[belief_id]

        # Confidence degrades with transmission (telephone game effect)
        # Source: Kashima (2000). Maintaining Cultural Stereotypes
        degradation = 0.9 * trust  # Higher trust = less degradation

        belief.holders.add(target)
        belief.confidence *= degradation
        belief.transmission_count += 1

        self.agent_beliefs[target].add(belief_id)

    def _in_same_coalition(self, agent1: str, agent2: str) -> bool:
        """Check if two agents are in the same coalition."""
        for coalition in self.network.coalitions.values():
            if agent1 in coalition and agent2 in coalition:
                return True
        return False

    def get_agent_knowledge(self, agent_id: str) -> Dict[str, PropagatingBelief]:
        """Get all beliefs held by an agent."""
        return {
            bid: self.beliefs[bid]
            for bid in self.agent_beliefs.get(agent_id, set())
            if bid in self.beliefs
        }

    def create_knowledge_asymmetry_tensor(self, agents: List[str]) -> torch.Tensor:
        """
        Create a tensor representing knowledge asymmetry.

        This tensor encodes who knows what, enabling ToM reasoning
        about information states of other agents.

        Shape: [num_agents, num_agents] where entry [i,j] represents
        how much agent i knows that agent j doesn't.
        """
        n = len(agents)
        agent_to_idx = {a: i for i, a in enumerate(agents)}

        asymmetry = torch.zeros(n, n)

        for i, agent_i in enumerate(agents):
            beliefs_i = self.agent_beliefs.get(agent_i, set())
            for j, agent_j in enumerate(agents):
                if i == j:
                    continue
                beliefs_j = self.agent_beliefs.get(agent_j, set())

                # What i knows that j doesn't
                exclusive = beliefs_i - beliefs_j
                asymmetry[i, j] = len(exclusive) / max(1, len(beliefs_i))

        return asymmetry


# =============================================================================
# ENVIRONMENTAL CO-EVOLUTION
# =============================================================================

class EnvironmentEvolutionStrategy(Enum):
    """Strategies for environment adaptation."""
    STATIC = auto()           # No adaptation (baseline)
    REACTIVE = auto()         # Respond to agent behavior
    ADVERSARIAL = auto()      # Actively challenge agents
    SCAFFOLDING = auto()      # Gradually increase difficulty
    ECOLOGICAL = auto()       # Realistic evolutionary pressure


@dataclass
class EnvironmentalPressure:
    """
    A pressure that shapes NPC behavior and psychology.

    These pressures evolve based on agent population fitness,
    creating the co-evolutionary dynamic.
    """
    pressure_id: str
    pressure_type: str  # "resource_scarcity", "social_threat", "cognitive_demand"
    intensity: float    # 0-1 scale
    target_dimension: str  # Which soul map dimension this affects

    # Evolution tracking
    generation_introduced: int
    effectiveness: float = 0.5  # How well this pressure differentiates agents


class PsychosocialEnvironmentEvolution:
    """
    The core co-evolution engine.

    This system tracks agent population fitness and evolves the
    environment to maintain selection pressure. The key insight is
    that as agents become better at ToM reasoning, the environment
    must present increasingly sophisticated social challenges.

    EVOLUTION DYNAMICS:
    1. Track agent fitness distribution
    2. Identify which challenges agents have "solved"
    3. Introduce new pressures that require deeper ToM
    4. Remove pressures that no longer differentiate

    This creates an arms race where neither side fully "wins" -
    the hallmark of genuine co-evolution.
    """

    def __init__(
        self,
        strategy: EnvironmentEvolutionStrategy = EnvironmentEvolutionStrategy.ECOLOGICAL,
        evolution_rate: float = 0.1,
    ):
        self.strategy = strategy
        self.evolution_rate = evolution_rate

        # Environmental pressures
        self.pressures: Dict[str, EnvironmentalPressure] = {}
        self.pressure_history: List[Dict[str, float]] = []

        # Agent population tracking
        self.fitness_history: List[List[float]] = []
        self.fitness_variance_history: List[float] = []

        # NPC population parameters (evolve these)
        self.npc_base_parameters: Dict[str, float] = {
            "deception_sophistication": 0.3,  # How complex NPC deception is
            "coalition_fluidity": 0.5,        # How often coalitions change
            "belief_complexity": 0.4,         # Depth of NPC belief systems
            "emotional_volatility": 0.5,      # How much NPC emotions shift
            "communication_noise": 0.2,       # Reliability of NPC communication
        }

        # Generation counter
        self.generation = 0

        # Initialize base pressures
        self._initialize_pressures()

    def _initialize_pressures(self):
        """Initialize starting environmental pressures."""
        base_pressures = [
            EnvironmentalPressure(
                pressure_id="resource_competition",
                pressure_type="resource_scarcity",
                intensity=0.3,
                target_dimension="survival_drive",
                generation_introduced=0
            ),
            EnvironmentalPressure(
                pressure_id="social_trust_challenge",
                pressure_type="social_threat",
                intensity=0.3,
                target_dimension="trust_default",
                generation_introduced=0
            ),
            EnvironmentalPressure(
                pressure_id="belief_reasoning_demand",
                pressure_type="cognitive_demand",
                intensity=0.3,
                target_dimension="tom_depth",
                generation_introduced=0
            ),
        ]

        for pressure in base_pressures:
            self.pressures[pressure.pressure_id] = pressure

    def record_generation_fitness(self, fitness_values: List[float]):
        """Record fitness distribution for current generation."""
        self.fitness_history.append(fitness_values)
        self.fitness_variance_history.append(np.var(fitness_values))

    def evolve_environment(self, agent_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Evolve the environment based on agent population performance.

        Returns a dict of changes to apply to NPCs and environment.
        """
        self.generation += 1

        if self.strategy == EnvironmentEvolutionStrategy.STATIC:
            return {}

        changes = {}

        # Analyze agent performance patterns
        performance_analysis = self._analyze_performance(agent_performance)

        if self.strategy == EnvironmentEvolutionStrategy.ECOLOGICAL:
            changes = self._ecological_evolution(performance_analysis)
        elif self.strategy == EnvironmentEvolutionStrategy.ADVERSARIAL:
            changes = self._adversarial_evolution(performance_analysis)
        elif self.strategy == EnvironmentEvolutionStrategy.SCAFFOLDING:
            changes = self._scaffolding_evolution(performance_analysis)
        elif self.strategy == EnvironmentEvolutionStrategy.REACTIVE:
            changes = self._reactive_evolution(performance_analysis)

        # Record pressure states
        self.pressure_history.append({
            pid: p.intensity for pid, p in self.pressures.items()
        })

        return changes

    def _analyze_performance(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze agent performance to identify adaptation targets."""
        analysis = {
            "mean_fitness": np.mean(list(performance.values())),
            "variance": np.var(list(performance.values())),
            "top_performers": [],
            "struggling_areas": [],
            "solved_challenges": [],
        }

        # Identify which challenges are no longer differentiating
        if len(self.fitness_variance_history) >= 5:
            recent_variance = self.fitness_variance_history[-5:]
            if all(v < 0.05 for v in recent_variance):
                # Low variance = challenges are "solved"
                analysis["solved_challenges"] = list(self.pressures.keys())

        return analysis

    def _ecological_evolution(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ecological evolution: realistic selective pressure.

        Mimics natural selection dynamics where:
        1. Successful strategies create their own challenges
        2. Unused niches open opportunities
        3. Overfitting to one challenge leaves vulnerability
        """
        changes = {}

        # If mean fitness is high, increase environmental complexity
        if analysis["mean_fitness"] > 0.7:
            # Increase sophistication parameters
            for param in self.npc_base_parameters:
                self.npc_base_parameters[param] = min(
                    1.0,
                    self.npc_base_parameters[param] + self.evolution_rate
                )
            changes["npc_parameters"] = dict(self.npc_base_parameters)

            # Introduce new pressure if old ones are solved
            if analysis["solved_challenges"]:
                new_pressure = self._generate_novel_pressure()
                if new_pressure:
                    self.pressures[new_pressure.pressure_id] = new_pressure
                    changes["new_pressure"] = new_pressure.pressure_id

        # If variance is too low, increase diversity
        if analysis["variance"] < 0.1:
            changes["increase_diversity"] = True
            # Add conflicting pressures that favor different strategies
            self._add_diversity_pressure()

        return changes

    def _adversarial_evolution(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adversarial evolution: actively challenge agent strategies.

        Identifies what agents are good at and creates obstacles.
        """
        changes = {}

        # Identify successful agent strategies (from performance patterns)
        # Then create counter-pressures

        for pressure_id, pressure in list(self.pressures.items()):
            # If agents are handling this well, intensify it
            if pressure.effectiveness < 0.3:
                pressure.intensity = min(1.0, pressure.intensity + 0.2)
                changes[f"intensify_{pressure_id}"] = pressure.intensity

        return changes

    def _scaffolding_evolution(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scaffolding evolution: gradual difficulty increase.

        Provides a curriculum where complexity increases
        proportional to agent mastery.
        """
        changes = {}

        # Calculate mastery level
        mastery = analysis["mean_fitness"]

        # Gradual intensity increase
        target_intensity = min(1.0, 0.2 + mastery * 0.6)

        for pressure in self.pressures.values():
            old_intensity = pressure.intensity
            pressure.intensity = old_intensity + (target_intensity - old_intensity) * 0.1

        changes["target_complexity"] = target_intensity

        return changes

    def _reactive_evolution(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reactive evolution: respond to agent behavior patterns.

        Less aggressive than adversarial, more realistic homeostasis.
        """
        changes = {}

        # Maintain moderate difficulty
        target_variance = 0.15  # Enough to differentiate

        if analysis["variance"] < target_variance * 0.5:
            # Too easy, increase pressure
            for pressure in self.pressures.values():
                pressure.intensity = min(1.0, pressure.intensity + 0.05)
        elif analysis["variance"] > target_variance * 2:
            # Too hard, decrease pressure
            for pressure in self.pressures.values():
                pressure.intensity = max(0.1, pressure.intensity - 0.05)

        return changes

    def _generate_novel_pressure(self) -> Optional[EnvironmentalPressure]:
        """Generate a novel environmental pressure."""
        pressure_templates = [
            ("deception_detection", "cognitive_demand", "metacognitive_awareness"),
            ("coalition_navigation", "social_threat", "cooperation_tendency"),
            ("emotional_manipulation", "social_threat", "emotional_stability"),
            ("information_warfare", "cognitive_demand", "uncertainty_tolerance"),
            ("status_competition", "resource_scarcity", "status_drive"),
        ]

        # Pick one not currently active
        existing = set(self.pressures.keys())
        available = [t for t in pressure_templates if t[0] not in existing]

        if not available:
            return None

        template = random.choice(available)

        return EnvironmentalPressure(
            pressure_id=template[0],
            pressure_type=template[1],
            intensity=0.3,
            target_dimension=template[2],
            generation_introduced=self.generation
        )

    def _add_diversity_pressure(self):
        """Add pressures that favor diverse strategies."""
        # Create opposing pressures
        oppositions = [
            ("cooperation_pressure", "competition_pressure"),
            ("stability_pressure", "adaptability_pressure"),
            ("individual_pressure", "collective_pressure"),
        ]

        for pos, neg in oppositions:
            if pos not in self.pressures and neg not in self.pressures:
                # Add both with moderate intensity
                self.pressures[pos] = EnvironmentalPressure(
                    pressure_id=pos,
                    pressure_type="diversity",
                    intensity=0.3,
                    target_dimension="cooperation_tendency",
                    generation_introduced=self.generation
                )
                break

    def apply_to_npc(self, npc: BaseNPC, tick: int) -> SoulMapDelta:
        """
        Apply current environmental pressures to an NPC.

        Returns a SoulMapDelta representing the psychological
        effect of the environment on the NPC.
        """
        delta = SoulMapDelta()

        for pressure in self.pressures.values():
            # Calculate pressure effect based on NPC psychology
            vulnerability = self._calculate_vulnerability(npc, pressure)
            effect = pressure.intensity * vulnerability * 0.1

            # Apply effect to target dimension
            if hasattr(delta, pressure.target_dimension):
                current = getattr(delta, pressure.target_dimension)
                setattr(delta, pressure.target_dimension, current + effect)

        return delta

    def _calculate_vulnerability(self, npc: BaseNPC, pressure: EnvironmentalPressure) -> float:
        """Calculate how vulnerable an NPC is to a specific pressure."""
        # Base vulnerability from soul map
        if hasattr(npc.soul_map, pressure.target_dimension):
            current_value = getattr(npc.soul_map, pressure.target_dimension)
            # NPCs with extreme values are more vulnerable
            vulnerability = abs(current_value - 0.5) * 2
        else:
            vulnerability = 0.5

        # Modify by NPC archetype
        if npc.archetype and "resilient" in npc.archetype.lower():
            vulnerability *= 0.5

        return vulnerability

    def get_evolution_state(self) -> Dict[str, Any]:
        """Get current state of environmental evolution."""
        return {
            "generation": self.generation,
            "strategy": self.strategy.name,
            "pressures": {
                pid: {"intensity": p.intensity, "type": p.pressure_type}
                for pid, p in self.pressures.items()
            },
            "npc_parameters": self.npc_base_parameters,
            "fitness_trend": self.fitness_history[-10:] if self.fitness_history else [],
        }


# =============================================================================
# INTEGRATED CO-EVOLUTION SYSTEM
# =============================================================================

class PsychosocialCoevolutionEngine:
    """
    Master engine integrating all co-evolution components.

    This is the main interface for running psychosocial co-evolution.
    It coordinates:
    1. Social network dynamics
    2. Belief propagation
    3. Environmental evolution
    4. NPC psychological adaptation

    The result is an environment that genuinely co-evolves with
    agents, providing the selection pressure necessary for
    developing sophisticated Theory of Mind.
    """

    def __init__(
        self,
        evolution_strategy: EnvironmentEvolutionStrategy = EnvironmentEvolutionStrategy.ECOLOGICAL,
        enable_belief_propagation: bool = True,
        enable_social_dynamics: bool = True,
    ):
        """
        Initialize the co-evolution engine.

        Args:
            evolution_strategy: How the environment should evolve
            enable_belief_propagation: Enable belief spreading mechanics
            enable_social_dynamics: Enable social network dynamics
        """
        self.social_network = SocialNetwork()
        self.belief_engine = BeliefPropagationEngine(self.social_network)
        self.env_evolution = PsychosocialEnvironmentEvolution(evolution_strategy)

        self.enable_belief_propagation = enable_belief_propagation
        self.enable_social_dynamics = enable_social_dynamics

        # Tick counter
        self.tick = 0

        # Metrics
        self.metrics = {
            "social_complexity": [],
            "belief_entropy": [],
            "coalition_count": [],
            "hierarchy_gini": [],
        }

    def register_agents(self, agent_ids: List[str]):
        """Register agents with the co-evolution system."""
        for agent_id in agent_ids:
            self.social_network.hierarchy[agent_id] = 0.5

    def process_interaction(
        self,
        agent1_id: str,
        agent2_id: str,
        agent1_action: Dict[str, Any],
        agent2_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process an interaction between two agents.

        This is the main entry point for recording social interactions.
        """
        # Determine cooperation
        a1_cooperated = self._was_cooperative(agent1_action)
        a2_cooperated = self._was_cooperative(agent2_action)

        # Update social network
        if self.enable_social_dynamics:
            self.social_network.record_interaction(
                agent1_id, agent2_id, a1_cooperated, a2_cooperated, self.tick
            )

        # Propagate beliefs from interaction
        if self.enable_belief_propagation:
            self.belief_engine.propagate(self.tick, [(agent1_id, agent2_id)])

        # Determine interaction outcome
        outcome = self._determine_outcome(agent1_action, agent2_action)

        # Update hierarchy if competitive
        if outcome.get("competitive"):
            winner = outcome.get("winner")
            loser = outcome.get("loser")
            if winner and loser:
                self.social_network.update_hierarchy(winner, loser)

        return outcome

    def _was_cooperative(self, action: Dict[str, Any]) -> bool:
        """Determine if an action was cooperative."""
        action_type = action.get("type", "")
        if isinstance(action_type, Enum):
            action_type = action_type.value

        cooperative_actions = {"interact", "validate", "share", "help", "cooperate"}
        return action_type.lower() in cooperative_actions

    def _determine_outcome(
        self,
        action1: Dict[str, Any],
        action2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the outcome of an interaction."""
        # Simplified outcome determination
        outcome = {"competitive": False}

        # Check for competitive interaction
        if action1.get("type") == "compete" or action2.get("type") == "compete":
            outcome["competitive"] = True
            # Random winner for now (could be based on status/resources)
            outcome["winner"] = random.choice([
                action1.get("agent_id"), action2.get("agent_id")
            ])
            outcome["loser"] = (
                action2.get("agent_id") if outcome["winner"] == action1.get("agent_id")
                else action1.get("agent_id")
            )

        return outcome

    def tick_world(self, npc_list: List[BaseNPC]) -> Dict[str, Any]:
        """
        Advance the world by one tick.

        This applies all dynamics and returns changes to apply.
        """
        self.tick += 1

        changes = {
            "npc_deltas": {},
            "new_coalitions": {},
            "belief_updates": {},
        }

        # Tick social network
        if self.enable_social_dynamics:
            self.social_network.tick(self.tick)
            changes["new_coalitions"] = self.social_network.coalitions

        # Apply environmental pressure to NPCs
        for npc in npc_list:
            delta = self.env_evolution.apply_to_npc(npc, self.tick)
            changes["npc_deltas"][npc.npc_id] = delta

        # Record metrics
        self._record_metrics()

        return changes

    def evolve_generation(self, fitness_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Trigger environmental evolution at generation boundary.

        This should be called after evaluating agent fitness each generation.
        """
        self.env_evolution.record_generation_fitness(list(fitness_values.values()))
        return self.env_evolution.evolve_environment(fitness_values)

    def _record_metrics(self):
        """Record complexity metrics."""
        # Social complexity: number of significant relationships
        significant_edges = sum(
            1 for e in self.social_network.edges.values()
            if e.familiarity > 0.3
        )
        self.metrics["social_complexity"].append(significant_edges)

        # Coalition count
        self.metrics["coalition_count"].append(len(self.social_network.coalitions))

        # Hierarchy Gini coefficient
        if self.social_network.hierarchy:
            statuses = sorted(self.social_network.hierarchy.values())
            n = len(statuses)
            if n > 1:
                gini = sum(
                    (2 * i - n - 1) * s
                    for i, s in enumerate(statuses, 1)
                ) / (n * sum(statuses)) if sum(statuses) > 0 else 0
                self.metrics["hierarchy_gini"].append(abs(gini))

    def get_social_observation(self, agent_id: str) -> torch.Tensor:
        """
        Get social observation tensor for an agent.

        This encodes the agent's view of the social world,
        including relationships and coalitions.
        """
        obs = []

        # Agent's relationships (top 15 by familiarity)
        agent_edges = [
            e for e in self.social_network.edges.values()
            if e.source_id == agent_id
        ]
        agent_edges.sort(key=lambda e: e.familiarity, reverse=True)

        for i in range(TheoreticalConstants.DUNBAR_NUMBER):
            if i < len(agent_edges):
                edge = agent_edges[i]
                obs.extend([
                    edge.trust,
                    edge.familiarity,
                    edge.affect,
                    edge.perceived_trust,
                    float(edge.get_relationship_type().value) / 6
                ])
            else:
                obs.extend([0.5, 0.0, 0.0, 0.5, 0.0])

        # Coalition membership
        in_coalition = any(
            agent_id in members
            for members in self.social_network.coalitions.values()
        )
        obs.append(float(in_coalition))

        # Hierarchy position
        status = self.social_network.hierarchy.get(agent_id, 0.5)
        obs.append(status)

        # Knowledge asymmetry summary
        knowledge = self.belief_engine.get_agent_knowledge(agent_id)
        obs.append(len(knowledge) / 100)  # Normalized belief count

        return torch.tensor(obs, dtype=torch.float32)

    def get_tom_challenge_level(self) -> int:
        """
        Get current ToM challenge level.

        This indicates what order of ToM reasoning the environment
        currently demands. Used for curriculum learning.
        """
        # Based on environmental parameters
        deception = self.env_evolution.npc_base_parameters["deception_sophistication"]
        belief_complexity = self.env_evolution.npc_base_parameters["belief_complexity"]

        challenge = (deception + belief_complexity) / 2

        if challenge < 0.3:
            return 1  # First-order ToM sufficient
        elif challenge < 0.5:
            return 2  # Second-order needed
        elif challenge < 0.7:
            return 3  # Third-order needed
        elif challenge < 0.9:
            return 4  # Fourth-order needed
        else:
            return 5  # Maximum ToM depth required

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of co-evolution state."""
        return {
            "tick": self.tick,
            "social_network": {
                "num_edges": len(self.social_network.edges),
                "num_coalitions": len(self.social_network.coalitions),
                "hierarchy_size": len(self.social_network.hierarchy),
            },
            "belief_engine": {
                "active_beliefs": len(self.belief_engine.beliefs),
                "total_holders": sum(
                    len(b.holders) for b in self.belief_engine.beliefs.values()
                ),
            },
            "environment": self.env_evolution.get_evolution_state(),
            "tom_challenge_level": self.get_tom_challenge_level(),
            "metrics": {
                k: v[-10:] if v else []
                for k, v in self.metrics.items()
            },
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Constants
    'TheoreticalConstants',

    # Social Network
    'RelationshipType',
    'SocialEdge',
    'SocialNetwork',

    # Belief Propagation
    'PropagatingBelief',
    'BeliefPropagationEngine',

    # Environmental Evolution
    'EnvironmentEvolutionStrategy',
    'EnvironmentalPressure',
    'PsychosocialEnvironmentEvolution',

    # Main Engine
    'PsychosocialCoevolutionEngine',
]
