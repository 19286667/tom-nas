"""
Narrative Emergence System
==========================

This module creates meaningful, self-explanatory, and entertaining
scenarios from the emergent dynamics of psychosocial co-evolution.

The key insight: emergent complexity without interpretation is noise.
This system transforms raw social dynamics into comprehensible
narratives that reveal Theory of Mind in action.

DESIGN PHILOSOPHY
-----------------
1. EMERGENCE OVER SCRIPTING: Narratives arise from dynamics, not templates
2. PSYCHOLOGICAL TRUTH: Every scenario reflects genuine psychological principles
3. DISCOVERABILITY: Players should feel they're uncovering, not being told
4. CONSEQUENTIALITY: Actions have meaningful, traceable consequences
5. SCIENTIFIC GROUNDING: Entertainment serves pedagogical function

NARRATIVE SOURCES
-----------------
Narratives emerge from:
- Coalition dynamics (alliances, betrayals, loyalty tests)
- Belief asymmetry (secrets, revelations, information warfare)
- Status competition (rises, falls, challenges)
- Psychological crises (identity conflicts, value clashes)
- Environmental pressures (scarcity, threat, opportunity)

Each source generates archetypal patterns that humans naturally
recognize and find meaningful.
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


from .npcs.base_npc import BaseNPC
from .psychosocial_coevolution import (
    PsychosocialCoevolutionEngine,
)

# =============================================================================
# NARRATIVE ARCHETYPES
# =============================================================================


class NarrativeArchetype(Enum):
    """
    Universal narrative patterns that emerge from social dynamics.

    These archetypes are cross-culturally recognized and provide
    intuitive framing for complex social situations.

    Each archetype has:
    - Clear dramatic structure
    - Recognizable character roles
    - Implicit ToM requirements to understand/resolve
    """

    # ALLIANCE ARCHETYPES
    RELUCTANT_ALLIANCE = auto()  # Former enemies must cooperate
    BETRAYAL_BREWING = auto()  # Coalition member planning defection
    LOYALTY_TEST = auto()  # Trust being verified through trial
    POWER_STRUGGLE = auto()  # Coalition leadership contested

    # DECEPTION ARCHETYPES
    HIDDEN_IDENTITY = auto()  # Someone isn't who they claim
    DOUBLE_AGENT = auto()  # Playing both sides
    FALSE_FLAG = auto()  # Action attributed to wrong party
    MANIPULATION_WEB = auto()  # Complex multi-party deception

    # STATUS ARCHETYPES
    RISING_CHALLENGER = auto()  # Low-status agent ascending
    FALLEN_LEADER = auto()  # High-status agent in decline
    USURPER_PLOT = auto()  # Conspiracy to overthrow
    MERITOCRACY_TEST = auto()  # Status earned through demonstration

    # PSYCHOLOGICAL ARCHETYPES
    IDENTITY_CRISIS = auto()  # Agent questioning core beliefs
    VALUE_CONFLICT = auto()  # Competing drives forcing choice
    REDEMPTION_ARC = auto()  # Past wrongs seeking absolution
    TRANSFORMATION = auto()  # Fundamental psychological shift

    # INFORMATION ARCHETYPES
    SECRET_KNOWLEDGE = auto()  # Asymmetric information exploitation
    REVELATION_PENDING = auto()  # Truth about to emerge
    PROPHECY_UNFOLDING = auto()  # Predicted events manifesting
    EPISTEMIC_CRISIS = auto()  # What's known is wrong


@dataclass
class NarrativeElement:
    """
    A single narrative element (character, event, or relationship).
    """

    element_id: str
    element_type: str  # "character", "event", "relationship", "location"
    description: str
    participants: List[str]  # NPC IDs involved
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmergentNarrative:
    """
    A complete emergent narrative arising from dynamics.

    This represents a storyline that has emerged from the
    underlying social simulation and has been recognized
    as matching an archetypal pattern.
    """

    narrative_id: str
    archetype: NarrativeArchetype
    title: str
    description: str

    # Participants and roles
    protagonists: List[str]  # NPCs in protagonist roles
    antagonists: List[str]  # NPCs in antagonist roles
    supporting_cast: List[str]  # Other involved NPCs

    # Narrative structure
    elements: List[NarrativeElement] = field(default_factory=list)
    current_act: int = 1  # 3-act structure: 1=setup, 2=conflict, 3=resolution
    tension_level: float = 0.0  # 0-1, narrative tension

    # Emergence tracking
    tick_emerged: int = 0
    tick_resolved: Optional[int] = None
    detection_confidence: float = 0.0  # How clearly this pattern matches

    # ToM requirements
    min_tom_depth: int = 1  # Minimum ToM to understand this narrative
    understanding_checkpoints: List[str] = field(default_factory=list)

    # Outcome tracking
    predicted_outcomes: List[str] = field(default_factory=list)
    actual_outcome: Optional[str] = None

    def get_dramatic_summary(self) -> str:
        """Generate a dramatic summary of the narrative."""
        act_descriptions = {1: "The stage is set...", 2: "Conflict intensifies...", 3: "Resolution approaches..."}
        return f"[{self.archetype.name}] {self.title}\n{act_descriptions.get(self.current_act, '')}\n{self.description}"


# =============================================================================
# NARRATIVE DETECTION
# =============================================================================


class NarrativeDetector:
    """
    Detects emergent narrative patterns from social dynamics.

    This system watches the co-evolution engine and identifies
    when patterns match recognizable narrative archetypes.

    The detection is based on:
    - Structural patterns (relationship configurations)
    - Dynamic patterns (how relationships are changing)
    - Psychological patterns (agent states and conflicts)
    """

    def __init__(self, coevolution_engine: PsychosocialCoevolutionEngine):
        self.engine = coevolution_engine
        self.detected_narratives: Dict[str, EmergentNarrative] = {}
        self.narrative_counter = 0

        # Detection thresholds
        self.min_confidence = 0.6

        # Pattern matchers
        self.matchers: Dict[NarrativeArchetype, Callable] = {
            NarrativeArchetype.BETRAYAL_BREWING: self._detect_betrayal_brewing,
            NarrativeArchetype.RELUCTANT_ALLIANCE: self._detect_reluctant_alliance,
            NarrativeArchetype.POWER_STRUGGLE: self._detect_power_struggle,
            NarrativeArchetype.RISING_CHALLENGER: self._detect_rising_challenger,
            NarrativeArchetype.HIDDEN_IDENTITY: self._detect_hidden_identity,
            NarrativeArchetype.IDENTITY_CRISIS: self._detect_identity_crisis,
            NarrativeArchetype.SECRET_KNOWLEDGE: self._detect_secret_knowledge,
            NarrativeArchetype.LOYALTY_TEST: self._detect_loyalty_test,
        }

    def scan_for_narratives(self, npcs: Dict[str, BaseNPC], tick: int) -> List[EmergentNarrative]:
        """
        Scan current state for emergent narratives.

        Returns newly detected narratives.
        """
        new_narratives = []

        for archetype, matcher in self.matchers.items():
            # Skip if we already have an active narrative of this type
            # with the same participants
            detection = matcher(npcs, tick)

            if detection and detection["confidence"] >= self.min_confidence:
                # Check for duplicates
                if not self._is_duplicate(detection, archetype):
                    narrative = self._create_narrative(detection, archetype, tick)
                    self.detected_narratives[narrative.narrative_id] = narrative
                    new_narratives.append(narrative)

        return new_narratives

    def _is_duplicate(self, detection: Dict, archetype: NarrativeArchetype) -> bool:
        """Check if this narrative is duplicate of existing one."""
        for narrative in self.detected_narratives.values():
            if narrative.archetype != archetype:
                continue
            if narrative.tick_resolved is not None:
                continue  # Already resolved

            # Check participant overlap
            existing_participants = set(narrative.protagonists + narrative.antagonists + narrative.supporting_cast)
            new_participants = set(detection.get("participants", []))

            overlap = len(existing_participants & new_participants)
            if overlap > len(new_participants) * 0.7:
                return True

        return False

    def _create_narrative(self, detection: Dict, archetype: NarrativeArchetype, tick: int) -> EmergentNarrative:
        """Create narrative from detection result."""
        self.narrative_counter += 1

        return EmergentNarrative(
            narrative_id=f"narrative_{self.narrative_counter}",
            archetype=archetype,
            title=detection.get("title", f"Untitled {archetype.name}"),
            description=detection.get("description", ""),
            protagonists=detection.get("protagonists", []),
            antagonists=detection.get("antagonists", []),
            supporting_cast=detection.get("supporting_cast", []),
            tick_emerged=tick,
            detection_confidence=detection.get("confidence", 0.0),
            min_tom_depth=detection.get("tom_depth", 1),
            predicted_outcomes=detection.get("predicted_outcomes", []),
            tension_level=detection.get("tension", 0.3),
        )

    # =========================================================================
    # PATTERN MATCHERS
    # =========================================================================

    def _detect_betrayal_brewing(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect brewing betrayal within a coalition.

        Pattern:
        - Agent A is in coalition with B
        - A's trust of B is declining
        - A has positive relationship with someone outside coalition
        - A's affect toward B is becoming negative
        """
        network = self.engine.social_network

        for coalition_id, members in network.coalitions.items():
            if len(members) < 2:
                continue

            for potential_betrayer in members:
                # Check relationships with coalition members
                coalition_trust = []
                outside_connections = []

                for target in network.edges:
                    edge = network.edges.get((potential_betrayer, target[1]))
                    if edge is None:
                        continue

                    if target[1] in members:
                        coalition_trust.append(edge)
                    elif edge.trust > 0.5 and edge.affect > 0:
                        outside_connections.append(edge)

                # Check betrayal conditions
                if coalition_trust and outside_connections:
                    declining_trust = any(
                        len(e.cooperation_history) > 3 and e.cooperation_history[-1] == False for e in coalition_trust
                    )
                    negative_affect = any(e.affect < -0.1 for e in coalition_trust)

                    if declining_trust and negative_affect:
                        betrayed = [e.target_id for e in coalition_trust if e.affect < 0]
                        if betrayed:
                            npc = npcs.get(potential_betrayer)
                            npc_name = npc.name if npc else potential_betrayer

                            return {
                                "confidence": 0.7,
                                "title": f"The Wavering Loyalty of {npc_name}",
                                "description": (
                                    f"{npc_name} grows distant from their allies, eyes turning toward new horizons. Trust, once solid, shows cracks."
                                ),
                                "protagonists": betrayed[:1],
                                "antagonists": [potential_betrayer],
                                "supporting_cast": list(members - {potential_betrayer} - set(betrayed)),
                                "participants": list(members),
                                "tom_depth": 2,  # Need to understand betrayer's hidden intentions
                                "predicted_outcomes": [
                                    "Betrayal executed, coalition fractures",
                                    "Betrayer discovered, expelled",
                                    "Reconciliation, trust rebuilt",
                                ],
                                "tension": 0.6,
                            }

        return None

    def _detect_reluctant_alliance(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect reluctant alliance forming.

        Pattern:
        - Agents A and B have negative history (low affect)
        - Both face common threat/pressure
        - Recent positive interaction despite history
        """
        network = self.engine.social_network

        for (source, target), edge in network.edges.items():
            # Need negative history but recent cooperation
            if edge.familiarity < 0.3:
                continue
            if edge.affect > 0:  # Already positive
                continue
            if len(edge.cooperation_history) < 2:
                continue

            # Check for recent cooperation despite negative affect
            recent_cooperation = edge.cooperation_history[-2:]
            if sum(recent_cooperation) >= 1:  # At least one recent cooperation
                reverse_edge = network.edges.get((target, source))
                if reverse_edge and reverse_edge.affect < 0:
                    # Both view each other negatively but cooperating
                    npc1 = npcs.get(source)
                    npc2 = npcs.get(target)
                    name1 = npc1.name if npc1 else source
                    name2 = npc2.name if npc2 else target

                    return {
                        "confidence": 0.65,
                        "title": f"Strange Bedfellows: {name1} and {name2}",
                        "description": (
                            f"Old wounds cannot heal, yet circumstances force {name1} and {name2} to stand together. Neither trusts the other, but necessity makes allies of enemies."
                        ),
                        "protagonists": [source, target],
                        "antagonists": [],
                        "supporting_cast": [],
                        "participants": [source, target],
                        "tom_depth": 2,
                        "predicted_outcomes": [
                            "Alliance strengthens into genuine bond",
                            "Temporary truce, old enmity returns",
                            "One betrays the other at crucial moment",
                        ],
                        "tension": 0.5,
                    }

        return None

    def _detect_power_struggle(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect power struggle within coalition or hierarchy.

        Pattern:
        - Two high-status agents in same coalition
        - Both have support networks
        - Recent competitive interactions
        """
        network = self.engine.social_network
        hierarchy = network.hierarchy

        # Find high-status agents in same coalition
        for coalition_id, members in network.coalitions.items():
            high_status = [m for m in members if hierarchy.get(m, 0.5) > 0.6]

            if len(high_status) >= 2:
                # Check for competition between them
                contender1, contender2 = high_status[:2]
                edge = network.edges.get((contender1, contender2))

                if edge and edge.affect < 0.2 and edge.familiarity > 0.3:
                    npc1 = npcs.get(contender1)
                    npc2 = npcs.get(contender2)
                    name1 = npc1.name if npc1 else contender1
                    name2 = npc2.name if npc2 else contender2

                    return {
                        "confidence": 0.75,
                        "title": "The Contest for Dominance",
                        "description": (
                            f"Two titans within the same alliance: {name1} and {name2}. The group cannot have two heads. One must yield, or both must fall."
                        ),
                        "protagonists": [contender1],
                        "antagonists": [contender2],
                        "supporting_cast": list(members - {contender1, contender2}),
                        "participants": list(members),
                        "tom_depth": 3,  # Need to predict strategic maneuvering
                        "predicted_outcomes": [
                            f"{name1} prevails, {name2} submitted",
                            f"{name2} prevails, {name1} submitted",
                            "Coalition splits along factional lines",
                            "External threat forces reconciliation",
                        ],
                        "tension": 0.8,
                    }

        return None

    def _detect_rising_challenger(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect rising challenger in hierarchy.

        Pattern:
        - Agent with low starting status
        - Recent string of successes (hierarchy increasing)
        - Established leader exists
        """
        network = self.engine.social_network
        hierarchy = network.hierarchy

        if len(hierarchy) < 3:
            return None

        # Find top status agent
        top_agent = max(hierarchy.items(), key=lambda x: x[1])

        # Find agents with increasing status
        # (Would need historical tracking, simplified here)
        for agent_id, status in hierarchy.items():
            if agent_id == top_agent[0]:
                continue
            if 0.4 < status < 0.6:  # Middle status, potentially rising
                # Check for positive trajectory (simplified)
                edge_to_top = network.edges.get((agent_id, top_agent[0]))
                if edge_to_top and edge_to_top.familiarity > 0.3:
                    npc = npcs.get(agent_id)
                    top_npc = npcs.get(top_agent[0])
                    challenger_name = npc.name if npc else agent_id
                    leader_name = top_npc.name if top_npc else top_agent[0]

                    return {
                        "confidence": 0.6,
                        "title": f"The Ascent of {challenger_name}",
                        "description": (
                            f"From the middle ranks emerges {challenger_name}, each success drawing them closer to {leader_name}'s throne. The old order watches warily."
                        ),
                        "protagonists": [agent_id],
                        "antagonists": [top_agent[0]],
                        "supporting_cast": [],
                        "participants": [agent_id, top_agent[0]],
                        "tom_depth": 2,
                        "predicted_outcomes": [
                            f"{challenger_name} achieves prominence",
                            f"{leader_name} suppresses the challenger",
                            "New alliance reshapes hierarchy",
                        ],
                        "tension": 0.5,
                    }

        return None

    def _detect_hidden_identity(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect hidden identity / deception scenario.

        Pattern:
        - Agent's behavior doesn't match their stated role
        - Inconsistencies in belief network
        - Others have conflicting beliefs about this agent
        """
        belief_engine = self.engine.belief_engine

        for npc_id, npc in npcs.items():
            if not hasattr(npc, "is_zombie") or not npc.is_zombie:
                continue

            # Zombie NPCs have hidden nature
            # Check if anyone suspects
            suspecting = []
            for other_id in npcs:
                if other_id == npc_id:
                    continue
                edge = self.engine.social_network.edges.get((other_id, npc_id))
                if edge and edge.trust < 0.3 and edge.familiarity > 0.3:
                    suspecting.append(other_id)

            if suspecting:
                npc_name = npc.name if hasattr(npc, "name") else npc_id

                return {
                    "confidence": 0.7,
                    "title": f"Who Is {npc_name}, Really?",
                    "description": (
                        f"Something is wrong with {npc_name}. The words say one thing, but the eyes say another. Some have begun to wonder what lies beneath the mask."
                    ),
                    "protagonists": suspecting[:2],
                    "antagonists": [npc_id],
                    "supporting_cast": [],
                    "participants": suspecting + [npc_id],
                    "tom_depth": 3,  # Need to see through deception
                    "predicted_outcomes": [
                        "True nature exposed",
                        "Suspicions dismissed as paranoia",
                        "Confrontation leads to revelation",
                    ],
                    "tension": 0.7,
                }

        return None

    def _detect_identity_crisis(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect psychological identity crisis.

        Pattern:
        - NPC's soul map shows internal conflict
        - Competing drives at extreme values
        - Recent dramatic experiences
        """
        for npc_id, npc in npcs.items():
            if not hasattr(npc, "soul_map"):
                continue

            soul_map = npc.soul_map

            # Check for value conflicts
            conflicts = []

            # Autonomy vs Affiliation conflict
            if hasattr(soul_map, "autonomy_drive") and hasattr(soul_map, "affiliation_drive"):
                if abs(soul_map.autonomy_drive - soul_map.affiliation_drive) > 0.5:
                    if soul_map.autonomy_drive > 0.7 and soul_map.affiliation_drive > 0.7:
                        conflicts.append("independence vs belonging")

            # Self coherence problems
            if hasattr(soul_map, "self_coherence"):
                if soul_map.self_coherence < 0.3:
                    conflicts.append("fragmented sense of self")

            if conflicts:
                npc_name = npc.name if hasattr(npc, "name") else npc_id

                return {
                    "confidence": 0.65,
                    "title": f"The Fracturing of {npc_name}",
                    "description": (
                        f"{npc_name} stands at a crossroads within their own soul. {' And '.join(conflicts)} tear at the foundations of who they thought they were."
                    ),
                    "protagonists": [npc_id],
                    "antagonists": [],
                    "supporting_cast": [],
                    "participants": [npc_id],
                    "tom_depth": 2,  # Understanding internal conflict
                    "predicted_outcomes": [
                        "Integration, stronger sense of self",
                        "Fragmentation continues, behavior erratic",
                        "Transformation into new identity",
                    ],
                    "tension": 0.5,
                }

        return None

    def _detect_secret_knowledge(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect asymmetric information scenario.

        Pattern:
        - Some agents have beliefs others lack
        - The unknown information is consequential
        """
        belief_engine = self.engine.belief_engine

        # Find beliefs held by few
        rare_beliefs = [(bid, belief) for bid, belief in belief_engine.beliefs.items() if 1 <= len(belief.holders) <= 3]

        if not rare_beliefs:
            return None

        # Pick most recent rare belief
        rare_beliefs.sort(key=lambda x: x[1].timestamp, reverse=True)
        belief_id, belief = rare_beliefs[0]

        holders = list(belief.holders)
        non_holders = [npc_id for npc_id in npcs if npc_id not in holders][:5]

        if holders and non_holders:
            holder_names = [npcs[h].name if h in npcs and hasattr(npcs[h], "name") else h for h in holders]

            return {
                "confidence": 0.6,
                "title": f"The Secret of {holder_names[0]}",
                "description": (
                    f"Knowledge is power, and {holder_names[0]} holds knowledge that others lack. What will they do with this advantage?"
                ),
                "protagonists": holders,
                "antagonists": [],
                "supporting_cast": non_holders,
                "participants": holders + non_holders,
                "tom_depth": 2,  # Understanding information asymmetry
                "predicted_outcomes": [
                    "Secret shared, leveling playing field",
                    "Secret exploited for advantage",
                    "Secret discovered by others",
                    "Secret becomes irrelevant",
                ],
                "tension": 0.4,
            }

        return None

    def _detect_loyalty_test(self, npcs: Dict[str, BaseNPC], tick: int) -> Optional[Dict]:
        """
        Detect loyalty test scenario.

        Pattern:
        - Coalition exists
        - Trust is uncertain within coalition
        - Opportunity to defect exists
        """
        network = self.engine.social_network

        for coalition_id, members in network.coalitions.items():
            if len(members) < 3:
                continue

            # Find member with uncertain trust
            for member in members:
                uncertain_trust = []
                for other in members:
                    if other == member:
                        continue
                    edge = network.edges.get((other, member))
                    if edge and 0.4 < edge.trust < 0.6:
                        uncertain_trust.append(other)

                if len(uncertain_trust) >= 2:
                    npc = npcs.get(member)
                    npc_name = npc.name if npc else member

                    return {
                        "confidence": 0.6,
                        "title": f"Testing {npc_name}",
                        "description": (
                            f"The coalition is uncertain about {npc_name}. A test approaches - will they prove their loyalty, or reveal their true colors?"
                        ),
                        "protagonists": uncertain_trust[:2],
                        "antagonists": [],
                        "supporting_cast": [member],
                        "participants": list(members),
                        "tom_depth": 2,
                        "predicted_outcomes": [
                            f"{npc_name} passes, trust strengthened",
                            f"{npc_name} fails, expelled from coalition",
                            "Test reveals unexpected alliance",
                        ],
                        "tension": 0.6,
                    }

        return None


# =============================================================================
# NARRATIVE PROGRESSION
# =============================================================================


class NarrativeProgressionEngine:
    """
    Tracks and progresses detected narratives over time.

    Narratives follow a 3-act structure:
    - Act 1 (Setup): Establish characters and conflict
    - Act 2 (Conflict): Tension builds, stakes raise
    - Act 3 (Resolution): Climax and outcome

    The engine monitors conditions for act transitions
    and narrative resolution.
    """

    def __init__(self, detector: NarrativeDetector):
        self.detector = detector
        self.coevolution = detector.engine

        # Resolution conditions per archetype
        self.resolution_conditions: Dict[NarrativeArchetype, Callable] = {
            NarrativeArchetype.BETRAYAL_BREWING: self._check_betrayal_resolution,
            NarrativeArchetype.POWER_STRUGGLE: self._check_power_resolution,
            NarrativeArchetype.RISING_CHALLENGER: self._check_rising_resolution,
        }

    def update_narratives(self, npcs: Dict[str, BaseNPC], tick: int) -> Dict[str, Any]:
        """
        Update all active narratives.

        Returns dict of narrative updates.
        """
        updates = {
            "act_transitions": [],
            "resolutions": [],
            "tension_changes": [],
        }

        for narrative_id, narrative in list(self.detector.detected_narratives.items()):
            if narrative.tick_resolved is not None:
                continue

            # Update tension based on dynamics
            new_tension = self._calculate_tension(narrative, npcs)
            if abs(new_tension - narrative.tension_level) > 0.1:
                updates["tension_changes"].append(
                    {
                        "narrative_id": narrative_id,
                        "old_tension": narrative.tension_level,
                        "new_tension": new_tension,
                    }
                )
                narrative.tension_level = new_tension

            # Check for act transitions
            new_act = self._determine_act(narrative, npcs, tick)
            if new_act != narrative.current_act:
                updates["act_transitions"].append(
                    {
                        "narrative_id": narrative_id,
                        "old_act": narrative.current_act,
                        "new_act": new_act,
                    }
                )
                narrative.current_act = new_act

            # Check for resolution
            resolution = self._check_resolution(narrative, npcs, tick)
            if resolution:
                narrative.tick_resolved = tick
                narrative.actual_outcome = resolution
                updates["resolutions"].append(
                    {
                        "narrative_id": narrative_id,
                        "outcome": resolution,
                    }
                )

        return updates

    def _calculate_tension(self, narrative: EmergentNarrative, npcs: Dict[str, BaseNPC]) -> float:
        """Calculate current tension level for narrative."""
        network = self.coevolution.social_network

        # Base tension from relationship states
        tension = 0.3

        # Add tension from protagonist-antagonist relationships
        for prot in narrative.protagonists:
            for ant in narrative.antagonists:
                edge = network.edges.get((prot, ant))
                if edge:
                    # More negative affect = more tension
                    tension += (1 - edge.affect) * 0.2
                    # Lower trust = more tension
                    tension += (1 - edge.trust) * 0.1

        # Add tension from instability
        tension += self.coevolution.env_evolution.npc_base_parameters.get("emotional_volatility", 0.5) * 0.2

        return min(1.0, tension)

    def _determine_act(self, narrative: EmergentNarrative, npcs: Dict[str, BaseNPC], tick: int) -> int:
        """Determine which act narrative is in."""
        ticks_since_emergence = tick - narrative.tick_emerged

        # Time-based baseline
        if ticks_since_emergence < 50:
            base_act = 1
        elif ticks_since_emergence < 150:
            base_act = 2
        else:
            base_act = 3

        # Tension-based adjustment
        if narrative.tension_level > 0.8 and base_act < 3:
            return 3
        if narrative.tension_level > 0.5 and base_act < 2:
            return 2

        return base_act

    def _check_resolution(self, narrative: EmergentNarrative, npcs: Dict[str, BaseNPC], tick: int) -> Optional[str]:
        """Check if narrative has resolved."""
        if narrative.current_act < 3:
            return None

        # Check archetype-specific resolution
        condition_fn = self.resolution_conditions.get(narrative.archetype)
        if condition_fn:
            return condition_fn(narrative, npcs)

        # Default: resolve after enough time in act 3
        if tick - narrative.tick_emerged > 200:
            return random.choice(narrative.predicted_outcomes)

        return None

    def _check_betrayal_resolution(self, narrative: EmergentNarrative, npcs: Dict[str, BaseNPC]) -> Optional[str]:
        """Check betrayal narrative resolution."""
        if not narrative.antagonists:
            return None

        betrayer = narrative.antagonists[0]
        network = self.coevolution.social_network

        # Check if betrayer is still in coalition
        in_coalition = any(betrayer in members for members in network.coalitions.values())

        if not in_coalition:
            return "Betrayal executed, coalition fractures"

        # Check if trust recovered
        for prot in narrative.protagonists:
            edge = network.edges.get((prot, betrayer))
            if edge and edge.trust > 0.7:
                return "Reconciliation, trust rebuilt"

        return None

    def _check_power_resolution(self, narrative: EmergentNarrative, npcs: Dict[str, BaseNPC]) -> Optional[str]:
        """Check power struggle resolution."""
        hierarchy = self.coevolution.social_network.hierarchy

        if narrative.protagonists and narrative.antagonists:
            prot_status = hierarchy.get(narrative.protagonists[0], 0.5)
            ant_status = hierarchy.get(narrative.antagonists[0], 0.5)

            if abs(prot_status - ant_status) > 0.3:
                if prot_status > ant_status:
                    return "Protagonist prevails, antagonist submitted"
                else:
                    return "Antagonist prevails, protagonist submitted"

        return None

    def _check_rising_resolution(self, narrative: EmergentNarrative, npcs: Dict[str, BaseNPC]) -> Optional[str]:
        """Check rising challenger resolution."""
        hierarchy = self.coevolution.social_network.hierarchy

        if narrative.protagonists:
            challenger_status = hierarchy.get(narrative.protagonists[0], 0.5)

            if challenger_status > 0.7:
                return "Challenger achieves prominence"
            elif challenger_status < 0.3:
                return "Challenger's rise arrested"

        return None


# =============================================================================
# NARRATIVE INTERFACE
# =============================================================================


class NarrativeEmergenceSystem:
    """
    Main interface for the narrative emergence system.

    This provides:
    1. Detection of emergent narratives
    2. Tracking of narrative progression
    3. Generation of narrative summaries for display
    4. ToM reasoning requirements for narrative understanding
    """

    def __init__(self, coevolution_engine: PsychosocialCoevolutionEngine):
        self.coevolution = coevolution_engine
        self.detector = NarrativeDetector(coevolution_engine)
        self.progression = NarrativeProgressionEngine(self.detector)

        # Display settings
        self.max_active_narratives = 5

    def tick(self, npcs: Dict[str, BaseNPC], current_tick: int) -> Dict[str, Any]:
        """
        Process one tick of narrative emergence.

        Returns updates for this tick.
        """
        results = {
            "new_narratives": [],
            "updates": {},
            "active_count": 0,
        }

        # Detect new narratives (not every tick for efficiency)
        if current_tick % 10 == 0:
            new_narratives = self.detector.scan_for_narratives(npcs, current_tick)
            results["new_narratives"] = [n.get_dramatic_summary() for n in new_narratives]

        # Update existing narratives
        results["updates"] = self.progression.update_narratives(npcs, current_tick)

        # Count active
        results["active_count"] = sum(1 for n in self.detector.detected_narratives.values() if n.tick_resolved is None)

        return results

    def get_active_narratives(self) -> List[EmergentNarrative]:
        """Get all active (unresolved) narratives."""
        return [n for n in self.detector.detected_narratives.values() if n.tick_resolved is None][
            : self.max_active_narratives
        ]

    def get_narrative_for_display(self, narrative_id: str) -> Optional[Dict[str, Any]]:
        """Get narrative formatted for display."""
        narrative = self.detector.detected_narratives.get(narrative_id)
        if narrative is None:
            return None

        return {
            "id": narrative.narrative_id,
            "archetype": narrative.archetype.name,
            "title": narrative.title,
            "description": narrative.description,
            "act": narrative.current_act,
            "tension": narrative.tension_level,
            "protagonists": narrative.protagonists,
            "antagonists": narrative.antagonists,
            "tom_required": narrative.min_tom_depth,
            "possible_outcomes": narrative.predicted_outcomes,
            "status": "resolved" if narrative.tick_resolved else "active",
            "outcome": narrative.actual_outcome,
        }

    def get_tom_learning_opportunities(self) -> List[Dict[str, Any]]:
        """
        Get current opportunities to learn/demonstrate ToM.

        These are narrative situations that require specific ToM
        reasoning to understand or influence.
        """
        opportunities = []

        for narrative in self.get_active_narratives():
            opportunity = {
                "narrative_id": narrative.narrative_id,
                "archetype": narrative.archetype.name,
                "tom_depth_required": narrative.min_tom_depth,
                "challenge_description": self._get_tom_challenge(narrative),
                "learning_objectives": self._get_learning_objectives(narrative),
            }
            opportunities.append(opportunity)

        return opportunities

    def _get_tom_challenge(self, narrative: EmergentNarrative) -> str:
        """Get description of ToM challenge for this narrative."""
        challenges = {
            NarrativeArchetype.BETRAYAL_BREWING: (
                "Understand the betrayer's hidden intentions and predict when they'll act"
            ),
            NarrativeArchetype.RELUCTANT_ALLIANCE: (
                "Navigate the tension between stated cooperation and underlying distrust"
            ),
            NarrativeArchetype.POWER_STRUGGLE: (
                "Predict strategic moves in a contest where both parties are modeling each other"
            ),
            NarrativeArchetype.HIDDEN_IDENTITY: "Detect inconsistencies between stated identity and actual behavior",
            NarrativeArchetype.SECRET_KNOWLEDGE: "Understand who knows what, and how that shapes their actions",
        }
        return challenges.get(narrative.archetype, "Understand the social dynamics at play")

    def _get_learning_objectives(self, narrative: EmergentNarrative) -> List[str]:
        """Get learning objectives for this narrative."""
        objectives = []

        depth = narrative.min_tom_depth
        if depth >= 1:
            objectives.append("Identify agents' immediate goals and intentions")
        if depth >= 2:
            objectives.append("Model what agents believe about each other")
        if depth >= 3:
            objectives.append("Understand recursive belief chains (A thinks B thinks...)")
        if depth >= 4:
            objectives.append("Predict strategic deception and counter-deception")
        if depth >= 5:
            objectives.append("Navigate adversarial ToM in real-time")

        return objectives

    def get_narrative_metrics(self) -> Dict[str, Any]:
        """Get metrics about narrative emergence."""
        all_narratives = list(self.detector.detected_narratives.values())

        return {
            "total_detected": len(all_narratives),
            "currently_active": len(self.get_active_narratives()),
            "resolved": sum(1 for n in all_narratives if n.tick_resolved is not None),
            "by_archetype": {
                archetype.name: sum(1 for n in all_narratives if n.archetype == archetype)
                for archetype in NarrativeArchetype
            },
            "average_duration": (
                sum((n.tick_resolved - n.tick_emerged) for n in all_narratives if n.tick_resolved)
                / max(1, sum(1 for n in all_narratives if n.tick_resolved))
            ),
            "average_tom_depth": sum(n.min_tom_depth for n in all_narratives) / max(1, len(all_narratives)),
        }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "NarrativeArchetype",
    "NarrativeElement",
    "EmergentNarrative",
    "NarrativeDetector",
    "NarrativeProgressionEngine",
    "NarrativeEmergenceSystem",
]
