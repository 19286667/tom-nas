"""
Integrated Agent System

Combines all ToM frameworks into a unified agent:
- Psychosocial Profile (10 layers, 80+ dims)
- Success State (9 domains, 120+ dims)
- tomsup k-ToM reasoning
- MetaMind three-stage pipeline
- BeliefNest nested belief tracking
- Hypothetical Minds hypothesis evaluation
- Generative Agents memory system
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict

# Import our taxonomies
import sys
sys.path.insert(0, '/home/user/tom-nas/src')
from taxonomy.psychosocial import PsychosocialProfile, AttachmentStyle
from taxonomy.success import SuccessState
from taxonomy.institutions import InstitutionalContext


class BeliefType(Enum):
    """Types of beliefs an agent can hold"""
    WORLD = "world"  # Beliefs about world state
    AGENT = "agent"  # Beliefs about other agents
    META = "meta"  # Beliefs about others' beliefs
    INTENTION = "intention"  # Beliefs about intentions


class StrategyType(Enum):
    """Possible strategies agents might employ"""
    COOPERATIVE = "cooperative"
    DEFECTOR = "defector"
    TIT_FOR_TAT = "tit_for_tat"
    RANDOM = "random"
    GRIM_TRIGGER = "grim_trigger"
    EXPLOITER = "exploiter"
    ALTRUIST = "altruist"


@dataclass
class Episode:
    """Single episodic memory (Generative Agents style)"""
    timestamp: int
    description: str
    agents_involved: List[int]
    emotional_valence: float  # -100 to 100
    importance: float  # 0 to 100
    location: Optional[Tuple[float, float, float]] = None
    action_taken: Optional[str] = None
    outcome: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.embedding is None:
            # Simple embedding based on content
            self.embedding = np.random.randn(64)


@dataclass
class RelationshipMemory:
    """Memory of relationship with another agent"""
    agent_id: int
    interactions: List[Episode] = field(default_factory=list)
    trust_level: float = 50.0
    cooperation_history: List[bool] = field(default_factory=list)
    last_interaction: Optional[int] = None
    perceived_traits: Dict[str, float] = field(default_factory=dict)
    perceived_strategy: Optional[StrategyType] = None
    relationship_quality: float = 50.0

    def update_from_interaction(self, cooperated: bool, my_cooperated: bool, timestamp: int):
        """Update relationship based on interaction outcome"""
        self.cooperation_history.append(cooperated)
        self.last_interaction = timestamp

        # Update trust based on cooperation
        if cooperated and my_cooperated:
            self.trust_level = min(100, self.trust_level + 5)
            self.relationship_quality = min(100, self.relationship_quality + 3)
        elif cooperated and not my_cooperated:
            self.trust_level = min(100, self.trust_level + 2)
        elif not cooperated and my_cooperated:
            self.trust_level = max(0, self.trust_level - 15)
            self.relationship_quality = max(0, self.relationship_quality - 10)
        else:
            self.trust_level = max(0, self.trust_level - 3)


@dataclass
class Insight:
    """Higher-level insight from reflection (Generative Agents)"""
    content: str
    supporting_memories: List[Episode]
    generated_at: int
    insight_type: str = "general"


@dataclass
class AgentMemory:
    """
    Complete memory system based on Generative Agents architecture.
    """
    decay_rate: float = 0.05  # From working_memory_depth

    # Memory stores
    episodic: List[Episode] = field(default_factory=list)
    semantic: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[int, RelationshipMemory] = field(default_factory=dict)
    self_narrative: List[str] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)

    # Cache
    _recency_scores: Dict[int, float] = field(default_factory=dict)

    def add_episode(self, episode: Episode):
        """Add new episodic memory"""
        self.episodic.append(episode)

        # Update relationship memories
        for agent_id in episode.agents_involved:
            if agent_id not in self.relationships:
                self.relationships[agent_id] = RelationshipMemory(agent_id=agent_id)
            self.relationships[agent_id].interactions.append(episode)
            self.relationships[agent_id].last_interaction = episode.timestamp

    def retrieve(self, query: str = "", n: int = 10,
                recency_weight: float = 0.5,
                importance_weight: float = 0.3,
                relevance_weight: float = 0.2,
                current_time: Optional[int] = None) -> List[Episode]:
        """
        Retrieve memories using Generative Agents' scoring.
        Combines recency, importance, and relevance.
        """
        if not self.episodic:
            return []

        if current_time is None:
            current_time = self.episodic[-1].timestamp if self.episodic else 0

        scores = []
        for episode in self.episodic:
            # Recency score (exponential decay)
            time_diff = current_time - episode.timestamp
            recency = np.exp(-self.decay_rate * time_diff)

            # Importance score
            importance = episode.importance / 100.0

            # Relevance score (simple keyword matching)
            relevance = self._compute_relevance(query, episode)

            # Combined score
            score = (recency_weight * recency +
                    importance_weight * importance +
                    relevance_weight * relevance)
            scores.append((episode, score))

        # Sort by score and return top n
        scores.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scores[:n]]

    def reflect(self, current_time: int) -> List[Insight]:
        """
        Generate insights from recent memories.
        Key innovation from Generative Agents (Smallville).
        """
        recent = self.retrieve("", n=100, recency_weight=1.0, current_time=current_time)

        if len(recent) < 5:
            return []

        # Generate reflection questions
        questions = self._generate_reflection_questions(recent)

        # Synthesize insights
        new_insights = []
        for question in questions:
            relevant = self.retrieve(question, n=20, current_time=current_time)
            insight = self._synthesize_insight(question, relevant, current_time)
            if insight:
                new_insights.append(insight)

        self.insights.extend(new_insights)
        return new_insights

    def _compute_relevance(self, query: str, episode: Episode) -> float:
        """Compute relevance score between query and episode"""
        if not query:
            return 0.5

        query_words = set(query.lower().split())
        desc_words = set(episode.description.lower().split())
        overlap = len(query_words & desc_words)
        return min(1.0, overlap / max(len(query_words), 1))

    def _generate_reflection_questions(self, memories: List[Episode]) -> List[str]:
        """Generate questions for reflection"""
        questions = []

        # Analyze patterns in recent memories
        agents_seen = set()
        positive_count = 0
        negative_count = 0

        for mem in memories:
            agents_seen.update(mem.agents_involved)
            if mem.emotional_valence > 30:
                positive_count += 1
            elif mem.emotional_valence < -30:
                negative_count += 1

        # Generate questions based on patterns
        if len(agents_seen) > 0:
            questions.append(f"What patterns do I notice in my interactions?")

        if negative_count > positive_count:
            questions.append("Why have my recent experiences been challenging?")
        elif positive_count > negative_count * 2:
            questions.append("What has been going well recently?")

        if len(memories) > 20:
            questions.append("What have I learned from recent events?")

        return questions

    def _synthesize_insight(self, question: str, memories: List[Episode],
                           timestamp: int) -> Optional[Insight]:
        """Synthesize insight from memories"""
        if len(memories) < 3:
            return None

        # Simple pattern-based insight generation
        # (In full implementation, would use LLM)
        avg_valence = np.mean([m.emotional_valence for m in memories])
        common_agents = self._find_common_agents(memories)

        if "interactions" in question and common_agents:
            content = f"I frequently interact with agents {common_agents}. " \
                     f"These interactions tend to be {'positive' if avg_valence > 0 else 'challenging'}."
        elif "challenging" in question:
            content = f"Recent challenges seem related to repeated patterns. " \
                     f"Average emotional experience: {avg_valence:.1f}"
        elif "well" in question:
            content = f"Recent successes are building positive momentum. " \
                     f"Key relationships are strengthening."
        else:
            content = f"Reflection on '{question}': patterns emerging from {len(memories)} memories."

        return Insight(
            content=content,
            supporting_memories=memories,
            generated_at=timestamp,
            insight_type="pattern"
        )

    def _find_common_agents(self, memories: List[Episode]) -> List[int]:
        """Find agents that appear frequently in memories"""
        counts = defaultdict(int)
        for mem in memories:
            for agent_id in mem.agents_involved:
                counts[agent_id] += 1

        # Return agents appearing in more than 20% of memories
        threshold = len(memories) * 0.2
        return [aid for aid, count in counts.items() if count > threshold]


@dataclass
class NestedBelief:
    """
    BeliefNest-style nested belief representation.
    Supports k-ToM depth reasoning.
    """
    # Level 0: Agent's own beliefs about world
    world_beliefs: Dict[str, Any] = field(default_factory=dict)

    # Level 1: Agent's beliefs about others' beliefs
    other_beliefs: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Level 2: Agent's beliefs about others' beliefs about self
    meta_beliefs: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Level 3+: Deeper recursion (up to k=5)
    deeper_beliefs: Dict[int, 'NestedBelief'] = field(default_factory=dict)

    # Confidence at each level (decays with depth)
    confidence_decay: float = 0.7

    def get_belief_at_depth(self, depth: int, path: List[int] = None) -> Dict[str, Any]:
        """
        Get beliefs at specified depth along path of agent IDs.
        depth=0: own beliefs
        depth=1: beliefs about agent path[0]
        depth=2: beliefs about path[0]'s beliefs about path[1]
        etc.
        """
        if path is None:
            path = []

        if depth == 0:
            return self.world_beliefs
        elif depth == 1 and len(path) >= 1:
            return self.other_beliefs.get(path[0], {})
        elif depth == 2 and len(path) >= 1:
            return self.meta_beliefs.get(path[0], {})
        elif depth >= 3 and len(path) >= 1:
            if path[0] in self.deeper_beliefs:
                return self.deeper_beliefs[path[0]].get_belief_at_depth(
                    depth - 1, path[1:]
                )
        return {}

    def update_belief(self, depth: int, path: List[int], key: str, value: Any):
        """Update a belief at specified depth"""
        if depth == 0:
            self.world_beliefs[key] = value
        elif depth == 1 and len(path) >= 1:
            if path[0] not in self.other_beliefs:
                self.other_beliefs[path[0]] = {}
            self.other_beliefs[path[0]][key] = value
        elif depth == 2 and len(path) >= 1:
            if path[0] not in self.meta_beliefs:
                self.meta_beliefs[path[0]] = {}
            self.meta_beliefs[path[0]][key] = value

    def get_confidence(self, depth: int) -> float:
        """Get confidence level at given depth"""
        return self.confidence_decay ** depth


@dataclass
class Hypothesis:
    """Hypothesis about another agent (Hypothetical Minds style)"""
    agent_id: int
    strategy: StrategyType
    beliefs: Dict[str, Any]
    desires: List[str]
    intentions: List[str]
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result of the MetaMind three-stage reasoning pipeline"""
    hypotheses: List[Hypothesis]
    refined_hypotheses: List[Hypothesis]
    selected_action: Dict[str, Any]
    reasoning_trace: List[str] = field(default_factory=list)


class KToMReasoner:
    """
    k-ToM reasoning based on tomsup formalism.
    Maps to psychosocial Layer 3 (tom_depth).
    """

    def __init__(self, tom_depth: int = 2, volatility: float = 0.5):
        self.k = tom_depth
        self.volatility = volatility

        # Belief tracking
        self.beliefs: Dict[int, Dict[str, float]] = {}  # agent_id -> beliefs
        self.opponent_k_estimates: Dict[int, float] = {}  # Estimated k of others

    def update_beliefs(self, agent_id: int, observation: Dict) -> Dict:
        """Update beliefs about agent using k-ToM Bayesian update"""
        if agent_id not in self.beliefs:
            self.beliefs[agent_id] = {'cooperation_prob': 0.5}
            self.opponent_k_estimates[agent_id] = 1.0  # Prior

        if 'action' in observation:
            # Update cooperation probability
            old_prob = self.beliefs[agent_id]['cooperation_prob']
            if observation['action'] == 'cooperate':
                new_prob = old_prob + self.volatility * (1 - old_prob)
            else:
                new_prob = old_prob - self.volatility * old_prob
            self.beliefs[agent_id]['cooperation_prob'] = np.clip(new_prob, 0.01, 0.99)

            # Update estimated k
            self._update_k_estimate(agent_id, observation)

        return {
            'beliefs': self.beliefs[agent_id].copy(),
            'estimated_k': self.opponent_k_estimates[agent_id],
            'confidence': 1.0 / (1 + self.volatility)
        }

    def _update_k_estimate(self, agent_id: int, observation: Dict):
        """Update estimate of other agent's k level"""
        # Simple heuristic: sophisticated behavior suggests higher k
        if 'behavior_complexity' in observation:
            complexity = observation['behavior_complexity']
            current = self.opponent_k_estimates[agent_id]
            # Bayesian-ish update toward complexity indicator
            self.opponent_k_estimates[agent_id] = (
                0.9 * current + 0.1 * min(5, complexity)
            )

    def predict_action(self, agent_id: int, context: Dict) -> str:
        """Predict other agent's action using k-ToM"""
        if agent_id not in self.beliefs:
            return 'cooperate' if np.random.random() < 0.5 else 'defect'

        coop_prob = self.beliefs[agent_id]['cooperation_prob']

        # Adjust based on k-level reasoning
        if self.k >= 2:
            # Consider what they think I'll do
            # This is simplified; full implementation would recurse
            estimated_k = self.opponent_k_estimates[agent_id]
            if estimated_k >= 1:
                # They're reasoning about me
                coop_prob *= 0.9  # Slightly more likely to defect

        return 'cooperate' if coop_prob > 0.5 else 'defect'


class HypothesisEvaluator:
    """
    Hypothetical Minds approach to ToM.
    Generate and evaluate competing hypotheses about other agents.
    """

    def __init__(self):
        self.strategy_space = list(StrategyType)
        self.hypothesis_scores: Dict[int, Dict[StrategyType, float]] = {}

    def generate_hypotheses(self, agent_id: int, observation: Dict) -> List[Hypothesis]:
        """Generate competing hypotheses about agent's strategy/goals"""
        hypotheses = []

        if agent_id not in self.hypothesis_scores:
            # Initialize with uniform prior
            self.hypothesis_scores[agent_id] = {
                s: 1.0 / len(self.strategy_space) for s in self.strategy_space
            }

        for strategy in self.strategy_space:
            likelihood = self.hypothesis_scores[agent_id][strategy]

            # Generate beliefs/desires/intentions for this strategy
            beliefs = self._infer_beliefs(strategy, observation)
            desires = self._infer_desires(strategy)
            intentions = self._infer_intentions(strategy, observation)

            hypotheses.append(Hypothesis(
                agent_id=agent_id,
                strategy=strategy,
                beliefs=beliefs,
                desires=desires,
                intentions=intentions,
                confidence=likelihood
            ))

        return hypotheses

    def update(self, agent_id: int, observation: Dict):
        """Update hypothesis scores based on observed action"""
        if 'action' not in observation:
            return

        action = observation['action']

        if agent_id not in self.hypothesis_scores:
            self.generate_hypotheses(agent_id, observation)

        for strategy in self.strategy_space:
            expected = self._expected_action(strategy, observation)
            if expected == action:
                self.hypothesis_scores[agent_id][strategy] *= 1.3
            else:
                self.hypothesis_scores[agent_id][strategy] *= 0.7

        # Normalize
        total = sum(self.hypothesis_scores[agent_id].values())
        for strategy in self.strategy_space:
            self.hypothesis_scores[agent_id][strategy] /= total

    def get_best_hypothesis(self, agent_id: int) -> Optional[Hypothesis]:
        """Get most likely hypothesis for an agent"""
        if agent_id not in self.hypothesis_scores:
            return None

        best_strategy = max(
            self.hypothesis_scores[agent_id],
            key=self.hypothesis_scores[agent_id].get
        )
        return Hypothesis(
            agent_id=agent_id,
            strategy=best_strategy,
            beliefs={},
            desires=self._infer_desires(best_strategy),
            intentions=[],
            confidence=self.hypothesis_scores[agent_id][best_strategy]
        )

    def _expected_action(self, strategy: StrategyType, context: Dict) -> str:
        """What action would this strategy produce in this context?"""
        if strategy == StrategyType.COOPERATIVE:
            return 'cooperate'
        elif strategy == StrategyType.DEFECTOR:
            return 'defect'
        elif strategy == StrategyType.TIT_FOR_TAT:
            return context.get('last_opponent_action', 'cooperate')
        elif strategy == StrategyType.RANDOM:
            return 'cooperate' if np.random.random() < 0.5 else 'defect'
        elif strategy == StrategyType.GRIM_TRIGGER:
            if context.get('opponent_ever_defected', False):
                return 'defect'
            return 'cooperate'
        elif strategy == StrategyType.EXPLOITER:
            return 'defect'
        elif strategy == StrategyType.ALTRUIST:
            return 'cooperate'
        return 'cooperate'

    def _infer_beliefs(self, strategy: StrategyType, observation: Dict) -> Dict:
        """Infer beliefs associated with a strategy"""
        beliefs = {}
        if strategy in [StrategyType.COOPERATIVE, StrategyType.ALTRUIST]:
            beliefs['others_cooperative'] = True
        elif strategy in [StrategyType.DEFECTOR, StrategyType.EXPLOITER]:
            beliefs['others_exploitable'] = True
        return beliefs

    def _infer_desires(self, strategy: StrategyType) -> List[str]:
        """Infer desires associated with a strategy"""
        if strategy == StrategyType.COOPERATIVE:
            return ['mutual_benefit', 'relationship']
        elif strategy == StrategyType.DEFECTOR:
            return ['personal_gain', 'dominance']
        elif strategy == StrategyType.TIT_FOR_TAT:
            return ['fairness', 'reciprocity']
        elif strategy == StrategyType.ALTRUIST:
            return ['others_welfare', 'meaning']
        else:
            return ['survival', 'resources']

    def _infer_intentions(self, strategy: StrategyType, observation: Dict) -> List[str]:
        """Infer intentions based on strategy and context"""
        intentions = []
        if strategy == StrategyType.EXPLOITER:
            intentions.append('exploit_cooperators')
        elif strategy == StrategyType.TIT_FOR_TAT:
            intentions.append('reciprocate_last_action')
        return intentions


@dataclass
class IntegratedAgent:
    """
    Complete agent integrating all ToM frameworks:
    - Psychosocial taxonomy (internal state)
    - Success taxonomy (fitness landscape)
    - k-ToM reasoning (tomsup)
    - MetaMind pipeline
    - BeliefNest (nested beliefs)
    - Hypothetical Minds (hypothesis evaluation)
    - Generative Agents (memory)
    """
    id: int

    # Position in world
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # === TAXONOMIES ===
    profile: PsychosocialProfile = field(default_factory=PsychosocialProfile)
    success: SuccessState = field(default_factory=SuccessState)
    institutional_context: InstitutionalContext = field(default_factory=InstitutionalContext)

    # === INTEGRATED FRAMEWORKS ===
    tom_reasoner: KToMReasoner = field(default=None)
    belief_tracker: NestedBelief = field(default_factory=NestedBelief)
    hypothesis_eval: HypothesisEvaluator = field(default_factory=HypothesisEvaluator)
    memory: AgentMemory = field(default=None)

    # Genome for evolution
    genome: Dict = field(default_factory=dict)

    # Is zombie (baseline without real ToM)?
    is_zombie: bool = False
    zombie_type: Optional[str] = None

    def __post_init__(self):
        """Initialize integrated components"""
        if self.tom_reasoner is None:
            self.tom_reasoner = KToMReasoner(
                tom_depth=self.profile.layer3.tom_depth,
                volatility=1.0 - self.profile.layer3.uncertainty_tolerance / 100.0
            )
        if self.memory is None:
            self.memory = AgentMemory(
                decay_rate=1.0 / max(1, self.profile.layer3.working_memory_capacity / 10)
            )

    @property
    def fitness(self) -> float:
        """Compute fitness using PRIVATE value weights"""
        if not self.success.alive:
            return 0.0
        weights = self.profile.get_success_weights()

        # Apply institutional modifiers
        inst_mods = self.institutional_context.get_combined_modifiers()
        base_fitness = self.success.compute_fitness(weights)

        # Institutional context affects fitness
        structural_advantage = self.institutional_context.get_structural_advantage()
        return base_fitness * (0.7 + 0.3 * structural_advantage)

    def perceive(self, world_state: Dict) -> Dict:
        """
        Layer 1: Perception with partial observability.
        Can only see within observation radius, not through walls.
        """
        observation = {
            'self': {
                'position': (self.x, self.y, self.z),
                'health': self.success.domain1.health,
                'resources': self.success.domain2.net_worth,
                'energy': self.profile.layer0.baseline_energy,
            },
            'nearby_agents': [],
            'nearby_resources': [],
            'timestamp': world_state.get('timestep', 0),
        }

        radius = self._get_observation_radius()

        # Filter visible agents
        for agent_state in world_state.get('agents', []):
            if agent_state['id'] == self.id:
                continue

            dist = np.sqrt(
                (self.x - agent_state['x'])**2 +
                (self.y - agent_state['y'])**2 +
                (self.z - agent_state.get('z', 0))**2
            )

            if dist <= radius:
                # Can only observe PUBLIC information
                observation['nearby_agents'].append({
                    'id': agent_state['id'],
                    'position': (agent_state['x'], agent_state['y'], agent_state.get('z', 0)),
                    'distance': dist,
                    'apparent_wealth': agent_state.get('visible_wealth', 50),
                    'reputation': agent_state.get('reputation', 50),
                    'last_action': agent_state.get('last_action'),
                    'apparent_emotion': agent_state.get('apparent_emotion', 'neutral'),
                })

        # Filter visible resources
        for resource in world_state.get('resources', []):
            dist = np.sqrt(
                (self.x - resource['x'])**2 +
                (self.y - resource['y'])**2 +
                (self.z - resource.get('z', 0))**2
            )
            if dist <= radius:
                observation['nearby_resources'].append({
                    'id': resource['id'],
                    'position': (resource['x'], resource['y'], resource.get('z', 0)),
                    'type': resource.get('type', 'generic'),
                    'value': resource.get('value', 10),
                    'distance': dist,
                })

        return observation

    def reason(self, observation: Dict) -> ReasoningResult:
        """
        Layer 2: ToM Reasoning
        Integrates: k-ToM + MetaMind + BeliefNest + Hypothetical Minds
        """
        if self.is_zombie:
            return self._zombie_reasoning(observation)

        reasoning_trace = []

        # Stage 1: Update memories (Generative Agents)
        self._update_memories(observation)
        reasoning_trace.append("Updated episodic memory")

        # Stage 2: Update nested beliefs (BeliefNest)
        for agent_obs in observation.get('nearby_agents', []):
            agent_id = agent_obs['id']

            # Update Level 0: World beliefs
            self.belief_tracker.update_belief(
                depth=0, path=[], key=f'agent_{agent_id}_position',
                value=agent_obs['position']
            )

            # Update Level 1: Beliefs about this agent
            self.belief_tracker.update_belief(
                depth=1, path=[agent_id], key='reputation',
                value=agent_obs['reputation']
            )

            reasoning_trace.append(f"Updated beliefs about agent {agent_id}")

        # Stage 3: k-ToM belief updates (tomsup)
        tom_updates = {}
        for agent_obs in observation.get('nearby_agents', []):
            if agent_obs.get('last_action'):
                tom_update = self.tom_reasoner.update_beliefs(
                    agent_obs['id'],
                    {'action': agent_obs['last_action']}
                )
                tom_updates[agent_obs['id']] = tom_update
        reasoning_trace.append(f"k-ToM updates for {len(tom_updates)} agents")

        # Stage 4: Generate hypotheses (Hypothetical Minds)
        all_hypotheses = []
        for agent_obs in observation.get('nearby_agents', []):
            hypotheses = self.hypothesis_eval.generate_hypotheses(
                agent_obs['id'], agent_obs
            )
            all_hypotheses.extend(hypotheses)

            # Update hypothesis scores
            if agent_obs.get('last_action'):
                self.hypothesis_eval.update(
                    agent_obs['id'],
                    {'action': agent_obs['last_action']}
                )
        reasoning_trace.append(f"Generated {len(all_hypotheses)} hypotheses")

        # Stage 5: Refine hypotheses via social norms (MetaMind Domain Agent)
        refined_hypotheses = self._refine_hypotheses(all_hypotheses, observation)
        reasoning_trace.append(f"Refined to {len(refined_hypotheses)} hypotheses")

        # Stage 6: Select action (MetaMind Response Agent)
        action = self._select_action(refined_hypotheses, observation)
        reasoning_trace.append(f"Selected action: {action.get('type', 'unknown')}")

        return ReasoningResult(
            hypotheses=all_hypotheses,
            refined_hypotheses=refined_hypotheses,
            selected_action=action,
            reasoning_trace=reasoning_trace
        )

    def _refine_hypotheses(self, hypotheses: List[Hypothesis],
                         observation: Dict) -> List[Hypothesis]:
        """
        MetaMind Domain Agent: Refine hypotheses via social norms.
        Uses Layer 6 Social + Layer 5 Values.
        """
        refined = []

        for hyp in hypotheses:
            # Check consistency with social norms
            norm_consistent = True

            # Trust check
            if hyp.strategy == StrategyType.EXPLOITER:
                if self.profile.layer6.trust_default > 70:
                    # High trust agents discount exploiter hypotheses
                    hyp.confidence *= 0.6
                    norm_consistent = False

            # Fairness check
            if hyp.strategy == StrategyType.DEFECTOR:
                if self.profile.layer6.fairness_sensitivity > 70:
                    hyp.confidence *= 0.7

            # Relationship memory check
            if hyp.agent_id in self.memory.relationships:
                rel = self.memory.relationships[hyp.agent_id]
                if rel.trust_level > 70 and hyp.strategy in [
                    StrategyType.DEFECTOR, StrategyType.EXPLOITER
                ]:
                    hyp.confidence *= 0.5
                elif rel.trust_level < 30 and hyp.strategy == StrategyType.COOPERATIVE:
                    hyp.confidence *= 0.7

            refined.append(hyp)

        # Normalize confidences
        total = sum(h.confidence for h in refined)
        if total > 0:
            for h in refined:
                h.confidence /= total

        return refined

    def _select_action(self, hypotheses: List[Hypothesis],
                      observation: Dict) -> Dict[str, Any]:
        """
        MetaMind Response Agent: Select action based on refined hypotheses.
        Uses Layer 2 Motivational + Layer 5 Values.
        """
        weights = self.profile.get_success_weights()

        possible_actions = self._get_possible_actions(observation)
        action_scores = {}

        for action in possible_actions:
            score = self._evaluate_action(action, hypotheses, observation, weights)
            action_scores[action['type']] = (action, score)

        # Select best action
        if action_scores:
            best = max(action_scores.values(), key=lambda x: x[1])
            return best[0]

        # Default: rest/wait
        return {'type': 'rest'}

    def _evaluate_action(self, action: Dict, hypotheses: List[Hypothesis],
                        observation: Dict, weights: Dict[str, float]) -> float:
        """Evaluate an action given hypotheses about others"""
        score = 0.0

        if action['type'] == 'cooperate':
            # Cooperation value depends on others' likely response
            for hyp in hypotheses:
                if hyp.strategy in [StrategyType.COOPERATIVE, StrategyType.TIT_FOR_TAT]:
                    score += hyp.confidence * 1.0
                elif hyp.strategy == StrategyType.DEFECTOR:
                    score += hyp.confidence * -0.5

            # Personal values influence
            score += weights.get('relational', 0) * 0.5
            score += self.profile.layer6.cooperation_tendency / 100.0

        elif action['type'] == 'defect':
            # Defection value
            for hyp in hypotheses:
                if hyp.strategy == StrategyType.COOPERATIVE:
                    score += hyp.confidence * 0.8  # Exploit cooperators
                elif hyp.strategy == StrategyType.DEFECTOR:
                    score += hyp.confidence * 0.2

            # Personal values influence
            score -= weights.get('relational', 0) * 0.3
            score -= self.profile.layer6.fairness_sensitivity / 200.0

        elif action['type'] == 'move':
            # Movement toward resources or away from threats
            resources = observation.get('nearby_resources', [])
            if resources:
                closest = min(resources, key=lambda r: r['distance'])
                if closest['distance'] > 1:
                    score += 0.5  # Move toward resource

        elif action['type'] == 'communicate':
            score += self.profile.layer7.extraversion / 100.0 * 0.5

        return score

    def _get_possible_actions(self, observation: Dict) -> List[Dict[str, Any]]:
        """Get list of possible actions in current context"""
        actions = [
            {'type': 'rest'},
            {'type': 'move', 'direction': 'random'},
        ]

        if observation.get('nearby_agents'):
            actions.extend([
                {'type': 'cooperate', 'target': observation['nearby_agents'][0]['id']},
                {'type': 'defect', 'target': observation['nearby_agents'][0]['id']},
                {'type': 'communicate', 'target': observation['nearby_agents'][0]['id']},
            ])

        if observation.get('nearby_resources'):
            actions.append({
                'type': 'collect',
                'target': observation['nearby_resources'][0]['id']
            })

        return actions

    def _zombie_reasoning(self, observation: Dict) -> ReasoningResult:
        """Simple reactive strategy for zombies (no real ToM)"""
        # Zombies use simple heuristics, not genuine mentalizing
        action = {'type': 'rest'}

        nearby = observation.get('nearby_agents', [])
        if nearby:
            # Simple pattern: mostly cooperate with occasional defection
            if np.random.random() < 0.6:
                action = {'type': 'cooperate', 'target': nearby[0]['id']}
            else:
                action = {'type': 'defect', 'target': nearby[0]['id']}
        elif observation.get('nearby_resources'):
            action = {'type': 'collect', 'target': observation['nearby_resources'][0]['id']}

        return ReasoningResult(
            hypotheses=[],
            refined_hypotheses=[],
            selected_action=action,
            reasoning_trace=['Zombie: reactive pattern only']
        )

    def _get_observation_radius(self) -> float:
        """Observation radius from Soul Map"""
        base = 5.0
        sensory = self.profile.layer0.sensory_sensitivity / 50
        attention = self.profile.layer3.attention_span / 50
        return base + sensory + attention

    def _update_memories(self, observation: Dict):
        """Add observation to memory"""
        timestamp = observation.get('timestamp', 0)

        for agent_obs in observation.get('nearby_agents', []):
            self.memory.add_episode(Episode(
                timestamp=timestamp,
                description=f"Observed agent {agent_obs['id']} at {agent_obs['position']}",
                agents_involved=[agent_obs['id']],
                emotional_valence=self.profile.layer1.joy - self.profile.layer1.fear,
                importance=30.0 + agent_obs.get('reputation', 0) * 0.2,
                location=(self.x, self.y, self.z),
            ))

    def reflect(self, timestep: int):
        """Periodic reflection (Generative Agents)"""
        insights = self.memory.reflect(timestep)

        # Update self-narrative
        for insight in insights:
            self.memory.self_narrative.append(insight.content)

    @classmethod
    def create_from_genome(cls, id: int, genome: Dict,
                          rng: Optional[np.random.Generator] = None) -> 'IntegratedAgent':
        """Create agent from evolutionary genome"""
        if rng is None:
            rng = np.random.default_rng(genome.get('profile_seed', id))

        # Sample profile based on genome parameters
        if genome.get('archetype'):
            profile = PsychosocialProfile.from_archetype(genome['archetype'], rng)
        else:
            profile = PsychosocialProfile.sample_random(rng)

        # Override ToM depth from genome
        if 'tom_depth' in genome:
            profile.layer3.tom_depth = genome['tom_depth']

        success = SuccessState.sample_random(rng)
        institutions = InstitutionalContext.sample_random(rng)

        return cls(
            id=id,
            profile=profile,
            success=success,
            institutional_context=institutions,
            genome=genome,
            is_zombie=genome.get('is_zombie', False),
            zombie_type=genome.get('zombie_type'),
        )

    def describe(self) -> str:
        """Generate human-readable description"""
        lines = [
            f"=== Agent {self.id} ===",
            f"Position: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f})",
            f"Fitness: {self.fitness:.3f}",
            f"Zombie: {self.is_zombie} ({self.zombie_type or 'N/A'})",
            "",
            f"ToM Depth: {self.profile.layer3.tom_depth}",
            f"Attachment: {self.profile.layer6.attachment_style.value}",
            f"Trust Default: {self.profile.layer6.trust_default:.0f}",
            f"Cooperation: {self.profile.layer6.cooperation_tendency:.0f}",
            "",
            f"Memories: {len(self.memory.episodic)} episodes",
            f"Relationships: {len(self.memory.relationships)}",
            f"Insights: {len(self.memory.insights)}",
        ]
        return '\n'.join(lines)
