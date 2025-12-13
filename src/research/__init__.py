"""
Research Cycle Engine

The complete end-to-end loop:
1. Observe environment → Form hypothesis (abductive reasoning)
2. Hypothesis → Synthesize experiment (λ-expression)
3. Execute experiment → Get results
4. Results → Update beliefs (Bayesian)
5. Updated beliefs → Publish findings
6. Publications → Influence other agents' beliefs
7. Successful patterns → Library compression (Stitch)
8. Repeat

This is NOT a visualization - it's the mechanism that creates
genuine selective pressure for Theory of Mind.
"""

import asyncio
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import logging
import math
import random

from src.config import get_logger
from src.config.constants import SOUL_MAP_DIMS

logger = get_logger(__name__)


class ResearchPhase(Enum):
    """Phases of the research cycle."""
    OBSERVING = "observing"
    HYPOTHESIZING = "hypothesizing"
    DESIGNING = "designing"
    EXECUTING = "executing"
    ANALYZING = "analyzing"
    PUBLISHING = "publishing"
    REVIEWING = "reviewing"
    EVOLVING = "evolving"


@dataclass
class Observation:
    """An observation from the environment."""
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    salience: float = 0.5  # How noteworthy this observation is


@dataclass
class Hypothesis:
    """A hypothesis formed from observations."""
    id: str
    statement: str
    observations: List[str]  # IDs of supporting observations
    confidence: float
    testable: bool
    created_at: datetime = field(default_factory=datetime.now)
    tested: bool = False
    supported: Optional[bool] = None


@dataclass
class ExperimentResult:
    """Results from running an experiment."""
    hypothesis_id: str
    success: bool
    data: Dict[str, Any]
    supports_hypothesis: bool
    confidence_delta: float  # How much this should change our belief
    execution_time: float
    error: Optional[str] = None


@dataclass
class BeliefUpdate:
    """A Bayesian update to beliefs."""
    prior: float
    likelihood: float
    posterior: float
    evidence_id: str
    timestamp: datetime = field(default_factory=datetime.now)


class ResearchCycle:
    """
    The complete research cycle for an agent.

    This is where cognition becomes science:
    - Observations trigger hypothesis formation
    - Hypotheses drive experimental design
    - Results update beliefs
    - Beliefs propagate through publication
    """

    def __init__(self, agent_id: str, synthesizer=None):
        self.agent_id = agent_id
        self.synthesizer = synthesizer
        self.phase = ResearchPhase.OBSERVING

        # State
        self.observations: List[Observation] = []
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.belief_state: Dict[str, float] = {}  # proposition -> confidence
        self.belief_history: List[BeliefUpdate] = []

        # Metrics
        self.experiments_run = 0
        self.hypotheses_confirmed = 0
        self.hypotheses_refuted = 0
        self.publications_created = 0

        # Configuration
        self.hypothesis_threshold = 0.3  # Min confidence to test
        self.publication_threshold = 0.7  # Min confidence to publish

    async def step(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step of the research cycle.

        Returns actions to take and any outputs (publications, etc.)
        """
        result = {
            "phase": self.phase.value,
            "actions": [],
            "outputs": [],
        }

        if self.phase == ResearchPhase.OBSERVING:
            observations = self._observe(environment_state)
            result["observations"] = len(observations)

            # Transition: if we have enough salient observations
            salient = [o for o in observations if o.salience > 0.5]
            if len(salient) >= 3 or len(self.observations) >= 10:
                self.phase = ResearchPhase.HYPOTHESIZING

        elif self.phase == ResearchPhase.HYPOTHESIZING:
            hypothesis = await self._form_hypothesis()
            if hypothesis:
                result["hypothesis"] = hypothesis.statement
                self.phase = ResearchPhase.DESIGNING
            else:
                self.phase = ResearchPhase.OBSERVING

        elif self.phase == ResearchPhase.DESIGNING:
            experiment = await self._design_experiment()
            if experiment:
                result["experiment"] = experiment
                self.phase = ResearchPhase.EXECUTING
            else:
                self.phase = ResearchPhase.OBSERVING

        elif self.phase == ResearchPhase.EXECUTING:
            execution_result = await self._execute_experiment()
            result["result"] = execution_result
            self.phase = ResearchPhase.ANALYZING

        elif self.phase == ResearchPhase.ANALYZING:
            analysis = self._analyze_results()
            result["analysis"] = analysis
            self.phase = ResearchPhase.PUBLISHING

        elif self.phase == ResearchPhase.PUBLISHING:
            publication = self._create_publication()
            if publication:
                result["outputs"].append(publication)
                self.publications_created += 1
            self.phase = ResearchPhase.EVOLVING

        elif self.phase == ResearchPhase.EVOLVING:
            capabilities = await self._evolve_capabilities()
            result["new_capabilities"] = capabilities
            self.phase = ResearchPhase.OBSERVING

        return result

    def _observe(self, env_state: Dict[str, Any]) -> List[Observation]:
        """Extract observations from environment state."""
        new_observations = []

        # Observe agents
        for agent_id, agent_data in env_state.get("agents", {}).items():
            if agent_id != self.agent_id:
                obs = Observation(
                    timestamp=datetime.now(),
                    source=f"agent:{agent_id}",
                    data={
                        "activity": agent_data.get("activity"),
                        "thought": agent_data.get("thought"),
                        "position": agent_data.get("position"),
                    },
                    salience=self._compute_salience(agent_data)
                )
                new_observations.append(obs)
                self.observations.append(obs)

        # Observe publications
        for pub in env_state.get("publications", []):
            obs = Observation(
                timestamp=datetime.now(),
                source=f"publication:{pub.get('id')}",
                data=pub,
                salience=0.7  # Publications are generally salient
            )
            new_observations.append(obs)
            self.observations.append(obs)

        # Observe simulations
        for sim in env_state.get("simulations", []):
            obs = Observation(
                timestamp=datetime.now(),
                source=f"simulation:{sim.get('id')}",
                data=sim,
                salience=0.8  # Simulations are very interesting
            )
            new_observations.append(obs)
            self.observations.append(obs)

        # Limit observation history
        if len(self.observations) > 100:
            self.observations = self.observations[-100:]

        return new_observations

    def _compute_salience(self, data: Dict[str, Any]) -> float:
        """Compute how noteworthy an observation is."""
        salience = 0.3  # Base salience

        # Unusual activities are more salient
        activity = data.get("activity", "idle")
        if activity in ["simulating", "publishing"]:
            salience += 0.3
        elif activity == "researching":
            salience += 0.2

        # Thoughts about ToM are very salient
        thought = data.get("thought", "").lower()
        if "belief" in thought or "think" in thought or "model" in thought:
            salience += 0.2

        return min(1.0, salience)

    async def _form_hypothesis(self) -> Optional[Hypothesis]:
        """Form a hypothesis from recent observations using abductive reasoning."""

        # Get recent salient observations
        recent = self.observations[-20:]
        salient = [o for o in recent if o.salience > 0.4]

        if not salient:
            return None

        # Pattern detection: look for regularities
        patterns = self._detect_patterns(salient)

        if not patterns:
            return None

        # Form hypothesis from strongest pattern
        pattern = max(patterns, key=lambda p: p["strength"])

        hypothesis_id = f"H_{len(self.hypotheses)}"
        hypothesis = Hypothesis(
            id=hypothesis_id,
            statement=pattern["statement"],
            observations=[o.source for o in pattern["observations"]],
            confidence=pattern["strength"],
            testable=pattern.get("testable", True),
        )

        self.hypotheses[hypothesis_id] = hypothesis
        self._current_hypothesis = hypothesis

        logger.info(f"Agent {self.agent_id} formed hypothesis: {hypothesis.statement}")

        return hypothesis

    def _detect_patterns(self, observations: List[Observation]) -> List[Dict[str, Any]]:
        """Detect patterns in observations that could form hypotheses."""
        patterns = []

        # Pattern 1: Agent behavior correlations
        agent_activities = {}
        for obs in observations:
            if obs.source.startswith("agent:"):
                agent_id = obs.source.split(":")[1]
                activity = obs.data.get("activity")
                agent_activities.setdefault(activity, []).append(agent_id)

        for activity, agents in agent_activities.items():
            if len(agents) >= 2:
                patterns.append({
                    "type": "correlation",
                    "statement": f"Multiple agents ({len(agents)}) engaged in {activity} simultaneously - possible coordination or common trigger",
                    "observations": [o for o in observations if o.data.get("activity") == activity],
                    "strength": 0.4 + 0.1 * len(agents),
                    "testable": True,
                })

        # Pattern 2: Publication influence
        pub_obs = [o for o in observations if o.source.startswith("publication:")]
        if pub_obs:
            patterns.append({
                "type": "influence",
                "statement": f"Publication activity detected - testing if this influences agent beliefs",
                "observations": pub_obs,
                "strength": 0.5,
                "testable": True,
            })

        # Pattern 3: Simulation spawning
        sim_obs = [o for o in observations if o.source.startswith("simulation:")]
        if sim_obs:
            patterns.append({
                "type": "recursion",
                "statement": f"Nested simulation detected - agents may be modeling other agents",
                "observations": sim_obs,
                "strength": 0.7,
                "testable": True,
            })

        # Pattern 4: Belief state changes
        thought_keywords = {}
        for obs in observations:
            thought = obs.data.get("thought", "")
            for keyword in ["hypothesis", "belief", "predict", "expect", "model"]:
                if keyword in thought.lower():
                    thought_keywords.setdefault(keyword, []).append(obs)

        for keyword, keyword_obs in thought_keywords.items():
            if len(keyword_obs) >= 2:
                patterns.append({
                    "type": "metacognition",
                    "statement": f"Agents are explicitly reasoning about '{keyword}' - possible ToM activity",
                    "observations": keyword_obs,
                    "strength": 0.6 + 0.05 * len(keyword_obs),
                    "testable": True,
                })

        return patterns

    async def _design_experiment(self) -> Optional[Dict[str, Any]]:
        """Design an experiment to test the current hypothesis."""
        if not hasattr(self, '_current_hypothesis'):
            return None

        hypothesis = self._current_hypothesis

        # Use synthesizer if available
        if self.synthesizer:
            try:
                program = await self.synthesizer.synthesize_async(
                    f"experiment to test: {hypothesis.statement}"
                )
                if program:
                    self._current_experiment = {
                        "hypothesis_id": hypothesis.id,
                        "program": program,
                        "design": f"λ-experiment for: {hypothesis.statement}",
                    }
                    return self._current_experiment
            except Exception as e:
                logger.warning(f"Synthesis failed: {e}")

        # Fallback: template-based experiment design
        experiment = {
            "hypothesis_id": hypothesis.id,
            "design": f"Observation-based test for: {hypothesis.statement}",
            "methodology": "observe_and_compare",
            "duration_ticks": 50,
        }

        self._current_experiment = experiment
        return experiment

    async def _execute_experiment(self) -> ExperimentResult:
        """Execute the current experiment."""
        if not hasattr(self, '_current_experiment'):
            return ExperimentResult(
                hypothesis_id="unknown",
                success=False,
                data={},
                supports_hypothesis=False,
                confidence_delta=0,
                execution_time=0,
                error="No experiment to execute"
            )

        experiment = self._current_experiment
        hypothesis_id = experiment["hypothesis_id"]

        import time
        start_time = time.time()

        try:
            # If we have a λ-program, execute it
            if "program" in experiment and self.synthesizer:
                result = self.synthesizer.evaluate(experiment["program"], {})
                execution_time = time.time() - start_time

                # Interpret result as support/refute
                if isinstance(result, bool):
                    supports = result
                elif isinstance(result, (int, float)):
                    supports = result > 0.5
                else:
                    supports = result is not None

                confidence_delta = 0.2 if supports else -0.15

            else:
                # Simulated experiment (observation-based)
                execution_time = time.time() - start_time

                # Stochastic result based on hypothesis quality
                hypothesis = self.hypotheses.get(hypothesis_id)
                if hypothesis:
                    # Better hypotheses more likely to be supported
                    supports = random.random() < (0.3 + hypothesis.confidence * 0.5)
                else:
                    supports = random.random() < 0.4

                confidence_delta = 0.15 if supports else -0.1

            self.experiments_run += 1

            return ExperimentResult(
                hypothesis_id=hypothesis_id,
                success=True,
                data={"raw_result": result if "program" in experiment else "observational"},
                supports_hypothesis=supports,
                confidence_delta=confidence_delta,
                execution_time=execution_time,
            )

        except Exception as e:
            return ExperimentResult(
                hypothesis_id=hypothesis_id,
                success=False,
                data={},
                supports_hypothesis=False,
                confidence_delta=-0.1,
                execution_time=time.time() - start_time,
                error=str(e)
            )

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results and update beliefs."""
        if not hasattr(self, '_last_result'):
            self._last_result = ExperimentResult(
                hypothesis_id="unknown",
                success=False,
                data={},
                supports_hypothesis=False,
                confidence_delta=0,
                execution_time=0,
            )

        result = self._last_result
        hypothesis = self.hypotheses.get(result.hypothesis_id)

        if not hypothesis:
            return {"error": "Hypothesis not found"}

        # Bayesian update
        prior = hypothesis.confidence
        likelihood = 0.7 if result.supports_hypothesis else 0.3

        # Bayes' rule (simplified)
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior)
        )

        # Apply update
        hypothesis.tested = True
        hypothesis.supported = result.supports_hypothesis
        hypothesis.confidence = posterior

        # Record update
        update = BeliefUpdate(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence_id=result.hypothesis_id,
        )
        self.belief_history.append(update)

        # Update proposition belief
        self.belief_state[hypothesis.statement] = posterior

        # Track statistics
        if result.supports_hypothesis:
            self.hypotheses_confirmed += 1
        else:
            self.hypotheses_refuted += 1

        logger.info(
            f"Agent {self.agent_id} updated belief: "
            f"{hypothesis.statement[:50]}... "
            f"({prior:.2f} → {posterior:.2f})"
        )

        return {
            "hypothesis_id": hypothesis.id,
            "prior": prior,
            "posterior": posterior,
            "supported": result.supports_hypothesis,
        }

    def _create_publication(self) -> Optional[Dict[str, Any]]:
        """Create a publication from confirmed hypotheses."""
        # Find hypotheses worth publishing
        publishable = [
            h for h in self.hypotheses.values()
            if h.tested and h.confidence >= self.publication_threshold
        ]

        if not publishable:
            return None

        # Take the best one
        best = max(publishable, key=lambda h: h.confidence)

        publication = {
            "id": f"pub_{self.agent_id}_{self.publications_created}",
            "author": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "title": f"On {best.statement[:50]}...",
            "hypothesis": best.statement,
            "confidence": best.confidence,
            "supported": best.supported,
            "methodology": "neurosymbolic_experiment" if self.synthesizer else "observational",
            "observations_used": len(best.observations),
        }

        return publication

    async def _evolve_capabilities(self) -> List[Dict[str, Any]]:
        """Compress successful patterns into library."""
        if not self.synthesizer:
            return []

        try:
            capabilities = self.synthesizer.evolve_library()
            return [c.to_agent_card() for c in capabilities]
        except Exception as e:
            logger.warning(f"Library evolution failed: {e}")
            return []

    def receive_publication(self, publication: Dict[str, Any]):
        """
        Receive and process a publication from another agent.

        This is how beliefs propagate through the population.
        """
        author = publication.get("author")
        if author == self.agent_id:
            return  # Don't update from own publications

        # Extract hypothesis and confidence
        hypothesis_statement = publication.get("hypothesis", "")
        pub_confidence = publication.get("confidence", 0.5)
        supported = publication.get("supported", True)

        # Get our current belief (or default)
        current_belief = self.belief_state.get(hypothesis_statement, 0.5)

        # Weight update by our trust in the author (simplified: equal trust)
        trust_weight = 0.3

        # Update our belief toward the publication's position
        if supported:
            new_belief = current_belief + trust_weight * (pub_confidence - current_belief)
        else:
            new_belief = current_belief - trust_weight * current_belief

        new_belief = max(0.0, min(1.0, new_belief))

        self.belief_state[hypothesis_statement] = new_belief

        logger.debug(
            f"Agent {self.agent_id} updated belief from publication: "
            f"{hypothesis_statement[:30]}... ({current_belief:.2f} → {new_belief:.2f})"
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current research state for visualization."""
        return {
            "agent_id": self.agent_id,
            "phase": self.phase.value,
            "observations_count": len(self.observations),
            "hypotheses_count": len(self.hypotheses),
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "publications_created": self.publications_created,
            "belief_state_size": len(self.belief_state),
            "current_hypothesis": getattr(self, '_current_hypothesis', Hypothesis(
                id="none", statement="", observations=[], confidence=0, testable=False
            )).statement[:50] if hasattr(self, '_current_hypothesis') else None,
        }


class PopulationResearchManager:
    """
    Manages research cycles across a population of agents.

    Handles:
    - Publication broadcasting
    - Belief propagation
    - Fitness computation
    - Selection pressure
    """

    def __init__(self):
        self.agents: Dict[str, ResearchCycle] = {}
        self.publications: List[Dict[str, Any]] = []
        self.generation = 0

    def register_agent(self, agent_id: str, synthesizer=None):
        """Register an agent for research tracking."""
        cycle = ResearchCycle(agent_id, synthesizer)
        self.agents[agent_id] = cycle
        return cycle

    async def step_all(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one research step for all agents."""
        results = {}

        # Include publications in environment state
        env_with_pubs = {**environment_state, "publications": self.publications[-20:]}

        # Step each agent
        for agent_id, cycle in self.agents.items():
            agent_env = {**env_with_pubs}
            result = await cycle.step(agent_env)
            results[agent_id] = result

            # Broadcast any new publications
            for output in result.get("outputs", []):
                if "hypothesis" in output:  # It's a publication
                    self.publications.append(output)
                    self._broadcast_publication(output)

        return results

    def _broadcast_publication(self, publication: Dict[str, Any]):
        """Broadcast a publication to all agents."""
        author = publication.get("author")
        for agent_id, cycle in self.agents.items():
            if agent_id != author:
                cycle.receive_publication(publication)

    def compute_fitness(self, agent_id: str) -> float:
        """
        Compute fitness for selection.

        Fitness = f(publications, accuracy, influence)
        """
        if agent_id not in self.agents:
            return 0.0

        cycle = self.agents[agent_id]

        # Publication count (normalized)
        pub_score = min(1.0, cycle.publications_created / 10)

        # Accuracy (confirmed / tested)
        total_tested = cycle.hypotheses_confirmed + cycle.hypotheses_refuted
        if total_tested > 0:
            accuracy = cycle.hypotheses_confirmed / total_tested
        else:
            accuracy = 0.5

        # Influence (how many agents updated beliefs from our publications)
        influence = self._compute_influence(agent_id)

        fitness = 0.3 * pub_score + 0.4 * accuracy + 0.3 * influence

        return fitness

    def _compute_influence(self, agent_id: str) -> float:
        """Compute how influential an agent's publications are."""
        my_pubs = [p for p in self.publications if p.get("author") == agent_id]
        if not my_pubs:
            return 0.0

        # Average confidence of our publications
        avg_confidence = sum(p.get("confidence", 0) for p in my_pubs) / len(my_pubs)

        return min(1.0, avg_confidence * len(my_pubs) / 5)

    def select_and_reproduce(self, population_size: int) -> List[str]:
        """
        Select high-fitness agents for reproduction.

        Returns list of agent IDs to reproduce (will be cloned with variation).
        """
        # Compute fitness for all
        fitness_scores = {
            agent_id: self.compute_fitness(agent_id)
            for agent_id in self.agents
        }

        # Sort by fitness
        sorted_agents = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)

        # Top 50% reproduce
        num_reproduce = max(1, len(sorted_agents) // 2)
        reproducers = [agent_id for agent_id, _ in sorted_agents[:num_reproduce]]

        self.generation += 1

        return reproducers

    def get_population_stats(self) -> Dict[str, Any]:
        """Get statistics about the population."""
        if not self.agents:
            return {"population": 0}

        fitness_scores = [self.compute_fitness(aid) for aid in self.agents]

        return {
            "population": len(self.agents),
            "generation": self.generation,
            "total_publications": len(self.publications),
            "mean_fitness": sum(fitness_scores) / len(fitness_scores),
            "max_fitness": max(fitness_scores),
            "min_fitness": min(fitness_scores),
        }
