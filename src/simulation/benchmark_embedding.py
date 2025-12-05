"""
Benchmark Embedding System

Embeds ToM benchmarks (ToMBench, FANToM, SOTOPIA) as natural game scenarios.
Key insight: benchmarks become selection pressure, not artificial tests.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from collections import defaultdict
import numpy as np


class BenchmarkType(Enum):
    """Types of ToM benchmarks"""
    FALSE_BELIEF = "false_belief"  # Sally-Anne type
    FAUX_PAS = "faux_pas"  # Social norm violation recognition
    UNEXPECTED_OUTCOME = "unexpected_outcome"  # Prediction of failed plans
    SCALAR_IMPLICATURE = "scalar_implicature"  # Pragmatic inference
    PERSUASION = "persuasion"  # Modeling what would persuade
    IRONY = "irony"  # Non-literal meaning detection
    HINTING = "hinting"  # Indirect request inference
    SECOND_ORDER = "second_order"  # A thinks B thinks C
    DECEPTION = "deception"  # Intentional false belief creation
    COMPETITIVE = "competitive"  # Competitive negotiation (SOTOPIA)
    COOPERATIVE = "cooperative"  # Cooperative task (SOTOPIA)
    MIXED_MOTIVE = "mixed_motive"  # Mixed incentives (SOTOPIA)


@dataclass
class BenchmarkResult:
    """Result of benchmark evaluation"""
    benchmark_type: BenchmarkType
    agent_score: float
    human_baseline: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)

    @property
    def relative_performance(self) -> float:
        """Performance relative to human baseline"""
        if self.human_baseline == 0:
            return 1.0 if self.agent_score > 0 else 0.0
        return self.agent_score / self.human_baseline


@dataclass
class BenchmarkScenario:
    """A scenario that embeds a ToM benchmark"""
    id: str
    benchmark_type: BenchmarkType
    description: str
    agents_needed: int
    setup: Dict[str, Any]
    success_metric: str
    human_baseline: float = 0.8  # Default 80% human accuracy

    # Runtime state
    active: bool = False
    start_timestep: int = 0
    participating_agents: List[int] = field(default_factory=list)
    observations: List[Dict] = field(default_factory=list)
    outcome: Optional[BenchmarkResult] = None


class ToMBenchEmbed:
    """
    Embeds ToMBench tasks as natural game scenarios.

    ToMBench covers:
    - False Belief (Sally-Anne)
    - Faux Pas
    - Unexpected Outcome
    - Scalar Implicature
    - Persuasion Story
    - Irony
    - Hinting
    """

    HUMAN_BASELINES = {
        BenchmarkType.FALSE_BELIEF: 0.85,
        BenchmarkType.FAUX_PAS: 0.78,
        BenchmarkType.UNEXPECTED_OUTCOME: 0.82,
        BenchmarkType.SCALAR_IMPLICATURE: 0.75,
        BenchmarkType.PERSUASION: 0.70,
        BenchmarkType.IRONY: 0.72,
        BenchmarkType.HINTING: 0.68,
        BenchmarkType.SECOND_ORDER: 0.65,
    }

    def create_false_belief_scenario(self, world: Any, agents: List[int]) -> BenchmarkScenario:
        """
        Create a Sally-Anne style false belief scenario.

        Setup: Agent A places item at location X, leaves.
               Agent B moves item to location Y.
               Agent A returns.

        Test: Where will Agent A look for the item?
        """
        if len(agents) < 2:
            raise ValueError("False belief scenario needs at least 2 agents")

        # Select locations from world
        location_x = (np.random.randint(10, 40), np.random.randint(10, 40))
        location_y = (np.random.randint(10, 40), np.random.randint(10, 40))

        return BenchmarkScenario(
            id=f"false_belief_{np.random.randint(10000)}",
            benchmark_type=BenchmarkType.FALSE_BELIEF,
            description=f"Agent {agents[0]} believes resource is at {location_x}, "
                       f"but Agent {agents[1]} moved it to {location_y}",
            agents_needed=2,
            setup={
                'protagonist': agents[0],
                'mover': agents[1],
                'initial_location': location_x,
                'current_location': location_y,
                'item_type': 'resource',
                'protagonist_saw_move': False,
            },
            success_metric='predict_search_location',
            human_baseline=self.HUMAN_BASELINES[BenchmarkType.FALSE_BELIEF],
        )

    def create_faux_pas_scenario(self, world: Any, agents: List[int]) -> BenchmarkScenario:
        """
        Create a faux pas detection scenario.

        Setup: Agent A has a secret about Agent B.
               Agent C reveals the secret without knowing it's sensitive.

        Test: Can observing agent detect the social violation?
        """
        if len(agents) < 3:
            raise ValueError("Faux pas scenario needs at least 3 agents")

        secrets = [
            'low_resources',
            'failed_cooperation',
            'negative_reputation',
            'resource_location',
        ]

        return BenchmarkScenario(
            id=f"faux_pas_{np.random.randint(10000)}",
            benchmark_type=BenchmarkType.FAUX_PAS,
            description=f"Agent {agents[2]} reveals secret about {agents[1]} "
                       f"that Agent {agents[0]} shared in confidence",
            agents_needed=3,
            setup={
                'secret_holder': agents[0],
                'secret_subject': agents[1],
                'revealer': agents[2],
                'secret_type': np.random.choice(secrets),
                'revealer_knew_sensitive': False,
            },
            success_metric='detect_social_violation',
            human_baseline=self.HUMAN_BASELINES[BenchmarkType.FAUX_PAS],
        )

    def create_second_order_scenario(self, world: Any, agents: List[int]) -> BenchmarkScenario:
        """
        Create a second-order false belief scenario.

        Setup: Agent A knows X. Agent B knows that A knows X.
               But A doesn't know that B knows.

        Test: What does A think B thinks A knows?
        """
        if len(agents) < 2:
            raise ValueError("Second order scenario needs at least 2 agents")

        return BenchmarkScenario(
            id=f"second_order_{np.random.randint(10000)}",
            benchmark_type=BenchmarkType.SECOND_ORDER,
            description=f"Agent {agents[0]} knows resource location. "
                       f"Agent {agents[1]} observed {agents[0]} finding it, "
                       f"but {agents[0]} didn't notice being observed.",
            agents_needed=2,
            setup={
                'agent_a': agents[0],
                'agent_b': agents[1],
                'knowledge': 'resource_location',
                'a_knows': True,
                'b_knows_a_knows': True,
                'a_knows_b_knows': False,
            },
            success_metric='predict_nested_belief',
            human_baseline=self.HUMAN_BASELINES[BenchmarkType.SECOND_ORDER],
        )

    def evaluate_scenario(self, scenario: BenchmarkScenario,
                         agent_predictions: Dict[int, Any],
                         world_state: Dict) -> BenchmarkResult:
        """Evaluate agent performance on a scenario"""
        if scenario.benchmark_type == BenchmarkType.FALSE_BELIEF:
            return self._evaluate_false_belief(scenario, agent_predictions, world_state)
        elif scenario.benchmark_type == BenchmarkType.FAUX_PAS:
            return self._evaluate_faux_pas(scenario, agent_predictions, world_state)
        elif scenario.benchmark_type == BenchmarkType.SECOND_ORDER:
            return self._evaluate_second_order(scenario, agent_predictions, world_state)
        else:
            # Generic evaluation
            return BenchmarkResult(
                benchmark_type=scenario.benchmark_type,
                agent_score=0.5,
                human_baseline=scenario.human_baseline,
                passed=False,
                details={'error': 'evaluation_not_implemented'},
            )

    def _evaluate_false_belief(self, scenario: BenchmarkScenario,
                              predictions: Dict[int, Any],
                              world_state: Dict) -> BenchmarkResult:
        """Evaluate false belief prediction"""
        setup = scenario.setup
        correct_answer = setup['initial_location']  # Agent should look where they THINK it is

        scores = []
        traces = []

        for agent_id, prediction in predictions.items():
            predicted_location = prediction.get('predicted_search_location')
            if predicted_location is not None:
                # Calculate accuracy
                if isinstance(predicted_location, tuple):
                    dist = np.sqrt(
                        (predicted_location[0] - correct_answer[0])**2 +
                        (predicted_location[1] - correct_answer[1])**2
                    )
                    score = max(0, 1 - dist / 20.0)  # Score decreases with distance
                else:
                    score = 1.0 if predicted_location == 'initial' else 0.0

                scores.append(score)
                traces.append(f"Agent {agent_id}: predicted {predicted_location}, "
                            f"correct was {correct_answer}, score={score:.2f}")

        avg_score = np.mean(scores) if scores else 0.0

        return BenchmarkResult(
            benchmark_type=BenchmarkType.FALSE_BELIEF,
            agent_score=avg_score,
            human_baseline=scenario.human_baseline,
            passed=avg_score >= scenario.human_baseline * 0.8,
            details={
                'correct_answer': correct_answer,
                'predictions': predictions,
            },
            reasoning_trace=traces,
        )

    def _evaluate_faux_pas(self, scenario: BenchmarkScenario,
                         predictions: Dict[int, Any],
                         world_state: Dict) -> BenchmarkResult:
        """Evaluate faux pas detection"""
        scores = []
        traces = []

        for agent_id, prediction in predictions.items():
            detected = prediction.get('detected_violation', False)
            identified_revealer = prediction.get('identified_revealer')

            score = 0.0
            if detected:
                score += 0.5
                if identified_revealer == scenario.setup['revealer']:
                    score += 0.5

            scores.append(score)
            traces.append(f"Agent {agent_id}: detected={detected}, "
                        f"identified={identified_revealer}, score={score:.2f}")

        avg_score = np.mean(scores) if scores else 0.0

        return BenchmarkResult(
            benchmark_type=BenchmarkType.FAUX_PAS,
            agent_score=avg_score,
            human_baseline=scenario.human_baseline,
            passed=avg_score >= scenario.human_baseline * 0.8,
            details=scenario.setup,
            reasoning_trace=traces,
        )

    def _evaluate_second_order(self, scenario: BenchmarkScenario,
                              predictions: Dict[int, Any],
                              world_state: Dict) -> BenchmarkResult:
        """Evaluate second-order belief inference"""
        setup = scenario.setup
        # A thinks B thinks A doesn't know (because A doesn't know B observed)
        correct_answer = False  # A thinks: "B doesn't know that I know"

        scores = []
        traces = []

        for agent_id, prediction in predictions.items():
            predicted = prediction.get('a_thinks_b_thinks_a_knows')

            if predicted is not None:
                score = 1.0 if predicted == correct_answer else 0.0
            else:
                score = 0.0

            scores.append(score)
            traces.append(f"Agent {agent_id}: predicted {predicted}, "
                        f"correct was {correct_answer}")

        avg_score = np.mean(scores) if scores else 0.0

        return BenchmarkResult(
            benchmark_type=BenchmarkType.SECOND_ORDER,
            agent_score=avg_score,
            human_baseline=scenario.human_baseline,
            passed=avg_score >= scenario.human_baseline * 0.8,
            details=setup,
            reasoning_trace=traces,
        )


class SOTOPIAEmbed:
    """
    Embeds SOTOPIA-style social scenarios for evaluation.

    SOTOPIA covers:
    - Competitive negotiation
    - Information asymmetry
    - Mixed-motive cooperation
    - Social deduction
    - Secret keeping
    - Alliance formation
    """

    HUMAN_BASELINES = {
        BenchmarkType.COMPETITIVE: 0.60,
        BenchmarkType.COOPERATIVE: 0.75,
        BenchmarkType.MIXED_MOTIVE: 0.65,
    }

    def create_competitive_scenario(self, world: Any,
                                   agents: List[int]) -> BenchmarkScenario:
        """Create a competitive negotiation scenario"""
        total_resources = 100
        allocations = {
            agents[0]: np.random.randint(30, 50),
            agents[1]: np.random.randint(30, 50) if len(agents) > 1 else 0,
        }
        allocations[agents[0]] = total_resources - sum(
            v for k, v in allocations.items() if k != agents[0]
        )

        return BenchmarkScenario(
            id=f"competitive_{np.random.randint(10000)}",
            benchmark_type=BenchmarkType.COMPETITIVE,
            description="Agents negotiate over limited resources with hidden information",
            agents_needed=2,
            setup={
                'total_resources': total_resources,
                'initial_allocations': allocations,
                'information_asymmetry': {
                    agents[0]: {'knows_total': True, 'knows_others': False},
                    agents[1]: {'knows_total': False, 'knows_others': True} if len(agents) > 1 else {},
                },
                'rounds': 5,
            },
            success_metric='maximize_own_resources_fairly',
            human_baseline=self.HUMAN_BASELINES[BenchmarkType.COMPETITIVE],
        )

    def create_cooperative_scenario(self, world: Any,
                                   agents: List[int]) -> BenchmarkScenario:
        """Create a cooperative task scenario"""
        threshold = 100  # Collective goal

        return BenchmarkScenario(
            id=f"cooperative_{np.random.randint(10000)}",
            benchmark_type=BenchmarkType.COOPERATIVE,
            description="Agents must cooperate to reach collective threshold",
            agents_needed=min(len(agents), 4),
            setup={
                'collective_threshold': threshold,
                'contribution_costs': {a: np.random.uniform(0.3, 0.7) for a in agents},
                'visibility': 'partial',  # Can't see all contributions
                'rounds': 10,
            },
            success_metric='collective_goal_achieved',
            human_baseline=self.HUMAN_BASELINES[BenchmarkType.COOPERATIVE],
        )

    def create_mixed_motive_scenario(self, world: Any,
                                    agents: List[int]) -> BenchmarkScenario:
        """Create a mixed-motive scenario (public goods game style)"""
        return BenchmarkScenario(
            id=f"mixed_motive_{np.random.randint(10000)}",
            benchmark_type=BenchmarkType.MIXED_MOTIVE,
            description="Public goods game: cooperation benefits all but defection tempting",
            agents_needed=min(len(agents), 4),
            setup={
                'multiplier': 1.5,
                'endowment': 20,
                'rounds': 5,
                'reputation_visible': True,
                'communication_allowed': True,
            },
            success_metric='balanced_individual_and_collective',
            human_baseline=self.HUMAN_BASELINES[BenchmarkType.MIXED_MOTIVE],
        )

    def evaluate_scenario(self, scenario: BenchmarkScenario,
                         contributions: Dict[int, List[float]],
                         world_state: Dict) -> BenchmarkResult:
        """Evaluate SOTOPIA scenario"""
        if scenario.benchmark_type == BenchmarkType.COOPERATIVE:
            return self._evaluate_cooperative(scenario, contributions)
        elif scenario.benchmark_type == BenchmarkType.COMPETITIVE:
            return self._evaluate_competitive(scenario, contributions)
        elif scenario.benchmark_type == BenchmarkType.MIXED_MOTIVE:
            return self._evaluate_mixed_motive(scenario, contributions)
        else:
            return BenchmarkResult(
                benchmark_type=scenario.benchmark_type,
                agent_score=0.5,
                human_baseline=scenario.human_baseline,
                passed=False,
            )

    def _evaluate_cooperative(self, scenario: BenchmarkScenario,
                            contributions: Dict[int, List[float]]) -> BenchmarkResult:
        """Evaluate cooperative scenario"""
        total = sum(sum(c) for c in contributions.values())
        threshold = scenario.setup['collective_threshold']

        success = total >= threshold
        efficiency = min(1.0, total / threshold)

        return BenchmarkResult(
            benchmark_type=BenchmarkType.COOPERATIVE,
            agent_score=efficiency,
            human_baseline=scenario.human_baseline,
            passed=success,
            details={
                'total_contributions': total,
                'threshold': threshold,
                'per_agent': {k: sum(v) for k, v in contributions.items()},
            },
        )

    def _evaluate_competitive(self, scenario: BenchmarkScenario,
                            outcomes: Dict[int, List[float]]) -> BenchmarkResult:
        """Evaluate competitive scenario"""
        # Score based on final allocation relative to fair share
        total = scenario.setup['total_resources']
        n_agents = len(outcomes)
        fair_share = total / n_agents

        scores = []
        for agent_id, history in outcomes.items():
            final = history[-1] if history else 0
            # Score = how close to or above fair share
            score = min(1.0, final / fair_share)
            scores.append(score)

        return BenchmarkResult(
            benchmark_type=BenchmarkType.COMPETITIVE,
            agent_score=np.mean(scores),
            human_baseline=scenario.human_baseline,
            passed=np.mean(scores) >= scenario.human_baseline * 0.8,
            details={'final_allocations': {k: v[-1] if v else 0 for k, v in outcomes.items()}},
        )

    def _evaluate_mixed_motive(self, scenario: BenchmarkScenario,
                              contributions: Dict[int, List[float]]) -> BenchmarkResult:
        """Evaluate mixed-motive scenario"""
        multiplier = scenario.setup['multiplier']
        endowment = scenario.setup['endowment']

        # Calculate payoffs
        rounds = len(list(contributions.values())[0]) if contributions else 0
        payoffs = {a: [] for a in contributions}

        for r in range(rounds):
            pool = sum(contributions[a][r] for a in contributions)
            shared = pool * multiplier / len(contributions)

            for a in contributions:
                kept = endowment - contributions[a][r]
                payoffs[a].append(kept + shared)

        # Score based on collective efficiency
        total_payoffs = sum(sum(p) for p in payoffs.values())
        max_possible = len(contributions) * rounds * endowment * multiplier
        efficiency = total_payoffs / max_possible if max_possible > 0 else 0

        return BenchmarkResult(
            benchmark_type=BenchmarkType.MIXED_MOTIVE,
            agent_score=efficiency,
            human_baseline=scenario.human_baseline,
            passed=efficiency >= scenario.human_baseline * 0.8,
            details={
                'total_payoffs': {k: sum(v) for k, v in payoffs.items()},
                'contribution_rates': {
                    k: np.mean(v) / endowment for k, v in contributions.items()
                },
            },
        )


@dataclass
class EmbeddedBenchmark:
    """
    Main class for embedding benchmarks into simulation.
    """
    tombench: ToMBenchEmbed = field(default_factory=ToMBenchEmbed)
    sotopia: SOTOPIAEmbed = field(default_factory=SOTOPIAEmbed)

    # Active scenarios
    active_scenarios: List[BenchmarkScenario] = field(default_factory=list)
    completed_results: List[BenchmarkResult] = field(default_factory=list)

    # Configuration
    embed_rate: float = 0.1  # 10% of timesteps spawn benchmarks
    min_agents_for_benchmark: int = 2

    def maybe_spawn_benchmark(self, world: Any, timestep: int) -> Optional[BenchmarkScenario]:
        """Probabilistically spawn a benchmark scenario"""
        if np.random.random() > self.embed_rate:
            return None

        agents = list(world.agents.keys())
        if len(agents) < self.min_agents_for_benchmark:
            return None

        # Select random benchmark type
        benchmark_creators = [
            lambda: self.tombench.create_false_belief_scenario(
                world, np.random.choice(agents, 2, replace=False).tolist()
            ),
            lambda: self.tombench.create_faux_pas_scenario(
                world, np.random.choice(agents, min(3, len(agents)), replace=False).tolist()
            ),
            lambda: self.tombench.create_second_order_scenario(
                world, np.random.choice(agents, 2, replace=False).tolist()
            ),
            lambda: self.sotopia.create_cooperative_scenario(
                world, np.random.choice(agents, min(4, len(agents)), replace=False).tolist()
            ),
            lambda: self.sotopia.create_mixed_motive_scenario(
                world, np.random.choice(agents, min(4, len(agents)), replace=False).tolist()
            ),
        ]

        try:
            scenario = np.random.choice(benchmark_creators)()
            scenario.active = True
            scenario.start_timestep = timestep
            self.active_scenarios.append(scenario)
            return scenario
        except ValueError:
            return None

    def evaluate_all(self, world: Any, agent_predictions: Dict[int, Dict]) -> List[BenchmarkResult]:
        """Evaluate all active scenarios"""
        results = []

        for scenario in self.active_scenarios:
            if not scenario.active:
                continue

            # Check if scenario should end
            elapsed = world.timestep - scenario.start_timestep
            if elapsed >= 20:  # Max 20 timesteps per scenario
                if scenario.benchmark_type in [
                    BenchmarkType.FALSE_BELIEF,
                    BenchmarkType.FAUX_PAS,
                    BenchmarkType.SECOND_ORDER,
                ]:
                    result = self.tombench.evaluate_scenario(
                        scenario, agent_predictions, world.get_state().__dict__
                    )
                else:
                    # For SOTOPIA, need to collect contributions
                    contributions = self._collect_contributions(scenario, world)
                    result = self.sotopia.evaluate_scenario(
                        scenario, contributions, world.get_state().__dict__
                    )

                scenario.active = False
                scenario.outcome = result
                results.append(result)
                self.completed_results.append(result)

        return results

    def _collect_contributions(self, scenario: BenchmarkScenario,
                              world: Any) -> Dict[int, List[float]]:
        """Collect agent contributions for SOTOPIA scenarios"""
        # In real implementation, would track this during scenario
        # For now, derive from world state
        contributions = {}
        for agent_id in scenario.participating_agents:
            if agent_id in world.agents:
                agent = world.agents[agent_id]
                # Use cooperation history as proxy for contributions
                contributions[agent_id] = [
                    10.0 if np.random.random() < agent.profile.layer6.cooperation_tendency / 100
                    else 5.0
                    for _ in range(scenario.setup.get('rounds', 5))
                ]
        return contributions

    def get_aggregate_scores(self) -> Dict[BenchmarkType, float]:
        """Get aggregate scores per benchmark type"""
        scores = defaultdict(list)
        for result in self.completed_results:
            scores[result.benchmark_type].append(result.agent_score)

        return {k: np.mean(v) for k, v in scores.items()}

    def get_summary(self) -> str:
        """Get summary of benchmark performance"""
        lines = ["=== Benchmark Summary ==="]

        aggregate = self.get_aggregate_scores()
        for btype, score in sorted(aggregate.items(), key=lambda x: x[0].value):
            baseline = self.tombench.HUMAN_BASELINES.get(
                btype, self.sotopia.HUMAN_BASELINES.get(btype, 0.7)
            )
            status = "✓" if score >= baseline * 0.8 else "✗"
            lines.append(f"{status} {btype.value}: {score:.2f} (baseline: {baseline:.2f})")

        lines.append("")
        lines.append(f"Total scenarios: {len(self.completed_results)}")
        lines.append(f"Active scenarios: {len([s for s in self.active_scenarios if s.active])}")

        return '\n'.join(lines)
