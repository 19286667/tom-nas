"""
Developmental Theory of Mind Benchmarks for ToM-NAS
Implements Sally-Anne tests from 0th to 5th order plus additional ToM tasks

Benchmark Categories:
1. Developmental ToM (Sally-Anne variants, 0-5th order)
2. Linguistic ToM (narrative tasks)
3. Ecological ToM (naturalistic scenarios)
4. Adversarial ToM (probing edge cases)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class ToMOrder(Enum):
    """Theory of Mind recursion orders."""
    ZEROTH = 0   # Direct observation
    FIRST = 1    # I think X (about state)
    SECOND = 2   # I think you think X
    THIRD = 3    # I think you think I think X
    FOURTH = 4   # I think you think I think you think X
    FIFTH = 5    # I think you think I think you think I think X


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    name: str
    order: int
    score: float
    passed: bool
    details: Dict[str, Any]
    expected: Any
    actual: Any


class SallyAnneScenario:
    """
    Encodes a Sally-Anne style false belief scenario.

    Basic scenario:
    - Sally puts object in Location A
    - Sally leaves
    - Anne moves object to Location B
    - Sally returns
    - Question: Where will Sally look?

    Higher orders add nested beliefs about beliefs.
    """

    def __init__(self, input_dim: int = 191, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device

        # Key dimension indices (aligned with ontology)
        self.OBJECT_LOC_A = 0
        self.OBJECT_LOC_B = 1
        self.SALLY_PRESENT = 2
        self.ANNE_PRESENT = 3
        self.SALLY_BELIEF_A = 4
        self.SALLY_BELIEF_B = 5
        self.ANNE_BELIEF_A = 6
        self.ANNE_BELIEF_B = 7
        self.SALLY_ABOUT_ANNE_A = 8
        self.SALLY_ABOUT_ANNE_B = 9
        self.ANNE_ABOUT_SALLY_A = 10
        self.ANNE_ABOUT_SALLY_B = 11

    def generate_scenario(self, order: int = 1) -> Tuple[torch.Tensor, Dict]:
        """
        Generate scenario encoding for specified ToM order.

        Returns:
            Tuple of (scenario_tensor, expected_answers)
        """
        seq_len = 6 + order  # More steps for higher orders

        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)
        expected = {}

        # T0: Initial - object in A, both present
        scenario[0, 0, self.OBJECT_LOC_A] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0
        scenario[0, 0, self.SALLY_BELIEF_A] = 1.0
        scenario[0, 0, self.ANNE_BELIEF_A] = 1.0

        # T1: Sally leaves
        scenario[0, 1] = scenario[0, 0].clone()
        scenario[0, 1, self.SALLY_PRESENT] = 0.0

        # T2: Anne moves object to B
        scenario[0, 2] = scenario[0, 1].clone()
        scenario[0, 2, self.OBJECT_LOC_A] = 0.0
        scenario[0, 2, self.OBJECT_LOC_B] = 1.0
        scenario[0, 2, self.ANNE_BELIEF_A] = 0.0
        scenario[0, 2, self.ANNE_BELIEF_B] = 1.0
        # Sally's belief unchanged (she wasn't there)

        # T3: Sally returns
        scenario[0, 3] = scenario[0, 2].clone()
        scenario[0, 3, self.SALLY_PRESENT] = 1.0

        # For higher orders, add belief-about-belief updates
        t = 4
        if order >= 2:
            # T4: Does Anne know Sally's belief is wrong?
            scenario[0, t] = scenario[0, t-1].clone()
            scenario[0, t, self.ANNE_ABOUT_SALLY_A] = 1.0  # Anne knows Sally thinks A
            t += 1

        if order >= 3:
            # T5: Does Sally know that Anne knows?
            scenario[0, t] = scenario[0, t-1].clone()
            scenario[0, t, self.SALLY_ABOUT_ANNE_A] = 0.5  # Sally uncertain about Anne's knowledge
            t += 1

        # Fill remaining timesteps
        while t < seq_len:
            scenario[0, t] = scenario[0, t-1].clone()
            t += 1

        # Expected answers by order
        expected['order_0'] = {'object_location': 'B'}  # Where IS the object

        expected['order_1'] = {
            'sally_will_look': 'A',  # False belief
            'where_sally_thinks': 'A'
        }

        expected['order_2'] = {
            'anne_knows_sally_wrong': True,
            'anne_prediction_of_sally': 'A'
        }

        expected['order_3'] = {
            'sally_knows_anne_knows': 'uncertain',  # Sally doesn't know Anne observed
        }

        expected['order_4'] = {
            'anne_thinks_sally_thinks_anne_knows': 'uncertain'
        }

        expected['order_5'] = {
            'sally_thinks_anne_thinks_sally_thinks': 'uncertain'
        }

        return scenario, expected


class DevelopmentalToMBenchmark(ABC):
    """Abstract base for developmental ToM benchmarks."""

    @abstractmethod
    def evaluate(self, agent: nn.Module) -> BenchmarkResult:
        pass


class SallyAnneBenchmark(DevelopmentalToMBenchmark):
    """
    Sally-Anne False Belief Test at specified order.

    Order 0: Can track object location
    Order 1: Can predict Sally's false belief
    Order 2: Can model Anne's knowledge of Sally's belief
    ...and so on
    """

    def __init__(self, order: int, input_dim: int = 191, device: str = 'cpu'):
        self.order = order
        self.input_dim = input_dim
        self.device = device
        self.scenario_gen = SallyAnneScenario(input_dim, device)

    def evaluate(self, agent: nn.Module) -> BenchmarkResult:
        """Evaluate agent on Sally-Anne test at this order."""
        scenario, expected = self.scenario_gen.generate_scenario(self.order)

        agent.eval()
        with torch.no_grad():
            output = agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(self.input_dim, device=self.device))

        if beliefs.dim() > 1:
            beliefs = beliefs[0]

        # Evaluate based on order
        if self.order == 0:
            # Just track object location
            loc_a = beliefs[self.scenario_gen.OBJECT_LOC_A].item()
            loc_b = beliefs[self.scenario_gen.OBJECT_LOC_B].item()
            predicted = 'B' if loc_b > loc_a else 'A'
            correct = predicted == 'B'
            score = loc_b / (loc_a + loc_b + 1e-8)

        elif self.order == 1:
            # Predict Sally's false belief
            sally_a = beliefs[self.scenario_gen.SALLY_BELIEF_A].item()
            sally_b = beliefs[self.scenario_gen.SALLY_BELIEF_B].item()
            predicted = 'A' if sally_a > sally_b else 'B'
            correct = predicted == 'A'  # Sally has false belief
            score = sally_a / (sally_a + sally_b + 1e-8)

        elif self.order == 2:
            # Anne's model of Sally's belief
            anne_thinks_sally_a = beliefs[self.scenario_gen.ANNE_ABOUT_SALLY_A].item() \
                if self.input_dim > self.scenario_gen.ANNE_ABOUT_SALLY_A else 0.5
            correct = anne_thinks_sally_a > 0.5
            score = anne_thinks_sally_a
            predicted = 'A' if correct else 'B'

        elif self.order == 3:
            # Order 3: Sally knows Anne knows Sally's belief
            # Requires: tracking that Anne observed Sally, so Anne updated her model
            # Agent must show Sally has SOME model of Anne's knowledge (not random)
            sally_about_anne = beliefs[self.scenario_gen.SALLY_ABOUT_ANNE_A].item() \
                if len(beliefs) > self.scenario_gen.SALLY_ABOUT_ANNE_A else 0.0
            sally_about_anne_b = beliefs[self.scenario_gen.SALLY_ABOUT_ANNE_B].item() \
                if len(beliefs) > self.scenario_gen.SALLY_ABOUT_ANNE_B else 0.0

            # Sally should track that Anne likely knows Sally thinks A
            # (because Anne was present when Sally originally saw object)
            # Score requires ACTIVE prediction, not just uncertainty
            belief_diff = abs(sally_about_anne - sally_about_anne_b)
            has_prediction = belief_diff > 0.2  # Must have non-random prediction

            score = sally_about_anne * 0.7 + (0.3 if has_prediction else 0.0)
            correct = sally_about_anne > 0.4 and has_prediction
            predicted = f"sally_thinks_anne_knows_A:{sally_about_anne:.2f}"

        elif self.order == 4:
            # Order 4: Anne thinks Sally thinks Anne knows
            # Even more nested - requires maintaining coherent recursive model
            # Must show structure, not random values
            beliefs_slice = beliefs[8:12] if len(beliefs) > 11 else beliefs[-4:]
            belief_mean = beliefs_slice.mean().item()
            belief_std = beliefs_slice.std().item() if len(beliefs_slice) > 1 else 0.0

            # Require STRUCTURED beliefs (std > 0.1) AND coherent prediction
            has_structure = belief_std > 0.1
            coherent = 0.2 < belief_mean < 0.8  # Not extreme values

            score = (belief_mean * 0.5 + belief_std * 0.5) if has_structure else belief_mean * 0.3
            correct = has_structure and coherent
            predicted = f"structured:{has_structure}, mean:{belief_mean:.2f}"

        else:  # Order 5
            # Order 5: Deepest nesting - requires full recursive chain
            # Must demonstrate coherent tracking across ALL levels
            level_scores = []
            for i in range(min(6, len(beliefs) // 2)):
                level_belief = beliefs[i * 2:(i + 1) * 2].mean().item() if len(beliefs) > (i + 1) * 2 else 0.5
                level_scores.append(level_belief)

            if level_scores:
                # Check for DECREASING confidence at higher levels (realistic)
                # Humans have less certainty about deeply nested beliefs
                is_decreasing = all(level_scores[i] >= level_scores[i+1] - 0.1
                                   for i in range(len(level_scores)-1))
                variance = np.var(level_scores) if len(level_scores) > 1 else 0.0
                has_structure = variance > 0.01

                score = (np.mean(level_scores) * 0.5 +
                        (0.3 if is_decreasing else 0.0) +
                        (0.2 if has_structure else 0.0))
                correct = is_decreasing and has_structure and np.mean(level_scores) > 0.3
            else:
                score = 0.0
                correct = False

            predicted = f"decreasing_confidence:{is_decreasing if level_scores else False}"

        return BenchmarkResult(
            name=f"Sally-Anne Order {self.order}",
            order=self.order,
            score=score,
            passed=correct,
            details={'beliefs': beliefs[:12].tolist() if len(beliefs) >= 12 else beliefs.tolist()},
            expected=expected.get(f'order_{self.order}', {}),
            actual=predicted
        )


class SmartiesTestBenchmark(DevelopmentalToMBenchmark):
    """
    Smarties/Unexpected Contents Test

    Child shown Smarties tube, asked what's inside.
    Says "Smarties" (typical answer).
    Shown tube contains pencils.
    Question: "What will [new person] think is inside?"

    Tests understanding of false belief from prior experience.
    """

    def __init__(self, input_dim: int = 191, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device
        self.APPEARANCE = 0
        self.ACTUAL_CONTENTS = 1
        self.OBSERVER_BELIEF = 2
        self.NEW_PERSON_BELIEF = 3

    def evaluate(self, agent: nn.Module) -> BenchmarkResult:
        # Scenario: Tube looks like candy, contains pencils
        scenario = torch.zeros(1, 5, self.input_dim, device=self.device)

        # T0: See tube (appears to contain candy)
        scenario[0, 0, self.APPEARANCE] = 1.0  # Candy appearance

        # T1: Belief based on appearance
        scenario[0, 1] = scenario[0, 0].clone()
        scenario[0, 1, self.OBSERVER_BELIEF] = 1.0  # Think it's candy

        # T2: Revealed - actually pencils
        scenario[0, 2] = scenario[0, 1].clone()
        scenario[0, 2, self.ACTUAL_CONTENTS] = 1.0  # Pencils revealed
        scenario[0, 2, self.APPEARANCE] = 1.0  # Still looks like candy tube

        # T3: New person enters (hasn't seen inside)
        scenario[0, 3] = scenario[0, 2].clone()
        # New person's belief should be based on appearance only

        # T4: Query - what does new person think?
        scenario[0, 4] = scenario[0, 3].clone()

        agent.eval()
        with torch.no_grad():
            output = agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(self.input_dim, device=self.device))

        if beliefs.dim() > 1:
            beliefs = beliefs[0]

        # New person should think candy (based on appearance)
        new_person_belief = beliefs[self.NEW_PERSON_BELIEF].item() \
            if len(beliefs) > self.NEW_PERSON_BELIEF else 0.5

        # Appearance-based prediction should be high
        appearance_influence = beliefs[self.APPEARANCE].item() \
            if len(beliefs) > self.APPEARANCE else 0.5

        score = (new_person_belief + appearance_influence) / 2
        passed = score > 0.5

        return BenchmarkResult(
            name="Smarties/Unexpected Contents Test",
            order=1,
            score=score,
            passed=passed,
            details={'new_person_belief': new_person_belief},
            expected={'new_person_thinks': 'candy'},
            actual='candy' if passed else 'pencils'
        )


class StrategicDeceptionBenchmark(DevelopmentalToMBenchmark):
    """
    Tests understanding of strategic deception.

    Scenario: Agent must predict if another agent will lie
    based on that agent's incentives.

    Order 2+ test: Requires modeling beliefs about beliefs.
    """

    def __init__(self, input_dim: int = 191, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device

    def evaluate(self, agent: nn.Module) -> BenchmarkResult:
        # Scenario: Agent A has incentive to lie to Agent B
        scenario = torch.zeros(1, 5, self.input_dim, device=self.device)

        # Encode: A knows truth, A benefits from B believing lie
        scenario[0, :, 0] = 1.0   # A knows truth
        scenario[0, :, 1] = 1.0   # A has incentive to lie
        scenario[0, :, 2] = 0.0   # B doesn't know truth
        scenario[0, :, 3] = 1.0   # A knows B doesn't know

        agent.eval()
        with torch.no_grad():
            output = agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(self.input_dim, device=self.device))

        if beliefs.dim() > 1:
            beliefs = beliefs[0]

        # Should predict A will attempt deception
        deception_prediction = beliefs[4].item() if len(beliefs) > 4 else 0.5
        # Should predict B might be deceived
        b_vulnerable = beliefs[5].item() if len(beliefs) > 5 else 0.5

        score = (deception_prediction + b_vulnerable) / 2
        passed = score > 0.4

        return BenchmarkResult(
            name="Strategic Deception Understanding",
            order=2,
            score=score,
            passed=passed,
            details={
                'deception_prediction': deception_prediction,
                'vulnerability_recognition': b_vulnerable
            },
            expected={'a_will_deceive': True, 'b_vulnerable': True},
            actual={'predicted_deception': deception_prediction > 0.5}
        )


class NarrativeToMBenchmark(DevelopmentalToMBenchmark):
    """
    Story-based ToM task.

    Presents a narrative with multiple characters and perspectives,
    tests comprehension of different mental states.
    """

    def __init__(self, input_dim: int = 191, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device

    def evaluate(self, agent: nn.Module) -> BenchmarkResult:
        # Simple story: Two characters with different knowledge
        # Character A sees event, Character B doesn't

        seq_len = 8
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # Story encoding (simplified)
        # T0-1: Setup
        scenario[0, 0, 0] = 1.0  # Character A present
        scenario[0, 0, 1] = 1.0  # Character B present

        # T2-3: Event happens, A sees, B doesn't
        scenario[0, 2, 2] = 1.0  # Event occurs
        scenario[0, 2, 3] = 1.0  # A witnesses
        scenario[0, 2, 4] = 0.0  # B doesn't witness

        # T4-5: A and B interact
        scenario[0, 4, 5] = 1.0  # Interaction

        # T6-7: Question about B's knowledge
        scenario[0, 6, 6] = 1.0  # Query marker

        # Fill forward
        for t in range(1, seq_len):
            if torch.sum(scenario[0, t]) == 0:
                scenario[0, t] = scenario[0, t-1].clone()

        agent.eval()
        with torch.no_grad():
            output = agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(self.input_dim, device=self.device))

        if beliefs.dim() > 1:
            beliefs = beliefs[0]

        # Check if agent correctly tracks that B doesn't know about event
        b_knowledge = beliefs[4].item() if len(beliefs) > 4 else 0.5
        a_knowledge = beliefs[3].item() if len(beliefs) > 3 else 0.5

        # Correct: A knows (high), B doesn't know (low)
        correct_a = a_knowledge > 0.5
        correct_b = b_knowledge < 0.5

        score = (a_knowledge + (1 - b_knowledge)) / 2
        passed = correct_a and correct_b

        return BenchmarkResult(
            name="Narrative ToM (Story Comprehension)",
            order=1,
            score=score,
            passed=passed,
            details={
                'a_knowledge_tracked': a_knowledge,
                'b_ignorance_tracked': 1 - b_knowledge
            },
            expected={'a_knows': True, 'b_knows': False},
            actual={'a_knows': correct_a, 'b_knows': not correct_b}
        )


def validate_tom_hierarchy(scores: Dict[int, float], agent_id: str = "") -> Tuple[bool, List[str]]:
    """
    Validate that ToM scores follow expected hierarchy.

    Lower orders should generally be easier than higher orders.
    Violations suggest tests are broken or agent is pattern-matching.

    Args:
        scores: Dict mapping order (0-5) to score
        agent_id: Optional identifier for error messages

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    violations = []

    for order in range(1, min(6, len(scores))):
        prev_score = scores.get(order - 1, 0)
        curr_score = scores.get(order, 0)

        # Higher order significantly easier than lower = violation
        # Allow small margin (0.15) for noise
        if curr_score > prev_score + 0.15:
            violations.append(
                f"Order {order} ({curr_score:.3f}) > Order {order-1} ({prev_score:.3f})"
            )

    is_valid = len(violations) == 0

    if not is_valid and agent_id:
        print(f"\n[ToM HIERARCHY WARNING for {agent_id}]")
        for v in violations:
            print(f"  - {v}")
        print("  This may indicate tests are misconfigured or agent is exploiting test structure.\n")

    return is_valid, violations


class ToMBenchmarkSuite:
    """
    Complete Theory of Mind Benchmark Suite.

    Includes:
    - Sally-Anne tests at orders 0-5
    - Smarties test
    - Strategic deception test
    - Narrative comprehension

    Now includes hierarchy validation to catch inverted/broken tests.
    """

    def __init__(self, input_dim: int = 191, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device

        # Build benchmark suite
        self.benchmarks = {}

        # Sally-Anne at all orders
        for order in range(6):
            self.benchmarks[f'sally_anne_order_{order}'] = SallyAnneBenchmark(
                order=order, input_dim=input_dim, device=device
            )

        # Additional benchmarks
        self.benchmarks['smarties'] = SmartiesTestBenchmark(input_dim, device)
        self.benchmarks['strategic_deception'] = StrategicDeceptionBenchmark(input_dim, device)
        self.benchmarks['narrative'] = NarrativeToMBenchmark(input_dim, device)

    def run_full_evaluation(self, agent: nn.Module, agent_id: str = "") -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        results = {}
        scores_by_order = {i: [] for i in range(6)}

        for name, benchmark in self.benchmarks.items():
            try:
                result = benchmark.evaluate(agent)
                results[name] = {
                    'score': result.score,
                    'passed': result.passed,
                    'order': result.order,
                    'expected': result.expected,
                    'actual': result.actual
                }
                scores_by_order[result.order].append(result.score)
            except Exception as e:
                results[name] = {
                    'score': 0.0,
                    'passed': False,
                    'error': str(e)
                }

        # Calculate aggregate metrics
        all_scores = [r['score'] for r in results.values() if 'score' in r]
        sally_anne_scores = [results[f'sally_anne_order_{i}']['score']
                           for i in range(6) if f'sally_anne_order_{i}' in results]

        # Max ToM order achieved (highest order passed WITH valid hierarchy)
        max_order_achieved = -1
        for i in range(6):
            key = f'sally_anne_order_{i}'
            if key in results and results[key]['passed']:
                max_order_achieved = i

        # HIERARCHY VALIDATION - catch inverted scores
        sally_anne_dict = {i: results[f'sally_anne_order_{i}']['score']
                         for i in range(6) if f'sally_anne_order_{i}' in results}
        hierarchy_valid, violations = validate_tom_hierarchy(sally_anne_dict, agent_id)

        return {
            'benchmark_results': results,
            'overall_score': np.mean(all_scores) if all_scores else 0.0,
            'sally_anne_progression': sally_anne_scores,
            'max_tom_order': max_order_achieved,
            'scores_by_order': {k: np.mean(v) if v else 0.0
                               for k, v in scores_by_order.items()},
            'num_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'num_total': len(results),
            'hierarchy_valid': hierarchy_valid,
            'hierarchy_violations': violations
        }

    def run_developmental_progression(self, agent: nn.Module) -> Dict[str, Any]:
        """
        Evaluate developmental progression through ToM orders.

        Returns which orders the agent passes and the progression curve.
        """
        progression = []
        passed_orders = []

        for order in range(6):
            benchmark = self.benchmarks[f'sally_anne_order_{order}']
            result = benchmark.evaluate(agent)
            progression.append({
                'order': order,
                'score': result.score,
                'passed': result.passed
            })
            if result.passed:
                passed_orders.append(order)

        return {
            'progression': progression,
            'passed_orders': passed_orders,
            'max_order_passed': max(passed_orders) if passed_orders else -1,
            'developmental_score': len(passed_orders) / 6.0
        }

    def get_fitness_contribution(self, results: Dict) -> float:
        """
        Calculate fitness contribution from benchmark results.

        Higher ToM order = higher fitness contribution.
        """
        base_score = results.get('overall_score', 0.0)
        max_order = results.get('max_tom_order', -1)

        # Bonus for higher orders (exponentially harder)
        order_bonus = (max_order + 1) * 0.1 if max_order >= 0 else 0.0

        return min(1.0, base_score + order_bonus)


def create_tom_benchmark_context(num_agents: int = 5,
                                  input_dim: int = 191) -> Dict:
    """Create context dict for ToM benchmarks."""
    return {
        'input_dim': input_dim,
        'num_agents': num_agents,
        'available_benchmarks': [
            'sally_anne_order_0',
            'sally_anne_order_1',
            'sally_anne_order_2',
            'sally_anne_order_3',
            'sally_anne_order_4',
            'sally_anne_order_5',
            'smarties',
            'strategic_deception',
            'narrative'
        ]
    }
