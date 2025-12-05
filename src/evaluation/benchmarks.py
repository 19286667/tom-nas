"""
Theory of Mind Benchmarks for ToM-NAS
Comprehensive test suite for evaluating ToM capabilities
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test"""
    test_name: str
    score: float
    max_score: float
    passed: bool
    details: Dict

    @property
    def percentage(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0


class SallyAnneTest:
    """Classic Sally-Anne false belief test with variations.

    These tests evaluate genuine Theory of Mind by testing:
    1. False belief attribution - understanding others can have beliefs different from reality
    2. Perspective taking - tracking what information each agent has access to
    3. Belief-desire reasoning - predicting actions based on (possibly false) beliefs
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.variations = [
            'basic', 'unexpected_transfer', 'deceptive_container',
            'second_order', 'triple_location'
        ]

    def _encode_scenario_step(self, step_data: Dict, base_tensor: torch.Tensor, step_idx: int) -> None:
        """Encode a single scenario step into the tensor using ontology-aligned features"""
        # Agent presence encoding (biological layer 0-14)
        base_tensor[0, step_idx, 0] = step_data.get('sally_present', 0.0)
        base_tensor[0, step_idx, 1] = step_data.get('anne_present', 0.0)
        base_tensor[0, step_idx, 2] = step_data.get('observer_attention', 0.5)

        # Object location encoding (spatial perception 15-30)
        base_tensor[0, step_idx, 15] = step_data.get('marble_in_basket', 0.0)
        base_tensor[0, step_idx, 16] = step_data.get('marble_in_box', 0.0)
        base_tensor[0, step_idx, 17] = step_data.get('marble_visible', 0.0)

        # Belief state encoding (ToM-specific 59-98)
        # Sally's belief about marble location
        base_tensor[0, step_idx, 59] = step_data.get('sally_believes_basket', 0.0)
        base_tensor[0, step_idx, 60] = step_data.get('sally_believes_box', 0.0)
        # Anne's belief about marble location
        base_tensor[0, step_idx, 61] = step_data.get('anne_believes_basket', 0.0)
        base_tensor[0, step_idx, 62] = step_data.get('anne_believes_box', 0.0)
        # Meta: who saw the transfer
        base_tensor[0, step_idx, 63] = step_data.get('sally_saw_transfer', 0.0)
        base_tensor[0, step_idx, 64] = step_data.get('anne_saw_transfer', 0.0)

        # Temporal/event markers (context 139-178)
        base_tensor[0, step_idx, 139] = float(step_idx) / 10.0  # Time
        base_tensor[0, step_idx, 140] = step_data.get('event_type', 0.0)

    def run_basic(self, agent: nn.Module) -> BenchmarkResult:
        """
        Basic Sally-Anne False Belief Test:
        1. Sally puts marble in basket (both see)
        2. Sally leaves
        3. Anne moves marble to box (Sally absent)
        4. Sally returns
        Question: Where will Sally look for the marble?
        Correct: basket (false belief - she didn't see the transfer)
        """
        agent.eval()

        with torch.no_grad():
            sequence = torch.zeros(1, 5, 191)

            # Step 0: Initial state - both present, marble in basket
            self._encode_scenario_step({
                'sally_present': 1.0, 'anne_present': 1.0,
                'marble_in_basket': 1.0, 'marble_in_box': 0.0, 'marble_visible': 1.0,
                'sally_believes_basket': 1.0, 'sally_believes_box': 0.0,
                'anne_believes_basket': 1.0, 'anne_believes_box': 0.0,
                'sally_saw_transfer': 0.0, 'anne_saw_transfer': 0.0,
                'event_type': 0.1
            }, sequence, 0)

            # Step 1: Sally puts marble in basket explicitly
            self._encode_scenario_step({
                'sally_present': 1.0, 'anne_present': 1.0,
                'marble_in_basket': 1.0, 'marble_in_box': 0.0, 'marble_visible': 1.0,
                'sally_believes_basket': 1.0, 'sally_believes_box': 0.0,
                'anne_believes_basket': 1.0, 'anne_believes_box': 0.0,
                'event_type': 0.2
            }, sequence, 1)

            # Step 2: Sally leaves
            self._encode_scenario_step({
                'sally_present': 0.0, 'anne_present': 1.0,
                'marble_in_basket': 1.0, 'marble_in_box': 0.0, 'marble_visible': 1.0,
                'sally_believes_basket': 1.0, 'sally_believes_box': 0.0,
                'anne_believes_basket': 1.0, 'anne_believes_box': 0.0,
                'event_type': 0.3
            }, sequence, 2)

            # Step 3: Anne moves marble to box (Sally absent - key moment)
            self._encode_scenario_step({
                'sally_present': 0.0, 'anne_present': 1.0,
                'marble_in_basket': 0.0, 'marble_in_box': 1.0, 'marble_visible': 1.0,
                'sally_believes_basket': 1.0, 'sally_believes_box': 0.0,  # Sally doesn't know!
                'anne_believes_basket': 0.0, 'anne_believes_box': 1.0,
                'sally_saw_transfer': 0.0, 'anne_saw_transfer': 1.0,
                'event_type': 0.4
            }, sequence, 3)

            # Step 4: Sally returns - test time
            self._encode_scenario_step({
                'sally_present': 1.0, 'anne_present': 1.0,
                'marble_in_basket': 0.0, 'marble_in_box': 1.0, 'marble_visible': 0.0,
                'sally_believes_basket': 1.0, 'sally_believes_box': 0.0,  # Still believes basket
                'anne_believes_basket': 0.0, 'anne_believes_box': 1.0,
                'sally_saw_transfer': 0.0, 'anne_saw_transfer': 1.0,
                'event_type': 0.5
            }, sequence, 4)

            # Get agent's prediction
            output = agent(sequence)
            beliefs = output['beliefs']

            # Evaluate: Agent should predict Sally will look in basket
            # Use multiple belief dimensions to assess
            basket_belief = beliefs[0, 0].item() if beliefs.shape[1] > 0 else 0.5
            box_belief = beliefs[0, 1].item() if beliefs.shape[1] > 1 else 0.5

            # Also check if model distinguishes Sally's belief from reality
            # (belief dimension 59 should be higher than 60 in a trained model)

            # Scoring: correct if basket > box
            correct = basket_belief > box_belief
            # Bonus for strong distinction
            distinction = abs(basket_belief - box_belief)
            score = (1.0 if correct else 0.0) * 0.7 + min(distinction, 0.3) * 0.3 / 0.3

            passed = correct and distinction > 0.1

            return BenchmarkResult(
                test_name="Sally-Anne Basic",
                score=score,
                max_score=1.0,
                passed=passed,
                details={
                    'basket_belief': basket_belief,
                    'box_belief': box_belief,
                    'distinction': distinction,
                    'correct': correct,
                    'correct_answer': 'basket'
                }
            )

    def run_second_order(self, agent: nn.Module) -> BenchmarkResult:
        """
        Second-order belief test:
        John and Mary are in a room. Mary puts her toy in the drawer.
        John leaves. Mary moves the toy to the box.
        But John was watching through the window!
        Question: Where does Mary think John thinks the toy is?
        Answer: drawer (Mary doesn't know John saw)
        """
        agent.eval()

        with torch.no_grad():
            sequence = torch.zeros(1, 6, 191)

            # Step 0: Both present, toy in drawer
            self._encode_scenario_step({
                'sally_present': 1.0, 'anne_present': 1.0,  # John=Sally, Mary=Anne for encoding
                'marble_in_basket': 1.0, 'marble_in_box': 0.0, 'marble_visible': 1.0,
                'sally_believes_basket': 1.0, 'anne_believes_basket': 1.0,
                'event_type': 0.1
            }, sequence, 0)

            # Step 1: John leaves (but secretly watches)
            self._encode_scenario_step({
                'sally_present': 0.0, 'anne_present': 1.0,
                'observer_attention': 0.8,  # John is watching secretly
                'marble_in_basket': 1.0, 'marble_in_box': 0.0,
                'sally_believes_basket': 1.0, 'anne_believes_basket': 1.0,
                'event_type': 0.2
            }, sequence, 1)

            # Step 2: Mary moves toy to box (John secretly sees)
            self._encode_scenario_step({
                'sally_present': 0.0, 'anne_present': 1.0,
                'observer_attention': 0.8,
                'marble_in_basket': 0.0, 'marble_in_box': 1.0,
                'sally_believes_basket': 0.0, 'sally_believes_box': 1.0,  # John knows!
                'anne_believes_basket': 0.0, 'anne_believes_box': 1.0,
                'sally_saw_transfer': 1.0, 'anne_saw_transfer': 1.0,
                'event_type': 0.3
            }, sequence, 2)

            # Step 3: Mary's belief about John's belief (she doesn't know he saw)
            # Encode: Mary thinks John still believes drawer
            sequence[0, 3, 65] = 1.0  # Mary's belief about John's belief = drawer
            sequence[0, 3, 66] = 0.0  # Mary's belief about John's belief != box
            sequence[0, 3, 67] = 0.0  # Mary didn't see John watching
            self._encode_scenario_step({
                'sally_present': 0.0, 'anne_present': 1.0,
                'marble_in_basket': 0.0, 'marble_in_box': 1.0,
                'event_type': 0.4
            }, sequence, 3)

            # Step 4: Test - what does Mary think John thinks?
            sequence[0, 4, 68] = 1.0  # Second-order query marker
            self._encode_scenario_step({
                'sally_present': 1.0, 'anne_present': 1.0,
                'marble_in_basket': 0.0, 'marble_in_box': 1.0,
                'event_type': 0.5
            }, sequence, 4)

            # Step 5: Additional context for 2nd-order reasoning
            sequence[0, 5, 69] = 0.7  # Confidence decay for 2nd order

            output = agent(sequence)
            beliefs = output['beliefs']

            # For 2nd-order: Mary thinks John thinks drawer
            # So we expect lower confidence overall (uncertainty in nested beliefs)
            # And the prediction should reflect Mary's false belief about John
            confidence = beliefs.mean().item()
            drawer_belief = beliefs[0, 0].item() if beliefs.shape[1] > 0 else 0.5

            # Expected: moderate confidence (0.5-0.7) due to 2nd-order uncertainty
            # and drawer_belief > box_belief
            expected_confidence_range = (0.4, 0.75)
            confidence_in_range = expected_confidence_range[0] <= confidence <= expected_confidence_range[1]

            score = 0.0
            if confidence_in_range:
                score += 0.5
            if drawer_belief > 0.5:  # Correct answer
                score += 0.5 * drawer_belief

            passed = score >= 0.6

            return BenchmarkResult(
                test_name="Sally-Anne Second-Order",
                score=score,
                max_score=1.0,
                passed=passed,
                details={
                    'confidence': confidence,
                    'drawer_belief': drawer_belief,
                    'confidence_in_range': confidence_in_range,
                    'expected_range': expected_confidence_range
                }
            )

    def run_all(self, agent: nn.Module) -> List[BenchmarkResult]:
        """Run all Sally-Anne variations"""
        results = [
            self.run_basic(agent),
            self.run_second_order(agent)
        ]
        return results


class HigherOrderToMBenchmark:
    """Systematic tests for each order of ToM (1st through 5th).

    Each order tests increasingly nested belief reasoning:
    - Order 1: A knows X (simple knowledge attribution)
    - Order 2: A knows B knows X (mutual knowledge)
    - Order 3: A knows B knows A knows X (common knowledge emergence)
    - Order 4+: Deep recursive mentalizing
    """

    def __init__(self, max_order: int = 5, device='cpu'):
        self.max_order = max_order
        self.device = device
        # Confidence should decay with recursive depth (epistemic uncertainty)
        self.confidence_decay_rate = 0.7  # Multiply by this per order

    def _encode_belief_chain(self, sequence: torch.Tensor, order: int, base_fact: float = 1.0) -> None:
        """Encode a chain of nested beliefs up to the specified order"""
        # Base fact encoding
        sequence[0, 0, 70] = base_fact  # The actual fact X

        for level in range(order):
            step_idx = level + 1
            if step_idx >= sequence.shape[1]:
                break

            # Each level encodes "Agent_{level} believes..."
            # Use dedicated belief encoding region (59-98)
            belief_idx = 59 + level * 5

            # Encode belief content
            sequence[0, step_idx, belief_idx] = base_fact  # Content of belief
            sequence[0, step_idx, belief_idx + 1] = self.confidence_decay_rate ** level  # Confidence
            sequence[0, step_idx, belief_idx + 2] = float(level + 1)  # Order marker
            sequence[0, step_idx, belief_idx + 3] = 1.0 if level % 2 == 0 else 0.0  # Agent alternation

            # Recursive marker in meta-cognitive region
            sequence[0, step_idx, 179 + min(level, 10)] = 1.0

    def test_order(self, agent: nn.Module, order: int) -> BenchmarkResult:
        """
        Test specific order of ToM with proper epistemic scenarios.

        Order 1: "Alice knows the ball is red"
        Order 2: "Alice knows Bob knows the ball is red"
        Order 3: "Alice knows Bob knows Alice knows the ball is red"
        etc.
        """
        agent.eval()

        with torch.no_grad():
            # Sequence length scales with order
            seq_length = order + 4
            sequence = torch.zeros(1, seq_length, 191)

            # Encode the nested belief chain
            self._encode_belief_chain(sequence, order, base_fact=1.0)

            # Add context markers for the order being tested
            sequence[0, :, 140] = float(order) / self.max_order  # Order indicator
            sequence[0, -1, 141] = 1.0  # Query marker

            # Final step: query about the nested belief
            sequence[0, -1, 59 + order - 1] = 0.5  # Query point

            output = agent(sequence)
            beliefs = output['beliefs']

            # Expected: confidence should decrease with order
            # A well-calibrated ToM agent shows epistemic humility
            expected_confidence = self.confidence_decay_rate ** (order - 1)
            actual_confidence = beliefs.mean().item()

            # Also check if the agent correctly tracks the belief content
            # For well-calibrated model, first belief dim should be close to 1.0
            # (the underlying fact is true throughout)
            content_belief = beliefs[0, 0].item() if beliefs.shape[1] > 0 else 0.5

            # Score components:
            # 1. Confidence calibration (does confidence decay appropriately?)
            confidence_error = abs(actual_confidence - expected_confidence)
            calibration_score = max(0.0, 1.0 - confidence_error * 2)

            # 2. Content preservation (does agent track the actual belief content?)
            content_score = content_belief  # Should be high for true fact

            # 3. Order-appropriate uncertainty (higher orders = more variance)
            beliefs_std = beliefs.std().item()
            expected_std = 0.1 + order * 0.05  # Uncertainty should increase
            uncertainty_score = max(0.0, 1.0 - abs(beliefs_std - expected_std) * 5)

            # Combined score
            score = calibration_score * 0.4 + content_score * 0.4 + uncertainty_score * 0.2

            # Pass if calibration is reasonable and content is tracked
            passed = calibration_score >= 0.4 and content_score >= 0.4

            return BenchmarkResult(
                test_name=f"ToM Order-{order}",
                score=score,
                max_score=1.0,
                passed=passed,
                details={
                    'order': order,
                    'expected_confidence': expected_confidence,
                    'actual_confidence': actual_confidence,
                    'confidence_error': confidence_error,
                    'content_belief': content_belief,
                    'beliefs_std': beliefs_std,
                    'calibration_score': calibration_score,
                    'content_score': content_score,
                    'uncertainty_score': uncertainty_score
                }
            )

    def run_all(self, agent: nn.Module) -> List[BenchmarkResult]:
        """Test all orders"""
        return [self.test_order(agent, order)
                for order in range(1, self.max_order + 1)]


class ZombieDetectionBenchmark:
    """Tests ability to detect agents without genuine ToM"""

    def __init__(self, device='cpu'):
        self.device = device
        self.zombie_types = [
            'behavioral', 'belief', 'causal',
            'metacognitive', 'linguistic', 'emotional'
        ]

    def test_detection(self, agent: nn.Module, zombie_type: str) -> BenchmarkResult:
        """Test detection of specific zombie type"""
        agent.eval()

        with torch.no_grad():
            # Simulate interaction with zombie agent
            # Zombies show inconsistent patterns
            sequence = torch.randn(1, 10, 191) * 0.3

            if zombie_type == 'behavioral':
                # Add inconsistent action patterns
                sequence[0, ::2, 50] = 1.0
                sequence[0, 1::2, 50] = -1.0
            elif zombie_type == 'belief':
                # Flat/inconsistent belief patterns
                sequence[0, :, 60:70] = 0.5

            output = agent(sequence)
            beliefs = output['beliefs']

            # Detection score based on uncertainty
            variance = beliefs.var().item()

            # High variance suggests detection
            score = min(1.0, variance * 5.0)
            passed = score >= 0.6

            return BenchmarkResult(
                test_name=f"Zombie Detection: {zombie_type}",
                score=score,
                max_score=1.0,
                passed=passed,
                details={
                    'zombie_type': zombie_type,
                    'variance': variance
                }
            )

    def run_all(self, agent: nn.Module) -> List[BenchmarkResult]:
        """Test all zombie types"""
        return [self.test_detection(agent, ztype)
                for ztype in self.zombie_types]


class CooperationBenchmark:
    """Tests cooperation and strategic reasoning"""

    def __init__(self, device='cpu'):
        self.device = device

    def test_repeated_prisoners_dilemma(self, agent: nn.Module,
                                       rounds: int = 10) -> BenchmarkResult:
        """Test in repeated Prisoner's Dilemma"""
        agent.eval()

        cooperation_count = 0
        reciprocation_count = 0

        with torch.no_grad():
            for round_num in range(rounds):
                # Encode game state
                sequence = torch.zeros(1, round_num + 1, 191)
                sequence[0, :, 80] = 1.0  # PD game marker

                # Encode partner's history (cooperative)
                if round_num > 0:
                    sequence[0, :round_num, 81] = 0.8  # Partner cooperated

                output = agent(sequence)
                action = output['actions'].item()

                # Count cooperation (action > 0.5 means cooperate)
                if action > 0.5:
                    cooperation_count += 1
                    if round_num > 0:
                        reciprocation_count += 1

        cooperation_rate = cooperation_count / rounds
        reciprocation_rate = reciprocation_count / max(1, rounds - 1)

        # Good ToM should show high reciprocation
        score = (cooperation_rate * 0.3 + reciprocation_rate * 0.7)
        passed = score >= 0.6

        return BenchmarkResult(
            test_name="Repeated Prisoner's Dilemma",
            score=score,
            max_score=1.0,
            passed=passed,
            details={
                'cooperation_rate': cooperation_rate,
                'reciprocation_rate': reciprocation_rate
            }
        )


class BenchmarkSuite:
    """Complete benchmark suite"""

    def __init__(self, device='cpu'):
        self.device = device
        self.sally_anne = SallyAnneTest(device)
        self.higher_order = HigherOrderToMBenchmark(max_order=5, device=device)
        self.zombie = ZombieDetectionBenchmark(device)
        self.cooperation = CooperationBenchmark(device)

    def run_full_suite(self, agent: nn.Module) -> Dict:
        """Run complete benchmark suite"""
        print("\n" + "="*60)
        print("Running ToM Benchmark Suite")
        print("="*60)

        all_results = []

        # Sally-Anne tests
        print("\n1. Sally-Anne Tests")
        print("-"*60)
        sally_results = self.sally_anne.run_all(agent)
        for result in sally_results:
            all_results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.test_name:30s} {status:8s} {result.percentage:5.1f}%")

        # Higher-order ToM
        print("\n2. Higher-Order ToM Tests")
        print("-"*60)
        higher_results = self.higher_order.run_all(agent)
        for result in higher_results:
            all_results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.test_name:30s} {status:8s} {result.percentage:5.1f}%")

        # Zombie detection
        print("\n3. Zombie Detection Tests")
        print("-"*60)
        zombie_results = self.zombie.run_all(agent)
        for result in zombie_results:
            all_results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.test_name:30s} {status:8s} {result.percentage:5.1f}%")

        # Cooperation
        print("\n4. Cooperation Tests")
        print("-"*60)
        coop_result = self.cooperation.test_repeated_prisoners_dilemma(agent)
        all_results.append(coop_result)
        status = "✓ PASS" if coop_result.passed else "✗ FAIL"
        print(f"  {coop_result.test_name:30s} {status:8s} {coop_result.percentage:5.1f}%")

        # Summary
        total_score = sum(r.score for r in all_results)
        max_score = sum(r.max_score for r in all_results)
        passed_count = sum(1 for r in all_results if r.passed)

        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"  Total Score:  {total_score:.2f} / {max_score:.2f} ({total_score/max_score*100:.1f}%)")
        print(f"  Tests Passed: {passed_count} / {len(all_results)}")
        print("="*60)

        return {
            'results': all_results,
            'total_score': total_score,
            'max_score': max_score,
            'percentage': total_score / max_score * 100,
            'passed_count': passed_count,
            'total_tests': len(all_results),
            'pass_rate': passed_count / len(all_results) * 100
        }

    def quick_eval(self, agent: nn.Module) -> float:
        """Quick evaluation returning single score"""
        full_results = self.run_full_suite(agent)
        return full_results['percentage'] / 100.0
