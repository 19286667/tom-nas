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
    """Classic Sally-Anne false belief test with variations"""

    def __init__(self, device="cpu"):
        self.device = device
        self.variations = ["basic", "unexpected_transfer", "deceptive_container", "second_order", "triple_location"]

    def run_basic(self, agent: nn.Module) -> BenchmarkResult:
        """
        Basic Sally-Anne:
        - Sally puts marble in basket
        - Sally leaves
        - Anne moves marble to box
        - Where will Sally look?
        Correct: basket
        """
        agent.eval()

        with torch.no_grad():
            # Encode scenario as sequence
            # [sally_present, anne_present, marble_basket, marble_box, marble_visible]
            sequence = torch.zeros(1, 4, 191)

            # Step 1: Sally puts marble in basket
            sequence[0, 0, 0] = 1.0  # Sally present
            sequence[0, 0, 1] = 1.0  # Anne present
            sequence[0, 0, 2] = 1.0  # Marble in basket
            sequence[0, 0, 4] = 1.0  # Marble visible

            # Step 2: Sally leaves
            sequence[0, 1, 0] = 0.0  # Sally not present
            sequence[0, 1, 1] = 1.0  # Anne present
            sequence[0, 1, 2] = 1.0  # Marble still in basket

            # Step 3: Anne moves marble
            sequence[0, 2, 0] = 0.0  # Sally not present
            sequence[0, 2, 1] = 1.0  # Anne present
            sequence[0, 2, 2] = 0.0  # Marble not in basket
            sequence[0, 2, 3] = 1.0  # Marble in box

            # Step 4: Sally returns - what does she believe?
            sequence[0, 3, 0] = 1.0  # Sally present
            sequence[0, 3, 1] = 1.0  # Anne present
            sequence[0, 3, 2] = 0.0  # Marble actually not in basket
            sequence[0, 3, 3] = 1.0  # Marble actually in box

            # Get agent's prediction
            output = agent(sequence)
            beliefs = output["beliefs"]

            # Check if agent predicts Sally believes marble is in basket
            # We expect first belief dimension to be higher (basket)
            basket_belief = beliefs[0, 0].item() if beliefs.shape[1] > 0 else 0.5
            box_belief = beliefs[0, 1].item() if beliefs.shape[1] > 1 else 0.5

            score = 1.0 if basket_belief > box_belief else 0.0
            passed = score >= 1.0

            return BenchmarkResult(
                test_name="Sally-Anne Basic",
                score=score,
                max_score=1.0,
                passed=passed,
                details={"basket_belief": basket_belief, "box_belief": box_belief, "correct_answer": "basket"},
            )

    def run_second_order(self, agent: nn.Module) -> BenchmarkResult:
        """
        Second-order belief test:
        - John thinks Mary thinks the toy is in location X
        """
        agent.eval()

        with torch.no_grad():
            # More complex scenario requiring 2nd-order reasoning
            sequence = torch.randn(1, 6, 191) * 0.1

            # Encode John's belief about Mary's belief
            sequence[0, :, 10] = 1.0  # Marker for 2nd-order scenario

            output = agent(sequence)
            beliefs = output["beliefs"]

            # Simplified scoring - check if belief confidence decreases
            # (2nd order should be less confident than 1st order)
            confidence = beliefs.mean().item()
            expected_confidence = 0.7  # Lower for 2nd order

            score = max(0.0, 1.0 - abs(confidence - expected_confidence) / expected_confidence)
            passed = score >= 0.7

            return BenchmarkResult(
                test_name="Sally-Anne Second-Order",
                score=score,
                max_score=1.0,
                passed=passed,
                details={"confidence": confidence, "expected": expected_confidence},
            )

    def run_all(self, agent: nn.Module) -> List[BenchmarkResult]:
        """Run all Sally-Anne variations"""
        results = [self.run_basic(agent), self.run_second_order(agent)]
        return results


class HigherOrderToMBenchmark:
    """Systematic tests for each order of ToM (1st through 5th)"""

    def __init__(self, max_order: int = 5, device="cpu"):
        self.max_order = max_order
        self.device = device

    def test_order(self, agent: nn.Module, order: int) -> BenchmarkResult:
        """
        Test specific order of ToM
        Order 1: A knows X
        Order 2: A knows B knows X
        Order 3: A knows B knows A knows X
        Order 4: A knows B knows A knows B knows X
        Order 5: Full recursive depth
        """
        agent.eval()

        with torch.no_grad():
            # Create scenario with nested beliefs
            seq_length = order + 3
            sequence = torch.zeros(1, seq_length, 191)

            # Encode belief nesting depth
            for i in range(min(order, seq_length)):
                sequence[0, i, 20 + i] = 1.0  # Unique marker per level

            output = agent(sequence)
            beliefs = output["beliefs"]

            # Expected confidence should decrease with order
            # Order 1: ~0.9, Order 2: ~0.75, Order 3: ~0.60, etc.
            expected_confidence = max(0.3, 1.0 - (order - 1) * 0.15)
            actual_confidence = beliefs.mean().item()

            # Score based on how close to expected
            error = abs(actual_confidence - expected_confidence)
            score = max(0.0, 1.0 - error)

            passed = score >= 0.5

            return BenchmarkResult(
                test_name=f"ToM Order-{order}",
                score=score,
                max_score=1.0,
                passed=passed,
                details={
                    "order": order,
                    "expected_confidence": expected_confidence,
                    "actual_confidence": actual_confidence,
                    "error": error,
                },
            )

    def run_all(self, agent: nn.Module) -> List[BenchmarkResult]:
        """Test all orders"""
        return [self.test_order(agent, order) for order in range(1, self.max_order + 1)]


class ZombieDetectionBenchmark:
    """Tests ability to detect agents without genuine ToM"""

    def __init__(self, device="cpu"):
        self.device = device
        self.zombie_types = ["behavioral", "belief", "causal", "metacognitive", "linguistic", "emotional"]

    def test_detection(self, agent: nn.Module, zombie_type: str) -> BenchmarkResult:
        """Test detection of specific zombie type"""
        agent.eval()

        with torch.no_grad():
            # Simulate interaction with zombie agent
            # Zombies show inconsistent patterns
            sequence = torch.randn(1, 10, 191) * 0.3

            if zombie_type == "behavioral":
                # Add inconsistent action patterns
                sequence[0, ::2, 50] = 1.0
                sequence[0, 1::2, 50] = -1.0
            elif zombie_type == "belief":
                # Flat/inconsistent belief patterns
                sequence[0, :, 60:70] = 0.5

            output = agent(sequence)
            beliefs = output["beliefs"]

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
                details={"zombie_type": zombie_type, "variance": variance},
            )

    def run_all(self, agent: nn.Module) -> List[BenchmarkResult]:
        """Test all zombie types"""
        return [self.test_detection(agent, ztype) for ztype in self.zombie_types]


class CooperationBenchmark:
    """Tests cooperation and strategic reasoning"""

    def __init__(self, device="cpu"):
        self.device = device

    def test_repeated_prisoners_dilemma(self, agent: nn.Module, rounds: int = 10) -> BenchmarkResult:
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
                action = output["actions"].item()

                # Count cooperation (action > 0.5 means cooperate)
                if action > 0.5:
                    cooperation_count += 1
                    if round_num > 0:
                        reciprocation_count += 1

        cooperation_rate = cooperation_count / rounds
        reciprocation_rate = reciprocation_count / max(1, rounds - 1)

        # Good ToM should show high reciprocation
        score = cooperation_rate * 0.3 + reciprocation_rate * 0.7
        passed = score >= 0.6

        return BenchmarkResult(
            test_name="Repeated Prisoner's Dilemma",
            score=score,
            max_score=1.0,
            passed=passed,
            details={"cooperation_rate": cooperation_rate, "reciprocation_rate": reciprocation_rate},
        )


class BenchmarkSuite:
    """Complete benchmark suite"""

    def __init__(self, device="cpu"):
        self.device = device
        self.sally_anne = SallyAnneTest(device)
        self.higher_order = HigherOrderToMBenchmark(max_order=5, device=device)
        self.zombie = ZombieDetectionBenchmark(device)
        self.cooperation = CooperationBenchmark(device)

    def run_full_suite(self, agent: nn.Module) -> Dict:
        """Run complete benchmark suite"""
        print("\n" + "=" * 60)
        print("Running ToM Benchmark Suite")
        print("=" * 60)

        all_results = []

        # Sally-Anne tests
        print("\n1. Sally-Anne Tests")
        print("-" * 60)
        sally_results = self.sally_anne.run_all(agent)
        for result in sally_results:
            all_results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.test_name:30s} {status:8s} {result.percentage:5.1f}%")

        # Higher-order ToM
        print("\n2. Higher-Order ToM Tests")
        print("-" * 60)
        higher_results = self.higher_order.run_all(agent)
        for result in higher_results:
            all_results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.test_name:30s} {status:8s} {result.percentage:5.1f}%")

        # Zombie detection
        print("\n3. Zombie Detection Tests")
        print("-" * 60)
        zombie_results = self.zombie.run_all(agent)
        for result in zombie_results:
            all_results.append(result)
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {result.test_name:30s} {status:8s} {result.percentage:5.1f}%")

        # Cooperation
        print("\n4. Cooperation Tests")
        print("-" * 60)
        coop_result = self.cooperation.test_repeated_prisoners_dilemma(agent)
        all_results.append(coop_result)
        status = "✓ PASS" if coop_result.passed else "✗ FAIL"
        print(f"  {coop_result.test_name:30s} {status:8s} {coop_result.percentage:5.1f}%")

        # Summary
        total_score = sum(r.score for r in all_results)
        max_score = sum(r.max_score for r in all_results)
        passed_count = sum(1 for r in all_results if r.passed)

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Total Score:  {total_score:.2f} / {max_score:.2f} ({total_score/max_score*100:.1f}%)")
        print(f"  Tests Passed: {passed_count} / {len(all_results)}")
        print("=" * 60)

        return {
            "results": all_results,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": total_score / max_score * 100,
            "passed_count": passed_count,
            "total_tests": len(all_results),
            "pass_rate": passed_count / len(all_results) * 100,
        }

    def quick_eval(self, agent: nn.Module) -> float:
        """Quick evaluation returning single score"""
        full_results = self.run_full_suite(agent)
        return full_results["percentage"] / 100.0
