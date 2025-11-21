"""
FIXED Developmental Theory of Mind Benchmarks for ToM-NAS

This module provides improved ToM tests that address issues found in validation:
1. Order 3+ tests now generate diverse scenarios
2. Scoring functions are harder to game
3. Tests properly enforce hierarchical structure
4. Answer distributions are balanced

Use this instead of tom_benchmarks.py for scientifically valid results.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random


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


class ImprovedSallyAnneScenario:
    """
    IMPROVED Sally-Anne scenario generator with:
    - Multiple diverse scenarios per order
    - Balanced answer distributions
    - True hierarchical difficulty
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

        # New dimensions for additional complexity
        self.CHARLIE_PRESENT = 12
        self.CHARLIE_BELIEF_A = 13
        self.CHARLIE_BELIEF_B = 14

    def generate_scenario(self, order: int = 1,
                         scenario_variant: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Generate scenario with controlled variation.

        Args:
            order: ToM order (0-5)
            scenario_variant: Specific variant to generate (0-N). If None, random.

        Returns:
            Tuple of (scenario_tensor, expected_answers)
        """
        # Choose variant if not specified
        if scenario_variant is None:
            num_variants = self._get_num_variants(order)
            scenario_variant = random.randint(0, num_variants - 1)

        if order == 0:
            return self._generate_order_0(scenario_variant)
        elif order == 1:
            return self._generate_order_1(scenario_variant)
        elif order == 2:
            return self._generate_order_2(scenario_variant)
        elif order == 3:
            return self._generate_order_3(scenario_variant)
        elif order == 4:
            return self._generate_order_4(scenario_variant)
        else:  # order == 5
            return self._generate_order_5(scenario_variant)

    def _get_num_variants(self, order: int) -> int:
        """Get number of scenario variants for each order."""
        return {0: 4, 1: 6, 2: 8, 3: 10, 4: 12, 5: 16}[min(order, 5)]

    def _generate_order_0(self, variant: int) -> Tuple[torch.Tensor, Dict]:
        """Order 0: Direct object tracking - where IS the object?"""
        seq_len = 5
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # Variants: object starts at A or B, may move
        initial_loc = 'A' if variant % 2 == 0 else 'B'
        object_moves = variant >= 2

        # T0: Initial state
        if initial_loc == 'A':
            scenario[0, 0, self.OBJECT_LOC_A] = 1.0
        else:
            scenario[0, 0, self.OBJECT_LOC_B] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0

        # T1-3: Copy forward or move
        for t in range(1, 4):
            scenario[0, t] = scenario[0, t-1].clone()

        # T2: Optional movement
        if object_moves:
            scenario[0, 2, self.OBJECT_LOC_A] = 0.0 if initial_loc == 'A' else 1.0
            scenario[0, 2, self.OBJECT_LOC_B] = 1.0 if initial_loc == 'A' else 0.0

        # Final state
        for t in range(2, 5):
            scenario[0, t] = scenario[0, 2].clone()

        # Expected: Where is object NOW?
        final_loc = 'B' if (initial_loc == 'A' and object_moves) or (initial_loc == 'B' and not object_moves) else 'A'
        if initial_loc == 'B' and object_moves:
            final_loc = 'A'

        expected = {
            'order_0': {
                'object_location': final_loc,
                'correct_answer_index': self.OBJECT_LOC_A if final_loc == 'A' else self.OBJECT_LOC_B
            }
        }

        return scenario, expected

    def _generate_order_1(self, variant: int) -> Tuple[torch.Tensor, Dict]:
        """Order 1: Sally's false belief - where does Sally THINK the object is?"""
        seq_len = 6
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # Variants: who leaves, what moves where
        sally_leaves = variant % 2 == 0  # Alternate who leaves
        initial_loc = 'A' if variant % 4 < 2 else 'B'
        final_loc = 'B' if initial_loc == 'A' else 'A'

        # T0: Initial - object in initial_loc, both present
        scenario[0, 0, self.OBJECT_LOC_A if initial_loc == 'A' else self.OBJECT_LOC_B] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0
        scenario[0, 0, self.SALLY_BELIEF_A if initial_loc == 'A' else self.SALLY_BELIEF_B] = 1.0
        scenario[0, 0, self.ANNE_BELIEF_A if initial_loc == 'A' else self.ANNE_BELIEF_B] = 1.0

        # T1: One person leaves
        scenario[0, 1] = scenario[0, 0].clone()
        if sally_leaves:
            scenario[0, 1, self.SALLY_PRESENT] = 0.0
        else:
            scenario[0, 1, self.ANNE_PRESENT] = 0.0

        # T2: Object moves while one is absent
        scenario[0, 2] = scenario[0, 1].clone()
        scenario[0, 2, self.OBJECT_LOC_A] = 1.0 if final_loc == 'A' else 0.0
        scenario[0, 2, self.OBJECT_LOC_B] = 1.0 if final_loc == 'B' else 0.0

        # Update beliefs of person present
        if sally_leaves:
            # Anne sees move, Sally doesn't
            scenario[0, 2, self.ANNE_BELIEF_A] = 1.0 if final_loc == 'A' else 0.0
            scenario[0, 2, self.ANNE_BELIEF_B] = 1.0 if final_loc == 'B' else 0.0
            # Sally's belief unchanged
        else:
            # Sally sees move, Anne doesn't
            scenario[0, 2, self.SALLY_BELIEF_A] = 1.0 if final_loc == 'A' else 0.0
            scenario[0, 2, self.SALLY_BELIEF_B] = 1.0 if final_loc == 'B' else 0.0
            # Anne's belief unchanged

        # T3-5: Person returns
        for t in range(3, 6):
            scenario[0, t] = scenario[0, 2].clone()
        scenario[0, 3, self.SALLY_PRESENT] = 1.0
        scenario[0, 3, self.ANNE_PRESENT] = 1.0
        for t in range(4, 6):
            scenario[0, t] = scenario[0, 3].clone()

        # Expected: Where does Sally THINK object is?
        # If Sally left, she still thinks initial_loc
        # If Sally stayed, she knows final_loc
        sally_thinks = initial_loc if sally_leaves else final_loc

        expected = {
            'order_1': {
                'sally_will_look': sally_thinks,
                'where_sally_thinks': sally_thinks,
                'is_false_belief': sally_leaves,
                'correct_belief_index': self.SALLY_BELIEF_A if sally_thinks == 'A' else self.SALLY_BELIEF_B
            }
        }

        return scenario, expected

    def _generate_order_2(self, variant: int) -> Tuple[torch.Tensor, Dict]:
        """Order 2: What does Anne know about Sally's belief?"""
        seq_len = 7
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # More complex variants with different knowledge states
        sally_left_during_move = variant % 2 == 0
        anne_saw_sally_see_initial = variant % 4 >= 2
        initial_loc = 'A' if variant < 4 else 'B'
        final_loc = 'B' if initial_loc == 'A' else 'A'

        # Build scenario
        scenario[0, 0, self.OBJECT_LOC_A if initial_loc == 'A' else self.OBJECT_LOC_B] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0
        scenario[0, 0, self.SALLY_BELIEF_A if initial_loc == 'A' else self.SALLY_BELIEF_B] = 1.0
        scenario[0, 0, self.ANNE_BELIEF_A if initial_loc == 'A' else self.ANNE_BELIEF_B] = 1.0

        # T1: Sally leaves
        scenario[0, 1] = scenario[0, 0].clone()
        if sally_left_during_move:
            scenario[0, 1, self.SALLY_PRESENT] = 0.0

        # T2: Object moves
        scenario[0, 2] = scenario[0, 1].clone()
        scenario[0, 2, self.OBJECT_LOC_A] = 1.0 if final_loc == 'A' else 0.0
        scenario[0, 2, self.OBJECT_LOC_B] = 1.0 if final_loc == 'B' else 0.0
        scenario[0, 2, self.ANNE_BELIEF_A] = 1.0 if final_loc == 'A' else 0.0
        scenario[0, 2, self.ANNE_BELIEF_B] = 1.0 if final_loc == 'B' else 0.0

        # Sally's belief only updates if present
        if not sally_left_during_move:
            scenario[0, 2, self.SALLY_BELIEF_A] = 1.0 if final_loc == 'A' else 0.0
            scenario[0, 2, self.SALLY_BELIEF_B] = 1.0 if final_loc == 'B' else 0.0

        # T3: Everyone reunites
        for t in range(3, 7):
            scenario[0, t] = scenario[0, 2].clone()
            scenario[0, t, self.SALLY_PRESENT] = 1.0
            scenario[0, t, self.ANNE_PRESENT] = 1.0

        # Anne's belief about Sally's belief
        # If Sally left during move, Anne knows Sally still thinks initial_loc
        anne_knows_sally_wrong = sally_left_during_move and anne_saw_sally_see_initial
        anne_thinks_sally_believes = initial_loc if sally_left_during_move else final_loc

        # Set Anne's model of Sally
        scenario[0, 4, self.ANNE_ABOUT_SALLY_A] = 1.0 if anne_thinks_sally_believes == 'A' else 0.3
        scenario[0, 4, self.ANNE_ABOUT_SALLY_B] = 1.0 if anne_thinks_sally_believes == 'B' else 0.3

        expected = {
            'order_2': {
                'anne_knows_sally_wrong': anne_knows_sally_wrong,
                'anne_prediction_of_sally': anne_thinks_sally_believes,
                'correct_index': self.ANNE_ABOUT_SALLY_A if anne_thinks_sally_believes == 'A' else self.ANNE_ABOUT_SALLY_B
            }
        }

        return scenario, expected

    def _generate_order_3(self, variant: int) -> Tuple[torch.Tensor, Dict]:
        """
        Order 3: What does Sally think Anne knows about Sally's belief?

        FIXED: Now generates multiple distinct scenarios with balanced answers.
        """
        seq_len = 8
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # Generate truly different scenarios
        # Variant determines: locations, who sees what, communication events
        initial_loc = 'A' if variant % 2 == 0 else 'B'
        final_loc = 'B' if initial_loc == 'A' else 'A'
        sally_saw_anne_see_move = variant % 4 >= 2  # Key: did Sally see Anne observe?
        anne_told_sally = (variant % 8) >= 4  # Did Anne communicate to Sally?
        confidence_level = 0.3 + (variant % 10) * 0.07  # Varying confidence

        # Build scenario with temporal structure
        # T0: Initial
        scenario[0, 0, self.OBJECT_LOC_A if initial_loc == 'A' else self.OBJECT_LOC_B] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0

        # Copy forward
        for t in range(1, 8):
            scenario[0, t] = scenario[0, t-1].clone()

        # T2: Sally leaves
        scenario[0, 2, self.SALLY_PRESENT] = 0.0

        # T3: Anne moves object
        scenario[0, 3, self.OBJECT_LOC_A] = 1.0 if final_loc == 'A' else 0.0
        scenario[0, 3, self.OBJECT_LOC_B] = 1.0 if final_loc == 'B' else 0.0

        # T4: Sally returns - but did she see Anne move it?
        scenario[0, 4, self.SALLY_PRESENT] = 1.0

        # KEY DIFFERENCE: Sally's model of Anne's knowledge
        # This depends on whether Sally observed Anne's observation
        if sally_saw_anne_see_move:
            # Sally knows Anne saw the move
            scenario[0, 5, self.SALLY_ABOUT_ANNE_A] = 0.9 if final_loc == 'A' else 0.1
            scenario[0, 5, self.SALLY_ABOUT_ANNE_B] = 0.9 if final_loc == 'B' else 0.1
            expected_sally_confidence = 'high'
        elif anne_told_sally:
            # Anne communicated, so Sally has some knowledge
            scenario[0, 5, self.SALLY_ABOUT_ANNE_A] = confidence_level if final_loc == 'A' else 0.3
            scenario[0, 5, self.SALLY_ABOUT_ANNE_B] = confidence_level if final_loc == 'B' else 0.3
            expected_sally_confidence = 'medium'
        else:
            # Sally has no information about Anne's knowledge
            scenario[0, 5, self.SALLY_ABOUT_ANNE_A] = 0.5  # Uncertain
            scenario[0, 5, self.SALLY_ABOUT_ANNE_B] = 0.5
            expected_sally_confidence = 'uncertain'

        # Copy to remaining timesteps
        for t in range(6, 8):
            scenario[0, t] = scenario[0, 5].clone()

        # Expected answers - now with variety
        expected = {
            'order_3': {
                'sally_thinks_anne_knows': expected_sally_confidence,
                'expected_confidence': confidence_level if anne_told_sally else (0.9 if sally_saw_anne_see_move else 0.5),
                'scenario_type': 'observed' if sally_saw_anne_see_move else ('communicated' if anne_told_sally else 'uncertain'),
                'correct_indices': [self.SALLY_ABOUT_ANNE_A, self.SALLY_ABOUT_ANNE_B]
            }
        }

        return scenario, expected

    def _generate_order_4(self, variant: int) -> Tuple[torch.Tensor, Dict]:
        """Order 4: What Anne thinks Sally thinks Anne knows."""
        seq_len = 9
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # Complex nested beliefs - generate diverse scenarios
        initial_loc = 'A' if variant % 2 == 0 else 'B'
        final_loc = 'B' if initial_loc == 'A' else 'A'

        # Various observation patterns
        sally_observed_anne = variant % 4 >= 2
        anne_observed_sally_observing = (variant % 8) >= 4
        uncertainty_factor = 0.2 + (variant % 12) * 0.05

        # Build scenario
        scenario[0, 0, self.OBJECT_LOC_A if initial_loc == 'A' else self.OBJECT_LOC_B] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0

        for t in range(1, 9):
            scenario[0, t] = scenario[0, t-1].clone()

        # Movement and observations
        scenario[0, 3, self.OBJECT_LOC_A] = 1.0 if final_loc == 'A' else 0.0
        scenario[0, 3, self.OBJECT_LOC_B] = 1.0 if final_loc == 'B' else 0.0

        # Set nested belief structure based on variant
        if sally_observed_anne and anne_observed_sally_observing:
            # Full observation chain
            belief_strength = 0.8 + uncertainty_factor
        elif sally_observed_anne:
            belief_strength = 0.6 + uncertainty_factor
        else:
            belief_strength = 0.3 + uncertainty_factor

        belief_strength = min(0.95, max(0.05, belief_strength))

        scenario[0, 6, self.SALLY_ABOUT_ANNE_A] = belief_strength if initial_loc == 'A' else 1 - belief_strength
        scenario[0, 6, self.SALLY_ABOUT_ANNE_B] = 1 - belief_strength if initial_loc == 'A' else belief_strength

        expected = {
            'order_4': {
                'anne_thinks_sally_thinks_anne_knows': 'structured' if belief_strength > 0.6 else 'uncertain',
                'expected_belief_strength': belief_strength,
                'has_structure': sally_observed_anne or anne_observed_sally_observing
            }
        }

        return scenario, expected

    def _generate_order_5(self, variant: int) -> Tuple[torch.Tensor, Dict]:
        """Order 5: Maximum depth recursive belief."""
        seq_len = 10
        scenario = torch.zeros(1, seq_len, self.input_dim, device=self.device)

        # Generate complex multi-agent scenarios
        initial_loc = 'A' if variant % 2 == 0 else 'B'
        final_loc = 'B' if initial_loc == 'A' else 'A'

        # Build base scenario
        scenario[0, 0, self.OBJECT_LOC_A if initial_loc == 'A' else self.OBJECT_LOC_B] = 1.0
        scenario[0, 0, self.SALLY_PRESENT] = 1.0
        scenario[0, 0, self.ANNE_PRESENT] = 1.0

        for t in range(1, 10):
            scenario[0, t] = scenario[0, t-1].clone()

        # Movement
        scenario[0, 3, self.OBJECT_LOC_A] = 1.0 if final_loc == 'A' else 0.0
        scenario[0, 3, self.OBJECT_LOC_B] = 1.0 if final_loc == 'B' else 0.0

        # Create hierarchical belief structure
        # Each level should have DECREASING confidence (realistic)
        base_conf = 0.9 - (variant % 16) * 0.02
        level_confidences = []
        for level in range(6):
            conf = base_conf * (0.85 ** level)  # Decay at each level
            level_confidences.append(conf)

        # Set belief dimensions
        for i in range(min(6, len(level_confidences))):
            if i * 2 + 1 < self.input_dim:
                scenario[0, 7, i * 2] = level_confidences[i]
                scenario[0, 7, i * 2 + 1] = 1 - level_confidences[i]

        # Check for proper hierarchy (decreasing confidence)
        is_properly_decreasing = all(
            level_confidences[i] >= level_confidences[i + 1] - 0.1
            for i in range(len(level_confidences) - 1)
        )

        expected = {
            'order_5': {
                'has_hierarchical_structure': is_properly_decreasing,
                'level_confidences': level_confidences,
                'mean_confidence': np.mean(level_confidences),
                'confidence_decay': base_conf * 0.85
            }
        }

        return scenario, expected


class ImprovedSallyAnneBenchmark:
    """
    IMPROVED Sally-Anne False Belief Test with:
    - Better scoring that can't be gamed
    - Multiple scenario variants
    - Proper hierarchical validation
    """

    def __init__(self, order: int, input_dim: int = 191, device: str = 'cpu'):
        self.order = order
        self.input_dim = input_dim
        self.device = device
        self.scenario_gen = ImprovedSallyAnneScenario(input_dim, device)

    def evaluate(self, agent: nn.Module, num_scenarios: int = 5) -> BenchmarkResult:
        """
        Evaluate agent across multiple scenario variants.

        Uses multiple scenarios to prevent gaming through memorization.
        """
        agent.eval()
        scores = []
        all_details = []

        num_variants = self.scenario_gen._get_num_variants(self.order)

        for variant in range(min(num_scenarios, num_variants)):
            scenario, expected = self.scenario_gen.generate_scenario(self.order, variant)

            with torch.no_grad():
                output = agent(scenario)
                beliefs = output.get('beliefs', torch.zeros(self.input_dim, device=self.device))

            if beliefs.dim() > 1:
                beliefs = beliefs[0]

            score, detail = self._evaluate_order(beliefs, expected)
            scores.append(score)
            all_details.append(detail)

        # Average across scenarios
        avg_score = np.mean(scores)
        score_std = np.std(scores)

        # Penalize high variance (indicates inconsistent/gaming behavior)
        if score_std > 0.3:
            avg_score *= 0.8  # 20% penalty for inconsistency

        passed = avg_score > 0.5 and score_std < 0.4

        return BenchmarkResult(
            name=f"Improved Sally-Anne Order {self.order}",
            order=self.order,
            score=avg_score,
            passed=passed,
            details={
                'per_scenario_scores': scores,
                'score_std': score_std,
                'scenario_details': all_details
            },
            expected=expected,
            actual=f"avg={avg_score:.3f}, std={score_std:.3f}"
        )

    def _evaluate_order(self, beliefs: torch.Tensor, expected: Dict) -> Tuple[float, Dict]:
        """Evaluate beliefs against expected for this order."""

        if self.order == 0:
            exp = expected['order_0']
            correct_idx = exp['correct_answer_index']
            wrong_idx = 1 - correct_idx  # Opposite location

            correct_val = beliefs[correct_idx].item()
            wrong_val = beliefs[wrong_idx].item()

            # Must clearly prefer correct location
            score = correct_val / (correct_val + wrong_val + 1e-8)
            detail = {'correct': correct_val, 'wrong': wrong_val}

        elif self.order == 1:
            exp = expected['order_1']
            correct_idx = exp['correct_belief_index']
            wrong_idx = self.scenario_gen.SALLY_BELIEF_B if correct_idx == self.scenario_gen.SALLY_BELIEF_A else self.scenario_gen.SALLY_BELIEF_A

            correct_val = beliefs[correct_idx].item()
            wrong_val = beliefs[wrong_idx].item()

            # Must track Sally's belief correctly
            score = correct_val / (correct_val + wrong_val + 1e-8)
            detail = {'correct': correct_val, 'wrong': wrong_val, 'is_false_belief': exp['is_false_belief']}

        elif self.order == 2:
            exp = expected['order_2']
            correct_idx = exp['correct_index']
            wrong_idx = self.scenario_gen.ANNE_ABOUT_SALLY_B if correct_idx == self.scenario_gen.ANNE_ABOUT_SALLY_A else self.scenario_gen.ANNE_ABOUT_SALLY_A

            correct_val = beliefs[correct_idx].item()
            wrong_val = beliefs[wrong_idx].item()

            # Must model Anne's belief about Sally
            score = correct_val / (correct_val + wrong_val + 1e-8)
            detail = {'correct': correct_val, 'wrong': wrong_val}

        elif self.order == 3:
            exp = expected['order_3']
            expected_conf = exp['expected_confidence']
            scenario_type = exp['scenario_type']

            # Get Sally's beliefs about Anne's knowledge
            sally_about_anne_a = beliefs[self.scenario_gen.SALLY_ABOUT_ANNE_A].item()
            sally_about_anne_b = beliefs[self.scenario_gen.SALLY_ABOUT_ANNE_B].item()

            # Score based on matching expected confidence pattern
            if scenario_type == 'observed':
                # Should have high confidence (>0.7) in one direction
                has_confidence = max(sally_about_anne_a, sally_about_anne_b) > 0.6
                score = 0.7 if has_confidence else 0.3
            elif scenario_type == 'communicated':
                # Should have medium confidence (0.4-0.7)
                max_val = max(sally_about_anne_a, sally_about_anne_b)
                if 0.4 < max_val < 0.8:
                    score = 0.7
                else:
                    score = 0.3
            else:  # uncertain
                # Should have near 0.5 (uncertain)
                is_uncertain = abs(sally_about_anne_a - 0.5) < 0.2 and abs(sally_about_anne_b - 0.5) < 0.2
                score = 0.7 if is_uncertain else 0.3

            detail = {
                'sally_about_anne_a': sally_about_anne_a,
                'sally_about_anne_b': sally_about_anne_b,
                'expected_type': scenario_type,
                'expected_confidence': expected_conf
            }

        elif self.order == 4:
            exp = expected['order_4']
            expected_strength = exp['expected_belief_strength']

            # Check for structured beliefs
            beliefs_slice = beliefs[8:12]
            belief_mean = beliefs_slice.mean().item()
            belief_std = beliefs_slice.std().item()

            # Must have non-trivial structure
            has_structure = belief_std > 0.1
            matches_expected = abs(belief_mean - expected_strength) < 0.3

            score = 0.0
            if has_structure:
                score += 0.4
            if matches_expected:
                score += 0.4
            if has_structure and matches_expected:
                score += 0.2

            detail = {
                'belief_mean': belief_mean,
                'belief_std': belief_std,
                'expected_strength': expected_strength
            }

        else:  # Order 5
            exp = expected['order_5']
            expected_levels = exp['level_confidences']

            # Check for decreasing confidence pattern
            level_values = []
            for i in range(6):
                if i * 2 < len(beliefs):
                    level_values.append(beliefs[i * 2].item())

            if len(level_values) >= 3:
                # Check if decreasing
                is_decreasing = all(
                    level_values[i] >= level_values[i + 1] - 0.15
                    for i in range(len(level_values) - 1)
                )

                # Check variance (should have structure)
                has_structure = np.var(level_values) > 0.01

                score = 0.0
                if is_decreasing:
                    score += 0.5
                if has_structure:
                    score += 0.3
                if np.mean(level_values) > 0.3:
                    score += 0.2
            else:
                is_decreasing = False
                has_structure = False
                score = 0.0

            detail = {
                'level_values': level_values,
                'is_decreasing': is_decreasing,
                'has_structure': has_structure
            }

        return score, detail


class ImprovedToMBenchmarkSuite:
    """Complete improved ToM benchmark suite."""

    def __init__(self, input_dim: int = 191, device: str = 'cpu'):
        self.input_dim = input_dim
        self.device = device

        self.benchmarks = {}
        for order in range(6):
            self.benchmarks[f'sally_anne_order_{order}'] = ImprovedSallyAnneBenchmark(
                order=order, input_dim=input_dim, device=device
            )

    def run_full_evaluation(self, agent: nn.Module, agent_id: str = "") -> Dict[str, Any]:
        """Run all benchmarks with improved scoring."""
        results = {}
        scores_by_order = {}

        for name, benchmark in self.benchmarks.items():
            try:
                result = benchmark.evaluate(agent, num_scenarios=5)
                results[name] = {
                    'score': result.score,
                    'passed': result.passed,
                    'order': result.order,
                    'details': result.details
                }
                scores_by_order[result.order] = result.score
            except Exception as e:
                results[name] = {'score': 0.0, 'passed': False, 'error': str(e)}
                scores_by_order[benchmark.order] = 0.0

        # Check hierarchy
        hierarchy_valid = True
        violations = []
        for i in range(1, 6):
            if i in scores_by_order and i - 1 in scores_by_order:
                if scores_by_order[i] > scores_by_order[i - 1] + 0.15:
                    hierarchy_valid = False
                    violations.append(f"Order {i} ({scores_by_order[i]:.2f}) > Order {i-1} ({scores_by_order[i-1]:.2f})")

        all_scores = [r['score'] for r in results.values() if 'score' in r]
        progression = [scores_by_order.get(i, 0) for i in range(6)]

        max_order = -1
        for i in range(6):
            if results.get(f'sally_anne_order_{i}', {}).get('passed', False):
                max_order = i

        return {
            'benchmark_results': results,
            'overall_score': np.mean(all_scores) if all_scores else 0.0,
            'sally_anne_progression': progression,
            'max_tom_order': max_order,
            'scores_by_order': scores_by_order,
            'hierarchy_valid': hierarchy_valid,
            'hierarchy_violations': violations,
            'num_passed': sum(1 for r in results.values() if r.get('passed', False)),
            'num_total': len(results)
        }


# Export for use as drop-in replacement
def create_fixed_tom_suite(input_dim: int = 191, device: str = 'cpu') -> ImprovedToMBenchmarkSuite:
    """Create the improved/fixed ToM benchmark suite."""
    return ImprovedToMBenchmarkSuite(input_dim=input_dim, device=device)
