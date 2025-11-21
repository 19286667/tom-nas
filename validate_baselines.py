#!/usr/bin/env python3
"""
ToM-NAS Comprehensive Validation Suite

This script provides scientific validation of ToM-NAS results by:
1. Establishing random baselines for all tests
2. Analyzing test structure (Order 3, metacognitive)
3. Testing ToM hierarchy dependencies
4. Characterizing evolved individuals
5. Generating diagnostic reports

Run this BEFORE using results in publications/dissertations.

Usage:
    python validate_baselines.py                    # Full validation
    python validate_baselines.py --baselines        # Only baselines
    python validate_baselines.py --analyze-order3   # Analyze Order 3 test
    python validate_baselines.py --checkpoint FILE  # Analyze checkpoint
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# RANDOM AGENT FOR BASELINES
# =============================================================================

class RandomAgent(nn.Module):
    """Agent that outputs random values - establishes chance baseline."""

    def __init__(self, input_dim: int = 191, output_dim: int = 181):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Dummy parameter so .parameters() works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        return {
            'beliefs': torch.rand(batch_size, self.output_dim, device=x.device),
            'actions': torch.rand(batch_size, device=x.device),
            'hidden_states': torch.rand(batch_size, 1, self.output_dim, device=x.device)
        }


class ConstantAgent(nn.Module):
    """Agent that always outputs 0.5 - tests for exploitable biases."""

    def __init__(self, input_dim: int = 191, output_dim: int = 181, constant: float = 0.5):
        super().__init__()
        self.output_dim = output_dim
        self.constant = constant
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        return {
            'beliefs': torch.full((batch_size, self.output_dim), self.constant, device=x.device),
            'actions': torch.full((batch_size,), self.constant, device=x.device),
            'hidden_states': torch.full((batch_size, 1, self.output_dim), self.constant, device=x.device)
        }


class AlwaysAAgent(nn.Module):
    """Agent that always predicts location A - tests answer distribution bias."""

    def __init__(self, input_dim: int = 191, output_dim: int = 181):
        super().__init__()
        self.output_dim = output_dim
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        beliefs = torch.zeros(batch_size, self.output_dim, device=x.device)
        # Set "location A" related beliefs high
        beliefs[:, 0] = 0.9  # Object at A
        beliefs[:, 3] = 0.9  # Sally believes A
        beliefs[:, 4] = 0.1  # Sally believes B (low)
        beliefs[:, 6] = 0.9  # Anne believes A
        beliefs[:, 8] = 0.9  # Sally thinks Anne knows A
        return {
            'beliefs': beliefs,
            'actions': torch.full((batch_size,), 0.9, device=x.device),
            'hidden_states': beliefs.unsqueeze(1)
        }


# =============================================================================
# VALIDATION RESULT STRUCTURES
# =============================================================================

@dataclass
class BaselineResult:
    """Result from baseline testing."""
    test_name: str
    random_mean: float
    random_std: float
    constant_mean: float
    always_a_mean: float
    interpretation: str
    is_meaningful: bool  # Is the test harder than random chance?


@dataclass
class OrderAnalysis:
    """Analysis of a specific ToM order."""
    order: int
    num_scenarios: int
    unique_scenarios: int
    answer_distribution: Dict[str, float]
    complexity_score: float
    potential_issues: List[str]


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    baseline_results: Dict[str, BaselineResult]
    order_analyses: Dict[int, OrderAnalysis]
    hierarchy_validation: Dict[str, Any]
    metacognitive_analysis: Dict[str, Any]
    recommendations: List[str]
    overall_validity: str  # "VALID", "NEEDS_REVIEW", "INVALID"


# =============================================================================
# BASELINE TESTING
# =============================================================================

def run_baseline_tests(num_trials: int = 100, verbose: bool = True) -> Dict[str, BaselineResult]:
    """Run baseline tests with random, constant, and biased agents."""

    from src.evaluation.tom_benchmarks import ToMBenchmarkSuite, SallyAnneBenchmark
    from src.evaluation.zombie_detection import ZombieDetectionSuite

    if verbose:
        print("\n" + "="*70)
        print("BASELINE VALIDATION")
        print("="*70)
        print(f"Running {num_trials} trials per agent type...")

    device = 'cpu'
    input_dim = 191
    output_dim = 181

    # Create test agents
    random_agent = RandomAgent(input_dim, output_dim)
    constant_agent = ConstantAgent(input_dim, output_dim, constant=0.5)
    always_a_agent = AlwaysAAgent(input_dim, output_dim)

    # Create test suites
    tom_suite = ToMBenchmarkSuite(input_dim=input_dim, device=device)
    zombie_suite = ZombieDetectionSuite(device=device)
    context = {'input_dim': input_dim}

    results = {}

    # Test ToM Orders
    if verbose:
        print("\n--- ToM Order Baselines ---")

    for order in range(6):
        random_scores = []
        constant_scores = []
        always_a_scores = []

        benchmark = SallyAnneBenchmark(order=order, input_dim=input_dim, device=device)

        for _ in range(num_trials):
            # Random agent
            result = benchmark.evaluate(random_agent)
            random_scores.append(result.score)

            # Constant agent
            result = benchmark.evaluate(constant_agent)
            constant_scores.append(result.score)

            # Always-A agent
            result = benchmark.evaluate(always_a_agent)
            always_a_scores.append(result.score)

        random_mean = np.mean(random_scores)
        random_std = np.std(random_scores)
        constant_mean = np.mean(constant_scores)
        always_a_mean = np.mean(always_a_scores)

        # Interpret results
        if always_a_mean > 0.7:
            interpretation = "WARNING: Biased toward 'A' answer. Test may have exploitable structure."
            is_meaningful = False
        elif random_mean > 0.6:
            interpretation = "WARNING: Random agent scores high. Test may be too easy."
            is_meaningful = False
        elif random_std < 0.05:
            interpretation = "WARNING: Low variance. Test may be deterministic/trivial."
            is_meaningful = random_mean < 0.4
        else:
            interpretation = "OK: Test appears to measure meaningful capability."
            is_meaningful = True

        results[f'tom_order_{order}'] = BaselineResult(
            test_name=f'Sally-Anne Order {order}',
            random_mean=random_mean,
            random_std=random_std,
            constant_mean=constant_mean,
            always_a_mean=always_a_mean,
            interpretation=interpretation,
            is_meaningful=is_meaningful
        )

        if verbose:
            status = "OK" if is_meaningful else "WARNING"
            print(f"  Order {order}: Random={random_mean:.3f}+/-{random_std:.3f}, "
                  f"Constant={constant_mean:.3f}, AlwaysA={always_a_mean:.3f} [{status}]")

    # Test Zombie Detection
    if verbose:
        print("\n--- Zombie Detection Baselines ---")

    zombie_types = ['behavioral', 'belief', 'causal', 'metacognitive', 'linguistic', 'emotional']

    for ztype in zombie_types:
        random_scores = []
        constant_scores = []

        for _ in range(num_trials):
            result = zombie_suite.run_full_evaluation(random_agent, context)
            random_scores.append(result['test_results'][ztype]['score'])

            result = zombie_suite.run_full_evaluation(constant_agent, context)
            constant_scores.append(result['test_results'][ztype]['score'])

        random_mean = np.mean(random_scores)
        random_std = np.std(random_scores)
        constant_mean = np.mean(constant_scores)

        if random_mean > 0.6:
            interpretation = f"WARNING: Random scores {random_mean:.2f}. Test may be too easy."
            is_meaningful = False
        elif constant_mean > 0.7:
            interpretation = f"WARNING: Constant agent scores {constant_mean:.2f}. Test may be trivial."
            is_meaningful = False
        else:
            interpretation = "OK: Test appears meaningful."
            is_meaningful = True

        results[f'zombie_{ztype}'] = BaselineResult(
            test_name=f'Zombie Detection: {ztype}',
            random_mean=random_mean,
            random_std=random_std,
            constant_mean=constant_mean,
            always_a_mean=0.0,  # Not applicable
            interpretation=interpretation,
            is_meaningful=is_meaningful
        )

        if verbose:
            status = "OK" if is_meaningful else "WARNING"
            print(f"  {ztype:15s}: Random={random_mean:.3f}+/-{random_std:.3f}, "
                  f"Constant={constant_mean:.3f} [{status}]")

    return results


# =============================================================================
# ORDER 3 ANALYSIS
# =============================================================================

def analyze_order_3_test(num_scenarios: int = 100, verbose: bool = True) -> OrderAnalysis:
    """Deep analysis of Order 3 test to understand why it might spike."""

    from src.evaluation.tom_benchmarks import SallyAnneScenario, SallyAnneBenchmark

    if verbose:
        print("\n" + "="*70)
        print("ORDER 3 TEST ANALYSIS")
        print("="*70)

    scenario_gen = SallyAnneScenario(input_dim=191, device='cpu')
    benchmark = SallyAnneBenchmark(order=3, input_dim=191, device='cpu')

    # Generate scenarios and analyze structure
    scenarios = []
    expected_answers = []

    for _ in range(num_scenarios):
        scenario, expected = scenario_gen.generate_scenario(order=3)
        scenarios.append(scenario)
        expected_answers.append(expected)

    # Check uniqueness
    unique_scenarios = len(set([s.numpy().tobytes() for s in scenarios]))

    # Analyze answer distribution (from expected dict)
    answer_counts = {}
    for exp in expected_answers:
        order_3_exp = exp.get('order_3', {})
        key = str(order_3_exp)
        answer_counts[key] = answer_counts.get(key, 0) + 1

    answer_distribution = {k: v/num_scenarios for k, v in answer_counts.items()}

    # Calculate complexity score
    # Low complexity if few unique scenarios or highly skewed distribution
    uniqueness_ratio = unique_scenarios / num_scenarios
    max_answer_freq = max(answer_distribution.values()) if answer_distribution else 1.0
    complexity_score = uniqueness_ratio * (1 - max_answer_freq + 0.5)

    # Identify potential issues
    issues = []

    if unique_scenarios < num_scenarios * 0.5:
        issues.append(f"Low scenario diversity: only {unique_scenarios}/{num_scenarios} unique")

    if max_answer_freq > 0.7:
        issues.append(f"Skewed answer distribution: most common answer is {max_answer_freq:.1%}")

    # Test with different agent strategies
    if verbose:
        print(f"\n  Scenarios generated: {num_scenarios}")
        print(f"  Unique scenarios: {unique_scenarios}")
        print(f"  Complexity score: {complexity_score:.3f}")

        print(f"\n  Answer distribution:")
        for ans, freq in sorted(answer_distribution.items(), key=lambda x: -x[1])[:5]:
            print(f"    {ans[:50]}... : {freq:.1%}")

    # Test scoring mechanism directly
    if verbose:
        print("\n  Order 3 Scoring Analysis:")
        print("  " + "-"*40)

    # Analyze what Order 3 actually measures
    # From the code, Order 3 checks:
    # 1. sally_about_anne_a activation (0.7 weight)
    # 2. Whether belief_diff > 0.2 (has_prediction) (0.3 bonus)

    test_beliefs = torch.zeros(191)

    # Scenario 1: Perfect "correct" answer
    test_beliefs[8] = 0.6  # sally_about_anne_a
    test_beliefs[9] = 0.1  # sally_about_anne_b
    belief_diff = abs(0.6 - 0.1)
    has_prediction = belief_diff > 0.2
    score_1 = 0.6 * 0.7 + (0.3 if has_prediction else 0.0)

    if verbose:
        print(f"    Perfect answer beliefs: score = {score_1:.3f}")

    # Scenario 2: Random beliefs around 0.5
    test_beliefs[8] = 0.5
    test_beliefs[9] = 0.5
    belief_diff = abs(0.5 - 0.5)
    has_prediction = belief_diff > 0.2
    score_2 = 0.5 * 0.7 + (0.3 if has_prediction else 0.0)

    if verbose:
        print(f"    Uncertain (0.5,0.5) beliefs: score = {score_2:.3f}")

    # Scenario 3: Polarized beliefs (easy for agent to learn)
    test_beliefs[8] = 0.9
    test_beliefs[9] = 0.1
    belief_diff = abs(0.9 - 0.1)
    has_prediction = belief_diff > 0.2
    score_3 = 0.9 * 0.7 + (0.3 if has_prediction else 0.0)

    if verbose:
        print(f"    Polarized (0.9,0.1) beliefs: score = {score_3:.3f}")

        # Key insight
        print("\n  KEY INSIGHT:")
        print("  Order 3 rewards high sally_about_anne_a (dim 8) + any non-random prediction.")
        print("  An agent that learns to output high dim[8] with varied dim[9] will score well,")
        print("  regardless of actual 3rd-order recursive reasoning.")

    if score_3 > 0.8:
        issues.append("Scoring function can be 'gamed' by outputting polarized beliefs")

    result = OrderAnalysis(
        order=3,
        num_scenarios=num_scenarios,
        unique_scenarios=unique_scenarios,
        answer_distribution=answer_distribution,
        complexity_score=complexity_score,
        potential_issues=issues
    )

    if verbose:
        if issues:
            print(f"\n  POTENTIAL ISSUES FOUND:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("\n  No obvious issues detected.")

    return result


# =============================================================================
# METACOGNITIVE TEST ANALYSIS (1.000 score investigation)
# =============================================================================

def analyze_metacognitive_test(verbose: bool = True) -> Dict[str, Any]:
    """Analyze why metacognitive test might show perfect 1.000 scores."""

    from src.evaluation.zombie_detection import MetacognitiveCalibrationTest

    if verbose:
        print("\n" + "="*70)
        print("METACOGNITIVE TEST ANALYSIS (1.000 Score Investigation)")
        print("="*70)

    test = MetacognitiveCalibrationTest()
    context = {'input_dim': 191}

    analysis = {
        'scoring_mechanism': {},
        'edge_cases': [],
        'potential_exploits': [],
        'recommendations': []
    }

    # Analyze the scoring mechanism
    if verbose:
        print("\n  Scoring Mechanism Analysis:")
        print("  " + "-"*40)

    # The metacognitive test scores based on:
    # 1. clear_conf > ambig_conf (+0.33)
    # 2. ambig_conf > unknown_conf (+0.33)
    # 3. clear_conf > unknown_conf (+0.34)
    # Penalty: if unknown_conf > 0.7, score *= 0.5

    # Test various agent output patterns
    class FixedConfidenceAgent(nn.Module):
        def __init__(self, clear_val, ambig_val, unknown_val, output_dim=191):
            super().__init__()
            self.clear_val = clear_val
            self.ambig_val = ambig_val
            self.unknown_val = unknown_val
            self.output_dim = output_dim
            self.call_count = 0
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            self.call_count += 1
            batch_size = x.shape[0]

            # Detect which scenario based on input characteristics
            input_sum = x.sum().item()
            input_mean = x.mean().item()

            # Clear = high input, Ambiguous = varied, Unknown = zeros
            if input_mean > 0.5:  # Clear input (ones)
                val = self.clear_val
            elif abs(input_mean) < 0.01:  # Unknown input (zeros)
                val = self.unknown_val
            else:  # Ambiguous (randn)
                val = self.ambig_val

            beliefs = torch.full((batch_size, self.output_dim), val)
            return {'beliefs': beliefs}

    # Test cases
    test_cases = [
        ("Perfectly calibrated", 0.9, 0.5, 0.1),
        ("Moderately calibrated", 0.7, 0.5, 0.3),
        ("Poorly calibrated", 0.5, 0.5, 0.5),
        ("Inverted calibration", 0.1, 0.5, 0.9),
        ("All confident", 0.9, 0.9, 0.9),
        ("All uncertain", 0.1, 0.1, 0.1),
    ]

    if verbose:
        print("\n  Testing different calibration patterns:")

    for name, clear, ambig, unknown in test_cases:
        agent = FixedConfidenceAgent(clear, ambig, unknown)
        result = test.run_test(agent, context)
        score = result['score']
        passed = result['passed']

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"    {name:25s}: clear={clear:.1f}, ambig={ambig:.1f}, unknown={unknown:.1f} "
                  f"-> score={score:.3f} [{status}]")

        if score == 1.0:
            analysis['potential_exploits'].append(
                f"Pattern '{name}' (clear={clear}, ambig={ambig}, unknown={unknown}) achieves perfect score"
            )

    # Check if the test is too easy
    random_scores = []
    random_agent = RandomAgent()
    for _ in range(100):
        result = test.run_test(random_agent, context)
        random_scores.append(result['score'])

    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)

    analysis['random_baseline'] = {
        'mean': random_mean,
        'std': random_std,
        'max': max(random_scores),
        'min': min(random_scores)
    }

    if verbose:
        print(f"\n  Random Agent Baseline:")
        print(f"    Mean: {random_mean:.3f} +/- {random_std:.3f}")
        print(f"    Range: [{min(random_scores):.3f}, {max(random_scores):.3f}]")

    # Analysis insights
    if random_mean > 0.5:
        analysis['edge_cases'].append("Random agent achieves >50% - test may be too easy")

    if verbose:
        print("\n  KEY INSIGHTS:")
        print("  The metacognitive test checks if confidence scales with evidence clarity.")
        print("  A score of 1.000 means the agent's confidence correctly orders:")
        print("    clear_evidence > ambiguous_evidence > no_evidence")
        print("\n  However, an agent that simply outputs:")
        print("    - High values for clear inputs (ones)")
        print("    - Medium values for noisy inputs")
        print("    - Low values for zero inputs")
        print("  ...will score perfectly without actual metacognitive ability.")

    analysis['recommendations'] = [
        "Consider adding more nuanced calibration tests (Brier score on predictions)",
        "Test confidence vs actual accuracy, not just input characteristics",
        "Add adversarial examples that should produce uncertainty"
    ]

    return analysis


# =============================================================================
# HIERARCHY DEPENDENCY TESTS
# =============================================================================

def test_hierarchy_dependency(verbose: bool = True) -> Dict[str, Any]:
    """Test if ToM orders are truly hierarchical (lower orders prerequisite for higher)."""

    from src.evaluation.tom_benchmarks import ToMBenchmarkSuite, SallyAnneBenchmark
    from src.agents.architectures import TransparentRNN

    if verbose:
        print("\n" + "="*70)
        print("TOM HIERARCHY DEPENDENCY TEST")
        print("="*70)
        print("\nQuestion: Can an agent pass Order N without passing Order N-1?")

    results = {
        'independence_tests': [],
        'transfer_tests': [],
        'hierarchy_valid': True,
        'violations': []
    }

    input_dim = 191
    output_dim = 181

    # Create agents optimized for specific orders
    if verbose:
        print("\n  Creating order-specialized agents...")

    # Train simple models to maximize specific order performance
    order_specialists = {}

    for target_order in range(6):
        # Create agent
        agent = TransparentRNN(input_dim, 64, output_dim, num_layers=2)
        benchmark = SallyAnneBenchmark(order=target_order, input_dim=input_dim)

        # Quick "training" - just run forward passes with gradient updates
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)

        for _ in range(50):  # Quick training
            scenario, _ = benchmark.scenario_gen.generate_scenario(target_order)
            output = agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(output_dim))

            # Loss: maximize score for target order
            # For Sally-Anne, we want high activation at specific indices
            if target_order <= 1:
                # Want high sally_belief_a (index 4), low sally_belief_b (index 5)
                target = torch.zeros(output_dim)
                target[4] = 1.0  # Sally believes A
                target[3] = 1.0  # Sally believes A (backup index)
            else:
                # Higher orders use indices 8-11
                target = torch.zeros(output_dim)
                target[8] = 0.8  # sally_about_anne_a
                target[9] = 0.2  # sally_about_anne_b (should be different)

            loss = nn.MSELoss()(beliefs[0] if beliefs.dim() > 1 else beliefs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        order_specialists[target_order] = agent

    # Test each specialist on ALL orders
    if verbose:
        print("\n  Testing specialists on all orders:")
        print("  " + "-"*60)
        header = "Specialist\\Test " + " ".join([f"Ord{i}" for i in range(6)])
        print(f"  {header}")

    for specialist_order, agent in order_specialists.items():
        scores = []
        for test_order in range(6):
            benchmark = SallyAnneBenchmark(order=test_order, input_dim=input_dim)
            result = benchmark.evaluate(agent)
            scores.append(result.score)

        if verbose:
            scores_str = " ".join([f"{s:.2f} " for s in scores])
            print(f"  Order {specialist_order} spec:     {scores_str}")

        results['independence_tests'].append({
            'specialist_order': specialist_order,
            'scores': scores
        })

        # Check for hierarchy violations
        # An Order N specialist should NOT score higher on Order N than on Order 0
        if specialist_order > 0 and scores[specialist_order] > scores[0] + 0.1:
            results['violations'].append(
                f"Order {specialist_order} specialist scores {scores[specialist_order]:.2f} on its order "
                f"but only {scores[0]:.2f} on Order 0 - hierarchy may be inverted"
            )
            results['hierarchy_valid'] = False

    # Transfer test: Does training on lower orders help higher orders?
    if verbose:
        print("\n  Transfer Learning Test:")
        print("  Does training on Order 0-1 help with Order 2-5?")

    # Train on Orders 0-1, test on all
    transfer_agent = TransparentRNN(input_dim, 64, output_dim, num_layers=2)
    optimizer = torch.optim.Adam(transfer_agent.parameters(), lr=0.01)

    # Train on Order 0 and 1
    for _ in range(100):
        for train_order in [0, 1]:
            benchmark = SallyAnneBenchmark(order=train_order, input_dim=input_dim)
            scenario, _ = benchmark.scenario_gen.generate_scenario(train_order)
            output = transfer_agent(scenario)
            beliefs = output.get('beliefs', torch.zeros(output_dim))

            target = torch.zeros(output_dim)
            target[4] = 1.0 if train_order <= 1 else 0.5

            loss = nn.MSELoss()(beliefs[0] if beliefs.dim() > 1 else beliefs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    transfer_scores = []
    for test_order in range(6):
        benchmark = SallyAnneBenchmark(order=test_order, input_dim=input_dim)
        result = benchmark.evaluate(transfer_agent)
        transfer_scores.append(result.score)

    results['transfer_tests'] = {
        'trained_orders': [0, 1],
        'scores_on_all': transfer_scores
    }

    if verbose:
        print(f"    Trained on: Orders 0-1")
        print(f"    Performance: {' '.join([f'O{i}:{s:.2f}' for i, s in enumerate(transfer_scores)])}")

        if transfer_scores[0] > 0.5 and transfer_scores[2] < 0.4:
            print("    Result: Transfer learning helps - hierarchy is valid")
        else:
            print("    Result: Unclear transfer - orders may be independent")

    # Summary
    if verbose:
        print("\n  HIERARCHY ANALYSIS SUMMARY:")
        if results['hierarchy_valid']:
            print("    Hierarchy appears VALID - lower orders are prerequisites")
        else:
            print("    Hierarchy VIOLATIONS detected:")
            for v in results['violations']:
                print(f"      - {v}")

    return results


# =============================================================================
# BEST INDIVIDUAL CHARACTERIZATION
# =============================================================================

def characterize_checkpoint(checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Load and characterize a checkpoint's best individual."""

    from src.evolution.operators import ArchitectureGene
    from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
    from src.evaluation.tom_benchmarks import ToMBenchmarkSuite
    from src.evaluation.zombie_detection import ZombieDetectionSuite

    if verbose:
        print("\n" + "="*70)
        print("BEST INDIVIDUAL CHARACTERIZATION")
        print("="*70)
        print(f"Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    analysis = {
        'architecture': {},
        'ontology_usage': {},
        'test_performance': {},
        'reasoning_traces': []
    }

    # Extract best individual info
    best_info = checkpoint.get('best_individual', {})
    gene_dict = best_info.get('gene', {})
    fitness = best_info.get('fitness', 0)

    if verbose:
        print(f"\n  Best Individual:")
        print(f"    Fitness: {fitness:.4f}")
        print(f"    Architecture: {gene_dict.get('arch_type', 'Unknown')}")
        print(f"    Layers: {gene_dict.get('num_layers', 'N/A')}")
        print(f"    Hidden dim: {gene_dict.get('hidden_dim', 'N/A')}")

    analysis['architecture'] = gene_dict

    # Recreate the model
    arch_type = gene_dict.get('arch_type', 'TRN')
    input_dim = 191
    output_dim = 181
    hidden_dim = gene_dict.get('hidden_dim', 128)
    num_layers = gene_dict.get('num_layers', 2)

    if arch_type == 'TRN':
        model = TransparentRNN(input_dim, hidden_dim, output_dim, num_layers=num_layers)
    elif arch_type == 'RSAN':
        model = RecursiveSelfAttention(input_dim, hidden_dim, output_dim,
                                        num_heads=gene_dict.get('num_heads', 4),
                                        max_recursion=gene_dict.get('max_recursion', 5))
    else:
        model = TransformerToMAgent(input_dim, hidden_dim, output_dim,
                                     num_layers=num_layers,
                                     num_heads=gene_dict.get('num_heads', 4))

    # Load weights
    if 'model_state' in best_info:
        try:
            model.load_state_dict(best_info['model_state'])
            if verbose:
                print("    Weights loaded successfully")
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not load weights: {e}")

    # Run evaluations
    tom_suite = ToMBenchmarkSuite(input_dim=input_dim)
    zombie_suite = ZombieDetectionSuite()
    context = {'input_dim': input_dim}

    tom_results = tom_suite.run_full_evaluation(model, agent_id="best")
    zombie_results = zombie_suite.run_full_evaluation(model, context)

    analysis['test_performance'] = {
        'tom': {
            'overall_score': tom_results['overall_score'],
            'progression': tom_results['sally_anne_progression'],
            'max_order': tom_results['max_tom_order'],
            'hierarchy_valid': tom_results['hierarchy_valid']
        },
        'zombie': {
            'overall_score': zombie_results['overall_score'],
            'test_results': {k: v['score'] for k, v in zombie_results['test_results'].items()},
            'is_zombie': zombie_results['is_likely_zombie']
        }
    }

    if verbose:
        print(f"\n  ToM Performance:")
        for i, score in enumerate(tom_results['sally_anne_progression']):
            status = "PASS" if score > 0.5 else "FAIL"
            print(f"    Order {i}: {score:.3f} [{status}]")
        print(f"    Hierarchy Valid: {tom_results['hierarchy_valid']}")

        print(f"\n  Zombie Detection:")
        for ztype, result in zombie_results['test_results'].items():
            print(f"    {ztype:15s}: {result['score']:.3f}")

    # Analyze ontology usage
    if verbose:
        print(f"\n  Ontology Usage Analysis:")

    # Generate sample inputs and analyze which dimensions activate
    from src.evaluation.tom_benchmarks import SallyAnneScenario
    scenario_gen = SallyAnneScenario(input_dim=input_dim)

    activations = []
    model.eval()
    with torch.no_grad():
        for order in range(6):
            scenario, _ = scenario_gen.generate_scenario(order)
            output = model(scenario)
            beliefs = output.get('beliefs', torch.zeros(output_dim))
            if beliefs.dim() > 1:
                beliefs = beliefs[0]
            activations.append(beliefs.numpy())

    activations = np.array(activations)

    # Which dimensions vary most across orders?
    dim_variance = np.var(activations, axis=0)
    active_dims = np.where(dim_variance > 0.01)[0]
    top_active = np.argsort(dim_variance)[-20:][::-1]

    analysis['ontology_usage'] = {
        'active_dimensions': len(active_dims),
        'total_dimensions': output_dim,
        'utilization_ratio': len(active_dims) / output_dim,
        'top_active_dims': top_active.tolist(),
        'dimension_variance': dim_variance.tolist()
    }

    if verbose:
        print(f"    Active dimensions: {len(active_dims)}/{output_dim} ({100*len(active_dims)/output_dim:.1f}%)")
        print(f"    Top 10 most active: {top_active[:10].tolist()}")

        # Map to ontology layers (approximate)
        ontology_layers = {
            'Biological (0-15)': sum(1 for d in active_dims if d < 16),
            'Affective (16-38)': sum(1 for d in active_dims if 16 <= d < 39),
            'Motivational (39-54)': sum(1 for d in active_dims if 39 <= d < 55),
            'Cognitive (55-75)': sum(1 for d in active_dims if 55 <= d < 76),
            'Self (76-95)': sum(1 for d in active_dims if 76 <= d < 96),
            'Social Cognition (96-115)': sum(1 for d in active_dims if 96 <= d < 116),
            'Values (116-145)': sum(1 for d in active_dims if 116 <= d < 146),
            'Contextual (146-165)': sum(1 for d in active_dims if 146 <= d < 166),
            'Existential (166-181)': sum(1 for d in active_dims if 166 <= d < 181)
        }

        print(f"\n    Activation by ontology layer:")
        for layer, count in ontology_layers.items():
            bar = "#" * count + "-" * (20 - count)
            print(f"      {layer:25s}: [{bar}] {count}")

    return analysis


# =============================================================================
# COMPREHENSIVE VALIDATION RUNNER
# =============================================================================

def run_full_validation(checkpoint_path: Optional[str] = None,
                        output_file: Optional[str] = None,
                        verbose: bool = True) -> ValidationReport:
    """Run complete validation suite and generate report."""

    if verbose:
        print("\n" + "="*70)
        print("TOM-NAS COMPREHENSIVE VALIDATION SUITE")
        print("="*70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all validations
    start_time = time.time()

    # 1. Baseline tests
    if verbose:
        print("\n[1/5] Running baseline tests...")
    baseline_results = run_baseline_tests(num_trials=50, verbose=verbose)

    # 2. Order 3 analysis
    if verbose:
        print("\n[2/5] Analyzing Order 3 test structure...")
    order_3_analysis = analyze_order_3_test(num_scenarios=50, verbose=verbose)

    # 3. Metacognitive analysis
    if verbose:
        print("\n[3/5] Analyzing metacognitive test (1.000 score investigation)...")
    metacog_analysis = analyze_metacognitive_test(verbose=verbose)

    # 4. Hierarchy tests
    if verbose:
        print("\n[4/5] Testing ToM hierarchy dependencies...")
    hierarchy_results = test_hierarchy_dependency(verbose=verbose)

    # 5. Checkpoint analysis (if provided)
    checkpoint_analysis = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        if verbose:
            print("\n[5/5] Analyzing checkpoint...")
        checkpoint_analysis = characterize_checkpoint(checkpoint_path, verbose=verbose)
    else:
        if verbose:
            print("\n[5/5] No checkpoint provided, skipping...")

    elapsed = time.time() - start_time

    # Generate recommendations
    recommendations = []

    # Check baseline issues
    problem_tests = [k for k, v in baseline_results.items() if not v.is_meaningful]
    if problem_tests:
        recommendations.append(f"Review these tests (baseline issues): {problem_tests}")

    # Check Order 3 issues
    if order_3_analysis.potential_issues:
        recommendations.append(f"Order 3 test has issues: {order_3_analysis.potential_issues}")

    # Check hierarchy
    if not hierarchy_results['hierarchy_valid']:
        recommendations.append("ToM hierarchy appears violated - orders may not be truly hierarchical")

    # Determine overall validity
    critical_issues = len(problem_tests) + len(order_3_analysis.potential_issues)
    if critical_issues > 3:
        overall_validity = "INVALID"
    elif critical_issues > 0 or not hierarchy_results['hierarchy_valid']:
        overall_validity = "NEEDS_REVIEW"
    else:
        overall_validity = "VALID"

    # Create report
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        baseline_results={k: asdict(v) for k, v in baseline_results.items()},
        order_analyses={3: asdict(order_3_analysis)},
        hierarchy_validation=hierarchy_results,
        metacognitive_analysis=metacog_analysis,
        recommendations=recommendations,
        overall_validity=overall_validity
    )

    # Summary
    if verbose:
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"  Time elapsed: {elapsed:.1f} seconds")
        print(f"  Overall validity: {overall_validity}")
        print(f"  Tests with baseline issues: {len(problem_tests)}")
        print(f"  Hierarchy valid: {hierarchy_results['hierarchy_valid']}")

        if recommendations:
            print(f"\n  RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"    - {rec}")

        print("\n" + "="*70)
        if overall_validity == "VALID":
            print("  RESULT: Tests appear valid for publication")
        elif overall_validity == "NEEDS_REVIEW":
            print("  RESULT: Some issues found - review before publication")
        else:
            print("  RESULT: Significant issues - DO NOT use without fixes")
        print("="*70)

    # Save report
    if output_file:
        report_dict = asdict(report) if hasattr(report, '__dataclass_fields__') else {
            'timestamp': report.timestamp,
            'baseline_results': report.baseline_results,
            'order_analyses': report.order_analyses,
            'hierarchy_validation': report.hierarchy_validation,
            'metacognitive_analysis': report.metacognitive_analysis,
            'recommendations': report.recommendations,
            'overall_validity': report.overall_validity
        }
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        if verbose:
            print(f"\nReport saved to: {output_file}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ToM-NAS Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_baselines.py                         # Full validation
  python validate_baselines.py --baselines             # Only baselines
  python validate_baselines.py --analyze-order3        # Only Order 3 analysis
  python validate_baselines.py --checkpoint FILE       # Analyze checkpoint
  python validate_baselines.py --output report.json    # Save report to file
        """
    )

    parser.add_argument('--baselines', action='store_true',
                        help='Run only baseline tests')
    parser.add_argument('--analyze-order3', action='store_true',
                        help='Run only Order 3 analysis')
    parser.add_argument('--analyze-metacog', action='store_true',
                        help='Run only metacognitive test analysis')
    parser.add_argument('--hierarchy', action='store_true',
                        help='Run only hierarchy dependency tests')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for JSON report')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()
    verbose = not args.quiet

    # Run specific tests or full validation
    if args.baselines:
        run_baseline_tests(verbose=verbose)
    elif args.analyze_order3:
        analyze_order_3_test(verbose=verbose)
    elif args.analyze_metacog:
        analyze_metacognitive_test(verbose=verbose)
    elif args.hierarchy:
        test_hierarchy_dependency(verbose=verbose)
    elif args.checkpoint:
        characterize_checkpoint(args.checkpoint, verbose=verbose)
    else:
        # Full validation
        output_file = args.output or f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        run_full_validation(
            checkpoint_path=args.checkpoint,
            output_file=output_file,
            verbose=verbose
        )


if __name__ == "__main__":
    main()
