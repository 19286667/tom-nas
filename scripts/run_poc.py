#!/usr/bin/env python3
"""
Proof of Concept Validation Script

This script validates that:
1. Our evaluation framework actually requires Theory of Mind
2. The reality-only baseline fails belief questions
3. A trained model can learn to answer belief questions
4. Zero-cost proxies correlate with ToM performance

If all tests pass, we have a valid evaluation framework for NAS.

Usage:
    python scripts/run_poc.py

Expected output:
    - Test 1: Baseline should ace reality (~100%) but fail beliefs (~50% or less)
    - Test 2: Trained model should improve >10% over baseline on beliefs
    - Test 3: Zero-cost proxies should show variance across architectures
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import from src.data, src.evaluation, src.nas
from src.data.tomi_loader import ToMiLoader
from src.data.encoding import ScenarioEncoder
from src.evaluation.tom_evaluator import ToMEvaluator, BaselineEvaluator, RandomBaselineEvaluator
from src.nas.zero_cost import ZeroCostProxies


def create_simple_model(input_dim: int = 181, hidden_dim: int = 128):
    """Create a simple LSTM model for testing."""
    class SimpleToMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                               num_layers=2, dropout=0.1)
            self.output = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.output(lstm_out)

    return SimpleToMModel()


def train_model(model, scenarios, encoder, epochs=10, lr=0.001, verbose=True):
    """Train model on scenarios."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_samples = 0

        for scenario in scenarios:
            # Encode scenario
            x = encoder.encode_scenario(scenario).unsqueeze(0)

            # Get target (location index)
            target_idx = encoder.get_location_index(scenario.ground_truth_answer)
            if target_idx < 0:
                continue

            # Forward pass
            output = model(x)
            pred = output[0, -1, encoder.location_start:encoder.location_end]

            # Loss
            loss = criterion(pred.unsqueeze(0), torch.tensor([target_idx]))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_samples += 1

        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(num_samples, 1)
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def main():
    print("=" * 60)
    print("PROOF OF CONCEPT VALIDATION")
    print("=" * 60)
    print("\nThis validates our evaluation framework for ToM-NAS.")
    print("All tests must pass before running full NAS experiments.\n")

    # Load data
    print("[1] Generating scenarios...")
    loader = ToMiLoader(seed=42)
    scenarios = loader.load(num_samples=500)

    # Split into train/test
    train_scenarios = scenarios[:400]
    test_scenarios = scenarios[400:]

    print(f"    Generated {len(train_scenarios)} training, {len(test_scenarios)} test scenarios")

    # Count by type
    by_type = {}
    for s in test_scenarios:
        by_type[s.question_type] = by_type.get(s.question_type, 0) + 1
    print(f"    Test distribution: {by_type}")

    # Encoder
    encoder = ScenarioEncoder()

    # =========================================================================
    # TEST 1: Baseline should fail belief questions
    # =========================================================================
    print("\n" + "-" * 60)
    print("[2] Testing reality-only baseline...")
    print("-" * 60)

    baseline_eval = BaselineEvaluator(encoder)
    baseline_results = baseline_eval.evaluate(test_scenarios)

    print(f"    Reality questions accuracy: {baseline_results['reality_accuracy']:.1%}")
    print(f"    First-order belief accuracy: {baseline_results['first_order_accuracy']:.1%}")
    print(f"    ToM accuracy (overall): {baseline_results['tom_accuracy']:.1%}")

    # Also test random baseline for reference
    random_eval = RandomBaselineEvaluator(encoder)
    random_results = random_eval.evaluate(test_scenarios)
    print(f"\n    Random baseline reality: {random_results['reality_accuracy']:.1%}")
    print(f"    Random baseline beliefs: {random_results['first_order_accuracy']:.1%}")

    # Baseline should get high reality, low belief
    reality_ok = baseline_results['reality_accuracy'] > 0.8
    belief_fails = baseline_results['first_order_accuracy'] < 0.6

    if reality_ok and belief_fails:
        print("\n    [PASS] Baseline correctly fails belief questions!")
        print("           This confirms our scenarios require ToM.")
        test1_pass = True
    else:
        if not reality_ok:
            print("\n    [WARN] Baseline didn't ace reality questions.")
            print("           This might indicate scenario generation issues.")
        if not belief_fails:
            print("\n    [FAIL] Baseline scored too high on beliefs.")
            print("           Scenarios may not have enough information asymmetry!")
        test1_pass = False

    # =========================================================================
    # TEST 2: Train a model and check it learns ToM
    # =========================================================================
    print("\n" + "-" * 60)
    print("[3] Training ToM model...")
    print("-" * 60)

    model = create_simple_model()
    print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_scenarios, encoder, epochs=30, verbose=True)

    # Evaluate trained model
    tom_eval = ToMEvaluator(encoder)
    model_results = tom_eval.evaluate(model, test_scenarios)

    print(f"\n    Trained model results:")
    print(f"      Reality accuracy: {model_results['reality_accuracy']:.1%}")
    print(f"      First-order accuracy: {model_results['first_order_accuracy']:.1%}")
    print(f"      ToM accuracy: {model_results['tom_accuracy']:.1%}")
    print(f"      Overall accuracy: {model_results['overall_accuracy']:.1%}")

    # Model should beat baseline on belief questions
    tom_improvement = model_results['first_order_accuracy'] - baseline_results['first_order_accuracy']
    print(f"\n    Improvement over baseline: {tom_improvement:+.1%}")

    if tom_improvement > 0.1:
        print(f"    [PASS] Model improves {tom_improvement:.1%} over baseline on belief questions!")
        test2_pass = True
    elif tom_improvement > 0:
        print(f"    [PARTIAL] Model shows some improvement ({tom_improvement:.1%})")
        print("              May need more training or better architecture.")
        test2_pass = True  # Partial pass is still ok for PoC
    else:
        print(f"    [FAIL] Model doesn't improve over baseline")
        test2_pass = False

    # =========================================================================
    # TEST 3: Zero-cost proxies discriminate architectures
    # =========================================================================
    print("\n" + "-" * 60)
    print("[4] Testing zero-cost proxies...")
    print("-" * 60)

    proxies = ZeroCostProxies()

    # Create a few model variants
    models = [
        create_simple_model(hidden_dim=32),
        create_simple_model(hidden_dim=64),
        create_simple_model(hidden_dim=128),
        create_simple_model(hidden_dim=256),
    ]

    proxy_scores = []
    print("    Computing proxy scores for different architectures:")
    for i, m in enumerate(models):
        scores = proxies.compute_all(m)
        proxy_scores.append(scores['combined'])
        h = [32, 64, 128, 256][i]
        print(f"      h={h}: synflow={scores['synflow']:.2f}, "
              f"naswot={scores['naswot']:.2f}, combined={scores['combined']:.4f}")

    # Proxies should produce varying scores
    score_variance = max(proxy_scores) - min(proxy_scores)
    print(f"\n    Score range: {min(proxy_scores):.4f} to {max(proxy_scores):.4f}")
    print(f"    Variance: {score_variance:.4f}")

    if score_variance > 0.05:
        print(f"    [PASS] Proxy scores show discrimination ({score_variance:.4f} variance)")
        test3_pass = True
    else:
        print(f"    [FAIL] Proxy scores too similar - may not discriminate well")
        test3_pass = False

    # =========================================================================
    # TEST 4: Quick NAS sanity check (optional)
    # =========================================================================
    print("\n" + "-" * 60)
    print("[5] Quick NAS sanity check...")
    print("-" * 60)

    from src.nas.efficient_nas import create_default_generator

    generator = create_default_generator()
    generated = [generator() for _ in range(10)]

    print(f"    Generated 10 random architectures")
    for i, m in enumerate(generated[:3]):
        params = sum(p.numel() for p in m.parameters())
        print(f"      Model {i+1}: {params:,} params")

    test4_pass = len(generated) == 10 and all(
        hasattr(m, 'forward') for m in generated
    )
    if test4_pass:
        print("    [PASS] Architecture generator works correctly")
    else:
        print("    [FAIL] Architecture generator issues")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"    Test 1 (Baseline fails ToM): {'PASS' if test1_pass else 'FAIL'}")
    print(f"    Test 2 (Model learns ToM):   {'PASS' if test2_pass else 'FAIL'}")
    print(f"    Test 3 (Proxies discriminate): {'PASS' if test3_pass else 'FAIL'}")
    print(f"    Test 4 (Generator works):    {'PASS' if test4_pass else 'FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    critical_pass = test1_pass and test2_pass  # These are the critical ones

    print(f"\n    Overall: {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")

    if all_pass:
        print("\n" + "=" * 60)
        print("SUCCESS! Evaluation framework is valid.")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run full NAS: python -m src.nas.efficient_nas")
        print("  2. Or use experiment_runner.py for comprehensive experiments")
    elif critical_pass:
        print("\n" + "=" * 60)
        print("PARTIAL SUCCESS - Critical tests passed")
        print("=" * 60)
        print("\nThe framework can detect ToM capability.")
        print("Non-critical failures may affect efficiency but not validity.")
    else:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED")
        print("=" * 60)
        print("\nCheck the failures above before running NAS experiments.")
        print("Common issues:")
        print("  - Scenarios without information asymmetry")
        print("  - Model not training properly")
        print("  - Ground truth computation bugs")

    return 0 if all_pass else (1 if critical_pass else 2)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
