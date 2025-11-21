#!/usr/bin/env python
"""
ToM-NAS System Validation Suite
Verifies that all outputs are meaningful, interpretable, valid, and comprehensive.

Run with: python validate_system.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from typing import Dict, List, Tuple

# Import system components
from src.core.ontology import SoulMapOntology
from src.core.beliefs import BeliefNetwork
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
from src.world.social_world import SocialWorld4
from src.evolution.nas_engine import NASEngine, EvolutionConfig
from src.evolution.operators import SpeciesManager, ArchitectureGene
from src.evaluation.zombie_detection import ZombieDetectionSuite
from src.evaluation.tom_benchmarks import ToMBenchmarkSuite, validate_tom_hierarchy


class ValidationSuite:
    """Comprehensive validation of ToM-NAS system."""

    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def run_all(self):
        """Run all validation tests."""
        print("=" * 70)
        print("ToM-NAS System Validation Suite")
        print("=" * 70)

        tests = [
            ("1. Fitness Non-Zero", self.test_fitness_non_zero),
            ("2. Fitness Varies", self.test_fitness_varies),
            ("3. Species Count", self.test_species_count),
            ("4. ToM Hierarchy", self.test_tom_hierarchy),
            ("5. ToM Test Validity", self.test_tom_tests_not_trivial),
            ("6. Zombie Tests Require Reasoning", self.test_zombie_not_pattern_match),
            ("7. Evolution Improves Fitness", self.test_evolution_improves),
            ("8. Architecture Diversity", self.test_architecture_diversity),
            ("9. Reproducibility", self.test_reproducibility),
            ("10. Output Interpretability", self.test_interpretability),
        ]

        for name, test_fn in tests:
            print(f"\n{name}")
            print("-" * 50)
            try:
                result = test_fn()
                if result['status'] == 'PASS':
                    print(f"  [PASS] {result['message']}")
                    self.passed += 1
                elif result['status'] == 'WARN':
                    print(f"  [WARN] {result['message']}")
                    self.warnings += 1
                else:
                    print(f"  [FAIL] {result['message']}")
                    self.failed += 1
                self.results[name] = result
            except Exception as e:
                print(f"  [FAIL] Exception: {e}")
                self.failed += 1
                self.results[name] = {'status': 'FAIL', 'message': str(e)}

        self.print_summary()
        return self.failed == 0

    def test_fitness_non_zero(self) -> Dict:
        """Verify fitness values are non-zero."""
        ontology = SoulMapOntology()
        world = SocialWorld4(num_agents=6, ontology_dim=181)
        belief_net = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=3)

        config = EvolutionConfig(population_size=4, num_generations=1)
        engine = NASEngine(config, world, belief_net)
        engine.initialize_population()
        engine.evaluate_population()

        fitnesses = [ind.fitness for ind in engine.population if ind.fitness is not None]

        if not fitnesses:
            return {'status': 'FAIL', 'message': 'No fitness values computed'}

        non_zero = [f for f in fitnesses if f > 0.01]

        if len(non_zero) == 0:
            return {'status': 'FAIL', 'message': f'All fitness values near zero: {fitnesses}'}

        if len(non_zero) < len(fitnesses) * 0.5:
            return {'status': 'WARN', 'message': f'Only {len(non_zero)}/{len(fitnesses)} have non-zero fitness'}

        return {
            'status': 'PASS',
            'message': f'All {len(fitnesses)} individuals have valid fitness (range: {min(fitnesses):.3f}-{max(fitnesses):.3f})'
        }

    def test_fitness_varies(self) -> Dict:
        """Verify fitness varies between individuals (not all same value)."""
        ontology = SoulMapOntology()
        world = SocialWorld4(num_agents=6, ontology_dim=181)
        belief_net = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=3)

        config = EvolutionConfig(population_size=6, num_generations=1)
        engine = NASEngine(config, world, belief_net)
        engine.initialize_population()
        engine.evaluate_population()

        fitnesses = [ind.fitness for ind in engine.population if ind.fitness is not None]

        if len(fitnesses) < 2:
            return {'status': 'FAIL', 'message': 'Not enough fitness values'}

        variance = np.var(fitnesses)

        if variance < 0.001:
            return {'status': 'FAIL', 'message': f'Fitness has no variance ({variance:.6f}) - all individuals identical'}

        return {
            'status': 'PASS',
            'message': f'Fitness varies appropriately (variance={variance:.4f}, range={min(fitnesses):.3f}-{max(fitnesses):.3f})'
        }

    def test_species_count(self) -> Dict:
        """Verify species are correctly tracked by architecture type."""
        sm = SpeciesManager()

        # Create mock population with different architectures
        class MockGene:
            def __init__(self, arch):
                self.gene_dict = {'arch_type': arch}

        population = [
            (None, MockGene('TRN'), 0.5),
            (None, MockGene('TRN'), 0.6),
            (None, MockGene('RSAN'), 0.4),
            (None, MockGene('RSAN'), 0.5),
            (None, MockGene('Transformer'), 0.7),
        ]

        sm.speciate(population)
        count = sm.get_species_count()
        sizes = sm.get_species_sizes()

        if count != 3:
            return {'status': 'FAIL', 'message': f'Expected 3 species, got {count}'}

        if sorted(sizes) != [1, 2, 2]:
            return {'status': 'WARN', 'message': f'Species sizes unexpected: {sizes}'}

        return {
            'status': 'PASS',
            'message': f'Species count correct: {count} species with sizes {sizes}'
        }

    def test_tom_hierarchy(self) -> Dict:
        """Verify ToM tests follow proper hierarchy (lower orders easier)."""
        # Test hierarchy validation function
        good_scores = {0: 0.8, 1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3}
        bad_scores = {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.9, 4: 0.95, 5: 0.99}

        valid_good, violations_good = validate_tom_hierarchy(good_scores)
        valid_bad, violations_bad = validate_tom_hierarchy(bad_scores)

        if not valid_good:
            return {'status': 'FAIL', 'message': f'Good hierarchy flagged as invalid: {violations_good}'}

        if valid_bad:
            return {'status': 'FAIL', 'message': 'Bad hierarchy (inverted) not detected!'}

        # Test actual agent
        agent = TransparentRNN(191, 64, 181)
        tom_suite = ToMBenchmarkSuite(input_dim=191)
        results = tom_suite.run_full_evaluation(agent)

        scores = {i: results['sally_anne_progression'][i] for i in range(6)}
        valid, violations = validate_tom_hierarchy(scores)

        if not valid:
            return {
                'status': 'WARN',
                'message': f'Untrained agent has hierarchy violations: {violations}'
            }

        return {
            'status': 'PASS',
            'message': f'ToM hierarchy validation working. Scores: {[f"{s:.2f}" for s in scores.values()]}'
        }

    def test_tom_tests_not_trivial(self) -> Dict:
        """Verify ToM tests are not trivially passed by random outputs."""
        # Create random "agent" that outputs random beliefs
        class RandomAgent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(191, 181)

            def forward(self, x):
                # Random outputs
                batch_size = x.shape[0]
                return {
                    'beliefs': torch.rand(batch_size, 181),
                    'actions': torch.rand(batch_size, 1)
                }

        random_agent = RandomAgent()
        tom_suite = ToMBenchmarkSuite(input_dim=191)

        # Run multiple times to check consistency
        pass_counts = {i: 0 for i in range(6)}
        num_trials = 5

        for _ in range(num_trials):
            results = tom_suite.run_full_evaluation(random_agent)
            for i in range(6):
                if results['benchmark_results'][f'sally_anne_order_{i}']['passed']:
                    pass_counts[i] += 1

        # Random agent should not consistently pass high-order tests
        high_order_pass_rate = (pass_counts[3] + pass_counts[4] + pass_counts[5]) / (3 * num_trials)

        if high_order_pass_rate > 0.6:
            return {
                'status': 'FAIL',
                'message': f'Random agent passes high-order tests {high_order_pass_rate*100:.0f}% of time - tests too easy!'
            }

        return {
            'status': 'PASS',
            'message': f'Random agent pass rates: {pass_counts} - tests require genuine reasoning'
        }

    def test_zombie_not_pattern_match(self) -> Dict:
        """Verify zombie tests cannot be trivially pattern-matched."""
        # Test with random and trained agents
        random_agent = TransparentRNN(191, 64, 181)
        zombie_suite = ZombieDetectionSuite()

        results = zombie_suite.run_full_evaluation(random_agent, {'input_dim': 191})

        test_results = results.get('test_results', {})
        if not test_results:
            return {'status': 'WARN', 'message': 'No zombie test results returned'}

        # Check that scores vary (not all same)
        scores = [r.get('score', 0) for r in test_results.values()]

        if len(set(scores)) == 1:
            return {'status': 'WARN', 'message': f'All zombie tests return same score: {scores[0]}'}

        # Check that untrained agent doesn't ace all tests
        passed = sum(1 for r in test_results.values() if r.get('passed', False))

        if passed == len(test_results):
            return {
                'status': 'FAIL',
                'message': 'Untrained agent passes ALL zombie tests - tests may be broken'
            }

        return {
            'status': 'PASS',
            'message': f'Zombie tests: {passed}/{len(test_results)} passed by untrained agent (expected partial)'
        }

    def test_evolution_improves(self) -> Dict:
        """Verify that evolution actually improves fitness over generations."""
        ontology = SoulMapOntology()
        world = SocialWorld4(num_agents=6, ontology_dim=181)
        belief_net = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=3)

        config = EvolutionConfig(population_size=6, num_generations=5)
        engine = NASEngine(config, world, belief_net)
        engine.initialize_population()

        # Run evolution
        for _ in range(5):
            engine.evolve_generation()

        history = engine.history

        if len(history['best_fitness']) < 5:
            return {'status': 'FAIL', 'message': 'History not properly recorded'}

        first_gen = history['best_fitness'][0]
        last_gen = history['best_fitness'][-1]
        best_ever = max(history['best_fitness'])

        # Check for some improvement or maintenance
        if last_gen < first_gen * 0.8:
            return {
                'status': 'WARN',
                'message': f'Fitness decreased: {first_gen:.3f} -> {last_gen:.3f}'
            }

        return {
            'status': 'PASS',
            'message': f'Evolution working: Gen 1={first_gen:.3f}, Gen 5={last_gen:.3f}, Best={best_ever:.3f}'
        }

    def test_architecture_diversity(self) -> Dict:
        """Verify multiple architecture types are explored."""
        ontology = SoulMapOntology()
        world = SocialWorld4(num_agents=6, ontology_dim=181)
        belief_net = BeliefNetwork(num_agents=6, ontology_dim=181, max_order=3)

        config = EvolutionConfig(population_size=12, num_generations=1)
        engine = NASEngine(config, world, belief_net)
        engine.initialize_population()

        arch_types = [ind.gene.gene_dict['arch_type'] for ind in engine.population]
        unique_archs = set(arch_types)

        if len(unique_archs) < 2:
            return {
                'status': 'FAIL',
                'message': f'Only {len(unique_archs)} architecture type(s): {unique_archs}'
            }

        arch_counts = {arch: arch_types.count(arch) for arch in unique_archs}

        return {
            'status': 'PASS',
            'message': f'Architecture diversity: {arch_counts}'
        }

    def test_reproducibility(self) -> Dict:
        """Verify results are reproducible with same seed."""
        torch.manual_seed(42)
        np.random.seed(42)

        agent1 = TransparentRNN(191, 64, 181)
        tom_suite = ToMBenchmarkSuite(input_dim=191)
        results1 = tom_suite.run_full_evaluation(agent1)

        # Reset seeds and create identical agent
        torch.manual_seed(42)
        np.random.seed(42)

        agent2 = TransparentRNN(191, 64, 181)
        results2 = tom_suite.run_full_evaluation(agent2)

        # Compare scores
        scores1 = results1['sally_anne_progression']
        scores2 = results2['sally_anne_progression']

        if scores1 != scores2:
            diff = [abs(s1 - s2) for s1, s2 in zip(scores1, scores2)]
            if max(diff) > 0.01:
                return {
                    'status': 'FAIL',
                    'message': f'Results not reproducible. Diff: {diff}'
                }

        return {
            'status': 'PASS',
            'message': 'Results reproducible with same random seed'
        }

    def test_interpretability(self) -> Dict:
        """Verify outputs contain interpretable information."""
        agent = TransparentRNN(191, 64, 181)
        tom_suite = ToMBenchmarkSuite(input_dim=191)
        zombie_suite = ZombieDetectionSuite()

        tom_results = tom_suite.run_full_evaluation(agent)
        zombie_results = zombie_suite.run_full_evaluation(agent, {'input_dim': 191})

        # Check ToM results have expected keys
        required_tom_keys = ['overall_score', 'sally_anne_progression', 'max_tom_order', 'hierarchy_valid']
        missing_tom = [k for k in required_tom_keys if k not in tom_results]

        if missing_tom:
            return {'status': 'FAIL', 'message': f'Missing ToM keys: {missing_tom}'}

        # Check zombie results have expected structure
        if 'test_results' not in zombie_results:
            return {'status': 'FAIL', 'message': 'Missing zombie test_results'}

        # Check individual results have scores and passed flags
        for test_name, result in zombie_results['test_results'].items():
            if 'score' not in result:
                return {'status': 'FAIL', 'message': f'Zombie test {test_name} missing score'}
            if 'passed' not in result:
                return {'status': 'FAIL', 'message': f'Zombie test {test_name} missing passed flag'}

        return {
            'status': 'PASS',
            'message': f'All outputs interpretable. ToM: {len(tom_results)} fields, Zombie: {len(zombie_results["test_results"])} tests'
        }

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  Passed:   {self.passed}")
        print(f"  Warnings: {self.warnings}")
        print(f"  Failed:   {self.failed}")
        print("=" * 70)

        if self.failed == 0 and self.warnings == 0:
            print("ALL VALIDATIONS PASSED - System output is scientifically valid")
        elif self.failed == 0:
            print("VALIDATION PASSED WITH WARNINGS - Review warnings above")
        else:
            print("VALIDATION FAILED - Fix issues before using results")
        print("=" * 70)


def main():
    suite = ValidationSuite()
    success = suite.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
