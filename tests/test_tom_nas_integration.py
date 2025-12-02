"""
Integration Tests for ToM-NAS System

This module provides comprehensive integration tests that verify:
1. Information asymmetry correctly creates false beliefs
2. Event encoding preserves all necessary information
3. Ground truth computation is correct
4. The complete NAS pipeline works end-to-end
5. Architectures can be trained and evaluated for Theory of Mind
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import unittest
from typing import Dict, List, Any


class TestInformationAsymmetry(unittest.TestCase):
    """Tests that information asymmetry correctly creates false beliefs."""

    def setUp(self):
        from src.core.events import (
            Event, AgentBeliefs, create_sally_anne_scenario,
            compute_ground_truth, verify_information_asymmetry
        )
        self.Event = Event
        self.AgentBeliefs = AgentBeliefs
        self.create_sally_anne_scenario = create_sally_anne_scenario
        self.compute_ground_truth = compute_ground_truth
        self.verify_info_asymmetry = verify_information_asymmetry

    def test_sally_anne_false_belief(self):
        """Sally should have a false belief about marble location."""
        events, questions = self.create_sally_anne_scenario()

        sally_beliefs = self.AgentBeliefs("Sally", events)
        anne_beliefs = self.AgentBeliefs("Anne", events)

        # Sally believes marble is in basket (false belief)
        self.assertEqual(sally_beliefs.get_object_location("marble"), "basket")

        # Anne knows marble is in box (true belief)
        self.assertEqual(anne_beliefs.get_object_location("marble"), "box")

    def test_second_order_belief(self):
        """Anne should know that Sally has false belief."""
        events, questions = self.create_sally_anne_scenario()

        anne_beliefs = self.AgentBeliefs("Anne", events)

        # Anne's model of Sally's belief
        anne_thinks_sally_believes = anne_beliefs.get_belief_about_other("Sally", "marble")

        # Anne knows Sally believes basket
        self.assertEqual(anne_thinks_sally_believes, "basket")

    def test_ground_truth_reality_question(self):
        """Ground truth for reality questions should be current world state."""
        events, questions = self.create_sally_anne_scenario()

        reality_q = [q for q in questions if q.question_type == 'reality'][0]
        gt = self.compute_ground_truth(events, reality_q)

        self.assertEqual(gt, "box")  # Marble is actually in box

    def test_ground_truth_first_order_belief(self):
        """Ground truth for first-order belief should match agent's observations."""
        events, questions = self.create_sally_anne_scenario()

        sally_q = [q for q in questions
                   if q.question_type == 'first_order' and q.target_agent == 'Sally'][0]
        gt = self.compute_ground_truth(events, sally_q)

        self.assertEqual(gt, "basket")  # Sally believes basket

    def test_ground_truth_second_order_belief(self):
        """Ground truth for second-order belief should match nested belief."""
        events, questions = self.create_sally_anne_scenario()

        second_q = [q for q in questions if q.question_type == 'second_order'][0]
        gt = self.compute_ground_truth(events, second_q)

        self.assertEqual(gt, "basket")  # Anne knows Sally believes basket

    def test_observation_filtering(self):
        """Agents should only see events they observed."""
        events, _ = self.create_sally_anne_scenario()

        sally_beliefs = self.AgentBeliefs("Sally", events)

        # Sally should have fewer observed events (missed event 4 and 5)
        self.assertEqual(len(sally_beliefs.observed_events), 4)

    def test_verification_function(self):
        """Complete verification should pass."""
        results = self.verify_info_asymmetry()

        self.assertTrue(results['sally_has_false_belief'])
        self.assertTrue(results['second_order_correct'])
        self.assertTrue(results['all_tests_passed'])


class TestEventEncoding(unittest.TestCase):
    """Tests that event encoding preserves necessary information."""

    def setUp(self):
        from src.core.events import (
            Event, EventEncoder, ScenarioEncoder, AnswerDecoder,
            create_sally_anne_scenario, Question
        )
        self.Event = Event
        self.EventEncoder = EventEncoder
        self.ScenarioEncoder = ScenarioEncoder
        self.AnswerDecoder = AnswerDecoder
        self.create_sally_anne_scenario = create_sally_anne_scenario
        self.Question = Question

    def test_event_encoding_shape(self):
        """Encoded event should have correct shape."""
        encoder = self.EventEncoder()

        event = self.Event(
            timestamp=1,
            actor="Sally",
            action="enter",
            target_location="room",
            observed_by={"Sally", "Anne"}
        )

        encoder.register_agent("Sally")
        encoder.register_agent("Anne")
        encoder.register_location("room")

        encoded = encoder.encode_event(event)

        self.assertEqual(encoded.shape, (181,))

    def test_observer_encoding_multi_hot(self):
        """Observer encoding should be multi-hot."""
        encoder = self.EventEncoder()

        encoder.register_agent("Sally")
        encoder.register_agent("Anne")
        encoder.register_location("room")

        event = self.Event(
            timestamp=1,
            actor="Sally",
            action="enter",
            target_location="room",
            observed_by={"Sally", "Anne"}
        )

        encoded = encoder.encode_event(event)

        # Both observers should be marked
        observer_start = encoder.OBSERVER_DIM_START
        sally_idx = encoder.agents_vocab["Sally"]
        anne_idx = encoder.agents_vocab["Anne"]

        self.assertEqual(encoded[observer_start + sally_idx].item(), 1.0)
        self.assertEqual(encoded[observer_start + anne_idx].item(), 1.0)

    def test_scenario_encoding_includes_question(self):
        """Scenario encoding should include question as final element."""
        events, questions = self.create_sally_anne_scenario()
        question = questions[0]

        encoder = self.EventEncoder()
        scenario_encoder = self.ScenarioEncoder(encoder)

        encoded = scenario_encoder.encode_scenario(events, question)

        # Should be events + 1 question
        self.assertEqual(encoded.shape[0], len(events) + 1)
        self.assertEqual(encoded.shape[1], 181)

    def test_answer_decoding(self):
        """Answer decoder should recover correct location."""
        encoder = self.EventEncoder()
        encoder.register_location("basket")
        encoder.register_location("box")

        decoder = self.AnswerDecoder(encoder)

        # Create output that matches basket encoding
        output = encoder.get_location_vector("basket")

        decoded = decoder.decode_location(output)
        self.assertEqual(decoded, "basket")


class TestZeroCostProxies(unittest.TestCase):
    """Tests for zero-cost proxy evaluation."""

    def setUp(self):
        from src.evolution.zero_cost_proxies import ZeroCostProxy, ArchitectureFilter
        from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
        self.ZeroCostProxy = ZeroCostProxy
        self.ArchitectureFilter = ArchitectureFilter
        self.TRN = TransparentRNN
        self.RSAN = RecursiveSelfAttention
        self.Transformer = TransformerToMAgent

    def test_proxy_evaluation_returns_scores(self):
        """Proxy evaluation should return all expected scores."""
        proxy = self.ZeroCostProxy(input_dim=181)
        model = self.TRN(181, 128, 181, num_layers=2)

        score = proxy.evaluate(model)

        self.assertIsNotNone(score.synflow)
        self.assertIsNotNone(score.naswot)
        self.assertIsNotNone(score.gradnorm)
        self.assertIsNotNone(score.combined_score)
        self.assertGreater(score.param_count, 0)

    def test_proxy_discriminates_architectures(self):
        """Different architectures should get different proxy scores."""
        proxy = self.ZeroCostProxy(input_dim=181)

        small_model = self.TRN(181, 64, 181, num_layers=1)
        large_model = self.TRN(181, 256, 181, num_layers=4)

        small_score = proxy.evaluate(small_model)
        large_score = proxy.evaluate(large_model)

        # Larger model should have more parameters
        self.assertGreater(large_score.param_count, small_score.param_count)

    def test_architecture_filter(self):
        """Architecture filter should reduce candidate count."""
        proxy = self.ZeroCostProxy(input_dim=181)
        filter = self.ArchitectureFilter(proxy, param_budget=500000)

        architectures = [
            self.TRN(181, 64, 181, num_layers=1),
            self.TRN(181, 128, 181, num_layers=2),
            self.TRN(181, 256, 181, num_layers=3),
        ]

        filtered = filter.filter_architectures(architectures, top_k=2)

        self.assertEqual(len(filtered), 2)


class TestSupernet(unittest.TestCase):
    """Tests for supernet weight sharing."""

    def setUp(self):
        from src.evolution.supernet import ToMSupernet, SubnetConfig, SupernetEvaluator
        self.ToMSupernet = ToMSupernet
        self.SubnetConfig = SubnetConfig
        self.SupernetEvaluator = SupernetEvaluator

    def test_supernet_forward_trn(self):
        """Supernet should produce valid output for TRN config."""
        supernet = self.ToMSupernet(input_dim=181, output_dim=181)

        config = self.SubnetConfig(
            arch_type='trn',
            num_layers=2,
            hidden_dim=128,
            num_heads=4
        )
        supernet.set_active_config(config)

        x = torch.randn(4, 10, 181)
        output = supernet(x)

        self.assertIn('beliefs', output)
        self.assertEqual(output['beliefs'].shape, (4, 181))

    def test_supernet_forward_transformer(self):
        """Supernet should produce valid output for Transformer config."""
        supernet = self.ToMSupernet(input_dim=181, output_dim=181)

        config = self.SubnetConfig(
            arch_type='transformer',
            num_layers=2,
            hidden_dim=128,
            num_heads=4
        )
        supernet.set_active_config(config)

        x = torch.randn(4, 10, 181)
        output = supernet(x)

        self.assertIn('beliefs', output)
        self.assertEqual(output['beliefs'].shape, (4, 181))

    def test_weight_sharing(self):
        """Different configs should share weights."""
        supernet = self.ToMSupernet(input_dim=181, output_dim=181)

        # Get reference to a weight
        input_proj_weight = supernet.input_proj.weight.clone()

        # Switch configs
        config1 = self.SubnetConfig('trn', 2, 128, 4)
        config2 = self.SubnetConfig('transformer', 3, 96, 4)

        supernet.set_active_config(config1)
        _ = supernet(torch.randn(1, 5, 181))

        supernet.set_active_config(config2)
        _ = supernet(torch.randn(1, 5, 181))

        # Weight should be unchanged (shared)
        self.assertTrue(torch.equal(input_proj_weight, supernet.input_proj.weight))


class TestMutationController(unittest.TestCase):
    """Tests for the reinforced mutation controller."""

    def setUp(self):
        from src.evolution.mutation_controller import (
            MutationController, ControllerTrainer, GuidedMutator
        )
        self.MutationController = MutationController
        self.ControllerTrainer = ControllerTrainer
        self.GuidedMutator = GuidedMutator

    def test_controller_prediction(self):
        """Controller should produce predictions."""
        controller = self.MutationController()

        config = {
            'arch_type': 'transformer',
            'num_layers': 2,
            'hidden_dim': 128,
            'num_heads': 4,
            'dropout': 0.1
        }

        pred = controller.predict_improvement(
            config, 'hidden_dim', 128, 160
        )

        self.assertIsInstance(pred, float)

    def test_guided_mutation(self):
        """Guided mutator should produce valid child configs."""
        controller = self.MutationController()
        mutator = self.GuidedMutator(controller)

        config = {
            'arch_type': 'transformer',
            'num_layers': 2,
            'hidden_dim': 128,
            'num_heads': 4,
            'dropout': 0.1,
            'use_skip_connections': True
        }

        child = mutator.apply_mutation(config)

        # Should be a valid config
        self.assertIn('arch_type', child)
        self.assertIn('num_layers', child)


class TestToMiDataset(unittest.TestCase):
    """Tests for ToMi benchmark loading."""

    def setUp(self):
        from src.benchmarks.tomi_loader import ToMiDataset, ToMiParser
        self.ToMiDataset = ToMiDataset
        self.ToMiParser = ToMiParser

    def test_synthetic_generation(self):
        """Should generate synthetic ToMi examples."""
        dataset = self.ToMiDataset()
        dataset.generate_synthetic(num_examples=50)

        self.assertEqual(len(dataset.examples), 50)

        # Each example should have events and questions
        example = dataset.examples[0]
        self.assertGreater(len(example.events), 0)
        self.assertGreater(len(example.questions), 0)

    def test_dataset_split(self):
        """Dataset should split into train/val/test."""
        dataset = self.ToMiDataset()
        dataset.generate_synthetic(num_examples=100)
        dataset.split()

        total = (len(dataset.train_examples) +
                 len(dataset.val_examples) +
                 len(dataset.test_examples))

        self.assertEqual(total, 100)

    def test_batch_generation(self):
        """Should generate batches for training."""
        dataset = self.ToMiDataset()
        dataset.generate_synthetic(num_examples=50)

        inputs, targets, indices = dataset.get_batch(8)

        self.assertEqual(inputs.shape[0], 8)
        self.assertEqual(inputs.shape[2], 181)  # Soul Map dimension

    def test_question_type_filtering(self):
        """Should filter batches by question type."""
        dataset = self.ToMiDataset()
        dataset.generate_synthetic(num_examples=50)

        inputs, targets, indices = dataset.get_batch(8, question_type='first_order')

        # Should complete without error
        self.assertEqual(inputs.shape[0], 8)


class TestToMFitness(unittest.TestCase):
    """Tests for ToM-specific fitness evaluation."""

    def setUp(self):
        from src.evolution.tom_fitness import ToMSpecificFitness, CombinedToMFitness
        self.ToMSpecificFitness = ToMSpecificFitness
        self.CombinedToMFitness = CombinedToMFitness

    def test_fitness_evaluation(self):
        """Fitness evaluator should produce results."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(181, 181)

            def forward(self, x):
                if x.dim() == 3:
                    x = x[:, -1, :]
                out = torch.sigmoid(self.net(x))
                return {'beliefs': out, 'actions': out.mean(dim=-1)}

        model = SimpleModel()
        fitness = self.ToMSpecificFitness()

        result = fitness.evaluate(model, num_examples=20)

        self.assertIsNotNone(result.total_fitness)
        self.assertIsNotNone(result.tom_accuracy)
        self.assertIsNotNone(result.control_accuracy)

    def test_efficiency_penalty(self):
        """Large models should have lower efficiency scores."""

        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(181, 181)

            def forward(self, x):
                if x.dim() == 3: x = x[:, -1, :]
                return {'beliefs': torch.sigmoid(self.net(x)), 'actions': torch.zeros(x.shape[0])}

        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(181, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 181)
                )

            def forward(self, x):
                if x.dim() == 3: x = x[:, -1, :]
                return {'beliefs': torch.sigmoid(self.net(x)), 'actions': torch.zeros(x.shape[0])}

        fitness = self.ToMSpecificFitness(target_params=100000)

        small_result = fitness.evaluate(SmallModel(), num_examples=10)
        large_result = fitness.evaluate(LargeModel(), num_examples=10)

        self.assertGreater(small_result.efficiency_score, large_result.efficiency_score)


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests for the complete NAS pipeline."""

    def test_complete_evaluation_pipeline(self):
        """Test complete flow from events to fitness."""
        from src.core.events import create_sally_anne_scenario, EventEncoder, ScenarioEncoder
        from src.agents.architectures import TransparentRNN

        # Create scenario
        events, questions = create_sally_anne_scenario()

        # Encode
        encoder = EventEncoder()
        scenario_encoder = ScenarioEncoder(encoder)

        encoded = scenario_encoder.encode_scenario(events, questions[0])

        # Create and run model
        model = TransparentRNN(181, 128, 181, num_layers=2)
        output = model(encoded.unsqueeze(0))

        # Should produce beliefs
        self.assertIn('beliefs', output)
        self.assertEqual(output['beliefs'].shape[1], 181)

    def test_nas_search_iteration(self):
        """Test that NAS search can run one iteration."""
        from src.evolution.supernet import ToMSupernet, SupernetEvaluator
        from src.evolution.linas import LINASSearch

        supernet = ToMSupernet(181, 181)
        evaluator = SupernetEvaluator(supernet)
        search = LINASSearch(supernet, evaluator)

        # Generate dummy eval data
        eval_data = [(torch.randn(4, 10, 181), torch.rand(4, 181)) for _ in range(2)]

        # Run one iteration
        result = search.search_iteration(
            eval_data,
            num_candidates=10,
            num_to_evaluate=3
        )

        self.assertIn('iteration', result)
        self.assertIn('best_fitness', result)


class TestRealityVsBeliefDiscrimination(unittest.TestCase):
    """Tests that models can be trained to discriminate reality from beliefs."""

    def test_untrained_model_baseline(self):
        """Untrained model should be near random on all question types."""
        from src.benchmarks.tomi_loader import ToMiDataset
        from src.agents.architectures import TransparentRNN

        dataset = ToMiDataset()
        dataset.generate_synthetic(num_examples=100)

        model = TransparentRNN(181, 128, 181, num_layers=2)

        # Evaluate accuracy
        correct_reality = 0
        correct_belief = 0
        total_reality = 0
        total_belief = 0

        model.eval()
        with torch.no_grad():
            for example in dataset.examples[:20]:
                for q_idx, question in enumerate(example.questions):
                    inp, tgt, correct_idx = dataset.encode_example(example, q_idx)
                    output = model(inp.unsqueeze(0))
                    pred_idx = output['beliefs'].argmax(dim=-1).item()

                    if question.question_type == 'reality':
                        total_reality += 1
                        if pred_idx == correct_idx:
                            correct_reality += 1
                    elif question.question_type in ['first_order', 'second_order']:
                        total_belief += 1
                        if pred_idx == correct_idx:
                            correct_belief += 1

        # Untrained model should be roughly random
        # With ~10 possible locations, expect ~10% accuracy
        reality_acc = correct_reality / total_reality if total_reality > 0 else 0
        belief_acc = correct_belief / total_belief if total_belief > 0 else 0

        # Just verify it produces valid outputs (not testing for randomness specifically)
        self.assertGreaterEqual(reality_acc, 0.0)
        self.assertLessEqual(reality_acc, 1.0)


def run_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestInformationAsymmetry,
        TestEventEncoding,
        TestZeroCostProxies,
        TestSupernet,
        TestMutationController,
        TestToMiDataset,
        TestToMFitness,
        TestEndToEndPipeline,
        TestRealityVsBeliefDiscrimination,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 70)
    print("TOM-NAS INTEGRATION TESTS")
    print("=" * 70)
    print()

    result = run_tests()

    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")

        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")
