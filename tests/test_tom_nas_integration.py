"""
Integration Tests for ToM-NAS System

Tests the complete system including:
- Information asymmetry and false belief scenarios
- Benchmark loaders and evaluators
- Supernet elastic architectures
- Social games and ToM reasoning
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.events import (
    create_sally_anne_scenario,
    verify_information_asymmetry,
    InformationAsymmetryTracker,
    EventType,
)
from src.core.beliefs import BeliefNetwork, RecursiveBeliefState
from src.core.ontology import SoulMapOntology
from src.benchmarks import (
    ToMiDataset,
    ToMiEvaluator,
    SocialIQADataset,
    SocialGameBenchmark,
    UnifiedBenchmark,
)
from src.evolution.supernet import (
    ElasticLSTMCell,
    ElasticTransparentRNN,
    ElasticTransformer,
    ZeroCostProxy,
)
from src.liminal import LiminalEnvironment, SoulMap


class TestInformationAsymmetry(unittest.TestCase):
    """Test the core information asymmetry system."""

    def test_sally_anne_scenario_creates_events(self):
        """Test that Sally-Anne scenario creates correct number of events."""
        events, questions = create_sally_anne_scenario()
        self.assertEqual(len(events), 6)  # enter, enter, place, exit, move, enter

    def test_sally_has_false_belief(self):
        """Test that Sally has a false belief about the marble location."""
        results = verify_information_asymmetry()
        self.assertEqual(results['sally_marble_belief'], 'basket')
        self.assertEqual(results['reality'], 'box')
        self.assertTrue(results['sally_has_false_belief'])

    def test_anne_has_true_belief(self):
        """Test that Anne has the correct belief about marble location."""
        results = verify_information_asymmetry()
        self.assertEqual(results['anne_marble_belief'], 'box')
        self.assertTrue(results['anne_has_true_belief'])

    def test_observer_sees_all(self):
        """Test that Observer sees reality correctly."""
        results = verify_information_asymmetry()
        self.assertEqual(results['observer_marble_belief'], 'box')
        self.assertTrue(results['observer_has_true_belief'])

    def test_sally_observed_events(self):
        """Test that Sally observed the correct number of events.

        Sally should see 5 events:
        1. Sally enters
        2. Anne enters
        3. Sally places marble
        4. Sally exits (she's the actor, but doesn't observe from room)
        5. Sally returns

        Sally does NOT see Anne moving the marble (event 5 in sequence).
        """
        events, questions = create_sally_anne_scenario()
        tracker = questions[0]['_tracker']
        sally_beliefs = tracker.agent_beliefs['Sally']

        # Sally observes 5 events (not the move event)
        self.assertEqual(len(sally_beliefs.observed_events), 5)

    def test_anne_observed_events(self):
        """Test that Anne observed all events in the room."""
        events, questions = create_sally_anne_scenario()
        tracker = questions[0]['_tracker']
        anne_beliefs = tracker.agent_beliefs['Anne']

        # Anne observes all 6 events
        self.assertEqual(len(anne_beliefs.observed_events), 6)

    def test_all_verification_tests_pass(self):
        """Verify that all information asymmetry tests pass."""
        results = verify_information_asymmetry()
        self.assertTrue(results['all_tests_passed'])


class TestBeliefNetwork(unittest.TestCase):
    """Test the recursive belief network."""

    def test_belief_network_creation(self):
        """Test creating a belief network."""
        network = BeliefNetwork(num_agents=5, ontology_dim=181, max_order=5)
        self.assertEqual(network.num_agents, 5)
        self.assertEqual(len(network.agent_beliefs), 5)

    def test_belief_update(self):
        """Test updating beliefs in the network."""
        network = BeliefNetwork(num_agents=3, ontology_dim=181, max_order=3)

        # Update agent 0's belief about agent 1
        content = torch.randn(181)
        success = network.update_agent_belief(0, order=1, target=1, content=content)
        self.assertTrue(success)

        # Retrieve the belief
        belief_state = network.get_agent_belief_state(0)
        self.assertIsNotNone(belief_state)

    def test_recursive_belief_confidence_decay(self):
        """Test that higher-order beliefs have lower confidence."""
        belief_state = RecursiveBeliefState(agent_id=0, ontology_dim=181, max_order=5)

        # Add beliefs at different orders
        content = torch.randn(181)
        belief_state.update_belief(order=1, target=1, content=content, confidence=1.0)
        belief_state.update_belief(order=2, target=1, content=content, confidence=1.0)
        belief_state.update_belief(order=3, target=1, content=content, confidence=1.0)

        # Higher order should have lower confidence
        b1 = belief_state.get_belief(1, 1)
        b2 = belief_state.get_belief(2, 1)
        b3 = belief_state.get_belief(3, 1)

        self.assertGreater(b1.confidence, b2.confidence)
        self.assertGreater(b2.confidence, b3.confidence)


class TestOntology(unittest.TestCase):
    """Test the Soul Map ontology."""

    def test_ontology_dimensions(self):
        """Test ontology has correct dimensions."""
        ontology = SoulMapOntology()
        self.assertEqual(ontology.total_dims, 181)

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding states."""
        ontology = SoulMapOntology()

        # Create a state
        state = {'bio.vision': 0.8, 'bio.audition': 0.6}
        encoded = ontology.encode(state)

        self.assertEqual(encoded.shape[0], 181)
        self.assertEqual(encoded[ontology.name_to_idx['bio.vision']], 0.8)

    def test_default_state(self):
        """Test getting default state."""
        ontology = SoulMapOntology()
        default = ontology.get_default_state()
        self.assertEqual(default.shape[0], 181)
        self.assertTrue(torch.all(default == 0.5))


class TestToMiBenchmark(unittest.TestCase):
    """Test the ToMi benchmark loader."""

    def test_tomi_dataset_creation(self):
        """Test creating ToMi dataset with synthetic examples."""
        dataset = ToMiDataset()  # Uses synthetic examples
        self.assertGreater(len(dataset), 0)

    def test_tomi_batch_generation(self):
        """Test generating batches from ToMi dataset."""
        dataset = ToMiDataset()
        events, targets, examples = dataset.get_batch(batch_size=8)

        self.assertEqual(events.shape[0], 8)  # Batch size
        self.assertEqual(targets.shape[0], 8)

    def test_tomi_examples_have_questions(self):
        """Test that ToMi examples have questions."""
        dataset = ToMiDataset()
        example = dataset[0]

        self.assertGreater(len(example.questions), 0)
        self.assertTrue(example.questions[0].requires_tom)


class TestSocialIQABenchmark(unittest.TestCase):
    """Test the SocialIQA benchmark loader."""

    def test_social_iqa_creation(self):
        """Test creating SocialIQA dataset."""
        dataset = SocialIQADataset()
        self.assertGreater(len(dataset), 0)

    def test_social_iqa_batch_generation(self):
        """Test generating batches."""
        dataset = SocialIQADataset()
        inputs, labels = dataset.get_batch(batch_size=4)

        self.assertEqual(inputs['context'].shape[0], 4)
        self.assertEqual(labels.shape[0], 4)

    def test_social_iqa_question_types(self):
        """Test that different question types are present."""
        dataset = SocialIQADataset()

        question_types = set()
        for example in dataset.examples[:50]:
            question_types.add(example.question_type)

        # Should have multiple question types
        self.assertGreater(len(question_types), 1)


class TestSocialGameBenchmark(unittest.TestCase):
    """Test the social game benchmark."""

    def test_social_game_creation(self):
        """Test creating social game benchmark."""
        benchmark = SocialGameBenchmark(num_agents=10, num_zombies=2)
        self.assertEqual(benchmark.num_agents, 10)

    def test_social_world_reset(self):
        """Test resetting the social world."""
        benchmark = SocialGameBenchmark(num_agents=10, num_zombies=2)
        benchmark.reset_world()
        self.assertEqual(benchmark.world.timestep, 0)

    def test_zombie_count(self):
        """Test that correct number of zombies are created."""
        benchmark = SocialGameBenchmark(num_agents=10, num_zombies=3)
        zombies = sum(1 for agent in benchmark.world.agents if agent.is_zombie)
        self.assertEqual(zombies, 3)


class TestElasticSupernet(unittest.TestCase):
    """Test the elastic supernet architectures."""

    def test_elastic_lstm_cell(self):
        """Test elastic LSTM cell with different hidden dims."""
        cell = ElasticLSTMCell(input_dim=64, max_hidden_dim=256)

        x = torch.randn(2, 64)  # Batch of 2, input dim 64
        h = torch.zeros(2, 256)
        c = torch.zeros(2, 256)

        # Test with full hidden dim
        h_new, (h_out, c_out) = cell(x, (h, c), active_hidden_dim=256)
        self.assertEqual(h_new.shape, torch.Size([2, 256]))

        # Test with smaller hidden dim
        h_new, (h_out, c_out) = cell(x, (h, c), active_hidden_dim=128)
        self.assertEqual(h_new.shape, torch.Size([2, 256]))  # Padded back to max

    def test_elastic_trn_forward(self):
        """Test elastic TRN forward pass."""
        model = ElasticTransparentRNN(
            input_dim=64, max_hidden_dim=256, output_dim=181, max_layers=4
        )

        x = torch.randn(2, 10, 64)  # Batch of 2, seq len 10

        # Full configuration
        output = model(x, {'hidden_dim': 256, 'num_layers': 4})
        self.assertEqual(output.shape, torch.Size([2, 181]))

        # Smaller configuration
        output = model(x, {'hidden_dim': 128, 'num_layers': 2})
        self.assertEqual(output.shape, torch.Size([2, 181]))

    def test_elastic_transformer_forward(self):
        """Test elastic transformer forward pass."""
        model = ElasticTransformer(
            input_dim=64, max_hidden_dim=256, output_dim=181,
            max_layers=4, max_heads=8
        )

        x = torch.randn(2, 10, 64)

        # Full configuration
        output = model(x, {'hidden_dim': 256, 'num_layers': 4, 'num_heads': 8})
        self.assertEqual(output.shape, torch.Size([2, 181]))

        # Smaller configuration (must divide evenly)
        output = model(x, {'hidden_dim': 128, 'num_layers': 2, 'num_heads': 4})
        self.assertEqual(output.shape, torch.Size([2, 181]))

    def test_layer_norm_dimension_fix(self):
        """Test that LayerNorm works with variable dimensions.

        This tests the fix for the bug where LayerNorm was initialized
        with max_hidden_dim but received tensors with active_hidden_dim.
        """
        cell = ElasticLSTMCell(input_dim=64, max_hidden_dim=256)

        # This should NOT raise a dimension mismatch error
        x = torch.randn(2, 64)
        h = torch.zeros(2, 256)
        c = torch.zeros(2, 256)

        # Test with 128 (half of max) - this was the bug
        try:
            h_new, _ = cell(x, (h, c), active_hidden_dim=128)
            success = True
        except RuntimeError as e:
            if "normalized_shape" in str(e):
                success = False
            else:
                raise

        self.assertTrue(success, "LayerNorm dimension mismatch not fixed")


class TestZeroCostProxy(unittest.TestCase):
    """Test zero-cost proxy computation."""

    def test_synflow_computation(self):
        """Test SynFlow proxy."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        score = ZeroCostProxy.synflow(model, input_shape=(64,))
        self.assertGreater(score, 0)

    def test_jacob_cov_computation(self):
        """Test Jacobian covariance proxy."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        score = ZeroCostProxy.jacob_cov(model, input_shape=(64,), num_samples=8)
        self.assertGreaterEqual(score, 0)


class TestUnifiedBenchmark(unittest.TestCase):
    """Test the unified benchmark suite."""

    def test_unified_benchmark_creation(self):
        """Test creating unified benchmark."""
        benchmark = UnifiedBenchmark()

        # Check all components are initialized
        self.assertIsNotNone(benchmark.tomi)
        self.assertIsNotNone(benchmark.social_iqa)
        self.assertIsNotNone(benchmark.social_games)

    def test_unified_benchmark_summary(self):
        """Test getting benchmark summary."""
        benchmark = UnifiedBenchmark()
        summary = benchmark.get_benchmark_summary()

        self.assertIn('tomi', summary)
        self.assertIn('social_iqa', summary)
        self.assertIn('social_games', summary)


class TestLiminalEnvironment(unittest.TestCase):
    """Test the Liminal game environment."""

    def test_environment_creation(self):
        """Test creating Liminal environment."""
        env = LiminalEnvironment(population_size=50, include_heroes=True)
        self.assertGreater(len(env.npcs), 0)

    def test_environment_reset(self):
        """Test resetting the environment."""
        env = LiminalEnvironment(population_size=50)
        obs = env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual(env.tick, 0)

    def test_soul_map_creation(self):
        """Test Soul Map creation."""
        soul_map = SoulMap()
        tensor = soul_map.to_tensor()
        self.assertEqual(tensor.shape[0], 65)  # 60 + 5 realm modifiers


class TestIntegration(unittest.TestCase):
    """Full integration tests."""

    def test_end_to_end_evaluation(self):
        """Test evaluating a model on benchmarks end-to-end."""
        # Create a simple model matching benchmark input dimension (64)
        # Model needs to handle sequence input [batch, seq, 64] -> [batch, 181]
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 181)

            def forward(self, x):
                # x: [batch, seq, 64]
                x = torch.relu(self.fc1(x))  # [batch, seq, 128]
                x = x.mean(dim=1)  # Pool over sequence: [batch, 128]
                return self.fc2(x)  # [batch, 181]

        model = SimpleModel()

        # Quick evaluation
        benchmark = UnifiedBenchmark()
        results = benchmark.quick_evaluation(model, device='cpu')

        self.assertIn('tom_score', results)
        self.assertIn('control_score', results)
        self.assertIn('specificity', results)


if __name__ == '__main__':
    unittest.main()
