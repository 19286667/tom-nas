#!/usr/bin/env python
"""
Comprehensive Test Suite for ToM-NAS
Tests all major components
"""
import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.ontology import SoulMapOntology
from src.core.beliefs import RecursiveBeliefState, BeliefNetwork
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent
from src.world.social_world import SocialWorld4
from src.evolution.operators import ArchitectureGene, WeightMutation
from src.evaluation.benchmarks import SallyAnneTest, HigherOrderToMBenchmark


class TestOntology(unittest.TestCase):
    """Test Soul Map Ontology"""

    def setUp(self):
        self.ontology = SoulMapOntology()

    def test_initialization(self):
        """Test ontology initializes correctly"""
        self.assertEqual(self.ontology.total_dims, 181)
        self.assertTrue(len(self.ontology.dimensions) > 0)

    def test_default_state(self):
        """Test default state generation"""
        state = self.ontology.get_default_state()
        self.assertEqual(state.shape[0], 181)
        self.assertTrue(torch.all(state >= 0.0))
        self.assertTrue(torch.all(state <= 1.0))

    def test_encoding(self):
        """Test state encoding"""
        test_state = {'bio.vision': 0.8, 'bio.hunger': 0.3}
        encoded = self.ontology.encode(test_state)
        self.assertEqual(encoded.shape[0], 181)


class TestBeliefs(unittest.TestCase):
    """Test Belief System"""

    def setUp(self):
        self.belief_state = RecursiveBeliefState(
            agent_id=0, ontology_dim=181, max_order=5
        )

    def test_belief_creation(self):
        """Test creating beliefs at different orders"""
        content = torch.randn(181)

        for order in range(6):
            self.belief_state.update_belief(
                order=order, target=1, content=content,
                confidence=1.0, source="test"
            )

        # Check beliefs exist
        for order in range(6):
            belief = self.belief_state.get_belief(order, 1)
            if order <= 5:
                self.assertIsNotNone(belief)
                self.assertTrue(belief.confidence > 0)

    def test_confidence_decay(self):
        """Test confidence decays with order"""
        content = torch.randn(181)

        for order in range(4):
            self.belief_state.update_belief(
                order=order, target=1, content=content,
                confidence=1.0
            )

        # Higher orders should have lower confidence
        belief1 = self.belief_state.get_belief(1, 1)
        belief3 = self.belief_state.get_belief(3, 1)

        self.assertGreater(belief1.confidence, belief3.confidence)

    def test_belief_network(self):
        """Test multi-agent belief network"""
        network = BeliefNetwork(num_agents=5, ontology_dim=181, max_order=5)
        self.assertEqual(len(network.agent_beliefs), 5)


class TestArchitectures(unittest.TestCase):
    """Test Agent Architectures"""

    def setUp(self):
        self.input_dim = 191
        self.hidden_dim = 128
        self.output_dim = 181
        self.batch_size = 2
        self.seq_len = 10

    def test_trn_forward(self):
        """Test TRN forward pass"""
        model = TransparentRNN(self.input_dim, self.hidden_dim, self.output_dim)
        test_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        output = model(test_input)

        self.assertIn('beliefs', output)
        self.assertIn('actions', output)
        self.assertEqual(output['beliefs'].shape, (self.batch_size, self.output_dim))

    def test_rsan_forward(self):
        """Test RSAN forward pass"""
        model = RecursiveSelfAttention(self.input_dim, self.hidden_dim,
                                       self.output_dim, num_heads=4)
        test_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        output = model(test_input)

        self.assertIn('beliefs', output)
        self.assertIn('actions', output)
        self.assertEqual(output['beliefs'].shape, (self.batch_size, self.output_dim))

    def test_transformer_forward(self):
        """Test Transformer forward pass"""
        model = TransformerToMAgent(self.input_dim, self.hidden_dim,
                                    self.output_dim, num_layers=3)
        test_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        output = model(test_input)

        self.assertIn('beliefs', output)
        self.assertIn('actions', output)
        self.assertEqual(output['beliefs'].shape, (self.batch_size, self.output_dim))

    def test_output_ranges(self):
        """Test outputs are in valid ranges"""
        model = TransparentRNN(self.input_dim, self.hidden_dim, self.output_dim)
        test_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)

        output = model(test_input)

        # Beliefs should be in [0, 1]
        self.assertTrue(torch.all(output['beliefs'] >= 0.0))
        self.assertTrue(torch.all(output['beliefs'] <= 1.0))

        # Actions should be in [0, 1]
        self.assertTrue(torch.all(output['actions'] >= 0.0))
        self.assertTrue(torch.all(output['actions'] <= 1.0))


class TestSocialWorld(unittest.TestCase):
    """Test Social World 4"""

    def setUp(self):
        self.world = SocialWorld4(num_agents=6, ontology_dim=181, num_zombies=2)

    def test_initialization(self):
        """Test world initializes correctly"""
        self.assertEqual(self.world.num_agents, 6)
        self.assertEqual(len(self.world.agents), 6)

        # Check zombies were created
        zombie_count = sum(1 for agent in self.world.agents if agent.is_zombie)
        self.assertEqual(zombie_count, 2)

    def test_cooperation_game(self):
        """Test cooperation game"""
        result = self.world.play_cooperation_game(0, 1, 'cooperate', 'cooperate')

        self.assertEqual(result['game_type'], 'cooperation')
        self.assertIn('payoffs', result)
        self.assertEqual(len(result['payoffs']), 2)

    def test_resource_sharing(self):
        """Test resource sharing"""
        initial_resources = self.world.agents[0].resources
        result = self.world.play_resource_sharing_game(0, 1, 10.0)

        self.assertEqual(result['game_type'], 'resource_sharing')
        self.assertEqual(result['amount'], 10.0)

        # Giver should have less
        self.assertLess(self.world.agents[0].resources, initial_resources)

    def test_zombie_detection(self):
        """Test zombie detection"""
        # Find a zombie
        zombie_id = None
        for agent in self.world.agents:
            if agent.is_zombie:
                zombie_id = agent.id
                break

        if zombie_id is not None:
            result = self.world.attempt_zombie_detection(0, zombie_id)
            self.assertEqual(result['game_type'], 'zombie_detection')
            self.assertTrue(result['is_zombie'])

    def test_world_step(self):
        """Test world step function"""
        actions = [
            {'type': 'cooperate'} for _ in range(self.world.num_agents)
        ]

        result = self.world.step(actions)

        self.assertIn('timestep', result)
        self.assertIn('games', result)
        self.assertIn('agent_states', result)
        self.assertEqual(result['timestep'], 1)

    def test_coalition_formation(self):
        """Test coalition formation"""
        coalition_id = self.world.form_coalition([0, 1, 2])

        self.assertIn(coalition_id, self.world.coalitions)
        self.assertEqual(len(self.world.coalitions[coalition_id]), 3)

        # Check agents updated
        for agent_id in [0, 1, 2]:
            self.assertEqual(self.world.agents[agent_id].coalition, coalition_id)

    def test_world_reset(self):
        """Test world reset function"""
        # Modify world state
        self.world.timestep = 10
        self.world.agents[0].resources = 50.0
        coalition_id = self.world.form_coalition([0, 1])

        # Reset
        self.world.reset()

        # Check reset state
        self.assertEqual(self.world.timestep, 0)
        self.assertEqual(len(self.world.coalitions), 0)
        self.assertEqual(len(self.world.agents), self.world.num_agents)

        # Check zombies were created
        zombie_count = sum(1 for a in self.world.agents if a.is_zombie)
        self.assertEqual(zombie_count, self.world.num_zombies)


class TestEvolution(unittest.TestCase):
    """Test Evolution Components"""

    def test_architecture_gene_mutation(self):
        """Test gene mutation"""
        gene = ArchitectureGene()
        original_hidden = gene.gene_dict['hidden_dim']

        mutated = gene.mutate(mutation_rate=1.0)  # High rate for testing

        # At least one gene should change
        changed = False
        for key in gene.gene_dict.keys():
            if gene.gene_dict[key] != mutated.gene_dict[key]:
                changed = True
                break

        self.assertTrue(changed)

    def test_architecture_gene_crossover(self):
        """Test gene crossover"""
        gene1 = ArchitectureGene()
        gene1.gene_dict['hidden_dim'] = 128

        gene2 = ArchitectureGene()
        gene2.gene_dict['hidden_dim'] = 256

        child1, child2 = gene1.crossover(gene2)

        # Children should have genes from both parents
        self.assertIsNotNone(child1)
        self.assertIsNotNone(child2)

    def test_weight_mutation(self):
        """Test weight mutation"""
        model = TransparentRNN(191, 128, 181)
        original_param = list(model.parameters())[0].clone()

        mutated = WeightMutation.gaussian_noise(model, noise_std=0.01)

        mutated_param = list(mutated.parameters())[0]

        # Parameters should be different
        self.assertFalse(torch.allclose(original_param, mutated_param))


class TestBenchmarks(unittest.TestCase):
    """Test Benchmark Suite"""

    def setUp(self):
        self.model = TransparentRNN(191, 128, 181)

    def test_sally_anne_basic(self):
        """Test Sally-Anne test"""
        test = SallyAnneTest()
        result = test.run_basic(self.model)

        self.assertIsNotNone(result)
        self.assertEqual(result.test_name, "Sally-Anne Basic")
        self.assertTrue(0.0 <= result.score <= 1.0)

    def test_higher_order_tom(self):
        """Test higher-order ToM"""
        test = HigherOrderToMBenchmark(max_order=5)

        for order in range(1, 6):
            result = test.test_order(self.model, order)
            self.assertTrue(0.0 <= result.score <= 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_full_pipeline(self):
        """Test complete pipeline integration"""
        # Create all components
        ontology = SoulMapOntology()
        beliefs = BeliefNetwork(6, 181, 5)
        world = SocialWorld4(6, 181, 2)
        agent = TransparentRNN(191, 128, 181)

        # Generate data
        obs = world.get_observation(0)
        self.assertIn('own_resources', obs)

        # Forward pass
        test_input = torch.randn(1, 10, 191)
        output = agent(test_input)

        # Check outputs
        self.assertIn('beliefs', output)
        self.assertIn('actions', output)

        # Step world
        actions = [{'type': 'cooperate'} for _ in range(6)]
        result = world.step(actions, beliefs)

        self.assertIn('timestep', result)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestOntology))
    suite.addTests(loader.loadTestsFromTestCase(TestBeliefs))
    suite.addTests(loader.loadTestsFromTestCase(TestArchitectures))
    suite.addTests(loader.loadTestsFromTestCase(TestSocialWorld))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarks))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run:     {result.testsRun}")
    print(f"Successes:     {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print("="*80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
