"""
Tests for the Evolution/NAS modules.

This module tests the core evolution engine, fitness functions, genetic operators,
and zero-cost proxies used in Neural Architecture Search.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evolution.nas_engine import NASEngine, EvolutionConfig, Individual
from src.evolution.operators import ArchitectureGene, WeightMutation, PopulationOperators
from src.evolution.fitness import (
    ToMFitnessEvaluator, SallyAnneFitness, HigherOrderToMFitness,
    CompositeFitnessFunction
)
from src.evolution.zero_cost_proxies import ZeroCostProxy, ProxyScore, ArchitectureFilter
from src.agents.architectures import TransparentRNN, RecursiveSelfAttention, TransformerToMAgent


class TestEvolutionConfig:
    """Tests for EvolutionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvolutionConfig()

        assert config.population_size == 20
        assert config.num_generations == 100
        assert config.elite_size == 2
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7
        assert config.input_dim == 191
        assert config.output_dim == 181

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvolutionConfig(
            population_size=50,
            num_generations=200,
            mutation_rate=0.2
        )

        assert config.population_size == 50
        assert config.num_generations == 200
        assert config.mutation_rate == 0.2

    def test_convergence_config(self):
        """Test convergence detection configuration."""
        config = EvolutionConfig()

        assert config.convergence_window == 10
        assert config.convergence_threshold == 0.001
        assert config.enable_early_stopping is True


class TestArchitectureGene:
    """Tests for ArchitectureGene genetic encoding."""

    def test_gene_creation(self):
        """Test creating a new gene."""
        gene = ArchitectureGene()

        assert 'arch_type' in gene.gene_dict
        assert 'num_layers' in gene.gene_dict
        assert 'hidden_dim' in gene.gene_dict

    def test_gene_mutation(self):
        """Test gene mutation produces valid offspring."""
        gene = ArchitectureGene()
        gene.gene_dict['arch_type'] = 'TRN'
        gene.gene_dict['num_layers'] = 2
        gene.gene_dict['hidden_dim'] = 128

        # Mutate multiple times to test stability
        for _ in range(10):
            mutated = gene.mutate(mutation_rate=0.5)

            assert isinstance(mutated, ArchitectureGene)
            assert mutated.gene_dict['arch_type'] in ['TRN', 'RSAN', 'Transformer', 'Hybrid']
            assert 1 <= mutated.gene_dict['num_layers'] <= 5
            assert 64 <= mutated.gene_dict['hidden_dim'] <= 512

    def test_gene_crossover(self):
        """Test crossover between two genes."""
        gene1 = ArchitectureGene()
        gene1.gene_dict['arch_type'] = 'TRN'
        gene1.gene_dict['hidden_dim'] = 64

        gene2 = ArchitectureGene()
        gene2.gene_dict['arch_type'] = 'RSAN'
        gene2.gene_dict['hidden_dim'] = 256

        child1, child2 = gene1.crossover(gene2)

        assert isinstance(child1, ArchitectureGene)
        assert isinstance(child2, ArchitectureGene)

        # Children should have values from either parent
        assert child1.gene_dict['arch_type'] in ['TRN', 'RSAN']
        assert child2.gene_dict['arch_type'] in ['TRN', 'RSAN']


class TestIndividual:
    """Tests for Individual class."""

    def test_individual_creation(self):
        """Test creating an individual."""
        model = TransparentRNN(191, 128, 181)
        gene = ArchitectureGene()
        gene.gene_dict['arch_type'] = 'TRN'

        individual = Individual(model, gene, fitness=0.5, generation=1)

        assert individual.model is model
        assert individual.gene is gene
        assert individual.fitness == 0.5
        assert individual.generation == 1

    def test_individual_repr(self):
        """Test individual string representation."""
        model = TransparentRNN(191, 128, 181)
        gene = ArchitectureGene()
        gene.gene_dict['arch_type'] = 'TRN'

        individual = Individual(model, gene, fitness=0.75, generation=2)
        repr_str = repr(individual)

        assert 'TRN' in repr_str
        assert '0.75' in repr_str


class TestWeightMutation:
    """Tests for weight mutation operators."""

    def test_gaussian_noise(self):
        """Test Gaussian noise mutation."""
        model = TransparentRNN(191, 128, 181)

        # Get original weights
        original_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Apply mutation
        mutated = WeightMutation.gaussian_noise(model, noise_std=0.1)

        # Check weights have changed
        weights_changed = False
        for name, param in mutated.named_parameters():
            if not torch.allclose(param, original_weights[name]):
                weights_changed = True
                break

        assert weights_changed


class TestPopulationOperators:
    """Tests for population-level operators."""

    def test_tournament_selection(self):
        """Test tournament selection."""
        ops = PopulationOperators()

        # Create mock population with fitness values
        population = [
            (Mock(), 0.1),
            (Mock(), 0.5),
            (Mock(), 0.9),
            (Mock(), 0.3),
            (Mock(), 0.7)
        ]

        # Run selection multiple times
        selected_fitnesses = []
        for _ in range(100):
            selected = ops.tournament_selection(population, tournament_size=3)
            # Find the fitness of selected model
            for model, fitness in population:
                if model is selected:
                    selected_fitnesses.append(fitness)
                    break

        # Selection should favor higher fitness
        avg_fitness = np.mean(selected_fitnesses)
        assert avg_fitness > 0.4  # Should be above median

    def test_elitism_selection(self):
        """Test elitism selection."""
        ops = PopulationOperators()

        population = [
            (Mock(), 0.1),
            (Mock(), 0.9),
            (Mock(), 0.5),
            (Mock(), 0.3),
            (Mock(), 0.7)
        ]

        elite = ops.elitism_selection(population, elite_size=2)

        assert len(elite) == 2


class TestSallyAnneFitness:
    """Tests for Sally-Anne false belief test."""

    def test_evaluation(self):
        """Test Sally-Anne evaluation."""
        fitness = SallyAnneFitness()
        model = TransparentRNN(191, 128, 181)

        score = fitness.evaluate(model)

        assert 0.0 <= score <= 1.0

    def test_score_range(self):
        """Test score is in valid range for multiple models."""
        fitness = SallyAnneFitness()

        models = [
            TransparentRNN(191, 128, 181),
            RecursiveSelfAttention(191, 128, 181, num_heads=4),
            TransformerToMAgent(191, 128, 181)
        ]

        for model in models:
            score = fitness.evaluate(model)
            assert 0.0 <= score <= 1.0


class TestHigherOrderToMFitness:
    """Tests for higher-order ToM fitness."""

    def test_single_order(self):
        """Test evaluation of single order."""
        fitness = HigherOrderToMFitness(max_order=5)
        model = TransparentRNN(191, 128, 181)

        for order in range(1, 6):
            score = fitness.evaluate_order(model, order)
            assert 0.0 <= score <= 1.0

    def test_all_orders(self):
        """Test evaluation of all orders."""
        fitness = HigherOrderToMFitness(max_order=5)
        model = TransparentRNN(191, 128, 181)

        results = fitness.evaluate_all_orders(model)

        assert len(results) == 5
        for order in range(1, 6):
            assert order in results
            assert 0.0 <= results[order] <= 1.0


class TestCompositeFitnessFunction:
    """Tests for composite fitness evaluation."""

    @pytest.fixture
    def mock_world(self):
        """Create a mock world for testing."""
        world = Mock()
        world.num_agents = 4
        world.get_observation = Mock(return_value={
            'own_resources': 100.0,
            'own_energy': 50.0,
            'own_coalition': None,
            'observations': [
                {
                    'id': 1,
                    'estimated_resources': 80.0,
                    'estimated_energy': 40.0,
                    'reputation': 0.5,
                    'in_same_coalition': False
                }
            ]
        })
        world.step = Mock(return_value={
            'games': [],
            'zombie_detections': [],
            'agent_states': [
                {'resources': 100, 'energy': 50}
            ]
        })
        return world

    @pytest.fixture
    def mock_belief_network(self):
        """Create mock belief network."""
        return Mock()

    def test_evaluation_produces_valid_scores(self, mock_world, mock_belief_network):
        """Test that composite evaluation produces valid scores."""
        fitness = CompositeFitnessFunction(mock_world, mock_belief_network)
        model = TransparentRNN(191, 128, 181)

        results = fitness.evaluate(model, num_episodes=1)

        assert 'total_fitness' in results
        assert 0.0 <= results['total_fitness'] <= 1.0
        assert 'world_fitness' in results
        assert 'sally_anne' in results
        assert 'higher_order_tom' in results


class TestZeroCostProxy:
    """Tests for zero-cost proxy evaluations."""

    def test_synflow_computation(self):
        """Test SynFlow proxy computation."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)
        model = TransparentRNN(181, 64, 181)

        score = proxy.compute_synflow(model)

        assert isinstance(score, float)
        assert score >= 0  # SynFlow should be non-negative

    def test_naswot_computation(self):
        """Test NASWOT proxy computation."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)
        model = TransparentRNN(181, 64, 181)

        score = proxy.compute_naswot(model, num_samples=8)

        assert isinstance(score, float)

    def test_gradnorm_computation(self):
        """Test GradNorm proxy computation."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)
        model = TransparentRNN(181, 64, 181)

        score = proxy.compute_gradnorm(model)

        assert isinstance(score, float)
        assert score >= 0  # Norms are non-negative

    def test_full_evaluation(self):
        """Test full proxy evaluation."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)
        model = TransparentRNN(181, 64, 181)

        result = proxy.evaluate(model)

        assert isinstance(result, ProxyScore)
        assert result.param_count > 0
        assert result.evaluation_time_ms > 0
        assert 0.0 <= result.combined_score <= 1.0

    def test_different_architectures(self):
        """Test proxy evaluation on different architecture types."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)

        architectures = [
            TransparentRNN(181, 64, 181),
            RecursiveSelfAttention(181, 64, 181, num_heads=4),
            TransformerToMAgent(181, 64, 181, num_layers=2)
        ]

        for model in architectures:
            result = proxy.evaluate(model)
            assert isinstance(result, ProxyScore)
            assert 0.0 <= result.combined_score <= 1.0


class TestArchitectureFilter:
    """Tests for architecture filtering."""

    def test_filter_by_params(self):
        """Test filtering architectures by parameter count."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)
        filter = ArchitectureFilter(proxy, param_budget=100000, top_k_ratio=0.5)

        # Create architectures of varying sizes
        architectures = [
            TransparentRNN(181, 32, 181),   # Small
            TransparentRNN(181, 64, 181),   # Medium
            TransparentRNN(181, 256, 181),  # Large
        ]

        results = filter.filter_architectures(architectures, top_k=2)

        # Should return only architectures within budget
        assert len(results) <= 2
        for idx, model, score in results:
            assert score.param_count <= 100000

    def test_statistics(self):
        """Test filter statistics tracking."""
        proxy = ZeroCostProxy(input_dim=181, seq_len=5, batch_size=4)
        filter = ArchitectureFilter(proxy, param_budget=1000000)

        architectures = [TransparentRNN(181, 64, 181) for _ in range(3)]
        filter.filter_architectures(architectures, top_k=2)

        stats = filter.get_statistics()

        assert stats['total_evaluated'] == 3
        assert 'avg_evaluation_time_ms' in stats


class TestNASEngineConvergence:
    """Tests for NAS engine convergence detection."""

    @pytest.fixture
    def mock_world(self):
        """Create mock world."""
        world = Mock()
        world.num_agents = 4
        world.get_observation = Mock(return_value={
            'own_resources': 100.0,
            'own_energy': 50.0,
            'own_coalition': None,
            'observations': []
        })
        world.step = Mock(return_value={
            'games': [],
            'zombie_detections': [],
            'agent_states': []
        })
        return world

    @pytest.fixture
    def mock_belief_network(self):
        """Create mock belief network."""
        return Mock()

    def test_convergence_detection(self, mock_world, mock_belief_network):
        """Test that convergence is detected correctly."""
        config = EvolutionConfig(
            population_size=4,
            convergence_window=3,
            convergence_threshold=0.001,
            enable_early_stopping=True
        )

        engine = NASEngine(config, mock_world, mock_belief_network)

        # Simulate stagnant fitness history
        engine.history['best_fitness'] = [0.5, 0.5, 0.5, 0.5, 0.5]
        engine._stagnant_generations = 3

        engine._check_convergence()

        assert engine.has_converged()

    def test_no_false_convergence(self, mock_world, mock_belief_network):
        """Test that improving fitness doesn't trigger convergence."""
        config = EvolutionConfig(
            population_size=4,
            convergence_window=3,
            convergence_threshold=0.001,
            enable_early_stopping=True
        )

        engine = NASEngine(config, mock_world, mock_belief_network)

        # Simulate improving fitness history
        engine.history['best_fitness'] = [0.3, 0.4, 0.5, 0.6, 0.7]

        engine._check_convergence()

        assert not engine.has_converged()


class TestNASEnginePopulation:
    """Tests for NAS engine population management."""

    @pytest.fixture
    def mock_world(self):
        """Create mock world."""
        world = Mock()
        world.num_agents = 4
        world.get_observation = Mock(return_value={
            'own_resources': 100.0,
            'own_energy': 50.0,
            'own_coalition': None,
            'observations': []
        })
        world.step = Mock(return_value={
            'games': [],
            'zombie_detections': [],
            'agent_states': []
        })
        return world

    @pytest.fixture
    def mock_belief_network(self):
        """Create mock belief network."""
        return Mock()

    def test_population_initialization(self, mock_world, mock_belief_network):
        """Test population initialization creates correct number of individuals."""
        config = EvolutionConfig(population_size=5)
        engine = NASEngine(config, mock_world, mock_belief_network)

        engine.initialize_population()

        assert len(engine.population) == 5

        for individual in engine.population:
            assert isinstance(individual, Individual)
            assert individual.model is not None
            assert individual.gene is not None

    def test_gene_to_model_conversion(self, mock_world, mock_belief_network):
        """Test that genes produce valid models."""
        config = EvolutionConfig()
        engine = NASEngine(config, mock_world, mock_belief_network)

        # Test each architecture type
        for arch_type in ['TRN', 'RSAN', 'Transformer']:
            gene = ArchitectureGene()
            gene.gene_dict['arch_type'] = arch_type
            gene.gene_dict['num_layers'] = 2
            gene.gene_dict['hidden_dim'] = 128
            gene.gene_dict['num_heads'] = 4
            gene.gene_dict['max_recursion'] = 5

            model = engine._gene_to_model(gene)

            assert isinstance(model, nn.Module)

            # Test forward pass
            x = torch.randn(1, 5, 191)
            output = model(x)
            assert 'beliefs' in output
            assert output['beliefs'].shape[-1] == 181


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
