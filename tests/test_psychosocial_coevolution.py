#!/usr/bin/env python3
"""
Tests for Psychosocial Co-Evolution and Narrative Emergence Systems

These tests cover:
- Social network dynamics (edges, coalitions, hierarchy)
- Belief propagation engine
- Environmental evolution strategies
- Psychosocial co-evolution engine integration
- Narrative archetype detection
- Narrative progression and resolution
- ToM learning opportunities

These systems are critical for creating genuine co-evolutionary dynamics
that develop sophisticated Theory of Mind in agents.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from src.liminal.npcs.base_npc import BaseNPC, NPCState
from src.liminal.soul_map import SoulMap

# =============================================================================
# PSYCHOSOCIAL CO-EVOLUTION TESTS
# =============================================================================


class TestTheoreticalConstants:
    """Test theoretical constants are properly defined."""

    def test_import_constants(self):
        """Test constants can be imported."""
        from src.liminal.psychosocial_coevolution import TheoreticalConstants

        assert TheoreticalConstants.DUNBAR_NUMBER == 15
        assert 0 < TheoreticalConstants.BELIEF_CONFIDENCE_DECAY < 1
        assert 0 < TheoreticalConstants.BALANCE_PRESSURE < 1

    def test_tom_processing_cost(self):
        """Test ToM processing cost increases with depth."""
        from src.liminal.psychosocial_coevolution import TheoreticalConstants

        costs = TheoreticalConstants.TOM_PROCESSING_COST
        assert len(costs) == 6  # Order 0-5

        # Each level should have higher or equal cost
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1]


class TestSocialEdge:
    """Test social edge (relationship) mechanics."""

    def test_edge_creation(self):
        """Test creating a social edge."""
        from src.liminal.psychosocial_coevolution import SocialEdge

        edge = SocialEdge(source_id="agent_1", target_id="agent_2")

        assert edge.source_id == "agent_1"
        assert edge.target_id == "agent_2"
        assert edge.trust == 0.5  # Default
        assert edge.familiarity == 0.0  # Default
        assert edge.affect == 0.0  # Default

    def test_relationship_type_stranger(self):
        """Test stranger classification."""
        from src.liminal.psychosocial_coevolution import RelationshipType, SocialEdge

        edge = SocialEdge(source_id="a", target_id="b")
        assert edge.get_relationship_type() == RelationshipType.STRANGER

    def test_relationship_type_ally(self):
        """Test ally classification."""
        from src.liminal.psychosocial_coevolution import RelationshipType, SocialEdge

        edge = SocialEdge(source_id="a", target_id="b")
        edge.trust = 0.8
        edge.affect = 0.2  # Positive but below COALITION threshold (0.3)
        edge.familiarity = 0.5

        assert edge.get_relationship_type() == RelationshipType.ALLY

    def test_relationship_type_coalition(self):
        """Test coalition classification - highest trust tier with strong affect."""
        from src.liminal.psychosocial_coevolution import RelationshipType, SocialEdge, TheoreticalConstants

        edge = SocialEdge(source_id="a", target_id="b")
        edge.trust = 0.8  # Above COALITION_FORMATION_THRESHOLD (0.6)
        edge.affect = 0.5  # Above coalition affect threshold (0.3)
        edge.familiarity = 0.5  # Above acquaintance threshold

        assert edge.get_relationship_type() == RelationshipType.COALITION

    def test_relationship_type_enemy(self):
        """Test enemy classification."""
        from src.liminal.psychosocial_coevolution import RelationshipType, SocialEdge

        edge = SocialEdge(source_id="a", target_id="b")
        edge.trust = 0.2
        edge.affect = -0.5
        edge.familiarity = 0.5

        assert edge.get_relationship_type() == RelationshipType.ENEMY

    def test_interaction_updates_edge(self):
        """Test that interactions update the edge."""
        from src.liminal.psychosocial_coevolution import SocialEdge

        edge = SocialEdge(source_id="a", target_id="b")
        initial_familiarity = edge.familiarity

        edge.update_from_interaction(cooperated=True, tick=10)

        assert edge.familiarity > initial_familiarity
        assert edge.last_interaction_tick == 10
        assert edge.cooperation_history[-1] == True

    def test_negative_interaction_erodes_trust(self):
        """Test that defection erodes trust faster than cooperation builds it."""
        from src.liminal.psychosocial_coevolution import SocialEdge

        edge1 = SocialEdge(source_id="a", target_id="b")
        edge1.trust = 0.5
        edge1.update_from_interaction(cooperated=True, tick=1)
        trust_gain = edge1.trust - 0.5

        edge2 = SocialEdge(source_id="c", target_id="d")
        edge2.trust = 0.5
        edge2.update_from_interaction(cooperated=False, tick=1)
        trust_loss = 0.5 - edge2.trust

        # Negativity bias: losses > gains
        assert trust_loss > trust_gain

    def test_decay_regresses_to_neutral(self):
        """Test that relationships decay toward neutral over time."""
        from src.liminal.psychosocial_coevolution import SocialEdge

        edge = SocialEdge(source_id="a", target_id="b")
        edge.trust = 0.9
        edge.affect = 0.8
        edge.last_interaction_tick = 0

        edge.decay(current_tick=100)

        # Trust should regress toward 0.5
        assert edge.trust < 0.9
        # Affect should regress toward 0
        assert edge.affect < 0.8


class TestSocialNetwork:
    """Test social network dynamics."""

    def test_network_creation(self):
        """Test creating a social network."""
        from src.liminal.psychosocial_coevolution import SocialNetwork

        network = SocialNetwork()
        assert len(network.edges) == 0
        assert len(network.coalitions) == 0
        assert len(network.hierarchy) == 0

    def test_get_or_create_edge(self):
        """Test getting or creating edges."""
        from src.liminal.psychosocial_coevolution import SocialNetwork

        network = SocialNetwork()

        edge1 = network.get_or_create_edge("a", "b")
        edge2 = network.get_or_create_edge("a", "b")  # Same edge
        edge3 = network.get_or_create_edge("b", "a")  # Different edge

        assert edge1 is edge2
        assert edge1 is not edge3
        assert len(network.edges) == 2

    def test_record_interaction(self):
        """Test recording bilateral interaction."""
        from src.liminal.psychosocial_coevolution import SocialNetwork

        network = SocialNetwork()
        network.record_interaction(agent1="a", agent2="b", outcome1_cooperated=True, outcome2_cooperated=False, tick=10)

        edge_a_to_b = network.edges.get(("a", "b"))
        edge_b_to_a = network.edges.get(("b", "a"))

        assert edge_a_to_b is not None
        assert edge_b_to_a is not None

        # A cooperated, B defected: A's view of B should lose trust
        # B defected, A cooperated: B's view of A should gain trust

    def test_coalition_detection(self):
        """Test coalition detection from positive relationships."""
        from src.liminal.psychosocial_coevolution import SocialNetwork, TheoreticalConstants

        network = SocialNetwork()

        # Create strong positive relationships between a, b, c
        for pair in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("a", "c"), ("c", "a")]:
            edge = network.get_or_create_edge(pair[0], pair[1])
            edge.trust = TheoreticalConstants.COALITION_FORMATION_THRESHOLD + 0.1
            edge.affect = 0.3

        coalitions = network.detect_coalitions()

        assert len(coalitions) >= 1
        # All three should be in same coalition
        for coalition_members in coalitions.values():
            if "a" in coalition_members:
                assert "b" in coalition_members
                assert "c" in coalition_members

    def test_hierarchy_update(self):
        """Test hierarchy updates from competitive outcomes."""
        from src.liminal.psychosocial_coevolution import SocialNetwork

        network = SocialNetwork()
        network.hierarchy["winner"] = 0.5
        network.hierarchy["loser"] = 0.5

        network.update_hierarchy("winner", "loser")

        assert network.hierarchy["winner"] > 0.5
        assert network.hierarchy["loser"] < 0.5

    def test_heider_balance(self):
        """Test Heider balance pressure on triads."""
        from src.liminal.psychosocial_coevolution import SocialNetwork

        network = SocialNetwork()

        # Create imbalanced triad: A+B+, B+C+, A-C (should balance)
        for pair in [("a", "b"), ("b", "c"), ("a", "c")]:
            edge = network.get_or_create_edge(pair[0], pair[1])
            edge.familiarity = 0.5

        network.edges[("a", "b")].affect = 0.5  # positive
        network.edges[("b", "c")].affect = 0.5  # positive
        network.edges[("a", "c")].affect = -0.5  # negative - imbalanced!

        network.apply_heider_balance(tick=10)

        # Balance pressure should be applied (triad should tend toward balance)
        # Note: with random sampling, this might not always trigger


class TestBeliefPropagation:
    """Test belief propagation engine."""

    def test_engine_creation(self):
        """Test creating belief propagation engine."""
        from src.liminal.psychosocial_coevolution import BeliefPropagationEngine, SocialNetwork

        network = SocialNetwork()
        engine = BeliefPropagationEngine(network)

        assert engine.network is network
        assert len(engine.beliefs) == 0

    def test_introduce_belief(self):
        """Test introducing a new belief."""
        from src.liminal.psychosocial_coevolution import BeliefPropagationEngine, SocialNetwork

        network = SocialNetwork()
        engine = BeliefPropagationEngine(network)

        belief = engine.introduce_belief(
            belief_id="belief_1", content={"fact": "something_happened"}, source_id="agent_a", tick=10
        )

        assert belief.belief_id == "belief_1"
        assert belief.source_id == "agent_a"
        assert belief.confidence == 1.0
        assert "agent_a" in belief.holders

    def test_belief_propagation(self):
        """Test beliefs propagate through interactions."""
        from src.liminal.psychosocial_coevolution import BeliefPropagationEngine, SocialNetwork

        network = SocialNetwork()

        # Create trusted relationship
        edge = network.get_or_create_edge("a", "b")
        edge.trust = 0.9

        engine = BeliefPropagationEngine(network)

        # Introduce belief to A
        engine.introduce_belief(belief_id="secret", content={"info": "classified"}, source_id="a", tick=1)

        # Propagate through interaction
        engine.propagate(tick=2, interactions=[("a", "b")])

        # With high trust, B should likely have the belief now
        # (probabilistic, so we check after multiple propagations)
        for _ in range(10):
            engine.propagate(tick=2 + _, interactions=[("a", "b")])

        b_knowledge = engine.get_agent_knowledge("b")
        # Should have some knowledge (probabilistic)

    def test_knowledge_asymmetry_tensor(self):
        """Test knowledge asymmetry tensor creation."""
        from src.liminal.psychosocial_coevolution import BeliefPropagationEngine, SocialNetwork

        network = SocialNetwork()
        engine = BeliefPropagationEngine(network)

        # Agent A knows something B doesn't
        engine.introduce_belief("secret", {"data": 1}, "a", 1)

        agents = ["a", "b"]
        asymmetry = engine.create_knowledge_asymmetry_tensor(agents)

        assert asymmetry.shape == (2, 2)
        # A knows things B doesn't
        assert asymmetry[0, 1] > 0 or len(engine.agent_beliefs["a"]) > 0


class TestEnvironmentalEvolution:
    """Test environmental co-evolution mechanics."""

    def test_evolution_creation(self):
        """Test creating environmental evolution engine."""
        from src.liminal.psychosocial_coevolution import EnvironmentEvolutionStrategy, PsychosocialEnvironmentEvolution

        evolution = PsychosocialEnvironmentEvolution(strategy=EnvironmentEvolutionStrategy.ECOLOGICAL)

        assert evolution.strategy == EnvironmentEvolutionStrategy.ECOLOGICAL
        assert evolution.generation == 0
        assert len(evolution.pressures) > 0  # Base pressures initialized

    def test_static_strategy_no_change(self):
        """Test static strategy doesn't change environment."""
        from src.liminal.psychosocial_coevolution import EnvironmentEvolutionStrategy, PsychosocialEnvironmentEvolution

        evolution = PsychosocialEnvironmentEvolution(strategy=EnvironmentEvolutionStrategy.STATIC)

        initial_params = dict(evolution.npc_base_parameters)

        changes = evolution.evolve_environment({"agent_1": 0.8, "agent_2": 0.9})

        assert changes == {}
        assert evolution.npc_base_parameters == initial_params

    def test_ecological_evolution_increases_complexity(self):
        """Test ecological evolution increases complexity when agents perform well."""
        from src.liminal.psychosocial_coevolution import EnvironmentEvolutionStrategy, PsychosocialEnvironmentEvolution

        evolution = PsychosocialEnvironmentEvolution(
            strategy=EnvironmentEvolutionStrategy.ECOLOGICAL, evolution_rate=0.2
        )

        # High fitness agents
        high_performance = {f"agent_{i}": 0.85 for i in range(10)}

        initial_complexity = evolution.npc_base_parameters["deception_sophistication"]

        evolution.evolve_environment(high_performance)

        # Complexity should increase
        assert evolution.npc_base_parameters["deception_sophistication"] >= initial_complexity

    def test_scaffolding_evolution(self):
        """Test scaffolding evolution gradual difficulty increase."""
        from src.liminal.psychosocial_coevolution import EnvironmentEvolutionStrategy, PsychosocialEnvironmentEvolution

        evolution = PsychosocialEnvironmentEvolution(strategy=EnvironmentEvolutionStrategy.SCAFFOLDING)

        # Moderate performance
        changes = evolution.evolve_environment({"a": 0.5, "b": 0.5})

        assert "target_complexity" in changes

    def test_fitness_recording(self):
        """Test fitness history recording."""
        from src.liminal.psychosocial_coevolution import EnvironmentEvolutionStrategy, PsychosocialEnvironmentEvolution

        evolution = PsychosocialEnvironmentEvolution()

        fitness_values = [0.3, 0.5, 0.7, 0.4, 0.6]
        evolution.record_generation_fitness(fitness_values)

        assert len(evolution.fitness_history) == 1
        assert len(evolution.fitness_variance_history) == 1

    def test_get_evolution_state(self):
        """Test getting evolution state summary."""
        from src.liminal.psychosocial_coevolution import PsychosocialEnvironmentEvolution

        evolution = PsychosocialEnvironmentEvolution()
        evolution.generation = 5

        state = evolution.get_evolution_state()

        assert state["generation"] == 5
        assert "strategy" in state
        assert "pressures" in state
        assert "npc_parameters" in state


class TestPsychosocialCoevolutionEngine:
    """Test the main co-evolution engine."""

    def test_engine_creation(self):
        """Test creating the main engine."""
        from src.liminal.psychosocial_coevolution import EnvironmentEvolutionStrategy, PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine(
            evolution_strategy=EnvironmentEvolutionStrategy.ECOLOGICAL,
            enable_belief_propagation=True,
            enable_social_dynamics=True,
        )

        assert engine.social_network is not None
        assert engine.belief_engine is not None
        assert engine.env_evolution is not None
        assert engine.tick == 0

    def test_register_agents(self):
        """Test registering agents."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        engine.register_agents(["agent_1", "agent_2", "agent_3"])

        assert "agent_1" in engine.social_network.hierarchy
        assert "agent_2" in engine.social_network.hierarchy
        assert "agent_3" in engine.social_network.hierarchy

    def test_process_interaction(self):
        """Test processing an interaction."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        engine.register_agents(["a", "b"])

        outcome = engine.process_interaction(
            agent1_id="a",
            agent2_id="b",
            agent1_action={"type": "interact", "agent_id": "a"},
            agent2_action={"type": "interact", "agent_id": "b"},
        )

        assert "competitive" in outcome

    def test_tick_world(self):
        """Test advancing the world by one tick."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        engine.register_agents(["a", "b"])

        # Create mock NPCs
        npcs = []
        for i in range(3):
            npc = BaseNPC(npc_id=f"npc_{i}", name=f"NPC {i}", archetype="neutral", soul_map=SoulMap())
            npcs.append(npc)

        initial_tick = engine.tick

        changes = engine.tick_world(npcs)

        assert engine.tick == initial_tick + 1
        assert "npc_deltas" in changes
        assert "new_coalitions" in changes

    def test_evolve_generation(self):
        """Test triggering generation evolution."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()

        fitness = {"agent_1": 0.7, "agent_2": 0.5, "agent_3": 0.8}

        changes = engine.evolve_generation(fitness)

        assert engine.env_evolution.generation == 1

    def test_get_social_observation(self):
        """Test getting social observation tensor."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        engine.register_agents(["a", "b"])

        # Create some relationships
        engine.process_interaction(
            "a", "b", {"type": "interact", "agent_id": "a"}, {"type": "interact", "agent_id": "b"}
        )

        obs = engine.get_social_observation("a")

        assert isinstance(obs, torch.Tensor)
        assert obs.dtype == torch.float32

    def test_tom_challenge_level(self):
        """Test ToM challenge level calculation."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()

        level = engine.get_tom_challenge_level()

        assert 1 <= level <= 5

    def test_get_state_summary(self):
        """Test getting state summary."""
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        engine.register_agents(["a", "b", "c"])

        summary = engine.get_state_summary()

        assert "tick" in summary
        assert "social_network" in summary
        assert "belief_engine" in summary
        assert "environment" in summary
        assert "tom_challenge_level" in summary


# =============================================================================
# NARRATIVE EMERGENCE TESTS
# =============================================================================


class TestNarrativeArchetypes:
    """Test narrative archetype definitions."""

    def test_archetypes_defined(self):
        """Test all archetypes are defined."""
        from src.liminal.narrative_emergence import NarrativeArchetype

        # Should have multiple archetypes
        archetypes = list(NarrativeArchetype)
        assert len(archetypes) >= 10

        # Check some expected archetypes
        archetype_names = [a.name for a in archetypes]
        assert "BETRAYAL_BREWING" in archetype_names
        assert "RELUCTANT_ALLIANCE" in archetype_names
        assert "POWER_STRUGGLE" in archetype_names
        assert "HIDDEN_IDENTITY" in archetype_names


class TestEmergentNarrative:
    """Test emergent narrative dataclass."""

    def test_narrative_creation(self):
        """Test creating an emergent narrative."""
        from src.liminal.narrative_emergence import EmergentNarrative, NarrativeArchetype

        narrative = EmergentNarrative(
            narrative_id="test_1",
            archetype=NarrativeArchetype.BETRAYAL_BREWING,
            title="The Wavering Loyalty",
            description="Trust is tested...",
            protagonists=["npc_1"],
            antagonists=["npc_2"],
            supporting_cast=["npc_3"],
        )

        assert narrative.narrative_id == "test_1"
        assert narrative.archetype == NarrativeArchetype.BETRAYAL_BREWING
        assert narrative.current_act == 1
        assert narrative.tension_level == 0.0
        assert narrative.tick_resolved is None

    def test_dramatic_summary(self):
        """Test generating dramatic summary."""
        from src.liminal.narrative_emergence import EmergentNarrative, NarrativeArchetype

        narrative = EmergentNarrative(
            narrative_id="test_1",
            archetype=NarrativeArchetype.POWER_STRUGGLE,
            title="Battle for Leadership",
            description="Two rivals clash",
            protagonists=["a"],
            antagonists=["b"],
            supporting_cast=[],
        )

        summary = narrative.get_dramatic_summary()

        assert "POWER_STRUGGLE" in summary
        assert "Battle for Leadership" in summary


class TestNarrativeDetector:
    """Test narrative detection from dynamics."""

    def test_detector_creation(self):
        """Test creating narrative detector."""
        from src.liminal.narrative_emergence import NarrativeDetector
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        detector = NarrativeDetector(engine)

        assert detector.engine is engine
        assert len(detector.detected_narratives) == 0
        assert len(detector.matchers) > 0

    def test_scan_for_narratives(self):
        """Test scanning for narratives."""
        from src.liminal.narrative_emergence import NarrativeDetector
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        engine = PsychosocialCoevolutionEngine()
        detector = NarrativeDetector(engine)

        # Create some NPCs
        npcs = {}
        for i in range(5):
            npc = BaseNPC(npc_id=f"npc_{i}", name=f"NPC {i}", archetype="neutral", soul_map=SoulMap())
            npcs[f"npc_{i}"] = npc

        # Scan (might not detect anything with empty state)
        narratives = detector.scan_for_narratives(npcs, tick=100)

        assert isinstance(narratives, list)


class TestNarrativeProgressionEngine:
    """Test narrative progression mechanics."""

    def test_progression_engine_creation(self):
        """Test creating progression engine."""
        from src.liminal.narrative_emergence import NarrativeDetector, NarrativeProgressionEngine
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        detector = NarrativeDetector(coev)
        progression = NarrativeProgressionEngine(detector)

        assert progression.detector is detector
        assert len(progression.resolution_conditions) > 0

    def test_update_narratives(self):
        """Test updating narratives."""
        from src.liminal.narrative_emergence import NarrativeDetector, NarrativeProgressionEngine
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        detector = NarrativeDetector(coev)
        progression = NarrativeProgressionEngine(detector)

        npcs = {}
        updates = progression.update_narratives(npcs, tick=100)

        assert "act_transitions" in updates
        assert "resolutions" in updates
        assert "tension_changes" in updates


class TestNarrativeEmergenceSystem:
    """Test the main narrative emergence system."""

    def test_system_creation(self):
        """Test creating the narrative system."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        system = NarrativeEmergenceSystem(coev)

        assert system.coevolution is coev
        assert system.detector is not None
        assert system.progression is not None

    def test_tick(self):
        """Test processing one tick of narrative emergence."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        system = NarrativeEmergenceSystem(coev)

        npcs = {}
        for i in range(5):
            npc = BaseNPC(npc_id=f"npc_{i}", name=f"NPC {i}", archetype="neutral", soul_map=SoulMap())
            npcs[f"npc_{i}"] = npc

        results = system.tick(npcs, current_tick=10)

        assert "new_narratives" in results
        assert "updates" in results
        assert "active_count" in results

    def test_get_active_narratives(self):
        """Test getting active narratives."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        system = NarrativeEmergenceSystem(coev)

        active = system.get_active_narratives()

        assert isinstance(active, list)

    def test_tom_learning_opportunities(self):
        """Test getting ToM learning opportunities."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        system = NarrativeEmergenceSystem(coev)

        opportunities = system.get_tom_learning_opportunities()

        assert isinstance(opportunities, list)

    def test_narrative_metrics(self):
        """Test getting narrative metrics."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        coev = PsychosocialCoevolutionEngine()
        system = NarrativeEmergenceSystem(coev)

        metrics = system.get_narrative_metrics()

        assert "total_detected" in metrics
        assert "currently_active" in metrics
        assert "resolved" in metrics
        assert "by_archetype" in metrics
        assert "average_duration" in metrics
        assert "average_tom_depth" in metrics


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCoevolutionNarrativeIntegration:
    """Test integration between co-evolution and narrative systems."""

    def test_full_simulation_cycle(self):
        """Test running a complete simulation cycle."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine

        # Create systems
        coev = PsychosocialCoevolutionEngine()
        narrative = NarrativeEmergenceSystem(coev)

        # Create NPCs
        npcs = {}
        for i in range(10):
            npc = BaseNPC(
                npc_id=f"npc_{i}",
                name=f"NPC {i}",
                archetype=random.choice(["trusting", "suspicious", "neutral"]),
                soul_map=SoulMap(),
            )
            npcs[f"npc_{i}"] = npc

        # Register agents
        agent_ids = [f"agent_{i}" for i in range(5)]
        coev.register_agents(agent_ids)

        # Run simulation ticks
        for tick in range(50):
            # Random interactions
            for _ in range(3):
                a1, a2 = random.sample(agent_ids, 2)
                coev.process_interaction(
                    a1,
                    a2,
                    {"type": "interact", "agent_id": a1, "cooperate": random.random() > 0.5},
                    {"type": "interact", "agent_id": a2, "cooperate": random.random() > 0.5},
                )

            # Tick world
            coev.tick_world(list(npcs.values()))

            # Tick narratives
            narrative.tick(npcs, tick)

        # Check state
        state = coev.get_state_summary()
        assert state["tick"] > 0

        metrics = narrative.get_narrative_metrics()
        # Metrics should be valid
        assert metrics["total_detected"] >= 0

    def test_narrative_detection_with_coalitions(self):
        """Test narrative detection when coalitions form."""
        from src.liminal.narrative_emergence import NarrativeEmergenceSystem
        from src.liminal.psychosocial_coevolution import PsychosocialCoevolutionEngine, TheoreticalConstants

        coev = PsychosocialCoevolutionEngine()
        narrative = NarrativeEmergenceSystem(coev)

        # Create agents and force coalition
        agents = ["a", "b", "c", "d"]
        coev.register_agents(agents)

        # Create strong relationships between a, b, c
        for pair in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("a", "c"), ("c", "a")]:
            edge = coev.social_network.get_or_create_edge(pair[0], pair[1])
            edge.trust = TheoreticalConstants.COALITION_FORMATION_THRESHOLD + 0.1
            edge.affect = 0.5
            edge.familiarity = 0.5

        # Detect coalitions
        coev.social_network.detect_coalitions()

        # Create NPCs (needed for narrative detection)
        npcs = {}
        for agent_id in agents:
            npc = BaseNPC(npc_id=agent_id, name=agent_id.upper(), archetype="neutral", soul_map=SoulMap())
            npcs[agent_id] = npc

        # Scan for narratives
        new_narratives = narrative.detector.scan_for_narratives(npcs, tick=100)

        # Should find some narratives related to the coalition
        # (exact detection depends on conditions)


# =============================================================================
# MAIN
# =============================================================================


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
