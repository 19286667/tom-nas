#!/usr/bin/env python3
"""
Tests for Liminal Architectures Game Environment

Tests cover:
- Soul Map system
- Realm mechanics
- NPC creation (heroes and archetypes)
- Soul Scanner analysis
- Cognitive hazards
- Ontological instability
- Game environment integration
"""

import pytest
import torch
import sys

sys.path.insert(0, '.')

from src.liminal.soul_map import (
    SoulMap, SoulMapDelta, SoulMapCluster,
    COGNITIVE_DIMENSIONS, EMOTIONAL_DIMENSIONS,
    MOTIVATIONAL_DIMENSIONS, SOCIAL_DIMENSIONS, SELF_DIMENSIONS,
)
from src.liminal.realms import (
    Realm, RealmType, REALMS, get_realm, RealmTransition,
)
from src.liminal.npcs.base_npc import BaseNPC, NPCState, NPCBehavior
from src.liminal.npcs.heroes import HERO_NPCS, create_hero_npc, HERO_DEFINITIONS
from src.liminal.npcs.archetypes import (
    ARCHETYPES, create_archetype_npc, populate_realm,
)
from src.liminal.mechanics.soul_scanner import (
    SoulScanner, AnalysisDepth, AnalysisResult,
)
from src.liminal.mechanics.cognitive_hazards import (
    CognitiveHazard, HAZARD_REGISTRY, apply_hazard, HazardCategory,
)
from src.liminal.mechanics.ontological_instability import (
    OntologicalInstability, InstabilityLevel,
)
from src.liminal.game_environment import (
    LiminalEnvironment, GameState, Observation, ActionType,
)


class TestSoulMap:
    """Tests for the Soul Map system."""

    def test_soul_map_creation(self):
        """Test basic Soul Map creation."""
        soul = SoulMap()
        assert soul is not None
        assert len(soul.cognitive) == 12
        assert len(soul.emotional) == 12
        assert len(soul.motivational) == 12
        assert len(soul.social) == 12
        assert len(soul.self) == 12

    def test_soul_map_to_tensor(self):
        """Test conversion to tensor."""
        soul = SoulMap()
        tensor = soul.to_tensor()
        assert tensor.shape == (65,)  # 60 + 5 realm modifiers
        assert tensor.dtype == torch.float32

    def test_soul_map_from_tensor(self):
        """Test creation from tensor."""
        original = SoulMap()
        tensor = original.to_tensor()
        reconstructed = SoulMap.from_tensor(tensor)

        # Values should be preserved
        assert abs(original.cognitive["processing_speed"] -
                  reconstructed.cognitive["processing_speed"]) < 0.01

    def test_soul_map_from_archetype(self):
        """Test archetype-based creation."""
        soul = SoulMap.from_archetype("bureaucrat", variance=0.0)
        # Bureaucrat has high order drive
        assert soul.motivational["order_drive"] > 0.8

    def test_soul_map_delta_application(self):
        """Test applying deltas to Soul Map."""
        soul = SoulMap()
        initial_anxiety = soul.emotional["anxiety_baseline"]

        delta = SoulMapDelta.fear(intensity=0.3)
        delta.apply_to(soul)

        assert soul.emotional["anxiety_baseline"] > initial_anxiety

    def test_soul_map_stability_calculation(self):
        """Test stability metric."""
        soul = SoulMap()
        stability = soul.compute_stability()
        assert 0.0 <= stability <= 1.0

    def test_soul_map_threat_response(self):
        """Test threat response metric."""
        soul = SoulMap()
        threat = soul.compute_threat_response()
        assert 0.0 <= threat <= 1.0

    def test_soul_map_dominant_motivation(self):
        """Test finding dominant motivation."""
        soul = SoulMap()
        soul.motivational["novelty_drive"] = 1.0
        dominant, value = soul.get_dominant_motivation()
        assert dominant == "novelty"
        assert value == 1.0

    def test_soul_map_json_roundtrip(self):
        """Test JSON serialization."""
        soul = SoulMap()
        json_data = soul.to_json()
        reconstructed = SoulMap.from_json(json_data)

        assert abs(soul.cognitive["processing_speed"] -
                  reconstructed.cognitive["processing_speed"]) < 0.01


class TestRealms:
    """Tests for the Realm system."""

    def test_all_realms_exist(self):
        """Test all 6 realms are defined."""
        assert len(REALMS) == 6
        for realm_type in RealmType:
            assert realm_type in REALMS

    def test_realm_properties(self):
        """Test realm properties are set."""
        peregrine = get_realm(RealmType.PEREGRINE)
        assert peregrine.name == "Peregrine"
        assert peregrine.key_mechanic == "Complementarity"
        assert len(peregrine.traversal_methods) > 0

    def test_realm_vibe_compatibility(self):
        """Test vibe compatibility calculation."""
        ministry = get_realm(RealmType.MINISTRY)
        soul = SoulMap.from_archetype("bureaucrat", variance=0.0)

        compatibility = ministry.get_vibe_compatibility(soul)
        assert 0.0 <= compatibility <= 1.0
        # Bureaucrat should be compatible with Ministry
        assert compatibility > 0.5

    def test_realm_ambient_effects(self):
        """Test realm ambient effects application."""
        ministry = get_realm(RealmType.MINISTRY)
        soul = SoulMap()
        initial_order = soul.motivational["order_drive"]

        ministry.apply_ambient_effects(soul, duration=10.0)

        # Ministry increases order drive
        assert soul.motivational["order_drive"] > initial_order

    def test_realm_transition(self):
        """Test realm transition checks."""
        soul = SoulMap()
        peregrine = get_realm(RealmType.PEREGRINE)
        ministry = get_realm(RealmType.MINISTRY)

        can_transition, reason = RealmTransition.can_transition(
            peregrine, ministry, soul
        )
        assert can_transition is True


class TestNPCs:
    """Tests for NPC system."""

    def test_hero_npc_creation(self):
        """Test creating hero NPCs."""
        for hero_id in HERO_NPCS:
            npc = create_hero_npc(hero_id)
            assert npc is not None
            assert npc.is_hero is True
            assert npc.name == HERO_DEFINITIONS[hero_id]["name"]

    def test_arthur_peregrine_stats(self):
        """Test Arthur's specific Soul Map values."""
        arthur = create_hero_npc("arthur_peregrine")
        assert arthur.soul_map.emotional["anxiety_baseline"] > 0.7
        assert arthur.soul_map.emotional["threat_sensitivity"] > 0.8
        assert arthur.tom_depth == 4

    def test_agnes_high_tom(self):
        """Test Agnes has maximum ToM."""
        agnes = create_hero_npc("agnes_peregrine")
        assert agnes.soul_map.cognitive["tom_depth"] == 1.0  # Max
        assert agnes.tom_depth == 5

    def test_archetype_npc_creation(self):
        """Test creating archetype NPCs."""
        for archetype in ARCHETYPES.keys():
            npc = create_archetype_npc(archetype)
            assert npc is not None
            assert npc.archetype == archetype

    def test_populate_realm(self):
        """Test realm population."""
        npcs = populate_realm(RealmType.MINISTRY, count=20)
        assert len(npcs) > 0
        for npc in npcs:
            assert npc.current_realm == RealmType.MINISTRY

    def test_npc_observation_tensor(self):
        """Test NPC to observation tensor conversion."""
        npc = create_hero_npc("arthur_peregrine")
        tensor = npc.to_observation_tensor()
        assert tensor.shape[0] == 70  # 65 soul + 5 context

    def test_npc_belief_system(self):
        """Test NPC belief management."""
        npc = create_hero_npc("arthur_peregrine")
        npc.update_belief(
            target_id="player",
            belief_type="location",
            content=(10.0, 20.0),
            confidence=0.8,
            timestamp=100,
        )

        belief = npc.get_belief("player", "location")
        assert belief is not None
        assert belief.content == (10.0, 20.0)
        assert belief.confidence == 0.8

    def test_npc_action_decision(self):
        """Test NPC decision making."""
        npc = create_hero_npc("arthur_peregrine")
        context = {"threat_level": 0.8, "stranger_present": True}
        action = npc.decide_action(context)
        assert "action_type" in action


class TestSoulScanner:
    """Tests for Soul Scanner analysis."""

    def test_scanner_creation(self):
        """Test scanner initialization."""
        scanner = SoulScanner(player_tom_depth=3)
        assert scanner.player_tom_depth == 3

    def test_passive_scan(self):
        """Test passive scan returns basic info."""
        scanner = SoulScanner()
        target = create_hero_npc("arthur_peregrine")

        result = scanner.passive_scan(target)
        assert result.depth == AnalysisDepth.PASSIVE
        assert result.aura_color is not None
        assert 0.0 <= result.aura_intensity <= 1.0

    def test_moderate_scan(self):
        """Test moderate scan returns cluster summaries."""
        scanner = SoulScanner()
        target = create_hero_npc("arthur_peregrine")

        result = scanner.moderate_scan(target)
        assert result.depth == AnalysisDepth.MODERATE
        assert "cognitive" in result.cluster_summaries
        assert "emotional" in result.cluster_summaries

    def test_deep_scan(self):
        """Test deep scan returns full Soul Map."""
        scanner = SoulScanner(player_tom_depth=5)
        target = create_hero_npc("arthur_peregrine")

        result = scanner.deep_scan(target)
        assert result.depth == AnalysisDepth.DEEP
        assert result.full_soul_map is not None

    def test_predictive_scan(self):
        """Test predictive scan returns behavior predictions."""
        scanner = SoulScanner(player_tom_depth=4)
        target = create_hero_npc("arthur_peregrine")
        context = {"threat_present": True}

        result = scanner.predictive_scan(target, context)
        assert result.depth == AnalysisDepth.PREDICTIVE
        assert len(result.predicted_behaviors) > 0

    def test_analysis_accuracy_tom_mismatch(self):
        """Test accuracy decreases with ToM mismatch."""
        scanner_low = SoulScanner(player_tom_depth=2)
        scanner_high = SoulScanner(player_tom_depth=5)
        target = create_hero_npc("agnes_peregrine")  # ToM 5

        acc_low = scanner_low._calculate_analysis_accuracy(target)
        acc_high = scanner_high._calculate_analysis_accuracy(target)

        assert acc_high > acc_low


class TestCognitiveHazards:
    """Tests for cognitive hazards system."""

    def test_hazard_registry(self):
        """Test hazards are registered."""
        assert len(HAZARD_REGISTRY) > 0
        assert "doubt" in HAZARD_REGISTRY
        assert "fear" in HAZARD_REGISTRY
        assert "validation" in HAZARD_REGISTRY

    def test_hazard_application(self):
        """Test applying hazard to NPC."""
        target = create_archetype_npc("bureaucrat")
        initial_stability = target.soul_map.compute_stability()

        success, result = apply_hazard("doubt", target)

        assert success is True
        assert "stability_change" in result
        # Doubt should decrease stability
        assert target.soul_map.compute_stability() <= initial_stability

    def test_hazard_resistance(self):
        """Test hazard resistance calculation."""
        hazard = HAZARD_REGISTRY["doubt"]

        # High self-coherence NPC should resist destabilization
        strong_npc = create_hero_npc("director_thorne")  # High stability
        resistance = hazard._calculate_resistance(strong_npc)

        assert 0.0 <= resistance <= 1.0

    def test_positive_hazard(self):
        """Test positive hazard (validation) improves state."""
        target = create_archetype_npc("mourner")  # Low baseline
        initial_esteem = target.soul_map.self["esteem_stability"]

        success, _ = apply_hazard("validation", target)

        assert success is True
        assert target.soul_map.self["esteem_stability"] >= initial_esteem


class TestOntologicalInstability:
    """Tests for ontological instability system."""

    def test_instability_creation(self):
        """Test instability tracker initialization."""
        instability = OntologicalInstability()
        assert instability.instability == 0.0
        assert instability.current_level == InstabilityLevel.STABLE

    def test_instability_increase(self):
        """Test adding instability."""
        instability = OntologicalInstability()
        instability.add_instability(20.0, "test", "Test event")

        assert instability.instability == 20.0
        assert instability.current_level == InstabilityLevel.RIPPLES

    def test_instability_levels(self):
        """Test level progression."""
        instability = OntologicalInstability()

        levels = [
            (0, InstabilityLevel.STABLE),
            (20, InstabilityLevel.RIPPLES),
            (40, InstabilityLevel.DISTORTION),
            (60, InstabilityLevel.FRACTURING),
            (80, InstabilityLevel.MANIFESTATION),
            (100, InstabilityLevel.COLLAPSE),
        ]

        for amount, expected_level in levels:
            instability.instability = amount
            instability._update_level()
            assert instability.current_level == expected_level

    def test_instability_decay(self):
        """Test natural decay."""
        instability = OntologicalInstability(decay_rate=0.1)
        instability.add_instability(50.0, "test", "Test")

        initial = instability.instability
        instability.tick(10.0)

        assert instability.instability < initial

    def test_nothing_manifestation(self):
        """Test The Nothing manifests at high levels."""
        instability = OntologicalInstability()
        instability.add_instability(80.0, "test", "Major disruption")

        assert instability.nothing_manifested is True


class TestLiminalEnvironment:
    """Tests for the main game environment."""

    def test_environment_creation(self):
        """Test environment initialization."""
        env = LiminalEnvironment(
            population_size=50,
            include_heroes=True,
            seed=42,
        )

        assert len(env.npcs) > 0
        stats = env.get_statistics()
        assert stats["hero_count"] == len(HERO_NPCS)

    def test_environment_reset(self):
        """Test environment reset."""
        env = LiminalEnvironment(population_size=50, seed=42)
        obs1 = env.reset(seed=42)
        env.step({"type": ActionType.WAIT})
        obs2 = env.reset(seed=42)

        # Observations after reset should be consistent
        assert torch.allclose(obs1.full_tensor, obs2.full_tensor, atol=1e-5)

    def test_environment_step(self):
        """Test taking steps."""
        env = LiminalEnvironment(population_size=50, seed=42)
        env.reset()

        result = env.step({"type": ActionType.WAIT})

        assert result.observation is not None
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)

    def test_analyze_action(self):
        """Test analyze action."""
        env = LiminalEnvironment(population_size=50, seed=42)
        env.reset()

        # Get a nearby NPC
        game_state = env._get_game_state()
        if game_state.nearby_npcs:
            action = {
                "type": ActionType.ANALYZE,
                "target_id": game_state.nearby_npcs[0],
                "depth": AnalysisDepth.MODERATE,
            }
            result = env.step(action)
            assert result.reward > 0  # Analysis should give small positive reward

    def test_intervene_action(self):
        """Test intervene action."""
        env = LiminalEnvironment(population_size=50, seed=42)
        env.reset()

        game_state = env._get_game_state()
        if game_state.nearby_npcs:
            action = {
                "type": ActionType.INTERVENE,
                "target_id": game_state.nearby_npcs[0],
                "hazard": "doubt",
                "intensity": 0.5,
            }
            result = env.step(action)
            # Intervention should affect instability
            assert result.game_state.instability > 0

    def test_observation_shape(self):
        """Test observation tensor shape."""
        env = LiminalEnvironment(population_size=50, seed=42)
        obs = env.reset()

        assert obs.full_tensor.dim() == 1
        assert obs.player_soul_map.shape == (65,)
        assert len(obs.nearby_npc_states) == env.max_nearby_npcs

    def test_episode_termination(self):
        """Test episode terminates at max length."""
        env = LiminalEnvironment(
            population_size=20,
            max_episode_length=10,
            seed=42,
        )
        env.reset()

        done = False
        steps = 0
        while not done and steps < 20:
            result = env.step({"type": ActionType.WAIT})
            done = result.done
            steps += 1

        assert done is True
        assert steps <= 11  # Max length + 1

    def test_attach_nas_model(self):
        """Test attaching NAS model to NPC."""
        env = LiminalEnvironment(population_size=20, seed=42)
        env.reset()

        # Create dummy model
        class DummyModel:
            def __call__(self, x):
                return {"beliefs": torch.zeros(1, 181), "actions": torch.zeros(1)}

        # Attach to first NPC
        npc_id = list(env.npcs.keys())[0]
        result = env.attach_nas_model(npc_id, DummyModel())
        assert result is True


class TestIntegration:
    """Integration tests."""

    def test_full_episode_run(self):
        """Test running a complete episode."""
        env = LiminalEnvironment(
            population_size=100,
            include_heroes=True,
            max_episode_length=50,
            seed=42,
        )

        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Random action selection
            import random
            action_type = random.choice(list(ActionType))
            action = {"type": action_type}

            game_state = env._get_game_state()
            if action_type in [ActionType.ANALYZE, ActionType.INTERVENE,
                              ActionType.PREDICT]:
                if game_state.nearby_npcs:
                    action["target_id"] = game_state.nearby_npcs[0]
                    if action_type == ActionType.INTERVENE:
                        action["hazard"] = random.choice(list(HAZARD_REGISTRY.keys()))

            result = env.step(action)
            total_reward += result.reward
            done = result.done

        assert env.tick > 0
        stats = env.get_statistics()
        assert stats["current_tick"] == env.tick

    def test_hero_npc_in_environment(self):
        """Test hero NPCs are properly integrated."""
        env = LiminalEnvironment(
            population_size=50,
            include_heroes=True,
            seed=42,
        )
        env.reset()

        # Check Arthur exists
        arthur = None
        for npc in env.npcs.values():
            if npc.npc_id == "arthur_peregrine":
                arthur = npc
                break

        assert arthur is not None
        assert arthur.is_hero is True
        assert arthur.name == "Arthur Peregrine"

    def test_realm_effects_on_npcs(self):
        """Test realm effects properly applied during steps."""
        env = LiminalEnvironment(
            population_size=20,
            starting_realm=RealmType.MINISTRY,
            max_episode_length=100,
            seed=42,
        )
        env.reset()

        # Get a Ministry NPC
        ministry_npcs = env.get_npcs_in_realm(RealmType.MINISTRY)
        if ministry_npcs:
            npc = ministry_npcs[0]
            initial_order = npc.soul_map.motivational["order_drive"]

            # Run some steps
            for _ in range(20):
                env.step({"type": ActionType.WAIT})

            # Order drive should have increased (Ministry effect)
            # Note: might not change if NPC already at max
            assert npc.soul_map.motivational["order_drive"] >= initial_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
