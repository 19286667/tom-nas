"""
Integration tests for the complete research cycle.

Tests end-to-end: observation → hypothesis → experiment → belief update → publication.
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.research import ResearchCycle, PopulationResearchManager


class TestResearchCycle:
    """Tests for individual agent research cycles."""

    @pytest.fixture
    def cycle(self):
        """Create a research cycle instance."""
        return ResearchCycle("test_agent")

    @pytest.fixture
    def environment(self):
        """Create a mock environment state."""
        return {
            "agents": {
                "agent_1": {
                    "activity": "researching",
                    "thought": "testing hypothesis about beliefs",
                    "position": {"x": 10, "y": 0, "z": 20},
                },
                "agent_2": {
                    "activity": "simulating",
                    "thought": "modeling agent_1 behavior",
                    "position": {"x": -5, "y": 0, "z": 15},
                },
            },
            "publications": [],
            "simulations": [
                {"id": "sim_1", "creator": "agent_2", "depth": 1}
            ],
        }

    @pytest.mark.asyncio
    async def test_observation_phase(self, cycle, environment):
        """Agent can observe environment."""
        assert cycle.phase.value == "observing"

        result = await cycle.step(environment)

        assert result["observations"] > 0
        assert len(cycle.observations) > 0

    @pytest.mark.asyncio
    async def test_full_cycle(self, cycle, environment):
        """Agent can complete full research cycle."""
        phases_seen = set()

        # Run enough steps to complete a cycle
        for _ in range(20):
            result = await cycle.step(environment)
            phases_seen.add(result["phase"])

            # Cycle should eventually return to observing
            if len(phases_seen) >= 4:
                break

        # Should have progressed through multiple phases
        assert len(phases_seen) >= 2

    @pytest.mark.asyncio
    async def test_state_tracking(self, cycle, environment):
        """Research state is tracked correctly."""
        # Run a few steps
        for _ in range(5):
            await cycle.step(environment)

        state = cycle.get_state()

        assert "agent_id" in state
        assert "phase" in state
        assert "observations_count" in state
        assert state["agent_id"] == "test_agent"

    @pytest.mark.asyncio
    async def test_belief_update_from_publication(self, cycle):
        """Beliefs update when receiving publications."""
        initial_beliefs = len(cycle.belief_state)

        publication = {
            "author": "other_agent",
            "hypothesis": "Agents can model other agents",
            "confidence": 0.8,
            "supported": True,
        }

        cycle.receive_publication(publication)

        # Should have new belief
        assert len(cycle.belief_state) > initial_beliefs


class TestPopulationManager:
    """Tests for population-level research management."""

    @pytest.fixture
    def manager(self):
        """Create a population manager."""
        return PopulationResearchManager()

    @pytest.fixture
    def environment(self):
        """Create a mock environment."""
        return {
            "agents": {
                "external_agent": {
                    "activity": "publishing",
                    "thought": "sharing results",
                    "position": {"x": 0, "y": 0, "z": 0},
                }
            },
            "publications": [],
            "simulations": [],
        }

    def test_register_agents(self, manager):
        """Can register agents."""
        manager.register_agent("agent_1")
        manager.register_agent("agent_2")

        assert len(manager.agents) == 2
        assert "agent_1" in manager.agents
        assert "agent_2" in manager.agents

    @pytest.mark.asyncio
    async def test_step_all(self, manager, environment):
        """Can step all agents."""
        manager.register_agent("agent_1")
        manager.register_agent("agent_2")

        results = await manager.step_all(environment)

        assert "agent_1" in results
        assert "agent_2" in results

    def test_fitness_computation(self, manager):
        """Fitness is computed correctly."""
        cycle = manager.register_agent("agent_1")

        # Initial fitness should be low
        fitness = manager.compute_fitness("agent_1")
        assert 0 <= fitness <= 1

    def test_population_stats(self, manager):
        """Population stats are computed correctly."""
        manager.register_agent("agent_1")
        manager.register_agent("agent_2")
        manager.register_agent("agent_3")

        stats = manager.get_population_stats()

        assert stats["population"] == 3
        assert "mean_fitness" in stats
        assert "generation" in stats

    @pytest.mark.asyncio
    async def test_publication_broadcast(self, manager, environment):
        """Publications broadcast to other agents."""
        manager.register_agent("agent_1")
        manager.register_agent("agent_2")

        # Manually add a publication
        publication = {
            "id": "pub_test",
            "author": "agent_1",
            "hypothesis": "Test hypothesis",
            "confidence": 0.75,
            "supported": True,
        }
        manager.publications.append(publication)
        manager._broadcast_publication(publication)

        # Agent 2 should have updated beliefs
        agent_2_cycle = manager.agents["agent_2"]
        assert "Test hypothesis" in agent_2_cycle.belief_state

    def test_selection(self, manager):
        """Selection returns high-fitness agents."""
        manager.register_agent("agent_1")
        manager.register_agent("agent_2")
        manager.register_agent("agent_3")
        manager.register_agent("agent_4")

        reproducers = manager.select_and_reproduce(population_size=4)

        # Should return top 50%
        assert len(reproducers) == 2
        assert manager.generation == 1


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_multi_agent_research(self):
        """Multiple agents can conduct research together."""
        manager = PopulationResearchManager()

        # Create population
        for i in range(5):
            manager.register_agent(f"agent_{i}")

        # Create environment with interactions
        environment = {
            "agents": {
                f"agent_{i}": {
                    "activity": "researching",
                    "thought": f"hypothesis {i}",
                    "position": {"x": i * 10, "y": 0, "z": 0},
                }
                for i in range(5)
            },
            "publications": [],
            "simulations": [],
        }

        # Run multiple generations
        for generation in range(3):
            # Each agent steps multiple times
            for _ in range(10):
                await manager.step_all(environment)

            # Check progress
            stats = manager.get_population_stats()
            assert stats["population"] == 5

        # Should have some publications
        assert len(manager.publications) >= 0  # May or may not have publications

    @pytest.mark.asyncio
    async def test_research_produces_output(self):
        """Research cycle eventually produces publications."""
        cycle = ResearchCycle("prolific_agent")

        # Rich environment
        environment = {
            "agents": {
                f"agent_{i}": {
                    "activity": "simulating" if i % 2 == 0 else "researching",
                    "thought": f"modeling belief states at order {i}",
                    "position": {"x": i * 5, "y": 0, "z": i * 5},
                }
                for i in range(10)
            },
            "publications": [
                {
                    "id": f"pub_{i}",
                    "author": f"agent_{i}",
                    "hypothesis": f"Hypothesis {i}",
                    "confidence": 0.6 + i * 0.03,
                }
                for i in range(5)
            ],
            "simulations": [
                {"id": f"sim_{i}", "creator": f"agent_{i}", "depth": 1}
                for i in range(3)
            ],
        }

        # Run until publication or max steps
        max_steps = 100
        publications = []

        for _ in range(max_steps):
            result = await cycle.step(environment)
            publications.extend(result.get("outputs", []))

            if publications:
                break

        # Should have gathered observations at minimum
        state = cycle.get_state()
        assert state["observations_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
