"""
Psychosocial Co-Evolution Demo
==============================

This demonstration shows the complete integration of:
1. Bidirectional co-evolution (agents and environment evolve together)
2. Emergent social dynamics (coalitions, hierarchies, belief propagation)
3. Narrative emergence (meaningful stories from raw dynamics)
4. Scientific grounding (explicit theoretical foundations)

Run this demo to see Theory of Mind agents evolving within genuinely
complex psychosocial environments.

Usage:
    python -m src.liminal.coevolution_demo
    python -m src.liminal.coevolution_demo --generations 20 --verbose
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.liminal.game_environment import ActionType
from src.liminal.narrative_emergence import (
    NarrativeEmergenceSystem,
)
from src.liminal.npcs.base_npc import BaseNPC
from src.liminal.psychosocial_coevolution import (
    EnvironmentEvolutionStrategy,
    PsychosocialCoevolutionEngine,
    TheoreticalConstants,
)
from src.liminal.soul_map import SoulMap

# =============================================================================
# DEMO AGENT (Simple ToM-like agent for demonstration)
# =============================================================================


class DemoToMAgent:
    """
    A demonstration agent with basic ToM capabilities.

    For the full system, this would be replaced with NAS-evolved
    architectures (TRN, RSAN, Transformer, Hybrid).
    """

    def __init__(self, agent_id: str, tom_depth: int = 2):
        self.agent_id = agent_id
        self.tom_depth = tom_depth
        self.fitness = 0.0

        # Simple memory
        self.known_agents: Dict[str, Dict] = {}
        self.interaction_history: List[Dict] = []

    def select_action(
        self, observation: Dict[str, Any], game_state: Any, coevolution: PsychosocialCoevolutionEngine
    ) -> Dict[str, Any]:
        """Select action based on observation and social context."""
        # Get social observation
        social_obs = coevolution.get_social_observation(self.agent_id)

        # Simple strategy based on ToM depth
        if self.tom_depth >= 2:
            # Consider what others might do
            return self._strategic_action(observation, game_state, coevolution)
        else:
            # Basic reactive behavior
            return self._reactive_action(observation, game_state)

    def _strategic_action(
        self, observation: Dict, game_state: Any, coevolution: PsychosocialCoevolutionEngine
    ) -> Dict[str, Any]:
        """Strategic action using ToM reasoning."""
        network = coevolution.social_network

        # Find best interaction target
        target = None
        best_score = -float("inf")

        for other_id in network.hierarchy:
            if other_id == self.agent_id:
                continue

            edge = network.edges.get((self.agent_id, other_id))
            if edge:
                # Score based on potential benefit
                score = edge.trust * 0.5 + (1 - edge.familiarity) * 0.3

                # Prefer coalition members
                in_same_coalition = any(
                    self.agent_id in members and other_id in members for members in network.coalitions.values()
                )
                if in_same_coalition:
                    score += 0.2

                if score > best_score:
                    best_score = score
                    target = other_id

        if target:
            # Decide cooperation based on trust
            edge = network.edges.get((self.agent_id, target))
            will_cooperate = edge and edge.trust > 0.5

            return {
                "type": ActionType.INTERACT if will_cooperate else ActionType.ANALYZE,
                "target_id": target,
                "agent_id": self.agent_id,
                "cooperate": will_cooperate,
            }

        return {"type": ActionType.WAIT, "agent_id": self.agent_id}

    def _reactive_action(self, observation: Dict, game_state: Any) -> Dict[str, Any]:
        """Simple reactive action."""
        actions = [ActionType.WAIT, ActionType.MOVE, ActionType.ANALYZE]
        return {
            "type": random.choice(actions),
            "agent_id": self.agent_id,
        }

    def update_fitness(self, reward: float):
        """Update agent fitness."""
        self.fitness = self.fitness * 0.9 + reward * 0.1


# =============================================================================
# DEMO POPULATION
# =============================================================================


class DemoPopulation:
    """
    A population of demo agents for co-evolution demonstration.
    """

    def __init__(self, size: int = 20):
        self.agents: Dict[str, DemoToMAgent] = {}

        for i in range(size):
            agent_id = f"agent_{i}"
            tom_depth = random.randint(1, 3)  # Varied ToM depths
            self.agents[agent_id] = DemoToMAgent(agent_id, tom_depth)

    def get_agent_ids(self) -> List[str]:
        return list(self.agents.keys())

    def get_fitness_dict(self) -> Dict[str, float]:
        return {aid: a.fitness for aid, a in self.agents.items()}

    def evolve_generation(self):
        """
        Simple evolution: replace bottom performers with mutations of top.
        """
        sorted_agents = sorted(self.agents.items(), key=lambda x: x[1].fitness, reverse=True)

        # Keep top 50%
        survivors = sorted_agents[: len(sorted_agents) // 2]

        # Replace bottom 50% with mutations
        new_agents = {}
        for aid, agent in survivors:
            new_agents[aid] = agent

            # Create offspring
            offspring_id = f"agent_{len(new_agents)}"
            while offspring_id in new_agents:
                offspring_id = f"agent_{random.randint(100, 999)}"

            # Mutate ToM depth occasionally
            new_tom = agent.tom_depth
            if random.random() < 0.1:
                new_tom = max(1, min(5, new_tom + random.choice([-1, 1])))

            new_agents[offspring_id] = DemoToMAgent(offspring_id, new_tom)

        self.agents = new_agents


# =============================================================================
# DEMONSTRATION RUNNER
# =============================================================================


@dataclass
class DemoMetrics:
    """Metrics collected during demonstration."""

    generation: int
    mean_fitness: float
    max_fitness: float
    diversity: float
    coalition_count: int
    active_narratives: int
    tom_challenge_level: int
    env_complexity: float


def run_generation(
    population: DemoPopulation,
    coevolution: PsychosocialCoevolutionEngine,
    narrative_system: NarrativeEmergenceSystem,
    npcs: Dict[str, BaseNPC],
    ticks_per_generation: int = 100,
    verbose: bool = False,
) -> DemoMetrics:
    """
    Run one generation of co-evolution.
    """
    agent_ids = population.get_agent_ids()
    coevolution.register_agents(agent_ids)

    # Run ticks
    for tick in range(ticks_per_generation):
        coevolution.tick += 1

        # Agent interactions
        for i, agent_id in enumerate(agent_ids):
            agent = population.agents[agent_id]

            # Select random interaction partner
            others = [a for a in agent_ids if a != agent_id]
            if not others:
                continue
            partner_id = random.choice(others)
            partner = population.agents[partner_id]

            # Both agents choose actions
            action1 = agent.select_action({}, None, coevolution)
            action2 = partner.select_action({}, None, coevolution)

            # Process interaction
            outcome = coevolution.process_interaction(agent_id, partner_id, action1, action2)

            # Calculate rewards
            reward1 = 0.1 if action1.get("cooperate") and action2.get("cooperate") else -0.05
            reward2 = 0.1 if action1.get("cooperate") and action2.get("cooperate") else -0.05

            agent.update_fitness(reward1)
            partner.update_fitness(reward2)

        # Update world
        coevolution.tick_world(list(npcs.values()))

        # Update narratives
        narrative_result = narrative_system.tick(npcs, coevolution.tick)

        if verbose and tick % 20 == 0:
            print(f"    Tick {tick}: {narrative_result['active_count']} active narratives")

            if narrative_result["new_narratives"]:
                for summary in narrative_result["new_narratives"]:
                    print(f"      NEW: {summary.split(chr(10))[0]}")

    # Evolve environment based on agent fitness
    fitness_dict = population.get_fitness_dict()
    env_changes = coevolution.evolve_generation(fitness_dict)

    if verbose and env_changes:
        print(f"    Environment evolved: {list(env_changes.keys())}")

    # Calculate metrics
    fitnesses = list(fitness_dict.values())
    state = coevolution.get_state_summary()

    return DemoMetrics(
        generation=0,  # Set by caller
        mean_fitness=np.mean(fitnesses),
        max_fitness=np.max(fitnesses),
        diversity=np.std(fitnesses),
        coalition_count=state["social_network"]["num_coalitions"],
        active_narratives=len(narrative_system.get_active_narratives()),
        tom_challenge_level=state["tom_challenge_level"],
        env_complexity=sum(state["environment"]["npc_parameters"].values())
        / len(state["environment"]["npc_parameters"]),
    )


def create_demo_npcs(count: int = 50) -> Dict[str, BaseNPC]:
    """Create demo NPCs for the environment."""
    npcs = {}

    archetypes = [
        "trusting_cooperator",
        "suspicious_defector",
        "conditional_reciprocator",
        "status_seeker",
        "coalition_builder",
        "lone_wolf",
    ]

    for i in range(count):
        npc_id = f"npc_{i}"
        archetype = random.choice(archetypes)

        # Create NPC with soul map
        npc = BaseNPC(
            npc_id=npc_id,
            name=f"NPC {i}",
            archetype=archetype,
            soul_map=SoulMap(),
        )

        # Customize based on archetype
        if "cooperator" in archetype:
            npc.soul_map.trust_default = random.uniform(0.7, 0.9)
            npc.soul_map.cooperation_tendency = random.uniform(0.7, 0.9)
        elif "defector" in archetype:
            npc.soul_map.trust_default = random.uniform(0.2, 0.4)
            npc.soul_map.cooperation_tendency = random.uniform(0.2, 0.4)
        elif "status" in archetype:
            npc.soul_map.status_drive = random.uniform(0.7, 0.9)

        # Some NPCs are zombies (for validation)
        if random.random() < 0.1:
            npc.is_zombie = True
            npc.zombie_type = random.choice(["behavioral", "belief", "causal", "metacognitive"])

        npcs[npc_id] = npc

    return npcs


def print_theoretical_foundations():
    """Print the theoretical foundations of the system."""
    print("\n" + "=" * 70)
    print("THEORETICAL FOUNDATIONS")
    print("=" * 70)

    foundations = [
        ("Dunbar's Number", TheoreticalConstants.DUNBAR_NUMBER, "Cognitive limit on stable social relationships"),
        ("Belief Decay", TheoreticalConstants.BELIEF_CONFIDENCE_DECAY, "Confidence drops with ToM recursion depth"),
        ("Heider Balance", TheoreticalConstants.BALANCE_PRESSURE, "Triadic relationship tend toward balance"),
        ("Reputation Memory", TheoreticalConstants.REPUTATION_DECAY_RATE, "Temporal decay of social information"),
        ("Emotional Contagion", TheoreticalConstants.CONTAGION_RATE, "Affect spreads through proximity"),
        (
            "Coalition Threshold",
            TheoreticalConstants.COALITION_FORMATION_THRESHOLD,
            "Trust required for formal alliance",
        ),
    ]

    for name, value, description in foundations:
        print(f"  {name:.<30} {value:>6.3f}  | {description}")

    print("=" * 70 + "\n")


def print_generation_report(gen: int, metrics: DemoMetrics, narrative_system: NarrativeEmergenceSystem):
    """Print detailed generation report."""
    print(f"\n{'='*60}")
    print(f"GENERATION {gen} REPORT")
    print(f"{'='*60}")

    print("\n  Population Fitness:")
    print(f"    Mean:     {metrics.mean_fitness:>8.4f}")
    print(f"    Max:      {metrics.max_fitness:>8.4f}")
    print(f"    Std Dev:  {metrics.diversity:>8.4f}")

    print("\n  Social Dynamics:")
    print(f"    Coalitions:        {metrics.coalition_count:>3}")
    print(f"    Active Narratives: {metrics.active_narratives:>3}")
    print(f"    ToM Challenge:     {metrics.tom_challenge_level:>3}")

    print(f"\n  Environment Complexity: {metrics.env_complexity:.3f}")

    # Print active narratives
    active = narrative_system.get_active_narratives()
    if active:
        print("\n  Active Narratives:")
        for narrative in active[:3]:
            print(f"    - [{narrative.archetype.name}] {narrative.title}")
            print(f"      Act {narrative.current_act}, Tension: {narrative.tension_level:.2f}")

    # Print ToM learning opportunities
    opportunities = narrative_system.get_tom_learning_opportunities()
    if opportunities:
        print("\n  ToM Learning Opportunities:")
        for opp in opportunities[:2]:
            print(f"    - {opp['challenge_description']}")
            print(f"      Requires ToM depth: {opp['tom_depth_required']}")


def run_demo(
    num_generations: int = 10,
    population_size: int = 20,
    ticks_per_generation: int = 100,
    verbose: bool = False,
    seed: Optional[int] = None,
):
    """
    Run the complete psychosocial co-evolution demonstration.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    print("\n" + "=" * 70)
    print("PSYCHOSOCIAL CO-EVOLUTION DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration shows genuine bidirectional co-evolution:")
    print("  - Agents evolve to handle increasingly complex social dynamics")
    print("  - Environment evolves to maintain selection pressure")
    print("  - Narratives emerge from the underlying dynamics")
    print("  - All mechanisms are grounded in cognitive science\n")

    # Print theoretical foundations
    print_theoretical_foundations()

    # Initialize systems
    print("Initializing systems...")

    # Create co-evolution engine with ecological strategy
    coevolution = PsychosocialCoevolutionEngine(
        evolution_strategy=EnvironmentEvolutionStrategy.ECOLOGICAL,
        enable_belief_propagation=True,
        enable_social_dynamics=True,
    )

    # Create narrative emergence system
    narrative_system = NarrativeEmergenceSystem(coevolution)

    # Create NPC population (the environment)
    npcs = create_demo_npcs(50)

    # Create agent population
    population = DemoPopulation(population_size)

    print(f"  Created {len(npcs)} NPCs")
    print(f"  Created {population_size} agents")
    print(f"  Running {num_generations} generations")

    # Run generations
    all_metrics: List[DemoMetrics] = []

    for gen in range(num_generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/{num_generations}")
        print(f"{'='*60}")

        metrics = run_generation(
            population=population,
            coevolution=coevolution,
            narrative_system=narrative_system,
            npcs=npcs,
            ticks_per_generation=ticks_per_generation,
            verbose=verbose,
        )
        metrics.generation = gen + 1
        all_metrics.append(metrics)

        # Print generation report
        print_generation_report(gen + 1, metrics, narrative_system)

        # Evolve agent population
        population.evolve_generation()

        # Check for convergence
        if metrics.tom_challenge_level >= 4:
            print("\n  Environment has reached high complexity!")
            print("  Agents require 4th-order ToM reasoning.")

    # Final summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    print("\n  Evolution Summary:")
    print(f"    Initial mean fitness:    {all_metrics[0].mean_fitness:.4f}")
    print(f"    Final mean fitness:      {all_metrics[-1].mean_fitness:.4f}")
    print(f"    Initial ToM challenge:   {all_metrics[0].tom_challenge_level}")
    print(f"    Final ToM challenge:     {all_metrics[-1].tom_challenge_level}")
    print(f"    Initial env complexity:  {all_metrics[0].env_complexity:.3f}")
    print(f"    Final env complexity:    {all_metrics[-1].env_complexity:.3f}")

    narrative_metrics = narrative_system.get_narrative_metrics()
    print("\n  Narrative Emergence:")
    print(f"    Total narratives detected: {narrative_metrics['total_detected']}")
    print(f"    Narratives resolved:       {narrative_metrics['resolved']}")
    print(f"    Average ToM depth required: {narrative_metrics['average_tom_depth']:.1f}")

    print("\n  Top narrative archetypes:")
    sorted_archetypes = sorted(narrative_metrics["by_archetype"].items(), key=lambda x: x[1], reverse=True)
    for archetype, count in sorted_archetypes[:5]:
        if count > 0:
            print(f"    - {archetype}: {count}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    insights = [
        "Bidirectional co-evolution maintains selection pressure",
        "Social dynamics create genuine ToM challenges",
        "Narratives emerge naturally from principled mechanisms",
        "All dynamics are grounded in cognitive science research",
        "The system is simultaneously meaningful, entertaining, and rigorous",
    ]

    for insight in insights:
        print(f"  * {insight}")

    print("\n" + "=" * 70 + "\n")

    return all_metrics


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Psychosocial Co-Evolution Demonstration")
    parser.add_argument("--generations", "-g", type=int, default=10, help="Number of generations to run")
    parser.add_argument("--population", "-p", type=int, default=20, help="Agent population size")
    parser.add_argument("--ticks", "-t", type=int, default=100, help="Ticks per generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    run_demo(
        num_generations=args.generations,
        population_size=args.population,
        ticks_per_generation=args.ticks,
        verbose=args.verbose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
