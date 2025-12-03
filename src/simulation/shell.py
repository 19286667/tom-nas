"""
ToM-NAS Simulation Shell

Interactive command-line interface for running experiments,
configuring simulations, and monitoring evolution.
"""

import sys
import os
import cmd
import readline
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import time

# Add paths
sys.path.insert(0, '/home/user/tom-nas/src')

from simulation.integrated_agent import IntegratedAgent
from simulation.world import SimulationWorld, Action, ActionType
from simulation.benchmark_embedding import EmbeddedBenchmark
from simulation.poet_engine import POETEngine, AgentArchitectureGenome, EnvironmentGene
from taxonomy.psychosocial import PsychosocialProfile
from taxonomy.success import SuccessState
from taxonomy.institutions import InstitutionalContext
from taxonomy.sampling import AgentSampler, EnvironmentSampler


@dataclass
class SimulationConfig:
    """Configuration for simulation experiments"""
    # World settings
    world_size: int = 50
    num_agents: int = 10
    num_zombies: int = 2

    # Evolution settings
    population_size: int = 20
    num_generations: int = 100
    mutation_rate: float = 0.1

    # Benchmark settings
    benchmark_embed_rate: float = 0.1
    benchmarks_enabled: bool = True

    # Display settings
    verbose: bool = True
    display_interval: int = 10

    # Random seed
    seed: Optional[int] = None


class SimulationRunner:
    """Runs and manages simulations"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Components
        self.world: Optional[SimulationWorld] = None
        self.agents: Dict[int, IntegratedAgent] = {}
        self.benchmarks: Optional[EmbeddedBenchmark] = None
        self.poet: Optional[POETEngine] = None

        # State
        self.running = False
        self.paused = False
        self.timestep = 0

    def initialize_world(self):
        """Initialize the simulation world"""
        self.world = SimulationWorld(
            width=self.config.world_size,
            height=self.config.world_size,
            visibility_range=10.0,
        )

        if self.config.benchmarks_enabled:
            self.benchmarks = EmbeddedBenchmark(
                embed_rate=self.config.benchmark_embed_rate
            )

    def spawn_agents(self):
        """Spawn agents in the world"""
        sampler = AgentSampler(seed=self.config.seed)

        # Spawn regular agents
        num_regular = self.config.num_agents - self.config.num_zombies
        regular_agents = sampler.sample_population(num_regular, include_archetypes=True)

        for i, (profile, success) in enumerate(regular_agents):
            agent = IntegratedAgent(
                id=i,
                profile=profile,
                success=success,
                institutional_context=InstitutionalContext.sample_random(self.rng),
            )
            self.agents[i] = agent
            self.world.add_agent(agent)

        # Spawn zombies
        zombie_agents = sampler.sample_zombie_agents(self.config.num_zombies)
        for j, (profile, success) in enumerate(zombie_agents):
            agent = IntegratedAgent(
                id=num_regular + j,
                profile=profile,
                success=success,
                is_zombie=True,
                zombie_type='behavioral',
            )
            self.agents[num_regular + j] = agent
            self.world.add_agent(agent)

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step"""
        self.timestep += 1
        results = {}

        # Get observations for all agents
        observations = {}
        for agent_id, agent in self.agents.items():
            obs = self.world.get_observation(agent_id)
            observations[agent_id] = obs

        # Agents reason and choose actions
        actions = {}
        reasoning_results = {}
        for agent_id, agent in self.agents.items():
            obs_dict = observations[agent_id].to_dict()
            reasoning = agent.reason(obs_dict)
            reasoning_results[agent_id] = reasoning

            # Convert reasoning to action
            action_dict = reasoning.selected_action
            action = Action(
                agent_id=agent_id,
                type=ActionType(action_dict.get('type', 'rest')),
                target_id=action_dict.get('target'),
            )
            actions[agent_id] = action

        # Execute actions in world
        step_results = self.world.step(actions)

        # Maybe spawn benchmark
        if self.benchmarks:
            scenario = self.benchmarks.maybe_spawn_benchmark(self.world, self.timestep)
            if scenario:
                results['new_scenario'] = scenario.benchmark_type.value

            # Evaluate benchmarks
            predictions = {
                aid: {'predicted_search_location': 'initial'}
                for aid in self.agents
            }
            bench_results = self.benchmarks.evaluate_all(self.world, predictions)
            if bench_results:
                results['benchmark_results'] = [
                    {'type': r.benchmark_type.value, 'score': r.agent_score, 'passed': r.passed}
                    for r in bench_results
                ]

        # Periodic reflection
        if self.timestep % 50 == 0:
            for agent in self.agents.values():
                if not agent.is_zombie:
                    agent.reflect(self.timestep)

        results['timestep'] = self.timestep
        results['world_stats'] = self.world.get_statistics()

        return results

    def run(self, num_steps: int = 1000, callback: Optional[callable] = None):
        """Run simulation for specified steps"""
        self.running = True

        for _ in range(num_steps):
            if not self.running:
                break
            if self.paused:
                time.sleep(0.1)
                continue

            results = self.step()

            if callback:
                callback(results)

        self.running = False

    def stop(self):
        """Stop the simulation"""
        self.running = False

    def pause(self):
        """Pause the simulation"""
        self.paused = True

    def resume(self):
        """Resume the simulation"""
        self.paused = False


class ToMNASShell(cmd.Cmd):
    """Interactive shell for ToM-NAS simulation"""

    intro = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ToM-NAS Simulation Shell                            ║
║                    Theory of Mind Neural Architecture Search                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Commands:                                                                    ║
║    init       - Initialize simulation with current config                     ║
║    run [n]    - Run simulation for n steps (default: 100)                    ║
║    step       - Execute single simulation step                               ║
║    status     - Show simulation status                                        ║
║    agents     - List all agents                                              ║
║    agent <id> - Show details for agent                                       ║
║    config     - Show/modify configuration                                    ║
║    evolve [n] - Run POET evolution for n generations                         ║
║    benchmark  - Show benchmark results                                       ║
║    help       - Show this help                                               ║
║    quit       - Exit shell                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    prompt = 'ToM-NAS> '

    def __init__(self):
        super().__init__()
        self.config = SimulationConfig()
        self.runner: Optional[SimulationRunner] = None
        self.poet: Optional[POETEngine] = None

    def do_init(self, arg):
        """Initialize the simulation world and agents"""
        print("Initializing simulation...")
        self.runner = SimulationRunner(self.config)
        self.runner.initialize_world()
        self.runner.spawn_agents()
        print(f"✓ World initialized: {self.config.world_size}x{self.config.world_size}")
        print(f"✓ Agents spawned: {self.config.num_agents} ({self.config.num_zombies} zombies)")
        print(f"✓ Benchmarks: {'enabled' if self.config.benchmarks_enabled else 'disabled'}")

    def do_run(self, arg):
        """Run simulation for n steps: run [n]"""
        if self.runner is None:
            print("Error: Initialize simulation first with 'init'")
            return

        try:
            num_steps = int(arg) if arg else 100
        except ValueError:
            print("Usage: run [num_steps]")
            return

        print(f"Running simulation for {num_steps} steps...")
        start_time = time.time()

        def callback(results):
            if results['timestep'] % self.config.display_interval == 0:
                stats = results['world_stats']
                print(f"  Step {results['timestep']}: "
                      f"agents={stats['num_agents']}, "
                      f"coop_rate={stats['cooperation_rate']:.1%}, "
                      f"resources={stats['num_resources']}")

                if 'benchmark_results' in results:
                    for br in results['benchmark_results']:
                        status = '✓' if br['passed'] else '✗'
                        print(f"    {status} Benchmark {br['type']}: {br['score']:.2f}")

        self.runner.run(num_steps, callback)
        elapsed = time.time() - start_time
        print(f"Completed {num_steps} steps in {elapsed:.2f}s")

    def do_step(self, arg):
        """Execute a single simulation step"""
        if self.runner is None:
            print("Error: Initialize simulation first with 'init'")
            return

        results = self.runner.step()
        print(f"Step {results['timestep']}:")
        stats = results['world_stats']
        print(f"  Agents: {stats['num_agents']}")
        print(f"  Resources: {stats['num_resources']}")
        print(f"  Cooperation rate: {stats['cooperation_rate']:.1%}")
        print(f"  Interactions: {stats['total_interactions']}")

    def do_status(self, arg):
        """Show current simulation status"""
        if self.runner is None:
            print("Simulation not initialized. Use 'init' to start.")
            return

        print("\n" + self.runner.world.describe())

        if self.runner.benchmarks:
            print("\n" + self.runner.benchmarks.get_summary())

    def do_agents(self, arg):
        """List all agents"""
        if self.runner is None:
            print("Error: Initialize simulation first")
            return

        print(f"\n{'ID':>4} {'Type':>10} {'ToM':>5} {'Fitness':>8} {'Position':>15} {'Trust':>6}")
        print("-" * 60)

        for agent_id, agent in sorted(self.runner.agents.items()):
            agent_type = 'ZOMBIE' if agent.is_zombie else 'Normal'
            tom = agent.profile.layer3.tom_depth
            fitness = agent.fitness
            pos = f"({agent.x:.1f}, {agent.y:.1f})"
            trust = agent.profile.layer6.trust_default

            print(f"{agent_id:>4} {agent_type:>10} {tom:>5} {fitness:>8.3f} {pos:>15} {trust:>6.0f}")

    def do_agent(self, arg):
        """Show details for specific agent: agent <id>"""
        if self.runner is None:
            print("Error: Initialize simulation first")
            return

        try:
            agent_id = int(arg)
        except ValueError:
            print("Usage: agent <id>")
            return

        if agent_id not in self.runner.agents:
            print(f"Agent {agent_id} not found")
            return

        agent = self.runner.agents[agent_id]
        print(agent.describe())
        print("\n" + agent.profile.describe())
        print("\n" + agent.success.describe())

    def do_config(self, arg):
        """Show or modify configuration: config [param] [value]"""
        parts = arg.split() if arg else []

        if len(parts) == 0:
            print("\n=== Current Configuration ===")
            print(f"  world_size:          {self.config.world_size}")
            print(f"  num_agents:          {self.config.num_agents}")
            print(f"  num_zombies:         {self.config.num_zombies}")
            print(f"  population_size:     {self.config.population_size}")
            print(f"  num_generations:     {self.config.num_generations}")
            print(f"  mutation_rate:       {self.config.mutation_rate}")
            print(f"  benchmark_embed_rate: {self.config.benchmark_embed_rate}")
            print(f"  benchmarks_enabled:  {self.config.benchmarks_enabled}")
            print(f"  verbose:             {self.config.verbose}")
            print(f"  display_interval:    {self.config.display_interval}")
            print(f"  seed:                {self.config.seed}")
            print("\nUse 'config <param> <value>' to modify")

        elif len(parts) == 2:
            param, value = parts
            if hasattr(self.config, param):
                old_value = getattr(self.config, param)
                try:
                    if isinstance(old_value, bool):
                        new_value = value.lower() in ('true', '1', 'yes')
                    elif isinstance(old_value, int):
                        new_value = int(value)
                    elif isinstance(old_value, float):
                        new_value = float(value)
                    else:
                        new_value = value

                    setattr(self.config, param, new_value)
                    print(f"Set {param}: {old_value} -> {new_value}")
                except ValueError as e:
                    print(f"Error: {e}")
            else:
                print(f"Unknown parameter: {param}")
        else:
            print("Usage: config [param] [value]")

    def do_evolve(self, arg):
        """Run POET co-evolution: evolve [generations]"""
        try:
            num_gens = int(arg) if arg else 10
        except ValueError:
            print("Usage: evolve [generations]")
            return

        print("Initializing POET co-evolution engine...")
        self.poet = POETEngine(
            population_size=self.config.population_size,
            mutation_rate=self.config.mutation_rate,
            seed=self.config.seed,
        )
        self.poet.initialize_population()

        print(f"Running evolution for {num_gens} generations...")

        def simple_evaluator(agent_genome, env_genome):
            """Simple fitness evaluation for testing"""
            # Simulate: higher ToM + appropriate difficulty = better fitness
            tom_depth = agent_genome.architecture.tom_depth
            difficulty = env_genome.difficulty
            tom_pressure = env_genome.get_tom_pressure()

            # Fitness: ToM should match pressure
            fit = 1.0 - abs(tom_depth / 5.0 - tom_pressure)

            # Penalize complexity
            complexity = agent_genome.architecture.complexity_score()
            fit -= complexity * 0.1

            # Benchmark scores (simulated)
            bench_scores = {
                'false_belief': min(1.0, tom_depth / 3.0),
                'cooperation': 0.5 + 0.1 * tom_depth,
                'zombie_detection': 0.3 + 0.15 * tom_depth,
            }

            return max(0, fit), bench_scores

        for gen in range(num_gens):
            stats = self.poet.evolve_generation(simple_evaluator)
            print(f"  Gen {stats['generation']:>3}: "
                  f"best={stats['best_fitness']:.3f}, "
                  f"mean={stats['mean_fitness']:.3f}, "
                  f"diversity={stats['diversity']:.3f}, "
                  f"best_tom={stats['best_tom_depth']}")

        print("\n" + self.poet.describe())

    def do_benchmark(self, arg):
        """Show benchmark results"""
        if self.runner is None or self.runner.benchmarks is None:
            print("No benchmark data available")
            return

        print(self.runner.benchmarks.get_summary())

    def do_quit(self, arg):
        """Exit the shell"""
        print("Goodbye!")
        return True

    def do_exit(self, arg):
        """Exit the shell"""
        return self.do_quit(arg)

    def do_help(self, arg):
        """Show help"""
        if arg:
            super().do_help(arg)
        else:
            print(self.intro)

    def emptyline(self):
        """Do nothing on empty line"""
        pass

    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands")


def run_shell():
    """Main entry point for the shell"""
    try:
        shell = ToMNASShell()
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")


if __name__ == "__main__":
    run_shell()
