"""
ToM-NAS Opening Menu System

Interactive menu for selecting experiments, configuring simulations,
and managing the ToM-NAS research platform.
"""

import sys
import os
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Ensure path is set
sys.path.insert(0, '/home/user/tom-nas/src')


class MenuState(Enum):
    """Current state of the menu system"""
    MAIN = "main"
    EXPERIMENTS = "experiments"
    CONFIG = "config"
    VISUALIZATION = "visualization"
    BENCHMARKS = "benchmarks"
    EVOLUTION = "evolution"
    RUNNING = "running"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str
    world_size: int = 50
    num_agents: int = 10
    num_zombies: int = 2
    num_steps: int = 1000
    benchmark_rate: float = 0.1
    evolution_enabled: bool = False
    evolution_generations: int = 50
    visualization_mode: str = "text"  # text, ascii, or pygame
    seed: Optional[int] = None


# Predefined experiments
PREDEFINED_EXPERIMENTS = {
    'quick_test': ExperimentConfig(
        name="Quick Test",
        description="Fast 100-step test to verify system",
        num_steps=100,
        num_agents=6,
        evolution_enabled=False,
    ),
    'tom_emergence': ExperimentConfig(
        name="ToM Emergence Study",
        description="Evolve agents to observe ToM emergence",
        num_steps=500,
        num_agents=12,
        num_zombies=3,
        evolution_enabled=True,
        evolution_generations=30,
    ),
    'benchmark_validation': ExperimentConfig(
        name="Benchmark Validation",
        description="Run all ToM benchmarks intensively",
        num_steps=2000,
        benchmark_rate=0.3,
        num_agents=8,
    ),
    'zombie_detection': ExperimentConfig(
        name="Zombie Detection Challenge",
        description="Can agents identify zombies (no real ToM)?",
        num_agents=10,
        num_zombies=4,
        num_steps=1000,
    ),
    'social_complexity': ExperimentConfig(
        name="Social Complexity Scaling",
        description="Test ToM with increasing agent count",
        num_agents=20,
        num_zombies=4,
        num_steps=1500,
    ),
    'poet_coevolution': ExperimentConfig(
        name="POET Co-Evolution",
        description="Full POET agent-environment co-evolution",
        evolution_enabled=True,
        evolution_generations=100,
        num_steps=500,
    ),
}


class MenuSystem:
    """Main menu system for ToM-NAS"""

    BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ████████╗ ██████╗ ███╗   ███╗      ███╗   ██╗ █████╗ ███████╗              ║
║  ╚══██╔══╝██╔═══██╗████╗ ████║      ████╗  ██║██╔══██╗██╔════╝              ║
║     ██║   ██║   ██║██╔████╔██║█████╗██╔██╗ ██║███████║███████╗              ║
║     ██║   ██║   ██║██║╚██╔╝██║╚════╝██║╚██╗██║██╔══██║╚════██║              ║
║     ██║   ╚██████╔╝██║ ╚═╝ ██║      ██║ ╚████║██║  ██║███████║              ║
║     ╚═╝    ╚═════╝ ╚═╝     ╚═╝      ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝              ║
║                                                                              ║
║             Theory of Mind - Neural Architecture Search                      ║
║                                                                              ║
║  Discovering neural architectures that enable genuine Theory of Mind         ║
║  through co-evolution with social environments.                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

    def __init__(self):
        self.state = MenuState.MAIN
        self.config = ExperimentConfig(
            name="Custom",
            description="Custom experiment configuration"
        )
        self.running = True

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self, title: str):
        """Print a section header"""
        width = 78
        print("╔" + "═" * width + "╗")
        print("║" + title.center(width) + "║")
        print("╚" + "═" * width + "╝")

    def print_menu(self, options: List[tuple], title: str = "Options"):
        """Print a menu with numbered options"""
        print(f"\n{title}:")
        print("-" * 40)
        for i, (key, label) in enumerate(options, 1):
            print(f"  [{key}] {label}")
        print("-" * 40)

    def get_input(self, prompt: str = "Select: ") -> str:
        """Get user input"""
        try:
            return input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return 'q'

    def run(self):
        """Main menu loop"""
        while self.running:
            self.clear_screen()

            if self.state == MenuState.MAIN:
                self._show_main_menu()
            elif self.state == MenuState.EXPERIMENTS:
                self._show_experiments_menu()
            elif self.state == MenuState.CONFIG:
                self._show_config_menu()
            elif self.state == MenuState.VISUALIZATION:
                self._show_visualization_menu()
            elif self.state == MenuState.BENCHMARKS:
                self._show_benchmarks_menu()
            elif self.state == MenuState.EVOLUTION:
                self._show_evolution_menu()

    def _show_main_menu(self):
        """Show the main menu"""
        print(self.BANNER)

        self.print_menu([
            ('1', 'Run Experiment'),
            ('2', 'Predefined Experiments'),
            ('3', 'Configure Experiment'),
            ('4', 'Visualization Options'),
            ('5', 'Benchmark Settings'),
            ('6', 'Evolution Settings'),
            ('7', 'Interactive Shell'),
            ('8', 'Quick Demo'),
            ('q', 'Quit'),
        ], "Main Menu")

        choice = self.get_input()

        if choice == '1':
            self._run_experiment()
        elif choice == '2':
            self.state = MenuState.EXPERIMENTS
        elif choice == '3':
            self.state = MenuState.CONFIG
        elif choice == '4':
            self.state = MenuState.VISUALIZATION
        elif choice == '5':
            self.state = MenuState.BENCHMARKS
        elif choice == '6':
            self.state = MenuState.EVOLUTION
        elif choice == '7':
            self._run_shell()
        elif choice == '8':
            self._run_quick_demo()
        elif choice == 'q':
            self.running = False

    def _show_experiments_menu(self):
        """Show predefined experiments menu"""
        self.print_header("Predefined Experiments")

        print("\nAvailable Experiments:\n")
        for i, (key, exp) in enumerate(PREDEFINED_EXPERIMENTS.items(), 1):
            print(f"  [{i}] {exp.name}")
            print(f"      {exp.description}")
            print(f"      Agents: {exp.num_agents}, Steps: {exp.num_steps}, "
                  f"Evolution: {'Yes' if exp.evolution_enabled else 'No'}")
            print()

        print("  [b] Back to main menu")
        print()

        choice = self.get_input()

        if choice == 'b':
            self.state = MenuState.MAIN
        else:
            try:
                idx = int(choice) - 1
                exp_key = list(PREDEFINED_EXPERIMENTS.keys())[idx]
                self.config = PREDEFINED_EXPERIMENTS[exp_key]
                self._run_experiment()
            except (ValueError, IndexError):
                pass

    def _show_config_menu(self):
        """Show configuration menu"""
        self.print_header("Experiment Configuration")

        print(f"\nCurrent Configuration: {self.config.name}")
        print("-" * 50)
        print(f"  [1] World Size:        {self.config.world_size}")
        print(f"  [2] Number of Agents:  {self.config.num_agents}")
        print(f"  [3] Number of Zombies: {self.config.num_zombies}")
        print(f"  [4] Simulation Steps:  {self.config.num_steps}")
        print(f"  [5] Benchmark Rate:    {self.config.benchmark_rate}")
        print(f"  [6] Random Seed:       {self.config.seed or 'None (random)'}")
        print("-" * 50)
        print("  [r] Run with this config")
        print("  [s] Save config")
        print("  [b] Back to main menu")
        print()

        choice = self.get_input()

        if choice == 'b':
            self.state = MenuState.MAIN
        elif choice == 'r':
            self._run_experiment()
        elif choice == '1':
            self._edit_value('world_size', int, 20, 200)
        elif choice == '2':
            self._edit_value('num_agents', int, 2, 50)
        elif choice == '3':
            self._edit_value('num_zombies', int, 0, 20)
        elif choice == '4':
            self._edit_value('num_steps', int, 10, 10000)
        elif choice == '5':
            self._edit_value('benchmark_rate', float, 0.0, 1.0)
        elif choice == '6':
            self._edit_seed()

    def _show_visualization_menu(self):
        """Show visualization options menu"""
        self.print_header("Visualization Options")

        modes = ['text', 'ascii', 'detailed']
        current_idx = modes.index(self.config.visualization_mode) if self.config.visualization_mode in modes else 0

        print("\nVisualization Mode:")
        print("-" * 40)
        for i, mode in enumerate(modes, 1):
            marker = "●" if mode == self.config.visualization_mode else "○"
            print(f"  [{i}] {marker} {mode.capitalize()}")
            if mode == 'text':
                print("       Simple text output with statistics")
            elif mode == 'ascii':
                print("       ASCII art world representation")
            elif mode == 'detailed':
                print("       Verbose output with agent details")
        print("-" * 40)
        print("  [b] Back to main menu")
        print()

        choice = self.get_input()

        if choice == 'b':
            self.state = MenuState.MAIN
        else:
            try:
                idx = int(choice) - 1
                self.config.visualization_mode = modes[idx]
                print(f"Set visualization mode to: {modes[idx]}")
                time.sleep(0.5)
            except (ValueError, IndexError):
                pass

    def _show_benchmarks_menu(self):
        """Show benchmark settings menu"""
        self.print_header("Benchmark Settings")

        print("\nToM Benchmark Configuration:")
        print("-" * 50)
        print(f"  [1] Embed Rate:     {self.config.benchmark_rate:.1%}")
        print()
        print("Benchmark Types:")
        print("  • False Belief (Sally-Anne)")
        print("  • Faux Pas Detection")
        print("  • Second-Order Belief")
        print("  • SOTOPIA Social Scenarios")
        print("  • Cooperative/Competitive Tasks")
        print("-" * 50)
        print("  [2] Run benchmark-focused experiment")
        print("  [b] Back to main menu")
        print()

        choice = self.get_input()

        if choice == 'b':
            self.state = MenuState.MAIN
        elif choice == '1':
            self._edit_value('benchmark_rate', float, 0.0, 1.0)
        elif choice == '2':
            self.config = PREDEFINED_EXPERIMENTS['benchmark_validation']
            self._run_experiment()

    def _show_evolution_menu(self):
        """Show evolution settings menu"""
        self.print_header("POET Evolution Settings")

        print("\nCo-Evolution Configuration:")
        print("-" * 50)
        print(f"  [1] Evolution Enabled:  {'Yes' if self.config.evolution_enabled else 'No'}")
        print(f"  [2] Generations:        {self.config.evolution_generations}")
        print("-" * 50)
        print("\nPOET co-evolves:")
        print("  • Agent neural architectures (TRN, RSAN, Transformer)")
        print("  • ToM depth (0-5 levels of recursive reasoning)")
        print("  • Environment complexity (social pressure)")
        print("-" * 50)
        print("  [3] Run evolution experiment")
        print("  [b] Back to main menu")
        print()

        choice = self.get_input()

        if choice == 'b':
            self.state = MenuState.MAIN
        elif choice == '1':
            self.config.evolution_enabled = not self.config.evolution_enabled
        elif choice == '2':
            self._edit_value('evolution_generations', int, 1, 500)
        elif choice == '3':
            self.config = PREDEFINED_EXPERIMENTS['poet_coevolution']
            self._run_experiment()

    def _edit_value(self, attr: str, type_fn, min_val, max_val):
        """Edit a configuration value"""
        current = getattr(self.config, attr)
        print(f"\nCurrent {attr}: {current}")
        print(f"Enter new value ({min_val}-{max_val}): ", end="")

        try:
            new_val = type_fn(input())
            if min_val <= new_val <= max_val:
                setattr(self.config, attr, new_val)
                print(f"Set {attr} to {new_val}")
            else:
                print(f"Value out of range")
        except ValueError:
            print("Invalid input")

        time.sleep(0.5)

    def _edit_seed(self):
        """Edit random seed"""
        print("\nEnter random seed (or 'none' for random): ", end="")
        val = input().strip()
        if val.lower() == 'none':
            self.config.seed = None
        else:
            try:
                self.config.seed = int(val)
            except ValueError:
                print("Invalid seed")
        time.sleep(0.5)

    def _run_experiment(self):
        """Run the configured experiment"""
        self.clear_screen()
        self.print_header(f"Running: {self.config.name}")

        print(f"\nConfiguration:")
        print(f"  World: {self.config.world_size}x{self.config.world_size}")
        print(f"  Agents: {self.config.num_agents} ({self.config.num_zombies} zombies)")
        print(f"  Steps: {self.config.num_steps}")
        print(f"  Benchmark rate: {self.config.benchmark_rate:.1%}")
        print(f"  Evolution: {'Enabled' if self.config.evolution_enabled else 'Disabled'}")
        print()

        print("Press Enter to start, or 'c' to cancel...")
        if self.get_input() == 'c':
            return

        # Import here to avoid circular imports
        from simulation.shell import SimulationRunner, SimulationConfig

        config = SimulationConfig(
            world_size=self.config.world_size,
            num_agents=self.config.num_agents,
            num_zombies=self.config.num_zombies,
            benchmark_embed_rate=self.config.benchmark_rate,
            seed=self.config.seed,
        )

        runner = SimulationRunner(config)
        runner.initialize_world()
        runner.spawn_agents()

        print("\n" + "=" * 60)
        print("SIMULATION RUNNING")
        print("=" * 60 + "\n")

        start_time = time.time()

        def callback(results):
            step = results['timestep']
            stats = results['world_stats']

            if self.config.visualization_mode == 'detailed':
                print(f"\n[Step {step}]")
                print(f"  Agents: {stats['num_agents']}")
                print(f"  Resources: {stats['num_resources']}")
                print(f"  Cooperation: {stats['cooperation_rate']:.1%}")
                print(f"  Interactions: {stats['total_interactions']}")

                if 'benchmark_results' in results:
                    for br in results['benchmark_results']:
                        status = '✓' if br['passed'] else '✗'
                        print(f"  {status} {br['type']}: {br['score']:.2f}")

            elif self.config.visualization_mode == 'ascii':
                self._print_ascii_world(runner)

            else:  # text mode
                if step % 10 == 0:
                    print(f"Step {step:>5}: coop={stats['cooperation_rate']:.1%}, "
                          f"resources={stats['num_resources']}")

        try:
            runner.run(self.config.num_steps, callback)
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted!")

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"\nRan {runner.timestep} steps in {elapsed:.2f}s")
        print(f"Final stats: {runner.world.get_statistics()}")

        if runner.benchmarks:
            print("\n" + runner.benchmarks.get_summary())

        print("\nPress Enter to continue...")
        input()

    def _print_ascii_world(self, runner):
        """Print ASCII representation of the world"""
        world = runner.world
        size = min(30, world.width)  # Limit display size

        print("\n" + "-" * (size + 2))
        for y in range(size):
            row = "|"
            for x in range(size):
                # Check for agent at this position
                agent_here = None
                for agent in runner.agents.values():
                    if int(agent.x) == x and int(agent.y) == y:
                        agent_here = agent
                        break

                if agent_here:
                    if agent_here.is_zombie:
                        row += "Z"
                    else:
                        row += str(agent_here.profile.layer3.tom_depth)
                elif world.terrain[x, y, 0] == 1:  # Wall
                    row += "█"
                else:
                    # Check for resource
                    has_resource = any(
                        not r.depleted and
                        int(r.location.x) == x and
                        int(r.location.y) == y
                        for r in world.resources
                    )
                    row += "*" if has_resource else "."

            row += "|"
            print(row)
        print("-" * (size + 2))
        print("Legend: 0-5=ToM depth, Z=Zombie, █=Wall, *=Resource")

    def _run_shell(self):
        """Launch the interactive shell"""
        self.clear_screen()
        from simulation.shell import run_shell
        run_shell()

    def _run_quick_demo(self):
        """Run a quick demonstration"""
        self.config = PREDEFINED_EXPERIMENTS['quick_test']
        self._run_experiment()


def main():
    """Main entry point"""
    menu = MenuSystem()
    try:
        menu.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
