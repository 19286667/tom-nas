"""
ToM-NAS Visualization System

Terminal-based ASCII renderer for the 3D simulation world.
Displays agents, resources, social interactions, and mental states.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import os
import sys
import time


class ColorCode(Enum):
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


@dataclass
class RenderConfig:
    """Configuration for the renderer."""
    use_colors: bool = True
    show_beliefs: bool = True
    show_resources: bool = True
    show_interactions: bool = True
    show_stats: bool = True
    cell_width: int = 3
    viewport_width: int = 40
    viewport_height: int = 20
    center_on_agent: Optional[int] = None
    animation_delay: float = 0.1


class TerminalRenderer:
    """ASCII-based terminal renderer for the simulation world."""

    # Agent display characters by ToM level
    AGENT_CHARS = {
        0: 'o',  # No ToM (basic reactive)
        1: '@',  # Level 1 (simple beliefs)
        2: '#',  # Level 2 (beliefs about beliefs)
        3: '$',  # Level 3 (deep recursion)
        4: '&',  # Level 4 (very deep)
        5: '*',  # Level 5 (maximum depth)
    }

    # Resource display characters
    RESOURCE_CHARS = {
        'food': 'F',
        'water': 'W',
        'shelter': 'H',
        'tool': 'T',
        'information': 'I',
        'social': 'S',
        'wealth': 'G',  # Gold
        'default': '?',
    }

    # Interaction symbols
    INTERACTION_CHARS = {
        'cooperate': '+',
        'defect': '-',
        'communicate': '~',
        'observe': '.',
        'trade': '$',
        'conflict': 'X',
    }

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.frame_buffer: List[str] = []
        self.last_interactions: List[Dict] = []
        self.stats_history: List[Dict] = []

    def _color(self, text: str, color: ColorCode, bold: bool = False) -> str:
        """Apply color to text if colors are enabled."""
        if not self.config.use_colors:
            return text
        prefix = ColorCode.BOLD.value if bold else ""
        return f"{prefix}{color.value}{text}{ColorCode.RESET.value}"

    def _get_agent_color(self, agent: Any) -> ColorCode:
        """Get color for agent based on their state."""
        if hasattr(agent, 'is_zombie') and agent.is_zombie:
            return ColorCode.DIM

        # Color by ToM level
        tom_level = getattr(agent, 'tom_level', 0)
        if hasattr(agent, 'tom_reasoner'):
            tom_level = agent.tom_reasoner.k_level

        colors = {
            0: ColorCode.WHITE,
            1: ColorCode.CYAN,
            2: ColorCode.GREEN,
            3: ColorCode.YELLOW,
            4: ColorCode.MAGENTA,
            5: ColorCode.BRIGHT_CYAN,
        }
        return colors.get(tom_level, ColorCode.WHITE)

    def _get_resource_color(self, resource_type: str) -> ColorCode:
        """Get color for resource type."""
        colors = {
            'food': ColorCode.GREEN,
            'water': ColorCode.BLUE,
            'shelter': ColorCode.YELLOW,
            'tool': ColorCode.WHITE,
            'information': ColorCode.CYAN,
            'social': ColorCode.MAGENTA,
            'wealth': ColorCode.BRIGHT_YELLOW,
        }
        return colors.get(resource_type.lower(), ColorCode.WHITE)

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def render_world(self, world: Any, agents: Dict[int, Any],
                     step: int = 0, show_header: bool = True) -> str:
        """Render the entire world state to a string."""
        lines = []

        if show_header:
            lines.extend(self._render_header(world, step, len(agents)))

        # Build the world grid
        grid_lines = self._render_grid(world, agents)
        lines.extend(grid_lines)

        if self.config.show_stats:
            lines.append("")
            lines.extend(self._render_stats(world, agents))

        if self.config.show_interactions and self.last_interactions:
            lines.append("")
            lines.extend(self._render_interactions())

        return "\n".join(lines)

    def _render_header(self, world: Any, step: int, num_agents: int) -> List[str]:
        """Render the header with simulation info."""
        width = getattr(world, 'width', 50)
        height = getattr(world, 'height', 50)
        depth = getattr(world, 'depth', 1)

        header = self._color("=" * 60, ColorCode.CYAN)
        title = self._color(f" ToM-NAS Simulation - Step {step} ", ColorCode.BRIGHT_CYAN, bold=True)
        info = f" World: {width}x{height}x{depth} | Agents: {num_agents}"
        footer = self._color("=" * 60, ColorCode.CYAN)

        return [header, title, info, footer, ""]

    def _render_grid(self, world: Any, agents: Dict[int, Any]) -> List[str]:
        """Render the world grid with agents and resources."""
        width = getattr(world, 'width', 50)
        height = getattr(world, 'height', 50)

        # Determine viewport
        vw = min(self.config.viewport_width, width)
        vh = min(self.config.viewport_height, height)

        # Calculate viewport offset
        offset_x, offset_y = 0, 0
        if self.config.center_on_agent and self.config.center_on_agent in agents:
            agent = agents[self.config.center_on_agent]
            if hasattr(agent, 'position'):
                pos = agent.position
                offset_x = max(0, min(pos[0] - vw // 2, width - vw))
                offset_y = max(0, min(pos[1] - vh // 2, height - vh))

        # Build position lookup for agents
        agent_positions: Dict[Tuple[int, int, int], List[Any]] = {}
        for agent_id, agent in agents.items():
            if hasattr(agent, 'position'):
                pos = tuple(agent.position) if hasattr(agent.position, '__iter__') else (0, 0, 0)
                if len(pos) == 2:
                    pos = (pos[0], pos[1], 0)
                if pos not in agent_positions:
                    agent_positions[pos] = []
                agent_positions[pos].append(agent)

        # Build position lookup for resources
        resource_positions: Dict[Tuple[int, int, int], List[Any]] = {}
        if hasattr(world, 'resources'):
            for resource in world.resources:
                if hasattr(resource, 'position'):
                    pos = tuple(resource.position) if hasattr(resource.position, '__iter__') else (0, 0, 0)
                    if len(pos) == 2:
                        pos = (pos[0], pos[1], 0)
                    if pos not in resource_positions:
                        resource_positions[pos] = []
                    resource_positions[pos].append(resource)

        lines = []

        # Top border
        border_top = "+" + "-" * (vw * self.config.cell_width) + "+"
        lines.append(self._color(border_top, ColorCode.DIM))

        # Grid rows
        for y in range(offset_y, offset_y + vh):
            row_chars = []
            for x in range(offset_x, offset_x + vw):
                cell = self._render_cell(x, y, 0, agent_positions, resource_positions)
                row_chars.append(cell)

            row_str = "|" + "".join(row_chars) + "|"
            lines.append(row_str)

        # Bottom border
        border_bottom = "+" + "-" * (vw * self.config.cell_width) + "+"
        lines.append(self._color(border_bottom, ColorCode.DIM))

        # Coordinates
        coord_str = f"  ({offset_x},{offset_y}) to ({offset_x + vw - 1},{offset_y + vh - 1})"
        lines.append(self._color(coord_str, ColorCode.DIM))

        return lines

    def _render_cell(self, x: int, y: int, z: int,
                     agent_positions: Dict[Tuple[int, int, int], List],
                     resource_positions: Dict[Tuple[int, int, int], List]) -> str:
        """Render a single grid cell."""
        pos = (x, y, z)
        cw = self.config.cell_width

        # Check for agents at this position
        if pos in agent_positions:
            agents_here = agent_positions[pos]
            agent = agents_here[0]  # Show first agent

            # Get display character
            tom_level = 0
            if hasattr(agent, 'tom_reasoner'):
                tom_level = agent.tom_reasoner.k_level
            elif hasattr(agent, 'tom_level'):
                tom_level = agent.tom_level

            char = self.AGENT_CHARS.get(tom_level, '@')
            color = self._get_agent_color(agent)

            # Mark zombies differently
            if hasattr(agent, 'is_zombie') and agent.is_zombie:
                char = 'Z'

            # Show count if multiple agents
            if len(agents_here) > 1:
                display = f"{char}{len(agents_here)}"[:cw]
            else:
                display = char.center(cw)

            return self._color(display, color, bold=True)

        # Check for resources at this position
        if pos in resource_positions and self.config.show_resources:
            resources_here = resource_positions[pos]
            resource = resources_here[0]

            res_type = getattr(resource, 'type', 'default')
            if hasattr(res_type, 'value'):
                res_type = res_type.value

            char = self.RESOURCE_CHARS.get(res_type.lower(), '?')
            color = self._get_resource_color(res_type)

            if len(resources_here) > 1:
                display = f"{char}{len(resources_here)}"[:cw]
            else:
                display = char.center(cw)

            return self._color(display, color)

        # Empty cell
        return " " * cw

    def _render_stats(self, world: Any, agents: Dict[int, Any]) -> List[str]:
        """Render statistics about the simulation."""
        lines = []

        lines.append(self._color("Statistics:", ColorCode.CYAN, bold=True))

        # Count agents by ToM level
        tom_counts = {i: 0 for i in range(6)}
        zombie_count = 0
        total_fitness = 0.0

        for agent in agents.values():
            tom_level = 0
            if hasattr(agent, 'tom_reasoner'):
                tom_level = agent.tom_reasoner.k_level
            elif hasattr(agent, 'tom_level'):
                tom_level = agent.tom_level
            tom_counts[tom_level] = tom_counts.get(tom_level, 0) + 1

            if hasattr(agent, 'is_zombie') and agent.is_zombie:
                zombie_count += 1

            if hasattr(agent, 'success') and hasattr(agent.success, 'compute_fitness'):
                if hasattr(agent, 'profile') and hasattr(agent.profile, 'get_success_weights'):
                    weights = agent.profile.get_success_weights()
                    total_fitness += agent.success.compute_fitness(weights)

        avg_fitness = total_fitness / len(agents) if agents else 0.0

        # ToM distribution
        tom_str = " | ".join([f"L{k}:{v}" for k, v in tom_counts.items() if v > 0])
        lines.append(f"  ToM Levels: {tom_str}")
        lines.append(f"  Zombies: {zombie_count} | Avg Fitness: {avg_fitness:.3f}")

        # Resource count
        if hasattr(world, 'resources'):
            lines.append(f"  Resources: {len(world.resources)}")

        return lines

    def _render_interactions(self) -> List[str]:
        """Render recent interactions."""
        lines = []
        lines.append(self._color("Recent Interactions:", ColorCode.YELLOW, bold=True))

        for interaction in self.last_interactions[-5:]:  # Show last 5
            agent1 = interaction.get('agent1', '?')
            agent2 = interaction.get('agent2', '?')
            action = interaction.get('action', 'interact')
            outcome = interaction.get('outcome', '')

            symbol = self.INTERACTION_CHARS.get(action, '?')
            line = f"  Agent {agent1} {symbol} Agent {agent2}"
            if outcome:
                line += f" -> {outcome}"
            lines.append(line)

        return lines

    def add_interaction(self, agent1_id: int, agent2_id: int,
                       action: str, outcome: str = ""):
        """Record an interaction for display."""
        self.last_interactions.append({
            'agent1': agent1_id,
            'agent2': agent2_id,
            'action': action,
            'outcome': outcome,
        })
        # Keep only last 20 interactions
        if len(self.last_interactions) > 20:
            self.last_interactions = self.last_interactions[-20:]

    def render_agent_detail(self, agent: Any) -> str:
        """Render detailed view of a single agent."""
        lines = []

        agent_id = getattr(agent, 'id', '?')
        lines.append(self._color(f"=== Agent {agent_id} Detail ===", ColorCode.CYAN, bold=True))

        # Position
        if hasattr(agent, 'position'):
            pos = agent.position
            lines.append(f"Position: ({pos[0]}, {pos[1]}, {pos[2] if len(pos) > 2 else 0})")

        # ToM level
        tom_level = 0
        if hasattr(agent, 'tom_reasoner'):
            tom_level = agent.tom_reasoner.k_level
        lines.append(f"ToM Level: {tom_level}")

        # Zombie status
        if hasattr(agent, 'is_zombie'):
            status = self._color("ZOMBIE", ColorCode.RED) if agent.is_zombie else self._color("Active", ColorCode.GREEN)
            lines.append(f"Status: {status}")

        # Psychosocial profile summary
        if hasattr(agent, 'profile'):
            lines.append("")
            lines.append(self._color("Psychosocial Profile:", ColorCode.MAGENTA, bold=True))
            profile = agent.profile

            # Show key traits
            if hasattr(profile, 'layer1'):  # Affective
                layer1 = profile.layer1
                lines.append(f"  Emotional Range: {getattr(layer1, 'emotional_range', 0.5):.2f}")

            if hasattr(profile, 'layer3'):  # Cognitive
                layer3 = profile.layer3
                lines.append(f"  Intelligence: {getattr(layer3, 'fluid_intelligence', 0.5):.2f}")

            if hasattr(profile, 'layer6'):  # Social
                layer6 = profile.layer6
                lines.append(f"  Social Style: {getattr(layer6, 'interaction_style', 'unknown')}")

        # Success state
        if hasattr(agent, 'success'):
            lines.append("")
            lines.append(self._color("Success State:", ColorCode.GREEN, bold=True))
            success = agent.success

            if hasattr(success, 'get_domain_scores'):
                scores = success.get_domain_scores()
                for domain, score in list(scores.items())[:5]:
                    bar = self._render_bar(score, 20)
                    lines.append(f"  {domain[:12]:12s} {bar} {score:.2f}")

        # Beliefs
        if self.config.show_beliefs and hasattr(agent, 'belief_tracker'):
            lines.append("")
            lines.append(self._color("Beliefs:", ColorCode.CYAN, bold=True))
            tracker = agent.belief_tracker

            if hasattr(tracker, 'beliefs_about'):
                for other_id, beliefs in list(tracker.beliefs_about.items())[:3]:
                    lines.append(f"  About Agent {other_id}:")
                    for key, value in list(beliefs.items())[:2]:
                        lines.append(f"    {key}: {value}")

        # Memory summary
        if hasattr(agent, 'memory'):
            lines.append("")
            lines.append(self._color("Memory:", ColorCode.YELLOW, bold=True))
            memory = agent.memory

            episodic = len(getattr(memory, 'episodic', []))
            semantic = len(getattr(memory, 'semantic', {}))
            relational = len(getattr(memory, 'relational', {}))

            lines.append(f"  Episodic: {episodic} | Semantic: {semantic} | Relational: {relational}")

        return "\n".join(lines)

    def _render_bar(self, value: float, width: int = 20) -> str:
        """Render a progress bar."""
        filled = int(value * width)
        empty = width - filled

        bar = "[" + "#" * filled + " " * empty + "]"

        if self.config.use_colors:
            if value < 0.3:
                return self._color(bar, ColorCode.RED)
            elif value < 0.7:
                return self._color(bar, ColorCode.YELLOW)
            else:
                return self._color(bar, ColorCode.GREEN)
        return bar

    def render_beliefs_network(self, agents: Dict[int, Any]) -> str:
        """Render a network view of beliefs between agents."""
        lines = []
        lines.append(self._color("=== Belief Network ===", ColorCode.CYAN, bold=True))

        # Build adjacency representation
        for agent_id, agent in list(agents.items())[:10]:  # Limit to 10 agents
            if hasattr(agent, 'belief_tracker') and hasattr(agent.belief_tracker, 'beliefs_about'):
                beliefs_about = agent.belief_tracker.beliefs_about
                if beliefs_about:
                    targets = list(beliefs_about.keys())[:5]
                    target_str = ", ".join([str(t) for t in targets])
                    lines.append(f"  Agent {agent_id} -> [{target_str}]")

        return "\n".join(lines)

    def animate(self, world: Any, agents: Dict[int, Any],
                steps: int, step_fn=None):
        """Run animated display of simulation steps."""
        for step in range(steps):
            self.clear_screen()

            # Execute step if function provided
            if step_fn:
                step_fn()

            # Render current state
            output = self.render_world(world, agents, step)
            print(output)

            time.sleep(self.config.animation_delay)


class DetailedRenderer(TerminalRenderer):
    """Extended renderer with more detailed information."""

    def render_world(self, world: Any, agents: Dict[int, Any],
                     step: int = 0, show_header: bool = True) -> str:
        """Render world with additional detail panels."""
        lines = []

        # Main world view
        main_view = super().render_world(world, agents, step, show_header)
        lines.append(main_view)

        # Add legend
        lines.append("")
        lines.extend(self._render_legend())

        # Add top performers
        lines.append("")
        lines.extend(self._render_top_agents(agents))

        return "\n".join(lines)

    def _render_legend(self) -> List[str]:
        """Render a legend for the display symbols."""
        lines = []
        lines.append(self._color("Legend:", ColorCode.DIM, bold=True))

        # Agents
        agent_legend = "  Agents: "
        for level, char in sorted(self.AGENT_CHARS.items()):
            agent_legend += f"L{level}={char} "
        agent_legend += "Z=Zombie"
        lines.append(agent_legend)

        # Resources
        resource_legend = "  Resources: "
        for res_type, char in list(self.RESOURCE_CHARS.items())[:6]:
            resource_legend += f"{res_type[0].upper()}={char} "
        lines.append(resource_legend)

        return lines

    def _render_top_agents(self, agents: Dict[int, Any], top_n: int = 3) -> List[str]:
        """Render the top performing agents."""
        lines = []
        lines.append(self._color("Top Performers:", ColorCode.GREEN, bold=True))

        # Calculate fitness for each agent
        fitness_scores = []
        for agent_id, agent in agents.items():
            if hasattr(agent, 'success') and hasattr(agent.success, 'compute_fitness'):
                if hasattr(agent, 'profile') and hasattr(agent.profile, 'get_success_weights'):
                    weights = agent.profile.get_success_weights()
                    fitness = agent.success.compute_fitness(weights)
                    fitness_scores.append((agent_id, fitness, agent))

        # Sort and display top N
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (agent_id, fitness, agent) in enumerate(fitness_scores[:top_n]):
            tom_level = 0
            if hasattr(agent, 'tom_reasoner'):
                tom_level = agent.tom_reasoner.k_level

            zombie = " (Z)" if getattr(agent, 'is_zombie', False) else ""
            lines.append(f"  {i+1}. Agent {agent_id} (L{tom_level}){zombie}: {fitness:.3f}")

        return lines


class MinimalRenderer(TerminalRenderer):
    """Minimal renderer for low-bandwidth/simple output."""

    def __init__(self, config: Optional[RenderConfig] = None):
        config = config or RenderConfig()
        config.use_colors = False
        config.show_beliefs = False
        config.show_interactions = False
        config.cell_width = 1
        super().__init__(config)

    def render_world(self, world: Any, agents: Dict[int, Any],
                     step: int = 0, show_header: bool = True) -> str:
        """Minimal render of the world."""
        lines = []

        if show_header:
            lines.append(f"Step {step} | Agents: {len(agents)}")

        # Simple grid
        grid_lines = self._render_grid(world, agents)
        lines.extend(grid_lines)

        return "\n".join(lines)


# Factory function
def create_renderer(mode: str = 'standard') -> TerminalRenderer:
    """Create a renderer based on the specified mode."""
    modes = {
        'standard': TerminalRenderer,
        'detailed': DetailedRenderer,
        'minimal': MinimalRenderer,
    }

    renderer_class = modes.get(mode.lower(), TerminalRenderer)
    return renderer_class()


# Demo function
def demo_visualization():
    """Demonstrate the visualization system with mock data."""
    from dataclasses import dataclass

    @dataclass
    class MockAgent:
        id: int
        position: Tuple[int, int, int]
        tom_level: int = 1
        is_zombie: bool = False

    @dataclass
    class MockResource:
        position: Tuple[int, int, int]
        type: str = 'food'

    @dataclass
    class MockWorld:
        width: int = 30
        height: int = 15
        depth: int = 1
        resources: List = field(default_factory=list)

    # Create mock world
    world = MockWorld()
    world.resources = [
        MockResource(position=(5, 5, 0), type='food'),
        MockResource(position=(10, 10, 0), type='water'),
        MockResource(position=(15, 7, 0), type='shelter'),
        MockResource(position=(20, 3, 0), type='wealth'),
    ]

    # Create mock agents
    agents = {
        1: MockAgent(id=1, position=(3, 3, 0), tom_level=0),
        2: MockAgent(id=2, position=(8, 5, 0), tom_level=1),
        3: MockAgent(id=3, position=(12, 8, 0), tom_level=2),
        4: MockAgent(id=4, position=(18, 10, 0), tom_level=3),
        5: MockAgent(id=5, position=(25, 5, 0), tom_level=1, is_zombie=True),
    }

    # Create renderer and display
    renderer = create_renderer('detailed')
    renderer.add_interaction(1, 2, 'cooperate', 'mutual gain')
    renderer.add_interaction(2, 3, 'communicate', 'shared info')
    renderer.add_interaction(4, 5, 'defect', 'betrayal')

    output = renderer.render_world(world, agents, step=42)
    print(output)

    print("\n")

    # Show agent detail for one agent
    print(renderer.render_agent_detail(agents[3]))


if __name__ == '__main__':
    demo_visualization()
