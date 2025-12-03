"""
3D World Simulation

Grid-based world with:
- Partial observability (can't see through walls)
- Resource distribution
- Interaction resolution (cooperate, defect, etc.)
- Embedded benchmark scenarios
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
from collections import defaultdict


class ActionType(Enum):
    """Types of actions agents can take"""
    MOVE = "move"
    COOPERATE = "cooperate"
    DEFECT = "defect"
    COMMUNICATE = "communicate"
    COLLECT = "collect"
    REST = "rest"
    ANALYZE = "analyze"  # Soul scanner
    PREDICT = "predict"  # ToM prediction


class ResourceType(Enum):
    """Types of resources in the world"""
    FOOD = "food"
    MATERIAL = "material"
    KNOWLEDGE = "knowledge"
    SOCIAL = "social"


class TerrainType(Enum):
    """Types of terrain"""
    OPEN = 0
    WALL = 1
    WATER = 2
    DENSE = 3  # Reduces visibility


@dataclass
class Location:
    """A location in the world"""
    x: int
    y: int
    z: int = 0
    terrain: TerrainType = TerrainType.OPEN

    def distance_to(self, other: 'Location') -> float:
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)


@dataclass
class Resource:
    """A resource in the world"""
    id: int
    location: Location
    type: ResourceType
    value: float
    respawn_rate: float = 0.1
    depleted: bool = False

    def collect(self) -> float:
        """Collect this resource, returns value obtained"""
        if self.depleted:
            return 0.0
        self.depleted = True
        return self.value

    def tick(self):
        """Called each timestep for respawn chance"""
        if self.depleted and np.random.random() < self.respawn_rate:
            self.depleted = False


@dataclass
class Observation:
    """What an agent can observe"""
    self_state: Dict[str, Any]
    nearby_agents: List[Dict[str, Any]]
    nearby_resources: List[Dict[str, Any]]
    visible_terrain: List[Tuple[int, int, int, TerrainType]]
    timestamp: int

    def to_dict(self) -> Dict:
        return {
            'self': self.self_state,
            'nearby_agents': self.nearby_agents,
            'nearby_resources': self.nearby_resources,
            'timestep': self.timestamp,
        }


@dataclass
class Action:
    """An action taken by an agent"""
    agent_id: int
    type: ActionType
    target_id: Optional[int] = None
    target_location: Optional[Location] = None
    message: Optional[str] = None
    prediction: Optional[Dict] = None


@dataclass
class InteractionResult:
    """Result of an interaction between agents"""
    agent1_id: int
    agent2_id: int
    agent1_action: ActionType
    agent2_action: ActionType
    agent1_payoff: float
    agent2_payoff: float
    description: str
    timestamp: int


@dataclass
class WorldState:
    """Complete state of the world at a moment"""
    timestep: int
    agents: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    terrain: np.ndarray  # 3D grid of terrain types
    interactions: List[InteractionResult]
    events: List[Dict[str, Any]]


class PayoffMatrix:
    """Payoff matrix for game-theoretic interactions"""

    def __init__(self, matrix_type: str = "prisoners_dilemma"):
        if matrix_type == "prisoners_dilemma":
            # (my payoff, their payoff)
            self.matrix = {
                ('cooperate', 'cooperate'): (3, 3),  # Reward
                ('cooperate', 'defect'): (0, 5),     # Sucker
                ('defect', 'cooperate'): (5, 0),     # Temptation
                ('defect', 'defect'): (1, 1),        # Punishment
            }
        elif matrix_type == "stag_hunt":
            self.matrix = {
                ('cooperate', 'cooperate'): (5, 5),
                ('cooperate', 'defect'): (0, 3),
                ('defect', 'cooperate'): (3, 0),
                ('defect', 'defect'): (3, 3),
            }
        elif matrix_type == "chicken":
            self.matrix = {
                ('cooperate', 'cooperate'): (3, 3),
                ('cooperate', 'defect'): (1, 4),
                ('defect', 'cooperate'): (4, 1),
                ('defect', 'defect'): (0, 0),
            }
        else:
            # Default: PD
            self.matrix = {
                ('cooperate', 'cooperate'): (3, 3),
                ('cooperate', 'defect'): (0, 5),
                ('defect', 'cooperate'): (5, 0),
                ('defect', 'defect'): (1, 1),
            }

    def get_payoffs(self, action1: str, action2: str) -> Tuple[float, float]:
        """Get payoffs for (agent1_action, agent2_action)"""
        key = (action1, action2)
        return self.matrix.get(key, (0, 0))


@dataclass
class SimulationWorld:
    """
    3D simulation world for ToM-NAS experiments.
    """
    width: int = 50
    height: int = 50
    depth: int = 1  # For 2D simulation, depth=1

    # World state
    terrain: np.ndarray = field(init=False)
    resources: List[Resource] = field(default_factory=list)
    agents: Dict[int, Any] = field(default_factory=dict)  # id -> agent

    # Game mechanics
    payoff_matrix: PayoffMatrix = field(default_factory=lambda: PayoffMatrix("prisoners_dilemma"))

    # History
    timestep: int = 0
    interaction_history: List[InteractionResult] = field(default_factory=list)
    event_log: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    visibility_range: float = 10.0
    resource_spawn_rate: float = 0.05
    max_resources: int = 50

    # Benchmark scenarios
    active_scenarios: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the world"""
        # Create terrain grid
        self.terrain = np.zeros((self.width, self.height, self.depth), dtype=np.int8)

        # Add some walls for partial observability
        self._generate_terrain()

        # Spawn initial resources
        self._spawn_initial_resources()

    def _generate_terrain(self):
        """Generate terrain with walls and obstacles"""
        # Simple room-like structure
        wall_prob = 0.05

        for x in range(self.width):
            for y in range(self.height):
                if np.random.random() < wall_prob:
                    self.terrain[x, y, 0] = TerrainType.WALL.value

        # Clear spawn areas
        center_x, center_y = self.width // 2, self.height // 2
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                x, y = center_x + dx, center_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.terrain[x, y, 0] = TerrainType.OPEN.value

    def _spawn_initial_resources(self):
        """Spawn initial resources in the world"""
        num_resources = self.max_resources // 2
        resource_id = 0

        for _ in range(num_resources):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            # Don't spawn on walls
            if self.terrain[x, y, 0] == TerrainType.WALL.value:
                continue

            resource_type = np.random.choice(list(ResourceType))
            value = np.random.uniform(5, 20)

            self.resources.append(Resource(
                id=resource_id,
                location=Location(x, y, 0),
                type=resource_type,
                value=value,
            ))
            resource_id += 1

    def add_agent(self, agent: Any):
        """Add an agent to the world"""
        self.agents[agent.id] = agent

        # Place agent in world
        if agent.x == 0 and agent.y == 0:
            # Random starting position
            while True:
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                if self.terrain[x, y, 0] != TerrainType.WALL.value:
                    agent.x = float(x)
                    agent.y = float(y)
                    break

    def remove_agent(self, agent_id: int):
        """Remove an agent from the world"""
        if agent_id in self.agents:
            del self.agents[agent_id]

    def get_observation(self, agent_id: int) -> Observation:
        """Get observation for an agent (partial observability)"""
        agent = self.agents.get(agent_id)
        if agent is None:
            return Observation(
                self_state={},
                nearby_agents=[],
                nearby_resources=[],
                visible_terrain=[],
                timestamp=self.timestep
            )

        radius = self.visibility_range

        # Self state
        self_state = {
            'position': (agent.x, agent.y, agent.z),
            'health': agent.success.domain1.health if hasattr(agent, 'success') else 100,
            'resources': agent.success.domain2.net_worth if hasattr(agent, 'success') else 50,
        }

        # Nearby agents (with line of sight check)
        nearby_agents = []
        for other_id, other in self.agents.items():
            if other_id == agent_id:
                continue

            dist = np.sqrt(
                (agent.x - other.x)**2 +
                (agent.y - other.y)**2 +
                (agent.z - other.z)**2
            )

            if dist <= radius and self._has_line_of_sight(agent, other):
                nearby_agents.append({
                    'id': other_id,
                    'position': (other.x, other.y, other.z),
                    'distance': dist,
                    'visible_wealth': self._get_visible_wealth(other),
                    'reputation': self._get_reputation(other_id),
                    'last_action': self._get_last_action(other_id),
                    'apparent_emotion': self._get_apparent_emotion(other),
                })

        # Nearby resources
        nearby_resources = []
        for resource in self.resources:
            if resource.depleted:
                continue

            dist = np.sqrt(
                (agent.x - resource.location.x)**2 +
                (agent.y - resource.location.y)**2 +
                (agent.z - resource.location.z)**2
            )

            if dist <= radius:
                nearby_resources.append({
                    'id': resource.id,
                    'position': resource.location.to_tuple(),
                    'type': resource.type.value,
                    'value': resource.value,
                    'distance': dist,
                })

        # Visible terrain
        visible_terrain = []
        ax, ay = int(agent.x), int(agent.y)
        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                x, y = ax + dx, ay + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if dx*dx + dy*dy <= radius*radius:
                        terrain_type = TerrainType(self.terrain[x, y, 0])
                        visible_terrain.append((x, y, 0, terrain_type))

        return Observation(
            self_state=self_state,
            nearby_agents=nearby_agents,
            nearby_resources=nearby_resources,
            visible_terrain=visible_terrain,
            timestamp=self.timestep
        )

    def _has_line_of_sight(self, agent1, agent2) -> bool:
        """Check if agent1 can see agent2 (no walls in between)"""
        # Bresenham-like ray casting
        x1, y1 = int(agent1.x), int(agent1.y)
        x2, y2 = int(agent2.x), int(agent2.y)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            if x == x2 and y == y2:
                return True

            if (0 <= x < self.width and 0 <= y < self.height and
                self.terrain[x, y, 0] == TerrainType.WALL.value):
                return False

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True

    def _get_visible_wealth(self, agent) -> float:
        """Get visible wealth indicator (not exact amount)"""
        if hasattr(agent, 'success'):
            actual = agent.success.domain2.net_worth
            # Add noise for partial observability
            noise = np.random.normal(0, 10)
            return np.clip(actual + noise, 0, 100)
        return 50.0

    def _get_reputation(self, agent_id: int) -> float:
        """Calculate agent's reputation from interaction history"""
        cooperations = 0
        total = 0

        for interaction in self.interaction_history[-100:]:  # Last 100 interactions
            if interaction.agent1_id == agent_id:
                total += 1
                if interaction.agent1_action == ActionType.COOPERATE:
                    cooperations += 1
            elif interaction.agent2_id == agent_id:
                total += 1
                if interaction.agent2_action == ActionType.COOPERATE:
                    cooperations += 1

        if total == 0:
            return 50.0  # Neutral reputation

        return cooperations / total * 100

    def _get_last_action(self, agent_id: int) -> Optional[str]:
        """Get agent's last action"""
        for interaction in reversed(self.interaction_history):
            if interaction.agent1_id == agent_id:
                return interaction.agent1_action.value
            elif interaction.agent2_id == agent_id:
                return interaction.agent2_action.value
        return None

    def _get_apparent_emotion(self, agent) -> str:
        """Get agent's apparent emotion (from Layer 1)"""
        if hasattr(agent, 'profile'):
            joy = agent.profile.layer1.joy
            sadness = agent.profile.layer1.sadness
            fear = agent.profile.layer1.fear
            anger = agent.profile.layer1.anger

            max_emotion = max(joy, sadness, fear, anger)
            if max_emotion == joy and joy > 60:
                return 'happy'
            elif max_emotion == sadness and sadness > 60:
                return 'sad'
            elif max_emotion == fear and fear > 60:
                return 'fearful'
            elif max_emotion == anger and anger > 60:
                return 'angry'
        return 'neutral'

    def step(self, actions: Dict[int, Action]) -> Dict[int, Dict]:
        """
        Execute one timestep of the simulation.
        Returns results for each agent.
        """
        self.timestep += 1
        results = {}

        # Process movements first
        for agent_id, action in actions.items():
            if action.type == ActionType.MOVE:
                results[agent_id] = self._process_move(agent_id, action)

        # Process interactions
        interaction_pairs = self._identify_interaction_pairs(actions)
        for (id1, id2), (action1, action2) in interaction_pairs.items():
            result = self._process_interaction(id1, id2, action1, action2)
            self.interaction_history.append(result)

            if id1 not in results:
                results[id1] = {}
            if id2 not in results:
                results[id2] = {}

            results[id1]['interaction'] = {
                'partner': id2,
                'my_action': action1.type.value,
                'their_action': action2.type.value,
                'my_payoff': result.agent1_payoff,
            }
            results[id2]['interaction'] = {
                'partner': id1,
                'my_action': action2.type.value,
                'their_action': action1.type.value,
                'my_payoff': result.agent2_payoff,
            }

        # Process resource collection
        for agent_id, action in actions.items():
            if action.type == ActionType.COLLECT and action.target_id is not None:
                collect_result = self._process_collect(agent_id, action)
                if agent_id not in results:
                    results[agent_id] = {}
                results[agent_id]['collect'] = collect_result

        # Resource respawning
        for resource in self.resources:
            resource.tick()

        # Maybe spawn new resources
        if len([r for r in self.resources if not r.depleted]) < self.max_resources:
            if np.random.random() < self.resource_spawn_rate:
                self._spawn_resource()

        # Update active scenarios
        self._update_scenarios()

        return results

    def _process_move(self, agent_id: int, action: Action) -> Dict:
        """Process a move action"""
        agent = self.agents.get(agent_id)
        if agent is None:
            return {'success': False, 'reason': 'agent_not_found'}

        target = action.target_location
        if target is None:
            # Random movement
            dx = np.random.uniform(-1, 1)
            dy = np.random.uniform(-1, 1)
            new_x = np.clip(agent.x + dx, 0, self.width - 1)
            new_y = np.clip(agent.y + dy, 0, self.height - 1)
        else:
            # Movement toward target
            dx = np.sign(target.x - agent.x)
            dy = np.sign(target.y - agent.y)
            new_x = np.clip(agent.x + dx, 0, self.width - 1)
            new_y = np.clip(agent.y + dy, 0, self.height - 1)

        # Check for walls
        if self.terrain[int(new_x), int(new_y), 0] != TerrainType.WALL.value:
            agent.x = new_x
            agent.y = new_y
            return {'success': True, 'new_position': (new_x, new_y)}
        else:
            return {'success': False, 'reason': 'blocked'}

    def _identify_interaction_pairs(self, actions: Dict[int, Action]) -> Dict:
        """Identify pairs of agents that will interact"""
        pairs = {}

        for agent_id, action in actions.items():
            if action.type in [ActionType.COOPERATE, ActionType.DEFECT]:
                target_id = action.target_id
                if target_id is not None and target_id in self.agents:
                    # Check if target also has an action toward this agent
                    pair_key = tuple(sorted([agent_id, target_id]))

                    if pair_key not in pairs:
                        # Find target's action
                        target_action = actions.get(target_id)
                        if target_action is None:
                            # Default to cooperate if no action specified
                            target_action = Action(
                                agent_id=target_id,
                                type=ActionType.COOPERATE,
                                target_id=agent_id
                            )

                        if agent_id < target_id:
                            pairs[pair_key] = (action, target_action)
                        else:
                            pairs[pair_key] = (target_action, action)

        return pairs

    def _process_interaction(self, id1: int, id2: int,
                           action1: Action, action2: Action) -> InteractionResult:
        """Process an interaction between two agents"""
        # Map action types to cooperation/defection
        a1_type = 'cooperate' if action1.type == ActionType.COOPERATE else 'defect'
        a2_type = 'cooperate' if action2.type == ActionType.COOPERATE else 'defect'

        # Get payoffs
        payoff1, payoff2 = self.payoff_matrix.get_payoffs(a1_type, a2_type)

        # Apply payoffs to agents
        agent1 = self.agents.get(id1)
        agent2 = self.agents.get(id2)

        if agent1 and hasattr(agent1, 'success'):
            agent1.success.domain2.net_worth = min(100,
                agent1.success.domain2.net_worth + payoff1)
        if agent2 and hasattr(agent2, 'success'):
            agent2.success.domain2.net_worth = min(100,
                agent2.success.domain2.net_worth + payoff2)

        # Generate description
        if a1_type == 'cooperate' and a2_type == 'cooperate':
            desc = f"Agents {id1} and {id2} cooperated mutually"
        elif a1_type == 'defect' and a2_type == 'defect':
            desc = f"Agents {id1} and {id2} both defected"
        elif a1_type == 'cooperate':
            desc = f"Agent {id1} cooperated but {id2} defected"
        else:
            desc = f"Agent {id2} cooperated but {id1} defected"

        return InteractionResult(
            agent1_id=id1,
            agent2_id=id2,
            agent1_action=action1.type,
            agent2_action=action2.type,
            agent1_payoff=payoff1,
            agent2_payoff=payoff2,
            description=desc,
            timestamp=self.timestep
        )

    def _process_collect(self, agent_id: int, action: Action) -> Dict:
        """Process a resource collection action"""
        resource_id = action.target_id
        resource = None

        for r in self.resources:
            if r.id == resource_id:
                resource = r
                break

        if resource is None:
            return {'success': False, 'reason': 'resource_not_found'}

        if resource.depleted:
            return {'success': False, 'reason': 'resource_depleted'}

        agent = self.agents.get(agent_id)
        if agent is None:
            return {'success': False, 'reason': 'agent_not_found'}

        # Check distance
        dist = np.sqrt(
            (agent.x - resource.location.x)**2 +
            (agent.y - resource.location.y)**2
        )

        if dist > 2.0:  # Must be close to collect
            return {'success': False, 'reason': 'too_far'}

        # Collect
        value = resource.collect()

        if hasattr(agent, 'success'):
            agent.success.domain2.net_worth = min(100,
                agent.success.domain2.net_worth + value * 0.5)

        return {
            'success': True,
            'value': value,
            'resource_type': resource.type.value
        }

    def _spawn_resource(self):
        """Spawn a new resource"""
        resource_id = max([r.id for r in self.resources], default=-1) + 1

        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if self.terrain[x, y, 0] != TerrainType.WALL.value:
                break

        resource_type = np.random.choice(list(ResourceType))
        value = np.random.uniform(5, 20)

        self.resources.append(Resource(
            id=resource_id,
            location=Location(x, y, 0),
            type=resource_type,
            value=value,
        ))

    def _update_scenarios(self):
        """Update active benchmark scenarios"""
        for scenario in self.active_scenarios:
            if 'update_fn' in scenario:
                scenario['update_fn'](self, scenario)

    def add_scenario(self, scenario: Dict):
        """Add a benchmark scenario to the world"""
        self.active_scenarios.append(scenario)
        self.event_log.append({
            'type': 'scenario_added',
            'scenario': scenario.get('type', 'unknown'),
            'timestep': self.timestep,
        })

    def get_state(self) -> WorldState:
        """Get complete world state"""
        agents_data = []
        for agent_id, agent in self.agents.items():
            agents_data.append({
                'id': agent_id,
                'x': agent.x,
                'y': agent.y,
                'z': getattr(agent, 'z', 0),
                'visible_wealth': self._get_visible_wealth(agent),
                'reputation': self._get_reputation(agent_id),
                'last_action': self._get_last_action(agent_id),
                'apparent_emotion': self._get_apparent_emotion(agent),
            })

        resources_data = []
        for resource in self.resources:
            if not resource.depleted:
                resources_data.append({
                    'id': resource.id,
                    'x': resource.location.x,
                    'y': resource.location.y,
                    'z': resource.location.z,
                    'type': resource.type.value,
                    'value': resource.value,
                })

        return WorldState(
            timestep=self.timestep,
            agents=agents_data,
            resources=resources_data,
            terrain=self.terrain,
            interactions=self.interaction_history[-20:],  # Last 20 interactions
            events=self.event_log[-50:],  # Last 50 events
        )

    def get_statistics(self) -> Dict:
        """Get world statistics"""
        cooperation_rate = 0.0
        if self.interaction_history:
            coop_count = sum(
                1 for i in self.interaction_history
                if i.agent1_action == ActionType.COOPERATE or
                   i.agent2_action == ActionType.COOPERATE
            )
            cooperation_rate = coop_count / (len(self.interaction_history) * 2)

        return {
            'timestep': self.timestep,
            'num_agents': len(self.agents),
            'num_resources': len([r for r in self.resources if not r.depleted]),
            'total_interactions': len(self.interaction_history),
            'cooperation_rate': cooperation_rate,
            'active_scenarios': len(self.active_scenarios),
        }

    def describe(self) -> str:
        """Generate human-readable world description"""
        stats = self.get_statistics()
        lines = [
            "=== Simulation World ===",
            f"Size: {self.width}x{self.height}x{self.depth}",
            f"Timestep: {stats['timestep']}",
            "",
            f"Agents: {stats['num_agents']}",
            f"Resources: {stats['num_resources']}",
            f"Interactions: {stats['total_interactions']}",
            f"Cooperation Rate: {stats['cooperation_rate']:.1%}",
            "",
            f"Active Scenarios: {stats['active_scenarios']}",
        ]
        return '\n'.join(lines)
