# ToM-NAS Godot Integration

Complete integration of the Theory of Mind Neural Architecture Search system with Godot 4.x for embodied cognition through physics-grounded symbol manipulation.

## Overview

The Godot integration provides the physical simulation layer for ToM-NAS, enabling:

- **Embodied Cognition**: Agents perceive and act in a 3D world
- **Symbol Grounding**: Physical objects are mapped to semantic meaning
- **Social Simulation**: Multiple agents interact within institutional contexts
- **Theory of Mind Testing**: Classic ToM scenarios (Sally-Anne, etc.)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python Side                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  GodotBridge │  │   Symbol    │  │  Perception │             │
│  │  (WebSocket) │  │  Grounding  │  │  Processor  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────┐              │
│  │           Cognitive Core (Mentalese)           │              │
│  │    Recursive Self-Compression / TRM / ToM      │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                           │
                    WebSocket @ 9080
                           │
┌─────────────────────────────────────────────────────────────────┐
│                         Godot Side                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  TomBridge  │  │   World     │  │  Perception │             │
│  │  (Autoload) │  │   Manager   │  │   System    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                      │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐             │
│  │ Institution │  │   Event     │  │  Protocol   │             │
│  │   Manager   │  │    Bus      │  │   Handler   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌───────────────────────────────────────────────┐              │
│  │              3D Physics Simulation             │              │
│  │         Agents / Objects / Locations           │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Python Server

```bash
# Start just the Python WebSocket server
python run_godot_integration.py --no-godot

# Or with demo mode
python run_godot_integration.py --demo
```

### 2. Open Godot Project

```bash
# Open in Godot Editor
godot --path godot_project -e

# Or run directly
godot --path godot_project
```

### 3. Run Integration

```bash
# Full integration (launches both)
python run_godot_integration.py

# With specific scenario
python run_godot_integration.py --scenario sally_anne
```

## Directory Structure

```
godot_project/
├── project.godot           # Project configuration
├── autoloads/              # Global singletons
│   ├── tom_bridge.gd       # WebSocket communication
│   ├── world_manager.gd    # Entity registry
│   ├── protocol_handler.gd # Message serialization
│   ├── institution_manager.gd  # Social contexts
│   ├── perception_system.gd    # Agent vision
│   └── event_bus.gd        # Global events
├── scripts/
│   ├── main_world.gd       # Main scene controller
│   ├── entities/
│   │   ├── agent.gd        # Cognitive agent
│   │   ├── interactable_object.gd
│   │   ├── item.gd         # Tradeable items
│   │   └── location_zone.gd    # Institutional zones
│   └── scenarios/
│       └── sally_anne_scenario.gd
├── scenes/
│   ├── main_world.tscn     # Main game scene
│   ├── entities/
│   │   ├── agent.tscn
│   │   ├── interactable_object.tscn
│   │   ├── item.tscn
│   │   └── location_zone.tscn
│   └── scenarios/
│       └── sally_anne.tscn
└── resources/              # Materials, textures, etc.
```

## Communication Protocol

### Message Format

All messages are JSON with this structure:

```json
{
    "type": "MESSAGE_TYPE",
    "payload": { ... },
    "timestamp": 1234567890.123,
    "sequence_id": 42
}
```

### Message Types

#### Godot → Python

| Type | Description |
|------|-------------|
| `ENTITY_UPDATE` | Entity position/state changed |
| `AGENT_PERCEPTION` | What an agent perceives |
| `WORLD_STATE` | Full world snapshot |
| `COLLISION_EVENT` | Collision occurred |
| `INTERACTION_EVENT` | Agent interacted with object |
| `UTTERANCE_EVENT` | Agent spoke |

#### Python → Godot

| Type | Description |
|------|-------------|
| `AGENT_COMMAND` | Command for agent to execute |
| `SPAWN_ENTITY` | Create new entity |
| `MODIFY_ENTITY` | Change entity properties |
| `WORLD_COMMAND` | Global world control |

#### Bidirectional

| Type | Description |
|------|-------------|
| `HEARTBEAT` | Connection keepalive |
| `ACK` | Command acknowledgment |
| `ERROR` | Error notification |

#### Simulation Control

| Type | Description |
|------|-------------|
| `PAUSE` | Pause simulation |
| `RESUME` | Resume simulation |
| `RESET` | Reset world state |
| `STEP` | Single frame step |

### Agent Commands

Commands that can be sent to agents:

```gdscript
# Movement
{"command_type": "move", "target_position": {"x": 5, "y": 0, "z": 5}}
{"command_type": "turn", "target_entity_id": 12345}
{"command_type": "follow", "target_entity_id": 12345}
{"command_type": "flee", "target_entity_id": 12345}

# Object Interaction
{"command_type": "pick_up", "target_entity_id": 12345}
{"command_type": "put_down"}
{"command_type": "use", "target_entity_id": 12345}
{"command_type": "examine", "target_entity_id": 12345}
{"command_type": "give", "target_entity_id": 12345}  # Give to agent

# Social
{"command_type": "speak", "utterance_text": "Hello!"}
{"command_type": "gesture", "animation_name": "wave"}
{"command_type": "look_at", "target_entity_id": 12345}

# Control
{"command_type": "wait", "timeout_seconds": 2.0}
{"command_type": "cancel"}
```

## Institutions

Five institutional contexts affect agent behavior:

### The Hollow
- **Description**: Absence of structure, social anomie
- **Norms**: None
- **Pressure**: Existential

### The Market
- **Description**: Economic exchange and competition
- **Norms**: fair_exchange, property_rights, contract_honor
- **Roles**: buyer, seller, broker, merchant

### The Ministry
- **Description**: Bureaucratic hierarchy and rules
- **Norms**: follow_protocol, respect_hierarchy, document_actions
- **Roles**: official, clerk, petitioner, supervisor

### The Court
- **Description**: Justice and judgment
- **Norms**: truth_telling, evidence_based, due_process
- **Roles**: judge, prosecutor, defender, witness, accused

### The Temple
- **Description**: Sacred space and ritual
- **Norms**: reverence, ritual_compliance, purity
- **Roles**: priest, acolyte, pilgrim, devotee

## Perception System

Agents have configurable perception:

```gdscript
# Default configuration
var view_distance: float = 20.0
var view_angle: float = 120.0  # degrees
var hearing_distance: float = 15.0
```

Features:
- **Visual perception**: Raycasting with occlusion
- **Auditory perception**: Distance-based hearing
- **Proprioception**: Self-state awareness
- **Salience detection**: Movement, size, proximity

## Symbol Grounding

Physical objects are grounded to semantic meaning:

```python
# From Godot EntityUpdate to GroundedSymbol
symbol = grounder.ground_entity(entity_update)

# Attributes:
symbol.godot_id          # Godot node ID
symbol.semantic_node_id  # ID in Indra's Net
symbol.category          # Inferred category (chair, table, etc.)
symbol.physical_affordances  # What can be done with it
symbol.semantic_associations  # Linked concepts
```

## Scenarios

### Sally-Anne Test

Classic false belief test:

```python
# Load scenario
runner.load_scenario('sally_anne')

# Phases:
# 1. Sally places marble in basket
# 2. Sally leaves
# 3. Anne moves marble to box
# 4. Sally returns
# 5. Test: Where will Sally look?
```

### Market Exchange

Economic interaction scenario:

```python
runner.load_scenario('market_exchange')
```

### Zombie Detection

Theory of Mind detection scenario:

```python
runner.load_scenario('zombie_detection')
```

## API Reference

### TomBridge (Godot)

```gdscript
# Connection
TomBridge.connect_to_bridge(host, port)
TomBridge.disconnect_from_bridge()
TomBridge.is_connected_to_bridge() -> bool

# Sending messages
TomBridge.send_entity_update(entity)
TomBridge.send_agent_perception(agent, perception_data)
TomBridge.send_interaction_event(agent, target, type, success)
TomBridge.send_utterance_event(speaker, text, volume, target)
TomBridge.send_ack(command_id, success, changes, error)

# Signals
signal connection_state_changed(state)
signal message_received(type, payload)
signal command_received(agent_id, command)
signal simulation_control(control_type)
```

### GodotBridge (Python)

```python
# Connection
bridge.start(blocking=False)
bridge.stop()
bridge.is_connected() -> bool

# Sending commands
bridge.send_command(AgentCommand(...))
bridge.request_world_state()
bridge.pause_simulation()
bridge.resume_simulation()
bridge.reset_simulation()

# Message handling
@bridge.on(MessageType.AGENT_PERCEPTION)
def handle_perception(msg):
    ...
```

### WorldManager (Godot)

```gdscript
# Entity management
WorldManager.register_entity(entity)
WorldManager.unregister_entity(entity)
WorldManager.get_entity(id) -> Node3D
WorldManager.spawn_entity(type, name, position, properties)
WorldManager.modify_entity(id, modifications)

# Queries
WorldManager.get_entities_in_radius(center, radius, type)
WorldManager.get_nearest_entity(from, type, exclude)
WorldManager.raycast_entities(from, to, exclude)

# State
WorldManager.simulation_time
WorldManager.time_of_day
WorldManager.weather
```

## Testing

Run the test suite:

```bash
# Python tests
pytest tests/test_godot_bridge.py -v

# Integration test
python run_godot_integration.py --demo
```

## Troubleshooting

### Connection Issues

1. **Port in use**: Change port with `--port 9081`
2. **Firewall**: Allow localhost:9080
3. **Godot not found**: Use `--godot-path /path/to/godot`

### Performance

1. Reduce entity update frequency in WorldManager
2. Adjust perception update interval
3. Limit number of simultaneous agents

### Common Errors

```
"WebSocket connection failed"
→ Ensure Python server is running first

"Entity not found"
→ Check entity registration in WorldManager

"Command timeout"
→ Increase timeout_seconds in command
```

## Contributing

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Test with both Python and Godot sides

## References

- [Harnad's Symbol Grounding Problem](https://www.sciencedirect.com/science/article/abs/pii/S0167278998001318)
- [Godot 4.x Documentation](https://docs.godotengine.org/)
- [Theory of Mind in AI](https://www.sciencedirect.com/science/article/pii/S0004370216300790)
