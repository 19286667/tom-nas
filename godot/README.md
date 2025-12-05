# ToM-NAS Godot Validation Environment

This Godot 4.x project provides a 3D physics simulation for validating Theory of Mind Neural Architecture Search agents. It connects to the Python cognitive controller via WebSocket.

## Requirements

- Godot 4.2 or later
- Python server running (enhanced_server.py)

## Setup

1. Open this project in Godot 4.x
2. Start the Python server:
   ```bash
   cd /path/to/tom-nas
   python -c "
   from src.godot_bridge.enhanced_server import create_enhanced_server
   server = create_enhanced_server(port=9080)
   server.start(blocking=True)
   "
   ```
3. Run the Godot project (F5 or Play button)

## Architecture

### Autoloads (Singletons)

- **WebSocketBridge**: Manages WebSocket connection to Python
- **WorldState**: Tracks all entities and world state

### Scripts

- `agent_controller.gd`: Controls AI agents, handles perception and commands
- `websocket_bridge.gd`: WebSocket communication layer
- `world_state.gd`: World state management
- `main_scene.gd`: Main scene controller

### Scenes

- `main.tscn`: Main scene with agents, objects, and environment

## Communication Protocol

The bridge uses JSON messages with the following types:

### Godot -> Python
- `ENTITY_UPDATE`: Entity position/state changed
- `AGENT_PERCEPTION`: What an agent perceives
- `WORLD_STATE`: Full world snapshot
- `COLLISION_EVENT`: Collision occurred
- `INTERACTION_EVENT`: Agent interacted with object
- `UTTERANCE_EVENT`: Agent spoke

### Python -> Godot
- `AGENT_COMMAND`: Command for agent (move, interact, speak)
- `SPAWN_ENTITY`: Spawn new entity
- `MODIFY_ENTITY`: Modify existing entity
- `WORLD_COMMAND`: Global world command

### Bidirectional
- `HEARTBEAT`: Connection keepalive
- `ACK`: Acknowledgment

## Agent Commands

Agents respond to these command types:
- `move`: Move to position or entity
- `interact`: Interact with object
- `speak`: Say something
- `look`: Look at position or entity
- `pick_up`: Pick up object
- `put_down`: Put down held object
- `stop`: Stop current action

## Debug Controls

- `R`: Reset world
- `P`: Pause/Resume simulation
- `W`: Send world state to Python

## Integration with ToM-NAS

This environment enables:
1. **Grounded perception**: Agents perceive entities via raycasting
2. **Belief updates**: Observations update the BeliefNetwork
3. **Theory of Mind**: Higher-order belief inference from behavior
4. **Social dynamics**: Agent interactions and communication
