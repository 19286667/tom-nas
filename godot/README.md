# Liminal Architectures - Godot Integration

**Fully integrated** with the ToM-NAS cognitive architecture. This module connects
the Godot 4 game client to the complete tom-nas backend including:
- 5th-order recursive Theory of Mind
- 65-dimensional Soul Map psychological ontology
- Indra's Net semantic knowledge graph
- Neural Architecture Search evolved models

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GODOT 4.x                                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐│
│  │ Player        │  │ NPCs          │  │ UI                    ││
│  │ Controller    │  │ (Tier 0/1)    │  │ Debug Panel           ││
│  └───────┬───────┘  └───────┬───────┘  │ Soul Scanner          ││
│          │                  │          └───────────────────────┘│
│          └────────┬─────────┘                                   │
│                   ▼                                             │
│          ┌───────────────────┐                                  │
│          │    GameBridge     │  ◄─── WebSocket Client           │
│          │   (Autoload)      │                                  │
│          └─────────┬─────────┘                                  │
└────────────────────┼────────────────────────────────────────────┘
                     │ ws://localhost:9080
                     ▼
┌────────────────────┴────────────────────────────────────────────┐
│                    PYTHON BACKEND                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              godot_server.py (ToM-NAS Integration)          ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  ││
│  │  │  Tier 2     │  │  Tier 3     │  │  Dialogue           │  ││
│  │  │  Strategic  │  │  Deep ToM   │  │  Generation         │  ││
│  │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  ││
│  └─────────┼────────────────┼────────────────────┼─────────────┘│
│            └────────────────┼────────────────────┘              │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    tom-nas modules                           ││
│  │  ✓ BeliefNetwork (5th-order ToM)    ✓ SoulMap (65-dim)     ││
│  │  ✓ IndrasNet (semantic graph)       ✓ RecursiveSimulator   ││
│  │  ✓ SoulMapDelta (hazards)          ✓ Archetype system      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: Run from Project Root (Recommended)

```bash
cd /path/to/tom-nas

# Start the integrated server
python run_godot_server.py

# Or with debug logging
python run_godot_server.py --debug
```

### Option 2: Run from Godot Directory

```bash
cd godot/python
pip install websockets
python godot_server.py
```

You should see:
```
============================================================
ToM-NAS Godot Bridge Server
============================================================
  ToM-NAS modules: AVAILABLE
  Indra's Net:     AVAILABLE
  PyTorch:         AVAILABLE
  WebSockets:      AVAILABLE
============================================================

2025-12-05 12:00:00 [INFO] Belief network initialized (5th-order ToM)
2025-12-05 12:00:00 [INFO] Starting Godot Bridge Server on ws://localhost:9080
2025-12-05 12:00:00 [INFO] ToM-NAS integration: ENABLED
```

### Open Godot Project

1. Open Godot 4.2+
2. Import the `godot/` folder as a project
3. Press F5 to run
4. Press **F3** to open Debug Panel and verify connection

## Tiered Processing

| Tier | Location | Latency | Responsibility |
|------|----------|---------|----------------|
| **0** | Godot | <16ms | Immediate reactive responses (flinch, startle) |
| **1** | Godot | <50ms | Heuristic decisions using local soul map |
| **2** | Python | <200ms | Strategic decisions via tom-nas SoulMap |
| **3** | Python | <500ms | Deep recursive ToM via BeliefNetwork |

## Project Structure

```
tom-nas/
├── godot/                          # Godot integration (this folder)
│   ├── project.godot               # Godot project config
│   ├── scenes/
│   │   ├── main.tscn               # Main test scene
│   │   ├── npc.tscn                # NPC prefab
│   │   ├── player.tscn             # Player prefab
│   │   └── debug_panel.tscn        # Debug UI
│   ├── scripts/
│   │   ├── autoload/
│   │   │   ├── game_bridge.gd      # WebSocket client
│   │   │   ├── soul_map_manager.gd # Local soul map storage (65 dims)
│   │   │   └── event_bus.gd        # Decoupled events
│   │   ├── npc/
│   │   │   └── npc_controller.gd   # NPC with tiered processing
│   │   ├── player/
│   │   │   └── player_controller.gd
│   │   └── ui/
│   │       └── debug_panel.gd
│   ├── python/
│   │   └── godot_server.py         # Integrated WebSocket server
│   └── README.md
├── src/
│   ├── godot_bridge/               # Original Python bridge module
│   ├── liminal/                    # Liminal game environment
│   │   └── soul_map.py             # 65-dim SoulMap class
│   ├── core/
│   │   └── beliefs.py              # BeliefNetwork (5th-order ToM)
│   └── knowledge_base/
│       └── indras_net.py           # Semantic knowledge graph
└── run_godot_server.py             # Unified entry point
```

## ToM-NAS Integration Details

### Soul Map Conversion

The `SoulMapConverter` class maps between Godot's 65 dimensions and tom-nas's
clustered SoulMap format:

| Godot Dimension | ToM-NAS Cluster | ToM-NAS Dimension |
|-----------------|-----------------|-------------------|
| trust_propensity | social | trust_default |
| theory_of_mind_depth | cognitive | tom_depth |
| emotional_intensity | emotional | intensity |
| risk_tolerance | motivational | risk_tolerance |
| ... | ... | ... |

### Belief Network Integration

When Deep ToM queries arrive, the server:
1. Converts Godot soul map to tom-nas `SoulMap`
2. Uses `SoulMap.get_tom_depth_int()` to determine reasoning depth (1-5)
3. Runs recursive simulation through `RecursiveSimulator`
4. Returns belief state, intentions, and deception probability

### Cognitive Hazards

Player actions can trigger `SoulMapDelta` effects:
- **Attack** → `SoulMapDelta.fear(0.3)` (increases anxiety, threat sensitivity)
- **Help** → `SoulMapDelta.validation(0.2)` (increases esteem, reduces anxiety)

These deltas are applied and synced back to Godot automatically.

## Message Protocol

### Godot → Python

| Type | Purpose | Payload |
|------|---------|---------|
| `PLAYER_ACTION` | Report player behavior | action_type, target_id, context |
| `QUERY_STRATEGIC` | Tier 2 decision request | npc_id, situation, soul_map |
| `QUERY_DEEP_TOM` | Tier 3 reasoning request | npc_id, target_id, depth, query_type |
| `DIALOGUE_REQUEST` | Generate dialogue | npc_id, context, history |
| `PERCEPTION_EVENT` | Report perceived entities | npc_id, entities[] |
| `WORLD_STATE` | Full state sync | entities[], relationships |

### Python → Godot

| Type | Purpose | Payload |
|------|---------|---------|
| `UPDATE_SOUL_MAP` | Push soul map changes | npc_id, soul_map |
| `SPAWN_NPC` | Create new NPC | position, archetype, soul_map |
| `NARRATIVE_BEAT` | Story event | event_type, data |
| `DIALOGUE_RESPONSE` | Generated dialogue | npc_id, text, choices[] |
| `COGNITIVE_HAZARD` | Apply hazard effect | npc_id, hazard_type, intensity |

## Test NPCs

The main scene includes three test NPCs with different archetypes:

| NPC | Archetype | Key Traits |
|-----|-----------|------------|
| `npc_paranoid` | Suspicious Stranger | Low trust, high vigilance |
| `npc_naive` | Trusting Villager | High trust, low deception detection |
| `npc_manipulative` | Charming Merchant | High deception, high ToM depth |

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| E | Interact with NPC |
| Q | Soul Scanner (inspect NPC) |
| F3 | Toggle Debug Panel |
| ESC | Toggle mouse capture |

## Troubleshooting

### "Python Backend: Disconnected"

1. Make sure `godot_server.py` is running
2. Check it's on port 9080: `netstat -an | grep 9080`
3. Verify tom-nas modules are importable: `python -c "from src.liminal.soul_map import SoulMap"`

### "ToM-NAS modules: NOT AVAILABLE"

1. Run from project root: `python run_godot_server.py`
2. Or install tom-nas: `pip install -e .`
3. Check Python path includes tom-nas/src

### NPCs not moving

1. Bake the navigation mesh in the NavigationRegion3D
2. Check NPC has NavigationAgent3D child

### No dialogue appearing

1. Implement the dialogue UI responding to `EventBus.show_dialogue_ui`
2. Check Python server logs for dialogue requests
