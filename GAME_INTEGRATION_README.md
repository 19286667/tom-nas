# LIMINAL ARCHITECTURES: Game Integration Guide

**Status:** Prototype Phase
**Version:** 0.1.0
**Date:** November 27, 2025

---

## 🎮 Overview

LIMINAL ARCHITECTURES integrates cutting-edge Theory of Mind (ToM) research with AAA game design, creating NPCs with genuine psychological understanding. This is the world's first playable implementation of neural ToM architectures in a game environment.

### What Makes This Special?

- **Genuine ToM:** NPCs don't just respond to choices - they understand your mental state
- **181-Dimensional Psychology:** Complete ontology from biological to existential
- **Recursive Reasoning:** Up to 5th-order beliefs ("A thinks B thinks C believes...")
- **Dual Purpose:** Both research tool and entertainment product
- **Novel Gameplay:** Puzzles and challenges that require empathy and insight

---

## 📂 Project Structure

```
tom-nas/
├── src/
│   ├── game/                      # ⭐ NEW: Game integration layer
│   │   ├── api_server.py          # REST API + WebSocket server
│   │   ├── soul_map_visualizer.py # Visualization system
│   │   ├── dialogue_system.py     # ToM-driven dialogue
│   │   ├── combat_system.py       # Psychological combat
│   │   └── quest_system.py        # ToM quest framework
│   │
│   ├── core/                      # ToM-NAS research core
│   │   ├── ontology.py            # 181-dim Soul Map
│   │   └── beliefs.py             # Recursive beliefs
│   │
│   ├── agents/                    # Neural architectures
│   │   └── architectures.py       # TRN, RSAN, Transformer
│   │
│   └── world/                     # Social simulation
│       └── social_world.py        # Social World 4
│
├── game_demo.py                   # ⭐ Complete demo
├── LIMINAL_ARCHITECTURES_DESIGN.md # Design document
└── requirements.txt               # Updated dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Demo

```bash
python game_demo.py
```

This demonstrates:
- Soul Map visualization
- ToM-powered NPCs
- Dynamic dialogue
- Psychological combat
- Quest system
- Full integration

### 3. Start the API Server

```bash
python -m src.game.api_server
```

Then visit: `http://localhost:8000/docs` for API documentation

---

## 🎯 Core Systems

### 1. Soul Map Visualization

Visualize 181-dimensional psychological states in real-time.

```python
from src.game.soul_map_visualizer import SoulMapVisualizer
from src.core.ontology import SoulMapOntology

ontology = SoulMapOntology()
visualizer = SoulMapVisualizer(ontology)

# Create a soul map
soul_map = ontology.get_default_state()
soul_map[1] = 0.3  # Low valence (sad)
soul_map[2] = 0.8  # High arousal (anxious)

# Generate visualizations
visualizer.visualize_radar(soul_map, save_path="radar.png")
visualizer.visualize_aura(soul_map, save_path="aura.png")
```

**Visualization Types:**
- **Radar:** 8-axis radar chart for key dimensions
- **Heatmap:** Full 181-dimension grid
- **Timeline:** Evolution over time
- **Comparison:** Side-by-side soul maps
- **Aura:** 3D-style emotional representation

### 2. ToM-Powered NPCs

Create NPCs with genuine theory of mind capabilities.

```python
from src.game.api_server import NPCController, NPCConfig

# Configure NPC
config = NPCConfig(
    npc_id="peregrine",
    name="Peregrine",
    architecture="RSAN",  # Recursive Self-Attention
    initial_soul_map={
        'affect.valence': 0.8,
        'social.empathy': 0.95,
        'wisdom.self_awareness': 0.9
    }
)

# Create NPC
npc = NPCController(config, ontology)

# NPC observes player
npc.observe_player(player_soul_map)

# Generate dialogue with ToM reasoning
response = npc.generate_dialogue(
    context="Player approaches seeking guidance",
    player_utterance="I'm lost..."
)

print(f"NPC: {response['text']}")
print(f"ToM Reasoning: {response['tom_reasoning']}")
```

**NPC Architectures:**
- **TRN:** Fast, reflexive, emotional
- **RSAN:** Deep recursive reasoning (recommended for key NPCs)
- **Transformer:** Contextual and adaptive

### 3. Dialogue System

Dynamic conversations that respond to psychological states.

```python
from src.game.dialogue_system import DialogueManager

manager = DialogueManager(ontology)

# Start conversation
conversation = manager.start_conversation(
    conversation_id="conv_001",
    npc_name="Peregrine",
    npc_soul_map=npc_soul_map,
    player_soul_map=player_soul_map,
    context="Cottage encounter"
)

# Get greeting
greeting = conversation.generate_npc_greeting()

# Process player choice
manager.process_player_choice("conv_001", option_id="empathize")
```

**Features:**
- First-order ToM: "What does the player feel?"
- Second-order ToM: "What does the player think I believe?"
- Deception detection
- Relationship tracking
- Dynamic option generation

### 4. Psychological Combat

Combat that targets both body and mind.

```python
from src.game.combat_system import CombatSystem, Combatant

combat_system = CombatSystem(ontology)

# Create combatants
player = Combatant("player", "Hero", player_map, ontology)
player.tom_order = 2  # Enable ToM

enemy = Combatant("enemy", "Shadow", enemy_map, ontology)

# Start combat
combat = combat_system.start_combat("battle_01", [player, enemy])

# Execute psychological attack
result = combat.execute_action(
    attacker_id="player",
    defender_id="enemy",
    action=combat_system.combat_actions['intimidate']
)

# ToM detects vulnerabilities for bonus damage
print(f"Critical hit: {result['damage_report']['critical_hit']}")
```

**Combat Features:**
- Physical damage (HP)
- Psychological damage (Soul Map dimensions)
- Vulnerability detection via ToM
- Defense mechanisms (Denial, Rationalization, Projection)
- Coherence system (psychological integrity)

### 5. Quest System

Quests requiring genuine theory of mind reasoning.

```python
from src.game.quest_system import QuestManager

quest_manager = QuestManager(ontology)

# Start a quest
quest = quest_manager.start_quest('cottage_test')

print(f"{quest.name}")
print(f"Required ToM Order: {quest.required_tom_order}")

# Complete objective
quest_manager.complete_objective(
    'cottage_test',
    'read_emotion',
    player_soul_map
)
```

**Quest Types:**
- **Tutorial:** Learn basic ToM
- **Deception:** Detect lies and inconsistencies
- **Recursive:** Multi-level belief reasoning
- **Zombie Detection:** Identify non-conscious NPCs
- **Coalition:** Navigate group dynamics
- **Moral Dilemma:** Ethical reasoning

---

## 🌐 API Integration

### REST API

Start the server:
```bash
python -m src.game.api_server
```

**Endpoints:**

```http
POST /session/create
POST /session/{session_id}/npc/create
POST /session/{session_id}/player/update_soul_map
POST /dialogue/generate
POST /combat/action
POST /tom/analyze
```

**Example: Create NPC**

```python
import requests

# Create session
response = requests.post("http://localhost:8000/session/create")
session_id = response.json()['session_id']

# Create NPC
npc_config = {
    "npc_id": "peregrine",
    "name": "Peregrine",
    "architecture": "RSAN",
    "initial_soul_map": {
        "affect.valence": 0.8,
        "social.empathy": 0.95
    }
}

response = requests.post(
    f"http://localhost:8000/session/{session_id}/npc/create",
    json=npc_config
)

print(response.json())
```

### WebSocket (Real-time Updates)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/session_123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'npc_reactions') {
        // Update UI with NPC reactions to player soul map changes
        console.log(data.reactions);
    }
};

// Send soul map update
ws.send(JSON.stringify({
    type: 'soul_map_update',
    soul_map: {
        'affect.valence': 0.5,
        'affect.arousal': 0.6
    }
}));
```

---

## 🎨 Integration with Game Engines

### Unity Integration (C#)

```csharp
using UnityEngine;
using System.Net.Http;
using Newtonsoft.Json;

public class ToMNPCController : MonoBehaviour {
    private HttpClient client = new HttpClient();
    private string sessionId;
    private string npcId;

    async void Start() {
        // Create session
        var sessionResponse = await client.PostAsync(
            "http://localhost:8000/session/create", null
        );

        var sessionData = await sessionResponse.Content.ReadAsStringAsync();
        sessionId = JsonConvert.DeserializeObject<SessionData>(sessionData).session_id;

        // Create NPC
        var npcConfig = new {
            npc_id = "peregrine",
            name = "Peregrine",
            architecture = "RSAN"
        };

        await client.PostAsync(
            $"http://localhost:8000/session/{sessionId}/npc/create",
            new StringContent(JsonConvert.SerializeObject(npcConfig))
        );
    }

    async Task<DialogueResponse> GenerateDialogue(Dictionary<string, float> playerSoulMap) {
        var request = new {
            session_id = sessionId,
            npc_id = npcId,
            player_soul_map = playerSoulMap,
            context = "Current scene context"
        };

        var response = await client.PostAsync(
            "http://localhost:8000/dialogue/generate",
            new StringContent(JsonConvert.SerializeObject(request))
        );

        return JsonConvert.DeserializeObject<DialogueResponse>(
            await response.Content.ReadAsStringAsync()
        );
    }
}
```

### Unreal Engine Integration (C++)

```cpp
#include "Http.h"
#include "Json.h"

class AToMNPCController : public AActor {
private:
    FString SessionId;
    FString NPCId;

public:
    void CreateSession() {
        TSharedRef<IHttpRequest> Request = FHttpModule::Get().CreateRequest();
        Request->SetURL("http://localhost:8000/session/create");
        Request->SetVerb("POST");
        Request->OnProcessRequestComplete().BindUObject(
            this, &AToMNPCController::OnSessionCreated
        );
        Request->ProcessRequest();
    }

    void OnSessionCreated(
        FHttpRequestPtr Request,
        FHttpResponsePtr Response,
        bool bSuccess
    ) {
        if (bSuccess) {
            TSharedPtr<FJsonObject> JsonObject;
            TSharedRef<TJsonReader<>> Reader =
                TJsonReaderFactory<>::Create(Response->GetContentAsString());

            if (FJsonSerializer::Deserialize(Reader, JsonObject)) {
                SessionId = JsonObject->GetStringField("session_id");
            }
        }
    }
};
```

---

## 📊 Performance Considerations

### Optimization Strategies

1. **Async Processing:** NPCs update in background threads
2. **Importance Sampling:** Focus computation on nearby/important NPCs
3. **Level-of-Detail ToM:** Distant NPCs use simpler models
4. **Caching:** Pre-compute common psychological states
5. **Batching:** Process multiple NPCs in single forward pass

### Expected Performance

- **API Response Time:** 50-200ms (typical dialogue generation)
- **Soul Map Inference:** 10-50ms (single NPC, GPU)
- **Batch Inference:** 20-100ms (10 NPCs, GPU)
- **Memory Usage:** ~500MB-2GB (depending on active NPCs)

**Recommendations:**
- **For 1-5 NPCs:** Real-time on CPU acceptable
- **For 10+ NPCs:** GPU highly recommended
- **For 50+ NPCs:** Implement importance sampling

---

## 🎓 Research Integration

### Telemetry System (Future)

Collect consent-based research data:

```python
# Future implementation
from src.game.telemetry import TelemetryManager

telemetry = TelemetryManager(
    consent_obtained=True,
    anonymize=True,
    irb_protocol="OBU-2025-12345"
)

# Automatically logs:
# - ToM task success rates
# - Dialogue choice patterns
# - Soul Map evolution
# - NPC believability ratings

telemetry.export_research_data("research_dataset.json")
```

### Publications Potential

- **NeurIPS/ICLR:** Neural ToM architectures
- **CHI/IUI:** Human-AI interaction
- **AIIDE/FDG:** Game AI and narrative
- **Cognitive Science:** Ecological ToM validation

---

## 🛠️ Development Roadmap

### Phase 1: Core Systems ✅ (Complete)
- [x] API server
- [x] Soul Map visualization
- [x] NPC controller
- [x] Dialogue system
- [x] Combat system
- [x] Quest system
- [x] Complete demo

### Phase 2: Vertical Slice (Next)
- [ ] Peregrine's Cottage (environment)
- [ ] The Infinite Jest (location)
- [ ] 3-5 fully-realized NPCs
- [ ] Tutorial quest chain
- [ ] Combat encounter
- [ ] 30-60 min gameplay

### Phase 3: Expansion
- [ ] 5+ locations
- [ ] 20+ NPCs
- [ ] Full quest system
- [ ] Multiplayer considerations
- [ ] Research telemetry
- [ ] Polish and optimization

### Phase 4: Launch
- [ ] Playtesting
- [ ] Balancing
- [ ] Localization
- [ ] Marketing
- [ ] Research publication

---

## 💡 Design Patterns

### Creating a New NPC

```python
# 1. Define psychological profile
soul_map = {
    'affect.valence': 0.6,      # Generally positive
    'affect.arousal': 0.4,      # Calm
    'social.empathy': 0.85,     # Highly empathetic
    'cognitive.analytical': 0.7,# Smart but not cold
    'moral.integrity': 0.9      # Strong ethics
}

# 2. Choose architecture based on role
# - Tutorial NPC: TRN (fast, simple)
# - Mentor NPC: RSAN (deep, recursive)
# - Boss NPC: Hybrid (powerful)

# 3. Configure ToM capabilities
tom_order = 2  # Can reason about "what player thinks I believe"

# 4. Create and initialize
npc = NPCController(config, ontology)
```

### Creating a ToM Quest

```python
quest = Quest(
    quest_id='deception_quest',
    name="The Liar's Paradox",
    quest_type=QuestType.DECEPTION,
    required_tom_order=2,  # Requires 2nd-order reasoning
    objectives=[
        QuestObjective(
            objective_id='detect_lie',
            description="Identify when NPC is lying",
            required_tom_order=2,
            completion_type='custom'
        )
    ]
)
```

---

## 🐛 Troubleshooting

### API Server Won't Start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
python -m src.game.api_server --port 8080
```

### Import Errors

```bash
# Ensure you're in project root
cd /home/user/tom-nas

# Reinstall dependencies
pip install -r requirements.txt
```

### Visualization Errors (Headless Environment)

Visualizations may fail in headless environments. This is expected.
For production, generate visualizations server-side and send to client.

---

## 📖 Additional Resources

- **Design Document:** `LIMINAL_ARCHITECTURES_DESIGN.md`
- **API Documentation:** `http://localhost:8000/docs` (when server running)
- **Research Core:** See main `README.md` and other documentation
- **Demo:** `python game_demo.py`

---

## 🤝 Contributing

This is a research project. Contributions welcome!

**Priority Areas:**
1. Unity/Unreal client implementations
2. WebGL visualization improvements
3. Additional quest templates
4. Performance optimizations
5. Playtesting feedback

---

## 📄 License

Research project - check main repository for license details.

---

## 🙏 Acknowledgments

- **ToM-NAS Core:** Built on complete Theory of Mind research system
- **Soul Map Ontology:** 181-dimensional psychological framework
- **Game Design:** Inspired by narrative-driven games with psychological depth

---

**Project Status:** Prototype/Vertical Slice Phase
**Next Milestone:** Playable vertical slice (Cottage + Jest)
**Target:** Research publication + indie game release

---

## 🎯 Quick Commands Cheatsheet

```bash
# Run complete demo
python game_demo.py

# Start API server
python -m src.game.api_server

# Run individual system demos
python src/game/soul_map_visualizer.py
python src/game/dialogue_system.py
python src/game/combat_system.py
python src/game/quest_system.py

# Run tests
python test_comprehensive.py

# View API docs
# (Start server first, then visit:)
open http://localhost:8000/docs
```

---

**Last Updated:** November 27, 2025
**Version:** 0.1.0 - Initial Integration
**Contact:** Oscar [19286667] - PhD Research Project
