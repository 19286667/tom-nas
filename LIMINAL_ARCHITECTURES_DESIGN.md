# LIMINAL ARCHITECTURES: ToM-NAS Game Integration

**Project:** Theory of Mind meets AAA Game Design
**Status:** Integration Phase
**Date:** November 27, 2025

---

## 🎮 Vision

LIMINAL ARCHITECTURES transforms ToM-NAS from a research project into an interactive AAA game where players explore consciousness, empathy, and theory of mind through gameplay. Every NPC is powered by genuine Theory of Mind neural architectures, creating unprecedented depth in character interaction.

---

## 🏗️ Architecture Overview

### Core Integration Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    GAME ENGINE (Unity/Unreal)                │
├─────────────────────────────────────────────────────────────┤
│                  Game Integration Layer                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │  Soul    │   NPC    │ Dialogue │ Combat   │  Quest   │  │
│  │   Map    │ Controller│  System  │ System   │ System   │  │
│  │  Visual  │          │          │          │          │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                ToM-NAS Research Core                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │ 181-dim  │ 5th Order│   TRN    │  RSAN    │Transform │  │
│  │ Ontology │ Beliefs  │   Agent  │  Agent   │   Agent  │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Social World 4                            │
│         Coalition · Reputation · Communication               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Core Systems

### 1. Soul Map Visualization System

**Purpose:** Real-time visualization of psychological states

**Features:**
- **Radar Display:** 181-dimensional ontology projected to 8-axis radar
- **Aura View:** Visual representation of emotional/mental state
- **Dimension Inspector:** Deep-dive into specific psychological layers
- **Prediction Overlay:** Show NPC's model of player's Soul Map
- **History Timeline:** Track psychological state changes over time

**Implementation:**
- WebGL/Three.js for browser version
- Custom shaders for Unity/Unreal
- Real-time updates from ToM-NAS backend

### 2. NPC Architecture (ToM-Powered Characters)

**Purpose:** Every NPC has genuine theory of mind

**Agent Types:**
- **TRN NPCs:** Fast, reflexive, emotional characters
- **RSAN NPCs:** Deep, recursive thinkers with nested beliefs
- **Transformer NPCs:** Contextual, adaptive characters
- **Hybrid NPCs:** Boss characters and key story NPCs

**NPC Features:**
- Maintains own Soul Map (181 dimensions)
- Models player's Soul Map (theory of mind)
- Recursive belief reasoning (up to 5th order)
- Memory of past interactions
- Dynamic relationship evolution
- Coalition formation with other NPCs

### 3. Dialogue System with ToM Reasoning

**Purpose:** Conversations that respond to psychological state

**Features:**
- **Intent Prediction:** NPC predicts what player will say
- **Subtext Analysis:** Hidden meanings and implications
- **Emotional Resonance:** Dialogue options affect Soul Map
- **Belief Revelation:** Discover NPC's nested beliefs
- **Deception Detection:** Spot lies and inconsistencies
- **Persuasion Mechanics:** Use ToM to influence NPCs

**Dialogue Structure:**
```json
{
  "node_id": "cottage_01_greeting",
  "speaker": "Peregrine",
  "text": "You seem troubled, traveler.",
  "tom_reasoning": {
    "observes": ["player.affect.sadness > 0.7"],
    "believes": "Player is experiencing emotional distress",
    "second_order": "Player knows I can see their distress",
    "response_rationale": "Offer empathy to build trust"
  },
  "options": [
    {
      "text": "I'm fine. Just tired.",
      "soul_map_delta": {"affect.defensiveness": +0.2},
      "tom_effect": "Peregrine detects deception",
      "next": "cottage_01_deception"
    },
    {
      "text": "Yes... I've lost something important.",
      "soul_map_delta": {"affect.vulnerability": +0.3, "social.trust": +0.2},
      "tom_effect": "Peregrine notes honesty, increases empathy",
      "next": "cottage_01_honest"
    }
  ]
}
```

### 4. Psychological Combat System

**Purpose:** Combat that targets mind as well as body

**Combat Wheels:**
- **Physical Damage:** Traditional HP system
- **Psychological Damage:** Affects Soul Map dimensions
- **Vulnerability Mapping:** Use ToM to discover weaknesses
- **Emotional Attacks:** Fear, Shame, Confusion
- **Defense Mechanisms:** Rationalization, Denial, Projection
- **Belief Shattering:** Challenge core beliefs for massive damage

**Combat Example:**
```
Player uses "Reveal Hypocrisy" on Enemy
→ Enemy's Soul Map analyzed for inconsistencies
→ Enemy has high moral.justice but low moral.compassion
→ Attack exploits this contradiction
→ Psychological damage: -30 Coherence
→ Enemy becomes Confused (reduced accuracy)
```

### 5. ToM-Driven Quest System

**Purpose:** Quests that require understanding mental states

**Quest Types:**

**Tutorial Quest: "The Cottage Test"**
- Learn Soul Map basics
- Practice first-order ToM (what does Peregrine want?)
- Simple belief reasoning

**Recursive Quest: "The Infinite Jest"**
- Recursive ToM required (what does A think B thinks C wants?)
- Coalition dynamics
- Deception and counter-deception
- Multiple solution paths based on psychological insight

**Zombie Detection Quest: "The Hollow Ones"**
- Identify NPCs without genuine ToM
- Behavioral inconsistency detection
- Philosophical implications of consciousness

**Quest Structure:**
```python
class ToMQuest:
    def __init__(self, quest_id, required_tom_order=1):
        self.id = quest_id
        self.required_tom_order = required_tom_order
        self.objectives = []

    def check_completion(self, player_action):
        # Did player demonstrate required ToM depth?
        if player_action.tom_depth >= self.required_tom_order:
            return True
        return False
```

### 6. Research Telemetry System

**Purpose:** Collect consent-based research data

**Features:**
- **Opt-in Consent:** Clear, granular permissions
- **Anonymization:** No PII collected
- **IRB Compliance:** Ethical research standards
- **Data Dashboard:** Players see their contributions
- **Research Credits:** Players acknowledged in publications

**Data Collected (with consent):**
- ToM task success rates
- Dialogue choice patterns
- Soul Map evolution over time
- Prediction accuracy
- Quest solving strategies
- NPC believability ratings

---

## 🎨 Vertical Slice: "The Infinite Jest & Peregrine's Cottage"

### Setting

**The Infinite Jest** - A mysterious tavern where identities are fluid

**Peregrine's Cottage** - A safe space for self-reflection

### Characters (Fully Implemented)

**Peregrine (RSAN-powered)**
- **Role:** Mentor, Empathic Guide
- **Soul Map:** High wisdom.self_awareness, compassion, patience
- **ToM Capability:** 5th order (deeply understands nested beliefs)
- **Dialogue Style:** Socratic, reflective, non-judgmental

**The Barkeep (Transformer-powered)**
- **Role:** Mysterious Host, Information Broker
- **Soul Map:** High cognitive.strategic_thinking, low emotional.vulnerability
- **ToM Capability:** 4th order (reads customers masterfully)
- **Dialogue Style:** Cryptic, playful, observant

**The Stranger (TRN-powered)**
- **Role:** Antagonist, Deceiver
- **Soul Map:** High motivation.power, low moral.honesty
- **ToM Capability:** 3rd order (manipulative but not deep)
- **Dialogue Style:** Charming, deceptive, agenda-driven

### Tutorial Flow

1. **Awakening** - Player gains Soul Map vision
2. **First Contact** - Meet Peregrine, learn basic ToM
3. **The Cottage Test** - Practice reading emotional states
4. **Journey to the Jest** - Navigate social dynamics
5. **The Recursive Challenge** - Solve multi-layered deception
6. **Revelation** - Discover your own psychological truth

---

## 🛠️ Technical Implementation

### Technology Stack

**Backend (Python)**
- ToM-NAS core (existing)
- Flask/FastAPI REST API
- WebSocket for real-time updates
- Redis for session state
- PostgreSQL for player data

**Frontend (Unity/Unreal)**
- C# (Unity) / C++ (Unreal) game logic
- HTTP client for API calls
- WebSocket client for real-time ToM updates
- Custom shaders for Soul Map visualization
- UI framework for dialogue/combat

**Middleware**
- Python ↔ Unity/Unreal bridge
- JSON for data serialization
- Efficient batching for performance

### Performance Optimization

**Challenge:** ToM inference is computationally expensive

**Solutions:**
1. **Pre-compute Common States:** Cache frequent Soul Map patterns
2. **Async Processing:** NPCs update in background
3. **Importance Sampling:** Focus computation on nearby/important NPCs
4. **Level-of-Detail ToM:** Distant NPCs use simpler models
5. **GPU Acceleration:** PyTorch CUDA for batch inference

---

## 📊 Development Roadmap

### Phase 1: Core Integration (Weeks 1-4)
- [x] Design architecture
- [ ] Build REST API for ToM-NAS
- [ ] Create Soul Map visualization (WebGL prototype)
- [ ] Implement basic NPC controller
- [ ] Design dialogue system schema

### Phase 2: Vertical Slice (Weeks 5-12)
- [ ] Build Peregrine's Cottage (environment)
- [ ] Implement 3 main NPCs (Peregrine, Barkeep, Stranger)
- [ ] Create tutorial quest chain
- [ ] Develop Soul Map UI
- [ ] Basic combat prototype

### Phase 3: Polish & Expand (Weeks 13-24)
- [ ] Full combat system
- [ ] Additional locations and NPCs
- [ ] Quest variety expansion
- [ ] Performance optimization
- [ ] Playtesting and iteration

### Phase 4: Research Integration (Weeks 25-30)
- [ ] Telemetry system
- [ ] Consent framework
- [ ] Data analysis pipeline
- [ ] Publication preparation

---

## 🎓 Research Contributions

### Novel Aspects

1. **First playable ToM system** - Not just benchmarks, but gameplay
2. **Human-AI ToM interaction** - How do humans reason about AI minds?
3. **Ecological validity** - ToM in rich, naturalistic contexts
4. **Comparative architectures** - Players experience TRN vs RSAN vs Transformer NPCs
5. **Explainable AI gaming** - Transparent reasoning creates gameplay

### Publications Potential

- **NeurIPS/ICLR:** ToM architectures and evaluation
- **CHI/IUI:** Human-AI interaction and UX research
- **AIIDE/FDG:** Game AI and narrative systems
- **Cognitive Science:** Theory of mind in naturalistic settings

---

## 💰 Funding & Viability

### MVP Budget (Indie Scope)

**Team:** 5-10 people, 6-12 months
- 1 Technical Director
- 2 AI/ML Engineers
- 2 Game Developers
- 1 UI/UX Designer
- 1 Narrative Designer
- 1 3D Artist (environment)

**Estimated Cost:** $500K - $1M
**Funding Sources:** Research grants, indie publisher, Kickstarter

### AAA Budget (Full Vision)

**Team:** 50-100 people, 24-36 months
**Estimated Cost:** $50M - $100M
**Funding Sources:** Major publisher, VC investment

---

## 🚀 Next Steps (Immediate Actions)

### 1. Build REST API (Today)
```bash
cd /home/user/tom-nas
python -m src.game.api_server
```

### 2. Create Soul Map Visualizer (This Week)
- WebGL prototype using Three.js
- Real-time updates from API
- Interactive exploration

### 3. Implement First NPC (This Week)
- Peregrine character controller
- Basic dialogue tree
- Soul Map integration

### 4. Prototype Combat (Next Week)
- Physical + Psychological damage
- Vulnerability system
- Simple enemy AI

### 5. Build Demo Scene (2 Weeks)
- Cottage environment
- Tutorial quest
- Full interaction loop

---

## 📖 Documentation Structure

```
/docs/
├── DESIGN_DOCUMENT.md           # This file
├── API_SPECIFICATION.md          # REST API docs
├── SOUL_MAP_GUIDE.md            # Psychology ontology
├── NPC_DEVELOPMENT_GUIDE.md     # Creating ToM NPCs
├── DIALOGUE_SYSTEM.md           # Writing ToM dialogue
├── COMBAT_MECHANICS.md          # Psychological combat
├── QUEST_DESIGN.md              # ToM quest patterns
└── RESEARCH_PROTOCOL.md         # Ethics & data collection
```

---

## ✨ Unique Selling Points

### For Players
- **Unprecedented NPC Depth:** Characters that truly understand you
- **Meaningful Choices:** Psychological consequences, not just branching paths
- **Self-Discovery:** Explore your own mind through gameplay
- **Intellectual Challenge:** Puzzles requiring empathy and insight

### For Researchers
- **Ecological ToM Evaluation:** Real-world complexity
- **Large-Scale Data:** Thousands of ToM interactions
- **Human Baselines:** Compare human vs AI theory of mind
- **Open Science:** Reproducible, published research

### For Industry
- **Next-Gen AI:** Beyond chatbots to genuine understanding
- **Innovation Showcase:** Cutting-edge ML in production
- **Market Differentiation:** No other game has this technology
- **Awards Potential:** Technical innovation + artistic vision

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)
- ✅ ToM-NAS core functional
- [ ] 1 playable location (Cottage or Jest)
- [ ] 3 fully-realized NPCs
- [ ] Soul Map visualization working
- [ ] Tutorial quest completable
- [ ] 30-60 minutes of gameplay

### Exceptional Launch Product
- [ ] 5 locations across multiple realms
- [ ] 20+ unique NPCs
- [ ] Full combat and quest systems
- [ ] 10-20 hours of gameplay
- [ ] Research paper published
- [ ] Positive critical reception

---

## 🌟 Tagline Ideas

*"The game that reads your mind... and has one of its own."*

*"Where every conversation matters. Because they remember."*

*"Explore consciousness. Question reality. Understand others."*

*"LIMINAL ARCHITECTURES: The boundary between minds."*

---

**Last Updated:** November 27, 2025
**Status:** Architecture Complete, Implementation Beginning
**Next Milestone:** Working API + Soul Map Visualizer
