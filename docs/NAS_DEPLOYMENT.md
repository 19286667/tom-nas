# ToM-NAS: Immersive Simulation Environment

## Overview

This is not a dashboard - it's a **living world** you walk through.

The simulation runs continuously on your NAS. You connect via browser from any device on your network and explore:

- **Watch agents** conduct research, form hypotheses, publish findings
- **Enter institutions** (research labs, corporate R&D, government agencies)
- **Observe recursive simulations** - agents creating simulations containing agents
- **Interact** with the world - ask agents about their beliefs, trigger events

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR NAS (Old Laptop)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    Docker Container                         │     │
│  │                                                             │     │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │     │
│  │  │  Simulation │◄──►│  WebSocket  │◄──►│   Web UI    │    │     │
│  │  │   Engine    │    │   Bridge    │    │  (Three.js) │    │     │
│  │  │             │    │             │    │             │    │     │
│  │  │  • Agents   │    │  Real-time  │    │  • 3D World │    │     │
│  │  │  • Beliefs  │    │   state     │    │  • HUD      │    │     │
│  │  │  • Research │    │   sync      │    │  • Menus    │    │     │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    │     │
│  │         │                                     │            │     │
│  │         └─────────────────────────────────────┘            │     │
│  │                    Shared State                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│                              ▼                                       │
│                    ┌─────────────────┐                              │
│                    │   Port 8080     │                              │
│                    │   (Web Access)  │                              │
│                    └─────────────────┘                              │
│                              │                                       │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Your Browser       │
                    │  (Any device on LAN)│
                    │                     │
                    │  http://nas:8080    │
                    └─────────────────────┘
```

## The World

### Realms (Explorable Areas)

```
THE HOLLOW          THE MARKET          THE MINISTRY
┌───────────┐       ┌───────────┐       ┌───────────┐
│           │       │ $  $  $   │       │ ══════    │
│   Pure    │  ──►  │  Trade    │  ──►  │  Rules    │
│  Research │       │  Ideas    │       │  Policy   │
│           │       │           │       │           │
└───────────┘       └───────────┘       └───────────┘
     │                                        │
     ▼                                        ▼
THE COURT                              THE TEMPLE
┌───────────┐                          ┌───────────┐
│   ⚖       │                          │    ∞      │
│   Power   │         ◄────────        │ Meaning   │
│  Dynamics │                          │ Purpose   │
└───────────┘                          └───────────┘
```

### What You See

- **Agents** walking, thinking, interacting (visual representations)
- **Thought bubbles** showing current hypotheses
- **Publication boards** with recent research
- **Simulation portals** where agents run recursive experiments
- **Energy fields** showing constraint propagation (NSHE)
- **Relationship lines** between agents (collaboration/competition)

## Installation

### Prerequisites

1. **Hardware**: Any computer with at least 2GB RAM (old laptop works fine)
2. **OS**: Linux (Ubuntu, Debian, or any distro)
3. **Docker**: Container runtime

### Step 1: Install Linux

If repurposing an old laptop, we recommend **Ubuntu Server** for minimal overhead:

```bash
# Download Ubuntu Server from ubuntu.com
# Create bootable USB with Rufus/Etcher
# Boot and follow installer
```

### Step 2: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Add your user to docker group (then log out/in)
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin
```

### Step 3: Clone and Run

```bash
# Clone the repository
git clone https://github.com/19286667/tom-nas.git
cd tom-nas

# Make scripts executable
chmod +x start-nas.sh stop-nas.sh

# Start the simulation
./start-nas.sh

# Access from any device on your network
# http://<nas-ip>:80
```

### Finding Your NAS IP

```bash
# On the NAS itself:
ip addr show | grep inet

# Or from another device, check your router's connected devices
```

## Quick Start (for existing Docker setups)

```bash
# Start everything
docker-compose -f docker-compose.nas.yml up -d

# View logs
docker-compose -f docker-compose.nas.yml logs -f

# Stop
docker-compose -f docker-compose.nas.yml down
```

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look around |
| E | Interact with agent/object |
| Tab | Open world map |
| I | Inventory (collected publications) |
| P | Pause simulation |
| ~ | Console (advanced commands) |

## HUD Elements

```
┌─────────────────────────────────────────────────────────────────┐
│ [TIME: 2847] [AGENTS: 47] [ACTIVE SIMS: 3]         [⚙ MENU]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                                                                  │
│                        3D WORLD VIEW                             │
│                                                                  │
│                                                                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ NEARBY: Dr. Chen (Cognitive Science) - "Testing false belief"   │
│ RECENT: Publication #1847 - "Emergent ToM in recursive sims"   │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Details

### Services

| Service | Port | Description |
|---------|------|-------------|
| Simulation | 8000 (internal) | Core simulation engine |
| Bridge | 8765 | WebSocket real-time updates |
| Web UI | 80 | Three.js exploration interface |
| Proxy | 80 | Nginx reverse proxy |

### Resource Usage

- **RAM**: ~1.5GB with 50 agents
- **CPU**: Low (simulation is tick-based, not real-time)
- **Disk**: ~500MB for containers
- **Network**: Minimal (local only)

### API Endpoints

```bash
# Health check
curl http://localhost/health

# Get current state
curl http://localhost/api/state

# List agents
curl http://localhost/api/agents

# Get specific agent
curl http://localhost/api/agents/agent_hollow_0
```

### WebSocket Protocol

Connect to `ws://<nas-ip>:8765/ws` for real-time updates:

```javascript
// Receive state updates
{
  "type": "state_update",
  "tick": 1234,
  "agents": [...],
  "stats": {...}
}

// Send commands
ws.send(JSON.stringify({ "type": "pause" }));
ws.send(JSON.stringify({ "type": "resume" }));
```

## Troubleshooting

### Services not starting

```bash
# Check container status
docker ps -a

# View logs
docker-compose -f docker-compose.nas.yml logs simulation

# Restart a specific service
docker-compose -f docker-compose.nas.yml restart simulation
```

### Memory issues

If you see OOM errors, reduce agent count in `docker-compose.nas.yml`:
```yaml
environment:
  - TOM_MAX_AGENTS=30  # Reduce from default 100
```

### Network issues

If you can't connect from other devices:
```bash
# Check firewall
sudo ufw allow 80/tcp
sudo ufw allow 8765/tcp

# Or disable firewall temporarily
sudo ufw disable
```

## Auto-start on Boot

To start the simulation automatically when the NAS boots:

```bash
# Create systemd service
sudo tee /etc/systemd/system/tom-nas.service << EOF
[Unit]
Description=ToM-NAS Simulation
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/$USER/tom-nas
ExecStart=/usr/bin/docker compose -f docker-compose.nas.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.nas.yml down

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable tom-nas
sudo systemctl start tom-nas
```

## The Simulation

This isn't just visualization - the simulation IS the mechanism:

1. **Agents** have recursive beliefs (up to 5th order)
2. **Research** produces lambda calculus programs (intrinsically safe)
3. **Nested simulations** create genuine selective pressure for ToM
4. **Library compression** (Stitch) evolves reusable abstractions
5. **Verification** (NSHE, PIMMUR, PAN) ensures scientific rigor

Walking through the world lets you observe emergence in real-time.
