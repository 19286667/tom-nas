"""
Simulation Engine Runner

Wraps the fractal simulation tree and exposes it via FastAPI.
This is the entry point for the Docker container.
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .fractal_node import (
    SimulationConfig,
    RootSimulationNode,
    RSCAgent,
    create_simulation,
)
from src.config import get_logger

logger = get_logger(__name__)


class EngineState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class SimulationEngine:
    """
    Main simulation engine that runs the world.

    Manages the simulation loop and exposes state to connected clients.
    """

    tick_rate: float = 10.0  # Ticks per second
    max_agents: int = 100

    def __post_init__(self):
        self.state = EngineState.STOPPED
        self.root: Optional[RootSimulationNode] = None
        self.tick_count = 0
        self.start_time: Optional[float] = None

        # Connected WebSocket clients
        self.clients: List[WebSocket] = []
        self._lock = threading.Lock()

        # Event queue for client commands
        self.command_queue: asyncio.Queue = asyncio.Queue()

        # Stats
        self.stats = {
            "ticks": 0,
            "agents": 0,
            "active_sims": 0,
            "avg_tick_ms": 0.0,
        }

    def initialize(self, config: Optional[SimulationConfig] = None):
        """Initialize the simulation world."""
        config = config or SimulationConfig(
            max_recursive_depth=5,
            entropy_threshold=0.1,
            max_ticks_per_step=100,
            enable_visualization=True,
        )

        self.root = RootSimulationNode(config=config)
        self.root.initialize({})

        # Populate with initial agents
        self._spawn_initial_agents()

        logger.info("Simulation engine initialized")

    def _spawn_initial_agents(self):
        """Create initial population of researcher agents."""
        from src.institutions.researcher_agent import ResearcherAgent

        # Create agents across different realms
        realms = ["hollow", "market", "ministry", "court", "temple"]

        for i, realm in enumerate(realms):
            for j in range(3):  # 3 agents per realm initially
                agent_id = f"agent_{realm}_{j}"
                agent = ResearcherAgent(
                    agent_id=agent_id,
                    name=f"Dr. {realm.title()} {j+1}",
                    specialty=realm,
                    institution_type="research_lab",
                )
                self.root.add_agent(agent)

        logger.info(f"Spawned {len(self.root.agents)} initial agents")

    async def run(self):
        """Main simulation loop."""
        self.state = EngineState.RUNNING
        self.start_time = time.time()
        tick_duration = 1.0 / self.tick_rate

        logger.info(f"Simulation running at {self.tick_rate} ticks/sec")

        while self.state == EngineState.RUNNING:
            tick_start = time.time()

            # Process any queued commands
            await self._process_commands()

            # Run simulation tick
            if self.root:
                result = self.root.step()
                self.tick_count += 1

                # Update stats
                self.stats["ticks"] = self.tick_count
                self.stats["agents"] = len(self.root.agents)
                self.stats["active_sims"] = self._count_active_sims()

                # Broadcast state to clients
                await self._broadcast_state(result)

            # Maintain tick rate
            tick_elapsed = time.time() - tick_start
            self.stats["avg_tick_ms"] = tick_elapsed * 1000

            sleep_time = max(0, tick_duration - tick_elapsed)
            await asyncio.sleep(sleep_time)

    def _count_active_sims(self) -> int:
        """Count all active child simulations."""
        if not self.root:
            return 0

        count = 0
        def count_children(node):
            nonlocal count
            count += len(node.child_nodes)
            for child in node.child_nodes.values():
                count_children(child)

        count_children(self.root)
        return count

    async def _process_commands(self):
        """Process commands from clients."""
        while not self.command_queue.empty():
            try:
                cmd = await asyncio.wait_for(
                    self.command_queue.get(),
                    timeout=0.01
                )
                await self._handle_command(cmd)
            except asyncio.TimeoutError:
                break

    async def _handle_command(self, cmd: Dict[str, Any]):
        """Handle a client command."""
        cmd_type = cmd.get("type")

        if cmd_type == "pause":
            self.state = EngineState.PAUSED
            logger.info("Simulation paused")

        elif cmd_type == "resume":
            self.state = EngineState.RUNNING
            logger.info("Simulation resumed")

        elif cmd_type == "query_agent":
            agent_id = cmd.get("agent_id")
            if self.root and agent_id in self.root.agents:
                agent = self.root.agents[agent_id]
                # Return agent details to requesting client

        elif cmd_type == "interact":
            # Player interaction with agent/object
            target_id = cmd.get("target_id")
            interaction = cmd.get("interaction")
            # Handle interaction

    async def _broadcast_state(self, tick_result: Dict[str, Any]):
        """Broadcast simulation state to all connected clients."""
        state_update = self._build_state_update(tick_result)
        message = json.dumps(state_update)

        disconnected = []
        for client in self.clients:
            try:
                await client.send_text(message)
            except:
                disconnected.append(client)

        # Clean up disconnected clients
        for client in disconnected:
            self.clients.remove(client)

    def _build_state_update(self, tick_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build state update message for clients."""
        agents_data = []

        if self.root:
            for agent_id, agent in self.root.agents.items():
                agents_data.append({
                    "id": agent_id,
                    "name": getattr(agent, 'name', agent_id),
                    "position": getattr(agent, 'position', {"x": 0, "y": 0, "z": 0}),
                    "activity": getattr(agent, 'current_activity', "idle"),
                    "thought": getattr(agent, 'current_thought', ""),
                    "realm": getattr(agent, 'current_realm', "hollow"),
                })

        return {
            "type": "state_update",
            "tick": self.tick_count,
            "time": time.time() - (self.start_time or time.time()),
            "stats": self.stats,
            "agents": agents_data,
            "tree": self.root.get_tree_structure() if self.root else {},
            "events": tick_result.get("agent_actions", [])[:10],  # Last 10 events
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {
            "engine_state": self.state.value,
            "tick": self.tick_count,
            "stats": self.stats,
            "tree": self.root.get_tree_structure() if self.root else {},
        }

    def pause(self):
        """Pause the simulation."""
        self.state = EngineState.PAUSED

    def resume(self):
        """Resume the simulation."""
        self.state = EngineState.RUNNING


# FastAPI application
app = FastAPI(
    title="ToM-NAS Simulation",
    description="Immersive Theory of Mind Simulation Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = SimulationEngine()


@app.on_event("startup")
async def startup():
    """Initialize and start simulation on server startup."""
    engine.initialize()
    asyncio.create_task(engine.run())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine": engine.state.value,
        "tick": engine.tick_count,
    }


@app.get("/state")
async def get_state():
    """Get current simulation state."""
    return engine.get_state()


@app.get("/agents")
async def get_agents():
    """Get all agents."""
    if not engine.root:
        return {"agents": []}
    return {
        "agents": [
            {
                "id": aid,
                "name": getattr(a, 'name', aid),
                "specialty": getattr(a, 'specialty', 'general'),
            }
            for aid, a in engine.root.agents.items()
        ]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details."""
    if not engine.root or agent_id not in engine.root.agents:
        return {"error": "Agent not found"}

    agent = engine.root.agents[agent_id]
    return {
        "id": agent_id,
        "name": getattr(agent, 'name', agent_id),
        "specialty": getattr(agent, 'specialty', 'general'),
        "tom_depth": agent.tom_depth,
        "beliefs": getattr(agent, 'beliefs', {}),
        "publications": getattr(agent, 'publications', []),
    }


@app.post("/command")
async def send_command(command: Dict[str, Any]):
    """Send command to simulation."""
    await engine.command_queue.put(command)
    return {"status": "queued"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    engine.clients.append(websocket)

    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_text()
            cmd = json.loads(data)
            await engine.command_queue.put(cmd)
    except WebSocketDisconnect:
        engine.clients.remove(websocket)


def main():
    """Entry point for Docker container."""
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "src.simulation.engine:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
