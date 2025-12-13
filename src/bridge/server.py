"""
WebSocket Bridge Server

Connects multiple web clients to the simulation engine.
Handles:
- Real-time state broadcasting
- Client command forwarding
- Connection management
"""

import asyncio
import json
import os
import logging
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, field
import aiohttp
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BridgeServer:
    """
    WebSocket bridge between web clients and simulation.
    """

    simulation_host: str = "simulation"
    simulation_port: int = 8000
    websocket_port: int = 8765
    max_clients: int = 10

    def __post_init__(self):
        self.clients: Set[WebSocketServerProtocol] = set()
        self.simulation_url = f"http://{self.simulation_host}:{self.simulation_port}"
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the bridge server."""
        logger.info(f"Starting bridge server on port {self.websocket_port}")
        logger.info(f"Connecting to simulation at {self.simulation_url}")

        self._running = True

        # Start polling simulation for state updates
        self._poll_task = asyncio.create_task(self.poll_simulation())

        # Start WebSocket server
        async with websockets.serve(
            self.handle_client,
            "0.0.0.0",
            self.websocket_port,
            ping_interval=20,
            ping_timeout=60,
        ):
            logger.info(f"Bridge server listening on ws://0.0.0.0:{self.websocket_port}")
            await asyncio.Future()  # Run forever

    async def stop(self):
        """Stop the bridge server."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()

        # Close all client connections
        for client in self.clients:
            await client.close()

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new client connection."""
        if len(self.clients) >= self.max_clients:
            await websocket.close(1008, "Server full")
            return

        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected. Total: {len(self.clients)}")

        try:
            # Send current state immediately
            state = await self.get_simulation_state()
            if state:
                await websocket.send(json.dumps({
                    "type": "state_update",
                    **state
                }))

            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client {client_id} disconnected. Total: {len(self.clients)}")

    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "pause":
                await self.send_command({"type": "pause"})

            elif msg_type == "resume":
                await self.send_command({"type": "resume"})

            elif msg_type == "query_agent":
                agent_id = data.get("agent_id")
                agent_data = await self.get_agent_details(agent_id)
                if agent_data:
                    await websocket.send(json.dumps({
                        "type": "agent_detail",
                        "data": agent_data
                    }))

            elif msg_type == "interact":
                await self.send_command({
                    "type": "interact",
                    "target_id": data.get("target_id"),
                    "interaction": data.get("interaction")
                })

            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def poll_simulation(self):
        """Poll simulation for state updates and broadcast to clients."""
        poll_interval = 0.1  # 10 updates per second

        while self._running:
            try:
                state = await self.get_simulation_state()
                if state and self.clients:
                    message = json.dumps({
                        "type": "state_update",
                        **state
                    })
                    await self.broadcast(message)

            except Exception as e:
                logger.error(f"Error polling simulation: {e}")

            await asyncio.sleep(poll_interval)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Clean up disconnected clients
        self.clients -= disconnected

    async def get_simulation_state(self) -> Optional[Dict[str, Any]]:
        """Get current state from simulation."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.simulation_url}/state") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except aiohttp.ClientError as e:
            logger.debug(f"Could not reach simulation: {e}")
        return None

    async def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about an agent."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.simulation_url}/agents/{agent_id}") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except aiohttp.ClientError as e:
            logger.debug(f"Could not get agent details: {e}")
        return None

    async def send_command(self, command: Dict[str, Any]):
        """Send command to simulation."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.simulation_url}/command",
                    json=command
                ) as resp:
                    return resp.status == 200
        except aiohttp.ClientError as e:
            logger.error(f"Could not send command: {e}")
        return False


async def main():
    """Entry point."""
    server = BridgeServer(
        simulation_host=os.getenv("SIMULATION_HOST", "simulation"),
        simulation_port=int(os.getenv("SIMULATION_PORT", "8000")),
        websocket_port=int(os.getenv("WEBSOCKET_PORT", "8765")),
        max_clients=int(os.getenv("MAX_CLIENTS", "10")),
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
