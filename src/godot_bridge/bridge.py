"""
Godot Bridge - WebSocket Communication Layer

Provides robust bidirectional communication between the Python
cognitive controller and the Godot 4.x physics simulation.

The bridge handles:
1. WebSocket server for Godot to connect
2. Message serialization/deserialization
3. Event dispatching to appropriate handlers
4. Connection management and heartbeats

This is the infrastructure layer that enables embodied cognition
by connecting the mind (Python) to the body (Godot).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Awaitable
import threading
from queue import Queue

from .protocol import GodotMessage, MessageType, EntityUpdate, AgentPerception, WorldState, AgentCommand
from .symbol_grounding import SymbolGrounder, GroundingContext
from .perception import PerceptionProcessor, PerceptualField
from .action import ActionExecutor, GodotAction, ActionResult

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """State of the Godot connection."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


@dataclass
class BridgeConfig:
    """Configuration for the Godot bridge."""

    host: str = "localhost"
    port: int = 9080
    heartbeat_interval_ms: float = 1000.0
    reconnect_delay_ms: float = 5000.0
    message_queue_size: int = 1000
    enable_logging: bool = True
    log_messages: bool = False  # Log all messages (verbose)


@dataclass
class ConnectionStats:
    """Statistics about the connection."""

    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    errors: int = 0


class GodotBridge:
    """
    WebSocket bridge for Godot-Python communication.

    Manages the connection to Godot and dispatches messages
    to appropriate handlers (perception, action, etc.).
    """

    def __init__(self, config: BridgeConfig, symbol_grounder: Optional[SymbolGrounder] = None, knowledge_base=None):
        """
        Initialize the Godot bridge.

        Args:
            config: Bridge configuration
            symbol_grounder: Symbol grounding system (created if None)
            knowledge_base: Indra's Net knowledge graph
        """
        self.config = config
        self.knowledge_base = knowledge_base

        # State
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()

        # Core systems
        self.grounder = symbol_grounder or SymbolGrounder(knowledge_base)
        self.perception_processor = PerceptionProcessor(self.grounder, knowledge_base)
        self.action_executor = ActionExecutor(self.grounder, self._send_message)

        # Message handling
        self.incoming_queue: Queue = Queue(maxsize=config.message_queue_size)
        self.outgoing_queue: Queue = Queue(maxsize=config.message_queue_size)

        # Event handlers
        self._handlers: Dict[MessageType, List[Callable]] = {msg_type: [] for msg_type in MessageType}

        # World state cache
        self.world_state: Optional[WorldState] = None
        self.last_world_update: Optional[datetime] = None

        # Agent perceptions cache
        self.agent_perceptions: Dict[int, PerceptualField] = {}

        # Websocket (set when running)
        self._websocket = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default message handlers."""

        @self.on(MessageType.ENTITY_UPDATE)
        def handle_entity_update(msg: GodotMessage):
            entity = EntityUpdate.from_dict(msg.payload)
            context = GroundingContext(simulation_time=msg.timestamp)
            self.grounder.ground_entity(entity, context)

        @self.on(MessageType.WORLD_STATE)
        def handle_world_state(msg: GodotMessage):
            self.world_state = WorldState(**msg.payload)
            self.last_world_update = datetime.now()

            # Ground all entities
            for entity_dict in msg.payload.get("entities", []):
                entity = EntityUpdate.from_dict(entity_dict)
                self.grounder.ground_entity(entity)

        @self.on(MessageType.AGENT_PERCEPTION)
        def handle_perception(msg: GodotMessage):
            perception = self._parse_perception(msg.payload)
            pfield = self.perception_processor.process_perception(perception)
            self.agent_perceptions[perception.agent_godot_id] = pfield

        @self.on(MessageType.HEARTBEAT)
        def handle_heartbeat(msg: GodotMessage):
            self.stats.last_heartbeat = datetime.now()
            # Send heartbeat response
            self._send_message(GodotMessage.create_heartbeat())

        @self.on(MessageType.ACK)
        def handle_ack(msg: GodotMessage):
            command_id = msg.payload.get("command_id")
            success = msg.payload.get("success", True)
            if command_id:
                self.action_executor.handle_result(
                    command_id, success, msg.payload.get("state_changes"), msg.payload.get("error")
                )

    def _parse_perception(self, payload: Dict) -> AgentPerception:
        """Parse perception payload into AgentPerception."""
        from .protocol import Vector3

        visible = [EntityUpdate.from_dict(e) for e in payload.get("visible_entities", [])]

        return AgentPerception(
            agent_godot_id=payload.get("agent_godot_id", 0),
            agent_name=payload.get("agent_name", ""),
            visible_entities=visible,
            occluded_entities=payload.get("occluded_entities", []),
            heard_utterances=payload.get("heard_utterances", []),
            own_position=Vector3(**payload.get("own_position", {})),
            own_velocity=Vector3(**payload.get("own_velocity", {})),
            own_orientation=Vector3(**payload.get("own_orientation", {})),
            energy_level=payload.get("energy_level", 1.0),
            held_object=payload.get("held_object"),
            current_institution=payload.get("current_institution"),
            timestamp=payload.get("timestamp", 0.0),
        )

    def on(self, message_type: MessageType):
        """
        Decorator to register a message handler.

        Usage:
            @bridge.on(MessageType.ENTITY_UPDATE)
            def handle_update(msg):
                ...
        """

        def decorator(handler: Callable[[GodotMessage], None]):
            self._handlers[message_type].append(handler)
            return handler

        return decorator

    def _dispatch_message(self, message: GodotMessage):
        """Dispatch a message to registered handlers."""
        handlers = self._handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Handler error for {message.message_type}: {e}")
                self.stats.errors += 1

    def _send_message(self, message: GodotMessage):
        """Queue a message to send to Godot."""
        if not self.outgoing_queue.full():
            self.outgoing_queue.put(message)
            self.stats.messages_sent += 1

    async def _send_loop(self):
        """Async loop for sending queued messages."""
        while self._running:
            try:
                # Non-blocking check
                if not self.outgoing_queue.empty():
                    message = self.outgoing_queue.get_nowait()
                    if self._websocket:
                        json_str = message.to_json()
                        await self._websocket.send(json_str)
                        self.stats.bytes_sent += len(json_str)

                        if self.config.log_messages:
                            logger.debug(f"Sent: {message.message_type.name}")

                await asyncio.sleep(0.01)  # Small delay to prevent busy loop

            except Exception as e:
                logger.error(f"Send error: {e}")
                self.stats.errors += 1

    async def _receive_loop(self):
        """Async loop for receiving messages."""
        while self._running and self._websocket:
            try:
                json_str = await self._websocket.recv()
                self.stats.bytes_received += len(json_str)
                self.stats.messages_received += 1

                message = GodotMessage.from_json(json_str)

                if self.config.log_messages:
                    logger.debug(f"Received: {message.message_type.name}")

                # Dispatch to handlers
                self._dispatch_message(message)

            except Exception as e:
                logger.error(f"Receive error: {e}")
                self.stats.errors += 1

    async def _heartbeat_loop(self):
        """Async loop for sending heartbeats."""
        interval_sec = self.config.heartbeat_interval_ms / 1000.0

        while self._running:
            await asyncio.sleep(interval_sec)
            self._send_message(GodotMessage.create_heartbeat())

    async def _run_async(self):
        """Main async run loop."""
        try:
            # Import websockets here to avoid import errors if not installed
            import websockets

            uri = f"ws://{self.config.host}:{self.config.port}"
            logger.info(f"Connecting to Godot at {uri}")
            self.state = ConnectionState.CONNECTING

            async with websockets.serve(self._handle_connection, self.config.host, self.config.port) as server:
                logger.info(f"WebSocket server listening on {uri}")
                self.state = ConnectionState.CONNECTED
                self.stats.connection_time = datetime.now()

                await asyncio.Future()  # Run forever

        except ImportError:
            logger.warning("websockets not installed, running in mock mode")
            self.state = ConnectionState.CONNECTED
            self.stats.connection_time = datetime.now()
            # Run without actual websocket
            while self._running:
                await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Bridge error: {e}")
            self.state = ConnectionState.ERROR
            self.stats.errors += 1

    async def _handle_connection(self, websocket, path):
        """Handle a websocket connection from Godot."""
        self._websocket = websocket
        logger.info("Godot connected!")

        try:
            # Start send/receive/heartbeat loops
            receive_task = asyncio.create_task(self._receive_loop())
            send_task = asyncio.create_task(self._send_loop())
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Wait for any to complete (usually means disconnect)
            done, pending = await asyncio.wait(
                [receive_task, send_task, heartbeat_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

        except Exception as e:
            logger.error(f"Connection error: {e}")

        finally:
            self._websocket = None
            logger.info("Godot disconnected")

    def start(self, blocking: bool = False):
        """
        Start the bridge.

        Args:
            blocking: If True, block until stopped. If False, run in background.
        """
        self._running = True

        if blocking:
            asyncio.run(self._run_async())
        else:
            # Run in background thread
            def run_in_thread():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_until_complete(self._run_async())

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()

    def stop(self):
        """Stop the bridge."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.state = ConnectionState.DISCONNECTED

    def is_connected(self) -> bool:
        """Check if connected to Godot."""
        return self.state == ConnectionState.CONNECTED and self._websocket is not None

    def send_command(self, command: AgentCommand):
        """Send a command to Godot."""
        message = GodotMessage.create_agent_command(command)
        self._send_message(message)

    def request_world_state(self):
        """Request full world state from Godot."""
        message = GodotMessage(message_type=MessageType.WORLD_COMMAND, payload={"command": "get_world_state"})
        self._send_message(message)

    def pause_simulation(self):
        """Pause the Godot simulation."""
        message = GodotMessage(message_type=MessageType.PAUSE)
        self._send_message(message)

    def resume_simulation(self):
        """Resume the Godot simulation."""
        message = GodotMessage(message_type=MessageType.RESUME)
        self._send_message(message)

    def step_simulation(self):
        """Step the simulation forward by one frame."""
        message = GodotMessage(message_type=MessageType.STEP)
        self._send_message(message)

    def reset_simulation(self):
        """Reset the Godot simulation."""
        message = GodotMessage(message_type=MessageType.RESET)
        self._send_message(message)

    def get_perception(self, agent_id: int) -> Optional[PerceptualField]:
        """Get latest perception for an agent."""
        return self.agent_perceptions.get(agent_id)

    def execute_action(self, action: GodotAction):
        """Execute an action through the action executor."""
        self.action_executor.queue_action(action)
        return self.action_executor.execute_next(action.agent_godot_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "connection": {
                "state": self.state.name,
                "connected_since": self.stats.connection_time.isoformat() if self.stats.connection_time else None,
                "last_heartbeat": self.stats.last_heartbeat.isoformat() if self.stats.last_heartbeat else None,
            },
            "messages": {
                "sent": self.stats.messages_sent,
                "received": self.stats.messages_received,
                "errors": self.stats.errors,
            },
            "bytes": {
                "sent": self.stats.bytes_sent,
                "received": self.stats.bytes_received,
            },
            "grounding": self.grounder.get_statistics(),
            "actions": self.action_executor.get_statistics(),
        }
