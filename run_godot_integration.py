#!/usr/bin/env python3
"""
Godot Integration Runner

Launches the Python WebSocket server and optionally the Godot simulation.
This is the main entry point for running ToM-NAS with full Godot integration.

Usage:
    python run_godot_integration.py [--no-godot] [--demo] [--scenario <name>]

Options:
    --no-godot      Don't launch Godot, only start the Python server
    --demo          Run a demonstration scenario
    --scenario      Specify scenario name (sally_anne, market_exchange, etc.)
    --host          WebSocket host (default: localhost)
    --port          WebSocket port (default: 9080)
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.godot_bridge.bridge import GodotBridge, BridgeConfig
from src.godot_bridge.protocol import MessageType, GodotMessage
from src.godot_bridge.symbol_grounding import SymbolGrounder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('GodotIntegration')


class GodotIntegrationRunner:
    """
    Manages the complete Godot-Python integration.

    Handles:
    - WebSocket server startup
    - Godot process management
    - Scenario execution
    - Demo mode
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 9080,
        godot_path: str = None,
        project_path: str = None
    ):
        self.host = host
        self.port = port
        self.godot_path = godot_path or self._find_godot()
        self.project_path = project_path or str(PROJECT_ROOT / 'godot_project')

        # Components
        self.bridge: GodotBridge = None
        self.godot_process: subprocess.Popen = None
        self.symbol_grounder: SymbolGrounder = None

        # State
        self.running = False
        self.scenario_active = False

    def _find_godot(self) -> str:
        """Find Godot executable."""
        # Common paths
        possible_paths = [
            'godot',
            'godot4',
            '/usr/bin/godot',
            '/usr/bin/godot4',
            '/usr/local/bin/godot',
            '/Applications/Godot.app/Contents/MacOS/Godot',
            os.path.expanduser('~/.local/bin/godot'),
        ]

        for path in possible_paths:
            if os.path.exists(path) or self._command_exists(path):
                return path

        return 'godot'  # Hope it's in PATH

    def _command_exists(self, cmd: str) -> bool:
        """Check if command exists."""
        try:
            subprocess.run(
                ['which', cmd],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def start_bridge(self) -> GodotBridge:
        """Start the WebSocket bridge server."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        config = BridgeConfig(
            host=self.host,
            port=self.port,
            enable_logging=True,
            log_messages=True
        )

        self.symbol_grounder = SymbolGrounder()
        self.bridge = GodotBridge(
            config=config,
            symbol_grounder=self.symbol_grounder
        )

        # Register custom handlers
        self._register_handlers()

        # Start bridge (non-blocking)
        self.bridge.start(blocking=False)

        logger.info("WebSocket server started")
        return self.bridge

    def _register_handlers(self):
        """Register custom message handlers."""

        @self.bridge.on(MessageType.WORLD_STATE)
        def on_world_state(msg: GodotMessage):
            logger.info(f"Received world state with {len(msg.payload.get('entities', []))} entities")

        @self.bridge.on(MessageType.AGENT_PERCEPTION)
        def on_perception(msg: GodotMessage):
            agent_name = msg.payload.get('agent_name', 'Unknown')
            visible_count = len(msg.payload.get('visible_entities', []))
            logger.debug(f"Agent {agent_name} perceives {visible_count} entities")

        @self.bridge.on(MessageType.INTERACTION_EVENT)
        def on_interaction(msg: GodotMessage):
            interaction_type = msg.payload.get('interaction_type', 'unknown')
            success = msg.payload.get('success', False)
            logger.info(f"Interaction: {interaction_type} - {'Success' if success else 'Failed'}")

        @self.bridge.on(MessageType.UTTERANCE_EVENT)
        def on_utterance(msg: GodotMessage):
            text = msg.payload.get('text', '')
            logger.info(f"Utterance: \"{text}\"")

    def start_godot(self, scene: str = None) -> subprocess.Popen:
        """Start the Godot process."""
        if not os.path.exists(self.project_path):
            logger.error(f"Godot project not found: {self.project_path}")
            return None

        cmd = [self.godot_path, '--path', self.project_path]

        if scene:
            cmd.extend(['--scene', scene])

        logger.info(f"Starting Godot: {' '.join(cmd)}")

        try:
            self.godot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"Godot started (PID: {self.godot_process.pid})")
            return self.godot_process
        except FileNotFoundError:
            logger.error(f"Godot executable not found: {self.godot_path}")
            return None

    def stop_godot(self):
        """Stop the Godot process."""
        if self.godot_process:
            logger.info("Stopping Godot...")
            self.godot_process.terminate()
            try:
                self.godot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.godot_process.kill()
            self.godot_process = None

    def stop_bridge(self):
        """Stop the WebSocket bridge."""
        if self.bridge:
            logger.info("Stopping WebSocket server...")
            self.bridge.stop()
            self.bridge = None

    def run_demo(self):
        """Run a demonstration of the integration."""
        logger.info("=== ToM-NAS Godot Integration Demo ===")

        # Wait for connection
        logger.info("Waiting for Godot to connect...")
        timeout = 30
        start_time = time.time()

        while not self.bridge.is_connected():
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for Godot connection")
                return False
            time.sleep(0.5)

        logger.info("Godot connected!")

        # Request world state
        self.bridge.request_world_state()
        time.sleep(1)

        # Get statistics
        stats = self.bridge.get_statistics()
        logger.info(f"Bridge statistics: {stats}")

        grounding_stats = self.symbol_grounder.get_statistics()
        logger.info(f"Grounding statistics: {grounding_stats}")

        return True

    def load_scenario(self, scenario_name: str):
        """Load a specific scenario."""
        logger.info(f"Loading scenario: {scenario_name}")

        if not self.bridge.is_connected():
            logger.error("Not connected to Godot")
            return False

        # Send scenario load command
        self.bridge._send_message(GodotMessage(
            message_type=MessageType.WORLD_COMMAND,
            payload={
                'command': 'load_scenario',
                'scenario': scenario_name
            }
        ))

        self.scenario_active = True
        return True

    async def run_async(
        self,
        launch_godot: bool = True,
        demo: bool = False,
        scenario: str = None
    ):
        """Run the integration asynchronously."""
        self.running = True

        try:
            # Start bridge
            self.start_bridge()

            # Start Godot if requested
            if launch_godot:
                self.start_godot()

            # Wait a moment for startup
            await asyncio.sleep(2)

            # Run demo if requested
            if demo:
                self.run_demo()

            # Load scenario if specified
            if scenario:
                self.load_scenario(scenario)

            # Main loop
            logger.info("Integration running. Press Ctrl+C to stop.")

            while self.running:
                await asyncio.sleep(0.1)

                # Check if Godot is still running
                if self.godot_process and self.godot_process.poll() is not None:
                    logger.info("Godot process ended")
                    self.running = False

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def run(
        self,
        launch_godot: bool = True,
        demo: bool = False,
        scenario: str = None
    ):
        """Run the integration (blocking)."""
        asyncio.run(self.run_async(launch_godot, demo, scenario))

    def stop(self):
        """Stop all components."""
        self.running = False
        self.stop_godot()
        self.stop_bridge()
        logger.info("Integration stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run ToM-NAS with Godot integration'
    )
    parser.add_argument(
        '--no-godot',
        action='store_true',
        help='Only start Python server, don\'t launch Godot'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration mode'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        help='Load specific scenario (sally_anne, market_exchange, etc.)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='WebSocket host'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9080,
        help='WebSocket port'
    )
    parser.add_argument(
        '--godot-path',
        type=str,
        help='Path to Godot executable'
    )

    args = parser.parse_args()

    runner = GodotIntegrationRunner(
        host=args.host,
        port=args.port,
        godot_path=args.godot_path
    )

    runner.run(
        launch_godot=not args.no_godot,
        demo=args.demo,
        scenario=args.scenario
    )


if __name__ == '__main__':
    main()
