#!/usr/bin/env python3
"""
ToM-NAS Godot Bridge Server - Unified Entry Point

This script starts the WebSocket server that connects Godot to the
full ToM-NAS cognitive architecture.

Usage:
    python run_godot_server.py              # Start server on localhost:9080
    python run_godot_server.py --debug      # With verbose logging
    python run_godot_server.py --port 8080  # Custom port

The server provides:
- Tier 2: Strategic decision-making using SoulMap psychological ontology
- Tier 3: Deep Theory of Mind via BeliefNetwork (up to 5th-order)
- Dialogue generation based on NPC psychology
- Cognitive hazard effects (fear, validation, etc.)
"""

import sys
import os

# Ensure we're running from project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the server
from godot.python.godot_server import main

if __name__ == "__main__":
    main()
