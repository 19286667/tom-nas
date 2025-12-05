"""
ToM-NAS Visualization Module

Streamlit-based visualization for the Liminal Architectures game environment
and Neural Architecture Search process.

Components:
- WorldRenderer: Visual rendering of Liminal realms and NPCs
- BeliefInspector: Visualization of information asymmetry and beliefs
- NASDashboard: Evolution progress and architecture analysis
- app: Main Streamlit application
"""

from .world_renderer import WorldRenderer
from .belief_inspector import BeliefInspector
from .nas_dashboard import NASDashboard

__all__ = [
    "WorldRenderer",
    "BeliefInspector",
    "NASDashboard",
]


def run_app():
    """Launch the Streamlit visualization app."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/visualization/app.py"])
