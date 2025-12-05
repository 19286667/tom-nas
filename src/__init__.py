"""
ToM-NAS: Theory of Mind Neural Architecture Search
The Fractal Semiotic Engine

A comprehensive framework for evolving neural architectures capable of
high-order, transparent Theory of Mind reasoning through:

1. SEMIOTIC KNOWLEDGE GRAPH (Indra's Net) - The omnipresent semantic web
   where every entity exists in hyperlinked meaning.

2. COGNITIVE CORE (Mentalese + RSC) - TypeScript-style cognitive blocks
   enabling recursive self-compression and N-th order ToM.

3. POET EVOLUTIONARY CONTROLLER - Paired open-ended evolution with
   sociological genotypes forcing emergence of deep social cognition.

4. GODOT PHYSICAL BRIDGE - Symbol grounding connecting physics to meaning.

5. FRACTAL SIMULATION ENGINE - Recursive simulation tree where agents
   can spawn nested simulations to predict futures.

The physical is cognitive. The cognitive is physical.
In Indra's Net, each pearl reflects all others.
"""

__version__ = "2.0.0"
__author__ = "ToM-NAS Research Team"

# Lazy imports to allow importing specific submodules without loading all dependencies
# Use: from src.godot_bridge import GodotBridge (works without torch)
# Or: import src; src.core (loads torch when accessed)

__all__ = [
    # Original modules
    'core',
    'agents',
    'world',
    'evolution',
    'evaluation',
    # Fractal Semiotic Engine modules
    'knowledge_base',
    'cognition',
    'godot_bridge',
    'simulation',
    # Metadata
    '__version__',
]


def __getattr__(name):
    """Lazy import of submodules."""
    if name in __all__ and name != '__version__':
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
