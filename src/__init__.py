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

from . import core
from . import agents
from . import world
from . import evolution
from . import evaluation
from . import knowledge_base
from . import cognition
from . import godot_bridge
from . import simulation

__all__ = [
    # Original modules
    "core",
    "agents",
    "world",
    "evolution",
    "evaluation",
    # Fractal Semiotic Engine modules
    "knowledge_base",
    "cognition",
    "godot_bridge",
    "simulation",
    # Metadata
    "__version__",
]
