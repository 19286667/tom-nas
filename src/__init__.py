"""
ToM-NAS: Theory of Mind Neural Architecture Search

A comprehensive framework for evolving neural architectures capable of
Theory of Mind reasoning, including recursive belief modeling, zombie detection,
and social world simulation.

Key modules:
- core: Soul Map ontology and recursive belief structures
- agents: Neural architecture implementations (TRN, RSAN, Transformer)
- world: Social world simulation with observation tracking
- evolution: NAS engine and fitness evaluation
- evaluation: Benchmarks and metrics
- data: Event representation and ToMi benchmark loading (NEW)
- nas: Efficient NAS with zero-cost proxies (NEW)
"""

__version__ = "1.1.0"
__author__ = "ToM-NAS Research Team"

from . import core
from . import agents
from . import world
from . import evolution
from . import evaluation
from . import data
from . import nas

__all__ = [
    'core',
    'agents',
    'world',
    'evolution',
    'evaluation',
    'data',
    'nas',
    '__version__',
]
