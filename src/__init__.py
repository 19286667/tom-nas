"""
ToM-NAS: Theory of Mind Neural Architecture Search

A comprehensive framework for evolving neural architectures capable of
Theory of Mind reasoning, including recursive belief modeling, zombie detection,
and social world simulation.
"""

__version__ = "1.0.0"
__author__ = "ToM-NAS Research Team"

from . import core
from . import agents
from . import world
from . import evolution
from . import evaluation

__all__ = [
    'core',
    'agents',
    'world',
    'evolution',
    'evaluation',
    '__version__',
]
