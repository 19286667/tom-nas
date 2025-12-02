"""
Neural Architecture Search module for Theory of Mind.

This module provides efficient NAS techniques for discovering
architectures that excel at Theory of Mind reasoning.

Key components:
- ZeroCostProxies: Fast architecture evaluation without training
- EfficientNAS: Combined zero-cost + evolution pipeline
"""

from .zero_cost import ZeroCostProxies, rank_architectures
from .efficient_nas import EfficientNAS

__all__ = [
    'ZeroCostProxies',
    'rank_architectures',
    'EfficientNAS',
]
