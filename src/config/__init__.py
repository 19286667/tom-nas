"""
ToM-NAS Configuration Module

Centralized configuration for all constants, dimensions, and settings.
This module provides a single source of truth for all configurable values.
"""

from .constants import *
from .settings import Settings, get_settings
from .logging_config import setup_logging, get_logger

__all__ = [
    # Constants
    'SOUL_MAP_DIMS',
    'INPUT_DIMS',
    'OUTPUT_DIMS',
    'DEFAULT_HIDDEN_DIMS',
    'MAX_BELIEF_ORDER',
    'CONFIDENCE_DECAY',
    'DEFAULT_NUM_AGENTS',
    'DEFAULT_POPULATION_SIZE',
    'DEFAULT_GENERATIONS',
    # Settings
    'Settings',
    'get_settings',
    # Logging
    'setup_logging',
    'get_logger',
]
