"""
ToM-NAS REST API Module

Provides HTTP endpoints for:
- Model inference
- Experiment management
- Health checks
- Metrics exposure
"""

from .main import app
from .routes import router

__all__ = ['app', 'router']
