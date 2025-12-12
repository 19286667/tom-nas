"""
Infrastructure Module - Operational Excellence & Reliability

Implements Well-Architected Framework patterns:
- Circuit breakers for fault isolation
- Health checks (liveness, readiness, startup)
- Graceful degradation
- Structured observability
- Resource management
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .health import HealthCheck, HealthStatus, ComponentHealth
from .observability import MetricsCollector, TraceContext, structured_log
from .resilience import retry_with_backoff, timeout, fallback
from .resource_manager import ResourcePool, ResourceLimits

__all__ = [
    'CircuitBreaker',
    'CircuitState',
    'HealthCheck',
    'HealthStatus',
    'ComponentHealth',
    'MetricsCollector',
    'TraceContext',
    'structured_log',
    'retry_with_backoff',
    'timeout',
    'fallback',
    'ResourcePool',
    'ResourceLimits',
]
