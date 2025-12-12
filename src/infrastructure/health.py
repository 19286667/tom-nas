"""
Health Check System

Implements three types of health checks per Kubernetes/Cloud Run patterns:
- Liveness: Is the process alive? (restart if not)
- Readiness: Can it accept traffic? (remove from LB if not)
- Startup: Has it finished initializing? (give it time)

Each component registers its health check function.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.config import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Partially functional
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Aggregate health of the system."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.components
            ]
        }


class HealthCheck:
    """
    Central health check registry and executor.

    Usage:
        health = HealthCheck()

        # Register components
        health.register("database", check_database)
        health.register("cache", check_cache, critical=False)

        # Check health
        status = health.check_all()
        if not status.is_ready:
            # Handle unhealthy state
    """

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
        self._checks: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last_check: Optional[SystemHealth] = None
        self._executor = ThreadPoolExecutor(max_workers=10)

        # Register self-check
        self.register("health_system", lambda: True, critical=False)

    def register(
        self,
        name: str,
        check_fn: Callable[[], bool],
        critical: bool = True,
        description: str = "",
    ):
        """
        Register a health check.

        Args:
            name: Unique identifier for the component
            check_fn: Function that returns True if healthy
            critical: If True, failure makes system unhealthy; else degraded
            description: Human-readable description
        """
        with self._lock:
            self._checks[name] = {
                "fn": check_fn,
                "critical": critical,
                "description": description,
            }
        logger.debug(f"Registered health check: {name}")

    def unregister(self, name: str):
        """Remove a health check."""
        with self._lock:
            self._checks.pop(name, None)

    def _run_check(self, name: str, check_info: dict) -> ComponentHealth:
        """Run a single health check with timeout."""
        start = time.time()
        try:
            future = self._executor.submit(check_info["fn"])
            result = future.result(timeout=self.timeout)
            latency = (time.time() - start) * 1000

            if result:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    latency_ms=latency,
                )
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Check returned False",
                    latency_ms=latency,
                )
        except FuturesTimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {self.timeout}s",
                latency_ms=self.timeout * 1000,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
            )

    def check_all(self) -> SystemHealth:
        """
        Run all health checks and return aggregate status.
        """
        with self._lock:
            checks = dict(self._checks)

        results = []
        for name, info in checks.items():
            result = self._run_check(name, info)
            results.append((result, info["critical"]))

        # Determine aggregate status
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY and critical
            for r, critical in results
        )
        any_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r, _ in results
        )

        if critical_unhealthy:
            status = HealthStatus.UNHEALTHY
        elif any_unhealthy:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        health = SystemHealth(
            status=status,
            components=[r for r, _ in results],
        )

        self._last_check = health
        return health

    def check_liveness(self) -> bool:
        """
        Liveness check: Is the process fundamentally alive?

        This should be fast and check only critical components.
        """
        # Just verify the health system itself works
        return True

    def check_readiness(self) -> SystemHealth:
        """
        Readiness check: Can the service accept traffic?

        Returns full health status.
        """
        return self.check_all()

    def check_startup(self) -> bool:
        """
        Startup check: Has initialization completed?

        Check if all components are at least degraded (not failed).
        """
        health = self.check_all()
        return health.is_ready

    def get_last_check(self) -> Optional[SystemHealth]:
        """Get cached result of last health check."""
        return self._last_check


# Global health check instance
_health_check: Optional[HealthCheck] = None


def get_health_check() -> HealthCheck:
    """Get global health check instance."""
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check


def register_health_check(
    name: str,
    check_fn: Callable[[], bool],
    critical: bool = True,
):
    """Convenience function to register a health check."""
    get_health_check().register(name, check_fn, critical)


# Standard health checks for common components
def check_torch_available() -> bool:
    """Check if PyTorch is available and functional."""
    try:
        import torch
        # Quick tensor operation to verify CUDA/CPU works
        x = torch.ones(1)
        return float(x.item()) == 1.0
    except Exception:
        return False


def check_memory_available(threshold_mb: float = 100) -> bool:
    """Check if sufficient memory is available."""
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 * 1024)
        return available > threshold_mb
    except ImportError:
        # psutil not available, assume OK
        return True


def check_disk_space(path: str = "/", threshold_mb: float = 500) -> bool:
    """Check if sufficient disk space is available."""
    try:
        import shutil
        free = shutil.disk_usage(path).free / (1024 * 1024)
        return free > threshold_mb
    except Exception:
        return True
