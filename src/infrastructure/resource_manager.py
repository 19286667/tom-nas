"""
Resource Management

Implements resource pooling and limits to prevent exhaustion:
- Connection pooling
- Memory management
- Compute budgeting
- Graceful resource release

Essential for cost optimization and reliability.
"""

import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, Generic, TypeVar, Callable, Optional, Any, List
from queue import Queue, Empty, Full
from contextlib import contextmanager

from src.config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class ResourceLimits:
    """Configurable resource limits."""
    max_memory_mb: float = 1024.0
    max_cpu_percent: float = 80.0
    max_concurrent_requests: int = 100
    max_queue_size: int = 1000
    request_timeout_seconds: float = 30.0

    # Evolution-specific limits
    max_population_size: int = 100
    max_simulation_depth: int = 5
    max_generations: int = 1000


class ResourcePool(Generic[T]):
    """
    Generic resource pool with lifecycle management.

    Reuses expensive resources (connections, models) rather than
    creating/destroying them repeatedly.

    Usage:
        pool = ResourcePool(
            factory=lambda: create_connection(),
            max_size=10,
            cleanup=lambda conn: conn.close()
        )

        with pool.acquire() as conn:
            conn.execute(...)
    """

    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 0,
        cleanup: Callable[[T], None] = None,
        health_check: Callable[[T], bool] = None,
        max_age_seconds: float = 3600.0,
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.cleanup_fn = cleanup
        self.health_check_fn = health_check
        self.max_age = max_age_seconds

        self._pool: Queue = Queue(maxsize=max_size)
        self._created: Dict[int, float] = {}  # id -> creation time
        self._in_use: int = 0
        self._lock = threading.Lock()

        # Stats
        self.total_created = 0
        self.total_destroyed = 0
        self.acquisitions = 0
        self.releases = 0
        self.timeouts = 0

        # Pre-warm pool
        self._warm_pool()

    def _warm_pool(self):
        """Create minimum number of resources."""
        for _ in range(self.min_size):
            try:
                resource = self._create_resource()
                self._pool.put_nowait(resource)
            except Exception as e:
                logger.warning(f"Failed to warm pool: {e}")
                break

    def _create_resource(self) -> T:
        """Create a new resource."""
        resource = self.factory()
        self._created[id(resource)] = time.time()
        self.total_created += 1
        return resource

    def _destroy_resource(self, resource: T):
        """Destroy a resource."""
        try:
            if self.cleanup_fn:
                self.cleanup_fn(resource)
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}")
        finally:
            self._created.pop(id(resource), None)
            self.total_destroyed += 1

    def _is_healthy(self, resource: T) -> bool:
        """Check if resource is still healthy."""
        # Check age
        created_at = self._created.get(id(resource), 0)
        if time.time() - created_at > self.max_age:
            return False

        # Custom health check
        if self.health_check_fn:
            try:
                return self.health_check_fn(resource)
            except Exception:
                return False

        return True

    @contextmanager
    def acquire(self, timeout: float = 30.0):
        """
        Acquire a resource from the pool.

        Context manager that automatically returns resource to pool.
        """
        resource = None
        acquired_from_pool = False

        try:
            # Try to get from pool
            try:
                resource = self._pool.get(timeout=timeout)
                acquired_from_pool = True

                # Health check
                if not self._is_healthy(resource):
                    self._destroy_resource(resource)
                    resource = self._create_resource()
                    acquired_from_pool = False

            except Empty:
                # Pool empty, create new if under limit
                with self._lock:
                    current_total = self._pool.qsize() + self._in_use
                    if current_total < self.max_size:
                        resource = self._create_resource()
                    else:
                        self.timeouts += 1
                        raise TimeoutError(
                            f"Resource pool exhausted (max={self.max_size})"
                        )

            with self._lock:
                self._in_use += 1
                self.acquisitions += 1

            yield resource

        finally:
            if resource is not None:
                with self._lock:
                    self._in_use -= 1
                    self.releases += 1

                # Return to pool if healthy
                if self._is_healthy(resource):
                    try:
                        self._pool.put_nowait(resource)
                    except Full:
                        self._destroy_resource(resource)
                else:
                    self._destroy_resource(resource)

    def drain(self):
        """Destroy all pooled resources."""
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                self._destroy_resource(resource)
            except Empty:
                break

    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "pool_size": self._pool.qsize(),
            "in_use": self._in_use,
            "max_size": self.max_size,
            "total_created": self.total_created,
            "total_destroyed": self.total_destroyed,
            "acquisitions": self.acquisitions,
            "releases": self.releases,
            "timeouts": self.timeouts,
            "utilization": self._in_use / max(1, self.max_size),
        }


class MemoryManager:
    """
    Tracks and limits memory usage.

    Essential for preventing OOM in long-running training jobs.
    """

    def __init__(self, max_memory_mb: float = 1024.0):
        self.max_memory_mb = max_memory_mb
        self._allocations: Dict[str, float] = {}
        self._lock = threading.Lock()

    def allocate(self, name: str, size_mb: float) -> bool:
        """
        Request memory allocation.

        Returns True if allocation succeeds, False if would exceed limit.
        """
        with self._lock:
            current_usage = sum(self._allocations.values())
            if current_usage + size_mb > self.max_memory_mb:
                logger.warning(
                    f"Memory allocation denied: {name} requested {size_mb}MB, "
                    f"current={current_usage}MB, max={self.max_memory_mb}MB"
                )
                return False

            self._allocations[name] = size_mb
            return True

    def release(self, name: str):
        """Release a memory allocation."""
        with self._lock:
            self._allocations.pop(name, None)

    def get_usage(self) -> dict:
        """Get current memory usage."""
        with self._lock:
            return {
                "allocations": dict(self._allocations),
                "total_mb": sum(self._allocations.values()),
                "max_mb": self.max_memory_mb,
                "available_mb": self.max_memory_mb - sum(self._allocations.values()),
            }

    @contextmanager
    def track(self, name: str, size_mb: float):
        """Context manager for memory tracking."""
        if not self.allocate(name, size_mb):
            raise MemoryError(f"Cannot allocate {size_mb}MB for {name}")
        try:
            yield
        finally:
            self.release(name)


class ComputeBudget:
    """
    Tracks compute usage for cost control.

    Prevents runaway costs from infinite loops or inefficient algorithms.
    """

    def __init__(
        self,
        max_cpu_seconds: float = 3600.0,
        max_gpu_seconds: float = 0.0,
        alert_threshold: float = 0.8,
    ):
        self.max_cpu_seconds = max_cpu_seconds
        self.max_gpu_seconds = max_gpu_seconds
        self.alert_threshold = alert_threshold

        self._cpu_used = 0.0
        self._gpu_used = 0.0
        self._lock = threading.Lock()
        self._alert_sent = False

    def consume_cpu(self, seconds: float):
        """Record CPU time consumption."""
        with self._lock:
            self._cpu_used += seconds
            self._check_budget()

    def consume_gpu(self, seconds: float):
        """Record GPU time consumption."""
        with self._lock:
            self._gpu_used += seconds
            self._check_budget()

    def _check_budget(self):
        """Check if budget is being exceeded."""
        cpu_ratio = self._cpu_used / max(1, self.max_cpu_seconds)
        gpu_ratio = self._gpu_used / max(1, self.max_gpu_seconds) if self.max_gpu_seconds else 0

        if not self._alert_sent and (cpu_ratio > self.alert_threshold or gpu_ratio > self.alert_threshold):
            logger.warning(
                f"Compute budget alert: CPU={cpu_ratio:.1%}, GPU={gpu_ratio:.1%}"
            )
            self._alert_sent = True

        if cpu_ratio >= 1.0:
            raise RuntimeError(
                f"CPU budget exhausted: {self._cpu_used}s / {self.max_cpu_seconds}s"
            )
        if self.max_gpu_seconds and gpu_ratio >= 1.0:
            raise RuntimeError(
                f"GPU budget exhausted: {self._gpu_used}s / {self.max_gpu_seconds}s"
            )

    @contextmanager
    def track_cpu(self):
        """Context manager to track CPU time."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.consume_cpu(elapsed)

    def get_usage(self) -> dict:
        """Get current compute usage."""
        with self._lock:
            return {
                "cpu_seconds_used": self._cpu_used,
                "cpu_seconds_max": self.max_cpu_seconds,
                "cpu_percent": self._cpu_used / max(1, self.max_cpu_seconds) * 100,
                "gpu_seconds_used": self._gpu_used,
                "gpu_seconds_max": self.max_gpu_seconds,
            }

    def reset(self):
        """Reset budget (for new billing period)."""
        with self._lock:
            self._cpu_used = 0.0
            self._gpu_used = 0.0
            self._alert_sent = False


# Global resource managers
_memory_manager: Optional[MemoryManager] = None
_compute_budget: Optional[ComputeBudget] = None
_resource_limits: Optional[ResourceLimits] = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_compute_budget() -> ComputeBudget:
    global _compute_budget
    if _compute_budget is None:
        _compute_budget = ComputeBudget()
    return _compute_budget


def get_resource_limits() -> ResourceLimits:
    global _resource_limits
    if _resource_limits is None:
        _resource_limits = ResourceLimits()
    return _resource_limits
