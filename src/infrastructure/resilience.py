"""
Resilience Patterns

Implements:
- Retry with exponential backoff
- Timeout decorator
- Fallback pattern
- Bulkhead (resource isolation)

These patterns are essential for building reliable distributed systems.
"""

import time
import random
import threading
from functools import wraps
from typing import Callable, TypeVar, Optional, Any, Type, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import signal

from src.config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RetryExhaustedError(Exception):
    """Raised when all retries are exhausted."""
    pass


class TimeoutExceededError(Exception):
    """Raised when operation times out."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Add randomness to prevent thundering herd
        retryable_exceptions: Exception types that trigger retry

    Usage:
        @retry_with_backoff(max_retries=3)
        def flaky_operation():
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{fn.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        raise RetryExhaustedError(
                            f"All {max_retries + 1} attempts failed"
                        ) from e

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    if jitter:
                        delay *= (0.5 + random.random())

                    logger.warning(
                        f"{fn.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise RetryExhaustedError("Retry logic error") from last_exception

        return wrapper
    return decorator


def timeout(seconds: float, fallback_value: Any = None):
    """
    Decorator to add timeout to a function.

    Args:
        seconds: Maximum execution time
        fallback_value: Value to return on timeout (if not raising)

    Usage:
        @timeout(5.0)
        def slow_operation():
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            # Use threading for cross-platform compatibility
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = fn(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                logger.warning(f"{fn.__name__} timed out after {seconds}s")
                if fallback_value is not None:
                    return fallback_value
                raise TimeoutExceededError(
                    f"{fn.__name__} exceeded timeout of {seconds}s"
                )

            if exception[0] is not None:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


def fallback(fallback_fn: Callable[..., T]):
    """
    Decorator to provide fallback on exception.

    Args:
        fallback_fn: Function to call on failure

    Usage:
        def get_cached():
            return cached_value

        @fallback(get_cached)
        def get_from_api():
            ...
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{fn.__name__} failed: {e}. Using fallback."
                )
                return fallback_fn(*args, **kwargs)
        return wrapper
    return decorator


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent executions to prevent resource exhaustion.
    Named after ship bulkheads that contain flooding.

    Usage:
        bulkhead = Bulkhead("api_calls", max_concurrent=10)

        with bulkhead:
            make_api_call()
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_waiting: int = 100,
        timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.timeout = timeout_seconds

        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._waiting = 0
        self._lock = threading.Lock()

        # Stats
        self.total_calls = 0
        self.rejected_calls = 0
        self.active_calls = 0

    def __enter__(self):
        with self._lock:
            if self._waiting >= self.max_waiting:
                self.rejected_calls += 1
                raise RuntimeError(
                    f"Bulkhead '{self.name}' queue full ({self.max_waiting})"
                )
            self._waiting += 1

        acquired = self._semaphore.acquire(timeout=self.timeout)

        with self._lock:
            self._waiting -= 1

        if not acquired:
            self.rejected_calls += 1
            raise TimeoutExceededError(
                f"Bulkhead '{self.name}' acquisition timed out"
            )

        with self._lock:
            self.total_calls += 1
            self.active_calls += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()
        with self._lock:
            self.active_calls -= 1
        return False

    def get_stats(self) -> dict:
        """Get bulkhead statistics."""
        with self._lock:
            return {
                "name": self.name,
                "max_concurrent": self.max_concurrent,
                "active_calls": self.active_calls,
                "waiting": self._waiting,
                "total_calls": self.total_calls,
                "rejected_calls": self.rejected_calls,
                "utilization": self.active_calls / self.max_concurrent,
            }


class GracefulDegradation:
    """
    Manages graceful degradation of system capabilities.

    When resources are constrained, disable non-essential features
    rather than failing entirely.

    Usage:
        degradation = GracefulDegradation()
        degradation.set_level(DegradationLevel.PARTIAL)

        if degradation.is_feature_enabled("detailed_logging"):
            log_details()
    """

    class Level:
        FULL = "full"           # All features enabled
        PARTIAL = "partial"     # Non-essential features disabled
        MINIMAL = "minimal"     # Only critical features
        EMERGENCY = "emergency" # Bare minimum for survival

    # Feature availability at each level
    FEATURE_LEVELS = {
        "detailed_logging": [Level.FULL],
        "metrics_collection": [Level.FULL, Level.PARTIAL],
        "caching": [Level.FULL, Level.PARTIAL, Level.MINIMAL],
        "core_inference": [Level.FULL, Level.PARTIAL, Level.MINIMAL, Level.EMERGENCY],
        "health_checks": [Level.FULL, Level.PARTIAL, Level.MINIMAL, Level.EMERGENCY],
    }

    def __init__(self):
        self._level = self.Level.FULL
        self._lock = threading.Lock()

    def set_level(self, level: str):
        """Set degradation level."""
        with self._lock:
            old_level = self._level
            self._level = level
            logger.warning(f"Degradation level changed: {old_level} -> {level}")

    def get_level(self) -> str:
        """Get current degradation level."""
        with self._lock:
            return self._level

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at current level."""
        with self._lock:
            levels = self.FEATURE_LEVELS.get(feature, [self.Level.FULL])
            return self._level in levels

    def auto_degrade(self, health_status: str):
        """Automatically adjust degradation based on health."""
        if health_status == "healthy":
            self.set_level(self.Level.FULL)
        elif health_status == "degraded":
            self.set_level(self.Level.PARTIAL)
        elif health_status == "unhealthy":
            self.set_level(self.Level.MINIMAL)
        else:
            self.set_level(self.Level.EMERGENCY)


# Global instances
_degradation = GracefulDegradation()


def get_degradation() -> GracefulDegradation:
    """Get global degradation manager."""
    return _degradation
