"""
Circuit Breaker Pattern

Prevents cascade failures by failing fast when a dependency is unhealthy.
Three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing fast, requests rejected immediately
- HALF_OPEN: Testing if dependency recovered

This is essential for reliability in distributed systems.
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, TypeVar, Generic
from functools import wraps

from src.config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes before closing (from half-open)
    timeout_seconds: float = 30.0       # Time before half-open test
    half_open_max_calls: int = 3        # Max calls in half-open state


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker for fault isolation.

    Usage:
        breaker = CircuitBreaker("external_service")

        @breaker
        def call_external_service():
            ...

        # Or explicit:
        result = breaker.call(lambda: risky_operation())
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        fallback: Callable[[], T] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_fn = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._lock = threading.RLock()
        self.stats = CircuitStats()

        logger.info(f"CircuitBreaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.timeout_seconds

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self.stats.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0

        logger.info(f"CircuitBreaker '{self.name}': {old_state.value} -> {new_state.value}")

    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, error: Exception):
        """Record a failed call."""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            logger.warning(f"CircuitBreaker '{self.name}' recorded failure: {error}")

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        state = self.state  # This checks for timeout transition

        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def call(self, fn: Callable[[], T]) -> T:
        """
        Execute a function through the circuit breaker.

        Raises CircuitBreakerError if circuit is open (unless fallback provided).
        """
        if not self._can_execute():
            self.stats.rejected_calls += 1
            if self.fallback_fn:
                logger.debug(f"CircuitBreaker '{self.name}': using fallback")
                return self.fallback_fn()
            raise CircuitBreakerError(
                f"Circuit '{self.name}' is {self.state.value}"
            )

        try:
            result = fn()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            if self.fallback_fn:
                return self.fallback_fn()
            raise

    def __call__(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator form."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return self.call(lambda: fn(*args, **kwargs))
        return wrapper

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"CircuitBreaker '{self.name}' manually reset")

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "rejected_calls": self.stats.rejected_calls,
            "failure_rate": (
                self.stats.failed_calls / max(1, self.stats.total_calls)
            ),
        }


# Pre-configured breakers for common use cases
class Breakers:
    """Registry of circuit breakers."""
    _breakers: dict = {}

    @classmethod
    def get(cls, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(name, **kwargs)
        return cls._breakers[name]

    @classmethod
    def get_all_stats(cls) -> dict:
        """Get stats for all breakers."""
        return {name: b.get_stats() for name, b in cls._breakers.items()}
