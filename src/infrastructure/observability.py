"""
Observability Infrastructure

Implements the three pillars of observability:
1. Metrics - Quantitative measurements (Prometheus-compatible)
2. Logs - Structured event records (Cloud Logging compatible)
3. Traces - Distributed request tracking (OpenTelemetry compatible)

This is not about pretty dashboards - it's about understanding
system behavior when things go wrong at 3am.
"""

import time
import json
import threading
import uuid
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from functools import wraps
from enum import Enum

from src.config import get_logger

logger = get_logger(__name__)


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class StructuredLogEntry:
    """A structured log entry compatible with Cloud Logging."""
    timestamp: str
    severity: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    # Cloud Logging specific
    logging_googleapis_com_trace: Optional[str] = None

    def to_dict(self) -> dict:
        entry = {
            "timestamp": self.timestamp,
            "severity": self.severity,
            "message": self.message,
        }
        if self.trace_id:
            entry["logging.googleapis.com/trace"] = self.trace_id
        if self.span_id:
            entry["logging.googleapis.com/spanId"] = self.span_id
        if self.labels:
            entry["labels"] = self.labels
        return entry

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def structured_log(
    level: LogLevel,
    message: str,
    trace_context: 'TraceContext' = None,
    **labels,
):
    """
    Emit a structured log entry.

    This produces JSON logs that Cloud Logging can parse and index.

    Usage:
        structured_log(
            LogLevel.INFO,
            "Processing request",
            trace_context=ctx,
            user_id="123",
            operation="inference"
        )
    """
    entry = StructuredLogEntry(
        timestamp=datetime.utcnow().isoformat() + "Z",
        severity=level.value,
        message=message,
        trace_id=trace_context.trace_id if trace_context else None,
        span_id=trace_context.span_id if trace_context else None,
        labels=labels,
    )

    # In production, emit JSON to stdout for Cloud Logging
    # In development, use standard logger
    if labels.get("_structured_output"):
        print(entry.to_json(), file=sys.stdout)
    else:
        log_fn = {
            LogLevel.DEBUG: logger.debug,
            LogLevel.INFO: logger.info,
            LogLevel.WARNING: logger.warning,
            LogLevel.ERROR: logger.error,
            LogLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)
        log_fn(f"{message} {labels}" if labels else message)


# =============================================================================
# DISTRIBUTED TRACING
# =============================================================================

@dataclass
class TraceContext:
    """
    Context for distributed tracing.

    Propagates trace/span IDs across service boundaries.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def create_child_span(self, name: str) -> 'TraceContext':
        """Create a child span within this trace."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            attributes={"span.name": name},
        )

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000


# Thread-local storage for trace context
_trace_context = threading.local()


def get_current_trace() -> Optional[TraceContext]:
    """Get current trace context."""
    return getattr(_trace_context, 'context', None)


def set_current_trace(ctx: TraceContext):
    """Set current trace context."""
    _trace_context.context = ctx


@contextmanager
def trace_span(name: str, **attributes):
    """
    Context manager for creating a traced span.

    Usage:
        with trace_span("process_request", user_id="123"):
            do_work()
    """
    parent = get_current_trace()

    if parent:
        ctx = parent.create_child_span(name)
    else:
        ctx = TraceContext(attributes={"span.name": name})

    for k, v in attributes.items():
        ctx.set_attribute(k, v)

    set_current_trace(ctx)

    try:
        yield ctx
    except Exception as e:
        ctx.set_attribute("error", True)
        ctx.set_attribute("error.message", str(e))
        raise
    finally:
        ctx.set_attribute("duration_ms", ctx.elapsed_ms())
        structured_log(
            LogLevel.DEBUG,
            f"Span completed: {name}",
            trace_context=ctx,
            duration_ms=ctx.elapsed_ms(),
        )
        set_current_trace(parent)


def traced(name: str = None):
    """
    Decorator to trace a function.

    Usage:
        @traced("expensive_operation")
        def expensive_operation():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            with trace_span(span_name):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# METRICS
# =============================================================================

class MetricType(Enum):
    COUNTER = "counter"         # Monotonically increasing
    GAUGE = "gauge"             # Point-in-time value
    HISTOGRAM = "histogram"     # Distribution of values


@dataclass
class Metric:
    """A single metric."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Prometheus-compatible metrics collector.

    Collects and exposes metrics in Prometheus text format.
    """

    def __init__(self, prefix: str = "tom_nas"):
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        # Standard buckets for histograms
        self._histogram_buckets = [
            0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
            0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')
        ]

    def _metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate unique key for metric + labels."""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if labels else name

    def counter(self, name: str, value: float = 1, **labels):
        """Increment a counter."""
        full_name = f"{self.prefix}_{name}"
        key = self._metric_key(full_name, labels)

        with self._lock:
            if key in self._metrics:
                self._metrics[key].value += value
            else:
                self._metrics[key] = Metric(
                    name=full_name,
                    type=MetricType.COUNTER,
                    value=value,
                    labels=labels,
                )

    def gauge(self, name: str, value: float, **labels):
        """Set a gauge value."""
        full_name = f"{self.prefix}_{name}"
        key = self._metric_key(full_name, labels)

        with self._lock:
            self._metrics[key] = Metric(
                name=full_name,
                type=MetricType.GAUGE,
                value=value,
                labels=labels,
            )

    def histogram(self, name: str, value: float, **labels):
        """Record a histogram observation."""
        full_name = f"{self.prefix}_{name}"
        key = self._metric_key(full_name, labels)

        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    @contextmanager
    def timer(self, name: str, **labels):
        """Context manager to time an operation."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.histogram(f"{name}_seconds", duration, **labels)

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        with self._lock:
            # Regular metrics
            for metric in self._metrics.values():
                labels_str = ",".join(
                    f'{k}="{v}"' for k, v in metric.labels.items()
                )
                if labels_str:
                    lines.append(f"{metric.name}{{{labels_str}}} {metric.value}")
                else:
                    lines.append(f"{metric.name} {metric.value}")

            # Histograms
            for key, values in self._histograms.items():
                if not values:
                    continue

                # Extract name and labels from key
                name = key.split("{")[0]

                # Compute bucket counts
                for bucket in self._histogram_buckets:
                    count = sum(1 for v in values if v <= bucket)
                    bucket_str = "+Inf" if bucket == float('inf') else str(bucket)
                    lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')

                lines.append(f"{name}_sum {sum(values)}")
                lines.append(f"{name}_count {len(values)}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        with self._lock:
            return {
                "metrics": {
                    k: {"value": m.value, "type": m.type.value}
                    for k, m in self._metrics.items()
                },
                "histogram_counts": {
                    k: len(v) for k, v in self._histograms.items()
                }
            }


# Global metrics collector
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# Convenience functions
def increment_counter(name: str, value: float = 1, **labels):
    """Increment a counter metric."""
    get_metrics().counter(name, value, **labels)


def set_gauge(name: str, value: float, **labels):
    """Set a gauge metric."""
    get_metrics().gauge(name, value, **labels)


def observe_histogram(name: str, value: float, **labels):
    """Record a histogram observation."""
    get_metrics().histogram(name, value, **labels)
