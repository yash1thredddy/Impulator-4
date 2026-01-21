"""
Application metrics for monitoring and observability.

Provides thread-safe metrics collection for:
- Job counters (created, completed, failed)
- API call statistics
- Cache hit rates
- Latency tracking

Usage:
    from backend.core.metrics import metrics

    # Increment counters
    metrics.increment('jobs_created')
    metrics.increment('api_calls_total')

    # Record latency
    metrics.record_latency('chembl', 150.5)

    # Get all metrics
    data = metrics.to_dict()
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from threading import Lock


@dataclass
class Metrics:
    """Thread-safe application metrics."""

    _lock: Lock = field(default_factory=Lock, repr=False)

    # Counters
    jobs_created: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_cancelled: int = 0
    api_calls_total: int = 0
    api_calls_failed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limit_exceeded: int = 0

    # Gauges (current values)
    active_jobs: int = 0
    pending_jobs: int = 0

    # Histograms (simplified - store recent samples)
    api_latencies: Dict[str, List[float]] = field(default_factory=dict)
    _max_latency_samples: int = field(default=1000, repr=False)

    # Start time for uptime calculation
    _start_time: float = field(default_factory=time.time, repr=False)

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric.

        Args:
            metric: Name of the metric to increment
            value: Amount to increment by (default 1)
        """
        with self._lock:
            current = getattr(self, metric, 0)
            setattr(self, metric, current + value)

    def decrement(self, metric: str, value: int = 1) -> None:
        """Decrement a gauge metric.

        Args:
            metric: Name of the metric to decrement
            value: Amount to decrement by (default 1)
        """
        with self._lock:
            current = getattr(self, metric, 0)
            setattr(self, metric, max(0, current - value))

    def set_gauge(self, metric: str, value: int) -> None:
        """Set a gauge to a specific value.

        Args:
            metric: Name of the gauge metric
            value: Value to set
        """
        with self._lock:
            setattr(self, metric, value)

    def record_latency(self, api: str, latency_ms: float) -> None:
        """Record an API call latency.

        Args:
            api: Name of the API (e.g., 'chembl', 'pdb')
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            if api not in self.api_latencies:
                self.api_latencies[api] = []
            self.api_latencies[api].append(latency_ms)
            # Keep only recent samples to prevent memory growth
            if len(self.api_latencies[api]) > self._max_latency_samples:
                self.api_latencies[api] = self.api_latencies[api][-self._max_latency_samples:]

    def get_latency_stats(self, api: str) -> Dict[str, Optional[float]]:
        """Get latency statistics for an API.

        Args:
            api: Name of the API

        Returns:
            Dict with min, max, avg, p50, p95, p99 latencies
        """
        with self._lock:
            samples = self.api_latencies.get(api, [])

            if not samples:
                return {
                    'count': 0,
                    'min': None,
                    'max': None,
                    'avg': None,
                    'p50': None,
                    'p95': None,
                    'p99': None,
                }

            sorted_samples = sorted(samples)
            count = len(sorted_samples)

            return {
                'count': count,
                'min': sorted_samples[0],
                'max': sorted_samples[-1],
                'avg': sum(sorted_samples) / count,
                'p50': sorted_samples[int(count * 0.5)],
                'p95': sorted_samples[int(count * 0.95)] if count >= 20 else sorted_samples[-1],
                'p99': sorted_samples[int(count * 0.99)] if count >= 100 else sorted_samples[-1],
            }

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self._start_time

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def job_success_rate(self) -> float:
        """Calculate job success rate (0.0 to 1.0)."""
        total = self.jobs_completed + self.jobs_failed
        if total == 0:
            return 1.0  # No jobs = 100% success (vacuously true)
        return self.jobs_completed / total

    def to_dict(self) -> dict:
        """Export all metrics as a dictionary.

        Returns:
            Dict containing all metric values
        """
        with self._lock:
            return {
                # Counters
                'jobs_created': self.jobs_created,
                'jobs_completed': self.jobs_completed,
                'jobs_failed': self.jobs_failed,
                'jobs_cancelled': self.jobs_cancelled,
                'api_calls_total': self.api_calls_total,
                'api_calls_failed': self.api_calls_failed,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'rate_limit_exceeded': self.rate_limit_exceeded,

                # Gauges
                'active_jobs': self.active_jobs,
                'pending_jobs': self.pending_jobs,

                # Computed metrics
                'cache_hit_rate': self.cache_hit_rate,
                'job_success_rate': self.job_success_rate,
                'uptime_seconds': self.uptime_seconds,

                # Latency stats by API
                'latencies': {
                    api: self.get_latency_stats(api)
                    for api in self.api_latencies.keys()
                },
            }

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.jobs_created = 0
            self.jobs_completed = 0
            self.jobs_failed = 0
            self.jobs_cancelled = 0
            self.api_calls_total = 0
            self.api_calls_failed = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.rate_limit_exceeded = 0
            self.active_jobs = 0
            self.pending_jobs = 0
            self.api_latencies.clear()
            self._start_time = time.time()


# Global metrics instance
metrics = Metrics()
