# Copyright 2026 Tobias Østen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pipeline Monitor — tracks per-stage metrics, timing, and health for the
14-stage cognitive pipeline.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class StageSpan:
    """
    Immutable record of a single pipeline stage execution.

    Attributes:
        stage_name: Name of the pipeline stage.
        request_id: Request identifier.
        start_time: Start timestamp.
        end_time: End timestamp.
        duration_ms: Duration in milliseconds.
        success: Whether the stage completed successfully.
        error: Error message if failed.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    stage_name: str
    request_id: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: str


@dataclass
class StageMetrics:
    """
    Mutable aggregated metrics for a pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        total_executions: Total number of executions.
        total_successes: Successful executions.
        total_failures: Failed executions.
        total_duration_ms: Cumulative duration in milliseconds.
        min_duration_ms: Minimum duration.
        max_duration_ms: Maximum duration.
        last_executed: Timestamp of last execution.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    stage_name: str
    total_executions: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    last_executed: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Average duration in milliseconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.total_executions == 0:
            return 0.0
        return self.total_duration_ms / self.total_executions

    @property
    def success_rate(self) -> float:
        """Success rate in [0.0, 1.0].

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.total_executions == 0:
            return 0.0
        return self.total_successes / self.total_executions


class PipelineMonitor:
    """
    Monitors pipeline execution with per-stage metrics and span tracking.

    Complexity:
        start_span: T(n) ∈ O(1), S(n) ∈ O(1).
        end_span: T(n) ∈ O(1), S(n) ∈ O(1).
        get_metrics: T(n) ∈ O(1), S(n) ∈ O(1).
        all_metrics: T(n) ∈ O(s), S(n) ∈ O(s).
    """

    __slots__ = ("_metrics", "_spans", "_active_spans", "_total_requests")

    def __init__(self) -> None:
        """Initialize the monitor.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._metrics: Dict[str, StageMetrics] = {}
        self._spans: List[StageSpan] = []
        self._active_spans: Dict[str, tuple[str, float]] = {}
        self._total_requests: int = 0

    def start_span(self, stage_name: str, request_id: str) -> str:
        """Start tracking a pipeline stage execution.

        Args:
            stage_name: Name of the stage.
            request_id: Request identifier.

        Returns:
            Span key for ending the span.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        span_key = f"{request_id}:{stage_name}"
        self._active_spans[span_key] = (stage_name, time.time())
        return span_key

    def end_span(self, span_key: str, success: bool = True, error: str = "") -> Optional[StageSpan]:
        """End tracking a pipeline stage execution.

        Args:
            span_key: Span key from start_span.
            success: Whether the stage succeeded.
            error: Error message if failed.

        Returns:
            StageSpan record, or None if span_key not found.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if span_key not in self._active_spans:
            return None

        stage_name, start_time = self._active_spans.pop(span_key)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000.0

        span = StageSpan(
            stage_name=stage_name,
            request_id=span_key.split(":")[0],
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )
        self._spans.append(span)

        # Update aggregated metrics
        if stage_name not in self._metrics:
            self._metrics[stage_name] = StageMetrics(stage_name=stage_name)

        metrics = self._metrics[stage_name]
        metrics.total_executions += 1
        metrics.total_duration_ms += duration_ms
        metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
        metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)
        metrics.last_executed = end_time
        if success:
            metrics.total_successes += 1
        else:
            metrics.total_failures += 1

        return span

    def get_metrics(self, stage_name: str) -> Optional[StageMetrics]:
        """Get aggregated metrics for a stage.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._metrics.get(stage_name)

    def all_metrics(self) -> Dict[str, StageMetrics]:
        """Get all stage metrics.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s).
        """
        return dict(self._metrics)

    def get_spans(self, limit: int = 100) -> List[StageSpan]:
        """Get recent spans.

        Args:
            limit: Maximum number of spans to return.

        Returns:
            List of recent StageSpan records.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._spans[-limit:])

    def increment_request_count(self) -> None:
        """Increment the total request counter.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._total_requests += 1

    def total_requests(self) -> int:
        """Return total number of requests.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._total_requests

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics as a serializable dict.

        Returns:
            Dictionary with all metrics, suitable for JSON export.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s).
        """
        return {
            "total_requests": self._total_requests,
            "stages": {
                name: {
                    "total_executions": m.total_executions,
                    "total_successes": m.total_successes,
                    "total_failures": m.total_failures,
                    "avg_duration_ms": round(m.avg_duration_ms, 2),
                    "min_duration_ms": round(m.min_duration_ms, 2) if m.min_duration_ms != float("inf") else 0.0,
                    "max_duration_ms": round(m.max_duration_ms, 2),
                    "success_rate": round(m.success_rate, 4),
                    "last_executed": m.last_executed,
                }
                for name, m in self._metrics.items()
            },
            "total_spans": len(self._spans),
        }

    def reset(self) -> None:
        """Reset all metrics and spans.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._metrics.clear()
        self._spans.clear()
        self._active_spans.clear()
        self._total_requests = 0
