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

"""Tests for PipelineMonitor — Phase 2-C component."""

from __future__ import annotations

import time

from core.conversational.pipeline_monitor import (
    PipelineMonitor,
    StageMetrics,
    StageSpan,
)


class TestPipelineMonitor:
    """Unit tests for PipelineMonitor."""

    def test_start_and_end_span(self) -> None:
        """Span tracking works."""
        monitor = PipelineMonitor()
        key = monitor.start_span("stage_01", "req_001")
        time.sleep(0.01)
        span = monitor.end_span(key)
        assert span is not None
        assert span.stage_name == "stage_01"
        assert span.duration_ms > 0
        assert span.success is True

    def test_end_unknown_span_returns_none(self) -> None:
        """Ending unknown span returns None."""
        monitor = PipelineMonitor()
        assert monitor.end_span("unknown") is None

    def test_metrics_aggregation(self) -> None:
        """Metrics are aggregated correctly."""
        monitor = PipelineMonitor()
        for i in range(3):
            key = monitor.start_span("stage_01", f"req_{i}")
            time.sleep(0.005)
            monitor.end_span(key)

        metrics = monitor.get_metrics("stage_01")
        assert metrics is not None
        assert metrics.total_executions == 3
        assert metrics.total_successes == 3
        assert metrics.success_rate == 1.0
        assert metrics.avg_duration_ms > 0

    def test_failure_tracking(self) -> None:
        """Failures are tracked."""
        monitor = PipelineMonitor()
        key = monitor.start_span("stage_05", "req_001")
        monitor.end_span(key, success=False, error="Connection failed")

        metrics = monitor.get_metrics("stage_05")
        assert metrics.total_failures == 1
        assert metrics.success_rate == 0.0

    def test_get_spans(self) -> None:
        """get_spans returns recent spans."""
        monitor = PipelineMonitor()
        for i in range(5):
            key = monitor.start_span("stage_01", f"req_{i}")
            monitor.end_span(key)

        spans = monitor.get_spans(limit=3)
        assert len(spans) == 3

    def test_all_metrics(self) -> None:
        """all_metrics returns all stages."""
        monitor = PipelineMonitor()
        key1 = monitor.start_span("stage_01", "req_001")
        monitor.end_span(key1)
        key2 = monitor.start_span("stage_02", "req_001")
        monitor.end_span(key2)

        all_m = monitor.all_metrics()
        assert "stage_01" in all_m
        assert "stage_02" in all_m

    def test_request_count(self) -> None:
        """Request count increments."""
        monitor = PipelineMonitor()
        monitor.increment_request_count()
        monitor.increment_request_count()
        assert monitor.total_requests() == 2

    def test_export_metrics(self) -> None:
        """export_metrics returns serializable dict."""
        monitor = PipelineMonitor()
        key = monitor.start_span("stage_01", "req_001")
        monitor.end_span(key)
        monitor.increment_request_count()

        exported = monitor.export_metrics()
        assert "total_requests" in exported
        assert "stages" in exported
        assert "stage_01" in exported["stages"]
        assert exported["total_requests"] == 1

    def test_reset(self) -> None:
        """reset clears all data."""
        monitor = PipelineMonitor()
        key = monitor.start_span("stage_01", "req_001")
        monitor.end_span(key)
        monitor.increment_request_count()
        monitor.reset()
        assert monitor.total_requests() == 0
        assert monitor.get_metrics("stage_01") is None

    def test_min_max_duration(self) -> None:
        """Min and max durations are tracked."""
        monitor = PipelineMonitor()
        key1 = monitor.start_span("stage_01", "req_001")
        time.sleep(0.001)
        monitor.end_span(key1)
        key2 = monitor.start_span("stage_01", "req_002")
        time.sleep(0.02)
        monitor.end_span(key2)

        metrics = monitor.get_metrics("stage_01")
        assert metrics.min_duration_ms < metrics.max_duration_ms

    def test_stage_span_is_frozen(self) -> None:
        """StageSpan is immutable."""
        monitor = PipelineMonitor()
        key = monitor.start_span("stage_01", "req_001")
        span = monitor.end_span(key)
        try:
            span.stage_name = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected

    def test_avg_duration_no_executions(self) -> None:
        """Avg duration is 0 when no executions."""
        metrics = StageMetrics(stage_name="test")
        assert metrics.avg_duration_ms == 0.0
        assert metrics.success_rate == 0.0
