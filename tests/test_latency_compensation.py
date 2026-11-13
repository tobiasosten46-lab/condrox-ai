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

"""Tests for LatencyCompensationLayer — Phase 2-F component."""

from __future__ import annotations

import time

from core.conversational.latency_compensation import (
    LatencyCompensationLayer,
    LatencyMeasurement,
    LatencyPrediction,
)


class TestLatencyCompensationLayer:
    """Unit tests for LatencyCompensationLayer."""

    def test_record_and_predict(self) -> None:
        """Recording measurements enables prediction."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.record("stage_01", 20.0)
        pred = layer.predict("stage_01")
        assert pred.predicted_ms > 0
        assert pred.based_on_samples == 2

    def test_predict_no_data(self) -> None:
        """Prediction with no data returns zeros."""
        layer = LatencyCompensationLayer()
        pred = layer.predict("unknown_stage")
        assert pred.predicted_ms == 0.0
        assert pred.confidence == 0.0

    def test_predict_overall(self) -> None:
        """Overall prediction uses all measurements."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.record("stage_02", 20.0)
        pred = layer.predict()
        assert pred.predicted_ms > 0
        assert pred.based_on_samples == 2

    def test_confidence_increases_with_samples(self) -> None:
        """More samples increase confidence."""
        layer = LatencyCompensationLayer()
        for i in range(20):
            layer.record("stage_01", 10.0 + i * 0.1)
        pred = layer.predict("stage_01")
        assert pred.confidence > 0.5

    def test_compensation_delay(self) -> None:
        """Compensation delay is calculated correctly."""
        layer = LatencyCompensationLayer()
        for _ in range(10):
            layer.record("stage_01", 5.0)
        delay = layer.get_compensation_delay(10.0, "stage_01")
        assert delay > 0  # Should need to wait to reach 10ms

    def test_compensation_delay_zero_when_over_target(self) -> None:
        """Delay is zero when predicted exceeds target."""
        layer = LatencyCompensationLayer()
        for _ in range(10):
            layer.record("stage_01", 50.0)
        delay = layer.get_compensation_delay(10.0, "stage_01")
        assert delay == 0.0

    def test_get_stage_names(self) -> None:
        """Stage names are returned."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.record("stage_02", 20.0)
        names = layer.get_stage_names()
        assert "stage_01" in names
        assert "stage_02" in names

    def test_measurement_count(self) -> None:
        """Measurement count is correct."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.record("stage_01", 20.0)
        assert layer.measurement_count() == 2

    def test_get_measurements(self) -> None:
        """get_measurements returns recent records."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.record("stage_02", 20.0)
        measurements = layer.get_measurements()
        assert len(measurements) == 2

    def test_clear(self) -> None:
        """Clear resets all data."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.clear()
        assert layer.measurement_count() == 0

    def test_export_stats(self) -> None:
        """export_stats returns serializable dict."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        layer.record("stage_01", 20.0)
        stats = layer.export_stats()
        assert "total_measurements" in stats
        assert "stages" in stats
        assert "stage_01" in stats["stages"]

    def test_latency_prediction_is_frozen(self) -> None:
        """LatencyPrediction is immutable."""
        layer = LatencyCompensationLayer()
        layer.record("stage_01", 10.0)
        pred = layer.predict("stage_01")
        try:
            pred.predicted_ms = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_latency_measurement_is_frozen(self) -> None:
        """LatencyMeasurement is immutable."""
        layer = LatencyCompensationLayer()
        m = layer.record("stage_01", 10.0)
        try:
            m.latency_ms = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_window_size_limits_measurements(self) -> None:
        """Window size limits stored measurements."""
        layer = LatencyCompensationLayer(window_size=5)
        for i in range(10):
            layer.record("stage_01", float(i))
        assert layer.measurement_count() == 5
