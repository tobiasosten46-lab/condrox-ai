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
Latency Compensation Layer — predicts and compensates for processing
latency by pre-computing responses and adjusting timing.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class LatencyPrediction:
    """
    Immutable latency prediction.

    Attributes:
        predicted_ms: Predicted latency in milliseconds.
        confidence: Prediction confidence in [0.0, 1.0].
        based_on_samples: Number of historical samples used.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    predicted_ms: float
    confidence: float
    based_on_samples: int


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    """
    Immutable latency measurement record.

    Attributes:
        stage_name: Pipeline stage name.
        latency_ms: Measured latency in milliseconds.
        timestamp: When the measurement was taken.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    stage_name: str
    latency_ms: float
    timestamp: float


class LatencyCompensationLayer:
    """
    Predicts and compensates for processing latency.

    Uses a sliding window of historical measurements to predict future
    latency and provides compensation strategies.

    Complexity:
        record: T(n) ∈ O(1), S(n) ∈ O(1).
        predict: T(n) ∈ O(w) for w window size, S(n) ∈ O(1).
    """

    __slots__ = (
        "_measurements",
        "_window_size",
        "_stage_history",
        "_total_predictions",
    )

    def __init__(self, window_size: int = 100) -> None:
        """Initialize the latency compensation layer.

        Args:
            window_size: Sliding window size for predictions.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._measurements: Deque[LatencyMeasurement] = deque(maxlen=window_size)
        self._window_size = window_size
        self._stage_history: Dict[str, Deque[float]] = {}
        self._total_predictions: int = 0

    def record(self, stage_name: str, latency_ms: float) -> LatencyMeasurement:
        """Record a latency measurement.

        Args:
            stage_name: Name of the pipeline stage.
            latency_ms: Measured latency in milliseconds.

        Returns:
            The recorded LatencyMeasurement.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        measurement = LatencyMeasurement(
            stage_name=stage_name,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )
        self._measurements.append(measurement)

        if stage_name not in self._stage_history:
            self._stage_history[stage_name] = deque(maxlen=self._window_size)
        self._stage_history[stage_name].append(latency_ms)

        return measurement

    def predict(self, stage_name: Optional[str] = None) -> LatencyPrediction:
        """Predict latency for a stage or the overall pipeline.

        Args:
            stage_name: Specific stage to predict, or None for overall.

        Returns:
            LatencyPrediction with predicted latency and confidence.

        Complexity: T(n) ∈ O(w) for w window size, S(n) ∈ O(1).
        """
        self._total_predictions += 1

        if stage_name is not None:
            history = self._stage_history.get(stage_name)
            if not history:
                return LatencyPrediction(
                    predicted_ms=0.0,
                    confidence=0.0,
                    based_on_samples=0,
                )
            values = list(history)
        else:
            if not self._measurements:
                return LatencyPrediction(
                    predicted_ms=0.0,
                    confidence=0.0,
                    based_on_samples=0,
                )
            values = [m.latency_ms for m in self._measurements]

        # Use exponential moving average for prediction
        alpha = 0.3
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema

        # Confidence based on sample count and variance
        sample_count = len(values)
        if sample_count < 2:
            confidence = sample_count / 10.0
        else:
            mean = sum(values) / sample_count
            variance = sum((v - mean) ** 2 for v in values) / sample_count
            std = variance ** 0.5
            cv = std / mean if mean > 0 else 1.0  # coefficient of variation
            confidence = max(0.0, min(1.0, 1.0 - cv)) * min(1.0, sample_count / 20.0)

        return LatencyPrediction(
            predicted_ms=ema,
            confidence=confidence,
            based_on_samples=sample_count,
        )

    def get_compensation_delay(self, target_ms: float, stage_name: Optional[str] = None) -> float:
        """Calculate compensation delay to meet a target latency.

        If predicted latency exceeds target, returns 0 (can't speed up).
        If predicted latency is below target, returns the delay needed
        to match the target (useful for pacing).

        Args:
            target_ms: Target latency in milliseconds.
            stage_name: Specific stage, or None for overall.

        Returns:
            Compensation delay in milliseconds.

        Complexity: T(n) ∈ O(w), S(n) ∈ O(1).
        """
        prediction = self.predict(stage_name)
        if prediction.predicted_ms >= target_ms:
            return 0.0
        return target_ms - prediction.predicted_ms

    def get_stage_names(self) -> List[str]:
        """Return all stage names with recorded measurements.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s).
        """
        return list(self._stage_history.keys())

    def get_measurements(self, limit: int = 50) -> List[LatencyMeasurement]:
        """Return recent measurements.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._measurements)[-limit:]

    def measurement_count(self) -> int:
        """Return total measurement count.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._measurements)

    def clear(self) -> None:
        """Clear all measurements.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._measurements.clear()
        self._stage_history.clear()
        self._total_predictions = 0

    def export_stats(self) -> Dict[str, any]:
        """Export statistics as a serializable dict.

        Complexity: T(n) ∈ O(s * w), S(n) ∈ O(s).
        """
        stage_stats: Dict[str, Dict[str, float]] = {}
        for stage, history in self._stage_history.items():
            values = list(history)
            if values:
                stage_stats[stage] = {
                    "count": len(values),
                    "avg_ms": round(sum(values) / len(values), 2),
                    "min_ms": round(min(values), 2),
                    "max_ms": round(max(values), 2),
                }
        return {
            "total_measurements": len(self._measurements),
            "total_predictions": self._total_predictions,
            "stages": stage_stats,
        }
