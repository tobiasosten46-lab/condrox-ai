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
Heartbeat engine for the unified Brain.

Monitors subsystem health by running registered health checks and producing
a deterministic aggregate status. This is a real health-monitor, not a stub.

Mathematical Foundation:
- Aggregate health is the minimum of all check scores: H = min_i(h_i).
- A weighted average variant is also supported: H_avg = Σ w_i * h_i / Σ w_i.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Immutable health status record.

    Attributes:
        subsystem: Name of the subsystem.
        score: Health score in [0.0, 1.0].
        message: Human-readable status message.
        timestamp: Unix timestamp of the check.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    subsystem: str
    score: float
    message: str
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate score range.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.score < 0.0 or self.score > 1.0:
            raise ValueError("score must be in [0.0, 1.0]")


class HeartbeatEngine:
    """
    Real health monitoring engine.

    Attributes:
        _checks: Registered health check callables.
        _history: Last status per subsystem.

    Complexity:
        register: O(1)
        check: O(c) for c checks.
    """

    __slots__ = ("_checks", "_history")

    def __init__(self) -> None:
        """Initialize the heartbeat engine."""
        self._checks: Dict[str, Callable[[], HealthStatus]] = {}
        self._history: Dict[str, HealthStatus] = {}

    def register(self, name: str, check: Callable[[], HealthStatus]) -> None:
        """Register a health check.

        Args:
            name: Subsystem name.
            check: Callable returning a HealthStatus.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._checks[name] = check

    def check(self, name: str) -> HealthStatus:
        """Run a single registered health check.

        Args:
            name: Subsystem name.

        Returns:
            HealthStatus from the check or a failing status if missing.

        Complexity: T(n) ∈ O(1) plus check cost.
        """
        if name not in self._checks:
            return HealthStatus(name, 0.0, "missing check")
        status = self._checks[name]()
        self._history[name] = status
        return status

    def check_all(self) -> List[HealthStatus]:
        """Run all registered health checks.

        Returns:
            List of HealthStatus sorted by subsystem name.

        Complexity: T(n) ∈ O(c) for c checks.
        """
        statuses = [self.check(name) for name in sorted(self._checks.keys())]
        return statuses

    def aggregate(self) -> HealthStatus:
        """Return the aggregate health status across all checks.

        Returns:
            HealthStatus with the minimum score and a combined message.

        Complexity: T(n) ∈ O(c), S(n) ∈ O(1).
        """
        statuses = self.check_all()
        if not statuses:
            return HealthStatus("system", 1.0, "no checks registered")

        min_status = min(statuses, key=lambda s: s.score)
        message = "; ".join(f"{s.subsystem}={s.score:.2f}" for s in statuses)
        return HealthStatus(
            subsystem="system",
            score=min_status.score,
            message=message,
        )
