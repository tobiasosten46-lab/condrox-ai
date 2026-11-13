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
AI-OS Monitor Loop — always-on background thread that monitors system health,
collects metrics, detects anomalies, and emits alerts.

Runs as a daemon thread at a configurable interval (default 5s).

Mathematical Foundation:
- Health score: H = min(h_i) for all subsystem checks i.
- Anomaly detection: threshold-based on rolling window averages.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.os.event_bus import AIEvent, EventBus, EventType
from core.os.service_manager import ServiceManager


@dataclass(frozen=True, slots=True)
class SystemHealth:
    """Immutable system health snapshot.

    Attributes:
        overall_score: Aggregate health in [0.0, 1.0].
        service_health: Per-service (healthy, message) dict.
        alert_count: Number of active alerts.
        timestamp: Unix timestamp.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(s) for services.
    """

    overall_score: float
    service_health: Dict[str, Tuple[bool, str]] = field(default_factory=dict, hash=False)
    alert_count: int = 0
    timestamp: float = field(default_factory=time.time)


class MonitorLoop:
    """
    Background monitoring thread for the AI-OS.

    Runs health checks, collects metrics, detects anomalies, and publishes
    alerts via the event bus.

    Attributes:
        _service_manager: Service manager for health checks.
        _event_bus: Event bus for publishing health/alert events.
        _interval: Monitoring interval in seconds.
        _thread: Background daemon thread.
        _running: Whether the loop is active.
        _health_history: Ring buffer of recent health snapshots.
        _anomaly_thresholds: Configured thresholds for anomaly detection.
        _lock: Thread lock.

    Complexity:
        start: O(1)
        _run_loop: O(s * h) per iteration for s services, h health checks.
    """

    __slots__ = (
        "_service_manager",
        "_event_bus",
        "_interval",
        "_thread",
        "_running",
        "_health_history",
        "_anomaly_thresholds",
        "_lock",
        "_last_health",
    )

    def __init__(
        self,
        service_manager: ServiceManager,
        event_bus: EventBus,
        interval: float = 5.0,
    ) -> None:
        """Initialize the monitor loop.

        Args:
            service_manager: Service manager for health checks.
            event_bus: Event bus for publishing events.
            interval: Monitoring interval in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._service_manager: ServiceManager = service_manager
        self._event_bus: EventBus = event_bus
        self._interval: float = interval
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._health_history: List[SystemHealth] = []
        self._anomaly_thresholds: Dict[str, float] = {
            "min_health_score": 0.5,
            "max_degraded_services": 3,
            "history_size": 60,  # 5 min at 5s interval
        }
        self._lock: threading.Lock = threading.Lock()
        self._last_health: Optional[SystemHealth] = None

    def start(self) -> None:
        """Start the monitoring loop as a daemon thread.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ai-os-monitor"
        )
        self._thread.start()
        self._event_bus.publish(AIEvent(
            event_type=EventType.HEALTH,
            source="monitor_loop",
            message=f"Monitor loop started (interval={self._interval}s)",
            severity="INFO",
        ))

    def stop(self) -> None:
        """Stop the monitoring loop.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval * 2)
        self._event_bus.publish(AIEvent(
            event_type=EventType.HEALTH,
            source="monitor_loop",
            message="Monitor loop stopped",
            severity="INFO",
        ))

    def _run_loop(self) -> None:
        """Main monitoring loop — runs until stopped.

        Complexity: T(n) ∈ O(i * s * h) for i iterations, s services, h checks.
        """
        while self._running:
            try:
                health = self._collect_health()
                self._detect_anomalies(health)
            except Exception as exc:
                self._event_bus.publish(AIEvent(
                    event_type=EventType.ALERT,
                    source="monitor_loop",
                    message=f"Monitor loop error: {exc}",
                    severity="ERROR",
                    data={"error": str(exc)},
                ))
            time.sleep(self._interval)

    def _collect_health(self) -> SystemHealth:
        """Collect health from all services.

        Returns:
            SystemHealth snapshot.

        Complexity: T(n) ∈ O(s * h) for s services, h health checks.
        """
        service_health = self._service_manager.check_health()
        scores = [1.0 if h else 0.0 for h, _ in service_health.values()]
        overall = min(scores) if scores else 1.0

        health = SystemHealth(
            overall_score=overall,
            service_health=service_health,
            alert_count=sum(1 for h, _ in service_health.values() if not h),
        )

        with self._lock:
            self._last_health = health
            self._health_history.append(health)
            max_history = int(self._anomaly_thresholds.get("history_size", 60))
            if len(self._health_history) > max_history:
                self._health_history = self._health_history[-max_history:]

        if health.alert_count > 0:
            self._event_bus.publish(AIEvent(
                event_type=EventType.HEALTH,
                source="monitor_loop",
                message=f"Health check: {health.alert_count} degraded services",
                severity="WARN" if health.alert_count <= 2 else "ERROR",
                data={
                    "overall_score": overall,
                    "degraded": [n for n, (h, _) in service_health.items() if not h],
                },
            ))

        return health

    def _detect_anomalies(self, health: SystemHealth) -> None:
        """Detect anomalies and emit alerts.

        Args:
            health: Current health snapshot.

        Complexity: T(n) ∈ O(1) for threshold checks.
        """
        min_score = self._anomaly_thresholds.get("min_health_score", 0.5)
        max_degraded = self._anomaly_thresholds.get("max_degraded_services", 3)

        if health.overall_score < min_score:
            self._event_bus.publish(AIEvent(
                event_type=EventType.ALERT,
                source="monitor_loop",
                message=f"ANOMALY: System health score {health.overall_score:.2f} below threshold {min_score}",
                severity="CRITICAL",
                data={"score": health.overall_score, "threshold": min_score},
            ))

        if health.alert_count > max_degraded:
            self._event_bus.publish(AIEvent(
                event_type=EventType.ALERT,
                source="monitor_loop",
                message=f"ANOMALY: {health.alert_count} degraded services exceeds threshold {max_degraded}",
                severity="CRITICAL",
                data={"degraded_count": health.alert_count, "threshold": max_degraded},
            ))

    def get_health(self) -> Optional[SystemHealth]:
        """Get the latest health snapshot.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return self._last_health

    def get_history(self, limit: int = 20) -> List[SystemHealth]:
        """Get recent health history.

        Args:
            limit: Maximum snapshots to return.

        Returns:
            List of health snapshots, newest first.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            history = list(self._health_history)
        history.reverse()
        return history[:limit]

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status as dict.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        health = self.get_health()
        return {
            "running": self._running,
            "interval": self._interval,
            "overall_score": health.overall_score if health else 1.0,
            "alert_count": health.alert_count if health else 0,
            "history_size": len(self._health_history),
            "thresholds": dict(self._anomaly_thresholds),
        }
