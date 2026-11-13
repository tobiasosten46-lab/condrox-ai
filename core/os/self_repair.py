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
AI-OS Self-Repair & Recovery — circuit breakers and auto-recovery actions.

Each service has a circuit breaker that opens after consecutive failures,
triggering auto-recovery (restart, flush cache, reload knowledge, etc.).

Circuit Breaker states: CLOSED → OPEN → HALF_OPEN → CLOSED/OPEN.

Mathematical Foundation:
- Failure window: sliding count of consecutive failures.
- Circuit opens when: failures >= threshold.
- Half-open probe: single test call after cooldown period.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.os.event_bus import AIEvent, EventBus, EventType
from core.os.service_manager import ServiceDescriptor, ServiceManager, ServiceState


class CircuitState(Enum):
    """Circuit breaker states.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Tripped — rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Per-service circuit breaker.

    Attributes:
        service_name: Name of the protected service.
        failure_threshold: Consecutive failures before opening.
        cooldown_seconds: Time before half-open probe.
        state: Current circuit state.
        consecutive_failures: Current failure streak.
        opened_at: Timestamp when circuit opened.
        total_trips: Total times circuit has tripped.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    service_name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    total_trips: int = 0


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    """Immutable record of a recovery action taken.

    Attributes:
        service_name: Service that was recovered.
        action: Recovery action name.
        success: Whether recovery succeeded.
        timestamp: When the action was taken.
        message: Human-readable detail.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    service_name: str
    action: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    message: str = ""


class SelfRepairManager:
    """
    Manages circuit breakers and auto-recovery for all services.

    Attributes:
        _breakers: Mapping from service name to CircuitBreaker.
        _service_manager: Service manager for restart operations.
        _event_bus: Event bus for recovery events.
        _recovery_log: Ring buffer of recovery actions.
        _lock: Thread lock.
        _max_log: Max recovery actions to retain.

    Complexity:
        record_success/failure: O(1).
        check_and_recover: O(s) for s services with open circuits.
    """

    __slots__ = (
        "_breakers",
        "_service_manager",
        "_event_bus",
        "_recovery_log",
        "_lock",
        "_max_log",
    )

    def __init__(
        self,
        service_manager: ServiceManager,
        event_bus: EventBus,
        max_log: int = 200,
    ) -> None:
        """Initialize the self-repair manager.

        Args:
            service_manager: Service manager for restart operations.
            event_bus: Event bus for publishing recovery events.
            max_log: Max recovery actions in log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._service_manager: ServiceManager = service_manager
        self._event_bus: EventBus = event_bus
        self._recovery_log: List[RecoveryAction] = []
        self._lock: threading.Lock = threading.Lock()
        self._max_log: int = max_log

    def register_breaker(
        self,
        service_name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        """Register a circuit breaker for a service.

        Args:
            service_name: Service to protect.
            failure_threshold: Failures before opening.
            cooldown_seconds: Time before half-open probe.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._breakers[service_name] = CircuitBreaker(
                service_name=service_name,
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
            )

    def record_success(self, service_name: str) -> None:
        """Record a successful operation — resets failure count.

        Args:
            service_name: Service that succeeded.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            breaker = self._breakers.get(service_name)
            if breaker:
                breaker.consecutive_failures = 0
                if breaker.state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                    breaker.state = CircuitState.CLOSED
                    self._event_bus.publish(AIEvent(
                        event_type=EventType.ALERT,
                        source="self_repair",
                        message=f"Circuit breaker CLOSED: {service_name} recovered",
                        severity="INFO",
                        data={"service": service_name, "circuit": "closed"},
                    ))

    def record_failure(self, service_name: str) -> CircuitState:
        """Record a failed operation — may trip the circuit.

        Args:
            service_name: Service that failed.

        Returns:
            Current circuit state.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            breaker = self._breakers.get(service_name)
            if breaker is None:
                return CircuitState.CLOSED

            breaker.consecutive_failures += 1

            if breaker.consecutive_failures >= breaker.failure_threshold:
                if breaker.state != CircuitState.OPEN:
                    breaker.state = CircuitState.OPEN
                    breaker.opened_at = time.time()
                    breaker.total_trips += 1
                    self._event_bus.publish(AIEvent(
                        event_type=EventType.ALERT,
                        source="self_repair",
                        message=f"Circuit breaker OPEN: {service_name} ({breaker.consecutive_failures} failures)",
                        severity="CRITICAL",
                        data={
                            "service": service_name,
                            "circuit": "open",
                            "failures": breaker.consecutive_failures,
                            "total_trips": breaker.total_trips,
                        },
                    ))

            return breaker.state

    def check_and_recover(self) -> List[RecoveryAction]:
        """Check all circuit breakers and attempt recovery.

        For OPEN circuits past cooldown, attempt restart (half-open probe).
        For services in FAILED state, attempt auto-restart.

        Returns:
            List of recovery actions taken.

        Complexity: T(n) ∈ O(s) for s services.
        """
        actions: List[RecoveryAction] = []
        now = time.time()

        with self._lock:
            breakers = list(self._breakers.values())

        for breaker in breakers:
            # Check if OPEN circuit is past cooldown → half-open
            if breaker.state == CircuitState.OPEN:
                if now - breaker.opened_at >= breaker.cooldown_seconds:
                    breaker.state = CircuitState.HALF_OPEN
                    self._event_bus.publish(AIEvent(
                        event_type=EventType.ALERT,
                        source="self_repair",
                        message=f"Circuit breaker HALF_OPEN: {breaker.service_name} (probing)",
                        severity="WARN",
                        data={"service": breaker.service_name, "circuit": "half_open"},
                    ))

            # Attempt restart for FAILED or HALF_OPEN services
            desc = self._service_manager.get(breaker.service_name)
            if desc is None:
                continue

            if desc.state == ServiceState.FAILED or breaker.state == CircuitState.HALF_OPEN:
                if desc.auto_restart and desc.restart_count < desc.max_restarts:
                    action = self._attempt_restart(desc, breaker)
                    actions.append(action)

        # Also check for FAILED services without registered breakers
        for desc in self._service_manager.list_services():
            if desc.state == ServiceState.FAILED and desc.auto_restart:
                if desc.restart_count < desc.max_restarts:
                    if desc.name not in self._breakers:
                        action = self._attempt_restart(desc, None)
                        actions.append(action)

        # Log actions
        with self._lock:
            self._recovery_log.extend(actions)
            if len(self._recovery_log) > self._max_log:
                self._recovery_log = self._recovery_log[-self._max_log:]

        return actions

    def _attempt_restart(
        self,
        desc: ServiceDescriptor,
        breaker: Optional[CircuitBreaker],
    ) -> RecoveryAction:
        """Attempt to restart a failed service.

        Args:
            desc: Service descriptor.
            breaker: Optional circuit breaker.

        Returns:
            Recovery action record.

        Complexity: T(n) ∈ O(s) for restart execution.
        """
        desc.restart_count += 1
        try:
            ok = self._service_manager.restart(desc.name)
            action = RecoveryAction(
                service_name=desc.name,
                action="restart",
                success=ok,
                message=f"Restart attempt {desc.restart_count}/{desc.max_restarts}",
            )
            if ok and breaker:
                self.record_success(desc.name)
            elif not ok and breaker:
                self.record_failure(desc.name)
        except Exception as exc:
            action = RecoveryAction(
                service_name=desc.name,
                action="restart",
                success=False,
                message=f"Restart error: {exc}",
            )
            if breaker:
                self.record_failure(desc.name)

        severity = "INFO" if action.success else "ERROR"
        self._event_bus.publish(AIEvent(
            event_type=EventType.ALERT,
            source="self_repair",
            message=f"Recovery: {action.action} {action.service_name} → {'OK' if action.success else 'FAILED'}",
            severity=severity,
            data={
                "service": action.service_name,
                "action": action.action,
                "success": action.success,
                "message": action.message,
            },
        ))

        return action

    def get_breaker_status(self) -> List[Dict[str, Any]]:
        """Get status of all circuit breakers.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            return [
                {
                    "service_name": b.service_name,
                    "state": b.state.value,
                    "consecutive_failures": b.consecutive_failures,
                    "failure_threshold": b.failure_threshold,
                    "total_trips": b.total_trips,
                    "opened_at": b.opened_at,
                    "cooldown_seconds": b.cooldown_seconds,
                }
                for b in self._breakers.values()
            ]

    def get_recovery_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recovery action log.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            log = list(reversed(self._recovery_log))
        return [
            {
                "service_name": a.service_name,
                "action": a.action,
                "success": a.success,
                "timestamp": a.timestamp,
                "message": a.message,
            }
            for a in log[:limit]
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get self-repair manager status.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            open_count = sum(1 for b in self._breakers.values() if b.state == CircuitState.OPEN)
            half_open = sum(1 for b in self._breakers.values() if b.state == CircuitState.HALF_OPEN)
            return {
                "breaker_count": len(self._breakers),
                "open_circuits": open_count,
                "half_open_circuits": half_open,
                "recovery_log_size": len(self._recovery_log),
            }
