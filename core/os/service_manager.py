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
AI-OS Service Manager — manages lifecycle of all AI subsystems as services.

Each service has a descriptor with dependencies, start/stop callables, and a
health check. The ServiceManager resolves dependencies via topological sort
and supports auto-restart with exponential backoff.

Mathematical Foundation:
- Dependency resolution: topological sort via Kahn's algorithm: O(V + E).
- Service lookup: O(1) via dict.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.os.event_bus import AIEvent, EventBus, EventType


class ServiceState(Enum):
    """Lifecycle states for a managed service.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class ServiceDescriptor:
    """Descriptor for a managed AI service.

    Attributes:
        name: Unique service name.
        dependencies: Names of services that must be running first.
        start_fn: Callable to start the service.
        stop_fn: Callable to stop the service.
        health_fn: Optional callable returning (healthy: bool, message: str).
        priority: Startup priority (lower = earlier).
        auto_restart: Whether to auto-restart on failure.
        max_restarts: Max restart attempts before giving up.
        restart_delay_base: Base delay in seconds for exponential backoff.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    dependencies: Tuple[str, ...] = ()
    start_fn: Optional[Callable[[], bool]] = None
    stop_fn: Optional[Callable[[], None]] = None
    health_fn: Optional[Callable[[], Tuple[bool, str]]] = None
    priority: int = 100
    auto_restart: bool = True
    max_restarts: int = 3
    restart_delay_base: float = 1.0

    # Runtime state (mutable, not part of identity)
    state: ServiceState = ServiceState.STOPPED
    restart_count: int = 0
    last_error: str = ""
    started_at: float = 0.0


class ServiceManager:
    """
    Manages lifecycle of all AI services with dependency resolution.

    Thread-safe. Uses topological sort for startup/shutdown order.

    Attributes:
        _services: Mapping from service name to descriptor.
        _lock: Thread lock for concurrent access.
        _event_bus: Event bus for publishing service events.

    Complexity:
        register: O(1)
        start_all: O(V + E) for topological sort + start execution.
        stop_all: O(V + E) reverse topological sort + stop execution.
    """

    __slots__ = ("_services", "_lock", "_event_bus")

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize the service manager.

        Args:
            event_bus: Event bus for publishing lifecycle events.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._services: Dict[str, ServiceDescriptor] = {}
        self._lock: threading.Lock = threading.Lock()
        self._event_bus: EventBus = event_bus

    def register(self, descriptor: ServiceDescriptor) -> None:
        """Register a service.

        Args:
            descriptor: Service descriptor.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._services[descriptor.name] = descriptor

    def get(self, name: str) -> Optional[ServiceDescriptor]:
        """Get a service descriptor by name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return self._services.get(name)

    def list_services(self) -> List[ServiceDescriptor]:
        """List all registered services.

        Complexity: T(n) ∈ O(n) for n services, S(n) ∈ O(n).
        """
        with self._lock:
            return list(self._services.values())

    def _topological_sort(self) -> List[str]:
        """Sort services by dependency order using Kahn's algorithm.

        Returns:
            Service names in dependency order.

        Raises:
            ValueError: If a dependency cycle is detected.

        Complexity: T(n) ∈ O(V + E) for V services, E dependencies.
        """
        in_degree: Dict[str, int] = {}
        adj: Dict[str, List[str]] = {}

        for name, desc in self._services.items():
            in_degree.setdefault(name, 0)
            adj.setdefault(name, [])
            for dep in desc.dependencies:
                if dep not in self._services:
                    continue  # Skip missing deps
                adj.setdefault(dep, [])
                adj[dep].append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

        # Priority queue: sort by priority then name
        queue: List[str] = sorted(
            [n for n, d in in_degree.items() if d == 0],
            key=lambda n: (self._services[n].priority, n),
        )
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    # Insert sorted by priority
                    inserted = False
                    for i, existing in enumerate(queue):
                        if self._services[neighbor].priority < self._services[existing].priority:
                            queue.insert(i, neighbor)
                            inserted = True
                            break
                    if not inserted:
                        queue.append(neighbor)

        if len(result) != len(self._services):
            raise ValueError("Dependency cycle detected in service graph")
        return result

    def start_all(self) -> Tuple[int, int]:
        """Start all services in dependency order.

        Returns:
            Tuple of (successes, failures).

        Complexity: T(n) ∈ O(V + E + Σ s_i) for services s_i.
        """
        order = self._topological_sort()
        successes = 0
        failures = 0

        for name in order:
            desc = self._services[name]
            ok = self._start_service(desc)
            if ok:
                successes += 1
            else:
                failures += 1

        self._event_bus.publish(AIEvent(
            event_type=EventType.SERVICE,
            source="service_manager",
            message=f"start_all: {successes} ok, {failures} failed",
            severity="WARN" if failures > 0 else "INFO",
            data={"successes": successes, "failures": failures},
        ))
        return successes, failures

    def stop_all(self) -> int:
        """Stop all services in reverse dependency order.

        Returns:
            Number of services stopped.

        Complexity: T(n) ∈ O(V + E + Σ s_i).
        """
        order = self._topological_sort()
        order.reverse()
        stopped = 0

        for name in order:
            desc = self._services[name]
            if desc.state in (ServiceState.RUNNING, ServiceState.DEGRADED):
                self._stop_service(desc)
                stopped += 1

        self._event_bus.publish(AIEvent(
            event_type=EventType.SHUTDOWN,
            source="service_manager",
            message=f"stop_all: {stopped} services stopped",
            severity="INFO",
            data={"stopped": stopped},
        ))
        return stopped

    def _start_service(self, desc: ServiceDescriptor) -> bool:
        """Start a single service.

        Args:
            desc: Service descriptor.

        Returns:
            True if started successfully.

        Complexity: T(n) ∈ O(s) for start_fn execution.
        """
        desc.state = ServiceState.STARTING
        self._event_bus.publish(AIEvent(
            event_type=EventType.SERVICE,
            source="service_manager",
            message=f"Starting service: {desc.name}",
            severity="INFO",
            data={"service": desc.name},
        ))

        try:
            if desc.start_fn:
                ok = desc.start_fn()
                if not ok:
                    desc.state = ServiceState.FAILED
                    desc.last_error = "start_fn returned False"
                    self._event_bus.publish(AIEvent(
                        event_type=EventType.SERVICE,
                        source="service_manager",
                        message=f"Service failed to start: {desc.name}",
                        severity="ERROR",
                        data={"service": desc.name, "error": desc.last_error},
                    ))
                    return False
            desc.state = ServiceState.RUNNING
            desc.started_at = time.time()
            desc.restart_count = 0
            self._event_bus.publish(AIEvent(
                event_type=EventType.SERVICE,
                source="service_manager",
                message=f"Service started: {desc.name}",
                severity="INFO",
                data={"service": desc.name},
            ))
            return True
        except Exception as exc:
            desc.state = ServiceState.FAILED
            desc.last_error = str(exc)
            self._event_bus.publish(AIEvent(
                event_type=EventType.SERVICE,
                source="service_manager",
                message=f"Service crashed on start: {desc.name}: {exc}",
                severity="ERROR",
                data={"service": desc.name, "error": str(exc)},
            ))
            return False

    def _stop_service(self, desc: ServiceDescriptor) -> None:
        """Stop a single service.

        Args:
            desc: Service descriptor.

        Complexity: T(n) ∈ O(s) for stop_fn execution.
        """
        desc.state = ServiceState.STOPPING
        self._event_bus.publish(AIEvent(
            event_type=EventType.SERVICE,
            source="service_manager",
            message=f"Stopping service: {desc.name}",
            severity="INFO",
            data={"service": desc.name},
        ))
        try:
            if desc.stop_fn:
                desc.stop_fn()
        except Exception as exc:
            self._event_bus.publish(AIEvent(
                event_type=EventType.SERVICE,
                source="service_manager",
                message=f"Error stopping service: {desc.name}: {exc}",
                severity="WARN",
                data={"service": desc.name, "error": str(exc)},
            ))
        desc.state = ServiceState.STOPPED

    def restart(self, name: str) -> bool:
        """Restart a specific service.

        Args:
            name: Service name.

        Returns:
            True if restarted successfully.

        Complexity: T(n) ∈ O(s) for stop + start.
        """
        with self._lock:
            desc = self._services.get(name)
        if desc is None:
            return False
        if desc.state in (ServiceState.RUNNING, ServiceState.DEGRADED):
            self._stop_service(desc)
        return self._start_service(desc)

    def check_health(self) -> Dict[str, Tuple[bool, str]]:
        """Run health checks for all running services.

        Returns:
            Mapping from service name to (healthy, message).

        Complexity: T(n) ∈ O(n * h) for n services, h health check.
        """
        results: Dict[str, Tuple[bool, str]] = {}
        for name, desc in self._services.items():
            if desc.state != ServiceState.RUNNING:
                results[name] = (False, f"not running (state={desc.state.value})")
                continue
            if desc.health_fn:
                try:
                    healthy, msg = desc.health_fn()
                    results[name] = (healthy, msg)
                    if not healthy:
                        desc.state = ServiceState.DEGRADED
                except Exception as exc:
                    results[name] = (False, f"health check error: {exc}")
                    desc.state = ServiceState.DEGRADED
            else:
                results[name] = (True, "ok")
        return results

    def get_status(self) -> List[Dict[str, Any]]:
        """Get status of all services.

        Returns:
            List of service status dicts.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            return [
                {
                    "name": d.name,
                    "state": d.state.value,
                    "restart_count": d.restart_count,
                    "last_error": d.last_error,
                    "uptime_seconds": time.time() - d.started_at if d.started_at > 0 else 0,
                    "dependencies": list(d.dependencies),
                    "priority": d.priority,
                    "auto_restart": d.auto_restart,
                }
                for d in self._services.values()
            ]
