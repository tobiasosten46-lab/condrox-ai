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
AI-OS Kernel — the central orchestrator that owns all OS-level components.

Manages the BootManager, ServiceManager, EventBus, MonitorLoop, and
StateManager. Provides a single entry point for boot, shutdown, and
status queries.

Mathematical Foundation:
- Boot: O(Σ s_i) for boot stages + O(V + E) for service topological sort.
- Shutdown: O(Σ s_i) reverse service stop + state checkpoint.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from core.os.boot_manager import BootManager, BootProgress, BootStage
from core.os.data_flow_manager import DataFlowManager
from core.os.event_bus import AIEvent, EventBus, EventType
from core.os.knowledge_hot_reload import KnowledgeHotReloader
from core.os.monitor_loop import MonitorLoop, SystemHealth
from core.os.self_repair import SelfRepairManager
from core.os.service_manager import ServiceDescriptor, ServiceManager, ServiceState
from core.os.state_manager import StateManager
from core.os.task_queue import TaskQueue


class AIOSKernel:
    """
    Central AI Operating System kernel.

    Owns all OS-level components and provides a unified interface for:
    - Boot (staged, with service orchestration)
    - Shutdown (graceful, with state persistence)
    - Health monitoring (always-on background thread)
    - State management (checkpoint + recovery)
    - Event bus (decoupled inter-service communication)
    - Task queue (background work scheduling)
    - Self-repair (circuit breakers, auto-recovery)

    Attributes:
        event_bus: Central event bus.
        service_manager: Service lifecycle manager.
        boot_manager: Staged boot manager.
        monitor_loop: Background health monitor.
        state_manager: State persistence manager.
        task_queue: Background task queue and scheduler.
        self_repair: Circuit breaker and auto-recovery manager.

    Complexity:
        boot: O(Σ s_i + V + E) for boot stages + service sort.
        shutdown: O(Σ s_i) for service stop + state checkpoint.
    """

    __slots__ = (
        "event_bus",
        "service_manager",
        "boot_manager",
        "monitor_loop",
        "state_manager",
        "task_queue",
        "self_repair",
        "hot_reloader",
        "data_flow",
        "_booted",
        "_boot_time",
    )

    def __init__(
        self,
        state_db_path: str = "Databases/condrox_os_state.db",
        monitor_interval: float = 5.0,
    ) -> None:
        """Initialize the AI-OS kernel.

        Args:
            state_db_path: Path to SQLite state database.
            monitor_interval: Health monitoring interval in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.event_bus: EventBus = EventBus()
        self.service_manager: ServiceManager = ServiceManager(self.event_bus)
        self.boot_manager: BootManager = BootManager(self.event_bus, self.service_manager)
        self.monitor_loop: MonitorLoop = MonitorLoop(
            self.service_manager, self.event_bus, interval=monitor_interval
        )
        self.state_manager: StateManager = StateManager(
            self.event_bus, db_path=state_db_path
        )
        self.task_queue: TaskQueue = TaskQueue(self.event_bus)
        self.self_repair: SelfRepairManager = SelfRepairManager(
            self.service_manager, self.event_bus
        )
        self.hot_reloader: Optional[KnowledgeHotReloader] = None
        self.data_flow: DataFlowManager = DataFlowManager(self.event_bus)
        self._booted: bool = False
        self._boot_time: float = 0.0

    def register_service(self, descriptor: ServiceDescriptor) -> None:
        """Register a service with the service manager.

        Args:
            descriptor: Service descriptor.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.service_manager.register(descriptor)

    def register_boot_stage(
        self,
        stage: BootStage,
        fn: Any,
    ) -> None:
        """Register a boot stage.

        Args:
            stage: Boot stage enum.
            fn: Callable returning (success, message).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.boot_manager.register_stage(stage, fn)

    def boot(self) -> BootProgress:
        """Run the full AI-OS boot sequence.

        1. Recover state from last checkpoint
        2. Run registered boot stages
        3. Start all services in dependency order
        4. Start the monitoring loop
        5. Save initial state checkpoint

        Returns:
            Final BootProgress.

        Complexity: T(n) ∈ O(r + Σ s_i + V + E) for recovery + stages + services.
        """
        boot_start = time.time()

        self.event_bus.publish(AIEvent(
            event_type=EventType.BOOT,
            source="aios_kernel",
            message="AI-OS boot sequence initiated",
            severity="INFO",
        ))

        # 1. Recover state
        self.state_manager.recover()

        # 2. Run boot stages
        progress = self.boot_manager.boot()

        # 3. Start all services
        successes, failures = self.service_manager.start_all()

        self.event_bus.publish(AIEvent(
            event_type=EventType.SERVICE,
            source="aios_kernel",
            message=f"Services started: {successes} ok, {failures} failed",
            severity="WARN" if failures > 0 else "INFO",
            data={"successes": successes, "failures": failures},
        ))

        # 4. Start monitoring
        self.monitor_loop.start()

        # 5. Start task queue
        self.task_queue.start()

        # 6. Start hot-reloader if configured
        if self.hot_reloader is not None:
            self.hot_reloader.start()

        # 7. Start data flow router
        self.data_flow.start_router()

        # 8. Initial checkpoint
        self.state_manager.set("boot_time", time.time() - boot_start)
        self.state_manager.set("boot_progress", progress.message)
        self.state_manager.set("services_started", successes)
        self.state_manager.checkpoint()

        self._booted = True
        self._boot_time = time.time() - boot_start

        self.event_bus.publish(AIEvent(
            event_type=EventType.BOOT,
            source="aios_kernel",
            message=f"AI-OS boot complete in {self._boot_time:.3f}s",
            severity="INFO",
            data={
                "boot_time": round(self._boot_time, 3),
                "services_ok": successes,
                "services_failed": failures,
                "failed_stages": progress.failed_stages,
            },
        ))

        return progress

    def shutdown(self) -> float:
        """Graceful AI-OS shutdown.

        1. Stop monitoring loop
        2. Save final state checkpoint
        3. Stop all services in reverse order
        4. Run boot manager shutdown

        Returns:
            Total shutdown duration in seconds.

        Complexity: T(n) ∈ O(Σ s_i) for service stops.
        """
        shutdown_start = time.time()

        self.event_bus.publish(AIEvent(
            event_type=EventType.SHUTDOWN,
            source="aios_kernel",
            message="AI-OS shutdown initiated",
            severity="INFO",
        ))

        # 1. Stop monitoring
        self.monitor_loop.stop()

        # 2. Stop task queue
        self.task_queue.stop()

        # 3. Stop hot-reloader if running
        if self.hot_reloader is not None:
            self.hot_reloader.stop()

        # 4. Stop data flow router and close all channels
        self.data_flow.close_all()

        # 5. Final checkpoint
        self.state_manager.set("shutdown_time", time.time())
        self.state_manager.checkpoint()

        # 6. Stop all services
        stopped = self.service_manager.stop_all()

        # 7. Boot manager shutdown
        self.boot_manager.shutdown()

        duration = time.time() - shutdown_start
        self._booted = False

        self.event_bus.publish(AIEvent(
            event_type=EventType.SHUTDOWN,
            source="aios_kernel",
            message=f"AI-OS shutdown complete in {duration:.3f}s ({stopped} services)",
            severity="INFO",
            data={"shutdown_time": round(duration, 3), "stopped": stopped},
        ))

        return duration

    @property
    def is_booted(self) -> bool:
        """Whether the AI-OS is booted.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._booted

    def get_status(self) -> Dict[str, Any]:
        """Get full AI-OS status.

        Returns:
            Dict with boot, services, health, events, and state info.

        Complexity: T(n) ∈ O(s + e) for s services, e events.
        """
        health = self.monitor_loop.get_health()
        return {
            "booted": self._booted,
            "boot_time": round(self._boot_time, 3),
            "boot_report": self.boot_manager.get_report(),
            "services": self.service_manager.get_status(),
            "health": {
                "overall_score": health.overall_score if health else 1.0,
                "alert_count": health.alert_count if health else 0,
                "degraded_services": [
                    n for n, (h, _) in (health.service_health if health else {}).items() if not h
                ],
            },
            "monitor": self.monitor_loop.get_status(),
            "task_queue": self.task_queue.get_status(),
            "self_repair": self.self_repair.get_status(),
            "hot_reloader": self.hot_reloader.get_status() if self.hot_reloader else None,
            "data_flow": self.data_flow.get_status(),
            "state": self.state_manager.get_status(),
            "events": {
                "total": self.event_bus.event_count(),
                "recent": [
                    {
                        "type": e.event_type.value,
                        "source": e.source,
                        "message": e.message,
                        "severity": e.severity,
                        "timestamp": e.timestamp,
                    }
                    for e in self.event_bus.get_log(limit=10)
                ],
            },
        }

    def get_health(self) -> Optional[SystemHealth]:
        """Get current system health.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self.monitor_loop.get_health()

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        severity: Optional[str] = None,
    ) -> List[AIEvent]:
        """Get events from the event bus.

        Args:
            event_type: Optional type filter.
            limit: Maximum events.
            severity: Optional severity filter.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return self.event_bus.get_log(event_type=event_type, limit=limit, severity=severity)

    def configure_hot_reloader(
        self,
        knowledge_root: str,
        backend: KnowledgeBackend,
        interval: float = 30.0,
    ) -> None:
        """Configure the knowledge hot-reloader.

        Must be called before boot() to activate hot-reloading.

        Args:
            knowledge_root: Path to knowledge root directory.
            backend: KnowledgeBackend to ingest into.
            interval: Scan interval in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.hot_reloader = KnowledgeHotReloader(
            knowledge_root=knowledge_root,
            backend=backend,
            event_bus=self.event_bus,
            interval=interval,
        )
