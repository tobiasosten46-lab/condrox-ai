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
AI-OS Boot Manager — staged boot with dependency-ordered service initialization,
progress tracking, and graceful shutdown.

Replaces the basic CoreBoot with a full staged boot sequence:
PRE_BOOT → INIT → LOAD_KNOWLEDGE → LOAD_MEMORY → WARM_CACHE → READY

Mathematical Foundation:
- Boot success is the conjunction of all stages: B = ∧ s_i.
- Progress is cumulative: P = (completed_stages / total_stages) * 100.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.os.event_bus import AIEvent, EventBus, EventType
from core.os.service_manager import ServiceManager


class BootStage(Enum):
    """Staged boot phases for the AI-OS.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    PRE_BOOT = "pre_boot"
    INIT = "init"
    LOAD_KNOWLEDGE = "load_knowledge"
    LOAD_MEMORY = "load_memory"
    WARM_CACHE = "warm_cache"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class BootProgress:
    """Immutable boot progress snapshot.

    Attributes:
        stage: Current boot stage.
        progress_pct: 0.0 to 100.0.
        message: Human-readable status.
        stage_timings: Mapping of stage name to duration in seconds.
        failed_stages: List of failed stage names.
        timestamp: Unix timestamp.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(s) for stages.
    """

    stage: str
    progress_pct: float
    message: str
    stage_timings: Dict[str, float] = field(default_factory=dict, hash=False)
    failed_stages: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class BootManager:
    """
    Staged boot manager for the AI-OS.

    Runs each boot stage in order, tracks progress, timing, and failures.
    Failed stages mark the system as degraded but don't halt boot.

    Attributes:
        _stages: Ordered list of (stage, callable) pairs.
        _event_bus: Event bus for publishing boot events.
        _service_manager: Service manager for coordinated startup.
        _progress: Current boot progress.
        _lock: Thread lock.
        _booted: Whether boot has completed.
        _boot_time: Total boot duration in seconds.

    Complexity:
        boot: O(Σ s_i) for stages s_i.
        shutdown: O(Σ s_i) for shutdown stages.
    """

    __slots__ = (
        "_stages",
        "_event_bus",
        "_service_manager",
        "_progress",
        "_lock",
        "_booted",
        "_boot_time",
    )

    def __init__(
        self,
        event_bus: EventBus,
        service_manager: ServiceManager,
    ) -> None:
        """Initialize the boot manager.

        Args:
            event_bus: Event bus for publishing boot events.
            service_manager: Service manager for coordinated startup.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._stages: List[Tuple[BootStage, Callable[[], Tuple[bool, str]]]] = []
        self._event_bus: EventBus = event_bus
        self._service_manager: ServiceManager = service_manager
        self._progress: BootProgress = BootProgress(
            stage=BootStage.PRE_BOOT.value,
            progress_pct=0.0,
            message="Not booted",
        )
        self._lock = __import__("threading").Lock()
        self._booted: bool = False
        self._boot_time: float = 0.0

    def register_stage(
        self,
        stage: BootStage,
        fn: Callable[[], Tuple[bool, str]],
    ) -> None:
        """Register a boot stage.

        Args:
            stage: Boot stage enum.
            fn: Callable returning (success, message).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._stages.append((stage, fn))

    def boot(self) -> BootProgress:
        """Run all boot stages in order.

        Failed stages are logged but don't halt boot — system continues
        in degraded mode.

        Returns:
            Final BootProgress.

        Complexity: T(n) ∈ O(Σ s_i) for stages s_i.
        """
        boot_start = time.time()
        total = len(self._stages)
        stage_timings: Dict[str, float] = {}
        failed_stages: List[str] = []

        self._event_bus.publish(AIEvent(
            event_type=EventType.BOOT,
            source="boot_manager",
            message=f"Boot sequence started ({total} stages)",
            severity="INFO",
            data={"total_stages": total},
        ))

        for i, (stage, fn) in enumerate(self._stages):
            stage_start = time.time()
            pct = (i / total) * 100.0 if total > 0 else 100.0

            with self._lock:
                self._progress = BootProgress(
                    stage=stage.value,
                    progress_pct=pct,
                    message=f"Running stage: {stage.value}",
                    stage_timings=stage_timings,
                    failed_stages=failed_stages,
                )

            self._event_bus.publish(AIEvent(
                event_type=EventType.BOOT,
                source="boot_manager",
                message=f"Boot stage: {stage.value}",
                severity="INFO",
                data={"stage": stage.value, "progress": pct},
            ))

            try:
                ok, msg = fn()
            except Exception as exc:
                ok = False
                msg = str(exc)

            duration = time.time() - stage_start
            stage_timings[stage.value] = round(duration, 3)

            if not ok:
                failed_stages.append(stage.value)
                self._event_bus.publish(AIEvent(
                    event_type=EventType.BOOT,
                    source="boot_manager",
                    message=f"Boot stage FAILED: {stage.value}: {msg}",
                    severity="WARN",
                    data={"stage": stage.value, "error": msg, "duration": duration},
                ))
            else:
                self._event_bus.publish(AIEvent(
                    event_type=EventType.BOOT,
                    source="boot_manager",
                    message=f"Boot stage OK: {stage.value} ({duration:.3f}s)",
                    severity="INFO",
                    data={"stage": stage.value, "duration": duration},
                ))

        self._boot_time = time.time() - boot_start
        self._booted = True

        final_pct = 100.0
        final_msg = "Boot complete"
        if failed_stages:
            final_msg = f"Boot complete (degraded: {', '.join(failed_stages)})"

        with self._lock:
            self._progress = BootProgress(
                stage=BootStage.READY.value,
                progress_pct=final_pct,
                message=final_msg,
                stage_timings=stage_timings,
                failed_stages=failed_stages,
            )

        self._event_bus.publish(AIEvent(
            event_type=EventType.BOOT,
            source="boot_manager",
            message=f"Boot complete in {self._boot_time:.3f}s ({len(failed_stages)} failures)",
            severity="WARN" if failed_stages else "INFO",
            data={
                "boot_time": round(self._boot_time, 3),
                "failed_stages": failed_stages,
                "stage_timings": stage_timings,
            },
        ))

        return self._progress

    def shutdown(self) -> float:
        """Graceful shutdown — stop all services in reverse order.

        Returns:
            Shutdown duration in seconds.

        Complexity: T(n) ∈ O(Σ s_i) for service stop calls.
        """
        shutdown_start = time.time()

        self._event_bus.publish(AIEvent(
            event_type=EventType.SHUTDOWN,
            source="boot_manager",
            message="Shutdown sequence started",
            severity="INFO",
        ))

        stopped = self._service_manager.stop_all()

        duration = time.time() - shutdown_start
        self._booted = False

        with self._lock:
            self._progress = BootProgress(
                stage="shutdown",
                progress_pct=0.0,
                message=f"Shutdown complete ({stopped} services, {duration:.3f}s)",
            )

        self._event_bus.publish(AIEvent(
            event_type=EventType.SHUTDOWN,
            source="boot_manager",
            message=f"Shutdown complete in {duration:.3f}s ({stopped} services)",
            severity="INFO",
            data={"shutdown_time": round(duration, 3), "stopped": stopped},
        ))

        return duration

    def get_progress(self) -> BootProgress:
        """Get current boot progress.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return self._progress

    @property
    def is_booted(self) -> bool:
        """Whether boot has completed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._booted

    @property
    def boot_time(self) -> float:
        """Total boot duration in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._boot_time

    def get_report(self) -> Dict[str, Any]:
        """Get full boot report as dict.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        p = self.get_progress()
        return {
            "booted": self._booted,
            "boot_time": round(self._boot_time, 3),
            "stage": p.stage,
            "progress_pct": p.progress_pct,
            "message": p.message,
            "stage_timings": p.stage_timings,
            "failed_stages": p.failed_stages,
        }
