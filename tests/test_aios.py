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
Tests for AI-OS components.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Tuple

import pytest

from core.os.aios_kernel import AIOSKernel
from core.os.boot_manager import BootManager, BootStage
from core.os.data_flow_manager import ChannelState, DataChannel, DataFlowManager
from core.os.event_bus import AIEvent, EventBus, EventType
from core.os.knowledge_hot_reload import KnowledgeHotReloader
from core.os.monitor_loop import MonitorLoop
from core.os.self_repair import CircuitState, SelfRepairManager
from core.os.service_manager import ServiceDescriptor, ServiceManager, ServiceState
from core.os.state_manager import StateManager
from core.os.task_queue import Task, TaskQueue, TaskState
from core.knowledge.knowledge_backend import KnowledgeBackend


# ─── EventBus tests ──────────────────────────────────────────────────────────


class TestEventBus:
    """Tests for the EventBus."""

    def test_publish_subscribe(self) -> None:
        """Events are delivered to subscribers."""
        bus = EventBus()
        received: list[AIEvent] = []
        bus.subscribe(EventType.BOOT, lambda e: received.append(e))
        bus.publish(AIEvent(
            event_type=EventType.BOOT,
            source="test",
            message="boot event",
        ))
        assert len(received) == 1
        assert received[0].message == "boot event"

    def test_no_subscribers_no_error(self) -> None:
        """Publishing with no subscribers doesn't error."""
        bus = EventBus()
        bus.publish(AIEvent(
            event_type=EventType.HEALTH,
            source="test",
            message="no subscribers",
        ))
        assert bus.event_count() == 1

    def test_event_log(self) -> None:
        """Event log stores events."""
        bus = EventBus(max_log=10)
        for i in range(5):
            bus.publish(AIEvent(
                event_type=EventType.SERVICE,
                source="test",
                message=f"event {i}",
            ))
        log = bus.get_log(limit=10)
        assert len(log) == 5
        # Newest first
        assert log[0].message == "event 4"

    def test_event_log_filter_by_type(self) -> None:
        """Event log filters by type."""
        bus = EventBus()
        bus.publish(AIEvent(event_type=EventType.BOOT, source="t", message="b"))
        bus.publish(AIEvent(event_type=EventType.HEALTH, source="t", message="h"))
        log = bus.get_log(event_type=EventType.BOOT)
        assert len(log) == 1
        assert log[0].event_type == EventType.BOOT

    def test_event_log_filter_by_severity(self) -> None:
        """Event log filters by severity."""
        bus = EventBus()
        bus.publish(AIEvent(event_type=EventType.ALERT, source="t", message="a", severity="WARN"))
        bus.publish(AIEvent(event_type=EventType.ALERT, source="t", message="b", severity="ERROR"))
        log = bus.get_log(severity="ERROR")
        assert len(log) == 1
        assert log[0].severity == "ERROR"

    def test_ring_buffer_max(self) -> None:
        """Ring buffer respects max_log size."""
        bus = EventBus(max_log=3)
        for i in range(5):
            bus.publish(AIEvent(event_type=EventType.BOOT, source="t", message=f"e{i}"))
        assert bus.event_count() == 3

    def test_handler_exception_no_crash(self) -> None:
        """A failing handler doesn't crash the bus."""
        bus = EventBus()
        received: list[AIEvent] = []
        def bad_handler(e: AIEvent) -> None:
            raise RuntimeError("boom")
        bus.subscribe(EventType.BOOT, bad_handler)
        bus.subscribe(EventType.BOOT, lambda e: received.append(e))
        bus.publish(AIEvent(event_type=EventType.BOOT, source="t", message="ok"))
        assert len(received) == 1

    def test_clear_log(self) -> None:
        """Clear log removes all events."""
        bus = EventBus()
        bus.publish(AIEvent(event_type=EventType.BOOT, source="t", message="x"))
        assert bus.event_count() == 1
        cleared = bus.clear_log()
        assert cleared == 1
        assert bus.event_count() == 0


# ─── ServiceManager tests ────────────────────────────────────────────────────


class TestServiceManager:
    """Tests for the ServiceManager."""

    def test_register_and_get(self) -> None:
        """Register and retrieve a service."""
        bus = EventBus()
        sm = ServiceManager(bus)
        desc = ServiceDescriptor(name="test_svc")
        sm.register(desc)
        assert sm.get("test_svc") is not None
        assert sm.get("nonexistent") is None

    def test_list_services(self) -> None:
        """List all registered services."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(name="a", priority=10))
        sm.register(ServiceDescriptor(name="b", priority=20))
        services = sm.list_services()
        assert len(services) == 2

    def test_start_all_no_deps(self) -> None:
        """Start all services without dependencies."""
        bus = EventBus()
        sm = ServiceManager(bus)
        started: list[str] = []
        sm.register(ServiceDescriptor(
            name="svc_a",
            start_fn=lambda: (started.append("a"), True)[1],
        ))
        sm.register(ServiceDescriptor(
            name="svc_b",
            start_fn=lambda: (started.append("b"), True)[1],
        ))
        ok, fail = sm.start_all()
        assert ok == 2
        assert fail == 0
        assert sm.get("svc_a").state == ServiceState.RUNNING
        assert sm.get("svc_b").state == ServiceState.RUNNING

    def test_start_all_with_deps(self) -> None:
        """Services start in dependency order."""
        bus = EventBus()
        sm = ServiceManager(bus)
        order: list[str] = []
        sm.register(ServiceDescriptor(
            name="dependent",
            dependencies=("base",),
            start_fn=lambda: (order.append("dependent"), True)[1],
            priority=50,
        ))
        sm.register(ServiceDescriptor(
            name="base",
            start_fn=lambda: (order.append("base"), True)[1],
            priority=10,
        ))
        sm.start_all()
        assert order[0] == "base"
        assert order[1] == "dependent"

    def test_start_failure_marks_failed(self) -> None:
        """Failed service start marks as FAILED."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(
            name="bad_svc",
            start_fn=lambda: False,
        ))
        ok, fail = sm.start_all()
        assert fail == 1
        assert sm.get("bad_svc").state == ServiceState.FAILED

    def test_start_exception_marks_failed(self) -> None:
        """Exception during start marks as FAILED."""
        bus = EventBus()
        sm = ServiceManager(bus)
        def crash() -> bool:
            raise RuntimeError("crash")
        sm.register(ServiceDescriptor(name="crash_svc", start_fn=crash))
        ok, fail = sm.start_all()
        assert fail == 1
        assert sm.get("crash_svc").state == ServiceState.FAILED
        assert "crash" in sm.get("crash_svc").last_error

    def test_stop_all(self) -> None:
        """Stop all running services."""
        bus = EventBus()
        sm = ServiceManager(bus)
        stopped: list[str] = []
        sm.register(ServiceDescriptor(
            name="svc",
            start_fn=lambda: True,
            stop_fn=lambda: stopped.append("svc"),
        ))
        sm.start_all()
        sm.stop_all()
        assert sm.get("svc").state == ServiceState.STOPPED
        assert stopped == ["svc"]

    def test_restart(self) -> None:
        """Restart a running service."""
        bus = EventBus()
        sm = ServiceManager(bus)
        starts: list[str] = []
        sm.register(ServiceDescriptor(
            name="svc",
            start_fn=lambda: (starts.append("start"), True)[1],
            stop_fn=lambda: None,
        ))
        sm.start_all()
        assert len(starts) == 1
        ok = sm.restart("svc")
        assert ok
        assert len(starts) == 2

    def test_check_health(self) -> None:
        """Health check returns per-service status."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(
            name="healthy",
            start_fn=lambda: True,
            health_fn=lambda: (True, "ok"),
        ))
        sm.register(ServiceDescriptor(
            name="unhealthy",
            start_fn=lambda: True,
            health_fn=lambda: (False, "degraded"),
        ))
        sm.start_all()
        health = sm.check_health()
        assert health["healthy"] == (True, "ok")
        assert health["unhealthy"] == (False, "degraded")

    def test_get_status(self) -> None:
        """Get status returns list of service dicts."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(name="svc", start_fn=lambda: True))
        sm.start_all()
        status = sm.get_status()
        assert len(status) == 1
        assert status[0]["name"] == "svc"
        assert status[0]["state"] == "running"

    def test_cycle_detection(self) -> None:
        """Dependency cycle raises ValueError."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(name="a", dependencies=("b",)))
        sm.register(ServiceDescriptor(name="b", dependencies=("a",)))
        with pytest.raises(ValueError, match="cycle"):
            sm.start_all()


# ─── BootManager tests ───────────────────────────────────────────────────────


class TestBootManager:
    """Tests for the BootManager."""

    def test_boot_success(self) -> None:
        """Boot completes with all stages passing."""
        bus = EventBus()
        sm = ServiceManager(bus)
        bm = BootManager(bus, sm)
        bm.register_stage(BootStage.INIT, lambda: (True, "init ok"))
        bm.register_stage(BootStage.READY, lambda: (True, "ready ok"))
        progress = bm.boot()
        assert bm.is_booted
        assert progress.progress_pct == 100.0
        assert len(progress.failed_stages) == 0

    def test_boot_partial_failure(self) -> None:
        """Boot continues despite stage failure."""
        bus = EventBus()
        sm = ServiceManager(bus)
        bm = BootManager(bus, sm)
        bm.register_stage(BootStage.INIT, lambda: (True, "ok"))
        bm.register_stage(BootStage.LOAD_KNOWLEDGE, lambda: (False, "no knowledge"))
        bm.register_stage(BootStage.READY, lambda: (True, "ok"))
        progress = bm.boot()
        assert bm.is_booted
        assert "load_knowledge" in progress.failed_stages

    def test_boot_exception_in_stage(self) -> None:
        """Exception in stage is caught and marked failed."""
        bus = EventBus()
        sm = ServiceManager(bus)
        bm = BootManager(bus, sm)
        def crash() -> Tuple[bool, str]:
            raise RuntimeError("boom")
        bm.register_stage(BootStage.INIT, crash)
        progress = bm.boot()
        assert "init" in progress.failed_stages

    def test_boot_progress_tracking(self) -> None:
        """Boot progress is tracked during boot."""
        bus = EventBus()
        sm = ServiceManager(bus)
        bm = BootManager(bus, sm)
        bm.register_stage(BootStage.INIT, lambda: (True, "ok"))
        bm.register_stage(BootStage.READY, lambda: (True, "ok"))
        bm.boot()
        report = bm.get_report()
        assert report["booted"] is True
        assert "init" in report["stage_timings"]
        assert "ready" in report["stage_timings"]

    def test_shutdown(self) -> None:
        """Shutdown stops services."""
        bus = EventBus()
        sm = ServiceManager(bus)
        bm = BootManager(bus, sm)
        sm.register(ServiceDescriptor(name="svc", start_fn=lambda: True, stop_fn=lambda: None))
        bm.boot()
        sm.start_all()
        duration = bm.shutdown()
        assert duration >= 0.0
        assert not bm.is_booted


# ─── StateManager tests ──────────────────────────────────────────────────────


class TestStateManager:
    """Tests for the StateManager."""

    def test_set_get(self) -> None:
        """Set and get state values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            sm = StateManager(bus, db_path=os.path.join(tmpdir, "state.db"))
            sm.set("key1", "value1")
            assert sm.get("key1") == "value1"
            assert sm.get("missing", "default") == "default"

    def test_checkpoint_and_recover(self) -> None:
        """Checkpoint saves and recover restores state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "state.db")
            bus = EventBus()
            sm1 = StateManager(bus, db_path=db_path)
            sm1.set("service_count", 5)
            sm1.set("boot_time", 1.23)
            ckpt_id = sm1.checkpoint()
            assert ckpt_id != ""

            # New manager recovers from same DB
            bus2 = EventBus()
            sm2 = StateManager(bus2, db_path=db_path)
            snapshot = sm2.recover()
            assert snapshot is not None
            assert sm2.get("service_count") == 5
            assert sm2.get("boot_time") == 1.23

    def test_recover_no_checkpoint(self) -> None:
        """Recover with no checkpoint returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            sm = StateManager(bus, db_path=os.path.join(tmpdir, "empty.db"))
            result = sm.recover()
            assert result is None

    def test_export_import(self) -> None:
        """Export and import state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            sm = StateManager(bus, db_path=os.path.join(tmpdir, "state.db"))
            sm.set("a", 1)
            sm.set("b", 2)
            exported = sm.export_state()
            assert exported == {"a": 1, "b": 2}

            sm.import_state({"c": 3})
            assert sm.get("c") == 3
            assert sm.get("a") is None

    def test_maybe_checkpoint(self) -> None:
        """Maybe_checkpoint only checkpoints after interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            sm = StateManager(
                bus,
                db_path=os.path.join(tmpdir, "state.db"),
                checkpoint_interval=10.0,
            )
            sm.set("x", 1)
            # First call should checkpoint (last_checkpoint is 0)
            ckpt1 = sm.maybe_checkpoint()
            assert ckpt1 is not None
            # Second call should not (interval not elapsed)
            ckpt2 = sm.maybe_checkpoint()
            assert ckpt2 is None

    def test_get_status(self) -> None:
        """Get status returns state info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus()
            sm = StateManager(bus, db_path=os.path.join(tmpdir, "state.db"))
            sm.set("x", 1)
            status = sm.get_status()
            assert status["state_keys"] == 1
            assert "db_path" in status


# ─── MonitorLoop tests ───────────────────────────────────────────────────────


class TestMonitorLoop:
    """Tests for the MonitorLoop."""

    def test_start_stop(self) -> None:
        """Monitor loop starts and stops."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(name="svc", start_fn=lambda: True, health_fn=lambda: (True, "ok")))
        sm.start_all()
        monitor = MonitorLoop(sm, bus, interval=0.1)
        monitor.start()
        time.sleep(0.3)
        assert monitor.get_health() is not None
        monitor.stop()

    def test_health_collection(self) -> None:
        """Health is collected from services."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(
            name="healthy_svc",
            start_fn=lambda: True,
            health_fn=lambda: (True, "ok"),
        ))
        sm.register(ServiceDescriptor(
            name="degraded_svc",
            start_fn=lambda: True,
            health_fn=lambda: (False, "degraded"),
        ))
        sm.start_all()
        monitor = MonitorLoop(sm, bus, interval=0.1)
        monitor.start()
        time.sleep(0.3)
        health = monitor.get_health()
        assert health is not None
        assert health.alert_count == 1
        monitor.stop()

    def test_get_status(self) -> None:
        """Get status returns monitor info."""
        bus = EventBus()
        sm = ServiceManager(bus)
        monitor = MonitorLoop(sm, bus, interval=2.0)
        status = monitor.get_status()
        assert status["running"] is False
        assert status["interval"] == 2.0


# ─── AIOSKernel integration tests ────────────────────────────────────────────


class TestAIOSKernel:
    """Integration tests for the AIOSKernel."""

    def test_boot_shutdown(self) -> None:
        """Full boot and shutdown cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AIOSKernel(
                state_db_path=os.path.join(tmpdir, "os_state.db"),
                monitor_interval=10.0,  # Slow to avoid thread during test
            )
            # Register a service
            kernel.register_service(ServiceDescriptor(
                name="test_svc",
                start_fn=lambda: True,
                stop_fn=lambda: None,
                health_fn=lambda: (True, "ok"),
            ))
            # Register a boot stage
            kernel.register_boot_stage(BootStage.INIT, lambda: (True, "init ok"))

            progress = kernel.boot()
            assert kernel.is_booted
            assert progress.progress_pct == 100.0

            # Check service is running
            svc = kernel.service_manager.get("test_svc")
            assert svc.state == ServiceState.RUNNING

            # Get status
            status = kernel.get_status()
            assert status["booted"] is True
            assert len(status["services"]) == 1

            # Shutdown
            duration = kernel.shutdown()
            assert duration >= 0.0
            assert not kernel.is_booted

    def test_events_published(self) -> None:
        """Boot publishes events to the event bus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kernel = AIOSKernel(
                state_db_path=os.path.join(tmpdir, "os_state.db"),
                monitor_interval=10.0,
            )
            kernel.register_boot_stage(BootStage.INIT, lambda: (True, "ok"))
            kernel.boot()
            events = kernel.get_events(event_type=EventType.BOOT)
            assert len(events) > 0
            kernel.shutdown()

    def test_state_persistence(self) -> None:
        """State persists across boot cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "os_state.db")
            kernel1 = AIOSKernel(state_db_path=db_path, monitor_interval=10.0)
            kernel1.register_boot_stage(BootStage.INIT, lambda: (True, "ok"))
            kernel1.boot()
            kernel1.state_manager.set("custom_key", "custom_value")
            kernel1.state_manager.checkpoint()
            kernel1.shutdown()

            # New kernel should recover state
            kernel2 = AIOSKernel(state_db_path=db_path, monitor_interval=10.0)
            kernel2.state_manager.recover()
            assert kernel2.state_manager.get("custom_key") == "custom_value"


# ─── TaskQueue tests ─────────────────────────────────────────────────────────


class TestTaskQueue:
    """Tests for the TaskQueue."""

    def test_enqueue_and_process(self) -> None:
        """Task is enqueued and processed by worker."""
        bus = EventBus()
        tq = TaskQueue(bus)
        results: list[str] = []
        task = Task(
            task_id="t1",
            name="test_task",
            fn=lambda: results.append("done"),
        )
        tq.enqueue(task)
        tq.start()
        time.sleep(0.5)
        tq.stop()
        assert results == ["done"]

    def test_priority_ordering(self) -> None:
        """Higher priority tasks execute first."""
        bus = EventBus()
        tq = TaskQueue(bus)
        order: list[str] = []
        tq.enqueue(Task(task_id="low", name="low", fn=lambda: order.append("low"), priority=200))
        tq.enqueue(Task(task_id="high", name="high", fn=lambda: order.append("high"), priority=10))
        tq.start()
        time.sleep(0.5)
        tq.stop()
        assert order[0] == "high"

    def test_retry_on_failure(self) -> None:
        """Failed tasks retry up to max_retries."""
        bus = EventBus()
        tq = TaskQueue(bus)
        attempts: list[int] = []
        def flaky() -> None:
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("not yet")
        task = Task(
            task_id="retry_task",
            name="retry",
            fn=flaky,
            max_retries=3,
            retry_delay_base=0.01,
        )
        tq.enqueue(task)
        tq.start()
        time.sleep(1.0)
        tq.stop()
        assert len(attempts) >= 2

    def test_cancel_queued(self) -> None:
        """Queued task can be cancelled."""
        bus = EventBus()
        tq = TaskQueue(bus)
        task = Task(task_id="cancel_me", name="cancel", fn=lambda: None)
        tq.enqueue(task)
        assert tq.cancel("cancel_me") is True
        assert tq.cancel("nonexistent") is False

    def test_scheduled_task(self) -> None:
        """Scheduled tasks run at intervals."""
        bus = EventBus()
        tq = TaskQueue(bus)
        counter: list[int] = []
        tq.schedule("periodic", lambda: counter.append(1), interval=0.1)
        tq.start()
        time.sleep(1.5)
        tq.stop()
        assert len(counter) >= 2

    def test_get_status(self) -> None:
        """Get status returns queue info."""
        bus = EventBus()
        tq = TaskQueue(bus)
        tq.enqueue(Task(task_id="t", name="t", fn=lambda: None))
        status = tq.get_status()
        assert status["pending_count"] == 1

    def test_get_pending(self) -> None:
        """Get pending returns queued tasks."""
        bus = EventBus()
        tq = TaskQueue(bus)
        tq.enqueue(Task(task_id="t1", name="task1", fn=lambda: None, priority=10))
        tq.enqueue(Task(task_id="t2", name="task2", fn=lambda: None, priority=20))
        pending = tq.get_pending()
        assert len(pending) == 2
        assert pending[0]["task_id"] == "t1"


# ─── SelfRepairManager tests ─────────────────────────────────────────────────


class TestSelfRepairManager:
    """Tests for the SelfRepairManager."""

    def test_record_success_resets(self) -> None:
        """Success resets failure count."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sr = SelfRepairManager(sm, bus)
        sr.register_breaker("svc", failure_threshold=3)
        sr.record_failure("svc")
        sr.record_failure("svc")
        assert sr._breakers["svc"].consecutive_failures == 2
        sr.record_success("svc")
        assert sr._breakers["svc"].consecutive_failures == 0

    def test_circuit_opens_on_threshold(self) -> None:
        """Circuit opens after threshold failures."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sr = SelfRepairManager(sm, bus)
        sr.register_breaker("svc", failure_threshold=3)
        sr.record_failure("svc")
        sr.record_failure("svc")
        assert sr._breakers["svc"].state == CircuitState.CLOSED
        sr.record_failure("svc")
        assert sr._breakers["svc"].state == CircuitState.OPEN
        assert sr._breakers["svc"].total_trips == 1

    def test_circuit_half_open_after_cooldown(self) -> None:
        """Circuit goes half-open after cooldown."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sr = SelfRepairManager(sm, bus)
        sr.register_breaker("svc", failure_threshold=2, cooldown_seconds=0.1)
        sr.record_failure("svc")
        sr.record_failure("svc")
        assert sr._breakers["svc"].state == CircuitState.OPEN
        time.sleep(0.15)
        # Trigger check via check_and_recover (no service registered, so no restart)
        sr.check_and_recover()
        assert sr._breakers["svc"].state == CircuitState.HALF_OPEN

    def test_circuit_closes_on_success(self) -> None:
        """Circuit closes when half-open and success recorded."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sr = SelfRepairManager(sm, bus)
        sr.register_breaker("svc", failure_threshold=2, cooldown_seconds=0.05)
        sr.record_failure("svc")
        sr.record_failure("svc")
        time.sleep(0.06)
        sr.check_and_recover()
        assert sr._breakers["svc"].state == CircuitState.HALF_OPEN
        sr.record_success("svc")
        assert sr._breakers["svc"].state == CircuitState.CLOSED

    def test_auto_restart_failed_service(self) -> None:
        """Failed service gets auto-restarted."""
        bus = EventBus()
        sm = ServiceManager(bus)
        starts: list[str] = []
        sm.register(ServiceDescriptor(
            name="svc",
            start_fn=lambda: (starts.append("start"), True)[1],
            stop_fn=lambda: None,
            auto_restart=True,
            max_restarts=3,
        ))
        sm.start_all()
        # Simulate failure
        sm.get("svc").state = ServiceState.FAILED
        sr = SelfRepairManager(sm, bus)
        actions = sr.check_and_recover()
        assert len(actions) == 1
        assert actions[0].service_name == "svc"
        assert actions[0].action == "restart"
        assert sm.get("svc").state == ServiceState.RUNNING

    def test_get_breaker_status(self) -> None:
        """Get breaker status returns all breakers."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sr = SelfRepairManager(sm, bus)
        sr.register_breaker("svc1")
        sr.register_breaker("svc2")
        status = sr.get_breaker_status()
        assert len(status) == 2

    def test_get_recovery_log(self) -> None:
        """Recovery log records actions."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sm.register(ServiceDescriptor(
            name="svc",
            start_fn=lambda: True,
            stop_fn=lambda: None,
            auto_restart=True,
        ))
        sm.start_all()
        sm.get("svc").state = ServiceState.FAILED
        sr = SelfRepairManager(sm, bus)
        sr.check_and_recover()
        log = sr.get_recovery_log()
        assert len(log) == 1
        assert log[0]["service_name"] == "svc"

    def test_get_status(self) -> None:
        """Get status returns self-repair info."""
        bus = EventBus()
        sm = ServiceManager(bus)
        sr = SelfRepairManager(sm, bus)
        sr.register_breaker("svc1")
        sr.register_breaker("svc2")
        sr.record_failure("svc1")
        sr.record_failure("svc1")
        sr.record_failure("svc1")  # Opens circuit
        status = sr.get_status()
        assert status["breaker_count"] == 2
        assert status["open_circuits"] == 1


# ─── KnowledgeHotReloader tests ──────────────────────────────────────────────


class TestKnowledgeHotReloader:
    """Tests for the KnowledgeHotReloader."""

    def test_scan_detects_new_files(self) -> None:
        """New files are detected and ingested on scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a domain directory with a knowledge file
            domain_dir = os.path.join(tmpdir, "reasoning")
            os.makedirs(domain_dir)
            file_path = os.path.join(domain_dir, "test.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("This is a test chunk about reasoning.\n\nSecond chunk about logic.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            items = reloader.scan_once()
            assert items == 2
            assert backend.count() == 2

    def test_scan_skips_unchanged_files(self) -> None:
        """Unchanged files are not re-ingested on second scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "logic")
            os.makedirs(domain_dir)
            file_path = os.path.join(domain_dir, "data.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("First chunk about logical reasoning.\n\nSecond chunk about predicates.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            # First scan — ingests
            items1 = reloader.scan_once()
            assert items1 == 2

            # Second scan — no changes
            items2 = reloader.scan_once()
            assert items2 == 0
            assert backend.count() == 2

    def test_scan_detects_modified_files(self) -> None:
        """Modified files are re-ingested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "ethics")
            os.makedirs(domain_dir)
            file_path = os.path.join(domain_dir, "moral.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Initial content about moral philosophy.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            # First scan
            items1 = reloader.scan_once()
            assert items1 == 1

            # Modify file
            time.sleep(0.1)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Updated content about moral philosophy and ethics.\n\nNew chunk about virtue ethics.\n\n")

            # Second scan — detects modification
            items2 = reloader.scan_once()
            assert items2 == 2

    def test_scan_removes_deleted_from_cache(self) -> None:
        """Deleted files are removed from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "culture")
            os.makedirs(domain_dir)
            file_path = os.path.join(domain_dir, "info.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Cultural knowledge chunk one.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            reloader.scan_once()
            assert len(reloader.get_tracked_files()) == 1

            # Delete file
            os.remove(file_path)
            reloader.scan_once()

            assert len(reloader.get_tracked_files()) == 0

    def test_prime_cache(self) -> None:
        """Prime cache populates file records without ingesting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "language")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "f1.txt"), "w", encoding="utf-8") as f:
                f.write("Language chunk one.\n\n")
            with open(os.path.join(domain_dir, "f2.txt"), "w", encoding="utf-8") as f:
                f.write("Language chunk two.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            primed = reloader.prime_cache()
            assert primed == 2
            assert backend.count() == 0  # No ingestion during prime

            # Subsequent scan should not ingest (already primed)
            items = reloader.scan_once()
            assert items == 0

    def test_ignores_downloads_and_meta(self) -> None:
        """Downloads and meta directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "downloads"))
            os.makedirs(os.path.join(tmpdir, "meta"))
            os.makedirs(os.path.join(tmpdir, "reasoning"))

            with open(os.path.join(tmpdir, "downloads", "skip.txt"), "w", encoding="utf-8") as f:
                f.write("This should be skipped.\n\n")
            with open(os.path.join(tmpdir, "meta", "skip.txt"), "w", encoding="utf-8") as f:
                f.write("This should also be skipped.\n\n")
            with open(os.path.join(tmpdir, "reasoning", "load.txt"), "w", encoding="utf-8") as f:
                f.write("This should be loaded.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            items = reloader.scan_once()
            assert items == 1
            assert backend.count() == 1

    def test_get_status(self) -> None:
        """Status returns correct info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "dialog")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "d.txt"), "w", encoding="utf-8") as f:
                f.write("Dialog chunk about conversation.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=15.0)

            reloader.scan_once()
            status = reloader.get_status()
            assert status["running"] is False
            assert status["interval"] == 15.0
            assert status["tracked_files"] == 1
            assert status["total_reloads"] == 1
            assert status["total_items_reloaded"] == 1

    def test_start_stop(self) -> None:
        """Start and stop control the background thread."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=0.2)

            reloader.start()
            assert reloader.get_status()["running"] is True

            time.sleep(0.5)
            reloader.stop()
            assert reloader.get_status()["running"] is False

    def test_publishes_events_on_reload(self) -> None:
        """KNOWLEDGE events are published on reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "world_knowledge")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "wk.txt"), "w", encoding="utf-8") as f:
                f.write("World knowledge fact one.\n\nWorld knowledge fact two.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            reloader.scan_once()

            events = bus.get_log(event_type=EventType.KNOWLEDGE)
            assert len(events) >= 1  # At least reload complete event
            assert any("new items" in e.message for e in events)

    def test_nonexistent_root(self) -> None:
        """Nonexistent root returns 0 items."""
        backend = KnowledgeBackend()
        bus = EventBus()
        reloader = KnowledgeHotReloader("/nonexistent/path", backend, bus, interval=999)

        assert reloader.scan_once() == 0
        assert backend.count() == 0

    def test_ignores_non_text_files(self) -> None:
        """Non-text files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_dir = os.path.join(tmpdir, "common_sense")
            os.makedirs(domain_dir)
            with open(os.path.join(domain_dir, "image.bin"), "wb") as f:
                f.write(b"\x00\x01\x02\x03")
            with open(os.path.join(domain_dir, "text.txt"), "w", encoding="utf-8") as f:
                f.write("Common sense knowledge chunk.\n\n")

            backend = KnowledgeBackend()
            bus = EventBus()
            reloader = KnowledgeHotReloader(tmpdir, backend, bus, interval=999)

            items = reloader.scan_once()
            assert items == 1  # Only the .txt file


# ─── DataFlowManager tests ───────────────────────────────────────────────────


class TestDataChannel:
    """Tests for the DataChannel."""

    def test_put_get(self) -> None:
        """Basic put and get works."""
        ch = DataChannel("test", capacity=10)
        assert ch.put("hello")
        assert ch.get(timeout=0.5) == "hello"

    def test_put_nowait_get_nowait(self) -> None:
        """Non-blocking put and get works."""
        ch = DataChannel("test", capacity=5)
        assert ch.put_nowait("item1")
        assert ch.put_nowait("item2")
        assert ch.get_nowait() == "item1"
        assert ch.get_nowait() == "item2"
        assert ch.get_nowait() is None

    def test_capacity_drops(self) -> None:
        """Non-blocking put drops when full."""
        ch = DataChannel("test", capacity=2)
        assert ch.put_nowait("a")
        assert ch.put_nowait("b")
        assert ch.put_nowait("c") is False  # Dropped
        metrics = ch.get_metrics()
        assert metrics.total_dropped == 1

    def test_blocking_get_timeout(self) -> None:
        """Get returns None on timeout."""
        ch = DataChannel("test", capacity=5)
        assert ch.get(timeout=0.1) is None

    def test_blocking_put_timeout(self) -> None:
        """Put returns False on timeout when full."""
        ch = DataChannel("test", capacity=1)
        ch.put("first")
        assert ch.put("second", timeout=0.1) is False

    def test_close(self) -> None:
        """Closed channel rejects puts."""
        ch = DataChannel("test", capacity=5)
        ch.put("item")
        ch.close()
        assert ch.state == ChannelState.CLOSED
        assert ch.put("new") is False
        # Can still get remaining items
        assert ch.get(timeout=0.1) == "item"

    def test_pause_resume(self) -> None:
        """Paused channel rejects puts, resumed works."""
        ch = DataChannel("test", capacity=5)
        ch.pause()
        assert ch.state == ChannelState.PAUSED
        assert ch.put("item", timeout=0.1) is False
        ch.resume()
        assert ch.state == ChannelState.OPEN
        assert ch.put("item")

    def test_metrics(self) -> None:
        """Metrics track put/get/dropped."""
        ch = DataChannel("test", capacity=3)
        ch.put_nowait("a")
        ch.put_nowait("b")
        ch.get_nowait()
        ch.put_nowait("c")
        ch.put_nowait("d")
        ch.put_nowait("e")  # This should be dropped (capacity=3, 2 in buffer)
        metrics = ch.get_metrics()
        assert metrics.channel_name == "test"
        assert metrics.total_put == 4
        assert metrics.total_get == 1
        assert metrics.max_buffer_size == 3


class TestDataFlowManager:
    """Tests for the DataFlowManager."""

    def test_create_channel(self) -> None:
        """Channels can be created and retrieved."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        ch = dfm.create_channel("input", capacity=50)
        assert ch.name == "input"
        assert dfm.get_channel("input") is ch

    def test_duplicate_channel_raises(self) -> None:
        """Duplicate channel name raises ValueError."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("test")
        with pytest.raises(ValueError, match="already exists"):
            dfm.create_channel("test")

    def test_put_get_via_manager(self) -> None:
        """Put and get through manager works."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("data")
        assert dfm.put("data", "hello")
        assert dfm.get("data", timeout=0.5) == "hello"

    def test_put_nonexistent_channel(self) -> None:
        """Put to non-existent channel returns False."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        assert dfm.put("nonexistent", "item") is False

    def test_remove_channel(self) -> None:
        """Channels can be removed."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("temp")
        assert dfm.remove_channel("temp") is True
        assert dfm.get_channel("temp") is None
        assert dfm.remove_channel("temp") is False

    def test_routing(self) -> None:
        """Routing rules forward items between channels."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("source")
        dfm.create_channel("dest")
        dfm.add_route("source", "dest")

        dfm.put_nowait("source", "item1")
        dfm.put_nowait("source", "item2")

        routed = dfm.route_once()
        assert routed == 2
        assert dfm.get_nowait("dest") == "item1"
        assert dfm.get_nowait("dest") == "item2"

    def test_routing_with_predicate(self) -> None:
        """Routing predicate filters items."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("input")
        dfm.create_channel("output")
        dfm.add_route("input", "output", predicate=lambda x: x > 10)

        dfm.put_nowait("input", 5)
        dfm.put_nowait("input", 15)
        dfm.put_nowait("input", 20)

        routed = dfm.route_once()
        assert routed == 2  # Only 15 and 20 pass predicate
        assert dfm.get_nowait("output") == 15
        assert dfm.get_nowait("output") == 20

    def test_routing_destination_full(self) -> None:
        """Routing stops when destination is full."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("src")
        dfm.create_channel("dst", capacity=1)
        dfm.add_route("src", "dst")

        dfm.put_nowait("src", "a")
        dfm.put_nowait("src", "b")

        routed = dfm.route_once()
        assert routed == 1  # Only one fits in dst
        # Source still has "b"
        assert dfm.get_nowait("src") == "b"

    def test_remove_route(self) -> None:
        """Routes can be removed."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("a")
        dfm.create_channel("b")
        dfm.add_route("a", "b")
        assert dfm.remove_route("a", "b") is True
        assert dfm.remove_route("a", "b") is False

    def test_get_all_metrics(self) -> None:
        """Get all channel metrics."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("ch1")
        dfm.create_channel("ch2")
        dfm.put_nowait("ch1", "x")

        metrics = dfm.get_all_metrics()
        assert len(metrics) == 2
        names = {m.channel_name for m in metrics}
        assert names == {"ch1", "ch2"}

    def test_get_status(self) -> None:
        """Status returns correct info."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("ch1", capacity=10)
        dfm.create_channel("ch2", capacity=20)
        dfm.add_route("ch1", "ch2")

        status = dfm.get_status()
        assert status["channels"] == 2
        assert status["routing_rules"] == 1
        assert status["router_running"] is False

    def test_start_stop_router(self) -> None:
        """Router thread starts and stops."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("src")
        dfm.create_channel("dst")
        dfm.add_route("src", "dst")

        dfm.put_nowait("src", "item")
        dfm.start_router(interval=0.1)

        time.sleep(0.3)
        assert dfm.get_status()["router_running"] is True

        # Item should have been routed
        assert dfm.get_nowait("dst") == "item"

        dfm.stop_router()
        assert dfm.get_status()["router_running"] is False

    def test_close_all(self) -> None:
        """Close all clears channels and stops router."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("a")
        dfm.create_channel("b")
        dfm.start_router(interval=1.0)

        dfm.close_all()
        assert len(dfm.get_channel_names()) == 0
        assert dfm.get_status()["router_running"] is False

    def test_channel_names(self) -> None:
        """Get channel names returns all names."""
        bus = EventBus()
        dfm = DataFlowManager(bus)
        dfm.create_channel("alpha")
        dfm.create_channel("beta")
        names = set(dfm.get_channel_names())
        assert names == {"alpha", "beta"}


# ─── BrainController + AIOSKernel integration tests ──────────────────────────


class TestBrainControllerAIOSIntegration:
    """Integration tests for BrainController + AIOSKernel wiring."""

    def test_boot_creates_aios_kernel(self) -> None:
        """boot_system creates and boots the AIOSKernel."""
        from core.brain.brain_controller import BrainController
        bc = BrainController()
        assert bc.aios_kernel is None  # Not created until boot
        bc.boot_system()
        assert bc.aios_kernel is not None
        assert bc.aios_kernel.is_booted is True
        bc.close()

    def test_close_shuts_down_aios(self) -> None:
        """close() shuts down the AIOSKernel."""
        from core.brain.brain_controller import BrainController
        bc = BrainController()
        bc.boot_system()
        assert bc.aios_kernel is not None
        assert bc.aios_kernel.is_booted is True
        bc.close()
        assert bc.aios_kernel.is_booted is False

    def test_aios_status_accessible(self) -> None:
        """AIOSKernel status is accessible after boot."""
        from core.brain.brain_controller import BrainController
        bc = BrainController()
        bc.boot_system()
        status = bc.aios_kernel.get_status()
        assert status["booted"] is True
        assert "services" in status
        assert "health" in status
        assert "events" in status
        assert "data_flow" in status
        bc.close()

    def test_aios_events_logged_on_boot(self) -> None:
        """Boot events are logged in the event bus."""
        from core.brain.brain_controller import BrainController
        bc = BrainController()
        bc.boot_system()
        events = bc.aios_kernel.get_events(event_type=EventType.BOOT)
        assert len(events) >= 2  # At least boot initiated + boot complete
        bc.close()

    def test_aios_data_flow_has_channels_after_boot(self) -> None:
        """Data flow manager is active after boot."""
        from core.brain.brain_controller import BrainController
        bc = BrainController()
        bc.boot_system()
        df_status = bc.aios_kernel.data_flow.get_status()
        assert df_status["router_running"] is True
        bc.close()

    def test_process_request_works_with_aios(self) -> None:
        """process_request works normally with AIOSKernel active."""
        from core.brain.brain_controller import BrainController
        bc = BrainController()
        bc.boot_system()
        result = bc.process_request("Hello, what is 2+2?")
        assert result.final_output  # Non-empty response
        assert bc.aios_kernel.is_booted is True
        bc.close()
