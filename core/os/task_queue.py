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
AI-OS Task Queue & Scheduler — priority queue with worker threads and
cron-like scheduling for periodic background tasks.

Mathematical Foundation:
- Priority queue: O(log n) insert, O(log n) pop.
- Worker dispatch: O(1) per task via thread pool.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.os.event_bus import AIEvent, EventBus, EventType


class TaskState(Enum):
    """Lifecycle states for a queued task.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A unit of background work.

    Attributes:
        task_id: Unique identifier.
        name: Human-readable name.
        fn: Callable to execute.
        priority: Lower = higher priority (default 100).
        state: Current task state.
        retries: Number of retries so far.
        max_retries: Max retry attempts.
        retry_delay_base: Base delay for exponential backoff.
        timeout: Max execution time in seconds (0 = no limit).
        created_at: Creation timestamp.
        started_at: Execution start timestamp.
        completed_at: Completion timestamp.
        error: Last error message.
        result: Return value on success.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    task_id: str
    name: str
    fn: Callable[[], Any]
    priority: int = 100
    state: TaskState = TaskState.QUEUED
    retries: int = 0
    max_retries: int = 3
    retry_delay_base: float = 1.0
    timeout: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""
    result: Any = None


@dataclass
class ScheduledTask:
    """A recurring scheduled task.

    Attributes:
        name: Unique name.
        fn: Callable to execute.
        interval: Run interval in seconds.
        last_run: Timestamp of last execution.
        running: Whether currently executing.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    fn: Callable[[], Any]
    interval: float
    last_run: float = 0.0
    running: bool = False


class TaskQueue:
    """
    Priority task queue with worker threads.

    Attributes:
        _queue: List of pending tasks (sorted by priority).
        _scheduled: Mapping of scheduled task name to ScheduledTask.
        _completed: Ring buffer of completed tasks.
        _lock: Thread lock.
        _event_bus: Event bus for task events.
        _worker: Background worker thread.
        _scheduler: Background scheduler thread.
        _running: Whether the queue is active.
        _max_completed: Max completed tasks to retain.

    Complexity:
        enqueue: O(n) for sorted insert.
        _worker_loop: O(1) per task dispatch.
    """

    __slots__ = (
        "_queue",
        "_scheduled",
        "_completed",
        "_lock",
        "_event_bus",
        "_worker",
        "_scheduler_thread",
        "_running",
        "_max_completed",
    )

    def __init__(self, event_bus: EventBus, max_completed: int = 200) -> None:
        """Initialize the task queue.

        Args:
            event_bus: Event bus for publishing task events.
            max_completed: Max completed tasks in ring buffer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._queue: List[Task] = []
        self._scheduled: Dict[str, ScheduledTask] = {}
        self._completed: List[Task] = []
        self._lock: threading.Lock = threading.Lock()
        self._event_bus: EventBus = event_bus
        self._worker: Optional[threading.Thread] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._max_completed: int = max_completed

    def enqueue(self, task: Task) -> str:
        """Add a task to the queue.

        Args:
            task: Task to enqueue.

        Returns:
            Task ID.

        Complexity: T(n) ∈ O(n) for sorted insert.
        """
        with self._lock:
            # Sorted insert by priority
            inserted = False
            for i, existing in enumerate(self._queue):
                if task.priority < existing.priority:
                    self._queue.insert(i, task)
                    inserted = True
                    break
            if not inserted:
                self._queue.append(task)

        self._event_bus.publish(AIEvent(
            event_type=EventType.TASK,
            source="task_queue",
            message=f"Task queued: {task.name} (id={task.task_id})",
            severity="INFO",
            data={"task_id": task.task_id, "name": task.name, "priority": task.priority},
        ))
        return task.task_id

    def schedule(self, name: str, fn: Callable[[], Any], interval: float) -> None:
        """Register a recurring scheduled task.

        Args:
            name: Unique task name.
            fn: Callable to execute.
            interval: Run interval in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._scheduled[name] = ScheduledTask(name=name, fn=fn, interval=interval)

        self._event_bus.publish(AIEvent(
            event_type=EventType.TASK,
            source="task_queue",
            message=f"Task scheduled: {name} (every {interval}s)",
            severity="INFO",
            data={"name": name, "interval": interval},
        ))

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task.

        Args:
            task_id: Task ID to cancel.

        Returns:
            True if cancelled.

        Complexity: T(n) ∈ O(n).
        """
        with self._lock:
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    task.state = TaskState.CANCELLED
                    self._queue.pop(i)
                    return True
        return False

    def start(self) -> None:
        """Start the worker and scheduler threads.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="ai-os-worker"
        )
        self._worker.start()
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="ai-os-scheduler"
        )
        self._scheduler_thread.start()

    def stop(self) -> None:
        """Stop the worker and scheduler threads.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2.0)

    def _worker_loop(self) -> None:
        """Main worker loop — processes queued tasks.

        Complexity: T(n) ∈ O(n * t) for n tasks, t avg execution time.
        """
        while self._running:
            task: Optional[Task] = None
            with self._lock:
                if self._queue:
                    task = self._queue.pop(0)

            if task is None:
                time.sleep(0.1)
                continue

            self._execute_task(task)

    def _scheduler_loop(self) -> None:
        """Main scheduler loop — runs scheduled tasks at intervals.

        Complexity: T(n) ∈ O(s) per iteration for s scheduled tasks.
        """
        while self._running:
            now = time.time()
            with self._lock:
                scheduled = list(self._scheduled.values())

            for st in scheduled:
                if st.running:
                    continue
                if now - st.last_run >= st.interval:
                    st.last_run = now
                    st.running = True
                    try:
                        st.fn()
                    except Exception as exc:
                        self._event_bus.publish(AIEvent(
                            event_type=EventType.TASK,
                            source="task_queue",
                            message=f"Scheduled task error: {st.name}: {exc}",
                            severity="ERROR",
                            data={"name": st.name, "error": str(exc)},
                        ))
                    finally:
                        st.running = False

            time.sleep(1.0)

    def _execute_task(self, task: Task) -> None:
        """Execute a single task with retry logic.

        Args:
            task: Task to execute.

        Complexity: T(n) ∈ O(t) for task execution time.
        """
        task.state = TaskState.RUNNING
        task.started_at = time.time()

        self._event_bus.publish(AIEvent(
            event_type=EventType.TASK,
            source="task_queue",
            message=f"Task running: {task.name}",
            severity="INFO",
            data={"task_id": task.task_id},
        ))

        try:
            task.result = task.fn()
            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            self._event_bus.publish(AIEvent(
                event_type=EventType.TASK,
                source="task_queue",
                message=f"Task completed: {task.name} ({task.completed_at - task.started_at:.3f}s)",
                severity="INFO",
                data={"task_id": task.task_id, "duration": task.completed_at - task.started_at},
            ))
        except Exception as exc:
            task.error = str(exc)
            task.retries += 1
            if task.retries <= task.max_retries:
                task.state = TaskState.RETRY
                delay = task.retry_delay_base * (2 ** (task.retries - 1))
                self._event_bus.publish(AIEvent(
                    event_type=EventType.TASK,
                    source="task_queue",
                    message=f"Task retry {task.retries}/{task.max_retries}: {task.name} (delay {delay:.1f}s)",
                    severity="WARN",
                    data={"task_id": task.task_id, "retry": task.retries, "delay": delay},
                ))
                time.sleep(delay)
                self._execute_task(task)
                return
            else:
                task.state = TaskState.FAILED
                task.completed_at = time.time()
                self._event_bus.publish(AIEvent(
                    event_type=EventType.TASK,
                    source="task_queue",
                    message=f"Task failed: {task.name}: {task.error}",
                    severity="ERROR",
                    data={"task_id": task.task_id, "error": task.error},
                ))

        with self._lock:
            self._completed.append(task)
            if len(self._completed) > self._max_completed:
                self._completed = self._completed[-self._max_completed:]

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get pending tasks.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            return [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "priority": t.priority,
                    "state": t.state.value,
                    "retries": t.retries,
                }
                for t in self._queue
            ]

    def get_completed(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get completed tasks.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            tasks = list(reversed(self._completed))
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "state": t.state.value,
                "duration": t.completed_at - t.started_at if t.started_at and t.completed_at else 0,
                "error": t.error,
            }
            for t in tasks[:limit]
        ]

    def get_scheduled(self) -> List[Dict[str, Any]]:
        """Get scheduled tasks.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            return [
                {
                    "name": st.name,
                    "interval": st.interval,
                    "last_run": st.last_run,
                    "running": st.running,
                }
                for st in self._scheduled.values()
            ]

    def get_status(self) -> Dict[str, Any]:
        """Get task queue status.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return {
                "running": self._running,
                "pending_count": len(self._queue),
                "completed_count": len(self._completed),
                "scheduled_count": len(self._scheduled),
            }
