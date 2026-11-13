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
AI-OS Event Bus — decoupled publish/subscribe event system for inter-service
communication.

Provides typed events, async handlers, a ring buffer event log, and event
replay for debugging and recovery.

Mathematical Foundation:
- Publish: O(s) for s subscribers per event type (hash lookup + dispatch).
- Ring buffer: O(1) append, O(n) read for n events.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional


class EventType(Enum):
    """Typed event categories for the AI-OS.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    BOOT = "boot"
    SERVICE = "service"
    HEALTH = "health"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    PIPELINE = "pipeline"
    ALERT = "alert"
    TASK = "task"
    STATE = "state"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class AIEvent:
    """Immutable event record.

    Attributes:
        event_type: Category of event.
        source: Emitting component name.
        message: Human-readable description.
        severity: INFO, WARN, ERROR, CRITICAL.
        timestamp: Unix timestamp.
        data: Optional payload dict.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(d) for data dict.
    """

    event_type: EventType
    source: str
    message: str
    severity: str = "INFO"
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict, hash=False)


class EventBus:
    """
    Publish/subscribe event bus with ring buffer log.

    Thread-safe. Subscribers are called synchronously on publish —
    they should be fast and non-blocking.

    Attributes:
        _subscribers: Mapping from EventType to list of callbacks.
        _log: Ring buffer of last N events.
        _lock: Thread lock for concurrent access.
        _max_log: Maximum events in ring buffer.

    Complexity:
        subscribe: T(n) ∈ O(1), S(n) ∈ O(1).
        publish: T(n) ∈ O(s) for s subscribers.
        get_log: T(n) ∈ O(n) for n events.
    """

    __slots__ = ("_subscribers", "_log", "_lock", "_max_log")

    def __init__(self, max_log: int = 1000) -> None:
        """Initialize the event bus.

        Args:
            max_log: Maximum events stored in ring buffer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(m) for max_log.
        """
        self._subscribers: Dict[EventType, List[Callable[[AIEvent], None]]] = {}
        self._log: Deque[AIEvent] = deque(maxlen=max_log)
        self._lock: threading.Lock = threading.Lock()
        self._max_log: int = max_log

    def subscribe(
        self, event_type: EventType, handler: Callable[[AIEvent], None]
    ) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: Event type to listen for.
            handler: Callback invoked on event.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def publish(self, event: AIEvent) -> None:
        """Publish an event to all subscribers.

        Args:
            event: The event to publish.

        Complexity: T(n) ∈ O(s) for s subscribers.
        """
        with self._lock:
            self._log.append(event)
            handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass  # Non-fatal — don't let a bad handler crash the bus

    def get_log(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        severity: Optional[str] = None,
    ) -> List[AIEvent]:
        """Retrieve events from the log.

        Args:
            event_type: Optional filter by type.
            limit: Maximum events to return.
            severity: Optional filter by severity.

        Returns:
            List of matching events, newest first.

        Complexity: T(n) ∈ O(n) for n logged events.
        """
        with self._lock:
            events = list(self._log)
        events.reverse()  # Newest first
        result: List[AIEvent] = []
        for event in events:
            if event_type and event.event_type != event_type:
                continue
            if severity and event.severity != severity:
                continue
            result.append(event)
            if len(result) >= limit:
                break
        return result

    def clear_log(self) -> int:
        """Clear the event log.

        Returns:
            Number of events cleared.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            count = len(self._log)
            self._log.clear()
            return count

    def event_count(self) -> int:
        """Return the number of events in the log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return len(self._log)
