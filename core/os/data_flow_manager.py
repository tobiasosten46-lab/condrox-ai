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
AI-OS Data Flow Manager — typed channels with backpressure and routing.

Provides typed data channels between AI-OS services for structured data
movement. Each channel has a bounded buffer with backpressure semantics
(blocking put when full, blocking get when empty). The manager tracks
flow metrics and supports routing rules.

Mathematical Foundation:
- Put: O(1) amortized (blocking with timeout).
- Get: O(1) amortized (blocking with timeout).
- Routing: O(r) for r routing rules per channel.
- Memory: O(Σ c_i * b) for c_i channels with b buffer capacity.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.os.event_bus import AIEvent, EventBus, EventType

logger = logging.getLogger(__name__)


class ChannelState(Enum):
    """State of a data flow channel."""

    OPEN = "open"
    CLOSED = "closed"
    PAUSED = "paused"


@dataclass
class FlowMetrics:
    """
    Immutable metrics snapshot for a single channel.

    Attributes:
        channel_name: Name of the channel.
        total_put: Total items put into the channel.
        total_get: Total items gotten from the channel.
        total_dropped: Items dropped due to full buffer (non-blocking put).
        current_buffer_size: Current number of items in buffer.
        max_buffer_size: Maximum buffer capacity.
        avg_latency_ms: Average put-to-get latency in milliseconds.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    channel_name: str
    total_put: int
    total_get: int
    total_dropped: int
    current_buffer_size: int
    max_buffer_size: int
    avg_latency_ms: float


@dataclass
class RoutingRule:
    """
    Immutable routing rule for channel forwarding.

    A routing rule forwards items from a source channel to a destination
    channel, optionally filtered by a predicate.

    Attributes:
        source: Source channel name.
        destination: Destination channel name.
        predicate: Optional filter function. If None, all items forwarded.
        priority: Routing priority (lower = higher priority).

    Complexity: T(n) ∈ O(p) for predicate evaluation p.
    """

    source: str
    destination: str
    predicate: Optional[Callable[[Any], bool]] = None
    priority: int = 0


class DataChannel:
    """
    Typed data channel with bounded buffer and backpressure.

    Uses a thread-safe queue with configurable capacity. When the buffer
    is full, `put` blocks (with timeout) or drops (non-blocking mode).
    When empty, `get` blocks (with timeout).

    Attributes:
        name: Channel name.
        capacity: Maximum buffer size.
        _buffer: Internal queue.
        _lock: Thread lock.
        _not_empty: Condition for non-empty buffer.
        _not_full: Condition for non-full buffer.
        _state: Channel state (OPEN, CLOSED, PAUSED).
        _total_put: Total items put.
        _total_get: Total items gotten.
        _total_dropped: Items dropped.
        _latency_sum: Sum of put-to-get latencies (ms).
        _latency_count: Number of latency samples.

    Complexity:
        put: O(1) amortized, blocking with timeout.
        get: O(1) amortized, blocking with timeout.
    """

    __slots__ = (
        "name",
        "capacity",
        "_buffer",
        "_lock",
        "_not_empty",
        "_not_full",
        "_state",
        "_total_put",
        "_total_get",
        "_total_dropped",
        "_latency_sum",
        "_latency_count",
    )

    def __init__(self, name: str, capacity: int = 100) -> None:
        """Initialize a data channel.

        Args:
            name: Channel name.
            capacity: Maximum buffer size.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(capacity).
        """
        self.name: str = name
        self.capacity: int = capacity
        self._buffer: List[Tuple[Any, float]] = []  # (item, put_timestamp)
        self._lock: threading.Lock = threading.Lock()
        self._not_empty: threading.Condition = threading.Condition(self._lock)
        self._not_full: threading.Condition = threading.Condition(self._lock)
        self._state: ChannelState = ChannelState.OPEN
        self._total_put: int = 0
        self._total_get: int = 0
        self._total_dropped: int = 0
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

    def put(self, item: Any, timeout: float = 1.0) -> bool:
        """Put an item into the channel (blocking with timeout).

        Args:
            item: Item to put.
            timeout: Maximum time to wait if buffer is full.

        Returns:
            True if item was put, False if timed out or channel closed.

        Complexity: T(n) ∈ O(1) amortized.
        """
        with self._not_full:
            if self._state == ChannelState.CLOSED:
                return False

            deadline = time.monotonic() + timeout
            while len(self._buffer) >= self.capacity and self._state == ChannelState.OPEN:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._not_full.wait(timeout=remaining)

            if self._state == ChannelState.CLOSED:
                return False
            if self._state == ChannelState.PAUSED:
                return False

            self._buffer.append((item, time.monotonic()))
            self._total_put += 1
            self._not_empty.notify()
            return True

    def put_nowait(self, item: Any) -> bool:
        """Put an item without blocking. Drops if full.

        Args:
            item: Item to put.

        Returns:
            True if put, False if dropped (buffer full or channel not open).

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            if self._state != ChannelState.OPEN:
                return False
            if len(self._buffer) >= self.capacity:
                self._total_dropped += 1
                return False
            self._buffer.append((item, time.monotonic()))
            self._total_put += 1
            self._not_empty.notify()
            return True

    def get(self, timeout: float = 1.0) -> Optional[Any]:
        """Get an item from the channel (blocking with timeout).

        Args:
            timeout: Maximum time to wait if buffer is empty.

        Returns:
            Item if available, None if timed out or channel closed.

        Complexity: T(n) ∈ O(1) amortized.
        """
        with self._not_empty:
            deadline = time.monotonic() + timeout
            while len(self._buffer) == 0 and self._state != ChannelState.CLOSED:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)

            if not self._buffer:
                return None

            item, put_time = self._buffer.pop(0)
            self._total_get += 1
            latency_ms = (time.monotonic() - put_time) * 1000.0
            self._latency_sum += latency_ms
            self._latency_count += 1
            self._not_full.notify()
            return item

    def get_nowait(self) -> Optional[Any]:
        """Get an item without blocking.

        Returns:
            Item if available, None if empty.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            if not self._buffer:
                return None
            item, put_time = self._buffer.pop(0)
            self._total_get += 1
            latency_ms = (time.monotonic() - put_time) * 1000.0
            self._latency_sum += latency_ms
            self._latency_count += 1
            self._not_full.notify()
            return item

    def close(self) -> None:
        """Close the channel. No more puts allowed. Gets drain remaining.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._state = ChannelState.CLOSED
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def pause(self) -> None:
        """Pause the channel. No puts allowed, gets still work.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._state = ChannelState.PAUSED

    def resume(self) -> None:
        """Resume a paused channel.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._state = ChannelState.OPEN

    @property
    def state(self) -> ChannelState:
        """Current channel state.

        Complexity: T(n) ∈ O(1).
        """
        return self._state

    @property
    def size(self) -> int:
        """Current buffer size.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return len(self._buffer)

    def get_metrics(self) -> FlowMetrics:
        """Get channel metrics snapshot.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            avg_latency = (
                self._latency_sum / self._latency_count
                if self._latency_count > 0
                else 0.0
            )
            return FlowMetrics(
                channel_name=self.name,
                total_put=self._total_put,
                total_get=self._total_get,
                total_dropped=self._total_dropped,
                current_buffer_size=len(self._buffer),
                max_buffer_size=self.capacity,
                avg_latency_ms=round(avg_latency, 3),
            )


class DataFlowManager:
    """
    Manages data flow channels with routing and backpressure.

    Provides a registry of named channels, routing rules for forwarding
    between channels, and aggregate metrics. Publishes flow events via
    the event bus.

    Attributes:
        _channels: Mapping from channel name to DataChannel.
        _routing_rules: List of routing rules sorted by priority.
        _event_bus: Event bus for publishing flow events.
        _lock: Thread lock for registry access.
        _router_thread: Background router thread.
        _running: Whether the router is active.

    Complexity:
        create_channel: O(1).
        put: O(1) amortized.
        get: O(1) amortized.
        add_route: O(r) for r existing rules (insertion sort).
        _route_all: O(c * r) for c channels with items, r routing rules.
    """

    __slots__ = (
        "_channels",
        "_routing_rules",
        "_event_bus",
        "_lock",
        "_router_thread",
        "_running",
    )

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize the data flow manager.

        Args:
            event_bus: Event bus for publishing flow events.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._channels: Dict[str, DataChannel] = {}
        self._routing_rules: List[RoutingRule] = []
        self._event_bus: EventBus = event_bus
        self._lock: threading.Lock = threading.Lock()
        self._router_thread: Optional[threading.Thread] = None
        self._running: bool = False

    def create_channel(self, name: str, capacity: int = 100) -> DataChannel:
        """Create a new named channel.

        Args:
            name: Unique channel name.
            capacity: Maximum buffer size.

        Returns:
            The created DataChannel.

        Raises:
            ValueError: If channel name already exists.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(capacity).
        """
        with self._lock:
            if name in self._channels:
                raise ValueError(f"Channel '{name}' already exists")
            channel = DataChannel(name, capacity)
            self._channels[name] = channel

            self._event_bus.publish(AIEvent(
                event_type=EventType.PIPELINE,
                source="data_flow_manager",
                message=f"Channel '{name}' created (capacity={capacity})",
                severity="INFO",
                data={"channel": name, "capacity": capacity},
            ))

            return channel

    def get_channel(self, name: str) -> Optional[DataChannel]:
        """Get a channel by name.

        Args:
            name: Channel name.

        Returns:
            DataChannel or None if not found.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return self._channels.get(name)

    def remove_channel(self, name: str) -> bool:
        """Remove a channel.

        Args:
            name: Channel name.

        Returns:
            True if removed, False if not found.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            channel = self._channels.pop(name, None)
            if channel:
                channel.close()
                # Remove routing rules for this channel
                self._routing_rules = [
                    r for r in self._routing_rules
                    if r.source != name and r.destination != name
                ]
            return channel is not None

    def put(self, channel_name: str, item: Any, timeout: float = 1.0) -> bool:
        """Put an item into a named channel.

        Args:
            channel_name: Target channel.
            item: Item to put.
            timeout: Blocking timeout.

        Returns:
            True if put, False otherwise.

        Complexity: T(n) ∈ O(1) amortized.
        """
        channel = self.get_channel(channel_name)
        if channel is None:
            logger.warning("Put to non-existent channel '%s'", channel_name)
            return False
        return channel.put(item, timeout=timeout)

    def put_nowait(self, channel_name: str, item: Any) -> bool:
        """Put an item without blocking.

        Args:
            channel_name: Target channel.
            item: Item to put.

        Returns:
            True if put, False if dropped or channel not found.

        Complexity: T(n) ∈ O(1).
        """
        channel = self.get_channel(channel_name)
        if channel is None:
            return False
        return channel.put_nowait(item)

    def get(self, channel_name: str, timeout: float = 1.0) -> Optional[Any]:
        """Get an item from a named channel.

        Args:
            channel_name: Source channel.
            timeout: Blocking timeout.

        Returns:
            Item or None if timed out / channel not found.

        Complexity: T(n) ∈ O(1) amortized.
        """
        channel = self.get_channel(channel_name)
        if channel is None:
            return None
        return channel.get(timeout=timeout)

    def get_nowait(self, channel_name: str) -> Optional[Any]:
        """Get an item without blocking.

        Args:
            channel_name: Source channel.

        Returns:
            Item or None if empty / channel not found.

        Complexity: T(n) ∈ O(1).
        """
        channel = self.get_channel(channel_name)
        if channel is None:
            return None
        return channel.get_nowait()

    def add_route(
        self,
        source: str,
        destination: str,
        predicate: Optional[Callable[[Any], bool]] = None,
        priority: int = 0,
    ) -> None:
        """Add a routing rule from source to destination channel.

        Args:
            source: Source channel name.
            destination: Destination channel name.
            predicate: Optional filter function.
            priority: Routing priority (lower = higher).

        Complexity: T(n) ∈ O(r) for r existing rules.
        """
        rule = RoutingRule(
            source=source,
            destination=destination,
            predicate=predicate,
            priority=priority,
        )
        with self._lock:
            self._routing_rules.append(rule)
            # Sort by priority (stable sort)
            self._routing_rules.sort(key=lambda r: r.priority)

    def remove_route(self, source: str, destination: str) -> bool:
        """Remove a routing rule.

        Args:
            source: Source channel name.
            destination: Destination channel name.

        Returns:
            True if removed, False if not found.

        Complexity: T(n) ∈ O(r) for r routing rules.
        """
        with self._lock:
            before = len(self._routing_rules)
            self._routing_rules = [
                r for r in self._routing_rules
                if not (r.source == source and r.destination == destination)
            ]
            return len(self._routing_rules) < before

    def route_once(self) -> int:
        """Perform a single routing pass across all channels.

        For each routing rule, attempts to move items from source to
        destination (non-blocking). Only moves items that pass the
        predicate filter.

        Returns:
            Number of items routed.

        Complexity: T(n) ∈ O(r * min(b, c)) for r rules, b buffer size, c capacity.
        """
        total_routed: int = 0

        with self._lock:
            rules = list(self._routing_rules)

        for rule in rules:
            src = self.get_channel(rule.source)
            dst = self.get_channel(rule.destination)
            if src is None or dst is None:
                continue
            if src.state != ChannelState.OPEN or dst.state != ChannelState.OPEN:
                continue

            # Drain all items from source, split by predicate
            items: List[Any] = []
            while True:
                item = src.get_nowait()
                if item is None:
                    break
                items.append(item)

            for item in items:
                if rule.predicate is not None and not rule.predicate(item):
                    # Put back to source (at end)
                    src.put_nowait(item)
                    continue
                if not dst.put_nowait(item):
                    # Destination full — put back to source
                    src.put_nowait(item)
                else:
                    total_routed += 1

        if total_routed > 0:
            self._event_bus.publish(AIEvent(
                event_type=EventType.PIPELINE,
                source="data_flow_manager",
                message=f"Routed {total_routed} items across channels",
                severity="INFO",
                data={"routed": total_routed},
            ))

        return total_routed

    def start_router(self, interval: float = 0.5) -> None:
        """Start the background router thread.

        Args:
            interval: Routing pass interval in seconds.

        Complexity: T(n) ∈ O(1).
        """
        if self._running:
            return
        self._running = True

        def _router_loop() -> None:
            while self._running:
                try:
                    self.route_once()
                except Exception as exc:
                    logger.error("Router error: %s", exc)
                time.sleep(interval)

        self._router_thread = threading.Thread(
            target=_router_loop, daemon=True, name="ai-os-data-flow-router"
        )
        self._router_thread.start()

    def stop_router(self) -> None:
        """Stop the background router thread.

        Complexity: T(n) ∈ O(1).
        """
        self._running = False
        if self._router_thread and self._router_thread.is_alive():
            self._router_thread.join(timeout=3.0)

    def get_channel_names(self) -> List[str]:
        """Get all channel names.

        Complexity: T(n) ∈ O(c) for c channels.
        """
        with self._lock:
            return list(self._channels.keys())

    def get_all_metrics(self) -> List[FlowMetrics]:
        """Get metrics for all channels.

        Complexity: T(n) ∈ O(c) for c channels.
        """
        with self._lock:
            return [ch.get_metrics() for ch in self._channels.values()]

    def get_status(self) -> Dict[str, Any]:
        """Get data flow manager status.

        Complexity: T(n) ∈ O(c) for c channels.
        """
        with self._lock:
            return {
                "channels": len(self._channels),
                "routing_rules": len(self._routing_rules),
                "router_running": self._running,
                "channel_metrics": {
                    name: {
                        "state": ch.state.value,
                        "size": ch.size,
                        "capacity": ch.capacity,
                        "total_put": ch.get_metrics().total_put,
                        "total_get": ch.get_metrics().total_get,
                        "total_dropped": ch.get_metrics().total_dropped,
                        "avg_latency_ms": ch.get_metrics().avg_latency_ms,
                    }
                    for name, ch in self._channels.items()
                },
            }

    def close_all(self) -> None:
        """Close all channels and stop router.

        Complexity: T(n) ∈ O(c) for c channels.
        """
        self.stop_router()
        with self._lock:
            for ch in self._channels.values():
                ch.close()
            self._channels.clear()
            self._routing_rules.clear()
