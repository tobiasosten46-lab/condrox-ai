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
Multi-Agent Memory Sync — shared memory buffer with locking and per-agent
priority for safe concurrent access.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class SyncEntry:
    """
    Immutable shared memory entry.

    Attributes:
        key: Entry key.
        value: Entry value.
        agent_id: Agent that wrote the entry.
        timestamp: Write timestamp.
        priority: Agent priority (higher = wins conflicts).
        version: Entry version number.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    key: str
    value: Any
    agent_id: str
    timestamp: float
    priority: int
    version: int


@dataclass(frozen=True, slots=True)
class SyncResult:
    """
    Immutable result of a sync operation.

    Attributes:
        success: Whether the operation succeeded.
        key: Entry key.
        conflict: Whether a conflict was detected.
        winner: Agent ID that won the conflict.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    success: bool
    key: str
    conflict: bool
    winner: str


class MultiAgentMemorySync:
    """
    Shared memory buffer with thread-safe locking and priority-based
    conflict resolution.

    Complexity:
        write: T(n) ∈ O(1), S(n) ∈ O(1).
        read: T(n) ∈ O(1), S(n) ∈ O(1).
        sync: T(n) ∈ O(e) for e entries, S(n) ∈ O(e).
    """

    __slots__ = ("_buffer", "_lock", "_agent_priorities", "_versions", "_conflict_log")

    def __init__(self) -> None:
        """Initialize the shared memory buffer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._buffer: Dict[str, SyncEntry] = {}
        self._lock = threading.RLock()
        self._agent_priorities: Dict[str, int] = {}
        self._versions: Dict[str, int] = {}
        self._conflict_log: List[Tuple[str, str, str, float]] = []

    def register_agent(self, agent_id: str, priority: int = 0) -> None:
        """Register an agent with a priority level.

        Args:
            agent_id: Unique agent identifier.
            priority: Priority (higher wins conflicts).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._agent_priorities[agent_id] = priority

    def write(self, key: str, value: Any, agent_id: str) -> SyncResult:
        """Write a value to the shared buffer.

        Args:
            key: Entry key.
            value: Value to write.
            agent_id: Agent performing the write.

        Returns:
            SyncResult indicating success and any conflict.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            agent_priority = self._agent_priorities.get(agent_id, 0)
            current = self._buffer.get(key)

            if current is not None:
                # Check for conflict
                if current.agent_id != agent_id:
                    if current.priority > agent_priority:
                        # Current entry has higher priority — conflict
                        self._conflict_log.append(
                            (key, agent_id, current.agent_id, time.time())
                        )
                        return SyncResult(
                            success=False,
                            key=key,
                            conflict=True,
                            winner=current.agent_id,
                        )

            # Write the entry
            version = self._versions.get(key, 0) + 1
            self._versions[key] = version
            entry = SyncEntry(
                key=key,
                value=value,
                agent_id=agent_id,
                timestamp=time.time(),
                priority=agent_priority,
                version=version,
            )
            self._buffer[key] = entry

            return SyncResult(
                success=True,
                key=key,
                conflict=False,
                winner=agent_id,
            )

    def read(self, key: str, default: Any = None) -> Optional[SyncEntry]:
        """Read a value from the shared buffer.

        Args:
            key: Entry key.
            default: Default value if key not found.

        Returns:
            SyncEntry or default.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            entry = self._buffer.get(key)
            if entry is None:
                return default
            return entry

    def read_value(self, key: str, default: Any = None) -> Any:
        """Read just the value from the buffer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        entry = self.read(key)
        if entry is None:
            return default
        return entry.value

    def delete(self, key: str, agent_id: str) -> bool:
        """Delete an entry from the buffer.

        Args:
            key: Entry key.
            agent_id: Agent performing the deletion.

        Returns:
            True if the entry was deleted.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            if key not in self._buffer:
                return False
            del self._buffer[key]
            self._versions.pop(key, None)
            return True

    def sync(self, agent_id: str, local_state: Dict[str, Any]) -> Dict[str, SyncEntry]:
        """Sync local agent state with the shared buffer.

        Args:
            agent_id: Agent performing the sync.
            local_state: Agent's local state to merge.

        Returns:
            Dictionary of entries that were updated or conflicted.

        Complexity: T(n) ∈ O(e) for e entries, S(n) ∈ O(e).
        """
        results: Dict[str, SyncEntry] = {}
        with self._lock:
            for key, value in local_state.items():
                result = self.write(key, value, agent_id)
                entry = self._buffer.get(key)
                if entry is not None:
                    results[key] = entry
        return results

    def get_conflicts(self, limit: int = 50) -> List[Tuple[str, str, str, float]]:
        """Return recent conflict log entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of (key, loser_agent, winner_agent, timestamp).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        with self._lock:
            return list(self._conflict_log[-limit:])

    def clear_conflicts(self) -> None:
        """Clear the conflict log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._conflict_log.clear()

    def keys(self) -> List[str]:
        """Return all keys in the buffer.

        Complexity: T(n) ∈ O(k), S(n) ∈ O(k).
        """
        with self._lock:
            return list(self._buffer.keys())

    def count(self) -> int:
        """Return the number of entries.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return len(self._buffer)

    def snapshot(self) -> Dict[str, SyncEntry]:
        """Return a snapshot of the entire buffer.

        Complexity: T(n) ∈ O(k), S(n) ∈ O(k).
        """
        with self._lock:
            return dict(self._buffer)

    def clear(self) -> None:
        """Clear all entries.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._buffer.clear()
            self._versions.clear()
