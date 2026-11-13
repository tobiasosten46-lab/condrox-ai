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
AI-OS State Manager — centralized state management with checkpointing,
persistence, and recovery.

Periodically snapshots system state to SQLite and restores on boot.

Mathematical Foundation:
- Checkpoint interval: T(n) ∈ O(n) for n state items.
- Recovery: O(n) for n stored items.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.os.event_bus import AIEvent, EventBus, EventType


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Immutable state snapshot.

    Attributes:
        version: Snapshot version for migration support.
        timestamp: Unix timestamp.
        state_data: Serialized state dict.
        checkpoint_id: Unique checkpoint identifier.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(d) for state data.
    """

    version: int
    timestamp: float
    state_data: Dict[str, Any] = field(default_factory=dict, hash=False)
    checkpoint_id: str = ""


class StateManager:
    """
    Centralized state manager with SQLite persistence.

    Attributes:
        _db_path: Path to SQLite state database.
        _state: In-memory state dict.
        _lock: Thread lock for concurrent access.
        _event_bus: Event bus for publishing state events.
        _checkpoint_interval: Auto-checkpoint interval in seconds.
        _last_checkpoint: Timestamp of last checkpoint.
        _max_snapshots: Maximum snapshots to retain.

    Complexity:
        checkpoint: O(n) for n state items.
        recover: O(n) for n stored items.
    """

    __slots__ = (
        "_db_path",
        "_state",
        "_lock",
        "_event_bus",
        "_checkpoint_interval",
        "_last_checkpoint",
        "_max_snapshots",
    )

    def __init__(
        self,
        event_bus: EventBus,
        db_path: str = "Databases/condrox_os_state.db",
        checkpoint_interval: float = 60.0,
        max_snapshots: int = 100,
    ) -> None:
        """Initialize the state manager.

        Args:
            event_bus: Event bus for publishing events.
            db_path: Path to SQLite state database.
            checkpoint_interval: Auto-checkpoint interval in seconds.
            max_snapshots: Maximum snapshots to retain in DB.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._db_path: str = db_path
        self._state: Dict[str, Any] = {}
        self._lock: threading.Lock = threading.Lock()
        self._event_bus: EventBus = event_bus
        self._checkpoint_interval: float = checkpoint_interval
        self._last_checkpoint: float = 0.0
        self._max_snapshots: int = max_snapshots
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite state database.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    checkpoint_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    state_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_ts
                ON state_snapshots(timestamp DESC)
            """)
            conn.commit()
        finally:
            conn.close()

    def set(self, key: str, value: Any) -> None:
        """Set a state value.

        Args:
            key: State key.
            value: State value (must be JSON-serializable).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value.

        Args:
            key: State key.
            default: Default if key not found.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return self._state.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get all state.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            return dict(self._state)

    def checkpoint(self) -> str:
        """Save current state to SQLite.

        Returns:
            Checkpoint ID.

        Complexity: T(n) ∈ O(n) for n state items.
        """
        with self._lock:
            state_copy = dict(self._state)

        checkpoint_id = f"ckpt_{int(time.time() * 1000)}"
        snapshot = StateSnapshot(
            version=1,
            timestamp=time.time(),
            state_data=state_copy,
            checkpoint_id=checkpoint_id,
        )

        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO state_snapshots VALUES (?, ?, ?, ?)",
                (
                    checkpoint_id,
                    snapshot.version,
                    snapshot.timestamp,
                    json.dumps(state_copy, default=str),
                ),
            )
            # Prune old snapshots
            conn.execute(
                "DELETE FROM state_snapshots WHERE checkpoint_id NOT IN "
                "(SELECT checkpoint_id FROM state_snapshots ORDER BY timestamp DESC LIMIT ?)",
                (self._max_snapshots,),
            )
            conn.commit()
        finally:
            conn.close()

        self._last_checkpoint = time.time()

        self._event_bus.publish(AIEvent(
            event_type=EventType.STATE,
            source="state_manager",
            message=f"State checkpoint saved: {checkpoint_id}",
            severity="INFO",
            data={"checkpoint_id": checkpoint_id, "keys": len(state_copy)},
        ))

        return checkpoint_id

    def recover(self) -> Optional[StateSnapshot]:
        """Restore state from the latest checkpoint.

        Returns:
            Restored snapshot, or None if no checkpoint exists.

        Complexity: T(n) ∈ O(n) for n state items.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT checkpoint_id, version, timestamp, state_json "
                "FROM state_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            self._event_bus.publish(AIEvent(
                event_type=EventType.STATE,
                source="state_manager",
                message="No checkpoint found — starting fresh",
                severity="INFO",
            ))
            return None

        checkpoint_id, version, timestamp, state_json = row
        state_data = json.loads(state_json)

        with self._lock:
            self._state = state_data

        snapshot = StateSnapshot(
            version=version,
            timestamp=timestamp,
            state_data=state_data,
            checkpoint_id=checkpoint_id,
        )

        self._event_bus.publish(AIEvent(
            event_type=EventType.STATE,
            source="state_manager",
            message=f"State recovered from checkpoint: {checkpoint_id}",
            severity="INFO",
            data={"checkpoint_id": checkpoint_id, "keys": len(state_data)},
        ))

        return snapshot

    def maybe_checkpoint(self) -> Optional[str]:
        """Auto-checkpoint if interval has elapsed.

        Returns:
            Checkpoint ID if checkpointed, None otherwise.

        Complexity: T(n) ∈ O(n) for n state items.
        """
        if time.time() - self._last_checkpoint < self._checkpoint_interval:
            return None
        return self.checkpoint()

    def export_state(self) -> Dict[str, Any]:
        """Export all state as JSON-serializable dict.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            return dict(self._state)

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from a dict.

        Args:
            state: State dict to import.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        with self._lock:
            self._state = dict(state)

    def get_status(self) -> Dict[str, Any]:
        """Get state manager status.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return {
                "db_path": self._db_path,
                "state_keys": len(self._state),
                "last_checkpoint": self._last_checkpoint,
                "checkpoint_interval": self._checkpoint_interval,
                "max_snapshots": self._max_snapshots,
            }
