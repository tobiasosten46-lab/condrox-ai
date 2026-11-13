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

"""Tests for MultiAgentMemorySync — Phase 2-D component."""

from __future__ import annotations

import threading

from core.conversational.multi_agent_memory_sync import (
    MultiAgentMemorySync,
    SyncEntry,
    SyncResult,
)


class TestMultiAgentMemorySync:
    """Unit tests for MultiAgentMemorySync."""

    def test_write_and_read(self) -> None:
        """Basic write and read works."""
        sync = MultiAgentMemorySync()
        sync.register_agent("agent_1", priority=5)
        result = sync.write("key", "value", "agent_1")
        assert result.success is True
        entry = sync.read("key")
        assert entry is not None
        assert entry.value == "value"

    def test_read_missing_key(self) -> None:
        """Missing key returns default."""
        sync = MultiAgentMemorySync()
        assert sync.read("missing") is None

    def test_read_value_with_default(self) -> None:
        """read_value returns default for missing key."""
        sync = MultiAgentMemorySync()
        assert sync.read_value("missing", "default") == "default"

    def test_priority_conflict(self) -> None:
        """Lower priority agent cannot overwrite higher priority."""
        sync = MultiAgentMemorySync()
        sync.register_agent("high", priority=9)
        sync.register_agent("low", priority=1)
        sync.write("key", "high_value", "high")
        result = sync.write("key", "low_value", "low")
        assert result.success is False
        assert result.conflict is True
        assert result.winner == "high"

    def test_same_agent_overwrite(self) -> None:
        """Same agent can overwrite its own entry."""
        sync = MultiAgentMemorySync()
        sync.register_agent("agent_1", priority=5)
        sync.write("key", "v1", "agent_1")
        result = sync.write("key", "v2", "agent_1")
        assert result.success is True
        assert sync.read_value("key") == "v2"

    def test_equal_priority_allows_write(self) -> None:
        """Equal priority agents can overwrite."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a", priority=5)
        sync.register_agent("b", priority=5)
        sync.write("key", "a_val", "a")
        result = sync.write("key", "b_val", "b")
        assert result.success is True

    def test_delete(self) -> None:
        """Delete removes an entry."""
        sync = MultiAgentMemorySync()
        sync.register_agent("agent_1")
        sync.write("key", "value", "agent_1")
        assert sync.delete("key", "agent_1") is True
        assert sync.read("key") is None

    def test_delete_missing_key(self) -> None:
        """Delete missing key returns False."""
        sync = MultiAgentMemorySync()
        assert sync.delete("missing", "agent_1") is False

    def test_sync_merges_local_state(self) -> None:
        """sync merges local state into shared buffer."""
        sync = MultiAgentMemorySync()
        sync.register_agent("agent_1", priority=5)
        local = {"a": 1, "b": 2}
        results = sync.sync("agent_1", local)
        assert len(results) == 2
        assert sync.read_value("a") == 1
        assert sync.read_value("b") == 2

    def test_conflict_log(self) -> None:
        """Conflict log records conflicts."""
        sync = MultiAgentMemorySync()
        sync.register_agent("high", priority=9)
        sync.register_agent("low", priority=1)
        sync.write("key", "high_val", "high")
        sync.write("key", "low_val", "low")
        conflicts = sync.get_conflicts()
        assert len(conflicts) > 0

    def test_clear_conflicts(self) -> None:
        """clear_conflicts empties the log."""
        sync = MultiAgentMemorySync()
        sync.register_agent("high", priority=9)
        sync.register_agent("low", priority=1)
        sync.write("key", "v", "high")
        sync.write("key", "v", "low")
        sync.clear_conflicts()
        assert len(sync.get_conflicts()) == 0

    def test_keys(self) -> None:
        """keys returns all buffer keys."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a")
        sync.write("k1", "v1", "a")
        sync.write("k2", "v2", "a")
        assert set(sync.keys()) == {"k1", "k2"}

    def test_count(self) -> None:
        """count returns entry count."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a")
        sync.write("k1", "v1", "a")
        sync.write("k2", "v2", "a")
        assert sync.count() == 2

    def test_snapshot(self) -> None:
        """snapshot returns a copy of the buffer."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a")
        sync.write("k1", "v1", "a")
        snap = sync.snapshot()
        assert "k1" in snap
        assert snap["k1"].value == "v1"

    def test_clear(self) -> None:
        """clear empties the buffer."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a")
        sync.write("k1", "v1", "a")
        sync.clear()
        assert sync.count() == 0

    def test_thread_safety(self) -> None:
        """Concurrent writes are safe."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a", priority=5)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for i in range(100):
                    sync.write(f"key_{i}", i, "a")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert sync.count() == 100

    def test_version_increments(self) -> None:
        """Version increments on each write."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a", priority=5)
        sync.write("key", "v1", "a")
        v1 = sync.read("key").version
        sync.write("key", "v2", "a")
        v2 = sync.read("key").version
        assert v2 > v1

    def test_sync_entry_is_frozen(self) -> None:
        """SyncEntry is immutable."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a")
        sync.write("key", "value", "a")
        entry = sync.read("key")
        try:
            entry.value = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_sync_result_is_frozen(self) -> None:
        """SyncResult is immutable."""
        sync = MultiAgentMemorySync()
        sync.register_agent("a")
        result = sync.write("key", "value", "a")
        try:
            result.success = False  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass
