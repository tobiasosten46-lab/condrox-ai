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

"""Tests for TemporalContextLayer — Phase 2-B component."""

from __future__ import annotations

import time

from core.conversational.temporal_context import (
    TemporalContextLayer,
    TemporalEntry,
)


class TestTemporalContextLayer:
    """Unit tests for TemporalContextLayer."""

    def test_set_and_get(self) -> None:
        """Basic set and get works."""
        layer = TemporalContextLayer(default_ttl=3600)
        layer.set("key", "value")
        assert layer.get("key") == "value"

    def test_get_missing_returns_default(self) -> None:
        """Missing key returns default."""
        layer = TemporalContextLayer()
        assert layer.get("missing", default="default") == "default"

    def test_expired_key_returns_default(self) -> None:
        """Expired key returns default."""
        layer = TemporalContextLayer(default_ttl=3600)
        base_time = time.time()
        layer.set("key", "value", ttl=1.0, current_time=base_time)
        # Access after TTL
        result = layer.get("key", default="expired", current_time=base_time + 2.0)
        assert result == "expired"

    def test_non_expired_key_returns_value(self) -> None:
        """Non-expired key returns value."""
        layer = TemporalContextLayer(default_ttl=3600)
        base_time = time.time()
        layer.set("key", "value", ttl=10.0, current_time=base_time)
        result = layer.get("key", default="expired", current_time=base_time + 5.0)
        assert result == "value"

    def test_ttl_zero_never_expires(self) -> None:
        """TTL of 0 means no expiration."""
        layer = TemporalContextLayer()
        base_time = time.time()
        layer.set("key", "value", ttl=0, current_time=base_time)
        result = layer.get("key", default="expired", current_time=base_time + 999999)
        assert result == "value"

    def test_refresh_extends_ttl(self) -> None:
        """Refresh extends a key's lifetime."""
        layer = TemporalContextLayer()
        base_time = time.time()
        layer.set("key", "value", ttl=10.0, current_time=base_time)
        # Refresh at base_time + 5
        layer.refresh("key", current_time=base_time + 5.0)
        # Should still be valid at base_time + 12 (5 + 10 = 15 > 12)
        result = layer.get("key", default="expired", current_time=base_time + 12.0)
        assert result == "value"

    def test_refresh_missing_key_returns_false(self) -> None:
        """Refresh of missing key returns False."""
        layer = TemporalContextLayer()
        assert layer.refresh("missing") is False

    def test_cleanup_expired(self) -> None:
        """Cleanup removes expired entries."""
        layer = TemporalContextLayer()
        base_time = time.time()
        layer.set("key1", "val1", ttl=5.0, current_time=base_time)
        layer.set("key2", "val2", ttl=100.0, current_time=base_time)
        removed = layer.cleanup_expired(current_time=base_time + 10.0)
        assert removed == 1
        assert layer.get("key2") == "val2"

    def test_freshness(self) -> None:
        """Freshness decreases over time."""
        layer = TemporalContextLayer()
        base_time = time.time()
        layer.set("key", "value", ttl=100.0, current_time=base_time)
        fresh = layer.freshness("key", current_time=base_time)
        stale = layer.freshness("key", current_time=base_time + 50.0)
        assert fresh > stale
        assert 0.0 < stale < 1.0

    def test_freshness_missing_key(self) -> None:
        """Missing key has freshness 0."""
        layer = TemporalContextLayer()
        assert layer.freshness("missing") == 0.0

    def test_count(self) -> None:
        """Count returns total entries."""
        layer = TemporalContextLayer()
        layer.set("a", 1)
        layer.set("b", 2)
        assert layer.count() == 2

    def test_active_count(self) -> None:
        """Active count excludes expired."""
        layer = TemporalContextLayer()
        base_time = time.time()
        layer.set("a", 1, ttl=5.0, current_time=base_time)
        layer.set("b", 2, ttl=100.0, current_time=base_time)
        assert layer.active_count(current_time=base_time + 10.0) == 1

    def test_keys(self) -> None:
        """Keys returns all keys."""
        layer = TemporalContextLayer()
        layer.set("a", 1)
        layer.set("b", 2)
        keys = layer.keys()
        assert set(keys) == {"a", "b"}

    def test_decay_weights(self) -> None:
        """Decay weights returns freshness scores."""
        layer = TemporalContextLayer()
        base_time = time.time()
        layer.set("a", 1, ttl=100.0, current_time=base_time)
        layer.set("b", 2, ttl=100.0, current_time=base_time - 50.0)
        weights = layer.decay_weights(current_time=base_time)
        assert "a" in weights
        assert "b" in weights
        assert weights["a"] > weights["b"]

    def test_access_count_increments(self) -> None:
        """Access count increments on get."""
        layer = TemporalContextLayer()
        layer.set("key", "value")
        layer.get("key")
        layer.get("key")
        entry = layer._entries["key"]
        assert entry.access_count == 2

    def test_temporal_entry_is_frozen(self) -> None:
        """TemporalEntry is immutable."""
        entry = TemporalEntry(
            value="test",
            created_at=time.time(),
            ttl=100.0,
            access_count=0,
            last_accessed=time.time(),
        )
        try:
            entry.value = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected
