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

"""Tests for MemoryStabilizationLayer — Phase 2-A critical component."""

from __future__ import annotations

import time

import numpy as np

from core.conversational.memory_stabilization import (
    MemoryCompressor,
    MemoryFallbackPolicy,
    MemoryFlushPolicy,
    MemoryPrioritizer,
    MemoryStabilizationLayer,
)
from core.memory.memory_models import MemoryItem
from core.memory.memory_store import MemoryStore


def _make_item(item_id: str, weight: float = 0.5, age: float = 0.0, tag: str = "test") -> MemoryItem:
    """Helper to create a MemoryItem."""
    return MemoryItem(
        item_id=item_id,
        content=f"content_{item_id}",
        embedding=np.array([1.0, 0.0, 0.0]),
        source="test",
        tags=(tag,),
        created_at=time.time() - age,
        weight=weight,
    )


class TestMemoryPrioritizer:
    """Tests for MemoryPrioritizer."""

    def test_prioritize_orders_by_weight(self) -> None:
        """Higher weight items come first."""
        prioritizer = MemoryPrioritizer()
        low = _make_item("low", weight=0.1)
        high = _make_item("high", weight=0.9)
        result = prioritizer.prioritize([low, high])
        assert result[0].item_id == "high"

    def test_prioritize_empty_list(self) -> None:
        """Empty list returns empty."""
        prioritizer = MemoryPrioritizer()
        assert prioritizer.prioritize([]) == []


class TestMemoryFlushPolicy:
    """Tests for MemoryFlushPolicy."""

    def test_should_flush_false_under_cap(self) -> None:
        """No flush needed when under cap."""
        policy = MemoryFlushPolicy(max_items=100)
        store = MemoryStore()
        store.add(_make_item("a"))
        assert policy.should_flush(store) is False

    def test_should_flush_true_over_cap(self) -> None:
        """Flush needed when over cap."""
        policy = MemoryFlushPolicy(max_items=2)
        store = MemoryStore()
        store.add(_make_item("a"))
        store.add(_make_item("b"))
        store.add(_make_item("c"))
        assert policy.should_flush(store) is True

    def test_select_eviction_candidates(self) -> None:
        """Lowest weight items are selected for eviction."""
        policy = MemoryFlushPolicy(max_items=2)
        store = MemoryStore()
        store.add(_make_item("low", weight=0.1))
        store.add(_make_item("mid", weight=0.5))
        store.add(_make_item("high", weight=0.9))
        candidates = policy.select_eviction_candidates(store, 1)
        assert "low" in candidates

    def test_no_eviction_under_cap(self) -> None:
        """No candidates when under cap."""
        policy = MemoryFlushPolicy(max_items=100)
        store = MemoryStore()
        store.add(_make_item("a"))
        assert policy.select_eviction_candidates(store, 10) == []


class TestMemoryCompressor:
    """Tests for MemoryCompressor."""

    def test_small_groups_not_compressed(self) -> None:
        """Groups under threshold are not compressed."""
        compressor = MemoryCompressor(threshold=10)
        items = [_make_item(f"item_{i}") for i in range(5)]
        result = compressor.compress(items)
        assert len(result) == 5

    def test_large_groups_compressed(self) -> None:
        """Groups over threshold are compressed."""
        compressor = MemoryCompressor(threshold=5, max_content_len=500)
        items = [_make_item(f"item_{i}", tag="big") for i in range(10)]
        result = compressor.compress(items)
        assert len(result) < 10
        assert any(item.source == "compression" for item in result)


class TestMemoryFallbackPolicy:
    """Tests for MemoryFallbackPolicy."""

    def test_fallback_returns_empty_and_reason(self) -> None:
        """Fallback returns empty list and reason string."""
        policy = MemoryFallbackPolicy()
        items, reason = policy.fallback("search failed")
        assert items == []
        assert "search failed" in reason


class TestMemoryStabilizationLayer:
    """Tests for the full stabilization layer."""

    def test_stabilize_evicts_excess(self) -> None:
        """Stabilize evicts items over the cap."""
        layer = MemoryStabilizationLayer(max_items=5, compression_threshold=100)
        store = MemoryStore()
        for i in range(10):
            store.add(_make_item(f"item_{i}", weight=0.1 + i * 0.01))
        layer.stabilize(store)
        assert store.count() <= 5

    def test_stabilize_under_cap_no_change(self) -> None:
        """Stabilize does nothing when under cap."""
        layer = MemoryStabilizationLayer(max_items=100, compression_threshold=100)
        store = MemoryStore()
        store.add(_make_item("a"))
        store.add(_make_item("b"))
        layer.stabilize(store)
        assert store.count() == 2

    def test_fallback(self) -> None:
        """Fallback returns safe result."""
        layer = MemoryStabilizationLayer()
        items, reason = layer.fallback("test reason")
        assert items == []
        assert "test reason" in reason

    def test_prioritize(self) -> None:
        """Prioritize returns sorted items."""
        layer = MemoryStabilizationLayer()
        items = [_make_item("low", weight=0.1), _make_item("high", weight=0.9)]
        result = layer.prioritize(items)
        assert result[0].item_id == "high"
