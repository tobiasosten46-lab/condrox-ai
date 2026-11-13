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
MemoryManager Stabilization Layer — adds prioritization, flush rules,
compression, and fallback policy to the existing MemoryManager.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.memory.memory_models import MemoryItem
from core.memory.memory_store import MemoryStore


class MemoryPrioritizer:
    """
    Re-ranks memory items by a composite score of recency, relevance, and
    importance.

    Score = 0.4 * relevance + 0.3 * recency + 0.3 * weight

    Complexity: T(n) ∈ O(n log n), S(n) ∈ O(n) for n items.
    """

    __slots__ = ("_halflife",)

    def __init__(self, decay_halflife: float = 86400.0) -> None:
        """Initialize with a decay half-life in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._halflife = decay_halflife

    def prioritize(
        self,
        items: List[MemoryItem],
        current_time: Optional[float] = None,
    ) -> List[MemoryItem]:
        """Return items sorted by descending priority score.

        Args:
            items: List of MemoryItem to rank.
            current_time: Reference time for recency; defaults to now.

        Returns:
            Items sorted by priority (highest first).

        Complexity: T(n) ∈ O(n log n), S(n) ∈ O(n).
        """
        now = current_time if current_time is not None else time.time()

        def score(item: MemoryItem) -> float:
            elapsed = max(0.0, now - item.created_at)
            recency = 0.5 ** (elapsed / self._halflife) if self._halflife > 0 else 1.0
            importance = item.weight
            access = 1.0 + 0.05 * item.access_count
            return (0.3 * recency + 0.3 * importance + 0.4 * importance * recency) * access

        return sorted(items, key=score, reverse=True)


class MemoryFlushPolicy:
    """
    Evicts low-weight items when the store exceeds a maximum count.

    Complexity: T(n) ∈ O(n log n), S(n) ∈ O(n) for n items.
    """

    __slots__ = ("_max_items",)

    def __init__(self, max_items: int = 10000) -> None:
        """Initialize with a maximum item count.

        Args:
            max_items: Soft cap on store size.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self._max_items = max_items

    def should_flush(self, store: MemoryStore) -> bool:
        """Return True if the store exceeds the cap.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return store.count() > self._max_items

    def select_eviction_candidates(
        self,
        store: MemoryStore,
        count: int,
    ) -> List[str]:
        """Select the lowest-priority item IDs for eviction.

        Args:
            store: The MemoryStore to inspect.
            count: Number of candidates to select.

        Returns:
            List of item_ids to evict.

        Complexity: T(n) ∈ O(n log n), S(n) ∈ O(n).
        """
        items = store.all_items()
        if len(items) <= self._max_items:
            return []

        # Sort by weight ascending, then by created_at ascending (oldest first).
        items.sort(key=lambda x: (x.weight, x.created_at))
        excess = len(items) - self._max_items
        return [item.item_id for item in items[:min(count, excess)]]


class MemoryCompressor:
    """
    Summarises old memory clusters into a single condensed memory item.

    Groups items by tag, merges content, and produces a single MemoryItem
    per group when the group exceeds a threshold.

    Complexity: T(n) ∈ O(n * g) for n items and g groups, S(n) ∈ O(n).
    """

    __slots__ = ("_threshold", "_max_content_len")

    def __init__(self, threshold: int = 50, max_content_len: int = 2000) -> None:
        """Initialize with a group threshold and max content length.

        Args:
            threshold: Minimum group size to trigger compression.
            max_content_len: Maximum length of compressed content.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self._threshold = threshold
        self._max_content_len = max_content_len

    def compress(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """Compress groups of items into condensed items.

        Args:
            items: List of MemoryItem to compress.

        Returns:
            List of compressed MemoryItem (one per large group) plus
            uncompressed small-group items.

        Complexity: T(n) ∈ O(n * g), S(n) ∈ O(n).
        """
        groups: Dict[str, List[MemoryItem]] = {}
        for item in items:
            key = item.tags[0] if item.tags else "_default"
            groups.setdefault(key, []).append(item)

        result: List[MemoryItem] = []
        for tag, group in groups.items():
            if len(group) < self._threshold:
                result.extend(group)
                continue
            # Merge content
            merged_content = " | ".join(item.content for item in group[:100])
            if len(merged_content) > self._max_content_len:
                merged_content = merged_content[:self._max_content_len] + "..."
            avg_weight = sum(item.weight for item in group) / len(group)
            compressed = MemoryItem(
                item_id=f"compressed_{tag}_{int(time.time())}",
                content=merged_content,
                embedding=None,
                source="compression",
                tags=(tag,),
                created_at=time.time(),
                access_count=sum(item.access_count for item in group),
                weight=min(1.0, avg_weight),
            )
            result.append(compressed)
        return result


class MemoryFallbackPolicy:
    """
    Returns a safe, empty result with a reason string when search fails.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ()

    def fallback(self, reason: str) -> Tuple[List[MemoryItem], str]:
        """Return a safe empty result and a reason.

        Args:
            reason: Why the fallback was triggered.

        Returns:
            Tuple of (empty list, reason string).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return [], f"memory_fallback: {reason}"


class MemoryStabilizationLayer:
    """
    Orchestrates prioritization, flushing, compression, and fallback for
    the MemoryManager.

    Complexity:
        stabilize: T(n) ∈ O(n log n), S(n) ∈ O(n).
        fallback: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_prioritizer", "_flush_policy", "_compressor", "_fallback_policy")

    def __init__(
        self,
        max_items: int = 10000,
        compression_threshold: int = 50,
        decay_halflife: float = 86400.0,
    ) -> None:
        """Initialize all sub-components.

        Args:
            max_items: Soft cap on memory store size.
            compression_threshold: Group size to trigger compression.
            decay_halflife: Half-life for recency scoring.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._prioritizer = MemoryPrioritizer(decay_halflife)
        self._flush_policy = MemoryFlushPolicy(max_items)
        self._compressor = MemoryCompressor(compression_threshold)
        self._fallback_policy = MemoryFallbackPolicy()

    def stabilize(self, store: MemoryStore) -> MemoryStore:
        """Apply flush and compression to a store.

        Args:
            store: The MemoryStore to stabilize.

        Returns:
            The same store (modified in-place by eviction).

        Complexity: T(n) ∈ O(n log n), S(n) ∈ O(n).
        """
        if self._flush_policy.should_flush(store):
            candidates = self._flush_policy.select_eviction_candidates(
                store, store.count() - self._flush_policy._max_items
            )
            for item_id in candidates:
                store.remove(item_id)

        # Compress remaining items if needed
        items = store.all_items()
        compressed = self._compressor.compress(items)
        if len(compressed) < len(items):
            # Rebuild store with compressed items
            for item in items:
                store.remove(item.item_id)
            for item in compressed:
                store.add(item)

        return store

    def prioritize(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """Return items sorted by priority.

        Complexity: T(n) ∈ O(n log n), S(n) ∈ O(n).
        """
        return self._prioritizer.prioritize(items)

    def fallback(self, reason: str) -> Tuple[List[MemoryItem], str]:
        """Return a safe fallback result.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._fallback_policy.fallback(reason)
