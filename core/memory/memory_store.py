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
In-memory memory storage with vector-based retrieval.

The MemoryStore holds MemoryItem instances, supports CRUD operations,
and ranks items by time-decayed cosine similarity against a query embedding.

Mathematical Foundation:
- Retrieval uses k-NN in vector space with cosine distance.
- Top-k selection runs in O(n log k) using a bounded max-heap.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.memory.memory_models import MemoryItem


class MemoryStore:
    """
    Thread-unsafe but deterministic in-memory memory store.

    This implementation is real and functional; persistence is intentionally
    left to adapter layers in later phases.

    Attributes:
        _items: Mapping from item_id to MemoryItem.

    Complexity:
        add: T(n) ∈ O(1), S(n) ∈ O(1)
        get: T(n) ∈ O(1), S(n) ∈ O(1)
        search: T(n) ∈ O(n log k) where n is item count and k is top_k.
    """

    __slots__ = ("_items", "_halflife")

    def __init__(self, decay_halflife: float = 86400.0) -> None:
        """Initialize an empty MemoryStore.

        Args:
            decay_halflife: Half-life in seconds for relevance decay.
        """
        self._items: Dict[str, MemoryItem] = {}
        self._halflife: float = decay_halflife

    def add(self, item: MemoryItem) -> MemoryItem:
        """Add or update a memory item.

        Args:
            item: MemoryItem to store.

        Returns:
            The stored MemoryItem.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._items[item.item_id] = item
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve a single memory by id.

        Args:
            item_id: Unique memory identifier.

        Returns:
            The MemoryItem or None.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._items.get(item_id)

    def remove(self, item_id: str) -> bool:
        """Remove a memory item by id.

        Args:
            item_id: Unique memory identifier.

        Returns:
            True if an item was removed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def count(self) -> int:
        """Return the number of stored memory items.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._items)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        tag_filter: Optional[str] = None,
        current_time: Optional[float] = None,
    ) -> List[MemoryItem]:
        """
        Return the top-k memory items ranked by relevance to the query.

        Args:
            query_embedding: 1-D numpy vector.
            top_k: Maximum number of results.
            tag_filter: Optional tag that results must possess.
            current_time: Optional reference time for decay.

        Returns:
            List of MemoryItem sorted by descending relevance.

        Complexity: T(n) ∈ O(n log k), S(n) ∈ O(k) for n items.
        """
        now = current_time if current_time is not None else time.time()
        top_k = max(1, top_k)

        heap: List[tuple[float, int, MemoryItem]] = []
        counter = 0

        for item in self._items.values():
            if tag_filter and tag_filter not in item.tags:
                continue
            score = item.relevance(query_embedding, now, self._halflife)
            # Use counter to break ties deterministically
            entry = (score, counter, item)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)
            counter += 1

        heap.sort(key=lambda x: x[0], reverse=True)
        return [item for _, _, item in heap]

    def all_items(self) -> List[MemoryItem]:
        """Return a snapshot of all stored memory items.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return list(self._items.values())

    def save(self, backend: "core.storage.StorageBackend", table: str = "memories") -> int:
        """Persist all memory items to a storage backend.

        Args:
            backend: Storage backend to write to.
            table: Target table name.

        Returns:
            Number of items saved.

        Complexity: T(n) ∈ O(n) for n items, S(n) ∈ O(n).
        """
        from core.storage import StorageBackend  # noqa: F811
        items_data: List[Dict[str, Any]] = []
        for item in self._items.values():
            embedding_list = (
                item.embedding.tolist() if item.embedding is not None else None
            )
            items_data.append({
                "item_id": item.item_id,
                "content": item.content,
                "embedding": embedding_list,
                "source": item.source,
                "tags": list(item.tags),
                "created_at": item.created_at,
                "access_count": item.access_count,
                "weight": item.weight,
            })
        backend.save_items(table, items_data)
        return len(items_data)

    def load(self, backend: "core.storage.StorageBackend", table: str = "memories") -> int:
        """Load memory items from a storage backend.

        Args:
            backend: Storage backend to read from.
            table: Source table name.

        Returns:
            Number of items loaded.

        Complexity: T(n) ∈ O(n) for n stored items, S(n) ∈ O(n).
        """
        from core.storage import StorageBackend  # noqa: F811
        items_data = backend.load_items(table)
        count = 0
        for data in items_data:
            embedding_raw = data.get("embedding")
            embedding = (
                np.array(embedding_raw, dtype=np.float64)
                if embedding_raw is not None
                else None
            )
            item = MemoryItem(
                item_id=data["item_id"],
                content=data["content"],
                embedding=embedding,
                source=data.get("source", "unknown"),
                tags=tuple(data.get("tags", [])),
                created_at=data.get("created_at", time.time()),
                access_count=data.get("access_count", 0),
                weight=data.get("weight", 1.0),
            )
            self._items[item.item_id] = item
            count += 1
        return count
