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
RAG Cache — LRU eviction cache for retrieval results.

Caches retrieval results keyed by query hash + indexed files signature.
Invalidation on file change. Bounded size with LRU eviction.

Complexity:
    get: T(n) in O(1) for cache hit, S(n) in O(1).
    put: T(n) in O(1) for cache miss, S(n) in O(1).
    invalidate: T(n) in O(1), S(n) in O(1).
    clear: T(n) in O(1), S(n) in O(1).

Invariants:
    - Cache size never exceeds max_size.
    - LRU eviction removes oldest accessed entry on overflow.
    - Cache key includes file signature for invalidation.

Edge Cases:
    - Empty query -> cache miss (no caching).
    - max_size=0 -> no caching (all gets return miss).
    - File signature change -> automatic invalidation.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """
    Immutable cache entry for a retrieval result.

    Complexity: T(n) in O(1), S(n) in O(result).
    Invariants: cached_at is a monotonic timestamp.
    Edge Cases: result may be empty tuple.
    """

    cache_key: str
    result: Any
    cached_at: float
    access_count: int
    file_signature: str


# ─── RAG Cache ───────────────────────────────────────────────────────────────


class RAGCache:
    """
    LRU eviction cache for RAG retrieval results.

    Uses OrderedDict for O(1) LRU operations (move_to_end + popitem).
    Cache key = SHA-256(query + file_signature). File signature changes
    trigger automatic invalidation on next get.

    Complexity:
        get: T(n) in O(1), S(n) in O(1).
        put: T(n) in O(1), S(n) in O(1).
        invalidate: T(n) in O(1), S(n) in O(1).
    Invariants:
        - Cache size never exceeds max_size.
        - LRU eviction on overflow.
        - Stale entries (file signature mismatch) treated as miss.
    Edge Cases:
        - Empty query -> miss.
        - max_size=0 -> no caching.
    """

    __slots__ = ("_cache", "_max_size", "_hits", "_misses", "_evictions")

    def __init__(self, max_size: int = 128) -> None:
        """
        Initialize RAG cache with maximum size.

        Complexity: T(n) in O(1), S(n) in O(max_size).
        Edge Cases: max_size=0 -> no caching (all gets return None).
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size: int = max_size
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """
        Current number of cached entries.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty cache -> 0.
        """
        return len(self._cache)

    @property
    def hits(self) -> int:
        """Total cache hits. Complexity: T(n) in O(1)."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses. Complexity: T(n) in O(1)."""
        return self._misses

    @property
    def evictions(self) -> int:
        """Total LRU evictions. Complexity: T(n) in O(1)."""
        return self._evictions

    @property
    def hit_rate(self) -> float:
        """
        Cache hit rate as fraction [0.0, 1.0].

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no requests -> 0.0.
        """
        total: int = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get(
        self, query: str, file_signature: str
    ) -> Optional[Any]:
        """
        Retrieve a cached result.

        Returns None on miss, cached result on hit.
        Updates LRU order on hit. Validates file signature.

        Complexity: T(n) in O(1), S(n) in O(1).
        Invariants: hit -> LRU order updated; miss -> order unchanged.
        Edge Cases: empty query -> None; stale signature -> None + eviction.
        """
        if self._max_size == 0 or not query.strip():
            self._misses += 1
            return None

        cache_key: str = self._make_key(query, file_signature)

        if cache_key not in self._cache:
            self._misses += 1
            return None

        entry: CacheEntry = self._cache[cache_key]

        if entry.file_signature != file_signature:
            del self._cache[cache_key]
            self._misses += 1
            return None

        self._cache.move_to_end(cache_key)
        self._hits += 1
        return entry.result

    def put(
        self,
        query: str,
        file_signature: str,
        result: Any,
    ) -> None:
        """
        Store a result in the cache.

        Evicts LRU entry if cache is full.

        Complexity: T(n) in O(1), S(n) in O(1).
        Invariants: cache size never exceeds max_size.
        Edge Cases: empty query -> no-op; max_size=0 -> no-op.
        """
        if self._max_size == 0 or not query.strip():
            return

        cache_key: str = self._make_key(query, file_signature)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)

        self._cache[cache_key] = CacheEntry(
            cache_key=cache_key,
            result=result,
            cached_at=time.time(),
            access_count=0,
            file_signature=file_signature,
        )

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._evictions += 1

    def invalidate(self, query: str, file_signature: str) -> bool:
        """
        Invalidate a specific cache entry.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: key not found -> False (no error).
        """
        cache_key: str = self._make_key(query, file_signature)
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False

    def invalidate_signature(self, file_signature: str) -> int:
        """
        Invalidate all entries matching a file signature.

        Complexity: T(n) in O(n) where n = cache size, S(n) in O(1).
        Edge Cases: no matching entries -> 0.
        """
        keys_to_remove: list[str] = [
            k for k, v in self._cache.items()
            if v.file_signature == file_signature
        ]
        for k in keys_to_remove:
            del self._cache[k]
        return len(keys_to_remove)

    def clear(self) -> int:
        """
        Clear all cache entries.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty cache -> returns 0.
        """
        count: int = len(self._cache)
        self._cache.clear()
        return count

    def get_stats(self) -> Tuple[int, int, int, int, float]:
        """
        Return cache statistics tuple.

        Returns: (size, hits, misses, evictions, hit_rate).

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty cache -> (0, hits, misses, evictions, hit_rate).
        """
        return (
            len(self._cache),
            self._hits,
            self._misses,
            self._evictions,
            self.hit_rate,
        )

    # ─── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(query: str, file_signature: str) -> str:
        """
        Generate cache key from query + file signature.

        Complexity: T(n) in O(n) where n = query + signature length,
                    S(n) in O(1).
        Edge Cases: empty inputs -> deterministic hash.
        """
        raw: str = f"{query}::{file_signature}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
