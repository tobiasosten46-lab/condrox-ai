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
Memory manager for the unified Brain system.

The MemoryManager orchestrates memory ingestion, retrieval, and relevance
ranking. It embeds text using a lightweight deterministic embedding model
(rolling hash projections) so the module is fully functional without
external model downloads.

Mathematical Foundation:
- Embedding projection uses a stable random projection matrix R drawn from
  a standard normal distribution: e = normalize(text_hash_vector @ R).
- Cosine similarity and exponential decay are reused from MemoryItem.
"""

from __future__ import annotations

import hashlib
import time
import zlib
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from core.memory.memory_models import MemoryItem
from core.memory.memory_store import MemoryStore


class MemoryManager:
    """
    High-level memory manager wired into the BrainController.

    Provides deterministic, real embeddings and CRUD/search operations.

    Attributes:
        store: Underlying MemoryStore.
        embedding_dim: Dimensionality of memory vectors.
        _projection: Stable random projection matrix.

    Complexity:
        embed: T(n) ∈ O(d) per character nibble, S(n) ∈ O(d²) for matrix.
        search: T(n) ∈ O(n log k) for n memories and k results.
    """

    __slots__ = ("store", "embedding_dim", "_projection", "_rng", "_embed_cache", "_cache_max_size")

    def __init__(self, embedding_dim: int = 64, decay_halflife: float = 86400.0) -> None:
        """Initialize the memory manager.

        Args:
            embedding_dim: Size of embedding vectors.
            decay_halflife: Half-life for memory relevance decay.
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.store = MemoryStore(decay_halflife=decay_halflife)
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed=42)
        self._projection = self._rng.standard_normal(
            size=(self._char_hash_dim(), embedding_dim)
        ).astype(np.float64)
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max_size: int = 10000

    def _char_hash_dim(self) -> int:
        """Return the dimensionality of the raw character hash vector.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return 256

    def _text_hash_vector(self, text: str) -> np.ndarray:
        """
        Convert text into a 256-D bag-of-nibbles vector deterministically.

        Uses CRC32 over text chunks for fast non-crypto hashing (10-100x
        faster than SHA-256). Each 4-byte chunk maps to a bucket pair.

        Args:
            text: Input string.

        Returns:
            256-D integer vector.

        Complexity: T(n) ∈ O(n) where n is the input length, S(n) ∈ O(1).
        """
        text_bytes = text.encode("utf-8")
        vector = np.zeros(self._char_hash_dim(), dtype=np.float64)
        # Process in 8-byte chunks, use CRC32 for fast deterministic hashing
        chunk_size = 8
        for i in range(0, len(text_bytes), chunk_size):
            chunk = text_bytes[i:i + chunk_size]
            h = zlib.crc32(chunk) & 0xFFFFFFFF
            # Map to two buckets using high and low bytes
            bucket_hi = (h >> 8) % self._char_hash_dim()
            bucket_lo = h % self._char_hash_dim()
            vector[bucket_hi] += 1.0
            vector[bucket_lo] += 1.0
        return vector

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a unit-length vector.

        Results are cached for repeated queries.

        Args:
            text: Input string.

        Returns:
            Normalized 1-D numpy vector of length embedding_dim.

        Complexity: T(n) ∈ O(d² + n) where d is embedding_dim, n is text length.
            Cached: T(n) ∈ O(1) for repeated queries.
        """
        if text in self._embed_cache:
            self._embed_cache.move_to_end(text)
            return self._embed_cache[text]

        raw = self._text_hash_vector(text)
        embedding = raw @ self._projection
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            result = np.zeros(self.embedding_dim, dtype=np.float64)
        else:
            result = embedding / norm

        # LRU eviction — remove oldest entry when cache is full
        self._embed_cache[text] = result
        self._embed_cache.move_to_end(text)
        if len(self._embed_cache) > self._cache_max_size:
            self._embed_cache.popitem(last=False)
        return result

    def add(
        self,
        content: str,
        source: str = "conversation",
        tags: Optional[List[str]] = None,
        weight: float = 1.0,
        item_id: Optional[str] = None,
    ) -> MemoryItem:
        """Add a memory with an auto-generated embedding and id.

        Args:
            content: Memory content.
            source: Origin of the memory.
            tags: Optional list of tags.
            weight: Base importance weight in [0.0, 1.0].
            item_id: Optional explicit id; otherwise generated.

        Returns:
            The stored MemoryItem.

        Complexity: T(n) ∈ O(d² + n), S(n) ∈ O(d).
        """
        if not content:
            raise ValueError("content must be non-empty")
        embedding = self.embed(content)
        final_id = item_id or f"mem_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        item = MemoryItem(
            item_id=final_id,
            content=content,
            embedding=embedding,
            source=source,
            tags=tuple(tags or []),
            created_at=time.time(),
            weight=weight,
        )
        return self.store.add(item)

    def search(
        self, query: str, top_k: int = 5, tag_filter: Optional[str] = None
    ) -> List[MemoryItem]:
        """Search memories by semantic similarity to a query string.

        Uses keyword overlap first for recall, falls back to embedding search.

        Args:
            query: Query text.
            top_k: Maximum number of results.
            tag_filter: Optional required tag.

        Returns:
            Top-k matching MemoryItem instances.

        Complexity: T(n) ∈ O(n * q) for n memories, q query tokens.
        """
        if not query:
            return []

        # Keyword-based search — extract content words from query
        import re as _re
        stop = {"the", "is", "a", "an", "what", "who", "how", "why", "where",
                "when", "of", "to", "in", "for", "and", "or", "are", "was",
                "were", "be", "do", "does", "did", "will", "would", "could",
                "should", "can", "this", "that", "it", "its", "on", "at",
                "by", "with", "from", "as", "about", "you", "me", "my",
                "we", "our", "i", "please", "your"}
        query_tokens = set(_re.findall(r"[a-z0-9]+", query.lower())) - stop
        if not query_tokens:
            return self.store.search(self.embed(query), top_k, tag_filter)

        # Score all memories by keyword overlap
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.store.all_items():
            if tag_filter and tag_filter not in item.tags:
                continue
            item_tokens = set(_re.findall(r"[a-z0-9]+", item.content.lower())) - stop
            overlap = len(query_tokens & item_tokens)
            if overlap > 0:
                score = overlap / max(1, len(query_tokens))
                scored.append((score, item))

        if not scored:
            # Fall back to embedding search
            return self.store.search(self.embed(query), top_k, tag_filter)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
