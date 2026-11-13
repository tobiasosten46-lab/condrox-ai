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
Memory models for the Brain memory system.

Provides immutable data structures for memory items with vector embeddings,
relevance scoring, and timestamp tracking.

Mathematical Foundation:
- Relevance decay follows exponential decay: r(t) = r0 * 2^(-t / h)
  where t is elapsed seconds and h is the half-life.
- Cosine similarity: sim(a,b) = (a·b) / (||a|| * ||b||)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """
    Immutable memory item with vector embedding and metadata.

    Attributes:
        item_id: Unique memory identifier.
        content: Human-readable memory content.
        embedding: Optional 1-D numpy vector for semantic retrieval.
        source: Origin of the memory (e.g., "conversation", "user_profile").
        tags: Categorical tags for filtering.
        created_at: Unix timestamp of creation.
        access_count: Number of times the memory has been retrieved.
        weight: Base importance weight in [0.0, 1.0].

    Complexity: T(init) ∈ O(d) where d is embedding dimension, S(init) ∈ O(d).
    """

    item_id: str
    content: str
    embedding: Optional[np.ndarray] = field(default=None, hash=False)
    source: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate invariants after construction.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not self.item_id:
            raise ValueError("MemoryItem.item_id must be non-empty")
        if self.weight < 0.0 or self.weight > 1.0:
            raise ValueError("weight must be in [0.0, 1.0]")
        if self.access_count < 0:
            raise ValueError("access_count must be non-negative")

    def with_access(self) -> "MemoryItem":
        """Return a new MemoryItem with incremented access count.

        Returns:
            A new immutable MemoryItem.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        from dataclasses import replace

        return replace(self, access_count=self.access_count + 1)

    def relevance(
        self,
        query_embedding: np.ndarray,
        current_time: Optional[float] = None,
        decay_halflife: float = 86400.0,
    ) -> float:
        """
        Compute time-decayed cosine similarity relevance score.

        Args:
            query_embedding: 1-D numpy vector to compare against.
            current_time: Optional reference time; defaults to now.
            decay_halflife: Half-life in seconds for recency decay.

        Returns:
            Relevance score in [0.0, 1.0].

        Complexity: T(n) ∈ O(d) where d is embedding dimension, S(n) ∈ O(1).
        """
        if self.embedding is None or query_embedding is None:
            return 0.0

        a = self.embedding.astype(np.float64)
        b = query_embedding.astype(np.float64)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))

        now = current_time if current_time is not None else time.time()
        elapsed = max(0.0, now - self.created_at)
        recency = math.pow(2.0, -elapsed / decay_halflife) if decay_halflife > 0.0 else 1.0

        usage_boost = 1.0 + math.log1p(self.access_count) * 0.05

        score = self.weight * (0.7 * (cosine + 1.0) / 2.0 + 0.3 * recency) * usage_boost
        return float(min(1.0, max(0.0, score)))
