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
Knowledge models for the Brain knowledge system.

Provides immutable KnowledgeItem objects with embedding support and
confidence-based scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class KnowledgeItem:
    """
    Immutable knowledge item stored in the knowledge backend.

    Attributes:
        item_id: Unique knowledge identifier.
        domain: Domain/category (e.g., "programming", "architecture").
        content: Human-readable knowledge content.
        embedding: Optional 1-D numpy vector for semantic retrieval.
        confidence: Source confidence in [0.0, 1.0].
        metadata: Free-form metadata (source, version, etc.).

    Complexity: T(init) ∈ O(d) where d is embedding dimension, S(init) ∈ O(d).
    """

    item_id: str
    domain: str
    content: str
    embedding: Optional[np.ndarray] = field(default=None, hash=False)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Validate invariants.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not self.item_id:
            raise ValueError("KnowledgeItem.item_id must be non-empty")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")

    def similarity(self, query_embedding: np.ndarray) -> float:
        """Compute cosine similarity with a query embedding.

        Args:
            query_embedding: 1-D numpy vector.

        Returns:
            Cosine similarity in [-1.0, 1.0].

        Complexity: T(n) ∈ O(d), S(n) ∈ O(1).
        """
        if self.embedding is None or query_embedding is None:
            return -1.0
        a = self.embedding.astype(np.float64)
        b = query_embedding.astype(np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        value = float(np.dot(a, b) / (norm_a * norm_b))
        return float(max(-1.0, min(1.0, value)))
