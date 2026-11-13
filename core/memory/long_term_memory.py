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
Long-Term Memory — cross-session episodic memory per user.

Stores condensed facts from previous conversations and recalls them based on
keyword overlap with the current query.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.storage.sqlite_storage import SQLiteStorage
from core.storage.storage_interface import StorageBackend


@dataclass(frozen=True, slots=True)
class EpisodicMemory:
    """
    Immutable episodic memory entry.

    Attributes:
        memory_id: Unique memory identifier.
        user_id: User this memory belongs to.
        session_id: Session identifier.
        timestamp: Unix timestamp of the event.
        summary: Condensed summary of the fact/event.
        entities: Key entities extracted from the event.
        emotion: Detected emotion label.

    Complexity: S(n) ∈ O(1).
    """

    memory_id: str
    user_id: str
    session_id: str
    timestamp: float
    summary: str
    entities: tuple[str, ...]
    emotion: str


class LongTermMemory:
    """
    Persistent long-term episodic memory store.

    Memories are stored per user and can be recalled by keyword overlap. A
    StorageBackend is used for durability; an in-memory cache is kept for reads.

    Attributes:
        _storage: Storage backend.
        _table: Table/collection name.
        _cache: Mapping from memory_id to EpisodicMemory.

    Complexity:
        add: T(n) ∈ O(1) cache + O(1) storage write.
        recall: T(n) ∈ O(u * e · q) for u user memories, e entities, q query tokens.
        load: T(n) ∈ O(n) for n memories.
    """

    __slots__ = ("_storage", "_table", "_cache")

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        table: str = "episodic_memories",
    ) -> None:
        """Initialize long-term memory.

        Args:
            storage: Optional StorageBackend. Defaults to SQLiteStorage in memory.
            table: Storage table name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._storage = storage if storage is not None else SQLiteStorage(":memory:")
        self._table: str = table
        self._cache: Dict[str, EpisodicMemory] = {}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text using the same stemming as the knowledge backend.

        Args:
            text: Input text.

        Returns:
            Set of normalized tokens.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        return set(KnowledgeBackend.tokenize(text))

    @staticmethod
    def _serialize(memory: EpisodicMemory) -> Dict[str, Any]:
        """Serialize a memory for storage.

        Complexity: T(n) ∈ O(1).
        """
        data = asdict(memory)
        data["item_id"] = memory.memory_id
        return data

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> EpisodicMemory:
        """Deserialize a storage dict to EpisodicMemory.

        Complexity: T(n) ∈ O(1).
        """
        data = dict(data)
        data.pop("item_id", None)
        if "entities" in data and not isinstance(data["entities"], tuple):
            data["entities"] = tuple(data["entities"])
        return EpisodicMemory(**data)

    def add(
        self,
        user_id: str,
        summary: str,
        session_id: str = "",
        entities: Optional[List[str]] = None,
        emotion: str = "neutral",
    ) -> EpisodicMemory:
        """Store a new episodic memory.

        Args:
            user_id: User identifier.
            summary: Condensed summary.
            session_id: Optional session identifier.
            entities: Optional list of key entities.
            emotion: Detected emotion label.

        Returns:
            The created EpisodicMemory.

        Complexity: T(n) ∈ O(1) + O(1) storage write.
        """
        memory_id = f"ltm_{uuid.uuid4().hex[:16]}"
        memory = EpisodicMemory(
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=time.time(),
            summary=summary,
            entities=tuple(entities) if entities else (),
            emotion=emotion,
        )
        self._cache[memory_id] = memory
        self._storage.save_items(self._table, [self._serialize(memory)])
        return memory

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> List[EpisodicMemory]:
        """Recall top-k memories for a user based on query overlap.

        Args:
            user_id: User to recall for.
            query: Query text.
            top_k: Maximum memories to return.

        Returns:
            List of EpisodicMemory sorted by descending relevance.

        Complexity: T(n) ∈ O(u * (s + e) · q) for u user memories,
            s summary length, e entity count, q query tokens.
        """
        if not self._cache:
            self.load()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[tuple[float, EpisodicMemory]] = []
        for memory in self._cache.values():
            if memory.user_id != user_id:
                continue
            summary_tokens = self._tokenize(memory.summary)
            entity_tokens: set[str] = set()
            for entity in memory.entities:
                entity_tokens.update(self._tokenize(entity))
            all_tokens = summary_tokens | entity_tokens
            overlap = len(query_tokens & all_tokens)
            if overlap == 0:
                continue
            scored.append((overlap, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def load(self) -> int:
        """Load all memories from storage into cache.

        Returns:
            Number of memories loaded.

        Complexity: T(n) ∈ O(n) for n memories.
        """
        self._cache.clear()
        rows = self._storage.load_items(self._table)
        for row in rows:
            memory = self._deserialize(row)
            self._cache[memory.memory_id] = memory
        return len(self._cache)

    def save(self) -> int:
        """Persist all cached memories.

        Returns:
            Number of memories saved.

        Complexity: T(n) ∈ O(n) for n memories.
        """
        items = [self._serialize(m) for m in self._cache.values()]
        self._storage.save_items(self._table, items)
        return len(items)

    def count(self) -> int:
        """Return total memory count.

        Complexity: T(n) ∈ O(1).
        """
        return len(self._cache) if self._cache else self.load()
