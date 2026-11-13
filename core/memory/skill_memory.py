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
Procedural Skill Memory — reusable user workflows and skills.

Stores named skill patterns with trigger phrases and ordered steps. The AI can
match a query against trigger phrases and execute the corresponding workflow.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.storage.sqlite_storage import SQLiteStorage
from core.storage.storage_interface import StorageBackend


@dataclass(frozen=True, slots=True)
class ProceduralSkill:
    """
    Immutable procedural skill.

    Attributes:
        skill_id: Unique skill identifier.
        name: Human-readable skill name.
        trigger_phrases: Phrases that activate this skill.
        steps: Ordered steps to execute.
        owner_user_id: User who owns the skill (empty for global).

    Complexity: S(n) ∈ O(s) where s is number of steps + triggers.
    """

    skill_id: str
    name: str
    trigger_phrases: tuple[str, ...]
    steps: tuple[str, ...]
    owner_user_id: str


class SkillMemory:
    """
    Persistent store for procedural skills.

    Skills can be matched by query-token overlap with trigger phrases and
    retrieved for execution. Storage backend provides durability.

    Attributes:
        _storage: Storage backend.
        _table: Table/collection name.
        _cache: Mapping from skill_id to ProceduralSkill.

    Complexity:
        add: T(n) ∈ O(1) cache + O(1) storage write.
        match: T(n) ∈ O(s · t · q) for s skills, t triggers, q query tokens.
        get_steps: T(n) ∈ O(1).
    """

    __slots__ = ("_storage", "_table", "_cache")

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        table: str = "procedural_skills",
    ) -> None:
        """Initialize skill memory.

        Args:
            storage: Optional StorageBackend. Defaults to SQLiteStorage in memory.
            table: Storage table name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._storage = storage if storage is not None else SQLiteStorage(":memory:")
        self._table: str = table
        self._cache: Dict[str, ProceduralSkill] = {}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text using the knowledge backend stemmer.

        Args:
            text: Input text.

        Returns:
            Set of normalized tokens.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        return set(KnowledgeBackend.tokenize(text))

    @staticmethod
    def _serialize(skill: ProceduralSkill) -> Dict[str, Any]:
        """Serialize a skill for storage.

        Complexity: T(n) ∈ O(s) for s steps/triggers.
        """
        data = asdict(skill)
        data["item_id"] = skill.skill_id
        return data

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> ProceduralSkill:
        """Deserialize a storage dict to ProceduralSkill.

        Complexity: T(n) ∈ O(s) for s steps/triggers.
        """
        data = dict(data)
        data.pop("item_id", None)
        for key in ("trigger_phrases", "steps"):
            if key in data and not isinstance(data[key], tuple):
                data[key] = tuple(data[key])
        return ProceduralSkill(**data)

    def add(
        self,
        name: str,
        trigger_phrases: List[str],
        steps: List[str],
        owner_user_id: str = "",
        skill_id: Optional[str] = None,
    ) -> ProceduralSkill:
        """Store a new skill.

        Args:
            name: Skill name.
            trigger_phrases: Phrases that activate this skill.
            steps: Ordered steps.
            owner_user_id: Optional owning user.
            skill_id: Optional explicit skill id.

        Returns:
            The created ProceduralSkill.

        Complexity: T(n) ∈ O(1) + O(1) storage write.
        """
        sid = skill_id or f"skill_{uuid.uuid4().hex[:16]}"
        skill = ProceduralSkill(
            skill_id=sid,
            name=name,
            trigger_phrases=tuple(trigger_phrases),
            steps=tuple(steps),
            owner_user_id=owner_user_id,
        )
        self._cache[sid] = skill
        self._storage.save_items(self._table, [self._serialize(skill)])
        return skill

    def get(self, skill_id: str) -> Optional[ProceduralSkill]:
        """Retrieve a skill by id.

        Args:
            skill_id: Skill identifier.

        Returns:
            ProceduralSkill or None.

        Complexity: T(n) ∈ O(1).
        """
        return self._cache.get(skill_id) or self._load_one(skill_id)

    def _load_one(self, skill_id: str) -> Optional[ProceduralSkill]:
        """Load a single skill from storage into cache.

        Complexity: T(n) ∈ O(n) for n stored skills.
        """
        for row in self._storage.load_items(self._table):
            if row.get("item_id") == skill_id:
                skill = self._deserialize(row)
                self._cache[skill_id] = skill
                return skill
        return None

    def match(self, query: str, top_k: int = 3) -> List[ProceduralSkill]:
        """Find skills whose trigger phrases overlap with the query.

        Args:
            query: Query text.
            top_k: Maximum skills to return.

        Returns:
            List of ProceduralSkill sorted by descending overlap.

        Complexity: T(n) ∈ O(s · t · q) for s skills, t triggers, q query tokens.
        """
        if not self._cache:
            self.load()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[tuple[int, ProceduralSkill]] = []
        for skill in self._cache.values():
            overlap = 0
            for phrase in skill.trigger_phrases:
                phrase_tokens = self._tokenize(phrase)
                overlap += len(query_tokens & phrase_tokens)
            if overlap > 0:
                scored.append((overlap, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def execute(self, skill_id: str) -> List[str]:
        """Return the ordered steps for a skill.

        Args:
            skill_id: Skill identifier.

        Returns:
            List of step strings, or empty if not found.

        Complexity: T(n) ∈ O(1).
        """
        skill = self.get(skill_id)
        return list(skill.steps) if skill is not None else []

    def load(self) -> int:
        """Load all skills from storage into cache.

        Returns:
            Number of skills loaded.

        Complexity: T(n) ∈ O(n) for n skills.
        """
        self._cache.clear()
        rows = self._storage.load_items(self._table)
        for row in rows:
            skill = self._deserialize(row)
            self._cache[skill.skill_id] = skill
        return len(self._cache)

    def save(self) -> int:
        """Persist all cached skills.

        Returns:
            Number of skills saved.

        Complexity: T(n) ∈ O(n) for n skills.
        """
        items = [self._serialize(s) for s in self._cache.values()]
        self._storage.save_items(self._table, items)
        return len(items)
