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
User Profile Store — persistent per-user preferences and summary.

Stores user-specific traits, preferences, and a free-text summary so the
AI can personalize responses across sessions.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.storage.sqlite_storage import SQLiteStorage
from core.storage.storage_interface import StorageBackend


@dataclass(frozen=True, slots=True)
class UserProfile:
    """
    Immutable user profile.

    Attributes:
        user_id: Unique user identifier.
        preferences: Key/value preferences as an immutable tuple of pairs
            (e.g., (("language", "no"), ("theme", "dark"))).
        summary: Free-text summary of what the system knows about the user.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.

    Complexity: S(n) ∈ O(p) where p is number of preference entries.
    """

    user_id: str
    preferences: Tuple[Tuple[str, str], ...]
    summary: str
    created_at: float
    updated_at: float


class UserProfileStore:
    """
    Persistent store for user profiles.

    Uses a StorageBackend (default SQLite) for durability. In-memory
    cache is kept for fast reads; writes flush to storage immediately.

    Attributes:
        _storage: Storage backend.
        _table: Table/collection name.
        _cache: In-memory cache keyed by user_id.

    Complexity:
        get: T(n) ∈ O(1) cache hit, O(n) cold load.
        upsert: T(n) ∈ O(1) cache + O(1) storage write.
        load: T(n) ∈ O(n) for n profiles.
        save: T(n) ∈ O(n) for n cached profiles.
    """

    __slots__ = ("_storage", "_table", "_cache")

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        table: str = "user_profiles",
    ) -> None:
        """Initialize the profile store.

        Args:
            storage: Optional StorageBackend. Defaults to SQLiteStorage.
            table: Storage table name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._storage = storage if storage is not None else SQLiteStorage(":memory:")
        self._table: str = table
        self._cache: Dict[str, UserProfile] = {}

    @staticmethod
    def _profile_id(user_id: str) -> str:
        """Generate a stable item_id from user_id.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return f"user_{user_id}"

    @staticmethod
    def _normalize_preferences(value: Any) -> Tuple[Tuple[str, str], ...]:
        """Normalize a preferences payload to an immutable tuple of pairs.

        Accepts a dict, a list of pairs, or an existing tuple of pairs.

        Args:
            value: Raw preferences data.

        Returns:
            Tuple of (key, value) pairs, sorted by key for determinism.

        Complexity: T(n) ∈ O(p log p) for p pairs due to sorting.
        """
        if isinstance(value, dict):
            pairs = list(value.items())
        else:
            pairs = [tuple(p) for p in value]
        pairs.sort(key=lambda pair: pair[0])
        return tuple((str(k), str(v)) for k, v in pairs)

    @staticmethod
    def _serialize(profile: UserProfile) -> Dict[str, Any]:
        """Serialize a UserProfile to a storage-compatible dict.

        Complexity: T(n) ∈ O(p), S(n) ∈ O(p).
        """
        data = asdict(profile)
        data["item_id"] = UserProfileStore._profile_id(profile.user_id)
        return data

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> UserProfile:
        """Deserialize a storage dict to a UserProfile.

        Complexity: T(n) ∈ O(p), S(n) ∈ O(p).
        """
        data = dict(data)
        data.pop("item_id", None)
        data["preferences"] = UserProfileStore._normalize_preferences(
            data.get("preferences", ())
        )
        return UserProfile(**data)

    def get(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve a user profile.

        Args:
            user_id: User identifier.

        Returns:
            UserProfile or None if not found.

        Complexity: T(n) ∈ O(1) cache hit; O(n) cold load.
        """
        if user_id in self._cache:
            return self._cache[user_id]

        item_id = self._profile_id(user_id)
        for row in self._storage.load_items(self._table):
            if row.get("item_id") == item_id:
                profile = self._deserialize(row)
                self._cache[user_id] = profile
                return profile
        return None

    def upsert(
        self,
        user_id: str,
        preferences: Optional[Dict[str, str]] = None,
        summary: Optional[str] = None,
    ) -> UserProfile:
        """Create or update a user profile.

        Args:
            user_id: User identifier.
            preferences: Optional preference dict to merge.
            summary: Optional summary to set/append.

        Returns:
            Updated UserProfile.

        Complexity: T(n) ∈ O(1) for cache + O(1) storage write.
        """
        now = time.time()
        existing = self.get(user_id)

        if existing is not None:
            merged_preferences = dict(existing.preferences)
            if preferences:
                merged_preferences.update(preferences)
            new_summary = summary if summary is not None else existing.summary
            created_at = existing.created_at
        else:
            merged_preferences = dict(preferences) if preferences else {}
            new_summary = summary or ""
            created_at = now

        profile = UserProfile(
            user_id=user_id,
            preferences=self._normalize_preferences(merged_preferences),
            summary=new_summary,
            created_at=created_at,
            updated_at=now,
        )
        self._cache[user_id] = profile
        self._storage.save_items(self._table, [self._serialize(profile)])
        return profile

    def list_users(self) -> List[str]:
        """Return all known user ids.

        Returns:
            List of user ids.

        Complexity: T(n) ∈ O(n) for n stored profiles.
        """
        if not self._cache:
            self.load()
        return list(self._cache.keys())

    def load(self) -> int:
        """Load all profiles from storage into the cache.

        Returns:
            Number of profiles loaded.

        Complexity: T(n) ∈ O(n) for n profiles.
        """
        self._cache.clear()
        rows = self._storage.load_items(self._table)
        for row in rows:
            profile = self._deserialize(row)
            self._cache[profile.user_id] = profile
        return len(self._cache)

    def save(self) -> int:
        """Persist all cached profiles to storage.

        Returns:
            Number of profiles saved.

        Complexity: T(n) ∈ O(n) for n cached profiles.
        """
        items = [self._serialize(p) for p in self._cache.values()]
        self._storage.save_items(self._table, items)
        return len(items)
