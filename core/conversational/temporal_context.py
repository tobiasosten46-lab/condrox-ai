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
ContextManager Temporal Layer — adds time-based key expiration and decay
to the existing ContextManager.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class TemporalEntry:
    """
    Immutable context entry with temporal metadata.

    Attributes:
        value: The stored value.
        created_at: Unix timestamp of creation.
        ttl: Time-to-live in seconds (0 = no expiration).
        access_count: Number of times accessed.
        last_accessed: Unix timestamp of last access.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    value: Any
    created_at: float
    ttl: float
    access_count: int
    last_accessed: float

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if the entry has expired.

        Args:
            current_time: Reference time; defaults to now.

        Returns:
            True if TTL > 0 and elapsed time exceeds TTL.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.ttl <= 0:
            return False
        now = current_time if current_time is not None else time.time()
        return (now - self.created_at) > self.ttl

    def freshness(self, current_time: Optional[float] = None) -> float:
        """Compute freshness score in [0.0, 1.0].

        Uses exponential decay: f(t) = e^(-t/tau) where tau = TTL.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.ttl <= 0:
            return 1.0
        now = current_time if current_time is not None else time.time()
        elapsed = max(0.0, now - self.created_at)
        return math.exp(-elapsed / self.ttl)


class TemporalContextLayer:
    """
    Temporal context layer with key expiration and freshness decay.

    Wraps a dictionary of temporal entries, automatically expiring stale
    keys and providing freshness-weighted retrieval.

    Complexity:
        set: T(n) ∈ O(1), S(n) ∈ O(1).
        get: T(n) ∈ O(1), S(n) ∈ O(1).
        cleanup_expired: T(n) ∈ O(n), S(n) ∈ O(n).
    """

    __slots__ = ("_entries", "_default_ttl")

    def __init__(self, default_ttl: float = 3600.0) -> None:
        """Initialize with a default TTL.

        Args:
            default_ttl: Default time-to-live in seconds.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._entries: Dict[str, TemporalEntry] = {}
        self._default_ttl = default_ttl

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> None:
        """Set a temporal context entry.

        Args:
            key: Context key.
            value: Value to store.
            ttl: Time-to-live in seconds; uses default if None.
            current_time: Reference time; defaults to now.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        now = current_time if current_time is not None else time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        self._entries[key] = TemporalEntry(
            value=value,
            created_at=now,
            ttl=effective_ttl,
            access_count=0,
            last_accessed=now,
        )

    def get(
        self,
        key: str,
        default: Any = None,
        current_time: Optional[float] = None,
    ) -> Any:
        """Get a value, returning default if expired or missing.

        Args:
            key: Context key.
            default: Value to return if missing or expired.
            current_time: Reference time; defaults to now.

        Returns:
            The stored value or default.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        entry = self._entries.get(key)
        if entry is None:
            return default
        if entry.is_expired(current_time):
            del self._entries[key]
            return default

        # Update access metadata
        now = current_time if current_time is not None else time.time()
        from dataclasses import replace

        self._entries[key] = replace(
            entry,
            access_count=entry.access_count + 1,
            last_accessed=now,
        )
        return entry.value

    def refresh(self, key: str, current_time: Optional[float] = None) -> bool:
        """Refresh a key's TTL without changing its value.

        Args:
            key: Context key.
            current_time: Reference time; defaults to now.

        Returns:
            True if the key was refreshed, False if not found.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        entry = self._entries.get(key)
        if entry is None:
            return False

        now = current_time if current_time is not None else time.time()
        from dataclasses import replace

        self._entries[key] = replace(entry, created_at=now)
        return True

    def cleanup_expired(self, current_time: Optional[float] = None) -> int:
        """Remove all expired entries.

        Args:
            current_time: Reference time; defaults to now.

        Returns:
            Number of entries removed.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        now = current_time if current_time is not None else time.time()
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.is_expired(now)
        ]
        for key in expired_keys:
            del self._entries[key]
        return len(expired_keys)

    def freshness(self, key: str, current_time: Optional[float] = None) -> float:
        """Get the freshness score of a key.

        Args:
            key: Context key.
            current_time: Reference time; defaults to now.

        Returns:
            Freshness in [0.0, 1.0], or 0.0 if not found.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        entry = self._entries.get(key)
        if entry is None:
            return 0.0
        return entry.freshness(current_time)

    def count(self) -> int:
        """Return the number of entries (including expired).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._entries)

    def active_count(self, current_time: Optional[float] = None) -> int:
        """Return the number of non-expired entries.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        now = current_time if current_time is not None else time.time()
        return sum(1 for entry in self._entries.values() if not entry.is_expired(now))

    def keys(self) -> List[str]:
        """Return all keys.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return list(self._entries.keys())

    def decay_weights(self, current_time: Optional[float] = None) -> Dict[str, float]:
        """Return freshness-weighted scores for all keys.

        Args:
            current_time: Reference time; defaults to now.

        Returns:
            Mapping from key to freshness score.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        now = current_time if current_time is not None else time.time()
        return {
            key: entry.freshness(now)
            for key, entry in self._entries.items()
            if not entry.is_expired(now)
        }
