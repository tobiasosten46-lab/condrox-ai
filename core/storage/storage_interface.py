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
Abstract storage backend interface.

Defines the contract for persistence implementations (SQLite, PostgreSQL,
file-based, etc.).

Mathematical Foundation:
- All operations are O(n) for n items in a batch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageBackend(ABC):
    """
    Abstract storage backend for persisting memory and knowledge items.

    Implementations must be thread-safe if used in concurrent contexts.

    Complexity:
        save_items: T(n) ∈ O(n), S(n) ∈ O(n).
        load_items: T(n) ∈ O(n), S(n) ∈ O(n).
    """

    @abstractmethod
    def save_items(self, table: str, items: List[Dict[str, Any]]) -> None:
        """Persist a batch of serialized items to the given table.

        Args:
            table: Target table/collection name.
            items: List of item dictionaries with serialized fields.

        Complexity: T(n) ∈ O(n) for n items.
        """

    @abstractmethod
    def load_items(self, table: str) -> List[Dict[str, Any]]:
        """Load all items from the given table.

        Args:
            table: Source table/collection name.

        Returns:
            List of item dictionaries with deserialized fields.

        Complexity: T(n) ∈ O(n) for n stored items.
        """

    @abstractmethod
    def clear_table(self, table: str) -> int:
        """Remove all items from the given table.

        Args:
            table: Target table/collection name.

        Returns:
            Number of items removed.

        Complexity: T(n) ∈ O(n) for n stored items.
        """

    @abstractmethod
    def count_items(self, table: str) -> int:
        """Return the number of items in the given table.

        Args:
            table: Target table/collection name.

        Returns:
            Item count.

        Complexity: T(n) ∈ O(1) for indexed tables.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend.

        Complexity: T(n) ∈ O(1).
        """
