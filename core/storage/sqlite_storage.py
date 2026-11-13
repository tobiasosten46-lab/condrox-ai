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
SQLite storage backend for persistent memory and knowledge.

Uses Python's built-in sqlite3 module — zero external dependencies.
Serializes numpy arrays as JSON arrays for portability.

Mathematical Foundation:
- Insert: T(n) ∈ O(n) for n items (batch insert with executemany).
- Load: T(n) ∈ O(n) for n rows.
- Space: S(n) ∈ O(n * d) where d is average item size.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.storage.storage_interface import StorageBackend


class SQLiteStorage(StorageBackend):
    """
    SQLite-based persistent storage backend.

    Attributes:
        _path: Filesystem path to the SQLite database file.
        _conn: SQLite connection (thread-local).
        _lock: Mutex for thread-safe writes.

    Complexity:
        save_items: T(n) ∈ O(n), S(n) ∈ O(n).
        load_items: T(n) ∈ O(n), S(n) ∈ O(n).
    """

    __slots__ = ("_path", "_local", "_lock", "_closed")
    _TABLE_RE: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, path: str | Path = "condrox.db") -> None:
        """Initialize the SQLite storage backend.

        Args:
            path: Filesystem path to the database file. Use ":memory:" for
                  an in-memory database (testing only).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._path: str = str(path)
        if self._path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self._path))
            os.makedirs(parent, exist_ok=True)
        self._local: threading.local = threading.local()
        self._lock: threading.Lock = threading.Lock()
        self._closed: bool = False

    @property
    def _connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection.

        Creates a new connection per thread on first access.

        Complexity: T(n) ∈ O(1) after first call.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    @classmethod
    def _validate_table_name(cls, table: str) -> None:
        """Validate a table name to prevent SQL injection.

        Only alphanumeric names starting with a letter or underscore are
        accepted. All public methods that build SQL with table names call
        this before executing.

        Args:
            table: Table name to validate.

        Raises:
            ValueError: If the table name contains unsafe characters.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not isinstance(table, str) or not cls._TABLE_RE.match(table):
            raise ValueError(f"Invalid SQLite table name: {table!r}")

    def _ensure_table(self, table: str) -> None:
        """Create the items table if it does not exist.

        Args:
            table: Table name to create.

        Complexity: T(n) ∈ O(1).
        """
        self._validate_table_name(table)
        self._connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "  item_id TEXT PRIMARY KEY,"
            "  data TEXT NOT NULL"
            ")"
        )
        self._connection.commit()

    def save_items(self, table: str, items: List[Dict[str, Any]]) -> None:
        """Persist a batch of serialized items.

        Uses INSERT OR REPLACE to handle upserts.

        Args:
            table: Target table name.
            items: List of item dictionaries. Each must have an "item_id" key.

        Complexity: T(n) ∈ O(n) for n items.
        """
        if self._closed:
            raise RuntimeError("Storage backend is closed")
        if not items:
            return

        self._ensure_table(table)
        rows = [(item.get("item_id", ""), json.dumps(item, default=str)) for item in items]

        with self._lock:
            self._connection.executemany(
                f"INSERT OR REPLACE INTO {table} (item_id, data) VALUES (?, ?)",
                rows,
            )
            self._connection.commit()

    def load_items(self, table: str) -> List[Dict[str, Any]]:
        """Load all items from the given table.

        Args:
            table: Source table name.

        Returns:
            List of item dictionaries.

        Complexity: T(n) ∈ O(n) for n stored items.
        """
        if self._closed:
            raise RuntimeError("Storage backend is closed")

        self._ensure_table(table)
        cursor = self._connection.execute(f"SELECT data FROM {table}")
        rows = cursor.fetchall()
        return [json.loads(row["data"]) for row in rows]

    def clear_table(self, table: str) -> int:
        """Remove all items from the given table.

        Args:
            table: Target table name.

        Returns:
            Number of items removed.

        Complexity: T(n) ∈ O(n) for n stored items.
        """
        if self._closed:
            raise RuntimeError("Storage backend is closed")

        self._ensure_table(table)
        cursor = self._connection.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        with self._lock:
            self._connection.execute(f"DELETE FROM {table}")
            self._connection.commit()
        return count

    def count_items(self, table: str) -> int:
        """Return the number of items in the given table.

        Args:
            table: Target table name.

        Returns:
            Item count.

        Complexity: T(n) ∈ O(1).
        """
        if self._closed:
            raise RuntimeError("Storage backend is closed")

        self._ensure_table(table)
        cursor = self._connection.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close all database connections.

        Complexity: T(n) ∈ O(1).
        """
        if self._closed:
            return
        self._closed = True
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
