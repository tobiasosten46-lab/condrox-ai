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
User authentication database for the Condrox dashboard.

Uses SQLite with PBKDF2-HMAC-SHA256 password hashing.
Zero external dependencies — uses only Python stdlib.

Mathematical Foundation:
- PBKDF2: O(i * n) where i = iterations, n = key length.
- User lookup: O(1) via indexed username column.
- Token validation: O(1) via in-memory token set lookup.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class User:
    """Immutable user record.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    id: int
    username: str
    role: str
    created_at: float


class UserDatabase:
    """
    SQLite-backed user database with PBKDF2 password hashing.

    Attributes:
        _path: Path to the SQLite database file.
        _lock: Mutex for thread-safe writes.

    Complexity:
        create_user: T(n) ∈ O(1), S(n) ∈ O(1).
        verify_user: T(n) ∈ O(i * n) for PBKDF2 iterations i, key length n.
        create_token: T(n) ∈ O(1), S(n) ∈ O(1).
        validate_token: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_path", "_lock", "_local", "_tokens")

    def __init__(self, path: str | Path = "Databases/condrox_users.db") -> None:
        """Initialize the user database and create tables if needed.

        Args:
            path: Filesystem path to the SQLite database file.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._path: str = str(path)
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        self._lock: threading.Lock = threading.Lock()
        self._local: threading.local = threading.local()
        self._tokens: dict[str, tuple[int, float]] = {}

        self._init_db()

    @property
    def _connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection.

        Complexity: T(n) ∈ O(1) after first call.
        """
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Create tables if they don't exist.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            conn = sqlite3.connect(self._path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'admin',
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
            conn.close()

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with salt using PBKDF2-HMAC-SHA256.

        Args:
            password: Plaintext password.
            salt: Hex-encoded salt string.

        Returns:
            Hex-encoded hash string.

        Complexity: T(n) ∈ O(i * k) for i iterations, k key length.
        """
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            100_000,
            32,
        ).hex()

    def create_user(self, username: str, password: str, role: str = "admin") -> User:
        """Create a new user.

        Args:
            username: Unique username.
            password: Plaintext password.
            role: User role (default 'admin').

        Returns:
            User record.

        Raises:
            ValueError: If username already exists.

        Complexity: T(n) ∈ O(i * k), S(n) ∈ O(1).
        """
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        created_at = time.time()

        with self._lock:
            conn = self._connection
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash, password_salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                    (username, password_hash, salt, role, created_at),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"User '{username}' already exists")

            row = conn.execute(
                "SELECT id, username, role, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        return User(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            created_at=row["created_at"],
        )

    def verify_user(self, username: str, password: str) -> Optional[User]:
        """Verify a user's credentials.

        Args:
            username: Username to verify.
            password: Plaintext password.

        Returns:
            User record if verified, None otherwise.

        Complexity: T(n) ∈ O(i * k) for PBKDF2 verification.
        """
        conn = self._connection
        row = conn.execute(
            "SELECT id, username, password_hash, password_salt, role, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None:
            return None

        expected_hash = row["password_hash"]
        computed_hash = self._hash_password(password, row["password_salt"])

        if not hmac.compare_digest(expected_hash, computed_hash):
            return None

        return User(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            created_at=row["created_at"],
        )

    def create_token(self, user_id: int) -> str:
        """Create a session token for a user.

        Args:
            user_id: User ID to create token for.

        Returns:
            Token string.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        token = secrets.token_urlsafe(32)
        self._tokens[token] = (user_id, time.time() + 86400)  # 24h expiry
        return token

    def validate_token(self, token: str) -> Optional[int]:
        """Validate a session token.

        Args:
            token: Token string to validate.

        Returns:
            User ID if valid, None otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        entry = self._tokens.get(token)
        if entry is None:
            return None
        user_id, expires = entry
        if time.time() > expires:
            del self._tokens[token]
            return None
        return user_id

    def revoke_token(self, token: str) -> None:
        """Revoke a session token.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._tokens.pop(token, None)

    def user_exists(self, username: str) -> bool:
        """Check if a user exists.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        conn = self._connection
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        """Close all connections.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
