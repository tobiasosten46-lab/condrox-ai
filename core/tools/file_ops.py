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
Safe file operations tool for Condrox AI.

All paths are validated against a root directory to prevent path traversal.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from core.security.security_engine import SecurityEngine


class FileOps:
    """
    Safe file operations with path validation.

    Complexity: T(op) ∈ O(1) for most ops, S(op) ∈ O(content)
    Invariants: all paths resolved within root_dir
    Edge Cases: path traversal → ValueError
    """

    __slots__ = ("_root", "_security")

    def __init__(self, root_dir: str, security: SecurityEngine | None = None) -> None:
        """
        Initialize FileOps with a root directory.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1)
        Edge Cases: root_dir not exists → created on first write
        """
        self._root: str = os.path.realpath(root_dir)
        self._security: SecurityEngine | None = security

    def _validate_path(self, path: str) -> str:
        """
        Validate and resolve path within root.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1)
        Edge Cases: traversal attempt → ValueError
        """
        full: str = os.path.realpath(os.path.join(self._root, path))
        if not full.startswith(self._root):
            raise ValueError(f"Path traversal denied: {path}")
        return full

    async def read(self, path: str) -> str:
        """
        Read file content.

        Complexity: T(n) ∈ O(n) where n = file size, S(n) ∈ O(content)
        Edge Cases: missing file → FileNotFoundError
        """
        full: str = self._validate_path(path)
        with open(full, "r", encoding="utf-8") as f:
            return f.read()

    async def write(self, path: str, content: str) -> int:
        """
        Write content to file.

        Complexity: T(n) ∈ O(n) where n = content size, S(n) ∈ O(content)
        Edge Cases: parent dir missing → created
        """
        full: str = self._validate_path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            return f.write(content)

    async def edit(self, path: str, old: str, new: str) -> bool:
        """
        Replace text in file.

        Complexity: T(n) ∈ O(n) where n = file size, S(n) ∈ O(content)
        Edge Cases: old not found → False
        """
        content: str = await self.read(path)
        if old not in content:
            return False
        updated: str = content.replace(old, new, 1)
        await self.write(path, updated)
        return True

    async def list_dir(self, path: str = ".") -> Tuple[Dict[str, Any], ...]:
        """
        List directory contents.

        Complexity: T(n) ∈ O(n) where n = entries, S(n) ∈ O(entries)
        Edge Cases: empty dir → empty tuple
        """
        full: str = self._validate_path(path)
        items: List[Dict[str, Any]] = []
        for entry in os.listdir(full):
            entry_path: str = os.path.join(full, entry)
            items.append({
                "name": entry,
                "type": "dir" if os.path.isdir(entry_path) else "file",
                "size": os.path.getsize(entry_path) if os.path.isfile(entry_path) else 0,
            })
        return tuple(items)

    async def delete(self, path: str) -> bool:
        """
        Delete a file.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1)
        Edge Cases: missing file → FileNotFoundError
        """
        full: str = self._validate_path(path)
        os.remove(full)
        return True
