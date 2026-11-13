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
Git operations tool for Condrox AI.

Wraps git commands via ShellExec.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from core.tools.shell import ShellExec


class GitOps:
    """
    Git operations via restricted shell.

    Complexity: T(op) ∈ O(git_op), S(op) ∈ O(output)
    Invariants: all commands go through ShellExec safety checks
    Edge Cases: not a git repo → RuntimeError
    """

    __slots__ = ("_shell",)

    def __init__(self, shell: ShellExec) -> None:
        self._shell: ShellExec = shell

    async def status(self, cwd: str = ".") -> str:
        """Get git status. Complexity: T(n) ∈ O(1)."""
        return await self._shell.execute_safe("git status --short", cwd)

    async def diff(self, cwd: str = ".") -> str:
        """Get git diff. Complexity: T(n) ∈ O(changes)."""
        return await self._shell.execute_safe("git diff", cwd)

    async def add(self, files: str = ".", cwd: str = ".") -> str:
        """Stage files. Complexity: T(n) ∈ O(files)."""
        return await self._shell.execute_safe(f"git add {files}", cwd)

    async def commit(self, message: str, cwd: str = ".") -> str:
        """Commit changes. Complexity: T(n) ∈ O(1)."""
        return await self._shell.execute_safe(
            f'git commit -m "{message}"', cwd
        )

    async def log(self, count: int = 10, cwd: str = ".") -> str:
        """Get commit log. Complexity: T(n) ∈ O(count)."""
        return await self._shell.execute_safe(
            f"git log --oneline -{count}", cwd
        )

    async def branch(self, cwd: str = ".") -> str:
        """List branches. Complexity: T(n) ∈ O(branches)."""
        return await self._shell.execute_safe("git branch", cwd)

    async def checkout(self, branch: str, cwd: str = ".") -> str:
        """Checkout branch. Complexity: T(n) ∈ O(1)."""
        return await self._shell.execute_safe(f"git checkout {branch}", cwd)
