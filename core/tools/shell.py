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
Restricted shell execution tool for Condrox AI.

Filters dangerous patterns and enforces timeouts.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import asyncio
import re
from typing import Tuple

_FORBIDDEN: tuple[re.Pattern[str], ...] = (
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"mkfs\."),
    re.compile(r"dd\s+if=.*of=/dev/"),
    re.compile(r":\(\)\{.*\|.*&\};:"),
    re.compile(r">\s*/dev/sda"),
    re.compile(r"chmod\s+777\s+/"),
    re.compile(r"curl\s+.*\|\s*sh"),
    re.compile(r"wget\s+.*\|\s*sh"),
)


class ShellExec:
    """
    Restricted shell execution with pattern filtering and timeout.

    Complexity: T(exec) ∈ O(timeout), S(exec) ∈ O(output)
    Invariants: forbidden patterns never executed
    Edge Cases: timeout → killed, forbidden → ValueError
    """

    __slots__ = ("_timeout",)

    def __init__(self, timeout: int = 30) -> None:
        self._timeout: int = timeout

    def _check_command(self, command: str) -> None:
        """
        Check command against forbidden patterns.

        Complexity: T(n) ∈ O(p) where p = patterns, S(n) ∈ O(1)
        Edge Cases: forbidden pattern → ValueError
        """
        for pattern in _FORBIDDEN:
            if pattern.search(command):
                raise ValueError(f"Forbidden command pattern detected: {command}")

    async def execute(self, command: str, cwd: str = ".") -> Tuple[str, str, int]:
        """
        Execute a shell command with timeout.

        Complexity: T(n) ∈ O(timeout), S(n) ∈ O(output)
        Edge Cases: timeout → killed, returns partial output
        """
        self._check_command(command)

        try:
            proc: asyncio.subprocess.Process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
            return (
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
                proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return "", "Command timed out", -1

    async def execute_safe(self, command: str, cwd: str = ".") -> str:
        """
        Execute and return only stdout, raising on error.

        Complexity: T(n) ∈ O(timeout), S(n) ∈ O(output)
        Edge Cases: non-zero exit → RuntimeError
        """
        stdout, stderr, code = await self.execute(command, cwd)
        if code != 0:
            raise RuntimeError(f"Command failed (exit {code}): {stderr}")
        return stdout
