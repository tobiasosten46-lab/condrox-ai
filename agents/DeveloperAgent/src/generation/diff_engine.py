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
Diff Engine — minimal patch generation for DeveloperAgent v3.0.

Generates unified diff format patches between original and modified code.
Supports line-level diffing with context lines.

Complexity:
    diff: T(n) in O(n * m) where n, m = line counts (LCS-based),
          S(n) in O(hunks).

Invariants:
    - Output is valid unified diff format.
    - Context lines configurable (default 3).
    - Hunk headers have correct line numbers.

Edge Cases:
    - Identical code -> empty diff.
    - Empty original -> all additions.
    - Empty modified -> all deletions.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import List, Tuple


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DiffHunk:
    """
    Immutable diff hunk.

    Complexity: T(n) in O(1), S(n) in O(lines).
    Invariants: old_start, new_start >= 0; old_count, new_count >= 0.
    Edge Cases: empty hunk -> all counts zero.
    """

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiffResult:
    """
    Immutable diff result.

    Complexity: T(n) in O(1), S(n) in O(hunks + patch).
    Invariants: hunks sorted by line number.
    Edge Cases: identical code -> empty hunks, empty patch.
    """

    hunks: Tuple[DiffHunk, ...]
    patch: str
    lines_added: int
    lines_removed: int
    lines_context: int
    is_empty: bool


# ─── Diff Engine ─────────────────────────────────────────────────────────────


class DiffEngine:
    """
    Minimal patch generation engine for DeveloperAgent v3.0.

    Uses Python's difflib for LCS-based diffing and generates unified
    diff format output with configurable context lines.

    Complexity:
        diff: T(n) in O(n * m) where n, m = line counts, S(n) in O(hunks).
    Invariants:
        - Output is valid unified diff format.
        - Hunk headers have correct line numbers.
    Edge Cases:
        - Identical code -> empty diff.
        - Empty inputs -> all additions or deletions.
    """

    __slots__ = ("_context_lines", "_truncation_threshold")

    def __init__(
        self,
        context_lines: int = 3,
        truncation_threshold: int = 10000,
    ) -> None:
        """
        Initialize DiffEngine with context line count.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: context_lines=0 -> no context in diff.
        """
        self._context_lines: int = context_lines
        self._truncation_threshold: int = truncation_threshold

    # ─── Public API ──────────────────────────────────────────────────────────

    def diff(
        self,
        original: str,
        modified: str,
        original_name: str = "original",
        modified_name: str = "modified",
    ) -> DiffResult:
        """
        Generate a unified diff between original and modified code.

        Complexity: T(n) in O(n * m) where n, m = line counts,
                    S(n) in O(hunks).
        Invariants: output is valid unified diff format.
        Edge Cases: identical code -> empty diff.
        """
        original_lines: List[str] = original.splitlines(keepends=True)
        modified_lines: List[str] = modified.splitlines(keepends=True)

        if original == modified:
            return DiffResult(
                hunks=(),
                patch="",
                lines_added=0,
                lines_removed=0,
                lines_context=0,
                is_empty=True,
            )

        diff_lines: List[str] = list(difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=original_name,
            tofile=modified_name,
            n=self._context_lines,
        ))

        patch: str = "".join(diff_lines)

        added: int = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        removed: int = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
        context: int = sum(1 for line in diff_lines if line.startswith(" "))

        hunks: Tuple[DiffHunk, ...] = self._parse_hunks(diff_lines)

        return DiffResult(
            hunks=hunks,
            patch=patch,
            lines_added=added,
            lines_removed=removed,
            lines_context=context,
            is_empty=False,
        )

    def apply_patch(
        self,
        original: str,
        patch: str,
    ) -> str:
        """
        Apply a unified diff patch to original code.

        Complexity: T(n) in O(n) where n = patch lines, S(n) in O(output).
        Invariants: returns modified code if patch is valid.
        Edge Cases: invalid patch -> original returned unchanged.
        """
        original_lines: List[str] = original.splitlines(keepends=True)
        patch_lines: List[str] = patch.splitlines(keepends=True)

        result: List[str] = []
        i: int = 0
        patch_idx: int = 0

        while patch_idx < len(patch_lines):
            line: str = patch_lines[patch_idx]

            if line.startswith("@@"):
                hunk_info = self._parse_hunk_header(line)
                if hunk_info is None:
                    patch_idx += 1
                    continue

                old_start, _, _, _ = hunk_info

                while i < old_start - 1 and i < len(original_lines):
                    result.append(original_lines[i])
                    i += 1

                patch_idx += 1
                while patch_idx < len(patch_lines) and not patch_lines[patch_idx].startswith("@@"):
                    hunk_line: str = patch_lines[patch_idx]
                    if hunk_line.startswith(" "):
                        result.append(hunk_line[1:] if len(hunk_line) > 1 else "\n")
                        i += 1
                    elif hunk_line.startswith("+"):
                        content: str = hunk_line[1:]
                        if not content.endswith("\n"):
                            content += "\n"
                        result.append(content)
                    elif hunk_line.startswith("-"):
                        i += 1
                    patch_idx += 1
            else:
                patch_idx += 1

        while i < len(original_lines):
            result.append(original_lines[i])
            i += 1

        return "".join(result)

    def format_summary(self, result: DiffResult) -> str:
        """
        Format diff result as a summary string.

        Complexity: T(n) in O(1), S(n) in O(summary).
        Edge Cases: empty diff -> "No changes".
        """
        if result.is_empty:
            return "No changes"

        return (
            f"Diff: +{result.lines_added} -{result.lines_removed} "
            f"(context: {result.lines_context}, hunks: {len(result.hunks)})"
        )

    # ─── Private ─────────────────────────────────────────────────────────────

    def _parse_hunks(
        self, diff_lines: List[str]
    ) -> Tuple[DiffHunk, ...]:
        """
        Parse diff lines into structured hunks.

        Complexity: T(n) in O(n) where n = diff lines, S(n) in O(hunks).
        Edge Cases: no hunks -> empty tuple.
        """
        hunks: List[DiffHunk] = []
        current_hunk_lines: List[str] = []
        current_header: Tuple[int, int, int, int] = (0, 0, 0, 0)

        for line in diff_lines:
            if line.startswith("@@"):
                if current_hunk_lines:
                    hunks.append(DiffHunk(
                        old_start=current_header[0],
                        old_count=current_header[1],
                        new_start=current_header[2],
                        new_count=current_header[3],
                        lines=tuple(current_hunk_lines),
                    ))
                    current_hunk_lines = []

                header = self._parse_hunk_header(line)
                if header:
                    current_header = header
            elif line.startswith((" ", "+", "-")) and not line.startswith(("+++", "---")):
                current_hunk_lines.append(line.rstrip("\n"))

        if current_hunk_lines:
            hunks.append(DiffHunk(
                old_start=current_header[0],
                old_count=current_header[1],
                new_start=current_header[2],
                new_count=current_header[3],
                lines=tuple(current_hunk_lines),
            ))

        return tuple(hunks)

    @staticmethod
    def _parse_hunk_header(line: str) -> Tuple[int, int, int, int]:
        """
        Parse a hunk header line (@@ -start,count +start,count @@).

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: malformed header -> None.
        """
        import re

        match = re.match(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line
        )
        if not match:
            return None

        old_start: int = int(match.group(1))
        old_count: int = int(match.group(2)) if match.group(2) else 1
        new_start: int = int(match.group(3))
        new_count: int = int(match.group(4)) if match.group(4) else 1

        return (old_start, old_count, new_start, new_count)
