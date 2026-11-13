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
scanner.py — Condrox Code Quality System

Scans the project tree and returns metadata for every code file.

Complexity: T(n) ∈ O(n * m), n = files, m = average file length.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Metadata for a scanned file."""

    path: Path
    relative: str
    extension: str
    lines: int
    size_bytes: int
    issues: List[str] = field(default_factory=list)


BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".webm",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".db", ".sqlite", ".pkl", ".pickle", ".npy",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".exe", ".dll", ".so", ".dylib",
}


def should_skip(path: Path, skip_patterns: Iterable[str] | None = None) -> bool:
    """Return True for directories/files that should not be scanned."""
    skip = set(skip_patterns or {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        "dist",
        "dist-electron",
        "Archive",
        "runtime",
        "knowledge",
        "downloads",
    })
    parts = {p.lower() for p in path.parts}
    if parts & skip:
        return True
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    return False


def scan_project(root: str | Path) -> List[FileInfo]:
    """Walk the project and collect file metadata."""
    root_path = Path(root).resolve()
    files: List[FileInfo] = []

    for dirpath, _dirnames, filenames in os.walk(root_path):
        dir_path = Path(dirpath)
        if should_skip(dir_path.relative_to(root_path)):
            continue
        for filename in filenames:
            file_path = dir_path / filename
            rel = file_path.relative_to(root_path)
            if should_skip(rel):
                continue
            try:
                size = file_path.stat().st_size
                lines = _count_lines(file_path)
            except OSError:
                continue
            files.append(
                FileInfo(
                    path=file_path,
                    relative=str(rel),
                    extension=file_path.suffix.lower(),
                    lines=lines,
                    size_bytes=size,
                )
            )

    return sorted(files, key=lambda f: f.relative)


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, UnicodeDecodeError):
        return 0
