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
File Watcher — monitors project files for changes and triggers re-indexing.

Uses polling-based detection (mtime + size hash) for cross-platform
compatibility without native filesystem watch dependencies.

Complexity:
    scan: T(n) in O(f) where f = watched files, S(n) in O(events).
    add_watch: T(n) in O(1), S(n) in O(1).
    remove_watch: T(n) in O(1), S(n) in O(1).

Invariants:
    - Only changed files generate events (mtime or size delta).
    - Deleted files generate DELETED events.
    - No duplicate events for same scan cycle.

Edge Cases:
    - File deleted between scans -> DELETED event.
    - File unchanged -> no event.
    - Watch path missing -> no events.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ─── Enums ───────────────────────────────────────────────────────────────────


class WatchEventType(Enum):
    """
    Type of file system event detected by watcher.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WatchEvent:
    """
    Immutable file watch event.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: event_type is a valid WatchEventType.
    Edge Cases: file_path may no longer exist (DELETED events).
    """

    event_type: WatchEventType
    file_path: str
    timestamp: float
    size: int
    mtime: float


# ─── File Watcher ────────────────────────────────────────────────────────────


class FileWatcher:
    """
    Polling-based file watcher for RAG auto-re-indexing.

    Tracks file mtime + size for each watched path. On each scan,
    compares current state to previous state and emits events for
    created, modified, and deleted files.

    Complexity:
        scan: T(n) in O(f) where f = watched files, S(n) in O(events).
        add_watch: T(n) in O(1), S(n) in O(1).
        remove_watch: T(n) in O(1), S(n) in O(1).
    Invariants:
        - Only changed files generate events.
        - Deleted files generate DELETED events.
        - No duplicate events per scan cycle.
    Edge Cases:
        - File deleted between scans -> DELETED event.
        - Watch path missing -> no events for that path.
        - No watches -> scan returns empty tuple.
    """

    __slots__ = (
        "_watches",
        "_file_states",
        "_extensions",
        "_last_scan_time",
    )

    _DEFAULT_EXTENSIONS: frozenset[str] = frozenset({
        ".py", ".ts", ".js", ".tsx", ".jsx",
        ".md", ".txt", ".json", ".yaml", ".yml",
    })

    def __init__(
        self,
        extensions: Optional[frozenset[str]] = None,
    ) -> None:
        """
        Initialize FileWatcher with file extensions to monitor.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: extensions=None -> default set used.
        """
        self._watches: Set[str] = set()
        self._file_states: Dict[str, Tuple[int, float]] = {}
        self._extensions: frozenset[str] = (
            extensions if extensions is not None else self._DEFAULT_EXTENSIONS
        )
        self._last_scan_time: float = 0.0

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def watch_count(self) -> int:
        """
        Number of watched paths (directories).

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no watches -> 0.
        """
        return len(self._watches)

    @property
    def tracked_file_count(self) -> int:
        """
        Number of currently tracked files.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no files tracked -> 0.
        """
        return len(self._file_states)

    def add_watch(self, path: str) -> None:
        """
        Add a directory or file to watch list.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: path missing -> added but no files tracked until exists.
        """
        self._watches.add(os.path.realpath(path))

    def remove_watch(self, path: str) -> bool:
        """
        Remove a path from watch list.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: path not watched -> False.
        """
        real_path: str = os.path.realpath(path)
        if real_path in self._watches:
            self._watches.discard(real_path)
            return True
        return False

    def scan(self) -> Tuple[WatchEvent, ...]:
        """
        Scan all watched paths and emit events for changes.

        Complexity: T(n) in O(f) where f = tracked + new files,
                    S(n) in O(events).
        Invariants: events sorted by timestamp.
        Edge Cases: no watches -> empty tuple.
        """
        if not self._watches:
            return ()

        self._last_scan_time = time.time()
        events: List[WatchEvent] = []
        current_files: Dict[str, Tuple[int, float]] = {}

        for watch_path in self._watches:
            files: List[str] = self._enumerate_files(watch_path)
            for file_path in files:
                try:
                    stat = os.stat(file_path)
                    current_files[file_path] = (stat.st_size, stat.st_mtime)
                except OSError:
                    continue

        for file_path, (size, mtime) in current_files.items():
            if file_path not in self._file_states:
                events.append(WatchEvent(
                    event_type=WatchEventType.CREATED,
                    file_path=file_path,
                    timestamp=self._last_scan_time,
                    size=size,
                    mtime=mtime,
                ))
            else:
                prev_size, prev_mtime = self._file_states[file_path]
                if size != prev_size or mtime != prev_mtime:
                    events.append(WatchEvent(
                        event_type=WatchEventType.MODIFIED,
                        file_path=file_path,
                        timestamp=self._last_scan_time,
                        size=size,
                        mtime=mtime,
                    ))

        for file_path in self._file_states:
            if file_path not in current_files:
                events.append(WatchEvent(
                    event_type=WatchEventType.DELETED,
                    file_path=file_path,
                    timestamp=self._last_scan_time,
                    size=0,
                    mtime=0.0,
                ))

        self._file_states = current_files
        events.sort(key=lambda e: e.timestamp)
        return tuple(events)

    def clear(self) -> None:
        """
        Clear all watches and tracked file states.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no watches -> no-op.
        """
        self._watches.clear()
        self._file_states.clear()

    # ─── Private ─────────────────────────────────────────────────────────────

    def _enumerate_files(self, path: str) -> List[str]:
        """
        Enumerate all watchable files under a path.

        Complexity: T(n) in O(f) where f = files in path, S(n) in O(files).
        Edge Cases: path missing -> empty list; single file -> [path].
        """
        if os.path.isfile(path):
            ext: str = os.path.splitext(path)[1].lower()
            if ext in self._extensions:
                return [path]
            return []

        if not os.path.isdir(path):
            return []

        files: List[str] = []
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self._extensions:
                    files.append(os.path.join(root, filename))
        return files
