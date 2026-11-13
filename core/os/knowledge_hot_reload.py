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
AI-OS Knowledge Hot-Reload — incremental file watcher with cache.

Monitors the knowledge root directory for new/modified files and
incrementally ingests them into the KnowledgeBackend without restarting
the system. Uses file modification timestamps + SHA-256 hashes for
change detection.

Mathematical Foundation:
- Scan: O(d * f) for d domains, f files per domain.
- Hash: O(b) per file for b bytes.
- Ingest: O(n * c) for n new chunks, c avg chunk length.
- Memory (cache): O(f) for f tracked files.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.os.event_bus import AIEvent, EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class FileTrackingRecord:
    """
    Immutable record tracking a single file's load state.

    Attributes:
        path: Absolute file path.
        domain: Domain label.
        mtime: Last modification time (epoch seconds).
        file_hash: SHA-256 hash of file content (first 8192 bytes).
        items_loaded: Number of knowledge items ingested from this file.
        last_loaded: Timestamp of last successful ingestion.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    path: str
    domain: str
    mtime: float
    file_hash: str
    items_loaded: int
    last_loaded: float


class KnowledgeHotReloader:
    """
    Incremental knowledge file watcher with change detection.

    Runs a background daemon thread that periodically scans the knowledge
    root directory, detects new or modified files via mtime + content hash,
    and ingests only the changed content into the KnowledgeBackend.

    Attributes:
        _knowledge_root: Root directory containing domain subdirectories.
        _backend: KnowledgeBackend to ingest into.
        _event_bus: Event bus for publishing reload events.
        _cache: Mapping from file path to FileTrackingRecord.
        _lock: Thread lock for cache access.
        _thread: Background watcher thread.
        _running: Whether the watcher is active.
        _interval: Scan interval in seconds.
        _max_chunk_chars: Max characters per chunk.
        _max_chunks_per_file: Max chunks to load per file.
        _total_reloads: Total reload operations performed.
        _total_items_reloaded: Total items ingested across all reloads.

    Complexity:
        scan: O(d * f * b) for d domains, f files, b avg file bytes (hashing).
        _watch_loop: O(scan) per iteration.
    """

    __slots__ = (
        "_knowledge_root",
        "_backend",
        "_event_bus",
        "_cache",
        "_lock",
        "_thread",
        "_running",
        "_interval",
        "_max_chunk_chars",
        "_max_chunks_per_file",
        "_total_reloads",
        "_total_items_reloaded",
    )

    def __init__(
        self,
        knowledge_root: str | Path,
        backend: KnowledgeBackend,
        event_bus: EventBus,
        interval: float = 30.0,
        max_chunk_chars: int = 2000,
        max_chunks_per_file: int = 50,
    ) -> None:
        """Initialize the hot-reloader.

        Args:
            knowledge_root: Root directory with domain subdirectories.
            backend: KnowledgeBackend to ingest into.
            event_bus: Event bus for publishing events.
            interval: Scan interval in seconds.
            max_chunk_chars: Max characters per chunk.
            max_chunks_per_file: Max chunks per file.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._knowledge_root: Path = Path(knowledge_root)
        self._backend: KnowledgeBackend = backend
        self._event_bus: EventBus = event_bus
        self._cache: Dict[str, FileTrackingRecord] = {}
        self._lock: threading.Lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._interval: float = interval
        self._max_chunk_chars: int = max_chunk_chars
        self._max_chunks_per_file: int = max_chunks_per_file
        self._total_reloads: int = 0
        self._total_items_reloaded: int = 0

    def start(self) -> None:
        """Start the background watcher thread.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="ai-os-knowledge-watcher"
        )
        self._thread.start()

        self._event_bus.publish(AIEvent(
            event_type=EventType.KNOWLEDGE,
            source="hot_reloader",
            message=f"Knowledge hot-reloader started (interval={self._interval}s)",
            severity="INFO",
            data={"interval": self._interval, "root": str(self._knowledge_root)},
        ))

    def stop(self) -> None:
        """Stop the background watcher thread.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._event_bus.publish(AIEvent(
            event_type=EventType.KNOWLEDGE,
            source="hot_reloader",
            message="Knowledge hot-reloader stopped",
            severity="INFO",
            data={
                "total_reloads": self._total_reloads,
                "total_items_reloaded": self._total_items_reloaded,
            },
        ))

    def scan_once(self) -> int:
        """Perform a single incremental scan and reload.

        Scans all domain directories, detects new/modified files,
        and ingests only changed content.

        Returns:
            Number of new knowledge items ingested.

        Complexity: T(n) ∈ O(d * f * b) for d domains, f files, b avg bytes.
        """
        if not self._knowledge_root.exists():
            return 0

        total_new_items: int = 0
        seen_paths: Set[str] = set()

        for domain_dir in sorted(self._knowledge_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name
            if domain_name in ("downloads", "meta"):
                continue

            for file_path in sorted(domain_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in (".txt", ".json", ".jsonl"):
                    continue

                path_str = str(file_path.resolve())
                seen_paths.add(path_str)

                try:
                    stat = file_path.stat()
                    mtime = stat.st_mtime

                    with self._lock:
                        cached = self._cache.get(path_str)

                    # Skip if unchanged (same mtime)
                    if cached and cached.mtime == mtime:
                        continue

                    # Hash first 8192 bytes for change detection
                    file_hash = self._hash_file(file_path)

                    # Skip if same hash (mtime changed but content didn't)
                    if cached and cached.file_hash == file_hash:
                        # Update mtime in cache
                        with self._lock:
                            self._cache[path_str] = FileTrackingRecord(
                                path=path_str,
                                domain=domain_name,
                                mtime=mtime,
                                file_hash=file_hash,
                                items_loaded=cached.items_loaded,
                                last_loaded=cached.last_loaded,
                            )
                        continue

                    # File is new or modified — ingest
                    items = self._ingest_file(file_path, domain_name)
                    total_new_items += items

                    with self._lock:
                        self._cache[path_str] = FileTrackingRecord(
                            path=path_str,
                            domain=domain_name,
                            mtime=mtime,
                            file_hash=file_hash,
                            items_loaded=items,
                            last_loaded=time.time(),
                        )

                    logger.info(
                        "Hot-reloaded %d items from %s (domain=%s)",
                        items, file_path.name, domain_name,
                    )

                except Exception as exc:
                    logger.warning("Hot-reload failed for %s: %s", file_path, exc)

        # Remove deleted files from cache
        with self._lock:
            stale = [p for p in self._cache if p not in seen_paths]
            for p in stale:
                del self._cache[p]

        if total_new_items > 0:
            self._total_reloads += 1
            self._total_items_reloaded += total_new_items

            self._event_bus.publish(AIEvent(
                event_type=EventType.KNOWLEDGE,
                source="hot_reloader",
                message=f"Hot-reload complete: {total_new_items} new items ingested",
                severity="INFO",
                data={
                    "new_items": total_new_items,
                    "total_reloads": self._total_reloads,
                    "total_items_reloaded": self._total_items_reloaded,
                    "tracked_files": len(self._cache),
                },
            ))

        # Ensure batched/disk backends persist hot-reloaded changes immediately.
        if hasattr(self._backend, "flush"):
            self._backend.flush()

        return total_new_items

    def _watch_loop(self) -> None:
        """Background watcher loop — scans at intervals.

        Complexity: T(n) ∈ O(scan * i) for i iterations.
        """
        while self._running:
            try:
                self.scan_once()
            except Exception as exc:
                logger.error("Knowledge watcher error: %s", exc)
                self._event_bus.publish(AIEvent(
                    event_type=EventType.KNOWLEDGE,
                    source="hot_reloader",
                    message=f"Watcher error: {exc}",
                    severity="ERROR",
                ))
            time.sleep(self._interval)

    def _hash_file(self, file_path: Path) -> str:
        """Compute SHA-256 hash of first 8192 bytes of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest string.

        Complexity: T(n) ∈ O(b) for b bytes read (max 8192).
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                hasher.update(f.read(8192))
        except Exception:
            return ""
        return hasher.hexdigest()

    def _ingest_file(self, file_path: Path, domain: str) -> int:
        """Ingest chunks from a single file into the backend.

        Args:
            file_path: Path to the file.
            domain: Domain label.

        Returns:
            Number of items ingested.

        Complexity: T(n) ∈ O(c * e) for c chunks, e embedding cost.
        """
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return 0

        if not text.strip():
            return 0

        chunks = self._split_chunks(text)
        loaded: int = 0

        for chunk in chunks:
            if loaded >= self._max_chunks_per_file:
                break
            clean = chunk.strip()[: self._max_chunk_chars]
            if len(clean) < 20:
                continue
            self._backend.ingest(
                content=clean,
                domain=domain,
                confidence=0.8,
                metadata={"source": str(file_path.name), "hot_reload": True},
            )
            loaded += 1

        return loaded

    def _split_chunks(self, text: str) -> List[str]:
        """Split text into chunks on double newlines or line groups.

        Args:
            text: Raw file text.

        Returns:
            List of text chunks.

        Complexity: T(n) ∈ O(n) for text length n, S(n) ∈ O(n).
        """
        if "\n\n" in text:
            return text.split("\n\n")

        lines = text.split("\n")
        chunks: List[str] = []
        current: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                continue
            current.append(line)
            if len(current) >= 5:
                chunks.append("\n".join(current))
                current = []

        if current:
            chunks.append("\n".join(current))

        return [c for c in chunks if len(c.strip()) >= 20]

    def get_tracked_files(self) -> List[Dict[str, object]]:
        """Get list of tracked files.

        Returns:
            List of file tracking dicts.

        Complexity: T(n) ∈ O(f), S(n) ∈ O(f) for f tracked files.
        """
        with self._lock:
            return [
                {
                    "path": r.path,
                    "domain": r.domain,
                    "mtime": r.mtime,
                    "file_hash": r.file_hash[:16],
                    "items_loaded": r.items_loaded,
                    "last_loaded": r.last_loaded,
                }
                for r in self._cache.values()
            ]

    def get_status(self) -> Dict[str, object]:
        """Get hot-reloader status.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return {
                "running": self._running,
                "interval": self._interval,
                "knowledge_root": str(self._knowledge_root),
                "tracked_files": len(self._cache),
                "total_reloads": self._total_reloads,
                "total_items_reloaded": self._total_items_reloaded,
            }

    def prime_cache(self) -> int:
        """Prime the cache with current file states without ingesting.

        Call this after an initial full load to prevent re-ingesting
        already-loaded content on the first scan.

        Returns:
            Number of files primed into cache.

        Complexity: T(n) ∈ O(d * f * b) for d domains, f files, b bytes.
        """
        if not self._knowledge_root.exists():
            return 0

        primed: int = 0

        for domain_dir in sorted(self._knowledge_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name
            if domain_name in ("downloads", "meta"):
                continue

            for file_path in sorted(domain_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix not in (".txt", ".json", ".jsonl"):
                    continue

                path_str = str(file_path.resolve())
                try:
                    stat = file_path.stat()
                    mtime = stat.st_mtime
                    file_hash = self._hash_file(file_path)

                    with self._lock:
                        self._cache[path_str] = FileTrackingRecord(
                            path=path_str,
                            domain=domain_name,
                            mtime=mtime,
                            file_hash=file_hash,
                            items_loaded=0,
                            last_loaded=0.0,
                        )
                    primed += 1
                except Exception:
                    pass

        return primed
