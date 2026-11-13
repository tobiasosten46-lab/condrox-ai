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
Full-scale knowledge ingestion pipeline for Condrox.

Streams every eligible file under the knowledge root into a DiskKnowledgeBackend
(SQLite + FTS5) so the downloaded corpus becomes searchable without loading it
all into RAM.

Features:
- Walks all subdirectories (excluding meta by default).
- Handles .txt, .json, .jsonl, .md, and .gz variants.
- Accumulates plain-text lines into chunks up to --chunk-size.
- Extracts the `text` field from JSONL and chunks it.
- Parses CSV assertions (e.g., ConceptNet) into text.
- Domain-tagged by directory name.
- Quality-gated (short/junk chunks rejected).
- Resumable: rerunning skips already-ingested content by content hash.
- Space-guarded: stops when the DB drive free space drops below --min-free-gb.
- Optional source deletion: deletes each source file after indexing.

Usage:
    python scripts/ingest_full_knowledge.py
    python scripts/ingest_full_knowledge.py --root data/knowledge --db Databases/full_knowledge.db
    python scripts/ingest_full_knowledge.py --delete-source --min-free-gb 10
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.knowledge.disk_knowledge_backend import DiskKnowledgeBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class OutOfSpace(Exception):
    """Raised when the DB drive drops below the configured free-space threshold."""


class KnowledgeStreamIngester:
    """Stream-ingest an on-disk knowledge corpus into DiskKnowledgeBackend.

    Attributes:
        backend: DiskKnowledgeBackend instance.
        chunk_size: Target max characters per chunk.
        min_chunk_size: Minimum characters for a chunk to be accepted.
        log_interval: Number of chunks between progress logs.
        delete_source: Whether to delete source files after successful indexing.
        min_free_bytes: Minimum free bytes required on the DB drive.
        _stats: Running ingestion statistics.

    Complexity:
        ingest_path: O(f) for file size f.
        run: O(total corpus size).
    """

    __slots__ = (
        "backend",
        "chunk_size",
        "min_chunk_size",
        "log_interval",
        "delete_source",
        "min_free_bytes",
        "_db_drive_root",
        "_stats",
    )

    def __init__(
        self,
        backend: DiskKnowledgeBackend,
        db_path: Path,
        chunk_size: int = 2000,
        min_chunk_size: int = 40,
        log_interval: int = 10000,
        delete_source: bool = False,
        min_free_gb: float = 5.0,
    ) -> None:
        """Initialize the stream ingester.

        Args:
            backend: DiskKnowledgeBackend to ingest into.
            db_path: Path to the output database (used to monitor drive free space).
            chunk_size: Target characters per chunk.
            min_chunk_size: Chunks shorter than this are rejected.
            log_interval: Progress log frequency in chunks.
            delete_source: Delete each source file after it is indexed.
            min_free_gb: Stop when free space on the DB drive drops below this.
        """
        self.backend = backend
        self.chunk_size = max(100, chunk_size)
        self.min_chunk_size = max(10, min_chunk_size)
        self.log_interval = max(1, log_interval)
        self.delete_source = delete_source
        self.min_free_bytes = int(min_free_gb * 1024 * 1024 * 1024)
        self._db_drive_root = Path(db_path.anchor) if db_path.anchor else Path.cwd()
        self._stats: Dict[str, Any] = {
            "files_seen": 0,
            "files_ingested": 0,
            "chunks_ingested": 0,
            "chunks_rejected": 0,
            "bytes_seen": 0,
            "sources_deleted": 0,
            "domains": set(),
            "start_time": time.time(),
        }

    def run(self, root: str | Path, skip_dirs: Optional[Set[str]] = None) -> None:
        """Ingest all eligible files under root, sorted by size (smallest first).

        Args:
            root: Knowledge root directory.
            skip_dirs: Directory names to skip (default meta only).
        """
        root_path = Path(root)
        skip = skip_dirs or {"meta"}
        if not root_path.exists():
            logger.error("Knowledge root does not exist: %s", root_path)
            return

        # Collect all eligible files first so we can process smallest first.
        # This maximizes domain coverage before large sources (C4, world_knowledge)
        # consume the remaining disk budget.
        files: List[Tuple[Path, str, int]] = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in skip]
            domain = (
                Path(dirpath).relative_to(root_path).parts[0]
                if dirpath != str(root_path)
                else "general"
            )
            for filename in filenames:
                file_path = Path(dirpath) / filename
                if not self._is_eligible(file_path):
                    continue
                try:
                    size = file_path.stat().st_size
                except OSError:
                    continue
                files.append((file_path, domain, size))

        # Smallest files first for maximum breadth.
        files.sort(key=lambda x: x[2])
        logger.info("Discovered %d eligible files to ingest", len(files))

        for file_path, domain, _ in files:
            self._stats["files_seen"] += 1
            try:
                self._check_free_space()
                self._ingest_file(file_path, domain)
            except OutOfSpace as exc:
                logger.warning("%s", exc)
                break
            except Exception as exc:
                logger.warning("Failed to ingest %s: %s", file_path, exc)

        self.backend.flush()
        self._log_summary()

    @staticmethod
    def _is_eligible(file_path: Path) -> bool:
        """Return True if the file extension is ingestible."""
        name = file_path.name.lower()
        if name.endswith(".gz"):
            inner = name[:-3]
            return inner.endswith((".txt", ".json", ".jsonl", ".csv", ".md"))
        return name.endswith((".txt", ".json", ".jsonl", ".csv", ".md", ".zip"))

    def _check_free_space(self) -> None:
        """Raise OutOfSpace if the DB drive has too little free space."""
        free = shutil.disk_usage(self._db_drive_root).free
        if free < self.min_free_bytes:
            free_gb = free / (1024 ** 3)
            raise OutOfSpace(
                f"Stopping: only {free_gb:.2f} GB free on {self._db_drive_root}"
            )

    def _ingest_file(self, file_path: Path, domain: str) -> None:
        """Ingest a single file and optionally delete the source."""
        chunks = self._open_and_stream(file_path)
        if chunks is None:
            return

        file_ingested = 0
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size:
                self._stats["chunks_rejected"] += 1
                continue
            item = self.backend.ingest(
                content=chunk[: self.chunk_size],
                domain=domain,
                confidence=0.8,
                metadata={"source": file_path.name},
            )
            if not item.metadata.get("__rejected"):
                file_ingested += 1
                self._stats["chunks_ingested"] += 1
                self._stats["bytes_seen"] += len(chunk)

        if file_ingested > 0:
            self._stats["files_ingested"] += 1
            self._stats["domains"].add(domain)
            self.backend.flush()
            free_gb = shutil.disk_usage(self._db_drive_root).free / (1024 ** 3)
            logger.info(
                "Indexed %s (domain=%s): %d chunks; DB count=%d; free=%.2f GB",
                file_path.name,
                domain,
                file_ingested,
                self.backend.count(),
                free_gb,
            )
            self._maybe_log()

            if self.delete_source:
                try:
                    file_path.unlink()
                    self._stats["sources_deleted"] += 1
                    logger.info("Deleted source %s", file_path)
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", file_path, exc)

    def _open_and_stream(self, file_path: Path) -> Optional[Iterable[str]]:
        """Open a file (plain, gzip, or zip) and return a chunk generator."""
        name = file_path.name.lower()

        if name.endswith(".zip"):
            return self._stream_zip(file_path)

        if name.endswith(".gz"):
            inner = name[:-3]
            handle = gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
            if inner.endswith((".json", ".jsonl")):
                return self._stream_jsonl_handle(handle)
            if inner.endswith(".csv"):
                return self._stream_csv_handle(handle)
            return self._stream_text_handle(handle)

        handle = open(file_path, "r", encoding="utf-8", errors="replace")
        if name.endswith((".json", ".jsonl")):
            return self._stream_jsonl_handle(handle)
        if name.endswith(".csv"):
            return self._stream_csv_handle(handle)
        return self._stream_text_handle(handle)

    def _stream_text_handle(self, fh) -> Iterable[str]:
        """Yield text chunks by accumulating lines from an open handle."""
        buffer: List[str] = []
        buffer_len = 0
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    if buffer:
                        yield "\n".join(buffer)
                        buffer = []
                        buffer_len = 0
                    continue

                line_len = len(line)
                if buffer_len + line_len > self.chunk_size and buffer:
                    yield "\n".join(buffer)
                    buffer = [line]
                    buffer_len = line_len
                else:
                    buffer.append(line)
                    buffer_len += line_len

            if buffer:
                yield "\n".join(buffer)
        finally:
            fh.close()

    def _stream_jsonl_handle(self, fh) -> Iterable[str]:
        """Yield text chunks from a JSONL handle.

        Tries common text-bearing fields and combines multi-part records
        (e.g., question + answer) when present.
        """
        text_fields = ("text", "content", "question", "instruction", "input", "context")
        aux_fields = ("answer", "output")
        try:
            for line in fh:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                parts: List[str] = []
                for field in text_fields:
                    value = obj.get(field)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                for field in aux_fields:
                    value = obj.get(field)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())

                if not parts:
                    continue
                text = "\n".join(parts)
                yield from self._chunk_text(text)
        finally:
            fh.close()

    def _stream_csv_handle(self, fh) -> Iterable[str]:
        """Yield text chunks from a CSV/TSV handle.

        For ConceptNet-style assertions, joins the relation, start, and end
        columns into a readable statement and skips trailing metadata JSON.
        """
        try:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                # ConceptNet has 5 columns: assertion, relation, start, end, metadata.
                # Use the first four columns (skip metadata JSON if present).
                text = " ".join(row[:4]).strip()
                if len(text) < self.min_chunk_size:
                    continue
                yield from self._chunk_text(text)
        finally:
            fh.close()

    def _stream_zip(self, file_path: Path) -> Optional[Iterable[str]]:
        """Stream the first eligible file inside a .zip archive."""

        def _generator():
            with zipfile.ZipFile(file_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename.lower()
                    if name.endswith((".txt", ".json", ".jsonl", ".csv", ".md")):
                        with zf.open(info) as inner:
                            text = inner.read().decode("utf-8", errors="replace")
                            yield from self._chunk_text(text)
                        break

        return _generator()

    def _chunk_text(self, text: str) -> Iterable[str]:
        """Split a large text string into overlapping chunks."""
        if len(text) <= self.chunk_size:
            yield text
            return

        sentences = re.split(r"(?<=[.!?])\s+", text)
        buffer: List[str] = []
        buffer_len = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_len = len(sentence)
            if buffer_len + sentence_len > self.chunk_size and buffer:
                yield " ".join(buffer)
                buffer = [sentence]
                buffer_len = sentence_len
            else:
                buffer.append(sentence)
                buffer_len += sentence_len

        if buffer:
            yield " ".join(buffer)

    def _maybe_log(self) -> None:
        """Emit progress if the log interval has been reached."""
        ingested = self._stats["chunks_ingested"]
        if ingested % self.log_interval == 0:
            elapsed = time.time() - self._stats["start_time"]
            rate = ingested / elapsed if elapsed > 0 else 0.0
            free_gb = shutil.disk_usage(self._db_drive_root).free / (1024 ** 3)
            logger.info(
                "Ingested %d chunks from %d files in %.1fs (%.1f chunks/s). "
                "Rejected: %d, deleted sources: %d, free space: %.2f GB",
                ingested,
                self._stats["files_ingested"],
                elapsed,
                rate,
                self._stats["chunks_rejected"],
                self._stats["sources_deleted"],
                free_gb,
            )

    def _log_summary(self) -> None:
        """Emit final summary."""
        elapsed = time.time() - self._stats["start_time"]
        free_gb = shutil.disk_usage(self._db_drive_root).free / (1024 ** 3)
        logger.info("=" * 60)
        logger.info("Ingestion complete")
        logger.info("Total files seen: %d", self._stats["files_seen"])
        logger.info("Files with indexed chunks: %d", self._stats["files_ingested"])
        logger.info("Chunks indexed: %d", self._stats["chunks_ingested"])
        logger.info("Chunks rejected: %d", self._stats["chunks_rejected"])
        logger.info("Sources deleted: %d", self._stats["sources_deleted"])
        logger.info("Total bytes seen: %d", self._stats["bytes_seen"])
        logger.info("Domains: %d", len(self._stats["domains"]))
        logger.info("Final DB count: %d", self.backend.count())
        logger.info("DB drive free space: %.2f GB", free_gb)
        logger.info("Elapsed: %.1f seconds", elapsed)
        logger.info("=" * 60)


def main() -> None:
    """CLI entry point for full-scale knowledge ingestion."""
    parser = argparse.ArgumentParser(
        description="Stream all downloaded knowledge into a disk-backed SQLite/FTS5 store."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("CONDROX_KNOWLEDGE_ROOT", r"data/knowledge"),
        help="Knowledge root directory",
    )
    parser.add_argument(
        "--db",
        default="Databases/full_knowledge.db",
        help="Output SQLite database",
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="Target chunk size")
    parser.add_argument("--min-chunk-size", type=int, default=40, help="Min chunk size")
    parser.add_argument("--batch-size", type=int, default=1000, help="SQLite batch size")
    parser.add_argument("--log-interval", type=int, default=10000, help="Progress log interval")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each source file after it is indexed (irreversible)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Stop when DB drive free space drops below this many GB",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing database and resume (skip already-ingested content)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not args.resume and db_path.exists():
        logger.info("Removing existing database %s", db_path)
        db_path.unlink()

    backend = DiskKnowledgeBackend(
        db_path=db_path,
        batch_size=args.batch_size,
        min_quality_score=0.15,
    )

    logger.info(
        "Starting full knowledge ingestion: root=%s, db=%s, chunk_size=%d, "
        "batch_size=%d, delete_source=%s, min_free_gb=%.1f",
        args.root,
        db_path,
        args.chunk_size,
        args.batch_size,
        args.delete_source,
        args.min_free_gb,
    )

    ingester = KnowledgeStreamIngester(
        backend=backend,
        db_path=db_path,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        log_interval=args.log_interval,
        delete_source=args.delete_source,
        min_free_gb=args.min_free_gb,
    )

    try:
        ingester.run(args.root)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
