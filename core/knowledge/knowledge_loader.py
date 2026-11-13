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
Knowledge loader — reads chunked datasets from disk into KnowledgeBackend.

Scans the knowledge root directory, reads text chunks from each domain,
and ingests them into the in-memory KnowledgeBackend for pipeline retrieval.

Mathematical Foundation:
- File scan: O(d * f) for d domains, f files per domain.
- Ingest: O(n * c) for n chunks, c avg chunk length.
- Memory: O(n * e) for n items, e embedding dimension.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_router import KnowledgeRouter

logger = logging.getLogger(__name__)


class KnowledgeLoader:
    """
    Loads chunked knowledge files from disk into the KnowledgeBackend.

    Attributes:
        _knowledge_root: Root directory containing domain subdirectories.
        _max_chunks_per_domain: Maximum chunks to load per domain (memory safety).
        _max_chunk_chars: Maximum characters per chunk to ingest.

    Complexity:
        load_all: T(n) ∈ O(d * f * c) for d domains, f files, c avg chunk size.
        Memory: S(n) ∈ O(n * e) for n items, e embedding dimension.
    """

    __slots__ = ("_knowledge_root", "_max_chunks_per_domain", "_max_chunk_chars", "_max_files_per_domain")

    def __init__(
        self,
        knowledge_root: str | Path = "E:\\Data\\condrox\\knowledge",
        max_chunks_per_domain: Optional[int] = None,
        max_chunk_chars: Optional[int] = None,
        max_files_per_domain: Optional[int] = None,
    ) -> None:
        """Initialize the knowledge loader.

        Args:
            knowledge_root: Root directory with domain subdirectories.
            max_chunks_per_domain: Max chunks to load per domain.
            max_chunk_chars: Max characters per chunk (truncated if longer).
            max_files_per_domain: Max files to sample per domain (for large dirs).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        import os

        self._knowledge_root: Path = Path(knowledge_root)
        self._max_chunks_per_domain: int = max_chunks_per_domain or int(
            os.environ.get("CONDROX_KL_MAX_CHUNKS", "10000")
        )
        self._max_chunk_chars: int = max_chunk_chars or int(
            os.environ.get("CONDROX_KL_MAX_CHUNK_CHARS", "2000")
        )
        self._max_files_per_domain: int = max_files_per_domain or int(
            os.environ.get("CONDROX_KL_MAX_FILES", "20")
        )

    def load_into(self, backend: KnowledgeBackend) -> int:
        """Load knowledge from disk into the backend.

        Args:
            backend: KnowledgeBackend to ingest into.

        Returns:
            Total number of knowledge items loaded.

        Complexity: T(n) ∈ O(d * f * c) for d domains, f files, c avg chunk size.
        """
        if not self._knowledge_root.exists():
            logger.warning(
                "Knowledge root does not exist: %s", self._knowledge_root
            )
            return 0

        total_loaded: int = 0

        for domain_dir in sorted(self._knowledge_root.iterdir()):
            if not domain_dir.is_dir():
                continue

            domain_name = domain_dir.name
            if domain_name in ("downloads", "meta"):
                continue

            loaded = self._load_domain(backend, domain_dir, domain_name)
            if loaded > 0:
                logger.info(
                    "Loaded %d knowledge items from domain '%s'", loaded, domain_name
                )
                total_loaded += loaded

        logger.info("Total knowledge items loaded: %d", total_loaded)
        return total_loaded

    def _load_domain(
        self,
        backend: KnowledgeBackend,
        domain_dir: Path,
        domain_name: str,
    ) -> int:
        """Load knowledge chunks from a single domain directory.

        Streams JSONL files line-by-line to handle 1GB+ files without
        loading them into memory. Samples across multiple files for
        broad coverage instead of only reading the first file.

        Args:
            backend: KnowledgeBackend to ingest into.
            domain_dir: Directory containing chunk files.
            domain_name: Domain label for ingested items.

        Returns:
            Number of items loaded from this domain.

        Complexity: T(n) ∈ O(f * c) for f files, c avg chunk size.
        """
        loaded: int = 0

        # Collect valid data files (skip metadata/README)
        data_files = sorted(
            f for f in domain_dir.iterdir()
            if f.is_file()
            and f.suffix in (".txt", ".json", ".jsonl")
            and not f.name.startswith("README")
            and not f.name.startswith("meta.")
        )

        if not data_files:
            return 0

        # Sample across files: distribute chunk budget across max_files files
        files_to_read = data_files[: self._max_files_per_domain]
        chunks_per_file = max(
            1,
            self._max_chunks_per_domain // len(files_to_read),
        )

        for file_path in files_to_read:
            if loaded >= self._max_chunks_per_domain:
                break

            try:
                file_loaded = self._stream_file(
                    backend, file_path, domain_name,
                    max_chunks=min(chunks_per_file, self._max_chunks_per_domain - loaded),
                )
                loaded += file_loaded
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)

        return loaded

    def _stream_file(
        self,
        backend: KnowledgeBackend,
        file_path: Path,
        domain_name: str,
        max_chunks: int,
    ) -> int:
        """Stream a JSONL or text file line-by-line, ingesting chunks.

        Handles JSONL files with {"id", "text", ...} format by streaming
        line-by-line — never loads the full file into memory.

        Args:
            backend: KnowledgeBackend to ingest into.
            file_path: Path to the data file.
            domain_name: Domain label.
            max_chunks: Maximum chunks to load from this file.

        Returns:
            Number of chunks ingested.

        Complexity: T(n) ∈ O(n) for file size n, S(n) ∈ O(line) per line.
        """
        import json

        loaded = 0
        is_jsonl = file_path.suffix in (".json", ".jsonl")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if loaded >= max_chunks:
                    break

                line = line.strip()
                if len(line) < 20:
                    continue

                # Try JSONL first — extract "text" field
                if is_jsonl and line.startswith("{"):
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                        if len(text) < 20:
                            continue
                        text = text[: self._max_chunk_chars]
                        item = backend.ingest(
                            content=text,
                            domain=domain_name,
                            confidence=0.8,
                            metadata={
                                "source": file_path.name,
                                "id": obj.get("id", ""),
                            },
                        )
                        checksum = hashlib.sha256(line.encode()).hexdigest()
                        backend.provenance.register(item.item_id, file_path, checksum)
                        loaded += 1
                        continue
                    except (json.JSONDecodeError, KeyError):
                        pass  # Fall through to text handling

                # Plain text: accumulate lines into chunks
                text = line[: self._max_chunk_chars]
                if len(text) >= 20:
                    item = backend.ingest(
                        content=text,
                        domain=domain_name,
                        confidence=0.8,
                        metadata={"source": file_path.name},
                    )
                    checksum = hashlib.sha256(line.encode()).hexdigest()
                    backend.provenance.register(item.item_id, file_path, checksum)
                    loaded += 1

        return loaded

    def _read_chunks(self, file_path: Path) -> List[str]:
        """Read text chunks from a small file (legacy support).

        For large JSONL files, use _stream_file instead.

        Args:
            file_path: Path to the chunk file.

        Returns:
            List of text chunks.

        Complexity: T(n) ∈ O(n) for file size n, S(n) ∈ O(n).
        """
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return []

        if not text.strip():
            return []

        if "\n\n" in text:
            chunks = text.split("\n\n")
        else:
            lines = text.split("\n")
            chunks = []
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
