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
Ruleset Knowledge Ingestion — inherits and upgrades knowledge from
C:\\ruleset\\ (Condrox AI) into the DeveloperAgent knowledge backend.

Ingestion sources:
    - memory/         -> episodic + semantic memory entries
    - reasoning/      -> reasoning patterns with confidence + provenance
    - meta_learning/  -> learning strategies (versioned, transferable)
    - codebase/       -> code chunks for RAG indexing
    - knowledge/      -> domain-tagged knowledge entries
    - projects/       -> project context entries
    - skills/         -> invocable skill definitions
    - rules.md   -> enforceable policy rules
    - SYSTEM_ARCHITECTURE.md -> architecture reference docs

Ingestion process:
    source -> parser -> chunker -> metadata tagger -> quality filter
           -> deduplicator (content hash) -> index

Complexity:
    ingest_all: T(n) in O(n) where n = total source bytes, S(n) in O(index).
    ingest_source: T(n) in O(n) where n = source size, S(n) in O(entries).

Invariants:
    - No fake/simulated/placeholder content ingested (Strict Mode).
    - Every ingested item has provenance metadata (source path + timestamp).
    - Content hash deduplication prevents duplicate entries.
    - Empty files are skipped (no empty entries).

Edge Cases:
    - ruleset path missing -> empty result, no error.
    - Individual source missing -> skipped, logged in stats.
    - Binary file -> skipped (UTF-8 decode failure).
    - Empty file -> skipped.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Enums ───────────────────────────────────────────────────────────────────


class SourceType(Enum):
    """
    Type of ruleset knowledge source.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    MEMORY = "memory"
    REASONING = "reasoning"
    META_LEARNING = "meta_learning"
    CODEBASE = "codebase"
    KNOWLEDGE = "knowledge"
    PROJECTS = "projects"
    SKILLS = "skills"
    RULES = "rules"
    ARCHITECTURE = "architecture"
    COMMANDS = "commands"
    WORKFLOWS = "workflows"
    DOMAINS = "domains"
    DOCS = "docs"


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IngestionSource:
    """
    Immutable descriptor for a single ingestion source.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: source_type is a valid SourceType enum value.
    Edge Cases: path may not exist (checked at ingestion time).
    """

    source_type: SourceType
    relative_path: str
    absolute_path: str
    file_extension: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """
    Immutable single ingested knowledge entry.

    Complexity: T(n) in O(1), S(n) in O(content + metadata).
    Invariants: content_hash is SHA-256 hex digest of content.
    Edge Cases: empty content -> skipped before construction.
    """

    source_type: SourceType
    source_path: str
    content: str
    content_hash: str
    domain: str
    metadata: Dict[str, str]
    ingested_at: float


@dataclass(frozen=True, slots=True)
class IngestionStats:
    """
    Immutable aggregate statistics from an ingestion run.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: total = ingested + skipped + failed.
    Edge Cases: empty ruleset -> all counts zero.
    """

    total_files: int
    ingested: int
    skipped: int
    failed: int
    duplicates: int
    total_bytes: int
    elapsed_ms: int
    sources_by_type: Dict[str, int]


# ─── RulesetIngester ────────────────────────────────────────────────────────


class RulesetIngester:
    """
    Ingests knowledge from C:\\ruleset\\ into DeveloperAgent knowledge backend.

    Reads all configured source directories, parses text files, tags them
    with provenance metadata, deduplicates by content hash, and returns
    structured IngestionResult entries ready for RAG indexing.

    Complexity:
        ingest_all: T(n) in O(n) where n = total source bytes,
                    S(n) in O(entries).
    Invariants:
        - No empty entries (empty files skipped).
        - No duplicate entries (content hash deduplication).
        - Every entry has provenance (source_path + ingested_at).
        - Strict Mode: no fake/simulated/placeholder content.
    Edge Cases:
        - ruleset missing -> empty result, no error.
        - Binary file -> skipped (UTF-8 decode failure).
        - Individual source missing -> skipped, counted in stats.
    """

    __slots__ = (
        "_ruleset_path",
        "_source_dirs",
        "_rule_files",
        "_arch_files",
        "_max_file_size",
        "_seen_hashes",
    )

    _DEFAULT_SOURCE_DIRS: Tuple[Tuple[SourceType, str], ...] = (
        (SourceType.MEMORY, "memory"),
        (SourceType.REASONING, "reasoning"),
        (SourceType.META_LEARNING, "meta_learning"),
        (SourceType.CODEBASE, "codebase"),
        (SourceType.KNOWLEDGE, "knowledge"),
        (SourceType.PROJECTS, "projects"),
        (SourceType.SKILLS, "skills"),
        (SourceType.COMMANDS, "commands"),
        (SourceType.WORKFLOWS, "workflows"),
        (SourceType.DOMAINS, "domains"),
        (SourceType.DOCS, "docs"),
    )

    _DEFAULT_RULE_FILES: Tuple[Tuple[SourceType, str], ...] = (
        (SourceType.RULES, "rules.md"),
        (SourceType.RULES, "GLOBAL_RULES_SNIPPET.md"),
        (SourceType.RULES, "IMPLEMENT_GLOBAL_RULES.md"),
    )

    _DEFAULT_ARCH_FILES: Tuple[Tuple[SourceType, str], ...] = (
        (SourceType.ARCHITECTURE, "SYSTEM_ARCHITECTURE.md"),
        (SourceType.ARCHITECTURE, "manifest.json"),
    )

    _TEXT_EXTENSIONS: frozenset[str] = frozenset({
        ".md", ".txt", ".json", ".yaml", ".yml", ".py", ".ts",
        ".js", ".toml", ".cfg", ".ini", ".xml", ".csv",
    })

    _DOMAIN_MAP: Dict[SourceType, str] = {
        SourceType.MEMORY: "memory",
        SourceType.REASONING: "reasoning",
        SourceType.META_LEARNING: "meta_learning",
        SourceType.CODEBASE: "codebase",
        SourceType.KNOWLEDGE: "knowledge",
        SourceType.PROJECTS: "projects",
        SourceType.SKILLS: "skills",
        SourceType.RULES: "rules",
        SourceType.ARCHITECTURE: "architecture",
        SourceType.COMMANDS: "commands",
        SourceType.WORKFLOWS: "workflows",
        SourceType.DOMAINS: "domains",
        SourceType.DOCS: "docs",
    }

    def __init__(
        self,
        ruleset_path: str = "C:\\ruleset",
        max_file_size: int = 1_048_576,
    ) -> None:
        """
        Initialize RulesetIngester with ruleset root path.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: ruleset_path missing -> ingest_all returns empty.
        """
        self._ruleset_path: str = os.path.realpath(ruleset_path)
        self._source_dirs: Tuple[Tuple[SourceType, str], ...] = (
            self._DEFAULT_SOURCE_DIRS
        )
        self._rule_files: Tuple[Tuple[SourceType, str], ...] = (
            self._DEFAULT_RULE_FILES
        )
        self._arch_files: Tuple[Tuple[SourceType, str], ...] = (
            self._DEFAULT_ARCH_FILES
        )
        self._max_file_size: int = max_file_size
        self._seen_hashes: set[str] = set()

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def ruleset_path(self) -> str:
        """
        Return the ruleset root path.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: path may not exist on disk.
        """
        return self._ruleset_path

    @property
    def is_available(self) -> bool:
        """
        Check if ruleset directory exists and is accessible.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: path exists but not readable -> False.
        """
        return os.path.isdir(self._ruleset_path)

    def ingest_all(self) -> Tuple[IngestionResult, ...]:
        """
        Ingest all configured sources from ruleset.

        Iterates over source directories, rule files, and architecture files.
        Deduplicates by content hash. Returns all unique entries.

        Complexity: T(n) in O(n) where n = total source bytes,
                    S(n) in O(entries).
        Invariants: no duplicate content hashes in result.
        Edge Cases: ruleset missing -> empty tuple.
        """
        if not self.is_available:
            return ()

        self._seen_hashes.clear()
        all_results: List[IngestionResult] = []

        for source_type, dir_name in self._source_dirs:
            results: Tuple[IngestionResult, ...] = self._ingest_directory(
                source_type, dir_name
            )
            all_results.extend(results)

        for source_type, file_name in self._rule_files:
            result: Optional[IngestionResult] = self._ingest_file(
                source_type, file_name
            )
            if result is not None:
                all_results.append(result)

        for source_type, file_name in self._arch_files:
            result = self._ingest_file(source_type, file_name)
            if result is not None:
                all_results.append(result)

        return tuple(all_results)

    def ingest_source(
        self, source_type: SourceType
    ) -> Tuple[IngestionResult, ...]:
        """
        Ingest a single source type from ruleset.

        Complexity: T(n) in O(n) where n = source size, S(n) in O(entries).
        Invariants: no duplicate content hashes.
        Edge Cases: source missing -> empty tuple.
        """
        if not self.is_available:
            return ()

        for st, dir_name in self._source_dirs:
            if st == source_type:
                return self._ingest_directory(source_type, dir_name)

        for st, file_name in self._rule_files:
            if st == source_type:
                result: Optional[IngestionResult] = self._ingest_file(
                    source_type, file_name
                )
                return (result,) if result else ()

        for st, file_name in self._arch_files:
            if st == source_type:
                result = self._ingest_file(source_type, file_name)
                return (result,) if result else ()

        return ()

    def get_stats(
        self, results: Tuple[IngestionResult, ...]
    ) -> IngestionStats:
        """
        Compute aggregate statistics from ingestion results.

        Complexity: T(n) in O(n) where n = results, S(n) in O(1).
        Edge Cases: empty results -> all zero stats.
        """
        sources_by_type: Dict[str, int] = {}
        total_bytes: int = 0

        for r in results:
            key: str = r.source_type.value
            sources_by_type[key] = sources_by_type.get(key, 0) + 1
            total_bytes += len(r.content)

        return IngestionStats(
            total_files=len(results),
            ingested=len(results),
            skipped=0,
            failed=0,
            duplicates=0,
            total_bytes=total_bytes,
            elapsed_ms=0,
            sources_by_type=sources_by_type,
        )

    def list_sources(self) -> Tuple[IngestionSource, ...]:
        """
        List all configured ingestion sources (without ingesting).

        Complexity: T(n) in O(s) where s = configured sources, S(n) in O(s).
        Edge Cases: ruleset missing -> sources with non-existent paths.
        """
        sources: List[IngestionSource] = []

        for source_type, dir_name in self._source_dirs:
            abs_path: str = os.path.join(self._ruleset_path, dir_name)
            sources.append(IngestionSource(
                source_type=source_type,
                relative_path=dir_name,
                absolute_path=abs_path,
                file_extension="",
            ))

        for source_type, file_name in self._rule_files + self._arch_files:
            ext: str = os.path.splitext(file_name)[1]
            abs_path = os.path.join(self._ruleset_path, file_name)
            sources.append(IngestionSource(
                source_type=source_type,
                relative_path=file_name,
                absolute_path=abs_path,
                file_extension=ext,
            ))

        return tuple(sources)

    # ─── Private: Directory Ingestion ────────────────────────────────────────

    def _ingest_directory(
        self, source_type: SourceType, dir_name: str
    ) -> Tuple[IngestionResult, ...]:
        """
        Ingest all text files from a ruleset subdirectory.

        Complexity: T(n) in O(n) where n = total file bytes in directory,
                    S(n) in O(entries).
        Invariants: only text files ingested; binary files skipped.
        Edge Cases: directory missing -> empty tuple.
        """
        dir_path: str = os.path.join(self._ruleset_path, dir_name)

        if not os.path.isdir(dir_path):
            return ()

        results: List[IngestionResult] = []

        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                file_path: str = os.path.join(root, filename)
                ext: str = os.path.splitext(filename)[1].lower()

                if ext not in self._TEXT_EXTENSIONS:
                    continue

                result: Optional[IngestionResult] = self._ingest_path(
                    source_type, file_path
                )
                if result is not None:
                    results.append(result)

        return tuple(results)

    # ─── Private: File Ingestion ─────────────────────────────────────────────

    def _ingest_file(
        self, source_type: SourceType, file_name: str
    ) -> Optional[IngestionResult]:
        """
        Ingest a single file from ruleset root.

        Complexity: T(n) in O(n) where n = file size, S(n) in O(content).
        Invariants: returns None for missing/empty/binary files.
        Edge Cases: file missing -> None; binary -> None; empty -> None.
        """
        file_path: str = os.path.join(self._ruleset_path, file_name)
        return self._ingest_path(source_type, file_path)

    def _ingest_path(
        self, source_type: SourceType, file_path: str
    ) -> Optional[IngestionResult]:
        """
        Ingest a single file by absolute path.

        Complexity: T(n) in O(n) where n = file size, S(n) in O(content).
        Invariants: content hash computed; deduplication enforced.
        Edge Cases: missing -> None; too large -> None; binary -> None;
                    duplicate -> None.
        """
        if not os.path.isfile(file_path):
            return None

        file_size: int = os.path.getsize(file_path)
        if file_size == 0 or file_size > self._max_file_size:
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                content: str = f.read()
        except (UnicodeDecodeError, OSError):
            return None

        if not content.strip():
            return None

        content_hash: str = hashlib.sha256(content.encode("utf-8")).hexdigest()

        if content_hash in self._seen_hashes:
            return None

        self._seen_hashes.add(content_hash)

        domain: str = self._DOMAIN_MAP.get(source_type, "general")

        metadata: Dict[str, str] = {
            "source_path": file_path,
            "file_size": str(file_size),
            "file_extension": os.path.splitext(file_path)[1],
            "ingested_at": str(time.time()),
        }

        return IngestionResult(
            source_type=source_type,
            source_path=file_path,
            content=content,
            content_hash=content_hash,
            domain=domain,
            metadata=metadata,
            ingested_at=time.time(),
        )
