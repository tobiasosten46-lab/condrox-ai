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
Massive Knowledge Ingestion Engine — Condrox AI

Reads ALL available datasets from E:\\Data\\condrox\\knowledge, categorizes
every chunk into the correct domain, and stores them into a SQLite database
for persistent retrieval by the cognitive pipeline.

Datasets processed:
  1. world_knowledge/   — C4 web text (3.7GB, 4 files)        → world_knowledge
  2. dialog/            — EmpatheticDialogues (7.5MB)          → dialog
  3. ethics/            — ETHICS dataset (14MB)                → ethics
  4. grammar/           — Europarl parallel text (58MB)        → grammar
  5. reasoning/         — GSM8K math problems (3.6MB)          → reasoning
  6. logic/             — ARC science questions (0.25MB)       → logic
  7. downloads/c4/      — Raw C4 gz files (2GB)                → world_knowledge
  8. downloads/conceptnet/ — ConceptNet assertions (68MB)      → common_sense

Empty domains filled from cross-source content:
  - cognitive_models    → extracted from world_knowledge (AI/ML/cognition text)
  - culture             → extracted from world_knowledge (culture/arts/history)
  - language            → extracted from grammar (language/linguistics)
  - morality            → extracted from ethics (moral dilemmas)
  - semantics           → extracted from ConceptNet (word meanings)
  - social_intelligence → extracted from dialog (social interactions)
  - values              → extracted from ethics (human values)

Complexity:
  - File scan:  O(d * f) for d domains, f files
  - Chunking:   O(n) for n lines
  - SQLite insert: O(n) batch insert
  - Memory:     O(batch_size) — streams, never loads full files

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("massive_ingest")

KNOWLEDGE_ROOT = Path(os.environ.get("CONDROX_KNOWLEDGE_ROOT", "E:\\Data\\condrox\\knowledge"))
DB_PATH = Path(os.environ.get("CONDROX_KNOWLEDGE_DB", "E:\\Data\\condrox\\knowledge_db.sqlite"))

# ─── Chunking parameters ──────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 2000       # Max chars per chunk
MIN_CHUNK_CHARS = 50         # Min chars per chunk (skip tiny fragments)
BATCH_SIZE = 500             # SQLite batch insert size
MAX_CHUNKS_PER_FILE = 50000  # Safety limit per file

# ─── Categorization keywords ─────────────────────────────────────────────────
# Used to re-classify world_knowledge chunks into specific domains
_DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "cognitive_models": (
        "cognitive", "cognition", "neural network", "brain function",
        "mental model", "cognitive atlas", "cognitive science",
        "attention mechanism", "memory formation", "perception",
    ),
    "culture": (
        "culture", "cultural", "tradition", "heritage", "festival",
        "custom", "ritual", "folklore", "indigenous", "ethnic",
        "art history", "literature", "poetry", "music history",
    ),
    "language": (
        "linguistic", "linguistics", "phonology", "morphology",
        "syntax", "semantics", "pragmatics", "language acquisition",
        "bilingual", "translation", "etymology", "phonetics",
    ),
    "morality": (
        "moral", "morality", "ethical dilemma", "right and wrong",
        "virtue", "conscience", "moral judgment", "moral reasoning",
    ),
    "semantics": (
        "semantic", "meaning", "definition", "synonym", "antonym",
        "word sense", "semantic relation", "lexical",
    ),
    "social_intelligence": (
        "social interaction", "interpersonal", "communication skill",
        "empathy", "social norm", "social cue", "body language",
        "conflict resolution", "negotiation", "cooperation",
    ),
    "values": (
        "value", "values", "integrity", "honesty", "fairness",
        "justice", "responsibility", "respect", "compassion",
        "human rights", "dignity", "equality",
    ),
}


@dataclass(frozen=True, slots=True)
class IngestionStats:
    """Immutable statistics for the ingestion run.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    total_chunks: int = 0
    total_domains: int = 0
    chunks_per_domain: Dict[str, int] = field(default_factory=dict)
    files_processed: int = 0
    files_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    db_size_mb: float = 0.0


class MassiveIngestionEngine:
    """Massive knowledge ingestion engine with smart categorization.

    Reads all datasets from the knowledge root, categorizes chunks into
    the correct domains, and stores them in SQLite for persistent retrieval.

    Complexity:
        run: T(n) ∈ O(n) for n total chunks across all files.
        Memory: S(n) ∈ O(batch_size) — streams files line-by-line.
    """

    __slots__ = ("_root", "_db_path", "_conn", "_stats", "_seen_hashes")

    def __init__(
        self,
        knowledge_root: Path = KNOWLEDGE_ROOT,
        db_path: Path = DB_PATH,
    ) -> None:
        """Initialize the ingestion engine.

        Args:
            knowledge_root: Root directory with domain subdirectories.
            db_path: Path to the output SQLite database.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._root = knowledge_root
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._stats = IngestionStats()
        self._seen_hashes: set[str] = set()

    def run(self) -> IngestionStats:
        """Run the full massive ingestion pipeline.

        Returns:
            IngestionStats with comprehensive statistics.

        Complexity: T(n) ∈ O(n) for n total chunks.
        """
        t0 = time.time()
        logger.info("=" * 70)
        logger.info("  MASSIVE KNOWLEDGE INGESTION ENGINE — Condrox AI")
        logger.info("  Knowledge root: %s", self._root)
        logger.info("  Database: %s", self._db_path)
        logger.info("=" * 70)

        if not self._root.exists():
            logger.error("Knowledge root does not exist: %s", self._root)
            return IngestionStats(errors=[f"Root not found: {self._root}"])

        # Initialize SQLite database
        self._init_db()

        chunks_per_domain: Dict[str, int] = {}
        files_processed = 0
        files_skipped = 0
        errors: List[str] = []
        total_chunks = 0

        # ─── Phase 1: Process pre-chunked datasets ────────────────────
        logger.info("")
        logger.info("PHASE 1: Pre-chunked datasets (JSONL/TXT)")
        logger.info("-" * 50)

        pre_chunked = [
            ("dialog", "dialog"),
            ("ethics", "ethics"),
            ("grammar", "grammar"),
            ("reasoning", "reasoning"),
            ("logic", "logic"),
        ]

        for domain_name, subdir in pre_chunked:
            domain_dir = self._root / subdir
            if not domain_dir.exists():
                logger.warning("  [SKIP] %s — directory not found", subdir)
                continue

            count = self._process_domain(domain_dir, domain_name)
            chunks_per_domain[domain_name] = chunks_per_domain.get(domain_name, 0) + count
            total_chunks += count
            if count > 0:
                files_processed += 1
                logger.info("  [OK] %-20s → %d chunks", domain_name, count)

        # ─── Phase 2: Process world_knowledge (C4 cleaned) ─────────────
        logger.info("")
        logger.info("PHASE 2: World knowledge (C4 cleaned — 3.7GB)")
        logger.info("-" * 50)

        wk_dir = self._root / "world_knowledge"
        if wk_dir.exists():
            count = self._process_world_knowledge(wk_dir)
            chunks_per_domain["world_knowledge"] = count
            total_chunks += count
            logger.info("  [OK] %-20s → %d chunks", "world_knowledge", count)
        else:
            logger.warning("  [SKIP] world_knowledge — not found")

        # ─── Phase 3: Process raw C4 downloads ─────────────────────────
        logger.info("")
        logger.info("PHASE 3: Raw C4 downloads (gz files — 2GB)")
        logger.info("-" * 50)

        c4_dir = self._root / "downloads" / "c4"
        if c4_dir.exists():
            gz_files = list(c4_dir.glob("*.json.gz"))
            if gz_files:
                count = self._process_c4_gz(gz_files)
                chunks_per_domain["world_knowledge"] = (
                    chunks_per_domain.get("world_knowledge", 0) + count
                )
                total_chunks += count
                logger.info("  [OK] %-20s → %d chunks (from %d gz files)",
                           "world_knowledge (+raw C4)", count, len(gz_files))
            else:
                logger.warning("  [SKIP] No .json.gz files in downloads/c4")
        else:
            logger.warning("  [SKIP] downloads/c4 — not found")

        # ─── Phase 4: Process ConceptNet → common_sense + semantics ────
        logger.info("")
        logger.info("PHASE 4: ConceptNet (68MB → common_sense + semantics)")
        logger.info("-" * 50)

        cn_path = self._root / "downloads" / "conceptnet" / "conceptnet-assertions-5.7.0.csv.gz"
        if cn_path.exists():
            cs_count, sem_count = self._process_conceptnet(cn_path)
            chunks_per_domain["common_sense"] = cs_count
            chunks_per_domain["semantics"] = sem_count
            total_chunks += cs_count + sem_count
            logger.info("  [OK] %-20s → %d chunks", "common_sense", cs_count)
            logger.info("  [OK] %-20s → %d chunks", "semantics", sem_count)
        else:
            logger.warning("  [SKIP] ConceptNet — not found")

        # ─── Phase 5: Cross-source domain filling ──────────────────────
        logger.info("")
        logger.info("PHASE 5: Cross-source domain filling")
        logger.info("-" * 50)

        # Clear dedup cache — same content CAN appear in multiple domains
        # (re-categorization, not duplication)
        self._seen_hashes.clear()

        # Re-classify world_knowledge chunks into specific domains
        fill_count = self._fill_empty_domains()
        for domain, count in fill_count.items():
            chunks_per_domain[domain] = chunks_per_domain.get(domain, 0) + count
            total_chunks += count
            if count > 0:
                logger.info("  [OK] %-20s → %d chunks (extracted)", domain, count)

        # ─── Phase 6: Process self-knowledge + basic knowledge pack ────
        logger.info("")
        logger.info("PHASE 6: Self-knowledge + basic knowledge pack")
        logger.info("-" * 50)

        # Self-knowledge from PreReleaseAgent
        self_knowledge_path = Path(__file__).parent.parent / "agents" / "PreReleaseAgent" / "condrox_self_knowledge.jsonl"
        if self_knowledge_path.exists():
            count = self._process_jsonl_file(self_knowledge_path, "self_knowledge")
            chunks_per_domain["self_knowledge"] = count
            total_chunks += count
            logger.info("  [OK] %-20s → %d chunks", "self_knowledge", count)

        # Basic knowledge pack
        bkp_path = Path(__file__).parent.parent / "agents" / "PreReleaseAgent" / "basic_knowledge_pack.jsonl"
        if bkp_path.exists():
            count = self._process_jsonl_file(bkp_path, "basic_knowledge")
            chunks_per_domain["basic_knowledge"] = count
            total_chunks += count
            logger.info("  [OK] %-20s → %d chunks", "basic_knowledge", count)

        # ─── Finalize ──────────────────────────────────────────────────
        self._conn.commit()
        self._conn.close()

        elapsed = time.time() - t0
        db_size = self._db_path.stat().st_size / (1024 * 1024) if self._db_path.exists() else 0

        stats = IngestionStats(
            total_chunks=total_chunks,
            total_domains=len(chunks_per_domain),
            chunks_per_domain=dict(sorted(chunks_per_domain.items(), key=lambda x: -x[1])),
            files_processed=files_processed,
            files_skipped=files_skipped,
            errors=errors,
            elapsed_seconds=elapsed,
            db_size_mb=round(db_size, 1),
        )

        # Print final report
        self._print_report(stats)

        return stats

    def _init_db(self) -> None:
        """Initialize the SQLite database with schema.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._db_path.exists():
            self._db_path.unlink()
            logger.info("  [DB] Removed existing database: %s", self._db_path)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.isolation_level = None
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON knowledge(domain)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON knowledge(content_hash)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge(source)")
        logger.info("  [DB] Initialized: %s", self._db_path)

    def _make_item_id(self, domain: str, content: str, idx: int) -> str:
        """Generate a unique item ID.

        Complexity: T(n) ∈ O(n) for content hashing, S(n) ∈ O(1).
        """
        content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"{domain}_{content_hash}_{idx:08d}"

    def _content_hash(self, content: str) -> str:
        """Generate a content hash for deduplication.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()

    def _insert_batch(self, batch: List[Tuple[str, str, str, str, str, str, float]]) -> int:
        """Insert a batch of knowledge items into SQLite.

        Args:
            batch: List of (item_id, domain, content, source, content_hash, metadata, created_at)

        Returns:
            Number of items actually inserted (after dedup).

        Complexity: T(n) ∈ O(n) for n items, S(n) ∈ O(batch_size).
        """
        if not batch or self._conn is None:
            return 0

        inserted = 0
        for item in batch:
            item_id, domain, content, source, content_hash, metadata, created_at = item
            if content_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(content_hash)
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO knowledge (item_id, domain, content, source, content_hash, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (item_id, domain, content, source, content_hash, metadata, created_at),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass

        return inserted

    def _process_domain(self, domain_dir: Path, domain_name: str) -> int:
        """Process all data files in a domain directory.

        Complexity: T(n) ∈ O(f * c) for f files, c chunks per file.
        """
        total = 0
        data_files = sorted(
            f for f in domain_dir.iterdir()
            if f.is_file()
            and f.suffix in (".txt", ".json", ".jsonl")
            and not f.name.startswith("README")
            and not f.name.startswith("meta.")
        )

        if not data_files:
            return 0

        for file_path in data_files:
            count = self._process_jsonl_file(file_path, domain_name)
            total += count

        return total

    def _process_jsonl_file(self, file_path: Path, domain_name: str) -> int:
        """Process a single JSONL or text file.

        Streams line-by-line, extracts text, chunks it, and inserts into SQLite.

        Complexity: T(n) ∈ O(n) for n lines, S(n) ∈ O(line) per line.
        """
        if self._conn is None:
            return 0

        loaded = 0
        batch: List[Tuple[str, str, str, str, str, str, float]] = []
        now = time.time()
        is_jsonl = file_path.suffix in (".json", ".jsonl")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f):
                    if loaded >= MAX_CHUNKS_PER_FILE:
                        break

                    line = line.strip()
                    if len(line) < 20:
                        continue

                    # Extract text from JSONL or use raw line
                    text = ""
                    metadata_str = "{}"

                    if is_jsonl and line.startswith("{"):
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                            if len(text) < MIN_CHUNK_CHARS:
                                continue
                            # Preserve metadata
                            meta = {k: v for k, v in obj.items() if k not in ("text", "id", "hash")}
                            metadata_str = json.dumps(meta, ensure_ascii=False)[:500]
                        except (json.JSONDecodeError, KeyError):
                            text = line
                    else:
                        # Plain text — skip chunk markers like "--- xxx ---"
                        if line.startswith("---") and line.endswith("---"):
                            continue
                        text = line

                    # Truncate to max chunk size
                    text = text[:MAX_CHUNK_CHARS]
                    if len(text) < MIN_CHUNK_CHARS:
                        continue

                    # Generate IDs and hash
                    content_hash = self._content_hash(text)
                    item_id = self._make_item_id(domain_name, text, loaded)
                    source = file_path.name

                    batch.append((item_id, domain_name, text, source, content_hash, metadata_str, now))

                    if len(batch) >= BATCH_SIZE:
                        loaded += self._insert_batch(batch)
                        batch.clear()

                        if loaded % 5000 == 0:
                            logger.info("    ... %d chunks ingested from %s", loaded, file_path.name)

        except Exception as exc:
            logger.warning("    [ERROR] Failed to read %s: %s", file_path.name, exc)

        # Flush remaining batch
        if batch:
            loaded += self._insert_batch(batch)

        return loaded

    def _process_world_knowledge(self, wk_dir: Path) -> int:
        """Process world_knowledge C4 cleaned files.

        These are 1GB+ JSONL files. Stream line-by-line with sampling.

        Complexity: T(n) ∈ O(n) for n lines, S(n) ∈ O(line) per line.
        """
        total = 0
        data_files = sorted(
            f for f in wk_dir.iterdir()
            if f.is_file()
            and f.suffix == ".json"
            and "progress" not in f.name
        )

        if not data_files:
            return 0

        # Distribute chunk budget across files
        chunks_per_file = min(MAX_CHUNKS_PER_FILE, 25000)
        logger.info("  Processing %d files, ~%d chunks each", len(data_files), chunks_per_file)

        for file_path in data_files:
            count = self._process_large_jsonl(file_path, "world_knowledge", chunks_per_file)
            total += count
            logger.info("    %s → %d chunks", file_path.name, count)

        return total

    def _process_large_jsonl(
        self, file_path: Path, domain_name: str, max_chunks: int
    ) -> int:
        """Process a large JSONL file with sampling.

        Uses systematic sampling — every Nth line — to get broad coverage
        without loading the entire file.

        Complexity: T(n) ∈ O(n) for n lines, S(n) ∈ O(line) per line.
        """
        if self._conn is None:
            return 0

        loaded = 0
        batch: List[Tuple[str, str, str, str, str, str, float]] = []
        now = time.time()

        # First pass: count lines to determine sampling rate
        # For 1GB files, we can't count all lines. Use estimated line count.
        file_size = file_path.stat().st_size
        # Average C4 line is ~2000 bytes
        est_lines = file_size // 2000
        sample_rate = max(1, est_lines // max_chunks) if est_lines > max_chunks else 1

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    if loaded >= max_chunks:
                        break

                    # Systematic sampling
                    if line_num % sample_rate != 0:
                        continue

                    line = line.strip()
                    if len(line) < 50:
                        continue

                    text = ""
                    if line.startswith("{"):
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                        except (json.JSONDecodeError, KeyError):
                            continue
                    else:
                        continue

                    if len(text) < MIN_CHUNK_CHARS:
                        continue

                    text = text[:MAX_CHUNK_CHARS]
                    content_hash = self._content_hash(text)
                    item_id = self._make_item_id(domain_name, text, loaded)
                    source = file_path.name

                    batch.append((item_id, domain_name, text, source, content_hash, "{}", now))

                    if len(batch) >= BATCH_SIZE:
                        loaded += self._insert_batch(batch)
                        batch.clear()

                        if loaded % 5000 == 0:
                            logger.info("    ... %d chunks from %s (line %d)",
                                       loaded, file_path.name, line_num)

        except Exception as exc:
            logger.warning("    [ERROR] %s: %s", file_path.name, exc)

        if batch:
            loaded += self._insert_batch(batch)

        return loaded

    def _process_c4_gz(self, gz_files: List[Path]) -> int:
        """Process raw C4 gz files.

        Decompresses and streams line-by-line.

        Complexity: T(n) ∈ O(n) for n lines, S(n) ∈ O(line) per line.
        """
        if self._conn is None:
            return 0

        total = 0
        chunks_per_file = min(MAX_CHUNKS_PER_FILE, 10000)
        now = time.time()

        for gz_path in gz_files:
            loaded = 0
            batch: List[Tuple[str, str, str, str, str, str, float]] = []

            try:
                with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f):
                        if loaded >= chunks_per_file:
                            break

                        line = line.strip()
                        if len(line) < 50 or not line.startswith("{"):
                            continue

                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                        except (json.JSONDecodeError, KeyError):
                            continue

                        if len(text) < MIN_CHUNK_CHARS:
                            continue

                        text = text[:MAX_CHUNK_CHARS]
                        content_hash = self._content_hash(text)
                        item_id = self._make_item_id("world_knowledge", text, total + loaded)
                        source = gz_path.name

                        batch.append((item_id, "world_knowledge", text, source, content_hash, "{}", now))

                        if len(batch) >= BATCH_SIZE:
                            loaded += self._insert_batch(batch)
                            batch.clear()

            except Exception as exc:
                logger.warning("    [ERROR] %s: %s", gz_path.name, exc)

            if batch:
                loaded += self._insert_batch(batch)

            total += loaded
            logger.info("    %s → %d chunks", gz_path.name, loaded)

        return total

    def _process_conceptnet(self, cn_path: Path) -> Tuple[int, int]:
        """Process ConceptNet CSV.gz file into common_sense and semantics.

        ConceptNet assertions format:
        /relation/start_node/end_node/weight   metadata

        Relations like IsA, CapableOf, HasProperty → common_sense
        Relations like Synonym, Antonym, RelatedTo → semantics

        Complexity: T(n) ∈ O(n) for n lines, S(n) ∈ O(line) per line.
        """
        if self._conn is None:
            return 0, 0

        common_sense_count = 0
        semantics_count = 0
        now = time.time()

        # ConceptNet relations → domain mapping
        common_sense_relations = {
            "/r/IsA", "/r/CapableOf", "/r/HasProperty", "/r/HasA",
            "/r/PartOf", "/r/UsedFor", "/r/MadeOf", "/r/HasSubevent",
            "/r/Causes", "/r/Desires", "/r/NotDesires",
            "/r/ReceivesAction", "/r/HasPrerequisite",
        }
        semantics_relations = {
            "/r/Synonym", "/r/Antonym", "/r/RelatedTo", "/r/FormOf",
            "/r/EtymologicallyRelatedTo", "/r/DerivedFrom",
            "/r/SimilarTo", "/r/HasContext",
        }

        cs_batch: List[Tuple[str, str, str, str, str, str, float]] = []
        sem_batch: List[Tuple[str, str, str, str, str, str, float]] = []

        try:
            with gzip.open(cn_path, "rt", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f):
                    if line_num >= 2000000:  # Safety limit
                        break

                    parts = line.strip().split("\t")
                    if len(parts) < 4:
                        continue

                    # ConceptNet format: assertion_uri, relation, start_node, end_node, metadata
                    relation = parts[1]
                    start_node = parts[2]
                    end_node = parts[3]

                    # Skip non-English — both nodes must be English
                    if "/c/en/" not in start_node or "/c/en/" not in end_node:
                        continue

                    start = start_node.replace("/c/en/", "").split("/")[0].replace("_", " ")
                    end = end_node.replace("/c/en/", "").split("/")[0].replace("_", " ")

                    # Build a natural language statement
                    rel_name = relation.replace("/r/", "").lower()
                    text = f"{start} {rel_name.replace('relatedto', 'is related to').replace('isa', 'is a').replace('capableof', 'is capable of').replace('hasproperty', 'has the property of').replace('hasa', 'has').replace('partof', 'is part of').replace('usedfor', 'is used for').replace('synonym', 'is synonymous with').replace('antonym', 'is the opposite of').replace('derivedfrom', 'is derived from').replace('similarto', 'is similar to').replace('madeof', 'is made of').replace('hassubevent', 'has subevent').replace('causes', 'causes').replace('desires', 'desires').replace('formof', 'is a form of')} {end}"

                    if len(text) < MIN_CHUNK_CHARS:
                        continue

                    text = text[:MAX_CHUNK_CHARS]
                    content_hash = self._content_hash(text)
                    source = cn_path.name

                    if relation in common_sense_relations:
                        item_id = self._make_item_id("common_sense", text, common_sense_count)
                        cs_batch.append((item_id, "common_sense", text, source, content_hash, "{}", now))
                        if len(cs_batch) >= BATCH_SIZE:
                            common_sense_count += self._insert_batch(cs_batch)
                            cs_batch.clear()
                    elif relation in semantics_relations:
                        item_id = self._make_item_id("semantics", text, semantics_count)
                        sem_batch.append((item_id, "semantics", text, source, content_hash, "{}", now))
                        if len(sem_batch) >= BATCH_SIZE:
                            semantics_count += self._insert_batch(sem_batch)
                            sem_batch.clear()

        except Exception as exc:
            logger.warning("    [ERROR] ConceptNet: %s", exc)

        if cs_batch:
            common_sense_count += self._insert_batch(cs_batch)
        if sem_batch:
            semantics_count += self._insert_batch(sem_batch)

        return common_sense_count, semantics_count

    def _fill_empty_domains(self) -> Dict[str, int]:
        """Fill empty domains by re-classifying world_knowledge chunks.

        Scans world_knowledge chunks in the database and re-categorizes
        them into specific domains based on keyword matching.

        Complexity: T(n) ∈ O(n * d) for n chunks, d domain keyword sets.
        """
        if self._conn is None:
            return {}

        result: Dict[str, int] = {}
        now = time.time()

        # Get all world_knowledge chunks
        cursor = self._conn.execute(
            "SELECT id, content, source FROM knowledge WHERE domain = 'world_knowledge' LIMIT 100000"
        )

        batch_per_domain: Dict[str, List[Tuple[str, str, str, str, str, str, float]]] = {
            domain: [] for domain in _DOMAIN_KEYWORDS
        }

        for row_id, content, source in cursor:
            content_lower = content.lower()

            for domain, keywords in _DOMAIN_KEYWORDS.items():
                # Check if content matches this domain's keywords
                match_count = sum(1 for kw in keywords if kw in content_lower)
                if match_count >= 1:  # At least 1 keyword match
                    content_hash = self._content_hash(content)
                    item_id = self._make_item_id(domain, content, len(batch_per_domain[domain]))
                    batch_per_domain[domain].append(
                        (item_id, domain, content, source, content_hash, "{}", now)
                    )
                    break  # One domain per chunk

        for domain, batch in batch_per_domain.items():
            if batch:
                result[domain] = self._insert_batch(batch)

        return result

    def _print_report(self, stats: IngestionStats) -> None:
        """Print a comprehensive ingestion report.

        Complexity: T(n) ∈ O(d) for d domains, S(n) ∈ O(1).
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("  INGESTION COMPLETE — Final Report")
        logger.info("=" * 70)
        logger.info("")
        logger.info("  Total chunks ingested:  %d", stats.total_chunks)
        logger.info("  Total domains:          %d", stats.total_domains)
        logger.info("  Files processed:        %d", stats.files_processed)
        logger.info("  Elapsed time:           %.1f seconds", stats.elapsed_seconds)
        logger.info("  Database size:          %.1f MB", stats.db_size_mb)
        logger.info("  Chunks/second:          %.0f", stats.total_chunks / max(1, stats.elapsed_seconds))
        logger.info("")
        logger.info("  Domain breakdown:")
        logger.info("  %-25s %10s %8s", "Domain", "Chunks", "Pct")
        logger.info("  " + "-" * 45)

        for domain, count in stats.chunks_per_domain.items():
            pct = count / max(1, stats.total_chunks) * 100
            bar = "#" * int(pct / 2)
            logger.info("  %-25s %10d %7.1f%% %s", domain, count, pct, bar)

        logger.info("  " + "-" * 45)
        logger.info("  %-25s %10d", "TOTAL", stats.total_chunks)
        logger.info("")

        if stats.errors:
            logger.info("  Errors (%d):", len(stats.errors))
            for err in stats.errors:
                logger.info("    - %s", err)
            logger.info("")

        logger.info("  Database: %s (%.1f MB)", self._db_path, stats.db_size_mb)
        logger.info("=" * 70)


def main() -> None:
    """Run the massive ingestion engine."""
    engine = MassiveIngestionEngine()
    stats = engine.run()

    # Save stats to JSON
    stats_path = Path("security/massive_ingestion_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": stats.total_chunks,
            "total_domains": stats.total_domains,
            "chunks_per_domain": stats.chunks_per_domain,
            "files_processed": stats.files_processed,
            "elapsed_seconds": round(stats.elapsed_seconds, 1),
            "db_size_mb": stats.db_size_mb,
            "db_path": str(DB_PATH),
        }, f, indent=2, ensure_ascii=False)
    logger.info("  Stats saved to: %s", stats_path)


if __name__ == "__main__":
    main()
