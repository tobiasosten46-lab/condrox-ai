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
Disk-backed knowledge backend for Condrox-scale datasets.

Uses SQLite + FTS5 for full-text search and BM25 ranking, stores documents
in a normal table, and batches writes for high-throughput ingestion of
hundreds of gigabytes of chunked text.

This backend is API-compatible with KnowledgeBackend for use by
KnowledgeRouter and KnowledgeQualityRanker. It is designed to ingest far
more data than fits in RAM by persisting every chunk and indexing it on disk.

Schema:
  documents(rowid, item_id, domain, content, embedding_json, confidence,
            metadata_json, token_count, char_count, quality_score, created_at)
  fts_knowledge(content) -- FTS5 external content table over documents

Complexity:
  ingest: O(t) tokenization per chunk, batched inserts O(b) per batch.
  search: O(q · df · log k) via FTS5 and BM25.
  candidate_ids: O(q · df) using the FTS5 inverted index.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_models import KnowledgeItem


class DiskKnowledgeBackend(KnowledgeBackend):
    """
    SQLite/FTS5-backed knowledge backend.

    Inherits from KnowledgeBackend for API/typing compatibility, but overrides
    all storage and retrieval methods to use a persistent SQLite database with
    an FTS5 full-text index.

    Attributes:
        _db_path: Path to the SQLite database file.
        _conn: Shared SQLite connection (not thread-safe across threads).
        _buffer: List of KnowledgeItem and values pending insertion.
        _batch_size: Number of items to accumulate before flushing to disk.
        _embedding_dim: Dimensionality of the compatibility embedding vector.
        _min_quality_score: Minimum quality score for an item to be accepted.

    Complexity:
        ingest: T(n) ∈ O(t) per chunk for tokenization.
        search: T(n) ∈ O(q · df · log k).
    """

    __slots__ = (
        "_db_path",
        "_conn",
        "_buffer",
        "_batch_size",
        "_min_quality_score",
        "_closed",
    )

    def __init__(
        self,
        db_path: str | Path = "condrox_knowledge.db",
        embedding_dim: int = 128,
        batch_size: int = 1000,
        min_quality_score: float = 0.0,
    ) -> None:
        """Initialize and create the SQLite/FTS5 schema.

        Args:
            db_path: SQLite database path. Parent directories are created.
            embedding_dim: Embedding vector dimension.
            batch_size: Number of items to batch before writing.
            min_quality_score: Items with a lower score are rejected.
        """
        # Satisfy the parent constructor; in-memory indexes remain unused.
        super().__init__(embedding_dim=embedding_dim)

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_size = max(1, batch_size)
        self._min_quality_score = max(0.0, min_quality_score)
        self._buffer: List[Dict[str, Any]] = []
        self._closed = False

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        # Tuned for bulk ingest and large databases.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-524288")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Reuse the in-memory backend tokenization for consistency."""
        return KnowledgeBackend.tokenize(text)

    @classmethod
    def stop_words(cls) -> frozenset[str]:
        """Reuse the in-memory backend stop-word list for consistency."""
        return KnowledgeBackend.stop_words()

    def _ensure_schema(self) -> None:
        """Create tables and FTS5 index if they do not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT UNIQUE NOT NULL,
                domain TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                metadata_json TEXT,
                token_count INTEGER NOT NULL DEFAULT 0,
                char_count INTEGER NOT NULL DEFAULT 0,
                quality_score REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_item_id ON documents(item_id)"
        )
        # External content FTS5 table; content lives in documents.content.
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_knowledge USING fts5(
                content,
                content='documents',
                content_rowid='rowid'
            )
            """
        )
        self._conn.commit()

    def _compute_quality_score(self, content: str) -> float:
        """Compute a simple quality score for a chunk.

        Rewards chunks that are substantive but not pathologically long.
        """
        length = len(content)
        if length < 20:
            return 0.0
        # Reward 100-2000 char chunks; penalize very short or very long.
        score = min(1.0, length / 800.0)
        if length > self._max_chunk_chars():
            score *= 0.8
        return score

    @staticmethod
    def _max_chunk_chars() -> int:
        """Default maximum characters per chunk."""
        return 2000

    def _embed(self, text: str) -> np.ndarray:
        """Create a simple normalized embedding vector for the chunk.

        Unlike the in-memory backend, this does not use live corpus statistics
        (which would require a full scan of the disk index). It hashes tokens
        into buckets and normalizes the vector.
        """
        tokens = self.tokenize(text)
        vec = np.zeros(self._embedding_dim, dtype=np.float64)
        for token in tokens:
            idx = zlib.crc32(token.encode("utf-8")) % self._embedding_dim
            # Simple IDF proxy: rarer tokens (fewer bucket collisions) are
            # weighted more by using the raw token hash for variance.
            weight = 1.0 + math.log(max(1, len(token)))
            vec[idx] += weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def ingest(
        self,
        content: str,
        domain: str = "general",
        item_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeItem:
        """Stage a knowledge item for batched insertion.

        Items are buffered until _batch_size is reached or flush() is called.

        Complexity: T(n) ∈ O(t) for t tokens.
        """
        if not content:
            raise ValueError("content must be non-empty")
        if self._closed:
            raise RuntimeError("backend is closed")

        score = self._compute_quality_score(content)
        if score < self._min_quality_score:
            # Still create a placeholder item so callers do not crash, but
            # do not index it.
            final_id = item_id or f"kno_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
            rejected_metadata = dict(metadata or {})
            rejected_metadata["__rejected"] = True
            return KnowledgeItem(
                item_id=final_id,
                domain=domain,
                content=content,
                embedding=None,
                confidence=confidence,
                metadata=rejected_metadata,
            )

        final_id = item_id or f"kno_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        tokens = self.tokenize(content)
        embedding = self._embed(content)
        item = KnowledgeItem(
            item_id=final_id,
            domain=domain,
            content=content,
            embedding=embedding,
            confidence=confidence,
            metadata=dict(metadata or {}),
        )

        self._buffer.append({
            "item": item,
            "score": score,
            "token_count": len(tokens),
            "char_count": len(content),
            "created_at": time.time(),
        })

        if len(self._buffer) >= self._batch_size:
            self.flush()

        return item

    def flush(self) -> int:
        """Flush the pending buffer to SQLite/FTS5.

        Returns:
            Number of items written.

        Complexity: O(b) for b buffered items.
        """
        if not self._buffer or self._closed:
            return 0

        rows: List[tuple[Any, ...]] = []
        for entry in self._buffer:
            item = entry["item"]
            rows.append((
                item.item_id,
                item.domain,
                item.content,
                json.dumps(item.embedding.tolist()) if item.embedding is not None else None,
                item.confidence,
                json.dumps(item.metadata) if item.metadata else None,
                entry["token_count"],
                entry["char_count"],
                entry["score"],
                entry["created_at"],
            ))

        with self._conn:
            cur = self._conn.executemany(
                """
                INSERT OR IGNORE INTO documents
                (item_id, domain, content, embedding_json, confidence,
                 metadata_json, token_count, char_count, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            # Collect rowids for the FTS5 external content table.
            item_ids = [entry["item"].item_id for entry in self._buffer]
            cur2 = self._conn.execute(
                f"SELECT rowid, item_id, content FROM documents WHERE item_id IN ({','.join('?'*len(item_ids))})",
                item_ids,
            )
            row_map = {row["item_id"]: (row["rowid"], row["content"]) for row in cur2.fetchall()}
            fts_rows = [
                (row_map[entry["item"].item_id][0], row_map[entry["item"].item_id][1])
                for entry in self._buffer
                if entry["item"].item_id in row_map
            ]
            if fts_rows:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO fts_knowledge (rowid, content) VALUES (?, ?)",
                    fts_rows,
                )

        written = len(self._buffer)
        self._buffer.clear()
        return written

    def get(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve a knowledge item by id from disk."""
        self.flush()
        row = self._conn.execute(
            "SELECT * FROM documents WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def _row_to_item(self, row: sqlite3.Row) -> KnowledgeItem:
        """Convert a documents row into a KnowledgeItem."""
        embedding: Optional[np.ndarray] = None
        if row["embedding_json"]:
            try:
                embedding = np.array(json.loads(row["embedding_json"]), dtype=np.float64)
            except (json.JSONDecodeError, ValueError):
                pass
        return KnowledgeItem(
            item_id=row["item_id"],
            domain=row["domain"],
            content=row["content"],
            embedding=embedding,
            confidence=row["confidence"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        )

    def items(self) -> Iterable[KnowledgeItem]:
        """Iterate over all stored items (not recommended for huge stores)."""
        self.flush()
        cur = self._conn.execute("SELECT * FROM documents")
        for row in cur:
            yield self._row_to_item(row)

    def count(self) -> int:
        """Return total number of indexed documents."""
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def _build_match_query(self, query_tokens: List[str]) -> str:
        """Build an FTS5 MATCH query from query tokens.

        Uses OR semantics so any matching token yields a candidate.
        Tokens are escaped to avoid FTS5 syntax injection.
        """
        escaped = [self._escape_fts_token(t) for t in query_tokens if t]
        return " OR ".join(escaped)

    @staticmethod
    def _escape_fts_token(token: str) -> str:
        """Escape a token for FTS5 MATCH queries.

        Wraps the token in double quotes and escapes embedded quotes.
        """
        safe = token.replace('"', '""')
        return f'"{safe}"'

    def candidate_ids(
        self,
        query_tokens: Iterable[str],
        domain_filter: Optional[str] = None,
    ) -> Set[str]:
        """Return item ids that contain any of the query tokens."""
        tokens = [t for t in query_tokens if t not in self.stop_words()]
        if not tokens:
            return set()

        match_query = self._build_match_query(tokens)
        if domain_filter:
            sql = """
                SELECT d.item_id
                FROM documents d
                WHERE d.rowid IN (
                    SELECT rowid FROM fts_knowledge WHERE content MATCH ?
                )
                AND d.domain = ?
            """
            params = (match_query, domain_filter)
        else:
            sql = """
                SELECT d.item_id
                FROM documents d
                JOIN fts_knowledge f ON d.rowid = f.rowid
                WHERE f.content MATCH ?
            """
            params = (match_query,)

        cur = self._conn.execute(sql, params)
        return {row[0] for row in cur.fetchall()}

    def item_token_set(self, item_id: str) -> Set[str]:
        """Return the set of tokens for a stored item."""
        row = self._conn.execute(
            "SELECT content FROM documents WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None or not row[0]:
            return set()
        return set(self.tokenize(row[0]))

    def bm25_score(
        self,
        query_tokens: List[str],
        item_id: str,
        original_query: str = "",
    ) -> float:
        """Compute an FTS5 BM25 score for a document against query tokens.

        The FTS5 bm25() function returns values where lower (more negative)
        indicates a better match, so we negate it to align with the ranker's
        higher-is-better convention.
        """
        tokens = [t for t in query_tokens if t not in self.stop_words()]
        if not tokens:
            return 0.0

        match_query = self._build_match_query(tokens)
        # Get the rowid and then the per-row bm25 score.
        row = self._conn.execute(
            """
            SELECT d.rowid, bm25(fts_knowledge)
            FROM documents d
            JOIN fts_knowledge f ON d.rowid = f.rowid
            WHERE d.item_id = ? AND f.content MATCH ?
            """,
            (item_id, match_query),
        ).fetchone()

        if row is None or row[1] is None:
            return 0.0

        # Apply an exact-match boost similar to the in-memory backend.
        score = -float(row[1])
        if original_query and score > 0:
            content_lower = self._content_lower(item_id)
            for word in re.findall(r"[a-z0-9]+", original_query.lower()):
                if len(word) > 3 and word in content_lower:
                    score *= 3.0
                    break
        return score

    def _content_lower(self, item_id: str) -> str:
        """Fetch lowercased content for an item, cached per call."""
        row = self._conn.execute(
            "SELECT content FROM documents WHERE item_id = ?", (item_id,)
        ).fetchone()
        return (row[0] or "").lower() if row else ""

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[KnowledgeItem]:
        """Search using FTS5 BM25 and return top-k items."""
        query_tokens = self.tokenize(query)
        tokens = [t for t in query_tokens if t not in self.stop_words()]
        if not tokens:
            return []

        match_query = self._build_match_query(tokens)
        if domain_filter:
            sql = """
                SELECT d.item_id
                FROM documents d
                JOIN fts_knowledge f ON d.rowid = f.rowid
                WHERE f.content MATCH ? AND d.domain = ?
                ORDER BY bm25(fts_knowledge)
                LIMIT ?
            """
            params = (match_query, domain_filter, top_k)
        else:
            sql = """
                SELECT d.item_id
                FROM documents d
                JOIN fts_knowledge f ON d.rowid = f.rowid
                WHERE f.content MATCH ?
                ORDER BY bm25(fts_knowledge)
                LIMIT ?
            """
            params = (match_query, top_k)

        cur = self._conn.execute(sql, params)
        results: List[KnowledgeItem] = []
        for row in cur.fetchall():
            item = self.get(row[0])
            if item is not None:
                results.append(item)
        return results

    def save(self, backend: Any, table: str = "knowledge") -> int:
        """Persist any pending buffer (data is already on disk)."""
        return self.flush()

    def load(self, backend: Any, table: str = "knowledge") -> int:
        """No-op: data is already in the SQLite database."""
        return self.count()

    def close(self) -> None:
        """Flush remaining items and close the database connection."""
        if not self._closed:
            self.flush()
            self._conn.commit()
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "DiskKnowledgeBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
