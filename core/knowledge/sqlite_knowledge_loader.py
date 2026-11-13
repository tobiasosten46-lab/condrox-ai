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
SQLite knowledge loader — loads chunks from the massive ingestion database.

Reads from the SQLite database created by scripts/massive_ingestion.py and
ingests them into the in-memory KnowledgeBackend for pipeline retrieval.

Supports domain-filtered loading and configurable chunk limits to keep
boot time bounded while maximizing knowledge coverage.

Mathematical Foundation:
- SQL query: O(n) for n rows, indexed by domain.
- Ingest: O(n * t) for n chunks, t avg tokens per chunk.
- Memory: O(n * e) for n items, e embedding dimension.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend

logger = logging.getLogger(__name__)


class SQLiteKnowledgeLoader:
    """Load knowledge chunks from a SQLite database into KnowledgeBackend.

    Reads from the massive ingestion database (227K+ chunks across 16 domains)
    with configurable per-domain limits to keep boot time bounded.

    Attributes:
        _db_path: Path to the SQLite knowledge database.
        _max_chunks_per_domain: Maximum chunks to load per domain.
        _domains: Optional list of domains to load (None = all).

    Complexity:
        load_into: T(n) ∈ O(n * t) for n chunks, t avg tokens.
        Memory: S(n) ∈ O(n * e) for n items, e embedding dim.
    """

    __slots__ = ("_db_path", "_max_chunks_per_domain", "_domains")

    def __init__(
        self,
        db_path: str | Path = "E:\\Data\\condrox\\knowledge_db.sqlite",
        max_chunks_per_domain: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> None:
        """Initialize the SQLite knowledge loader.

        Args:
            db_path: Path to the SQLite knowledge database.
            max_chunks_per_domain: Max chunks per domain (default from env or 5000).
            domains: Optional domain filter. None loads all domains.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._db_path: str = str(db_path)
        self._max_chunks_per_domain: int = max_chunks_per_domain or int(
            os.environ.get("CONDROX_SQLITE_KL_MAX_CHUNKS", "5000")
        )
        self._domains: Optional[List[str]] = domains

    def load_into(self, backend: KnowledgeBackend) -> int:
        """Load knowledge from SQLite into the backend.

        Args:
            backend: KnowledgeBackend to ingest into.

        Returns:
            Total number of knowledge items loaded.

        Complexity: T(n) ∈ O(n * t) for n chunks, t avg tokens.
        """
        if not Path(self._db_path).exists():
            logger.warning("SQLite knowledge DB not found: %s", self._db_path)
            return 0

        t0 = time.time()
        total_loaded: int = 0
        domain_counts: Dict[str, int] = {}

        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row

            # Get available domains
            if self._domains is None:
                cursor = conn.execute(
                    "SELECT DISTINCT domain FROM knowledge ORDER BY domain"
                )
                domains = [row["domain"] for row in cursor.fetchall()]
            else:
                domains = self._domains

            logger.info(
                "SQLite knowledge loader: %d domains available in %s",
                len(domains), self._db_path,
            )

            for domain in domains:
                if total_loaded >= self._max_chunks_per_domain * len(domains):
                    break

                # Query chunks for this domain, ordered by id for deterministic sampling
                cursor = conn.execute(
                    "SELECT item_id, domain, content, source, metadata "
                    "FROM knowledge WHERE domain = ? "
                    "ORDER BY id LIMIT ?",
                    (domain, self._max_chunks_per_domain),
                )

                domain_loaded = 0
                batch: List[tuple] = []

                for row in cursor:
                    content = row["content"]
                    if not content or len(content) < 20:
                        continue

                    # Parse metadata
                    metadata = {}
                    meta_str = row["metadata"]
                    if meta_str and meta_str != "{}":
                        try:
                            metadata = json.loads(meta_str)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    metadata["source"] = row["source"]

                    try:
                        backend.ingest(
                            content=content[:2000],
                            domain=domain,
                            confidence=0.8,
                            metadata=metadata,
                        )
                        domain_loaded += 1
                    except Exception:
                        pass  # Skip individual failures

                if domain_loaded > 0:
                    domain_counts[domain] = domain_loaded
                    total_loaded += domain_loaded
                    logger.info(
                        "  Loaded %d chunks from domain '%s'",
                        domain_loaded, domain,
                    )

            conn.close()

        except Exception as exc:
            logger.error("Failed to load from SQLite knowledge DB: %s", exc)

        elapsed = time.time() - t0
        logger.info(
            "SQLite knowledge load complete: %d chunks in %.1fs",
            total_loaded, elapsed,
        )

        return total_loaded

    def get_stats(self) -> Dict[str, int]:
        """Get statistics from the SQLite knowledge database.

        Returns:
            Dictionary mapping domain names to chunk counts.

        Complexity: T(n) ∈ O(d) for d domains (COUNT query).
        """
        if not Path(self._db_path).exists():
            return {}

        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            cursor = conn.execute(
                "SELECT domain, COUNT(*) as count FROM knowledge GROUP BY domain ORDER BY count DESC"
            )
            stats = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return stats
        except Exception:
            return {}
