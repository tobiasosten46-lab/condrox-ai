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

"""Tests for DiskKnowledgeBackend."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.knowledge.disk_knowledge_backend import DiskKnowledgeBackend


class TestDiskKnowledgeBackend:
    """Unit tests for the disk-backed knowledge backend."""

    def test_ingest_and_get(self, tmp_path: Path) -> None:
        """An ingested item can be retrieved by id."""
        db = tmp_path / "kb.db"
        backend = DiskKnowledgeBackend(db_path=db, batch_size=2)
        item = backend.ingest("The quick brown fox jumps.", domain="test")
        backend.flush()
        assert backend.count() == 1
        retrieved = backend.get(item.item_id)
        assert retrieved is not None
        assert retrieved.content == "The quick brown fox jumps."
        assert retrieved.domain == "test"
        backend.close()

    def test_search_ranking(self, tmp_path: Path) -> None:
        """Search returns relevant items ordered by FTS5/BM25."""
        db = tmp_path / "kb2.db"
        backend = DiskKnowledgeBackend(db_path=db, batch_size=10)
        backend.ingest("The quick brown fox", domain="animals")
        backend.ingest("A lazy dog sleeps", domain="animals")
        backend.ingest("Quantum computing uses qubits", domain="science")
        backend.flush()
        results = backend.search("quick brown fox", top_k=2)
        assert len(results) >= 1
        assert "fox" in results[0].content.lower()
        backend.close()

    def test_candidate_ids_and_domain_filter(self, tmp_path: Path) -> None:
        """candidate_ids supports domain filtering."""
        db = tmp_path / "kb3.db"
        backend = DiskKnowledgeBackend(db_path=db, batch_size=10)
        backend.ingest("Machine learning is powerful", domain="ai")
        backend.ingest("A recipe for pancakes", domain="cooking")
        backend.flush()
        ids = backend.candidate_ids(["machine", "learning"], domain_filter="ai")
        assert len(ids) == 1
        backend.close()

    def test_bm25_score_positive(self, tmp_path: Path) -> None:
        """bm25_score returns a positive score for a matching document."""
        db = tmp_path / "kb4.db"
        backend = DiskKnowledgeBackend(db_path=db, batch_size=10)
        item = backend.ingest("Recursion is a function that calls itself.", domain="cs")
        backend.flush()
        score = backend.bm25_score(["recursion"], item.item_id, original_query="recursion")
        assert score > 0
        backend.close()

    def test_flush_returns_written_count(self, tmp_path: Path) -> None:
        """flush returns the number of buffered items written."""
        db = tmp_path / "kb5.db"
        backend = DiskKnowledgeBackend(db_path=db, batch_size=5)
        for i in range(3):
            backend.ingest(f"Document {i} content here.", domain="test")
        written = backend.flush()
        assert written == 3
        backend.close()
