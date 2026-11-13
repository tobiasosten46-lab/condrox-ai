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

"""Tests for KnowledgeProvenance and loader integration."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_loader import KnowledgeLoader
from core.knowledge.provenance import KnowledgeProvenance, ProvenanceRecord


class TestKnowledgeProvenance:
    """Unit tests for the provenance registry."""

    def test_register_and_get(self) -> None:
        """Registering a record makes it retrievable by chunk id."""
        prov = KnowledgeProvenance()
        record = prov.register(
            chunk_id="chunk_001",
            source_path="/data/knowledge/programming/data.jsonl",
            checksum="abc123",
            ingested_at="2026-07-17T12:00:00+00:00",
        )

        assert isinstance(record, ProvenanceRecord)
        assert record.chunk_id == "chunk_001"
        assert record.source_name == "data.jsonl"

        fetched = prov.get("chunk_001")
        assert fetched is not None
        assert fetched.checksum == "abc123"

    def test_get_missing_returns_none(self) -> None:
        """Missing chunk ids return None."""
        prov = KnowledgeProvenance()
        assert prov.get("missing") is None

    def test_count_and_records(self) -> None:
        """Count and records() reflect registered items."""
        prov = KnowledgeProvenance()
        prov.register("a", "x.txt", "1")
        prov.register("b", "y.txt", "2")
        assert prov.count() == 2
        assert len(list(prov.records())) == 2

    def test_backend_has_provenance(self) -> None:
        """KnowledgeBackend exposes a provenance registry."""
        backend = KnowledgeBackend()
        item = backend.ingest("Python is a language.", domain="programming")
        assert backend.provenance is not None
        backend.provenance.register(item.item_id, "programming.txt", "checksum")
        assert backend.provenance.get(item.item_id) is not None

    def test_loader_registers_provenance(self) -> None:
        """KnowledgeLoader registers provenance for each ingested chunk."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            domain_dir = root / "programming"
            domain_dir.mkdir()
            data_file = domain_dir / "chunks.txt"
            raw_text = "Python is a high-level programming language.\n"
            data_file.write_text(raw_text, encoding="utf-8")

            backend = KnowledgeBackend()
            loader = KnowledgeLoader(
                knowledge_root=root,
                max_chunks_per_domain=10,
                max_files_per_domain=10,
            )
            loaded = loader.load_into(backend)
            assert loaded == 1

            item_id = next(iter(backend.items())).item_id
            record = backend.provenance.get(item_id)
            assert record is not None
            assert record.source_name == "chunks.txt"
            assert record.checksum == hashlib.sha256(
                raw_text.strip().encode()
            ).hexdigest()
