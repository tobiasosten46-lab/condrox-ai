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
Tests for SQLite storage backend and persistence integration.

Verifies save/load round-trips for MemoryStore and KnowledgeBackend,
and BrainController persistence wiring.

Run with:
    $env:PYTHONPATH = "."; python -m pytest tests/test_storage.py -v --tb=short
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from core.brain.brain_config import BrainConfig
from core.brain.brain_controller import BrainController
from core.knowledge.knowledge_backend import KnowledgeBackend
from core.memory.memory_models import MemoryItem
from core.memory.memory_store import MemoryStore
from core.storage import SQLiteStorage, StorageBackend


@pytest.fixture
def tmp_db() -> str:
    """Provide a temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Force garbage collection to release SQLite file handles on Windows
    import gc
    gc.collect()
    for _ in range(3):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            gc.collect()
            import time as _time
            _time.sleep(0.1)


class TestSQLiteStorage:
    """Unit tests for SQLiteStorage backend."""

    def test_save_and_load_items(self, tmp_db: str) -> None:
        """Items saved to SQLite can be loaded back."""
        storage = SQLiteStorage(tmp_db)
        items = [
            {"item_id": "m1", "content": "hello", "tags": ["a", "b"]},
            {"item_id": "m2", "content": "world", "tags": ["c"]},
        ]
        storage.save_items("memories", items)
        loaded = storage.load_items("memories")
        assert len(loaded) == 2
        assert loaded[0]["item_id"] == "m1"
        assert loaded[0]["content"] == "hello"
        storage.close()

    def test_upsert_replaces_existing(self, tmp_db: str) -> None:
        """Saving with same item_id replaces the existing item."""
        storage = SQLiteStorage(tmp_db)
        storage.save_items("memories", [{"item_id": "m1", "content": "old"}])
        storage.save_items("memories", [{"item_id": "m1", "content": "new"}])
        loaded = storage.load_items("memories")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "new"
        storage.close()

    def test_count_items(self, tmp_db: str) -> None:
        """count_items returns the correct count."""
        storage = SQLiteStorage(tmp_db)
        assert storage.count_items("memories") == 0
        storage.save_items("memories", [
            {"item_id": "m1", "content": "a"},
            {"item_id": "m2", "content": "b"},
        ])
        assert storage.count_items("memories") == 2
        storage.close()

    def test_clear_table(self, tmp_db: str) -> None:
        """clear_table removes all items and returns count."""
        storage = SQLiteStorage(tmp_db)
        storage.save_items("memories", [
            {"item_id": "m1", "content": "a"},
            {"item_id": "m2", "content": "b"},
        ])
        removed = storage.clear_table("memories")
        assert removed == 2
        assert storage.count_items("memories") == 0
        storage.close()

    def test_empty_save_is_noop(self, tmp_db: str) -> None:
        """Saving an empty list does not error."""
        storage = SQLiteStorage(tmp_db)
        storage.save_items("memories", [])
        assert storage.count_items("memories") == 0
        storage.close()

    def test_closed_backend_raises(self, tmp_db: str) -> None:
        """Operations on a closed backend raise RuntimeError."""
        storage = SQLiteStorage(tmp_db)
        storage.close()
        with pytest.raises(RuntimeError):
            storage.save_items("memories", [{"item_id": "m1"}])
        with pytest.raises(RuntimeError):
            storage.load_items("memories")


class TestMemoryStorePersistence:
    """Tests for MemoryStore save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_db: str) -> None:
        """Memory items survive a save/load cycle."""
        storage = SQLiteStorage(tmp_db)
        store = MemoryStore()
        item = MemoryItem(
            item_id="mem1",
            content="Test memory",
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            source="test",
            tags=("important", "unit"),
            weight=0.8,
        )
        store.add(item)
        saved_count = store.save(storage)
        assert saved_count == 1

        store2 = MemoryStore()
        loaded_count = store2.load(storage)
        assert loaded_count == 1

        loaded = store2.get("mem1")
        assert loaded is not None
        assert loaded.content == "Test memory"
        assert loaded.source == "test"
        assert loaded.weight == 0.8
        assert "important" in loaded.tags
        np.testing.assert_allclose(loaded.embedding, [0.1, 0.2, 0.3], atol=1e-10)
        storage.close()

    def test_load_empty_table(self, tmp_db: str) -> None:
        """Loading from an empty table returns 0 items."""
        storage = SQLiteStorage(tmp_db)
        store = MemoryStore()
        count = store.load(storage)
        assert count == 0
        assert store.count() == 0
        storage.close()


class TestKnowledgeBackendPersistence:
    """Tests for KnowledgeBackend save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_db: str) -> None:
        """Knowledge items survive a save/load cycle."""
        storage = SQLiteStorage(tmp_db)
        backend = KnowledgeBackend(embedding_dim=128)
        item = backend.ingest(
            content="Python is a programming language.",
            domain="programming",
            confidence=0.9,
            metadata={"source": "test"},
        )
        saved_count = backend.save(storage)
        assert saved_count == 1

        backend2 = KnowledgeBackend(embedding_dim=128)
        loaded_count = backend2.load(storage)
        assert loaded_count == 1

        loaded = backend2.get(item.item_id)
        assert loaded is not None
        assert loaded.content == "Python is a programming language."
        assert loaded.domain == "programming"
        assert loaded.confidence == 0.9
        storage.close()


class TestBrainControllerPersistence:
    """Integration tests for BrainController persistence wiring."""

    def test_persistence_roundtrip(self, tmp_db: str, monkeypatch) -> None:
        """BrainController saves and loads state across restarts."""
        monkeypatch.setenv("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")

        config = BrainConfig(
            persistence_path=tmp_db,
            enable_school_curriculum=False,
        )
        bc1 = BrainController(config=config)
        bc1.boot_system()
        # Clear boot-loaded data so the test is deterministic.
        bc1.memory_manager.store._items.clear()
        bc1.knowledge_router.backend._items.clear()
        bc1.ingest_knowledge("Python uses def for functions.", domain="programming")
        bc1.add_memory("User asked about Python functions.", source="conversation")
        total = bc1.save_state()
        assert total == 2  # 1 memory + 1 knowledge
        bc1.close()

        bc2 = BrainController(config=config)
        bc2.boot_system()
        assert bc2.memory_manager.store.count() == 1
        assert bc2.knowledge_router.backend.count() == 1
        bc2.close()

    def test_no_persistence_by_default(self) -> None:
        """Default config has no persistence — save_state returns 0."""
        bc = BrainController()
        bc.boot_system()
        assert bc.save_state() == 0
