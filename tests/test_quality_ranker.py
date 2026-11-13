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

"""Tests for KnowledgeQualityRanker and KnowledgeRouter integration."""

from __future__ import annotations

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_router import KnowledgeRouter
from core.knowledge.quality_ranker import KnowledgeQualityRanker


class TestKnowledgeQualityRanker:
    """Unit tests for the quality ranker."""

    def _sample_backend(self) -> KnowledgeBackend:
        """Return a backend with a mix of relevant and less relevant items."""
        backend = KnowledgeBackend()
        backend.ingest(
            content="Python is a high-level programming language used for scripting.",
            domain="programming",
            item_id="py_001",
        )
        backend.ingest(
            content="Python supports decorators, a powerful syntax for wrapping functions.",
            domain="programming",
            item_id="py_002",
        )
        backend.ingest(
            content="JavaScript runs in browsers and on servers via Node.js.",
            domain="programming",
            item_id="js_001",
        )
        return backend

    def test_search_returns_relevant_items(self) -> None:
        """Quality search returns relevant items even when BM25 score is low."""
        backend = self._sample_backend()
        ranker = KnowledgeQualityRanker(backend)

        results = ranker.search("python language", top_k=5)
        ids = [item.item_id for item in results]
        assert "py_001" in ids
        # py_002 is about decorators, not "python language" — may or may not appear

    def test_search_filters_stop_words(self) -> None:
        """Stop words in the query are ignored."""
        backend = self._sample_backend()
        ranker = KnowledgeQualityRanker(backend)

        results = ranker.search("what is python", top_k=5)
        ids = [item.item_id for item in results]
        assert "py_001" in ids

    def test_search_respects_domain_filter(self) -> None:
        """Domain filter restricts results to the requested domain."""
        backend = self._sample_backend()
        ranker = KnowledgeQualityRanker(backend)

        results = ranker.search("python", top_k=5, domain_filter="programming")
        assert all(item.domain == "programming" for item in results)

    def test_search_no_matches_returns_empty(self) -> None:
        """A query with no matching terms returns empty."""
        backend = self._sample_backend()
        ranker = KnowledgeQualityRanker(backend)

        results = ranker.search("quantum entanglement", top_k=5)
        assert results == []

    def test_rank_reorders_input_items(self) -> None:
        """rank() reorders a provided list by combined score."""
        backend = self._sample_backend()
        ranker = KnowledgeQualityRanker(backend)

        items = [backend.get("js_001"), backend.get("py_001")]
        items = [i for i in items if i is not None]
        results = ranker.rank("python language", items, top_k=5)
        assert results[0].item_id == "py_001"

    def test_router_uses_quality_ranker(self) -> None:
        """KnowledgeRouter.route returns results from the quality ranker."""
        router = KnowledgeRouter()
        router.ingest(
            content="Functions in Python use the def keyword.",
            domain="programming",
            item_id="py_func",
        )
        results = router.route(
            "How do I define a function?", domain="programming", top_k=5
        )
        assert len(results) > 0
        assert results[0].domain == "programming"
