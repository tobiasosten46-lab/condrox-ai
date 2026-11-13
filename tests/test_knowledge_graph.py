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

"""Tests for KnowledgeGraph — knowledge expansion layer."""

from __future__ import annotations

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
)
from core.knowledge.knowledge_router import KnowledgeRouter


class TestKnowledgeGraph:
    """Unit tests for KnowledgeGraph and KnowledgeRouter graph integration."""

    def _sample_backend(self) -> KnowledgeBackend:
        """Return a backend with three related programming chunks."""
        backend = KnowledgeBackend()
        backend.ingest(
            content="Python is a high-level programming language used for scripting and data science.",
            domain="programming",
            item_id="py_001",
            confidence=0.9,
        )
        backend.ingest(
            content="Python supports decorators, a powerful syntax for wrapping functions.",
            domain="programming",
            item_id="py_002",
            confidence=0.9,
        )
        backend.ingest(
            content="JavaScript runs in browsers and on servers via Node.js.",
            domain="programming",
            item_id="js_001",
            confidence=0.9,
        )
        return backend

    def test_add_item_creates_node(self) -> None:
        """Adding an item creates a graph node with tokens."""
        graph = KnowledgeGraph()
        backend = self._sample_backend()
        item = backend.get("py_001")
        assert item is not None
        graph.add_item(item)

        node = graph._nodes["py_001"]
        assert isinstance(node, KnowledgeGraphNode)
        assert node.domain == "programming"
        assert "python" in node.tokens

    def test_build_creates_edges(self) -> None:
        """Building the graph creates edges for related items."""
        backend = self._sample_backend()
        graph = KnowledgeGraph()
        graph.build(backend, min_jaccard=0.1, max_edges_per_node=10)

        assert graph._built
        # Two Python items share many tokens and should be linked.
        key = ("py_001", "py_002") if "py_001" < "py_002" else ("py_002", "py_001")
        assert key in graph._edges
        edge = graph._edges[key]
        assert isinstance(edge, KnowledgeGraphEdge)
        assert edge.relation == "related"
        assert edge.strength > 0.0

    def test_expand_returns_related_items(self) -> None:
        """Graph expansion returns related items not in the seed set."""
        backend = self._sample_backend()
        graph = KnowledgeGraph()
        graph.build(backend, min_jaccard=0.1, max_edges_per_node=10)

        # Seed is the first Python item; expansion should surface the second.
        expanded = graph.expand(seed_ids=["py_001"], query="python decorators", top_k=5)
        ids = [node_id for node_id, _ in expanded]
        assert "py_002" in ids

    def test_search_combines_bm25_and_graph(self) -> None:
        """Search includes graph-neighbors beyond BM25 matches."""
        backend = self._sample_backend()
        router = KnowledgeRouter(backend=backend)
        router.build_graph(min_jaccard=0.1, max_edges_per_node=10)

        # Query that matches py_001 should expand to py_002.
        results = router.route("python language", top_k=5, use_graph=True)
        ids = [item.item_id for item in results]
        assert "py_001" in ids
        assert "py_002" in ids

    def test_search_without_graph_falls_back(self) -> None:
        """Disabling graph returns plain BM25 results."""
        backend = self._sample_backend()
        router = KnowledgeRouter(backend=backend)
        # Graph intentionally not built.
        results = router.route("python language", top_k=5, use_graph=False)
        ids = [item.item_id for item in results]
        assert "py_001" in ids

    def test_empty_backend_build(self) -> None:
        """Building on an empty backend sets the built flag without error."""
        backend = KnowledgeBackend()
        graph = KnowledgeGraph()
        graph.build(backend)
        assert graph._built
        assert len(graph._nodes) == 0

    def test_domain_filter_respected(self) -> None:
        """Graph search respects the domain filter passed to the backend."""
        backend = self._sample_backend()
        router = KnowledgeRouter(backend=backend)
        router.build_graph(min_jaccard=0.1, max_edges_per_node=10)

        results = router.route("python", domain="programming", top_k=5, use_graph=True)
        assert all(item.domain == "programming" for item in results)
