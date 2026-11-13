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

"""Tests for ConversationalMemoryGraph — Phase 2-E component."""

from __future__ import annotations

from core.conversational.memory_graph import (
    ConversationalMemoryGraph,
    GraphEdge,
    MemoryNode,
)


class TestConversationalMemoryGraph:
    """Unit tests for ConversationalMemoryGraph."""

    def test_add_and_get_node(self) -> None:
        """Nodes can be added and retrieved."""
        graph = ConversationalMemoryGraph()
        graph.add_node("n1", "Python is a language", tags=("python",))
        node = graph.get_node("n1")
        assert node is not None
        assert node.content == "Python is a language"

    def test_add_edge(self) -> None:
        """Edges connect nodes."""
        graph = ConversationalMemoryGraph()
        graph.add_node("n1", "Python")
        graph.add_node("n2", "Programming")
        edge = graph.add_edge("n1", "n2", relation="related")
        assert edge is not None
        assert graph.edge_count() == 1

    def test_add_edge_missing_node(self) -> None:
        """Edge to missing node returns None."""
        graph = ConversationalMemoryGraph()
        graph.add_node("n1", "test")
        edge = graph.add_edge("n1", "missing")
        assert edge is None

    def test_get_neighbors(self) -> None:
        """Neighbors are found via BFS."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_node("b", "beta")
        graph.add_node("c", "gamma")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        neighbors = graph.get_neighbors("a", max_depth=2)
        assert len(neighbors) == 2
        ids = {n.node_id for n, _, _ in neighbors}
        assert "b" in ids
        assert "c" in ids

    def test_get_neighbors_filtered_by_relation(self) -> None:
        """Neighbors can be filtered by relation."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_node("b", "beta")
        graph.add_node("c", "gamma")
        graph.add_edge("a", "b", relation="related")
        graph.add_edge("a", "c", relation="causes")
        neighbors = graph.get_neighbors("a", relation="related")
        assert len(neighbors) == 1
        assert neighbors[0][0].node_id == "b"

    def test_search_by_keyword(self) -> None:
        """Search finds nodes by keyword overlap."""
        graph = ConversationalMemoryGraph()
        graph.add_node("n1", "Python programming language")
        graph.add_node("n2", "JavaScript framework")
        results = graph.search("Python language", top_k=2)
        assert len(results) > 0
        assert results[0][0].node_id == "n1"

    def test_search_with_tag_filter(self) -> None:
        """Search filters by tag."""
        graph = ConversationalMemoryGraph()
        graph.add_node("n1", "Python code", tags=("python",))
        graph.add_node("n2", "Python data", tags=("data",))
        results = graph.search("Python", top_k=5, tag_filter="python")
        assert all("python" in n.tags for n, _ in results)

    def test_remove_node(self) -> None:
        """Removing a node removes its edges."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_node("b", "beta")
        graph.add_edge("a", "b")
        assert graph.remove_node("a") is True
        assert graph.get_node("a") is None
        assert graph.edge_count() == 0

    def test_remove_missing_node(self) -> None:
        """Removing missing node returns False."""
        graph = ConversationalMemoryGraph()
        assert graph.remove_node("missing") is False

    def test_node_and_edge_count(self) -> None:
        """Counts are correct."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_node("b", "beta")
        graph.add_edge("a", "b")
        assert graph.node_count() == 2
        assert graph.edge_count() == 1

    def test_all_nodes(self) -> None:
        """all_nodes returns all nodes."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_node("b", "beta")
        assert len(graph.all_nodes()) == 2

    def test_clear(self) -> None:
        """Clear removes everything."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_edge("a", "a")
        graph.clear()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_get_neighbors_missing_node(self) -> None:
        """Neighbors of missing node returns empty."""
        graph = ConversationalMemoryGraph()
        assert graph.get_neighbors("missing") == []

    def test_memory_node_is_frozen(self) -> None:
        """MemoryNode is immutable."""
        graph = ConversationalMemoryGraph()
        graph.add_node("n1", "test")
        node = graph.get_node("n1")
        try:
            node.content = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_graph_edge_is_frozen(self) -> None:
        """GraphEdge is immutable."""
        graph = ConversationalMemoryGraph()
        graph.add_node("a", "alpha")
        graph.add_node("b", "beta")
        edge = graph.add_edge("a", "b")
        try:
            edge.relation = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass
