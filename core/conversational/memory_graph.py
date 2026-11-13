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
Conversational Memory Graph — graph-structured memory with semantic links
between memory nodes for neighbor-based retrieval.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class MemoryNode:
    """
    Immutable node in the memory graph.

    Attributes:
        node_id: Unique identifier.
        content: Memory content text.
        tags: Categorical tags.
        weight: Importance weight in [0, 1].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    node_id: str
    content: str
    tags: tuple[str, ...]
    weight: float


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """
    Immutable edge between memory nodes.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        relation: Edge relation type (e.g., "related", "causes", "contradicts").
        strength: Edge strength in [0, 1].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    source: str
    target: str
    relation: str
    strength: float


class ConversationalMemoryGraph:
    """
    Graph-structured memory with semantic links and neighbor retrieval.

    Complexity:
        add_node: T(n) ∈ O(1), S(n) ∈ O(1).
        add_edge: T(n) ∈ O(1), S(n) ∈ O(1).
        get_neighbors: T(n) ∈ O(d) for d degree, S(n) ∈ O(d).
        search: T(n) ∈ O(n log k) for n nodes, k results.
    """

    __slots__ = ("_nodes", "_edges", "_adjacency", "_reverse_adjacency")

    def __init__(self) -> None:
        """Initialize an empty memory graph.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._nodes: Dict[str, MemoryNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        self._reverse_adjacency: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        content: str,
        tags: Optional[Tuple[str, ...]] = None,
        weight: float = 0.5,
    ) -> MemoryNode:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier.
            content: Memory content.
            tags: Optional tags.
            weight: Importance weight.

        Returns:
            The created MemoryNode.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        node = MemoryNode(
            node_id=node_id,
            content=content,
            tags=tags or (),
            weight=max(0.0, min(1.0, weight)),
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related",
        strength: float = 0.5,
    ) -> Optional[GraphEdge]:
        """Add an edge between two nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            relation: Edge relation type.
            strength: Edge strength in [0, 1].

        Returns:
            The created GraphEdge, or None if nodes don't exist.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if source not in self._nodes or target not in self._nodes:
            return None

        edge_id = f"{source}->{target}:{relation}"
        edge = GraphEdge(
            source=source,
            target=target,
            relation=relation,
            strength=max(0.0, min(1.0, strength)),
        )
        self._edges[edge_id] = edge
        self._adjacency[source].append((target, relation, edge.strength))
        self._reverse_adjacency[target].append((source, relation, edge.strength))
        return edge

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        max_depth: int = 1,
    ) -> List[Tuple[MemoryNode, int, float]]:
        """Get neighbors of a node via BFS.

        Args:
            node_id: Starting node ID.
            relation: Optional relation filter.
            max_depth: Maximum BFS depth.

        Returns:
            List of (node, depth, cumulative_strength) tuples.

        Complexity: T(n) ∈ O(d^h) for d degree, h depth, S(n) ∈ O(n).
        """
        if node_id not in self._nodes:
            return []

        visited: Set[str] = {node_id}
        results: List[Tuple[MemoryNode, int, float]] = []
        queue: List[Tuple[str, int, float]] = [(node_id, 0, 1.0)]

        while queue:
            current_id, depth, strength = queue.pop(0)
            if depth >= max_depth:
                continue

            for neighbor_id, rel, edge_strength in self._adjacency.get(current_id, []):
                if neighbor_id in visited:
                    continue
                if relation is not None and rel != relation:
                    continue

                visited.add(neighbor_id)
                cumulative = strength * edge_strength
                node = self._nodes.get(neighbor_id)
                if node:
                    results.append((node, depth + 1, cumulative))
                    queue.append((neighbor_id, depth + 1, cumulative))

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        tag_filter: Optional[str] = None,
    ) -> List[Tuple[MemoryNode, float]]:
        """Search nodes by keyword overlap with query.

        Args:
            query: Search query text.
            top_k: Maximum results.
            tag_filter: Optional tag to filter by.

        Returns:
            List of (node, score) tuples sorted by descending score.

        Complexity: T(n) ∈ O(n log k) for n nodes, S(n) ∈ O(k).
        """
        query_words = set(query.lower().split())
        if not query_words:
            return []

        heap: List[Tuple[float, int, MemoryNode]] = []
        counter = 0

        for node in self._nodes.values():
            if tag_filter and tag_filter not in node.tags:
                continue

            node_words = set(node.content.lower().split())
            if not node_words:
                continue

            overlap = len(query_words & node_words) / len(query_words | node_words)
            score = overlap * node.weight

            if len(heap) < top_k:
                heapq.heappush(heap, (score, counter, node))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, counter, node))
            counter += 1

        heap.sort(key=lambda x: x[0], reverse=True)
        return [(node, score) for score, _, node in heap]

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges.

        Returns:
            True if a node was removed.

        Complexity: T(n) ∈ O(d) for d degree, S(n) ∈ O(1).
        """
        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]

        # Remove outgoing edges
        edges_to_remove = [
            eid for eid, edge in self._edges.items()
            if edge.source == node_id or edge.target == node_id
        ]
        for eid in edges_to_remove:
            del self._edges[eid]

        # Clean adjacency lists
        self._adjacency.pop(node_id, None)
        self._reverse_adjacency.pop(node_id, None)
        for adj in self._adjacency.values():
            adj[:] = [(n, r, s) for n, r, s in adj if n != node_id]
        for adj in self._reverse_adjacency.values():
            adj[:] = [(n, r, s) for n, r, s in adj if n != node_id]

        return True

    def node_count(self) -> int:
        """Return the number of nodes.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the number of edges.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._edges)

    def all_nodes(self) -> List[MemoryNode]:
        """Return all nodes.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return list(self._nodes.values())

    def clear(self) -> None:
        """Clear all nodes and edges.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
