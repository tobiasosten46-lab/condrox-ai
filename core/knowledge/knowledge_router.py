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

"""Knowledge router for the unified Brain system.

The KnowledgeRouter selects the appropriate knowledge backend and domain
based on the current state intent and context. It now combines BM25,
quality re-ranking, and optional graph expansion.
"""

from __future__ import annotations

from typing import List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_graph import KnowledgeGraph
from core.knowledge.knowledge_models import KnowledgeItem
from core.knowledge.quality_ranker import KnowledgeQualityRanker


class KnowledgeRouter:
    """
    Router that dispatches knowledge retrieval to the active backend.

    Layers:
      1. KnowledgeQualityRanker for combined BM25 + overlap scoring.
      2. KnowledgeGraph for expanding results with related items.

    Attributes:
        backend: Primary KnowledgeBackend instance.
        graph: KnowledgeGraph layer.
        ranker: KnowledgeQualityRanker instance.
        default_top_k: Default number of results.

    Complexity: T(route) ∈ O(1), S(route) ∈ O(1).
    """

    __slots__ = ("backend", "graph", "ranker", "default_top_k")

    def __init__(self, backend: Optional[KnowledgeBackend] = None, default_top_k: int = 5) -> None:
        """Initialize the router.

        Args:
            backend: Optional KnowledgeBackend; a new one is created if None.
            default_top_k: Default number of knowledge items to retrieve.
        """
        self.backend = backend if backend is not None else KnowledgeBackend()
        self.graph: KnowledgeGraph = KnowledgeGraph()
        self.ranker: KnowledgeQualityRanker = KnowledgeQualityRanker(self.backend)
        self.default_top_k = max(1, default_top_k)

    def build_graph(self, min_jaccard: float = 0.15, max_edges_per_node: int = 20) -> None:
        """Build or rebuild the knowledge graph from the current backend.

        Args:
            min_jaccard: Minimum Jaccard similarity for an edge.
            max_edges_per_node: Maximum edges retained per node.

        Complexity: T(n) ∈ O(backend build complexity), S(n) ∈ O(nodes + edges).
        """
        self.graph.build(
            self.backend,
            min_jaccard=min_jaccard,
            max_edges_per_node=max_edges_per_node,
        )

    def route(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
        use_graph: bool = True,
    ) -> List[KnowledgeItem]:
        """Route a knowledge query through ranker and optional graph expansion.

        Args:
            query: Query text.
            domain: Optional domain filter.
            top_k: Optional result count override.
            use_graph: Whether to expand results through the knowledge graph.

        Returns:
            List of relevant KnowledgeItem instances.

        Complexity: T(n) ∈ O(n log k + m + d²) for n items, query length m.
        """
        k = top_k if top_k is not None else self.default_top_k
        candidate_k = k * 3 if use_graph else k

        # 1. Quality-ranked candidate pool.
        candidates = self.ranker.search(
            query, top_k=candidate_k, domain_filter=domain
        )

        if not candidates:
            return []

        # 2. Optional graph expansion.
        if use_graph and self.graph is not None and self.graph.is_built:
            seed_ids = [item.item_id for item in candidates]
            expanded = self.graph.expand(
                seed_ids=seed_ids,
                query=query,
                top_k=candidate_k,
                max_hops=1,
            )
            seen = {item.item_id: item for item in candidates}
            for node_id, _ in expanded:
                if node_id in seen:
                    continue
                item = self.backend.get(node_id)
                if item is not None:
                    seen[node_id] = item
            pool = list(seen.values())
        else:
            pool = candidates

        # 3. Final quality ranking.
        return self.ranker.rank(query, pool, top_k=k)

    def ingest(
        self,
        content: str,
        domain: str = "general",
        item_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> KnowledgeItem:
        """Ingest knowledge through the router.

        Complexity: T(n) ∈ O(n + d²), S(n) ∈ O(d).
        """
        return self.backend.ingest(
            content=content,
            domain=domain,
            item_id=item_id,
            confidence=confidence,
            metadata=metadata,
        )
