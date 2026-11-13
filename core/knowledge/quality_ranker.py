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
Knowledge Quality Ranker — improves BM25 retrieval by combining BM25 scores
with explicit query-token overlap.

The ranker addresses cases where BM25 length normalization or low IDF
underranks highly relevant documents. It scores every candidate that matches
at least one query token and returns the top-k by a combined relevance score.

Mathematical Foundation:
- overlap(query, doc) = |Q ∩ D| / |Q|
- final_score = bm25_score + overlap_weight · overlap(query, doc) · overlap_scale
- Top-k selection with bounded min-heap: O(d log k) for d candidates.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.knowledge.knowledge_models import KnowledgeItem


@dataclass(frozen=True, slots=True)
class RankedKnowledgeItem:
    """
    Immutable result container for ranked knowledge.

    Attributes:
        item: The underlying KnowledgeItem.
        bm25_score: Raw BM25 score.
        overlap_score: Token overlap ratio in [0.0, 1.0].
        final_score: Combined score used for ranking.

    Complexity: S(n) ∈ O(1).
    """

    item: KnowledgeItem
    bm25_score: float
    overlap_score: float
    final_score: float


class KnowledgeQualityRanker:
    """
    Quality-aware ranker for KnowledgeBackend retrieval.

    Unlike plain BM25 search, the ranker does not discard candidates based
    solely on a fixed BM25 threshold. Instead it computes a combined score
    that rewards both classic BM25 relevance and direct query-token coverage.

    Attributes:
        _backend: KnowledgeBackend to search.
        _overlap_weight: Weight applied to the overlap term.
        _overlap_scale: Scaling factor for the overlap term.
        _min_final_score: Minimum combined score to include a result.

    Complexity:
        search: T(n) ∈ O(q · df_avg · log k) for q query terms.
        rank: T(n) ∈ O(d log k) for d input items.
    """

    __slots__ = (
        "_backend",
        "_overlap_weight",
        "_overlap_scale",
        "_min_final_score",
    )

    def __init__(
        self,
        backend: KnowledgeBackend,
        overlap_weight: float = 1.0,
        overlap_scale: float = 5.0,
        min_final_score: float = 0.01,
    ) -> None:
        """Initialize the quality ranker.

        Args:
            backend: KnowledgeBackend to search.
            overlap_weight: Weight applied to the overlap score.
            overlap_scale: Scale factor so overlap competes with BM25 scores.
            min_final_score: Minimum combined score for a candidate to be kept.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._backend = backend
        self._overlap_weight = overlap_weight
        self._overlap_scale = overlap_scale
        self._min_final_score = min_final_score

    def _query_tokens(self, query: str) -> List[str]:
        """Tokenize and filter stop words from a query.

        Args:
            query: Raw query string.

        Returns:
            List of content-bearing query tokens.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        tokens = self._backend.tokenize(query)
        stop_words = self._backend.stop_words()
        return [t for t in tokens if t not in stop_words]

    def _score(self, item: KnowledgeItem, query: str, query_tokens: List[str]) -> RankedKnowledgeItem:
        """Compute a combined relevance score for a single item.

        Args:
            item: Knowledge item to score.
            query: Original query string.
            query_tokens: Stop-filtered query tokens.

        Returns:
            RankedKnowledgeItem with component and final scores.

        Complexity: T(n) ∈ O(q + t) for q query tokens and t item tokens.
        """
        doc_tokens = self._backend.item_token_set(item.item_id)
        overlap = len(set(query_tokens) & doc_tokens) / max(len(query_tokens), 1)
        bm25 = self._backend.bm25_score(query_tokens, item.item_id, original_query=query)
        final = bm25 + self._overlap_weight * overlap * self._overlap_scale

        return RankedKnowledgeItem(
            item=item,
            bm25_score=bm25,
            overlap_score=overlap,
            final_score=final,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[KnowledgeItem]:
        """Search the backend using the combined quality score.

        Args:
            query: Query string.
            top_k: Maximum number of results.
            domain_filter: Optional domain constraint.

        Returns:
            Top-k KnowledgeItem instances sorted by descending final score.

        Complexity: T(n) ∈ O(q · df_avg · log k) for q query terms.
        """
        if not query or self._backend.count() == 0:
            return []

        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return []

        candidates = self._backend.candidate_ids(query_tokens, domain_filter=domain_filter)
        if not candidates:
            return []

        top_k = max(1, top_k)

        # ─── Term coverage filter ────────────────────────────────────────
        # For multi-term queries, require documents to match a minimum number
        # of content tokens. This prevents "star wars" from matching astronomy
        # articles about "stars" (which don't contain "wars").
        # - 1 token: any match (min 1)
        # - 2 tokens: both must match (min 2) — prevents single-word false positives
        # - 3+ tokens: at least 2/3 must match (min ceil(n * 2/3))
        unique_query_tokens = set(query_tokens)
        n_tokens = len(unique_query_tokens)
        if n_tokens <= 1:
            min_coverage = 1
        elif n_tokens == 2:
            min_coverage = 2
        else:
            min_coverage = max(2, (n_tokens * 2 + 2) // 3)

        heap: List[tuple[float, int, str]] = []
        counter = 0
        for item_id in candidates:
            item = self._backend.get(item_id)
            if item is None or len(item.content) < 20:
                continue
            # Check term coverage — skip docs that don't match enough terms
            doc_tokens = self._backend.item_token_set(item_id)
            matched = sum(1 for t in unique_query_tokens if t in doc_tokens)
            if matched < min_coverage:
                continue
            ranked = self._score(item, query, query_tokens)
            if ranked.final_score < self._min_final_score:
                continue
            entry = (ranked.final_score, counter, item_id)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif ranked.final_score > heap[0][0]:
                heapq.heapreplace(heap, entry)
            counter += 1

        # If coverage filter was too aggressive (no results), fall back to
        # accepting any single-term match
        if not heap and min_coverage > 1:
            for item_id in candidates:
                item = self._backend.get(item_id)
                if item is None or len(item.content) < 20:
                    continue
                ranked = self._score(item, query, query_tokens)
                if ranked.final_score < self._min_final_score:
                    continue
                entry = (ranked.final_score, counter, item_id)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif ranked.final_score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
                counter += 1

        heap.sort(key=lambda x: x[0], reverse=True)
        results: List[KnowledgeItem] = []
        for _, _, item_id in heap:
            item = self._backend.get(item_id)
            if item is not None:
                results.append(item)
        return results

    def rank(
        self,
        query: str,
        items: List[KnowledgeItem],
        top_k: int = 5,
    ) -> List[KnowledgeItem]:
        """Re-rank a pre-fetched list of knowledge items.

        Args:
            query: Query string.
            items: Items to rank.
            top_k: Maximum number of results.

        Returns:
            Top-k items sorted by descending final score.

        Complexity: T(n) ∈ O(d log k) for d input items.
        """
        if not items or not query:
            return []

        query_tokens = self._query_tokens(query)
        if not query_tokens:
            return items[:top_k]

        ranked = [self._score(item, query, query_tokens) for item in items]
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return [r.item for r in ranked[:top_k]]
