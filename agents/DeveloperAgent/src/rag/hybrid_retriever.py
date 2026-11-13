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
Hybrid Retriever — BM25 + keyword hybrid retrieval for DeveloperAgent RAG.

Combines lexical BM25 scoring with keyword relevance scoring to produce
a unified ranked result set. Configurable weights allow tuning the
balance between lexical precision and keyword recall.

Mathematical Foundation:
    BM25 score for document d and query q:
        BM25(d, q) = sum over terms t in q of:
            IDF(t) * (f(t, d) * (k1 + 1)) /
            (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

    Hybrid score = w_bm25 * normalize(BM25) + w_keyword * normalize(keyword)

    where normalize(x) = x / max(all_scores) if max > 0 else 0.

Complexity:
    index: T(n) in O(n) where n = chunks, S(n) in O(index).
    search: T(n) in O(k * c) where k = query terms, c = chunks,
            S(n) in O(results).

Invariants:
    - Results always sorted by hybrid score descending.
    - BM25 and keyword scores normalized to [0.0, 1.0] before weighting.
    - Empty index -> empty results.

Edge Cases:
    - Empty query -> empty results.
    - Single chunk -> BM25 may be zero (no IDF), keyword score used.
    - All scores zero -> results returned with score 0.0 (preserves order).

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.rag.chunker import CodeChunk
from core.rag.retriever import Retriever, RetrievalResult


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HybridResult:
    """
    Immutable hybrid retrieval result.

    Complexity: T(n) in O(1), S(n) in O(content).
    Invariants: hybrid_score = w_bm25 * bm25_norm + w_kw * kw_norm.
    Edge Cases: all scores zero when no terms match.
    """

    chunk: CodeChunk
    bm25_score: float
    keyword_score: float
    hybrid_score: float
    matched_terms: Tuple[str, ...]


# ─── Hybrid Retriever ───────────────────────────────────────────────────────


class HybridRetriever:
    """
    Hybrid BM25 + keyword retriever for DeveloperAgent RAG pipeline.

    Combines lexical BM25 scoring (term frequency / inverse document
    frequency with length normalization) with keyword relevance scoring
    (name match bonus + content frequency). Produces a unified ranked
    result set with configurable weights.

    Complexity:
        index_chunks: T(n) in O(n) where n = chunks, S(n) in O(index).
        index_file: T(n) in O(n) where n = code length, S(n) in O(chunks).
        search: T(n) in O(k * c) where k = query terms, c = chunks,
                S(n) in O(results).
    Invariants:
        - Results sorted by hybrid_score descending.
        - Scores normalized to [0.0, 1.0] before weighting.
        - Empty index -> empty results.
    Edge Cases:
        - Empty query -> empty tuple.
        - Single chunk -> BM25 may be zero, keyword score used.
        - All scores zero -> results with score 0.0.
    """

    __slots__ = (
        "_chunks",
        "_doc_lengths",
        "_avg_doc_length",
        "_df",
        "_num_docs",
        "_keyword_retriever",
        "_bm25_weight",
        "_keyword_weight",
        "_k1",
        "_b",
    )

    _STOP_WORDS: frozenset[str] = frozenset({
        "the", "a", "an", "is", "are", "was", "were",
        "how", "what", "why", "when", "where", "who",
        "do", "does", "did", "can", "could", "should",
        "to", "in", "on", "at", "for", "of", "with",
        "and", "or", "not", "but", "if", "then",
        "this", "that", "these", "those",
    })

    def __init__(
        self,
        bm25_weight: float = 0.6,
        keyword_weight: float = 0.4,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Initialize HybridRetriever with BM25 parameters.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: weights should sum to 1.0 (not enforced).
        """
        self._chunks: List[CodeChunk] = []
        self._doc_lengths: Dict[int, int] = {}
        self._avg_doc_length: float = 0.0
        self._df: Dict[str, int] = {}
        self._num_docs: int = 0
        self._keyword_retriever: Retriever = Retriever()
        self._bm25_weight: float = bm25_weight
        self._keyword_weight: float = keyword_weight
        self._k1: float = k1
        self._b: float = b

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        """
        Number of indexed chunks.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no chunks indexed -> 0.
        """
        return len(self._chunks)

    def index_chunks(
        self, chunks: Tuple[CodeChunk, ...]
    ) -> None:
        """
        Add chunks to the hybrid index.

        Complexity: T(n) in O(n) where n = chunks, S(n) in O(index).
        Invariants: doc lengths and DF counts updated.
        Edge Cases: empty tuple -> no-op.
        """
        for chunk in chunks:
            idx: int = len(self._chunks)
            self._chunks.append(chunk)
            terms: List[str] = self._tokenize(chunk.content)
            self._doc_lengths[idx] = len(terms)
            unique_terms: set[str] = set(terms)
            for term in unique_terms:
                self._df[term] = self._df.get(term, 0) + 1
        self._num_docs = len(self._chunks)
        if self._num_docs > 0:
            self._avg_doc_length = (
                sum(self._doc_lengths.values()) / self._num_docs
            )
        self._keyword_retriever.index_chunks(chunks)

    def index_file(
        self, file_path: str, code: str, language: str = "python"
    ) -> None:
        """
        Chunk and index a file.

        Complexity: T(n) in O(n) where n = code length, S(n) in O(chunks).
        Edge Cases: empty code -> no-op.
        """
        from core.rag.chunker import CodeChunker
        chunker: CodeChunker = CodeChunker()
        chunks: Tuple[CodeChunk, ...] = chunker.chunk_file(
            file_path, code, language
        )
        self.index_chunks(chunks)

    def search(
        self, query: str, top_k: int = 5
    ) -> Tuple[HybridResult, ...]:
        """
        Search indexed chunks using hybrid BM25 + keyword scoring.

        Complexity: T(n) in O(k * c) where k = query terms, c = chunks,
                    S(n) in O(results).
        Invariants: results sorted by hybrid_score descending.
        Edge Cases: empty index -> empty tuple; empty query -> empty tuple.
        """
        if not self._chunks or not query.strip():
            return ()

        query_terms: List[str] = self._tokenize(query)
        if not query_terms:
            return ()

        bm25_scores: Dict[int, float] = {}
        kw_scores: Dict[int, float] = {}
        matched_terms_map: Dict[int, set[str]] = {}

        for idx, chunk in enumerate(self._chunks):
            bm25: float = self._bm25_score(idx, query_terms)
            kw: float = self._keyword_score(chunk, query_terms)
            if bm25 > 0 or kw > 0:
                bm25_scores[idx] = bm25
                kw_scores[idx] = kw
                matched: set[str] = set()
                for term in query_terms:
                    if term in chunk.content.lower() or term in chunk.name.lower():
                        matched.add(term)
                matched_terms_map[idx] = matched

        if not bm25_scores and not kw_scores:
            return ()

        max_bm25: float = max(bm25_scores.values()) if bm25_scores else 0.0
        max_kw: float = max(kw_scores.values()) if kw_scores else 0.0

        results: List[HybridResult] = []
        for idx in bm25_scores:
            bm25_norm: float = (
                bm25_scores[idx] / max_bm25 if max_bm25 > 0 else 0.0
            )
            kw_norm: float = (
                kw_scores[idx] / max_kw if max_kw > 0 else 0.0
            )
            hybrid: float = (
                self._bm25_weight * bm25_norm
                + self._keyword_weight * kw_norm
            )
            results.append(HybridResult(
                chunk=self._chunks[idx],
                bm25_score=bm25_norm,
                keyword_score=kw_norm,
                hybrid_score=hybrid,
                matched_terms=tuple(sorted(matched_terms_map.get(idx, set()))),
            ))

        results.sort(key=lambda r: r.hybrid_score, reverse=True)
        return tuple(results[:top_k])

    def format_context(self, results: Tuple[HybridResult, ...]) -> str:
        """
        Format hybrid results as context string.

        Complexity: T(n) in O(r * c) where r = results, c = chunk content,
                    S(n) in O(output).
        Edge Cases: no results -> empty string.
        """
        if not results:
            return ""

        parts: List[str] = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.chunk.file_path}::{r.chunk.name} "
                f"(hybrid: {r.hybrid_score:.3f}, "
                f"bm25: {r.bm25_score:.3f}, kw: {r.keyword_score:.3f}, "
                f"lines {r.chunk.start_line}-{r.chunk.end_line})\n"
                f"{r.chunk.content}"
            )
        return "\n\n".join(parts)

    # ─── Private: BM25 Scoring ───────────────────────────────────────────────

    def _bm25_score(
        self, doc_idx: int, query_terms: List[str]
    ) -> float:
        """
        Calculate BM25 score for a document.

        Complexity: T(n) in O(k) where k = query terms, S(n) in O(1).
        Invariants: score >= 0.0.
        Edge Cases: term not in any doc -> IDF=0, contributes nothing.
        """
        doc_len: int = self._doc_lengths.get(doc_idx, 0)
        if doc_len == 0:
            return 0.0

        chunk: CodeChunk = self._chunks[doc_idx]
        doc_terms: List[str] = self._tokenize(chunk.content)
        term_freqs: Dict[str, int] = {}
        for term in doc_terms:
            term_freqs[term] = term_freqs.get(term, 0) + 1

        score: float = 0.0
        for term in query_terms:
            tf: int = term_freqs.get(term, 0)
            if tf == 0:
                continue

            df: int = self._df.get(term, 0)
            if df == 0:
                continue

            idf: float = math.log(
                (self._num_docs - df + 0.5) / (df + 0.5) + 1.0
            )

            numerator: float = tf * (self._k1 + 1)
            denominator: float = tf + self._k1 * (
                1 - self._b + self._b * doc_len / self._avg_doc_length
            )
            score += idf * (numerator / denominator)

        return max(0.0, score)

    def _keyword_score(
        self, chunk: CodeChunk, query_terms: List[str]
    ) -> float:
        """
        Calculate keyword relevance score for a chunk.

        Name matches get 2x weight; content matches get frequency-based
        score capped at 3.0.

        Complexity: T(n) in O(k) where k = query terms, S(n) in O(1).
        Edge Cases: no matches -> 0.0.
        """
        content_lower: str = chunk.content.lower()
        name_lower: str = chunk.name.lower()
        score: float = 0.0

        for term in query_terms:
            term_lower: str = term.lower()
            if term_lower in name_lower:
                score += 2.0
            elif term_lower in content_lower:
                count: int = content_lower.count(term_lower)
                score += min(count * 0.5, 3.0)

        return score

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase terms, removing stop words.

        Complexity: T(n) in O(n) where n = text length, S(n) in O(terms).
        Edge Cases: empty text -> empty list.
        """
        terms: List[str] = re.findall(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower()
        )
        return [
            t for t in terms
            if t not in self._STOP_WORDS and len(t) > 1
        ]
