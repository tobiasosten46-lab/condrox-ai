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
In-memory knowledge backend with BM25 retrieval.

The KnowledgeBackend stores KnowledgeItem instances, supports ingestion,
and performs top-k semantic search using BM25 (Best Matching 25) — the
standard ranking algorithm used in real search engines.

Mathematical Foundation:
- BM25: score(q, d) = Σ IDF(qi) · (f(qi,d)·(k1+1)) / (f(qi,d) + k1·(1−b+b·|d|/avgdl))
- IDF(qi) = log((N − n(qi) + 0.5) / (n(qi) + 0.5) + 1)
- Top-k retrieval with a bounded min-heap: O(n log k).
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
import zlib
from collections import Counter
from typing import Any, Dict, List, Optional, Set

import numpy as np

from core.knowledge.knowledge_models import KnowledgeItem
from core.knowledge.provenance import KnowledgeProvenance, ProvenanceRecord

# Standard English stop words removed from queries before BM25 scoring.
# These words appear in most documents and carry little discriminative value.
# Removing them prevents irrelevant results when only stop words match.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "is", "a", "an", "what", "who", "how", "why",
    "where", "when", "of", "to", "in", "for", "and", "or",
    "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this",
    "that", "these", "those", "it", "its", "on", "at",
    "by", "with", "from", "as", "into", "about", "your",
    "you", "me", "my", "we", "our", "i", "please",
    "tell", "explain", "describe",
})


class KnowledgeBackend:
    """
    In-memory knowledge backend with BM25 ranking.

    Uses an inverted index for O(q · df) retrieval instead of scanning
    all documents. BM25 provides term-frequency saturation and document
    length normalization — far superior to cosine similarity on
    bag-of-words hash buckets.

    Attributes:
        _items: Mapping from item_id to KnowledgeItem.
        _domain_index: Mapping from domain to item ids.
        _inverted_index: Mapping from token to set of item_ids containing it.
        _doc_freq: Mapping from token to number of documents containing it.
        _doc_tokens: Mapping from item_id to Counter of token frequencies.
        _doc_len: Mapping from item_id to document length (token count).
        _avg_doc_len: Average document length across all items.
        _bm25_k1: BM25 term frequency saturation parameter.
        _bm25_b: BM25 document length normalization parameter.

    Complexity:
        ingest: T(n) ∈ O(n) for token extraction, S(n) ∈ O(v) for vocabulary.
        search: T(n) ∈ O(q · df_avg · log k) for q query terms, df_avg docs per term.
    """

    __slots__ = (
        "_items",
        "_domain_index",
        "_inverted_index",
        "_doc_freq",
        "_doc_tokens",
        "_doc_len",
        "_avg_doc_len",
        "_bm25_k1",
        "_bm25_b",
        "_embedding_dim",
        "_min_score",
        "_provenance",
    )

    def __init__(self, embedding_dim: int = 128) -> None:
        """Initialize the knowledge backend.

        Args:
            embedding_dim: Retained for API compatibility; embeddings are
                generated lazily for save/load compatibility.
        """
        self._items: Dict[str, KnowledgeItem] = {}
        self._domain_index: Dict[str, Set[str]] = {}
        self._inverted_index: Dict[str, Set[str]] = {}
        self._doc_freq: Dict[str, int] = {}
        self._doc_tokens: Dict[str, Counter] = {}
        self._doc_len: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._bm25_k1: float = 1.5
        self._bm25_b: float = 0.75
        self._embedding_dim = embedding_dim
        self._min_score: float = 0.5  # increased minimum BM25 score for better relevance
        self._provenance: KnowledgeProvenance = KnowledgeProvenance()

    @property
    def provenance(self) -> KnowledgeProvenance:
        """Return the provenance registry.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._provenance

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize and stem text into lowercase alphanumeric words.

        Applies a lightweight suffix-stripping stemmer to normalize
        common English inflections (plurals, -ing, -ed, -ly) so that
        "functions" matches "function" and "running" matches "run".

        Args:
            text: Input string.

        Returns:
            List of normalized, stemmed tokens.

        Complexity: T(n) ∈ O(n) where n is text length, S(n) ∈ O(n).
        """
        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
        stemmed: List[str] = []
        for token in raw_tokens:
            if len(token) <= 3:
                stemmed.append(token)
                continue
            # Strip common suffixes — up to 2 passes so 'functions' → 'function' → 'funct'
            # matches 'function' → 'funct'. Ordered longest-first for correct matching.
            # Note: 'es' is omitted — 's' handles plurals (databases→databas matches database→databas).
            # 'es' caused double-stripping: databases→database→databa ≠ database→databas.
            for _ in range(2):
                for suffix in ("ingly", "edly", "ation", "ies", "ied", "ming", "ing", "ion", "ive", "ity", "ed", "ly", "s"):
                    if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                        token = token[: -len(suffix)]
                        break
                else:
                    break  # no suffix stripped this pass, stop
            stemmed.append(token)
        return stemmed

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Public tokenize wrapper.

        Args:
            text: Input string.

        Returns:
            List of normalized, stemmed tokens.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return KnowledgeBackend._tokenize(text)

    @classmethod
    def stop_words(cls) -> frozenset[str]:
        """Return the set of stop words used during retrieval.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return _STOP_WORDS

    @classmethod
    def is_stop_word(cls, token: str) -> bool:
        """Return True if the token is a stop word.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return token in _STOP_WORDS

    def _embed(self, text: str) -> np.ndarray:
        """Create a simple embedding for save/load compatibility.

        Generates a TF-IDF weighted vector using the current document
        frequencies. Used only for persistence — search uses BM25 directly.

        Args:
            text: Input string.

        Returns:
            1-D numpy vector (embedding_dim dimensions).

        Complexity: T(n) ∈ O(n) for n tokens.
        """
        tokens = self._tokenize(text)
        vec = np.zeros(self._embedding_dim, dtype=np.float64)
        n_docs = max(1, len(self._items))
        for token in tokens:
            idx = zlib.crc32(token.encode("utf-8")) % self._embedding_dim
            df = self._doc_freq.get(token, 0)
            idf = math.log((n_docs + 1.0) / (df + 1.0) + 1.0) if df > 0 else 1.0
            vec[idx] += idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def ingest(
        self,
        content: str,
        domain: str = "general",
        item_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> KnowledgeItem:
        """Ingest new knowledge into the backend.

        Args:
            content: Knowledge content.
            domain: Domain/category.
            item_id: Optional explicit id.
            confidence: Source confidence in [0.0, 1.0].
            metadata: Optional metadata.

        Returns:
            The stored KnowledgeItem.

        Complexity: T(n) ∈ O(n + d²), S(n) ∈ O(d).
        """
        if not content:
            raise ValueError("content must be non-empty")
        final_id = item_id or f"kno_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

        # Build inverted index entries
        tokens = self._tokenize(content)
        token_counts = Counter(tokens)
        self._doc_tokens[final_id] = token_counts
        self._doc_len[final_id] = len(tokens)
        for token in token_counts:
            self._inverted_index.setdefault(token, set()).add(final_id)
            self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        # Update average document length
        n = len(self._items)
        old_total = self._avg_doc_len * n
        self._avg_doc_len = (old_total + len(tokens)) / (n + 1)

        # Create embedding for save/load compatibility
        embedding = self._embed(content)
        item = KnowledgeItem(
            item_id=final_id,
            domain=domain,
            content=content,
            embedding=embedding,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._items[final_id] = item
        self._domain_index.setdefault(domain, set()).add(final_id)
        return item

    def get(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve a knowledge item by id.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._items.get(item_id)

    def items(self) -> Iterable[KnowledgeItem]:
        """Iterate over all stored knowledge items.

        Returns:
            Iterable of KnowledgeItem instances.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1) for the iterator.
        """
        return list(self._items.values())

    def count(self) -> int:
        """Return the number of stored knowledge items.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._items)

    def _bm25_score(self, query_tokens: List[str], item_id: str, original_query: str = "") -> float:
        """Compute BM25 score for a document against query tokens.

        BM25: score = Σ IDF(qi) · (f(qi,d)·(k1+1)) / (f(qi,d) + k1·(1−b+b·|d|/avgdl))

        Adds an exact match boost: if the original query word appears as a
        substring in the document content, that term's contribution is doubled.
        This prevents short docs matching a common term from beating longer
        docs containing the exact query word.

        Args:
            query_tokens: Tokenized query (stemmed, stop words removed).
            item_id: Document to score.
            original_query: Original query string for exact match boost.

        Returns:
            BM25 score (higher = more relevant).

        Complexity: T(n) ∈ O(q) for q unique query terms.
        """
        n_docs = len(self._items)
        if n_docs == 0:
            return 0.0

        doc_tokens = self._doc_tokens.get(item_id)
        if doc_tokens is None:
            return 0.0

        doc_len = self._doc_len.get(item_id, 0)
        avgdl = self._avg_doc_len if self._avg_doc_len > 0 else 1.0
        k1 = self._bm25_k1
        b = self._bm25_b

        # Extract original content words for exact match boost
        content_lower = self._items[item_id].content.lower() if item_id in self._items else ""
        orig_words = set(re.findall(r"[a-z0-9]+", original_query.lower())) if original_query else set()

        score = 0.0
        matched_terms = 0
        for term in set(query_tokens):
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            tf = doc_tokens.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / avgdl))
            term_score = idf * tf_norm

            # Exact match boost: if the original query word appears as substring
            # in the document content, boost that term's contribution by 3x.
            # This ensures a doc containing 'recursion' (exact word) beats a
            # shorter doc matching only 'programming' when the query asks about
            # 'recursion in programming'. Both match via stemming, but the exact
            # match is a stronger relevance signal than BM25 length normalization.
            if df < n_docs:
                for orig_word in orig_words:
                    if orig_word in content_lower and len(orig_word) > 3:
                        stemmed_orig = self._tokenize(orig_word)
                        if stemmed_orig and stemmed_orig[0] == term:
                            term_score *= 3.0
                            break

            score += term_score
            matched_terms += 1

        # Query coverage bonus: documents matching more unique query terms
        # are more relevant. Each additional matched term adds 50% bonus.
        if matched_terms > 1:
            score *= (1.0 + 0.5 * (matched_terms - 1))

        return score

    def bm25_score(
        self,
        query_tokens: List[str],
        item_id: str,
        original_query: str = "",
    ) -> float:
        """Public BM25 scoring wrapper.

        Args:
            query_tokens: Tokenized query (stemmed, stop words removed).
            item_id: Document to score.
            original_query: Original query string for exact match boost.

        Returns:
            BM25 score (higher = more relevant).

        Complexity: T(n) ∈ O(q) for q unique query terms.
        """
        return self._bm25_score(query_tokens, item_id, original_query)

    def candidate_ids(
        self,
        query_tokens: Iterable[str],
        domain_filter: Optional[str] = None,
    ) -> Set[str]:
        """Return item ids that contain any of the query tokens.

        Args:
            query_tokens: Iterable of query tokens.
            domain_filter: Optional domain to restrict results to.

        Returns:
            Set of candidate item ids.

        Complexity: T(n) ∈ O(q · df_avg) for q tokens.
        """
        candidates: Set[str] = set()
        for token in query_tokens:
            ids = self._inverted_index.get(token)
            if ids:
                # Snapshot the set to avoid "set changed size during iteration"
                # if another thread ingests knowledge concurrently.
                candidates.update(list(ids))
        if domain_filter and candidates:
            domain_ids = self._domain_index.get(domain_filter, set())
            candidates &= set(domain_ids)
        return candidates

    def item_token_set(self, item_id: str) -> Set[str]:
        """Return the set of tokens for a stored item.

        If the item is not indexed, its content is tokenized on demand.

        Args:
            item_id: Item identifier.

        Returns:
            Set of tokens.

        Complexity: T(n) ∈ O(t) for t tokens.
        """
        doc_tokens = self._doc_tokens.get(item_id)
        if doc_tokens is None:
            item = self._items.get(item_id)
            if item is None:
                return set()
            return set(self.tokenize(item.content))
        return set(doc_tokens.keys())

    def search(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[KnowledgeItem]:
        """Search knowledge using BM25 ranking with inverted index.

        Uses the inverted index to only score documents that contain at
        least one query term — O(q · df_avg) instead of O(n) scan.

        Args:
            query: Query text.
            top_k: Maximum number of results.
            domain_filter: Optional domain constraint.

        Returns:
            Top-k KnowledgeItem instances sorted by descending BM25 score.

        Complexity: T(n) ∈ O(q · df_avg · log k) for q query terms.
        """
        if not query or not self._items:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Filter stop words from query tokens — prevents irrelevant matches
        # on common words like "what", "is", "the" when key terms don't exist.
        content_tokens = [t for t in query_tokens if t not in _STOP_WORDS]
        if not content_tokens:
            # All query tokens are stop words — cannot match meaningfully
            return []

        top_k = max(1, top_k)

        # Collect candidate docs via inverted index using only content tokens
        candidates: Set[str] = set()
        for term in content_tokens:
            ids = self._inverted_index.get(term)
            if ids:
                # Snapshot to avoid concurrent modification during iteration
                candidates.update(list(ids))

        if domain_filter:
            domain_ids = self._domain_index.get(domain_filter, set())
            candidates &= set(domain_ids)

        if not candidates:
            return []

        # ─── Term coverage filter ────────────────────────────────────────
        # For multi-term queries, require documents to match a minimum number
        # of content tokens. This prevents "star wars" from matching astronomy
        # articles about "stars" (which don't contain "wars").
        # - 1 token: any match (min 1)
        # - 2 tokens: both must match (min 2)
        # - 3+ tokens: at least 2/3 must match
        unique_content_tokens = set(content_tokens)
        n_tokens = len(unique_content_tokens)
        if n_tokens <= 1:
            min_coverage = 1
        elif n_tokens == 2:
            min_coverage = 2
        else:
            min_coverage = max(2, (n_tokens * 2 + 2) // 3)

        # Score candidates with BM25 using content tokens only.
        # For very small collections the absolute threshold can be too
        # aggressive because IDF scores are naturally low; accept any match.
        effective_min_score = self._min_score if len(self._items) > 5 else 0.0

        # Score candidates with BM25 using content tokens only
        heap: List[tuple[float, int, str]] = []
        counter = 0
        for item_id in candidates:
            # Check term coverage — skip docs that don't match enough terms
            doc_tokens = self._doc_tokens.get(item_id, {})
            matched = sum(1 for t in unique_content_tokens if t in doc_tokens)
            if matched < min_coverage:
                continue
            score = self._bm25_score(content_tokens, item_id, original_query=query)
            if score < effective_min_score:
                continue
            entry = (score, counter, item_id)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)
            counter += 1

        # If coverage filter was too aggressive (no results), fall back to
        # accepting any single-term match
        if not heap and min_coverage > 1:
            for item_id in candidates:
                score = self._bm25_score(content_tokens, item_id, original_query=query)
                if score < effective_min_score:
                    continue
                entry = (score, counter, item_id)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
                counter += 1

        heap.sort(key=lambda x: x[0], reverse=True)
        return [self._items[item_id] for _, _, item_id in heap]

    def save(self, backend: "core.storage.StorageBackend", table: str = "knowledge") -> int:
        """Persist all knowledge items to a storage backend.

        Args:
            backend: Storage backend to write to.
            table: Target table name.

        Returns:
            Number of items saved.

        Complexity: T(n) ∈ O(n) for n items, S(n) ∈ O(n).
        """
        items_data: List[Dict[str, Any]] = []
        for item in list(self._items.values()):
            embedding_list = (
                item.embedding.tolist() if item.embedding is not None else None
            )
            items_data.append({
                "item_id": item.item_id,
                "domain": item.domain,
                "content": item.content,
                "embedding": embedding_list,
                "confidence": item.confidence,
                "metadata": dict(item.metadata) if item.metadata else {},
            })
        backend.save_items(table, items_data)
        return len(items_data)

    def load(self, backend: "core.storage.StorageBackend", table: str = "knowledge") -> int:
        """Load knowledge items from a storage backend.

        Rebuilds the inverted index and BM25 statistics from loaded content.

        Args:
            backend: Storage backend to read from.
            table: Source table name.

        Returns:
            Number of items loaded.

        Complexity: T(n) ∈ O(n · t) for n items, t avg tokens per item.
        """
        items_data = backend.load_items(table)
        count = 0
        for data in items_data:
            embedding_raw = data.get("embedding")
            embedding = (
                np.array(embedding_raw, dtype=np.float64)
                if embedding_raw is not None
                else None
            )
            item_id = data["item_id"]
            content = data["content"]
            item = KnowledgeItem(
                item_id=item_id,
                domain=data.get("domain", "general"),
                content=content,
                embedding=embedding,
                confidence=data.get("confidence", 1.0),
                metadata=data.get("metadata", {}),
            )
            self._items[item_id] = item
            self._domain_index.setdefault(item.domain, set()).add(item_id)

            # Rebuild inverted index
            tokens = self._tokenize(content)
            token_counts = Counter(tokens)
            self._doc_tokens[item_id] = token_counts
            self._doc_len[item_id] = len(tokens)
            for token in token_counts:
                self._inverted_index.setdefault(token, set()).add(item_id)
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

            count += 1

        # Recompute average document length
        if self._doc_len:
            self._avg_doc_len = sum(self._doc_len.values()) / len(self._doc_len)

        return count
