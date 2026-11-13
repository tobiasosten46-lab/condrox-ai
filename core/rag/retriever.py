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
Semantic retriever for Condrox AI RAG system.

Uses keyword-based retrieval (no external embedding model needed).
Can be upgraded to vector-based retrieval with ChromaDB later.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.rag.chunker import CodeChunk, CodeChunker


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    A single retrieval result.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(content)
    """

    chunk: CodeChunk
    score: float
    matched_terms: Tuple[str, ...]


class Retriever:
    """
    Keyword-based semantic retriever.

    Complexity: T(search) ∈ O(k×c) where k = keywords, c = chunks
    S(search) ∈ O(results)
    Invariants: results sorted by score descending
    Edge Cases: empty index → empty results
    """

    __slots__ = ("_index",)

    def __init__(self) -> None:
        self._index: List[CodeChunk] = []

    def index_chunks(self, chunks: Tuple[CodeChunk, ...]) -> None:
        """
        Add chunks to the search index.

        Complexity: T(n) ∈ O(n) where n = chunks, S(n) ∈ O(n)
        Edge Cases: empty tuple → no-op
        """
        self._index.extend(chunks)

    def index_file(self, file_path: str, code: str, language: str = "python") -> None:
        """
        Chunk and index a file.

        Complexity: T(n) ∈ O(n) where n = code length, S(n) ∈ O(chunks)
        Edge Cases: empty code → no-op
        """
        chunker: CodeChunker = CodeChunker()
        chunks: Tuple[CodeChunk, ...] = chunker.chunk_file(file_path, code, language)
        self.index_chunks(chunks)

    def search(
        self, query: str, top_k: int = 5
    ) -> Tuple[RetrievalResult, ...]:
        """
        Search indexed chunks by keyword relevance.

        Complexity: T(n) ∈ O(k×c) where k = query terms, c = chunks
        S(n) ∈ O(results)
        Edge Cases: empty index → empty tuple
        """
        if not self._index:
            return ()

        terms: List[str] = self._extract_terms(query)
        if not terms:
            return ()

        results: List[RetrievalResult] = []

        for chunk in self._index:
            content_lower: str = chunk.content.lower()
            name_lower: str = chunk.name.lower()

            matched: List[str] = []
            score: float = 0.0

            for term in terms:
                term_lower: str = term.lower()
                if term_lower in name_lower:
                    score += 2.0
                    matched.append(term)
                elif term_lower in content_lower:
                    count: int = content_lower.count(term_lower)
                    score += min(count * 0.5, 3.0)
                    matched.append(term)

            if score > 0:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=tuple(matched),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return tuple(results[:top_k])

    def _extract_terms(self, query: str) -> List[str]:
        """
        Extract search terms from query.

        Complexity: T(n) ∈ O(n) where n = query length, S(n) ∈ O(terms)
        Edge Cases: empty query → empty list
        """
        terms: List[str] = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)
        stop_words: set[str] = {
            "the", "a", "an", "is", "are", "was", "were",
            "how", "what", "why", "when", "where", "who",
            "do", "does", "did", "can", "could", "should",
            "to", "in", "on", "at", "for", "of", "with",
            "and", "or", "not", "but", "if", "then",
            "this", "that", "these", "those",
        }
        return [t for t in terms if t.lower() not in stop_words and len(t) > 2]

    def format_context(self, results: Tuple[RetrievalResult, ...]) -> str:
        """
        Format retrieval results as context string.

        Complexity: T(n) ∈ O(r×c) where r = results, c = chunk content
        S(n) ∈ O(output)
        Edge Cases: no results → empty string
        """
        if not results:
            return ""

        parts: List[str] = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.chunk.file_path}::{r.chunk.name} "
                f"(score: {r.score:.1f}, lines {r.chunk.start_line}-{r.chunk.end_line})\n"
                f"{r.chunk.content}"
            )
        return "\n\n".join(parts)

    @property
    def index_size(self) -> int:
        """Number of indexed chunks. Complexity: T(n) ∈ O(1)."""
        return len(self._index)
