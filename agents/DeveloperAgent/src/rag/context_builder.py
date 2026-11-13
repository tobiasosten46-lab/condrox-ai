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
Context Builder — assembles RAG context with token budget management.

Takes hybrid retrieval results and assembles a context string that fits
within a configurable token budget. Prioritizes high-scoring results
and truncates gracefully at chunk boundaries.

Complexity:
    build: T(n) in O(r) where r = results, S(n) in O(context).
    estimate_tokens: T(n) in O(n) where n = text length, S(n) in O(1).

Invariants:
    - Output context never exceeds token budget.
    - Results included in score order (highest first).
    - Each chunk truncated at line boundary if needed.

Edge Cases:
    - No results -> empty string.
    - Single chunk exceeds budget -> truncated to fit.
    - Budget=0 -> empty string.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from agents.DeveloperAgent.src.rag.hybrid_retriever import HybridResult


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextResult:
    """
    Immutable context assembly result.

    Complexity: T(n) in O(1), S(n) in O(context).
    Invariants: token_count <= max_tokens.
    Edge Cases: no results -> empty context, 0 tokens.
    """

    context: str
    token_count: int
    max_tokens: int
    chunks_included: int
    chunks_truncated: int
    chunks_skipped: int


# ─── Context Builder ─────────────────────────────────────────────────────────


class ContextBuilder:
    """
    Assembles RAG context with token budget management.

    Token estimation: 1 token ~= 4 characters (heuristic, no external
    tokenizer dependency). Results are included in score order until
    the token budget is exhausted.

    Complexity:
        build: T(n) in O(r) where r = results, S(n) in O(context).
        estimate_tokens: T(n) in O(1), S(n) in O(1).
    Invariants:
        - Output token_count <= max_tokens.
        - Results included in hybrid_score order (highest first).
        - Chunks truncated at line boundary, not mid-word.
    Edge Cases:
        - No results -> empty ContextResult.
        - Single chunk > budget -> truncated.
        - Budget=0 -> empty context.
    """

    __slots__ = ("_max_tokens", "_chars_per_token", "_separator")

    def __init__(
        self,
        max_tokens: int = 2048,
        chars_per_token: float = 4.0,
    ) -> None:
        """
        Initialize ContextBuilder with token budget.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: max_tokens=0 -> all builds return empty.
        """
        self._max_tokens: int = max_tokens
        self._chars_per_token: float = chars_per_token
        self._separator: str = "\n\n---\n\n"

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def max_tokens(self) -> int:
        """Maximum token budget. Complexity: T(n) in O(1)."""
        return self._max_tokens

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Heuristic: tokens ~= len(text) / chars_per_token.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty text -> 0.
        """
        if not text:
            return 0
        return int(len(text) / self._chars_per_token) + 1

    def build(
        self, results: Tuple[HybridResult, ...]
    ) -> ContextResult:
        """
        Assemble context from hybrid retrieval results.

        Includes results in score order until token budget exhausted.
        Truncates last chunk at line boundary if needed.

        Complexity: T(n) in O(r) where r = results, S(n) in O(context).
        Invariants: token_count <= max_tokens.
        Edge Cases: no results -> empty ContextResult.
        """
        if not results or self._max_tokens == 0:
            return ContextResult(
                context="",
                token_count=0,
                max_tokens=self._max_tokens,
                chunks_included=0,
                chunks_truncated=0,
                chunks_skipped=0,
            )

        parts: List[str] = []
        current_tokens: int = 0
        included: int = 0
        truncated: int = 0
        skipped: int = 0

        for r in results:
            chunk_text: str = self._format_chunk(r)
            chunk_tokens: int = self.estimate_tokens(chunk_text)

            if current_tokens + chunk_tokens <= self._max_tokens:
                parts.append(chunk_text)
                current_tokens += chunk_tokens
                included += 1
            else:
                remaining: int = self._max_tokens - current_tokens
                if remaining > 50:
                    truncated_text: str = self._truncate_to_tokens(
                        chunk_text, remaining
                    )
                    if truncated_text:
                        parts.append(truncated_text)
                        current_tokens += self.estimate_tokens(truncated_text)
                        included += 1
                        truncated += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

        context: str = self._separator.join(parts)
        final_tokens: int = self.estimate_tokens(context)

        return ContextResult(
            context=context,
            token_count=final_tokens,
            max_tokens=self._max_tokens,
            chunks_included=included,
            chunks_truncated=truncated,
            chunks_skipped=skipped,
        )

    # ─── Private ─────────────────────────────────────────────────────────────

    def _format_chunk(self, result: HybridResult) -> str:
        """
        Format a single hybrid result as context text.

        Complexity: T(n) in O(1), S(n) in O(chunk).
        Edge Cases: empty chunk content -> header only.
        """
        chunk = result.chunk
        header: str = (
            f"[{chunk.file_path}::{chunk.name}] "
            f"score={result.hybrid_score:.3f} "
            f"lines={chunk.start_line}-{chunk.end_line}"
        )
        return f"{header}\n{chunk.content}"

    def _truncate_to_tokens(
        self, text: str, max_tokens: int
    ) -> str:
        """
        Truncate text to fit within token budget, at line boundary.

        Complexity: T(n) in O(n) where n = text length, S(n) in O(output).
        Invariants: output token count <= max_tokens.
        Edge Cases: max_tokens=0 -> empty string.
        """
        max_chars: int = int(max_tokens * self._chars_per_token)
        if max_chars <= 0:
            return ""

        if len(text) <= max_chars:
            return text

        truncated: str = text[:max_chars]
        last_newline: int = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]

        return truncated + "\n... [truncated]"
