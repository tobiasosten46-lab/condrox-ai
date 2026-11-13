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
Consistency Engine — checks response consistency against prior turns and
memory to prevent contradictions.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """
    Immutable consistency check report.

    Attributes:
        consistent: Whether the response is consistent.
        contradictions: List of contradiction descriptions.
        confidence: Confidence in the consistency assessment [0, 1].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    consistent: bool
    contradictions: tuple[str, ...]
    confidence: float


class ConsistencyEngine:
    """
    Checks response consistency against prior statements and memory.

    Complexity:
        check: T(n) ∈ O(s * n) for s statements, n response length.
        add_statement: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_statements", "_negation_pairs", "_contradiction_log", "_stopwords")

    def __init__(self) -> None:
        """Initialize the consistency engine.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._statements: List[Tuple[str, str]] = []  # (text, source)
        self._negation_pairs: List[Tuple[str, str]] = [
            ("is", "is not"),
            ("is", "isn't"),
            ("can", "cannot"),
            ("can", "can't"),
            ("will", "will not"),
            ("will", "won't"),
            ("should", "should not"),
            ("should", "shouldn't"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("valid", "invalid"),
            ("yes", "no"),
            ("always", "never"),
            ("increases", "decreases"),
            ("higher", "lower"),
            ("more", "less"),
            ("present", "absent"),
        ]
        self._contradiction_log: List[str] = []
        self._stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "and", "but", "or", "if", "of", "at",
            "by", "for", "with", "to", "from", "in", "on", "not", "no",
        }

    def add_statement(self, text: str, source: str = "system") -> None:
        """Record a statement for future consistency checks.

        Args:
            text: The statement text.
            source: Who made the statement.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._statements.append((text.lower().strip(), source))

    def check(self, response: str) -> ConsistencyReport:
        """Check a response for consistency against recorded statements.

        Args:
            response: The response to check.

        Returns:
            ConsistencyReport with findings.

        Complexity: T(n) ∈ O(s * n) for s statements, n response length.
        """
        if not response or not self._statements:
            return ConsistencyReport(
                consistent=True,
                contradictions=(),
                confidence=1.0,
            )

        response_lower = response.lower().strip()
        contradictions: List[str] = []

        for statement, source in self._statements:
            contradiction = self._check_pair(statement, response_lower)
            if contradiction:
                contradictions.append(
                    f"Contradicts prior statement from {source}: {contradiction}"
                )

        consistent = len(contradictions) == 0
        confidence = 1.0 - (len(contradictions) * 0.2)
        confidence = max(0.0, min(1.0, confidence))

        if not consistent:
            self._contradiction_log.extend(contradictions)

        return ConsistencyReport(
            consistent=consistent,
            contradictions=tuple(contradictions),
            confidence=confidence,
        )

    def get_contradiction_log(self, limit: int = 50) -> List[str]:
        """Return recent contradiction log entries.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._contradiction_log[-limit:])

    def clear_contradiction_log(self) -> None:
        """Clear the contradiction log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._contradiction_log.clear()

    def clear_statements(self) -> None:
        """Clear all recorded statements.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._statements.clear()

    def statement_count(self) -> int:
        """Return the number of recorded statements.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._statements)

    def _check_pair(self, statement: str, response: str) -> Optional[str]:
        """Check if a response contradicts a prior statement.

        Uses negation pair detection: if the statement asserts X and the
        response asserts not-X (or vice versa), it's a contradiction.

        Complexity: T(n) ∈ O(p * n) for p negation pairs, n text length.
        """
        for positive, negative in self._negation_pairs:
            # Check if statement has positive and response has negative
            if self._contains_phrase(statement, positive) and self._contains_phrase(response, negative):
                # Verify they're talking about the same subject
                if self._same_subject(statement, response, positive, negative):
                    return f"'{positive}' vs '{negative}'"
            # Check reverse
            if self._contains_phrase(statement, negative) and self._contains_phrase(response, positive):
                if self._same_subject(statement, response, negative, positive):
                    return f"'{negative}' vs '{positive}'"

        return None

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        """Check if text contains a phrase as a whole word.

        Complexity: T(n) ∈ O(n) for n text length.
        """
        pattern = r"\b" + re.escape(phrase) + r"\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _same_subject(self, text_a: str, text_b: str, word_a: str, word_b: str) -> bool:
        """Heuristic: check if two texts share significant words around the
        negation pair, suggesting the same subject.

        Complexity: T(n) ∈ O(n) for n text length.
        """
        # Get words within 3 positions of the key word in each text
        context_a = self._get_context_words(text_a, word_a, radius=3)
        context_b = self._get_context_words(text_b, word_b, radius=3)

        if not context_a or not context_b:
            return True  # Can't determine, assume same subject

        overlap = context_a & context_b
        # Require at least 1 shared context word (excluding the negation words)
        overlap.discard(word_a)
        overlap.discard(word_b)
        return len(overlap) > 0

    def _get_context_words(self, text: str, word: str, radius: int = 3) -> Set[str]:
        """Get words within a radius of the target word, excluding stopwords.

        Complexity: T(n) ∈ O(n) for n words.
        """
        words = text.split()
        result: Set[str] = set()
        for i, w in enumerate(words):
            if word in w:
                start = max(0, i - radius)
                end = min(len(words), i + radius + 1)
                for j in range(start, end):
                    # Strip punctuation
                    clean = re.sub(r"[^\w]", "", words[j])
                    if clean and len(clean) >= 3 and clean not in self._stopwords:
                        result.add(clean)
        return result
