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
Semantic Repair — detects and repairs semantic drift in responses by
re-anchoring to the conversation's primary topic.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class RepairResult:
    """
    Immutable result of a semantic repair operation.

    Attributes:
        repaired_text: The repaired text.
        drift_detected: Whether semantic drift was detected.
        drift_score: Drift severity in [0.0, 1.0].
        repairs_applied: List of repair descriptions.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    repaired_text: str
    drift_detected: bool
    drift_score: float
    repairs_applied: tuple[str, ...]


class SemanticRepair:
    """
    Detects and repairs semantic drift in responses.

    Semantic drift occurs when a response wanders off-topic relative to
    the user's query. This component measures word overlap between the
    query and response, and if drift is detected, it re-anchors the
    response by trimming off-topic trailing content.

    Complexity:
        check_drift: T(n) ∈ O(q + r) for q query words, r response words.
        repair: T(n) ∈ O(r * s) for r response sentences, s query words.
    """

    __slots__ = ("_drift_threshold", "_min_overlap", "_stopwords")

    def __init__(
        self,
        drift_threshold: float = 0.15,
        min_overlap: float = 0.1,
    ) -> None:
        """Initialize the semantic repair engine.

        Args:
            drift_threshold: Below this overlap, drift is detected.
            min_overlap: Minimum overlap to consider a sentence on-topic.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._drift_threshold = drift_threshold
        self._min_overlap = min_overlap
        self._stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "what", "which", "who", "when", "where",
            "why", "how", "all", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "and", "but", "or",
            "if", "because", "as", "until", "while", "of", "at", "by",
            "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off",
            "over", "under", "again", "further", "then", "once",
        }

    def check_drift(self, query: str, response: str) -> float:
        """Measure semantic drift between query and response.

        Args:
            query: The user's query.
            response: The agent's response.

        Returns:
            Drift score in [0.0, 1.0] where higher = more drift.

        Complexity: T(n) ∈ O(q + r) for q query words, r response words.
        """
        query_words = self._extract_words(query)
        response_words = self._extract_words(response)

        if not query_words or not response_words:
            return 0.0

        overlap = len(query_words & response_words) / len(query_words | response_words)
        # Drift is the inverse of overlap
        return 1.0 - overlap

    def repair(self, query: str, response: str) -> RepairResult:
        """Repair semantic drift in a response.

        Args:
            query: The user's query.
            response: The agent's response.

        Returns:
            RepairResult with the repaired text and drift info.

        Complexity: T(n) ∈ O(r * s) for r response sentences, s query words.
        """
        if not response or not query:
            return RepairResult(
                repaired_text=response,
                drift_detected=False,
                drift_score=0.0,
                repairs_applied=(),
            )

        drift_score = self.check_drift(query, response)
        # drift_score is (1 - overlap). drift_detected when overlap < drift_threshold.
        overlap = 1.0 - drift_score
        drift_detected = overlap < self._drift_threshold

        if not drift_detected:
            return RepairResult(
                repaired_text=response,
                drift_detected=False,
                drift_score=drift_score,
                repairs_applied=(),
            )

        repairs: List[str] = []
        query_words = self._extract_words(query)
        sentences = self._split_sentences(response)

        if not sentences:
            return RepairResult(
                repaired_text=response,
                drift_detected=True,
                drift_score=drift_score,
                repairs_applied=("drift_detected_no_repair",),
            )

        # Score each sentence by overlap with query
        scored: List[Tuple[float, int, str]] = []
        for i, sentence in enumerate(sentences):
            sent_words = self._extract_words(sentence)
            if not sent_words:
                scored.append((0.0, i, sentence))
                continue
            overlap = len(query_words & sent_words) / len(query_words | sent_words)
            scored.append((overlap, i, sentence))

        # Keep sentences that meet the minimum overlap threshold
        on_topic = [(s, i, text) for s, i, text in scored if s >= self._min_overlap]
        off_topic = [(s, i, text) for s, i, text in scored if s < self._min_overlap]

        if not on_topic:
            # All sentences are off-topic; keep the first as a fallback
            on_topic = [scored[0]]
            repairs.append("all_sentences_off_topic_kept_first")

        if off_topic:
            repairs.append(f"removed_{len(off_topic)}_off_topic_sentences")

        # Rebuild response from on-topic sentences in original order
        on_topic.sort(key=lambda x: x[1])
        repaired_text = " ".join(text for _, _, text in on_topic)

        # Clean up whitespace
        repaired_text = re.sub(r"\s{2,}", " ", repaired_text).strip()

        return RepairResult(
            repaired_text=repaired_text,
            drift_detected=True,
            drift_score=drift_score,
            repairs_applied=tuple(repairs),
        )

    def _extract_words(self, text: str) -> Set[str]:
        """Extract meaningful words from text, excluding stopwords.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return {w for w in words if w not in self._stopwords}

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        # Simple sentence splitting on . ! ? followed by space or end
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
