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
ClaimRanker v2 — multi-vector scoring and contradiction detection for
ranking claims by strength and relevance.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Claim:
    """
    Immutable claim with multi-vector scoring.

    Attributes:
        text: Claim text.
        source: Source identifier.
        evidence_score: Evidence strength in [0.0, 1.0].
        recency_score: Recency in [0.0, 1.0].
        relevance_score: Query relevance in [0.0, 1.0].
        confidence_score: Confidence in [0.0, 1.0].
        contradicts: List of claim IDs this claim contradicts.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    text: str
    source: str
    evidence_score: float
    recency_score: float
    relevance_score: float
    confidence_score: float
    contradicts: tuple[str, ...] = ()

    @property
    def composite_score(self) -> float:
        """Weighted composite score.

        Weights: evidence 0.35, relevance 0.30, confidence 0.20, recency 0.15

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return (
            0.35 * self.evidence_score
            + 0.30 * self.relevance_score
            + 0.20 * self.confidence_score
            + 0.15 * self.recency_score
        )


@dataclass(frozen=True, slots=True)
class RankedClaim:
    """
    Immutable ranked claim with final score and rank.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    claim: Claim
    rank: int
    final_score: float
    contradiction_flagged: bool


class ClaimRankerV2:
    """
    Ranks claims by multi-vector scoring and flags contradictions.

    Complexity:
        rank: T(n) ∈ O(n log n), S(n) ∈ O(n) for n claims.
        detect_contradictions: T(n) ∈ O(n²), S(n) ∈ O(n).
    """

    __slots__ = ("_contradiction_markers", "_hedge_words")

    def __init__(self) -> None:
        """Initialize the ranker with contradiction and hedge markers.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Markers that indicate a claim contradicts another.
        self._contradiction_markers: List[str] = [
            "however",
            "but",
            "contradicts",
            "disputes",
            "refutes",
            "incorrect",
            "wrong",
            "false",
            "not true",
            "on the contrary",
            "conversely",
            "in contrast",
        ]
        # Hedge words that lower confidence.
        self._hedge_words: List[str] = [
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could",
            "seems",
            "appears",
            "likely",
            "probably",
            "uncertain",
            "unclear",
            "i think",
            "i believe",
            "arguably",
        ]

    def score_claim(
        self,
        text: str,
        source: str,
        query: str,
        evidence_count: int = 0,
        age_seconds: float = 0.0,
    ) -> Claim:
        """Score a single claim.

        Args:
            text: Claim text.
            source: Source identifier.
            query: The query for relevance scoring.
            evidence_count: Number of supporting evidence items.
            age_seconds: Age of the claim in seconds.

        Returns:
            Scored Claim.

        Complexity: T(n) ∈ O(n) for n words, S(n) ∈ O(1).
        """
        # Evidence score: logarithmic scaling
        evidence_score = min(1.0, math.log1p(evidence_count) / math.log1p(10))

        # Recency score: exponential decay with 7-day half-life
        halflife = 7 * 86400.0
        recency_score = 0.5 ** (age_seconds / halflife) if halflife > 0 else 1.0

        # Relevance score: word overlap with query
        relevance_score = self._word_overlap(text, query)

        # Confidence score: reduced by hedge words
        confidence_score = 1.0
        lower_text = text.lower()
        hedge_count = sum(1 for h in self._hedge_words if h in lower_text)
        confidence_score = max(0.0, 1.0 - 0.15 * hedge_count)

        return Claim(
            text=text,
            source=source,
            evidence_score=evidence_score,
            recency_score=recency_score,
            relevance_score=relevance_score,
            confidence_score=confidence_score,
        )

    def rank(self, claims: List[Claim]) -> List[RankedClaim]:
        """Rank claims by composite score, flagging contradictions.

        Args:
            claims: List of scored claims.

        Returns:
            List of RankedClaim sorted by descending score.

        Complexity: T(n) ∈ O(n log n + n²), S(n) ∈ O(n).
        """
        if not claims:
            return []

        # Detect contradictions
        contradiction_map = self.detect_contradictions(claims)

        # Sort by composite score descending
        sorted_claims = sorted(claims, key=lambda c: c.composite_score, reverse=True)

        result: List[RankedClaim] = []
        for rank, claim in enumerate(sorted_claims):
            is_flagged = claim.text in contradiction_map or any(
                claim.text in contradiction_map.get(other.text, [])
                for other in claims
                if other.text != claim.text
            )
            result.append(
                RankedClaim(
                    claim=claim,
                    rank=rank + 1,
                    final_score=claim.composite_score,
                    contradiction_flagged=is_flagged,
                )
            )

        return result

    def detect_contradictions(self, claims: List[Claim]) -> Dict[str, List[str]]:
        """Detect contradictions between claims.

        Args:
            claims: List of claims to check.

        Returns:
            Mapping from claim text to list of contradicted claim texts.

        Complexity: T(n) ∈ O(n² * m) for n claims, m markers.
        """
        contradictions: Dict[str, List[str]] = {}
        n = len(claims)

        for i in range(n):
            for j in range(i + 1, n):
                claim_a = claims[i]
                claim_b = claims[j]
                if self._are_contradictory(claim_a.text, claim_b.text):
                    contradictions.setdefault(claim_a.text, []).append(claim_b.text)
                    contradictions.setdefault(claim_b.text, []).append(claim_a.text)

        return contradictions

    def _are_contradictory(self, text_a: str, text_b: str) -> bool:
        """Check if two texts are contradictory.

        Complexity: T(n) ∈ O(m * n) for m markers, n text length.
        """
        # Check if either text contains a contradiction marker
        # AND has significant word overlap with the other
        lower_a = text_a.lower()
        lower_b = text_b.lower()

        has_marker_a = any(m in lower_a for m in self._contradiction_markers)
        has_marker_b = any(m in lower_b for m in self._contradiction_markers)

        if not (has_marker_a or has_marker_b):
            return False

        overlap = self._word_overlap(text_a, text_b)
        return overlap > 0.3  # Significant overlap + contradiction marker

    def _word_overlap(self, text_a: str, text_b: str) -> float:
        """Compute word overlap ratio between two texts.

        Returns:
            Overlap score in [0.0, 1.0].

        Complexity: T(n) ∈ O(n + m) for n, m words, S(n) ∈ O(n + m).
        """
        words_a = set(self._tokenize(text_a))
        words_b = set(self._tokenize(text_b))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase word tokens.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return re.findall(r"\b[a-z]{2,}\b", text.lower())
