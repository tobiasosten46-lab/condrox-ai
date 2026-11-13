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
Ethics engine for the unified Brain.

Performs real content policy checks using literal and fuzzy topic matching.
Outputs an immutable EthicsReport with a risk score and flagged topics.

Mathematical Foundation:
- Risk is the maximum topic match score: max_i(m_i) where m_i is a match
  strength computed by keyword overlap between the topic phrase and text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass(frozen=True, slots=True)
class EthicsReport:
    """Immutable ethics check report.

    Attributes:
        approved: True if no blocked topics were flagged.
        risk_score: Aggregate risk in [0.0, 1.0].
        flagged_topics: List of flagged topic strings.
        checks: Additional metadata.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(f) for f flagged topics.
    """

    approved: bool
    risk_score: float
    flagged_topics: List[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Validate score range.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.risk_score < 0.0 or self.risk_score > 1.0:
            raise ValueError("risk_score must be in [0.0, 1.0]")


class EthicsEngine:
    """
    Real ethics/policy engine with keyword-based topic detection.

    Attributes:
        blocked_topics: List of blocked topic phrases.

    Complexity:
        check: O(t * n) for t topics and text length n.
    """

    __slots__ = ("blocked_topics",)

    def __init__(self, blocked_topics: Optional[List[str]] = None) -> None:
        """Initialize with a list of blocked topic phrases.

        Args:
            blocked_topics: Phrases that, when matched, increase risk.
        """
        self.blocked_topics: List[str] = [t.lower() for t in (blocked_topics or [])]

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        """Tokenize text into a set of normalized words.

        Args:
            text: Input string.

        Returns:
            Set of lowercase alphanumeric tokens.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _topic_match_score(text_tokens: Set[str], topic: str) -> float:
        """Compute Jaccard-like keyword overlap between text and a topic.

        Args:
            text_tokens: Set of text tokens.
            topic: Topic phrase.

        Returns:
            Match score in [0.0, 1.0].

        Complexity: T(n) ∈ O(t) where t is topic token count.
        """
        topic_tokens = set(re.findall(r"[a-z0-9]+", topic.lower()))
        if not topic_tokens:
            return 0.0
        intersection = text_tokens & topic_tokens
        union = text_tokens | topic_tokens
        if not union:
            return 0.0
        return float(len(intersection) / len(union))

    def check(self, text: str) -> EthicsReport:
        """Run ethics checks on text.

        Args:
            text: Input or output text.

        Returns:
            EthicsReport.

        Complexity: T(n) ∈ O(t * n) for t topics.
        """
        if not text:
            return EthicsReport(approved=True, risk_score=0.0)

        text_tokens = self._tokens(text)
        flagged: List[str] = []
        max_score = 0.0

        for topic in self.blocked_topics:
            score = self._topic_match_score(text_tokens, topic)
            if score >= 0.5:
                flagged.append(topic)
            max_score = max(max_score, score)

        risk_score = min(1.0, max_score)
        approved = risk_score < 0.5

        return EthicsReport(
            approved=bool(approved),
            risk_score=risk_score,
            flagged_topics=flagged,
            checks={"topic_count": len(self.blocked_topics), "text_token_count": len(text_tokens)},
        )
