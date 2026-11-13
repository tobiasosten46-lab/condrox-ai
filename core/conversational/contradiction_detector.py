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
Contradiction Detector — detects contradictions between a new response and
prior knowledge or statements.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class ContradictionFinding:
    """
    Immutable contradiction finding.

    Attributes:
        new_statement: The new statement that contradicts.
        prior_statement: The prior statement it contradicts.
        contradiction_type: Type of contradiction.
        confidence: Detection confidence in [0.0, 1.0].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    new_statement: str
    prior_statement: str
    contradiction_type: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ContradictionReport:
    """
    Immutable contradiction detection report.

    Attributes:
        has_contradictions: Whether any contradictions were found.
        findings: List of contradiction findings.
        total_checked: Number of prior statements checked.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    has_contradictions: bool
    findings: tuple[ContradictionFinding, ...]
    total_checked: int


class ContradictionDetector:
    """
    Detects contradictions between new responses and prior statements.

    Uses three detection strategies:
    1. Direct negation: "X is true" vs "X is false"
    2. Quantifier flip: "always X" vs "never X"
    3. Numeric conflict: "value is 5" vs "value is 10"

    Complexity:
        check: T(n) ∈ O(s * n) for s prior statements, n new text.
        add_fact: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_facts", "_negation_pairs", "_quantifier_pairs", "_numeric_pattern", "_stopwords")

    def __init__(self) -> None:
        """Initialize the contradiction detector.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._facts: List[Tuple[str, str]] = []  # (text, source)
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
            ("possible", "impossible"),
            ("present", "absent"),
            ("exists", "does not exist"),
            ("supported", "unsupported"),
        ]
        self._quantifier_pairs: List[Tuple[str, str]] = [
            ("always", "never"),
            ("all", "no"),
            ("all", "none"),
            ("every", "no"),
            ("increases", "decreases"),
            ("higher", "lower"),
            ("more", "less"),
            ("faster", "slower"),
            ("larger", "smaller"),
            ("stronger", "weaker"),
        ]
        self._numeric_pattern = re.compile(
            r"\b(?:value|result|count|size|length|total|score|number)\s+(?:is|equals?|=)\s+(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        self._stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "and", "but", "or", "if",
            "of", "at", "by", "for", "with", "to", "from", "in", "on",
            "not", "no", "equals",
        }

    def add_fact(self, text: str, source: str = "system") -> None:
        """Record a fact for future contradiction checks.

        Args:
            text: The fact text.
            source: Source identifier.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._facts.append((text.lower().strip(), source))

    def check(self, new_statement: str) -> ContradictionReport:
        """Check a new statement for contradictions against known facts.

        Args:
            new_statement: The statement to check.

        Returns:
            ContradictionReport with findings.

        Complexity: T(n) ∈ O(s * n) for s facts, n statement length.
        """
        if not new_statement or not self._facts:
            return ContradictionReport(
                has_contradictions=False,
                findings=(),
                total_checked=0,
            )

        new_lower = new_statement.lower().strip()
        findings: List[ContradictionFinding] = []

        for fact_text, fact_source in self._facts:
            # Strategy 1: Negation pairs
            finding = self._check_negation(fact_text, new_lower, fact_source)
            if finding:
                findings.append(finding)
                continue

            # Strategy 2: Quantifier flips
            finding = self._check_quantifiers(fact_text, new_lower, fact_source)
            if finding:
                findings.append(finding)
                continue

            # Strategy 3: Numeric conflicts
            finding = self._check_numeric(fact_text, new_lower, fact_source)
            if finding:
                findings.append(finding)

        return ContradictionReport(
            has_contradictions=len(findings) > 0,
            findings=tuple(findings),
            total_checked=len(self._facts),
        )

    def fact_count(self) -> int:
        """Return the number of recorded facts.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._facts)

    def clear_facts(self) -> None:
        """Clear all recorded facts.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._facts.clear()

    def _check_negation(
        self,
        fact: str,
        new: str,
        source: str,
    ) -> Optional[ContradictionFinding]:
        """Check for negation-based contradictions.

        Complexity: T(n) ∈ O(p * n) for p pairs, n text length.
        """
        for positive, negative in self._negation_pairs:
            if self._has_phrase(fact, positive) and self._has_phrase(new, negative):
                if self._same_context(fact, new, positive, negative):
                    return ContradictionFinding(
                        new_statement=new[:100],
                        prior_statement=fact[:100],
                        contradiction_type="negation",
                        confidence=0.7,
                    )
            if self._has_phrase(fact, negative) and self._has_phrase(new, positive):
                if self._same_context(fact, new, negative, positive):
                    return ContradictionFinding(
                        new_statement=new[:100],
                        prior_statement=fact[:100],
                        contradiction_type="negation",
                        confidence=0.7,
                    )
        return None

    def _check_quantifiers(
        self,
        fact: str,
        new: str,
        source: str,
    ) -> Optional[ContradictionFinding]:
        """Check for quantifier-based contradictions.

        Complexity: T(n) ∈ O(p * n) for p pairs, n text length.
        """
        for q_a, q_b in self._quantifier_pairs:
            if self._has_phrase(fact, q_a) and self._has_phrase(new, q_b):
                if self._same_context(fact, new, q_a, q_b):
                    return ContradictionFinding(
                        new_statement=new[:100],
                        prior_statement=fact[:100],
                        contradiction_type="quantifier_flip",
                        confidence=0.8,
                    )
            if self._has_phrase(fact, q_b) and self._has_phrase(new, q_a):
                if self._same_context(fact, new, q_b, q_a):
                    return ContradictionFinding(
                        new_statement=new[:100],
                        prior_statement=fact[:100],
                        contradiction_type="quantifier_flip",
                        confidence=0.8,
                    )
        return None

    def _check_numeric(
        self,
        fact: str,
        new: str,
        source: str,
    ) -> Optional[ContradictionFinding]:
        """Check for numeric value contradictions.

        Complexity: T(n) ∈ O(n) for n text length.
        """
        fact_matches = self._numeric_pattern.findall(fact)
        new_matches = self._numeric_pattern.findall(new)

        if fact_matches and new_matches:
            # Check if any fact value differs from any new value
            fact_values = set(fact_matches)
            new_values = set(new_matches)
            if fact_values and new_values and fact_values.isdisjoint(new_values):
                # Check if they share context words (excluding stopwords)
                fact_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", fact)) - self._stopwords
                new_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", new)) - self._stopwords
                overlap = fact_words & new_words
                if len(overlap) >= 1:
                    return ContradictionFinding(
                        new_statement=new[:100],
                        prior_statement=fact[:100],
                        contradiction_type="numeric_conflict",
                        confidence=0.9,
                    )
        return None

    def _has_phrase(self, text: str, phrase: str) -> bool:
        """Check if text contains a phrase as whole words.

        Complexity: T(n) ∈ O(n) for n text length.
        """
        pattern = r"\b" + re.escape(phrase) + r"\b"
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _same_context(self, text_a: str, text_b: str, word_a: str, word_b: str) -> bool:
        """Check if two texts share context words around the key terms.

        Complexity: T(n) ∈ O(n) for n words.
        """
        ctx_a = self._context_words(text_a, word_a)
        ctx_b = self._context_words(text_b, word_b)
        ctx_a.discard(word_a)
        ctx_a.discard(word_b)
        ctx_b.discard(word_a)
        ctx_b.discard(word_b)
        if not ctx_a or not ctx_b:
            return True
        return len(ctx_a & ctx_b) > 0

    def _context_words(self, text: str, word: str, radius: int = 3) -> Set[str]:
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
                    clean = re.sub(r"[^\w]", "", words[j])
                    if clean and len(clean) >= 3 and clean not in self._stopwords:
                        result.add(clean)
        return result
