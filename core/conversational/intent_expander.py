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
Semantic Intent Expander — predicts intent chains and expands user intent
into sub-intents for deeper understanding.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class ExpandedIntent:
    """
    Immutable expanded intent with sub-intents.

    Attributes:
        primary: Primary intent label.
        sub_intents: Predicted sub-intent chain.
        confidence: Confidence in [0.0, 1.0].
        expansion_depth: How many levels were expanded.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    primary: str
    sub_intents: tuple[str, ...]
    confidence: float
    expansion_depth: int


# Intent expansion map: primary intent -> likely sub-intents.
_EXPANSION_MAP: Dict[str, List[str]] = {
    "code": ["analyze", "implement", "test", "optimize"],
    "explain": ["define", "contextualize", "exemplify", "summarize"],
    "question": ["clarify", "search", "verify", "summarize"],
    "fix": ["diagnose", "locate", "patch", "verify"],
    "improve": ["analyze", "plan", "implement", "test"],
    "diagnostic": ["inspect", "trace", "identify", "recommend"],
    "conversation": ["acknowledge", "respond", "elaborate"],
    "confirm": ["verify", "validate", "report"],
}

# Intent detection patterns.
_INTENT_PATTERNS: Dict[str, re.Pattern[str]] = {
    "code": re.compile(r"\b(code|function|class|method|implement|program|debug|compile)\b", re.IGNORECASE),
    "explain": re.compile(r"\b(explain|describe|what|how|why|elaborate)\b", re.IGNORECASE),
    "question": re.compile(r"\b(what|who|when|where|which|\?)\b", re.IGNORECASE),
    "fix": re.compile(r"\b(fix|repair|correct|broken|error|bug|crash)\b", re.IGNORECASE),
    "improve": re.compile(r"\b(improve|enhance|optimize|refactor|better)\b", re.IGNORECASE),
    "diagnostic": re.compile(r"\b(diagnose|analyze|inspect|troubleshoot|debug)\b", re.IGNORECASE),
    "conversation": re.compile(r"\b(hello|hi|thanks|please|okay|sure)\b", re.IGNORECASE),
    "confirm": re.compile(r"\b(confirm|verify|check|validate|test)\b", re.IGNORECASE),
}


class SemanticIntentExpander:
    """
    Expands user intent into sub-intent chains.

    Complexity:
        expand: T(n) ∈ O(p * n + d) for p patterns, n text length, d depth.
        detect: T(n) ∈ O(p * n) for p patterns, n text length.
    """

    __slots__ = ("_expansion_map", "_intent_patterns", "_max_depth")

    def __init__(
        self,
        expansion_map: Optional[Dict[str, List[str]]] = None,
        max_depth: int = 3,
    ) -> None:
        """Initialize the intent expander.

        Args:
            expansion_map: Custom intent expansion mapping.
            max_depth: Maximum expansion depth.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._expansion_map = expansion_map if expansion_map is not None else _EXPANSION_MAP
        self._intent_patterns = _INTENT_PATTERNS
        self._max_depth = max_depth

    def detect(self, text: str) -> Tuple[str, float]:
        """Detect the primary intent from text.

        Args:
            text: Input text.

        Returns:
            Tuple of (intent, confidence).

        Complexity: T(n) ∈ O(p * n) for p patterns, n text length.
        """
        if not text:
            return "unknown", 0.0

        # Check for question mark first — high priority for question intent
        if "?" in text:
            for intent, pattern in self._intent_patterns.items():
                if intent == "question":
                    matches = pattern.findall(text)
                    if matches:
                        return "question", min(1.0, len(matches) / 3.0)

        best_intent = "unknown"
        best_confidence = 0.0

        for intent, pattern in self._intent_patterns.items():
            matches = pattern.findall(text)
            if matches:
                confidence = min(1.0, len(matches) / 5.0)
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence

        return best_intent, best_confidence

    def expand(self, text: str, intent: Optional[str] = None) -> ExpandedIntent:
        """Expand an intent into sub-intents.

        Args:
            text: Input text.
            intent: Optional pre-classified intent; auto-detected if None.

        Returns:
            ExpandedIntent with sub-intent chain.

        Complexity: T(n) ∈ O(p * n + d) for p patterns, n text, d depth.
        """
        if intent is None or intent == "unknown":
            intent, confidence = self.detect(text)
        else:
            confidence = 0.8

        if intent == "unknown":
            return ExpandedIntent(
                primary="unknown",
                sub_intents=(),
                confidence=0.0,
                expansion_depth=0,
            )

        # Expand sub-intents
        sub_intents: List[str] = []
        current = intent
        for _ in range(self._max_depth):
            expansions = self._expansion_map.get(current, [])
            if not expansions:
                break
            # Pick the first unvisited sub-intent
            for sub in expansions:
                if sub not in sub_intents:
                    sub_intents.append(sub)
                    current = sub
                    break
            else:
                break

        return ExpandedIntent(
            primary=intent,
            sub_intents=tuple(sub_intents),
            confidence=confidence,
            expansion_depth=len(sub_intents),
        )

    def expand_chain(self, intent: str) -> List[str]:
        """Get the full expansion chain for an intent.

        Args:
            intent: Primary intent.

        Returns:
            List of sub-intents in order.

        Complexity: T(n) ∈ O(d * s) for d depth, s sub-intents per level.
        """
        chain: List[str] = []
        visited: set[str] = {intent}
        queue: List[str] = [intent]
        while queue and len(chain) < self._max_depth * 4:
            current = queue.pop(0)
            expansions = self._expansion_map.get(current, [])
            for sub in expansions:
                if sub not in visited:
                    chain.append(sub)
                    visited.add(sub)
                    queue.append(sub)
        return chain

    def register_expansion(self, intent: str, sub_intents: List[str]) -> None:
        """Register a custom intent expansion.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._expansion_map[intent] = sub_intents
