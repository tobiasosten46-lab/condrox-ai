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
Semantic Interrupt Handler — detects when a user's new input semantically
interrupts an ongoing response and handles the transition gracefully.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class InterruptType(Enum):
    """Types of semantic interrupts."""

    NONE = "none"
    TOPIC_CHANGE = "topic_change"
    CORRECTION = "correction"
    URGENCY = "urgency"
    ABORT = "abort"
    QUESTION = "question"


@dataclass(frozen=True, slots=True)
class InterruptResult:
    """
    Immutable interrupt detection result.

    Attributes:
        interrupt_type: Type of interrupt detected.
        is_interrupt: Whether an interrupt was detected.
        confidence: Detection confidence in [0.0, 1.0].
        should_abort: Whether the current response should be aborted.
        handoff_message: Optional message for graceful transition.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    interrupt_type: InterruptType
    is_interrupt: bool
    confidence: float
    should_abort: bool
    handoff_message: str


class SemanticInterruptHandler:
    """
    Detects semantic interrupts in user input.

    Complexity:
        detect: T(n) ∈ O(p * n) for p patterns, n text length.
    """

    __slots__ = ("_urgency_patterns", "_correction_patterns", "_abort_patterns", "_question_patterns", "_stopwords")

    def __init__(self) -> None:
        """Initialize the interrupt handler.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._urgency_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(stop|wait|hold on|hold up)\b", re.IGNORECASE),
            re.compile(r"\b(urgent|emergency|asap|immediately)\b", re.IGNORECASE),
            re.compile(r"\b(quick|now|right now)\b", re.IGNORECASE),
        ]
        self._correction_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(no wait|actually|sorry i meant|correction)\b", re.IGNORECASE),
            re.compile(r"\b(i meant|that's wrong|not that|wrong)\b", re.IGNORECASE),
            re.compile(r"\b(let me rephrase|disregard|ignore that)\b", re.IGNORECASE),
        ]
        self._abort_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(cancel|never mind|forget it|abort)\b", re.IGNORECASE),
            re.compile(r"\b(nevermind|nvm|stop everything)\b", re.IGNORECASE),
        ]
        self._question_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(what|why|how|when|where)\b.*\?", re.IGNORECASE),
            re.compile(r"\?\s*$"),
        ]
        self._stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "and", "but", "or", "if", "of", "at",
            "by", "for", "with", "to", "from", "in", "on", "not", "no",
        }

    def detect(self, new_input: str, current_topic: Optional[str] = None) -> InterruptResult:
        """Detect if the new input is a semantic interrupt.

        Args:
            new_input: The new user input.
            current_topic: Optional current conversation topic for topic change detection.

        Returns:
            InterruptResult with detection outcome.

        Complexity: T(n) ∈ O(p * n) for p patterns, n text length.
        """
        if not new_input or not new_input.strip():
            return InterruptResult(
                interrupt_type=InterruptType.NONE,
                is_interrupt=False,
                confidence=0.0,
                should_abort=False,
                handoff_message="",
            )

        # Check abort patterns first (highest priority)
        for pattern in self._abort_patterns:
            if pattern.search(new_input):
                return InterruptResult(
                    interrupt_type=InterruptType.ABORT,
                    is_interrupt=True,
                    confidence=0.95,
                    should_abort=True,
                    handoff_message="Aborting current response as requested.",
                )

        # Check urgency patterns
        for pattern in self._urgency_patterns:
            if pattern.search(new_input):
                return InterruptResult(
                    interrupt_type=InterruptType.URGENCY,
                    is_interrupt=True,
                    confidence=0.85,
                    should_abort=True,
                    handoff_message="Pausing current response for urgent input.",
                )

        # Check correction patterns
        for pattern in self._correction_patterns:
            if pattern.search(new_input):
                return InterruptResult(
                    interrupt_type=InterruptType.CORRECTION,
                    is_interrupt=True,
                    confidence=0.8,
                    should_abort=False,
                    handoff_message="Adjusting based on your correction.",
                )

        # Check question patterns
        for pattern in self._question_patterns:
            if pattern.search(new_input):
                return InterruptResult(
                    interrupt_type=InterruptType.QUESTION,
                    is_interrupt=True,
                    confidence=0.6,
                    should_abort=False,
                    handoff_message="Addressing your question.",
                )

        # Check topic change
        if current_topic:
            topic_overlap = self._compute_overlap(new_input, current_topic)
            if topic_overlap < 0.15:
                return InterruptResult(
                    interrupt_type=InterruptType.TOPIC_CHANGE,
                    is_interrupt=True,
                    confidence=0.5,
                    should_abort=False,
                    handoff_message="Transitioning to a new topic.",
                )

        return InterruptResult(
            interrupt_type=InterruptType.NONE,
            is_interrupt=False,
            confidence=0.0,
            should_abort=False,
            handoff_message="",
        )

    def _compute_overlap(self, text_a: str, text_b: str) -> float:
        """Compute word overlap between two texts.

        Complexity: T(n) ∈ O(n + m) for n, m text lengths.
        """
        words_a = self._extract_words(text_a)
        words_b = self._extract_words(text_b)

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _extract_words(self, text: str) -> Set[str]:
        """Extract meaningful words from text.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return {w for w in words if w not in self._stopwords}
