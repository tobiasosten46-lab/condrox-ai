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
Multi-Turn Reasoning Engine — maintains turn history and uses it to
influence output with goal-based reasoning.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """
    Immutable record of a single conversation turn.

    Attributes:
        turn_id: Turn identifier.
        user_input: What the user said.
        agent_output: What the agent responded.
        intent: Classified intent.
        timestamp: When the turn occurred.
        metadata: Additional turn metadata.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    turn_id: int
    user_input: str
    agent_output: str
    intent: str
    timestamp: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    """
    Immutable reasoning context derived from turn history.

    Attributes:
        turn_count: Number of turns in history.
        dominant_intent: Most frequent intent.
        topic_continuity: How much the topic has stayed consistent [0, 1].
        suggested_focus: Suggested focus for the next response.
        history_summary: Condensed summary of recent turns.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    turn_count: int
    dominant_intent: str
    topic_continuity: float
    suggested_focus: str
    history_summary: str


class MultiTurnReasoningEngine:
    """
    Tracks conversation turns and derives reasoning context.

    Complexity:
        record_turn: T(n) ∈ O(1), S(n) ∈ O(1).
        get_context: T(n) ∈ O(t) for t turns, S(n) ∈ O(t).
    """

    __slots__ = ("_turns", "_max_history", "_turn_counter")

    def __init__(self, max_history: int = 50) -> None:
        """Initialize the reasoning engine.

        Args:
            max_history: Maximum turns to retain.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._turns: List[TurnRecord] = []
        self._max_history = max_history
        self._turn_counter: int = 0

    def record_turn(
        self,
        user_input: str,
        agent_output: str,
        intent: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> TurnRecord:
        """Record a conversation turn.

        Args:
            user_input: User's input text.
            agent_output: Agent's response.
            intent: Classified intent.
            metadata: Optional metadata.

        Returns:
            The recorded TurnRecord.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._turn_counter += 1
        turn = TurnRecord(
            turn_id=self._turn_counter,
            user_input=user_input,
            agent_output=agent_output,
            intent=intent,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._turns.append(turn)
        if len(self._turns) > self._max_history:
            self._turns = self._turns[-self._max_history:]
        return turn

    def get_context(self, window: int = 10) -> ReasoningContext:
        """Derive reasoning context from recent turns.

        Args:
            window: Number of recent turns to analyze.

        Returns:
            ReasoningContext with derived insights.

        Complexity: T(n) ∈ O(t) for t turns in window, S(n) ∈ O(t).
        """
        if not self._turns:
            return ReasoningContext(
                turn_count=0,
                dominant_intent="unknown",
                topic_continuity=0.0,
                suggested_focus="",
                history_summary="",
            )

        recent = self._turns[-window:]
        turn_count = len(self._turns)

        # Find dominant intent
        intent_counts: Dict[str, int] = {}
        for turn in recent:
            intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1
        dominant_intent = max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"

        # Compute topic continuity via word overlap between turns
        topic_continuity = self._compute_continuity(recent)

        # Suggest focus based on dominant intent
        suggested_focus = self._suggest_focus(dominant_intent, recent)

        # Build history summary
        history_summary = self._summarize(recent)

        return ReasoningContext(
            turn_count=turn_count,
            dominant_intent=dominant_intent,
            topic_continuity=topic_continuity,
            suggested_focus=suggested_focus,
            history_summary=history_summary,
        )

    def get_turns(self, limit: int = 10) -> List[TurnRecord]:
        """Return recent turns.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._turns[-limit:])

    def get_turn(self, turn_id: int) -> Optional[TurnRecord]:
        """Get a specific turn by ID.

        Complexity: T(n) ∈ O(t), S(n) ∈ O(1).
        """
        for turn in self._turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def clear(self) -> None:
        """Clear all turn history.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._turns.clear()
        self._turn_counter = 0

    def turn_count(self) -> int:
        """Return the total number of turns recorded.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._turns)

    def _compute_continuity(self, turns: List[TurnRecord]) -> float:
        """Compute topic continuity as average word overlap between consecutive turns.

        v3.4.0 PATCH: Improved continuity scoring with three additions:
        1. Stopword filtering — common words (the, is, a, to) no longer
           inflate overlap scores.
        2. Semantic topic carryover — if consecutive turns share the same
           intent or domain metadata, continuity gets a floor of 0.5 even
           when word overlap is low (e.g., "quicksort" → "time complexity"
           should be high continuity).
        3. Topic-lock bonus — when the user's turns stay in the same
           technical domain (code/architect/analyze), continuity is boosted
           because topic switches in technical conversations are explicit.

        Complexity: T(n) ∈ O(t * w) for t turns, w words.
        """
        if len(turns) < 2:
            return 1.0

        # v3.4.0 PATCH: Stopwords that should not count toward overlap
        _stopwords = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "and", "or", "but", "not", "no", "yes", "i", "you", "he",
            "she", "it", "we", "they", "this", "that", "these", "those",
            "do", "does", "did", "have", "has", "had", "will", "would",
            "can", "could", "should", "may", "might", "must", "shall",
            "what", "how", "why", "when", "where", "who", "which",
            "my", "your", "his", "her", "its", "our", "their",
            "me", "him", "us", "them", "am", "if", "then", "so",
        })

        # v3.4.0 PATCH: Technical intents that get topic-lock bonus
        _technical_intents = frozenset({"code", "architect", "analyze", "explain"})

        overlaps: List[float] = []
        for i in range(1, len(turns)):
            words_a = set(turns[i - 1].user_input.lower().split()) - _stopwords
            words_b = set(turns[i].user_input.lower().split()) - _stopwords
            if not words_a or not words_b:
                # v3.4.0 PATCH: Check intent/domain metadata for semantic carryover
                prev_intent = turns[i - 1].metadata.get("intent", "")
                curr_intent = turns[i].metadata.get("intent", "")
                if prev_intent and curr_intent and prev_intent == curr_intent:
                    overlaps.append(0.5)
                else:
                    overlaps.append(0.0)
                continue
            intersection = words_a & words_b
            union = words_a | words_b
            word_overlap = len(intersection) / len(union)

            # v3.4.0 PATCH: Semantic carryover floor
            prev_intent = turns[i - 1].metadata.get("intent", "")
            curr_intent = turns[i].metadata.get("intent", "")
            prev_domain = turns[i - 1].metadata.get("domain", "")
            curr_domain = turns[i].metadata.get("domain", "")
            if prev_intent and curr_intent and prev_intent == curr_intent:
                word_overlap = max(word_overlap, 0.5)
            if prev_domain and curr_domain and prev_domain == curr_domain:
                word_overlap = max(word_overlap, 0.4)

            # v3.4.0 PATCH: Topic-lock bonus for technical conversations
            if curr_intent in _technical_intents and prev_intent in _technical_intents:
                word_overlap = min(1.0, word_overlap + 0.15)

            overlaps.append(word_overlap)

        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    def _suggest_focus(self, dominant_intent: str, turns: List[TurnRecord]) -> str:
        """Suggest a focus for the next response.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        focus_map = {
            "code": "Provide concrete code examples and implementation details.",
            "explain": "Focus on clarity and educational structure.",
            "question": "Answer directly and concisely.",
            "fix": "Focus on diagnosis and solution steps.",
            "improve": "Focus on actionable improvements.",
            "diagnostic": "Focus on systematic analysis.",
            "conversation": "Maintain natural conversational flow.",
            "confirm": "Focus on verification and evidence.",
        }
        return focus_map.get(dominant_intent, "Respond naturally to the user's input.")

    def _summarize(self, turns: List[TurnRecord]) -> str:
        """Build a condensed summary of recent turns.

        Complexity: T(n) ∈ O(t) for t turns.
        """
        if not turns:
            return ""
        parts: List[str] = []
        for turn in turns[-5:]:  # Last 5 turns
            user_excerpt = turn.user_input[:80]
            parts.append(f"Turn {turn.turn_id} [{turn.intent}]: {user_excerpt}")
        return " | ".join(parts)
