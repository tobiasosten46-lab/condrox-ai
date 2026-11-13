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

"""Behavior Analyzer — analyzes user behavior and preferences.

Tracks how users interact with the AI to build a behavioral profile:
- Communication style (formal, casual, technical)
- Knowledge level (beginner, intermediate, expert)
- Interest areas (what domains they ask about)
- Question patterns (how they phrase questions)
- Engagement patterns (when they're most active)
- Emotional patterns (what emotions they express)

This enables the AI to adapt its communication style to each user.

Complexity:
    analyze: T(n) ∈ O(n) for n interactions
    adapt_style: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class UserProfile:
    """Behavioral profile of a user.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    communication_style: str    # "formal", "casual", "technical", "mixed"
    knowledge_level: str        # "beginner", "intermediate", "expert"
    primary_interests: Tuple[str, ...]      # Top interest domains
    question_style: str         # "direct", "exploratory", "detailed", "brief"
    engagement_level: float     # [0.0, 1.0] — how engaged the user is
    emotional_tendency: str     # Most common emotion
    avg_message_length: float   # Average message length in characters
    prefers_code: bool          # Does the user prefer code examples?
    prefers_explanations: bool  # Does the user prefer detailed explanations?
    total_interactions: int     # Total conversation turns
    session_count: int          # Number of separate sessions
    last_active: float          # Last activity timestamp


class BehaviorAnalyzer:
    """Analyzes user behavior to build adaptive profiles.

    The analyzer tracks interaction patterns and produces a UserProfile
    that the AI uses to adapt its communication style.

    Complexity:
        record: T(n) ∈ O(1)
        analyze: T(n) ∈ O(n) for n interactions
    """

    __slots__ = (
        "_messages",
        "_intents",
        "_emotions",
        "_domains",
        "_message_lengths",
        "_response_lengths",
        "_code_requests",
        "_explanation_requests",
        "_timestamps",
        "_sessions",
        "_max_records",
    )

    def __init__(self, max_records: int = 5000) -> None:
        """Initialize the behavior analyzer."""
        self._messages: List[str] = []
        self._intents: List[str] = []
        self._emotions: List[str] = []
        self._domains: List[str] = []
        self._message_lengths: List[int] = []
        self._response_lengths: List[int] = []
        self._code_requests: int = 0
        self._explanation_requests: int = 0
        self._timestamps: List[float] = []
        self._sessions: int = 1
        self._max_records = max_records

    def record(
        self,
        message: str,
        intent: str,
        emotion: str,
        domain: str,
        response_length: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        """Record a user interaction for behavioral analysis.

        Complexity: T(n) ∈ O(1).
        """
        ts = timestamp or time.time()
        self._messages.append(message)
        self._intents.append(intent)
        self._emotions.append(emotion)
        self._domains.append(domain)
        self._message_lengths.append(len(message))
        self._response_lengths.append(response_length)
        self._timestamps.append(ts)

        if intent == "code":
            self._code_requests += 1
        if intent in ("explain", "question"):
            self._explanation_requests += 1

        # Trim if over limit
        if len(self._messages) > self._max_records:
            self._messages = self._messages[-self._max_records:]
            self._intents = self._intents[-self._max_records:]
            self._emotions = self._emotions[-self._max_records:]
            self._domains = self._domains[-self._max_records:]
            self._message_lengths = self._message_lengths[-self._max_records:]
            self._response_lengths = self._response_lengths[-self._max_records:]
            self._timestamps = self._timestamps[-self._max_records:]

    def new_session(self) -> None:
        """Record that a new session has started."""
        self._sessions += 1

    def analyze(self) -> UserProfile:
        """Analyze recorded behavior and produce a user profile.

        Complexity: T(n) ∈ O(n) for n interactions.
        """
        if not self._messages:
            return UserProfile(
                communication_style="neutral",
                knowledge_level="beginner",
                primary_interests=(),
                question_style="direct",
                engagement_level=0.0,
                emotional_tendency="neutral",
                avg_message_length=0.0,
                prefers_code=False,
                prefers_explanations=False,
                total_interactions=0,
                session_count=self._sessions,
                last_active=time.time(),
            )

        # Communication style
        style = self._detect_communication_style()

        # Knowledge level
        knowledge = self._detect_knowledge_level()

        # Primary interests
        domain_counts = Counter(self._domains)
        top_domains = tuple(d for d, _ in domain_counts.most_common(5))

        # Question style
        question_style = self._detect_question_style()

        # Engagement level
        engagement = self._calculate_engagement()

        # Emotional tendency
        emotion_counts = Counter(self._emotions)
        emotional_tendency = emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"

        # Average message length
        avg_len = sum(self._message_lengths) / len(self._message_lengths)

        # Preferences
        total = len(self._intents)
        prefers_code = self._code_requests > total * 0.3 if total > 0 else False
        prefers_explanations = self._explanation_requests > total * 0.3 if total > 0 else False

        return UserProfile(
            communication_style=style,
            knowledge_level=knowledge,
            primary_interests=top_domains,
            question_style=question_style,
            engagement_level=engagement,
            emotional_tendency=emotional_tendency,
            avg_message_length=avg_len,
            prefers_code=prefers_code,
            prefers_explanations=prefers_explanations,
            total_interactions=total,
            session_count=self._sessions,
            last_active=self._timestamps[-1] if self._timestamps else time.time(),
        )

    def _detect_communication_style(self) -> str:
        """Detect the user's communication style.

        Complexity: T(n) ∈ O(n) for n messages.
        """
        formal_signals = 0
        casual_signals = 0
        technical_signals = 0

        for msg in self._messages[-100:]:  # Last 100 messages
            msg_lower = msg.lower()
            # Formal signals
            if any(w in msg_lower for w in ("please", "thank you", "could you", "would you", "kindly")):
                formal_signals += 1
            # Casual signals
            if any(w in msg_lower for w in ("hey", "sup", "lol", "btw", "gonna", "wanna", "yeah", "nah")):
                casual_signals += 1
            # Technical signals
            if any(w in msg_lower for w in (
                "function", "class", "api", "endpoint", "schema", "query",
                "algorithm", "complexity", "async", "concurrent", "distributed",
            )):
                technical_signals += 1

        max_sig = max(formal_signals, casual_signals, technical_signals)
        if max_sig == 0:
            return "neutral"
        if max_sig == formal_signals:
            return "formal"
        if max_sig == casual_signals:
            return "casual"
        return "technical"

    def _detect_knowledge_level(self) -> str:
        """Detect the user's knowledge level.

        Complexity: T(n) ∈ O(n) for n intents.
        """
        if not self._intents:
            return "beginner"

        code_count = self._intents.count("code")
        architect_count = self._intents.count("architect")
        analyze_count = self._intents.count("analyze")
        greet_count = self._intents.count("greet")
        total = len(self._intents)

        advanced_ratio = (code_count + architect_count + analyze_count) / total
        if advanced_ratio > 0.5:
            return "expert"
        if advanced_ratio > 0.25:
            return "intermediate"
        return "beginner"

    def _detect_question_style(self) -> str:
        """Detect how the user phrases questions.

        Complexity: T(n) ∈ O(n) for n messages.
        """
        if not self._message_lengths:
            return "direct"

        avg_len = sum(self._message_lengths) / len(self._message_lengths)

        if avg_len < 30:
            return "brief"
        if avg_len < 100:
            return "direct"
        if avg_len < 250:
            return "detailed"
        return "exploratory"

    def _calculate_engagement(self) -> float:
        """Calculate how engaged the user is.

        Complexity: T(n) ∈ O(n) for n interactions.
        """
        if not self._timestamps:
            return 0.0

        # Engagement = frequency of interactions + message length + response to follow-ups
        total = len(self._messages)
        avg_len = sum(self._message_lengths) / max(1, total)

        # Normalize: 10+ interactions = high engagement, 1-2 = low
        frequency_score = min(1.0, total / 20.0)

        # Longer messages = more engaged (normalized: 200 chars = high)
        length_score = min(1.0, avg_len / 200.0)

        return (frequency_score * 0.6 + length_score * 0.4)

    @property
    def interaction_count(self) -> int:
        return len(self._messages)
