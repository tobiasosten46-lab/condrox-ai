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

"""Pattern Recognizer — detects recurring patterns in conversations and knowledge.

Analyzes conversation history, user behavior, and knowledge interactions to
identify patterns:
- Topic patterns (what topics recur across conversations)
- Question patterns (how users phrase questions)
- Response patterns (what response styles work best)
- Knowledge patterns (what knowledge is accessed together)
- Temporal patterns (when certain topics come up)

The pattern recognizer enables the AI to predict what the user might ask
next and proactively prepare relevant knowledge.

Complexity:
    recognize_patterns: T(n) ∈ O(n * p) for n interactions, p pattern types
    predict_next_topic: T(n) ∈ O(t * c) for t topics, c co-occurrences

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class Pattern:
    """A detected pattern in conversations or knowledge.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    pattern_type: str       # "topic_cooccurrence", "question_style", "response_success", etc.
    subject: str            # Pattern subject (e.g., "python+recursion")
    description: str        # Human-readable description
    frequency: int          # How often this pattern occurred
    confidence: float       # How confident we are [0.0, 1.0]
    first_seen: float       # When first detected
    last_seen: float        # When last detected
    metadata: Tuple[Tuple[str, str], ...] = ()  # Additional key-value pairs


@dataclass(slots=True)
class InteractionRecord:
    """Record of a single conversation interaction.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(t).
    """
    timestamp: float
    intent: str
    emotion: str
    topics: List[str] = field(default_factory=list)
    domain: str = "general"
    user_message_length: int = 0
    response_length: int = 0
    response_time_ms: float = 0.0
    follow_up_asked: bool = False
    user_returned: bool = False  # Did the user continue after this turn?
    curiosity_score: float = 0.0


class PatternRecognizer:
    """Recognizes patterns in conversation and knowledge data.

    Tracks interactions over time and identifies:
    1. Topic co-occurrence — which topics appear together
    2. Question patterns — how users phrase questions
    3. Response success — what response styles lead to continued conversation
    4. Knowledge clusters — what knowledge domains are accessed together
    5. Temporal patterns — when certain topics come up

    Complexity:
        record_interaction: T(n) ∈ O(1)
        recognize_patterns: T(n) ∈ O(n * p) for n interactions, p pattern types
        predict_next_topics: T(n) ∈ O(t * c) for t topics, c co-occurrences
    """

    __slots__ = (
        "_interactions",
        "_topic_cooccurrence",
        "_topic_frequency",
        "_intent_topic_map",
        "_response_success_by_intent",
        "_domain_transitions",
        "_max_interactions",
    )

    def __init__(self, max_interactions: int = 10000) -> None:
        """Initialize the pattern recognizer.

        Args:
            max_interactions: Maximum interactions to keep in memory.
        """
        self._interactions: List[InteractionRecord] = []
        self._topic_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._topic_frequency: Counter = Counter()
        self._intent_topic_map: Dict[str, Counter] = defaultdict(Counter)
        self._response_success_by_intent: Dict[str, List[float]] = defaultdict(list)
        self._domain_transitions: Dict[str, Counter] = defaultdict(Counter)
        self._max_interactions = max_interactions

    def record_interaction(self, record: InteractionRecord) -> None:
        """Record a conversation interaction for pattern analysis.

        Complexity: T(n) ∈ O(t²) for t topics (co-occurrence matrix).
        """
        self._interactions.append(record)
        if len(self._interactions) > self._max_interactions:
            self._interactions = self._interactions[-self._max_interactions:]

        # Track topic frequency
        for topic in record.topics:
            self._topic_frequency[topic] += 1

        # Track topic co-occurrence
        topics = record.topics
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                self._topic_cooccurrence[topics[i]][topics[j]] += 1
                self._topic_cooccurrence[topics[j]][topics[i]] += 1

        # Track intent → topic mapping
        for topic in record.topics:
            self._intent_topic_map[record.intent][topic] += 1

        # Track response success (user returned = success)
        success = 1.0 if record.user_returned else 0.0
        self._response_success_by_intent[record.intent].append(success)
        # Keep only last 100 records per intent
        if len(self._response_success_by_intent[record.intent]) > 100:
            self._response_success_by_intent[record.intent] = \
                self._response_success_by_intent[record.intent][-100:]

        # Track domain transitions
        if len(self._interactions) >= 2:
            prev_domain = self._interactions[-2].domain
            curr_domain = record.domain
            if prev_domain != curr_domain:
                self._domain_transitions[prev_domain][curr_domain] += 1

    def recognize_patterns(self) -> List[Pattern]:
        """Recognize all patterns from recorded interactions.

        Complexity: T(n) ∈ O(n + t² + i + d) for n interactions, t topics,
                    i intents, d domain transitions.
        """
        patterns: List[Pattern] = []
        now = time.time()

        # 1. Topic co-occurrence patterns
        for topic_a, cooccurrences in self._topic_cooccurrence.items():
            for topic_b, count in cooccurrences.items():
                if count >= 2 and topic_a < topic_b:  # Avoid duplicates
                    total_a = self._topic_frequency.get(topic_a, 1)
                    confidence = min(1.0, count / max(1, total_a))
                    patterns.append(Pattern(
                        pattern_type="topic_cooccurrence",
                        subject=f"{topic_a}+{topic_b}",
                        description=f"'{topic_a}' and '{topic_b}' appear together {count} times",
                        frequency=count,
                        confidence=confidence,
                        first_seen=now,
                        last_seen=now,
                        metadata=(("topic_a", topic_a), ("topic_b", topic_b)),
                    ))

        # 2. Intent-topic patterns
        for intent, topic_counts in self._intent_topic_map.items():
            for topic, count in topic_counts.most_common(5):
                if count >= 2:
                    patterns.append(Pattern(
                        pattern_type="intent_topic",
                        subject=f"{intent}→{topic}",
                        description=f"Intent '{intent}' is associated with topic '{topic}' ({count} times)",
                        frequency=count,
                        confidence=min(1.0, count / 10.0),
                        first_seen=now,
                        last_seen=now,
                        metadata=(("intent", intent), ("topic", topic)),
                    ))

        # 3. Response success patterns
        for intent, successes in self._response_success_by_intent.items():
            if len(successes) >= 3:
                avg_success = sum(successes) / len(successes)
                patterns.append(Pattern(
                    pattern_type="response_success",
                    subject=intent,
                    description=f"Intent '{intent}' has {avg_success:.0%} conversation continuation rate",
                    frequency=len(successes),
                    confidence=avg_success,
                    first_seen=now,
                    last_seen=now,
                    metadata=(("intent", intent), ("avg_success", f"{avg_success:.2f}")),
                ))

        # 4. Domain transition patterns
        for from_domain, transitions in self._domain_transitions.items():
            for to_domain, count in transitions.most_common(3):
                if count >= 2:
                    patterns.append(Pattern(
                        pattern_type="domain_transition",
                        subject=f"{from_domain}→{to_domain}",
                        description=f"Conversations shift from '{from_domain}' to '{to_domain}' ({count} times)",
                        frequency=count,
                        confidence=min(1.0, count / 5.0),
                        first_seen=now,
                        last_seen=now,
                        metadata=(("from", from_domain), ("to", to_domain)),
                    ))

        # 5. Temporal patterns — topics by time of day
        hour_topics: Dict[int, Counter] = defaultdict(Counter)
        for interaction in self._interactions:
            hour = int(time.gmtime(interaction.timestamp).tm_hour)
            for topic in interaction.topics:
                hour_topics[hour][topic] += 1

        for hour, topic_counts in hour_topics.items():
            top_topic, count = topic_counts.most_common(1)[0] if topic_counts else ("", 0)
            if count >= 3:
                patterns.append(Pattern(
                    pattern_type="temporal",
                    subject=f"hour_{hour}_{top_topic}",
                    description=f"'{top_topic}' is most discussed at hour {hour}:00 UTC ({count} times)",
                    frequency=count,
                    confidence=min(1.0, count / 10.0),
                    first_seen=now,
                    last_seen=now,
                    metadata=(("hour", str(hour)), ("topic", top_topic)),
                ))

        return patterns

    def predict_next_topics(self, current_topics: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict what topics might come next based on co-occurrence.

        Args:
            current_topics: Topics in the current conversation.
            top_k: Number of predictions to return.

        Returns:
            List of (predicted_topic, probability) tuples.

        Complexity: T(n) ∈ O(t * c) for t current topics, c co-occurrences.
        """
        if not current_topics:
            # Return most popular topics
            return [(t, c / max(1, sum(self._topic_frequency.values())))
                    for t, c in self._topic_frequency.most_common(top_k)]

        scores: Dict[str, float] = {}
        for topic in current_topics:
            cooccurrences = self._topic_cooccurrence.get(topic, {})
            total_cooc = sum(cooccurrences.values())
            if total_cooc == 0:
                continue
            for related, count in cooccurrences.items():
                if related in current_topics:
                    continue
                prob = count / total_cooc
                scores[related] = scores.get(related, 0.0) + prob

        # Normalize and sort
        if scores:
            max_score = max(scores.values())
            scores = {t: s / max_score for t, s in scores.items()}

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_dominant_pattern_type(self) -> str:
        """Get the most common pattern type.

        Complexity: T(n) ∈ O(n) for n patterns.
        """
        patterns = self.recognize_patterns()
        if not patterns:
            return "none"
        type_counts: Counter = Counter(p.pattern_type for p in patterns)
        return type_counts.most_common(1)[0][0]

    @property
    def interaction_count(self) -> int:
        return len(self._interactions)

    @property
    def unique_topics(self) -> int:
        return len(self._topic_frequency)
