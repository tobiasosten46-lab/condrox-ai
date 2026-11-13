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
User Pattern Learner — behavioral and emotional pattern extraction.

Analyzes each user interaction to extract and persist behavioral patterns,
emotional tendencies, conversational style, and topic preferences. These
patterns are stored as tagged memories so the AI can personalize responses
and demonstrate understanding of the user over time.

What gets learned:
- Emotional patterns: dominant emotions, emotional volatility, shifts
- Conversational style: formality, question vs statement ratio, verbosity
- Topic preferences: domains and subjects the user engages with most
- Communication patterns: greeting style, clarification frequency, code vs chat
- Behavioral signals: frustration indicators, curiosity level, engagement

What does NOT get learned:
- Raw personal facts (PII) — those are handled by Stage07bMemoryStore
- Irrelevant chitchat content — only patterns/signals are extracted

Complexity:
    learn: T(n) ∈ O(n) for input length n, S(n) ∈ O(1) per pattern.
    get_patterns: T(n) ∈ O(m) for m stored pattern memories.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


@dataclass(frozen=True, slots=True)
class UserPattern:
    """
    Immutable extracted user pattern from a single interaction.

    Attributes:
        pattern_type: Category of pattern (emotion, style, topic, behavior).
        pattern_key: Specific pattern identifier (e.g., "dominant_emotion").
        pattern_value: The observed value (e.g., "curious").
        confidence: How strong the signal is [0.0, 1.0].
        context: Additional context about when this was observed.

    Complexity: S(n) ∈ O(1).
    """

    pattern_type: str
    pattern_key: str
    pattern_value: str
    confidence: float
    context: str = ""


class UserPatternLearner:
    """
    Extracts and persists user behavioral/emotional patterns.

    Called after each pipeline run to analyze the interaction and store
    learned patterns as memories. Patterns are tagged for selective retrieval
    during future interactions.

    Complexity:
        learn: T(n) ∈ O(n) for input analysis, S(n) ∈ O(p) for p patterns.
    """

    __slots__ = (
        "_emotion_history",
        "_interaction_count",
        "_topic_history",
        "_style_history",
        "_max_history",
    )

    def __init__(self, max_history: int = 100) -> None:
        """Initialize the pattern learner.

        Args:
            max_history: Maximum interactions to keep in rolling history.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._emotion_history: List[Tuple[str, float, float]] = []
        self._interaction_count: int = 0
        self._topic_history: List[Tuple[str, float]] = []
        self._style_history: List[Tuple[str, float]] = []
        self._max_history: int = max_history

    def learn(
        self,
        controller: "BrainController",
        user_input: str,
        emotion: str,
        emotion_confidence: float,
        intent: str,
        mode: str,
        domain: str,
        final_output: str,
        store_memory: bool = True,
    ) -> List[UserPattern]:
        """Analyze an interaction and store learned patterns.

        Args:
            controller: BrainController with memory_manager.
            user_input: Raw user input text.
            emotion: Detected emotion label.
            emotion_confidence: Emotion detection confidence.
            intent: Classified intent.
            mode: Processing mode used.
            domain: Knowledge domain.
            final_output: The AI's response.
            store_memory: When False, skip memory writes (PreReleaseAgent).

        Returns:
            List of extracted UserPattern objects.

        Complexity: T(n) ∈ O(n) for input analysis, S(n) ∈ O(p) for p patterns.
        """
        if not user_input or not user_input.strip():
            return []

        self._interaction_count += 1
        now = time.time()
        patterns: List[UserPattern] = []

        # ─── 1. Emotional Pattern Analysis ───────────────────────────
        patterns.extend(
            self._analyze_emotion(emotion, emotion_confidence, now)
        )

        # ─── 2. Conversational Style Analysis ────────────────────────
        patterns.extend(
            self._analyze_style(user_input, intent, now)
        )

        # ─── 3. Topic/Domain Preference Analysis ─────────────────────
        patterns.extend(
            self._analyze_topic(domain, intent, mode, now)
        )

        # ─── 4. Behavioral Signal Analysis ───────────────────────────
        patterns.extend(
            self._analyze_behavior(user_input, intent, emotion, now)
        )

        # ─── 5. Engagement Pattern Analysis ──────────────────────────
        patterns.extend(
            self._analyze_engagement(user_input, final_output, now)
        )

        # Store patterns as memories
        if store_memory:
            self._store_patterns(controller, patterns, now)

        # Update rolling histories
        self._emotion_history.append((emotion, emotion_confidence, now))
        self._topic_history.append((domain, now))
        self._style_history.append((self._classify_style(user_input), now))

        # Trim histories
        if len(self._emotion_history) > self._max_history:
            self._emotion_history = self._emotion_history[-self._max_history:]
        if len(self._topic_history) > self._max_history:
            self._topic_history = self._topic_history[-self._max_history:]
        if len(self._style_history) > self._max_history:
            self._style_history = self._style_history[-self._max_history:]

        return patterns

    def _analyze_emotion(
        self, emotion: str, confidence: float, now: float
    ) -> List[UserPattern]:
        """Extract emotional patterns.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        patterns: List[UserPattern] = []

        # Track dominant emotion
        if emotion and emotion != "neutral":
            patterns.append(UserPattern(
                pattern_type="emotion",
                pattern_key="dominant_emotion",
                pattern_value=emotion,
                confidence=confidence,
                context=f"Observed at interaction #{self._interaction_count}",
            ))

        # Track emotional volatility — how many distinct emotions in recent history
        if len(self._emotion_history) >= 3:
            recent_emotions = [e[0] for e in self._emotion_history[-10:]]
            unique = set(recent_emotions)
            if len(unique) > 3:
                patterns.append(UserPattern(
                    pattern_type="emotion",
                    pattern_key="volatility",
                    pattern_value="high",
                    confidence=min(1.0, len(unique) / 5.0),
                    context=f"{len(unique)} distinct emotions in last 10 interactions",
                ))
            elif len(unique) <= 2 and len(recent_emotions) >= 5:
                patterns.append(UserPattern(
                    pattern_type="emotion",
                    pattern_key="volatility",
                    pattern_value="low",
                    confidence=0.7,
                    context=f"Consistent emotional state ({unique})",
                ))

        # Track emotional tendency — most common emotion
        if len(self._emotion_history) >= 5:
            emotion_counts: Dict[str, int] = {}
            for e, _, _ in self._emotion_history[-20:]:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            most_common = max(emotion_counts, key=emotion_counts.get)
            if most_common != "neutral":
                ratio = emotion_counts[most_common] / len(self._emotion_history[-20:])
                patterns.append(UserPattern(
                    pattern_type="emotion",
                    pattern_key="tendency",
                    pattern_value=most_common,
                    confidence=ratio,
                    context=f"{most_common} appears {emotion_counts[most_common]}/{len(self._emotion_history[-20:])} recent interactions",
                ))

        return patterns

    def _analyze_style(
        self, user_input: str, intent: str, now: float
    ) -> List[UserPattern]:
        """Extract conversational style patterns.

        Complexity: T(n) ∈ O(n) for input length n, S(n) ∈ O(1).
        """
        patterns: List[UserPattern] = []
        text_lower = user_input.lower()

        # Formality detection
        formal_markers = ("please", "thank you", "could you", "would you",
                          "may i", "might i", "i would like", "i appreciate")
        informal_markers = ("hey", "yo", "sup", "lol", "haha", "btw",
                           "gonna", "wanna", "gotta", "yeah", "nope")
        formal_count = sum(1 for m in formal_markers if m in text_lower)
        informal_count = sum(1 for m in informal_markers if m in text_lower)

        if formal_count > informal_count and formal_count > 0:
            patterns.append(UserPattern(
                pattern_type="style",
                pattern_key="formality",
                pattern_value="formal",
                confidence=min(1.0, formal_count / 3.0),
            ))
        elif informal_count > formal_count and informal_count > 0:
            patterns.append(UserPattern(
                pattern_type="style",
                pattern_key="formality",
                pattern_value="informal",
                confidence=min(1.0, informal_count / 3.0),
            ))

        # Verbosity — input length categories
        word_count = len(user_input.split())
        if word_count <= 5:
            verbosity = "terse"
            conf = 0.8
        elif word_count <= 15:
            verbosity = "concise"
            conf = 0.7
        elif word_count <= 40:
            verbosity = "moderate"
            conf = 0.6
        else:
            verbosity = "verbose"
            conf = 0.8
        patterns.append(UserPattern(
            pattern_type="style",
            pattern_key="verbosity",
            pattern_value=verbosity,
            confidence=conf,
            context=f"{word_count} words",
        ))

        # Question vs statement ratio
        question_marks = user_input.count("?")
        if question_marks > 0:
            patterns.append(UserPattern(
                pattern_type="style",
                pattern_key="communication_mode",
                pattern_value="inquisitive",
                confidence=min(1.0, question_marks / 3.0),
                context=f"{question_marks} questions in input",
            ))
        elif intent in ("code", "command", "architect"):
            patterns.append(UserPattern(
                pattern_type="style",
                pattern_key="communication_mode",
                pattern_value="directive",
                confidence=0.7,
            ))

        # Punctuation style
        if "!" in user_input and user_input.count("!") >= 2:
            patterns.append(UserPattern(
                pattern_type="style",
                pattern_key="punctuation_style",
                pattern_value="expressive",
                confidence=0.6,
            ))
        elif "..." in user_input:
            patterns.append(UserPattern(
                pattern_type="style",
                pattern_key="punctuation_style",
                pattern_value="hesitant",
                confidence=0.6,
            ))

        return patterns

    def _analyze_topic(
        self, domain: str, intent: str, mode: str, now: float
    ) -> List[UserPattern]:
        """Extract topic/domain preference patterns.

        Complexity: T(n) ∈ O(h) for history length h, S(n) ∈ O(1).
        """
        patterns: List[UserPattern] = []

        # Track domain preference
        if domain and domain != "general":
            patterns.append(UserPattern(
                pattern_type="topic",
                pattern_key="domain_preference",
                pattern_value=domain,
                confidence=0.6,
            ))

        # Track dominant domain from history
        if len(self._topic_history) >= 5:
            domain_counts: Dict[str, int] = {}
            for d, _ in self._topic_history[-20:]:
                domain_counts[d] = domain_counts.get(d, 0) + 1
            top_domain = max(domain_counts, key=domain_counts.get)
            if top_domain != "general":
                ratio = domain_counts[top_domain] / len(self._topic_history[-20:])
                patterns.append(UserPattern(
                    pattern_type="topic",
                    pattern_key="primary_interest",
                    pattern_value=top_domain,
                    confidence=ratio,
                    context=f"{top_domain} appears in {domain_counts[top_domain]}/{len(self._topic_history[-20:])} recent interactions",
                ))

        # Track mode preference (code vs chat vs analysis)
        mode_labels = {
            "code": "programming",
            "explain": "learning",
            "analyze": "analysis",
            "architect": "system_design",
            "conversational": "conversation",
            "creative": "creative",
        }
        mode_label = mode_labels.get(mode, mode)
        if mode_label != mode:
            patterns.append(UserPattern(
                pattern_type="topic",
                pattern_key="interaction_type",
                pattern_value=mode_label,
                confidence=0.7,
            ))

        return patterns

    def _analyze_behavior(
        self, user_input: str, intent: str, emotion: str, now: float
    ) -> List[UserPattern]:
        """Extract behavioral signal patterns.

        Complexity: T(n) ∈ O(n) for input length n, S(n) ∈ O(1).
        """
        patterns: List[UserPattern] = []
        text_lower = user_input.lower()

        # Frustration indicators
        frustration_markers = (
            "not working", "doesn't work", "broken", "stupid", "why won't",
            "keep getting", "error again", "still not", "wtf", "damn",
            "everydamn", "stop patch", "fix it", "not right",
        )
        frustration_count = sum(1 for m in frustration_markers if m in text_lower)
        if frustration_count > 0:
            patterns.append(UserPattern(
                pattern_type="behavior",
                pattern_key="frustration_level",
                pattern_value="elevated",
                confidence=min(1.0, frustration_count / 2.0),
                context=f"{frustration_count} frustration markers detected",
            ))

        # Curiosity indicators
        curiosity_markers = (
            "how does", "what if", "why does", "can you explain",
            "tell me more", "what happens when", "how would",
            "what's the difference", "could you elaborate",
        )
        curiosity_count = sum(1 for m in curiosity_markers if m in text_lower)
        if curiosity_count > 0:
            patterns.append(UserPattern(
                pattern_type="behavior",
                pattern_key="curiosity_level",
                pattern_value="high",
                confidence=min(1.0, curiosity_count / 2.0),
                context=f"{curiosity_count} curiosity markers detected",
            ))

        # Clarification frequency — user frequently corrects the AI
        clarification_markers = (
            "no wait", "actually no", "not that", "i mean",
            "sorry i meant", "that's not what", "no how",
        )
        clarification_count = sum(1 for m in clarification_markers if m in text_lower)
        if clarification_count > 0:
            patterns.append(UserPattern(
                pattern_type="behavior",
                pattern_key="clarification_frequency",
                pattern_value="high",
                confidence=0.7,
                context="User frequently corrects or clarifies",
            ))

        # Greeting pattern
        greeting_markers = ("hi", "hello", "hey", "good morning", "good evening",
                           "good afternoon", "howdy", "greetings")
        if any(text_lower.strip().startswith(g) for g in greeting_markers):
            patterns.append(UserPattern(
                pattern_type="behavior",
                pattern_key="greeting_style",
                pattern_value="warm",
                confidence=0.6,
            ))

        # Technical proficiency level
        advanced_markers = (
            "pipeline", "architecture", "async", "concurrency", "race condition",
            "memory leak", "deadlock", "bytestring", "serialization",
            "middleware", "dependency injection", "orm", "query optimization",
        )
        beginner_markers = (
            "what is python", "how to code", "basic", "beginner",
            "simple program", "easy code", "first time",
        )
        advanced_count = sum(1 for m in advanced_markers if m in text_lower)
        beginner_count = sum(1 for m in beginner_markers if m in text_lower)
        if advanced_count > beginner_count and advanced_count > 0:
            patterns.append(UserPattern(
                pattern_type="behavior",
                pattern_key="technical_level",
                pattern_value="advanced",
                confidence=min(1.0, advanced_count / 2.0),
            ))
        elif beginner_count > 0:
            patterns.append(UserPattern(
                pattern_type="behavior",
                pattern_key="technical_level",
                pattern_value="beginner",
                confidence=min(1.0, beginner_count / 2.0),
            ))

        return patterns

    def _analyze_engagement(
        self, user_input: str, final_output: str, now: float
    ) -> List[UserPattern]:
        """Extract engagement patterns from input/output interaction.

        Complexity: T(n) ∈ O(n) for input length, S(n) ∈ O(1).
        """
        patterns: List[UserPattern] = []

        # Response satisfaction signal — if user's next input references the
        # previous output, they're engaged in a multi-turn conversation
        if self._interaction_count > 1:
            # Check if this input references previous output
            reference_markers = (
                "that's not right", "not what i asked", "try again",
                "yes that's it", "perfect", "exactly", "that worked",
                "thanks", "thank you", "great", "awesome",
            )
            text_lower = user_input.lower()
            if any(m in text_lower for m in reference_markers):
                patterns.append(UserPattern(
                    pattern_type="engagement",
                    pattern_key="multi_turn",
                    pattern_value="active",
                    confidence=0.7,
                ))

        # Session depth — how deep into a session
        if self._interaction_count >= 10:
            patterns.append(UserPattern(
                pattern_type="engagement",
                pattern_key="session_depth",
                pattern_value="deep",
                confidence=min(1.0, self._interaction_count / 20.0),
                context=f"{self._interaction_count} interactions in session",
            ))
        elif self._interaction_count >= 3:
            patterns.append(UserPattern(
                pattern_type="engagement",
                pattern_key="session_depth",
                pattern_value="moderate",
                confidence=0.5,
            ))

        return patterns

    def _classify_style(self, user_input: str) -> str:
        """Classify the overall conversational style of an input.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        text_lower = user_input.lower()
        word_count = len(user_input.split())

        if word_count <= 5:
            return "terse"
        if "?" in user_input:
            return "inquisitive"
        if any(m in text_lower for m in ("please", "thank", "could you")):
            return "formal"
        if any(m in text_lower for m in ("hey", "lol", "yeah", "btw")):
            return "informal"
        return "standard"

    def _store_patterns(
        self,
        controller: "BrainController",
        patterns: List[UserPattern],
        timestamp: float,
    ) -> None:
        """Store extracted patterns as memories.

        Each pattern is stored with descriptive tags for selective retrieval.
        Patterns are deduplicated by pattern_key — only the most recent
        observation of each key is kept fresh.

        Complexity: T(n) ∈ O(p) for p patterns, S(n) ∈ O(1).
        """
        for p in patterns:
            # Build a descriptive memory content string
            content = self._format_pattern_memory(p, timestamp)
            item_id = f"pattern_{p.pattern_type}_{p.pattern_key}_{hashlib.sha256(content.encode()).hexdigest()[:8]}"

            tags = (
                "user_pattern",
                p.pattern_type,
                p.pattern_key,
                "learned",
            )

            try:
                controller.add_memory(
                    content=content,
                    source="pattern_learning",
                    tags=list(tags),
                    weight=p.confidence,
                    item_id=item_id,
                )
            except Exception:
                pass  # Memory storage failure is non-critical

    @staticmethod
    def _format_pattern_memory(p: UserPattern, timestamp: float) -> str:
        """Format a pattern into a human-readable memory string.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        parts = [
            f"[User Pattern — {p.pattern_type}]",
            f"{p.pattern_key}: {p.pattern_value}",
            f"confidence: {p.confidence:.0%}",
        ]
        if p.context:
            parts.append(f"context: {p.context}")
        return " | ".join(parts)

    def get_pattern_summary(self) -> Dict[str, str]:
        """Return a summary of observed patterns for introspection.

        Complexity: T(n) ∈ O(h) for history length h, S(n) ∈ O(k) for k keys.
        """
        summary: Dict[str, str] = {}

        # Emotion summary
        if self._emotion_history:
            emotion_counts: Dict[str, int] = {}
            for e, _, _ in self._emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            top_emotion = max(emotion_counts, key=emotion_counts.get)
            summary["dominant_emotion"] = top_emotion
            summary["emotion_distribution"] = ", ".join(
                f"{e}:{c}" for e, c in sorted(
                    emotion_counts.items(), key=lambda x: -x[1]
                )
            )

        # Topic summary
        if self._topic_history:
            domain_counts: Dict[str, int] = {}
            for d, _ in self._topic_history:
                domain_counts[d] = domain_counts.get(d, 0) + 1
            top_domain = max(domain_counts, key=domain_counts.get)
            summary["primary_domain"] = top_domain

        # Style summary
        if self._style_history:
            style_counts: Dict[str, int] = {}
            for s, _ in self._style_history:
                style_counts[s] = style_counts.get(s, 0) + 1
            top_style = max(style_counts, key=style_counts.get)
            summary["dominant_style"] = top_style

        summary["total_interactions"] = str(self._interaction_count)
        return summary
