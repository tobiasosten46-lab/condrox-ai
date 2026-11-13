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
Word Identity Engine — semantic lookup for the Condrox AI pipeline.

Replaces hardcoded keyword lists with a structured word identity system.
Each word has a type, category, intent, intent_weight, and optional emotion.

The engine loads modular JSON stores and provides O(1) word lookup.
Intent classification becomes: tokenize input → look up each word →
aggregate intent weights → pick the highest-scoring intent.

Mathematical Foundation:
- Intent score for intent I = sum(weight(w) for w in tokens if identity(w).intent == I)
- Emotion is determined by the highest-weight emotion-tagged word
- Domain is determined by the most common category among matched words

Complexity:
- Lookup: T(n) in O(1) per word, S(n) in O(1)
- Classify: T(n) in O(n) for n tokens, S(n) in O(i) for i intents
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class WordIdentity:
    """Identity record for a single word.

    Attributes:
        word: The lowercase word string.
        type: Grammatical or semantic type (noun, verb, action, concept, etc.).
        category: Semantic domain (programming, social, emotion, etc.).
        intent: Primary intent this word signals.
        intent_weight: Strength of intent signal [0.0, 1.0].
        meaning: Short definition.
        emotion: Optional emotion label if word carries emotional content.
        domain: Optional domain override for knowledge routing.
    """
    word: str
    type: str
    category: str
    intent: str
    intent_weight: float
    meaning: str
    emotion: str = ""
    domain: str = ""


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of classifying a full input string.

    Attributes:
        intent: Best-scoring intent label.
        confidence: Softmax confidence for the best intent [0.0, 1.0].
        emotion: Detected emotion label, or "neutral".
        domain: Best-matching knowledge domain.
        matched_words: Words that had identity entries.
        scores: Per-intent raw scores.
    """
    intent: str
    confidence: float
    emotion: str
    domain: str
    matched_words: List[str]
    scores: Dict[str, float]


class WordIdentityEngine:
    """Semantic word lookup engine for the Condrox AI pipeline.

    Loads modular JSON word identity stores and provides:
    - O(1) word lookup via dict
    - Intent classification from word identities
    - Emotion detection from word identities
    - Domain routing from word categories

    Complexity:
        init: T(n) in O(w) for w total words across all JSON files.
        lookup: T(n) in O(1).
        classify: T(n) in O(n) for n input tokens.
    """

    __slots__ = ("_words", "_intent_priority", "_emotion_fallback")

    def __init__(self, data_dir: Optional[str] = None) -> None:
        """Load all word identity JSON files from the data directory.

        Args:
            data_dir: Path to directory containing word identity JSON files.
                Defaults to the directory containing this module.

        Complexity: T(n) in O(w) for w total words.
        """
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(__file__))

        self._words: Dict[str, WordIdentity] = {}
        self._emotion_fallback: Dict[str, str] = {}

        # Intent priority for tie-breaking (lower = more specific = preferred)
        self._intent_priority: Dict[str, int] = {
            "clarify": 0,
            "greet": 1,
            "code": 2,
            "architect": 3,
            "analyze": 4,
            "emotional": 5,
            "command": 6,
            "explain": 7,
            "question": 8,
            "unknown": 9,
        }

        # Load all JSON files in the directory
        for filename in os.listdir(data_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                domain = data.get("domain", "general")
                for word, identity_data in data.get("words", {}).items():
                    # Dictionary entries (from dictionary.json) lack explicit
                    # intent fields and default to "unknown". Don't let them
                    # overwrite real identity entries from ai/programming/etc.
                    has_explicit_intent = "intent" in identity_data
                    if not has_explicit_intent and word in self._words:
                        continue
                    identity = WordIdentity(
                        word=word,
                        type=identity_data.get("type", "concept"),
                        category=identity_data.get("category", "general"),
                        intent=identity_data.get("intent", "unknown"),
                        intent_weight=float(identity_data.get("intent_weight", 0.5)),
                        meaning=identity_data.get("meaning", ""),
                        emotion=identity_data.get("emotion", ""),
                        domain=identity_data.get("domain", domain),
                    )
                    self._words[word] = identity
                    if identity.emotion:
                        self._emotion_fallback[word] = identity.emotion
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def lookup(self, word: str) -> Optional[WordIdentity]:
        """Look up a single word's identity.

        Args:
            word: Word to look up (case-insensitive).

        Returns:
            WordIdentity if found, None otherwise.

        Complexity: T(n) in O(1).
        """
        return self._words.get(word.lower())

    def classify(self, text: str) -> ClassificationResult:
        """Classify input text using word identities.

        Tokenizes the input, looks up each token, and aggregates
        intent weights to determine the best intent.

        Args:
            text: Input text to classify.

        Returns:
            ClassificationResult with intent, confidence, emotion, and domain.

        Complexity: T(n) in O(n) for n tokens.
        """
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        intent_scores: Dict[str, float] = {}
        emotion_scores: Dict[str, float] = {}
        domain_scores: Dict[str, float] = {}
        matched_words: List[str] = []

        for token in tokens:
            identity = self._words.get(token)
            if identity is None:
                continue
            matched_words.append(token)

            # Aggregate intent scores
            intent = identity.intent
            if intent and intent != "unknown":
                intent_scores[intent] = intent_scores.get(intent, 0.0) + identity.intent_weight

            # Aggregate emotion scores
            if identity.emotion:
                emotion_scores[identity.emotion] = (
                    emotion_scores.get(identity.emotion, 0.0) + identity.intent_weight
                )

            # Aggregate domain scores
            domain = identity.domain or identity.category
            if domain:
                domain_scores[domain] = domain_scores.get(domain, 0.0) + identity.intent_weight

        # Determine best intent
        if not intent_scores:
            return ClassificationResult(
                intent="unknown",
                confidence=0.0,
                emotion=self._pick_emotion(emotion_scores),
                domain=self._pick_domain(domain_scores),
                matched_words=matched_words,
                scores={},
            )

        # Softmax over intent scores
        max_score = max(intent_scores.values())
        exps = {k: np.exp(v - max_score) for k, v in intent_scores.items()}
        total = sum(exps.values())
        probabilities = {k: v / total for k, v in exps.items()} if total > 0 else {}

        # Tie-break by priority (lower priority value = more specific = preferred)
        best_intent = min(
            probabilities,
            key=lambda k: (-probabilities[k], self._intent_priority.get(k, 99)),
        )

        return ClassificationResult(
            intent=best_intent,
            confidence=probabilities[best_intent],
            emotion=self._pick_emotion(emotion_scores),
            domain=self._pick_domain(domain_scores),
            matched_words=matched_words,
            scores=intent_scores,
        )

    def _pick_emotion(self, emotion_scores: Dict[str, float]) -> str:
        """Pick the best emotion from scored candidates.

        Args:
            emotion_scores: Dict of emotion label to weight.

        Returns:
            Best emotion label, or "neutral" if none.

        Complexity: T(n) in O(e) for e emotions.
        """
        if not emotion_scores:
            return "neutral"
        return max(emotion_scores, key=lambda k: emotion_scores[k])

    def _pick_domain(self, domain_scores: Dict[str, float]) -> str:
        """Pick the best domain from scored candidates.

        Args:
            domain_scores: Dict of domain label to weight.

        Returns:
            Best domain label, or "general" if none.

        Complexity: T(n) in O(d) for d domains.
        """
        if not domain_scores:
            return "general"
        return max(domain_scores, key=lambda k: domain_scores[k])

    @property
    def word_count(self) -> int:
        """Return total number of words in the identity store."""
        return len(self._words)
