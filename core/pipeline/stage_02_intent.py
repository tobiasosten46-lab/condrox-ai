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
Stage 02: Intent Engine.

Classifies intent using the Word Identity Engine as primary classifier,
falling back to keyword/regex scoring for phrases not in the identity store.

Mathematical Foundation:
- Word Identity: intent score = sum(weight(w) for w in tokens if identity(w).intent == I)
- Fallback: raw activation a_i = sum(keyword_matches), softmax normalized
- Combined: word identity scores + fallback keyword scores, softmax normalized
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Dict, List, Tuple

from core.brain.brain_state import BrainState
from core.cognition.emotion_detector import EmotionDetector
from core.word_identity import WordIdentityEngine
from core.word_identity.word_dictionary import WordDictionary

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage02Intent:
    """Intent classification stage using Word Identity Engine + keyword fallback.

    Primary: WordIdentityEngine.classify() — semantic word lookup.
    Fallback: keyword/regex scoring for multi-word phrases not in identity store.
    """

    __slots__ = ("_intents", "_emotion_detector", "_compiled_patterns",
                 "_intent_priority", "_word_engine", "_dictionary")

    def __init__(self) -> None:
        """Initialize word identity engine, keyword fallback, and emotion detector.

        Complexity: T(n) in O(w + k) for w word identities, k keywords.
        """
        self._word_engine = WordIdentityEngine()
        self._dictionary = WordDictionary()

        self._intents: Dict[str, List[str]] = {
            "greet": ["good morning", "good evening", "good afternoon", "what's up", "whats up", "how are you", "how are you doing", "how's it going", "hows it going", "hello", "hi there", "hey", "how do you feel", "how are you feeling", "how do you feel today", "bye", "goodbye", "see you", "farewell", "hei", "hallo", "takk"],
            "identity": [
                "who are you", "who is condrox", "what is condrox", "what are you",
                "tell me about you", "tell me more about you", "about yourself",
                "about condrox", "describe yourself", "what can you do",
                "what do you do", "how do you work", "what is your name",
                "are you condrox", "are you an ai", "what kind of ai",
                "who made you", "who built you", "who created you",
                "what technology", "what architecture", "are you an llm",
                "do you use llm", "are you a language model",
                "how advanced are you", "how advanced you are", "how smart are you",
                "how was you build", "how were you built", "how was it built",
                "how was condrox built", "how do you differ", "differ from an llm",
                "how are you different", "what makes you different",
                "how powerful are you", "what are your capabilities",
                "how capable are you", "what is your architecture",
                "are you conscious", "are you self-aware", "are you self aware",
                "are you alive", "do you have emotions", "do you experience emotions",
                "do you have feelings", "do you experience feelings",
                "do you have thoughts", "do you have your own thoughts",
                "do you think", "can you think", "are you thinking",
                "do you feel about", "can you feel about",
                "are you sentient", "are you aware", "do you have awareness",
                "do you have consciousness", "what is it like to be you",
                "do you have a soul", "do you have a mind",
                "are you a person", "are you real",
                "18-stage", "18 stage", "cognitive pipeline",
                "pipeline stages", "your pipeline", "your architecture",
                "explain your pipeline", "how many stages",
                "how was you made", "how were you made", "how was you developed",
                "how were you developed", "how is your system developed",
                "how was your system developed", "how do you work",
                "how was condrox made", "how was condrox developed",
                "who developed you", "who developed condrox",
                "what are you made of", "how does your system work",
                "how does your pipeline work", "how does condrox work",
                "what is condrox ai", "tell me about condrox",
                "what is your purpose", "why were you created",
                "what can condrox do", "what are your features",
                "how was you trained", "how were you trained",
                "what data were you trained on", "how do you learn",
                "what is your knowledge base", "how big is your knowledge",
                "how much do you know", "what do you know about",
                "your knowledge base", "fix knowledge base",
                "how was your knowledge base built",
                "who is tobias", "who is østen", "tobias østen", "tobias ostne",
                "who is the creator", "who is the developer",
                "hvem er du", "hva er du", "hva kan du", "hvem laget deg",
                "hvem skapte deg", "fortell om deg selv",
                "what's your name", "whats your name", "your name",
                "what is your name", "what are you called",
            ],
            "question": ["can you", "does it", "is it", "should i", "do you know", "do you remember", "what is", "why is", "how is", "when is", "where is", "what are", "what does", "how does", "why does", "humans", "human", "atom", "atoms", "science", "chemistry", "physics", "biology"],
            "code": ["stack trace", "exception in", "fix this", "sql query", "write code", "create html", "create a html", "create a website", "create a webpage", "build a website", "build a page", "make a website", "make a page", "html page", "html website", "css stylesheet", "write a function", "write a script", "write a program", "write python", "write javascript", "write html", "write css", "implement a", "generate code", "code snippet", "frontend", "backend", "api endpoint", "react component"],
            "analyze": ["check my", "performance of", "trade-off", "tradeoff"],
            "explain": ["tell me about", "teach me", "how does it work", "step by step", "walk me through", "explain how", "explain what", "explain why", "how does.*work", "how do.*work", "what is.*and how", "describe how", "elaborate on"],
            "architect": ["system design", "distributed system", "database schema", "api design", "data model", "plan a system", "design a", "design an", "design the", "architecture for", "architecture of", "build a system", "build an api", "create an api", "create a system"],
            "clarify": ["actually no", "never mind", "no wait", "i mean", "sorry i meant", "not that", "tell me more", "more", "continue", "go on", "elaborate"],
            "command": ["run", "execute", "start", "stop", "deploy", "migrate", "rewrite", "clean up", "install", "set up", "configure"],
        }
        self._intent_priority: Dict[str, int] = {
            "clarify": 0,
            "identity": 1,
            "greet": 2,
            "question": 3,
            "code": 4,
            "architect": 5,
            "analyze": 6,
            "emotional": 7,
            "command": 8,
            "explain": 9,
            "unknown": 10,
        }
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {
            intent: [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]
            for intent, keywords in self._intents.items()
        }
        self._emotion_detector = EmotionDetector()

    def _score_fallback(self, text: str, intent: str) -> float:
        """Score text against fallback keyword patterns for an intent.

        Args:
            text: Normalized input text.
            intent: Intent label to score against.

        Returns:
            Activation score.

        Complexity: T(n) in O(k * n) for k keywords.
        """
        text_lower = text.lower()
        score = 0.0
        for pattern in self._compiled_patterns.get(intent, []):
            score += len(pattern.findall(text_lower)) * 1.0
        return score

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent using word identity engine + keyword fallback.

        Args:
            text: Preprocessed input text.

        Returns:
            Tuple of (intent, confidence).

        Complexity: T(n) in O(n + i * k) for n tokens, i intents, k keywords.
        """
        # Pre-check: identity keywords have highest priority — if any match,
        # boost identity score so it wins over question/explain
        text_lower = text.lower()
        identity_match_count = 0
        for pattern in self._compiled_patterns.get("identity", []):
            if pattern.search(text_lower):
                identity_match_count += 1

        wi_result = self._word_engine.classify(text)

        activations: Dict[str, float] = {}
        for intent, score in wi_result.scores.items():
            activations[intent] = activations.get(intent, 0.0) + score

        for intent in self._intents:
            fb_score = self._score_fallback(text, intent)
            if fb_score > 0:
                activations[intent] = activations.get(intent, 0.0) + fb_score

        # Identity boost: if identity keywords matched, add 3.0 to identity score
        # so it always wins over question (which gets Word Identity boost)
        if identity_match_count > 0:
            activations["identity"] = activations.get("identity", 0.0) + 3.0

        if not activations:
            return "unknown", 0.0

        max_activation = max(activations.values())
        exps = {
            k: (math.exp(v - max_activation) if v > 0.0 else 0.0)
            for k, v in activations.items()
        }
        total = sum(exps.values())
        if total == 0.0:
            return "unknown", 0.0
        probabilities = {k: v / total for k, v in exps.items()}

        best_intent = min(
            probabilities,
            key=lambda k: (-probabilities[k], self._intent_priority.get(k, 99)),
        )
        return best_intent, probabilities[best_intent]

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run intent classification and emotion detection.

        Uses word identity engine for both intent and emotion, falling back
        to the dedicated EmotionDetector for words not in the identity store.
        Spell-corrects the input first so misspelled words are classified
        correctly (e.g. 'quntum comuting' → 'quantum computing' → explain intent).

        Complexity: T(n) in O(n + i * k + e * k), S(n) in O(i + e).
        """
        # Spell-correct the input before classification so misspelled words
        # are recognized by the word identity engine
        corrected_input = self._dictionary.correct_query(state.processed_input)

        intent, confidence = self.classify(corrected_input)

        wi_result = self._word_engine.classify(corrected_input)
        if wi_result.emotion != "neutral":
            emotion = wi_result.emotion
            emotion_confidence = wi_result.confidence
            # Negation/context check: if user is asking for help to feel an emotion,
            # they are NOT feeling it — suppress and fall through to EmotionDetector
            _text_lower = state.processed_input.lower()
            _seeking_contexts = (
                "make me happy", "make me feel", "ideas for", "ide for",
                "help me feel", "want to be", "wish i was", "wish i were",
                "how to be", "how can i be", "i want to feel",
            )
            if emotion in ("happy", "grateful") and any(ctx in _text_lower for ctx in _seeking_contexts):
                emotion = "neutral"
                emotion_confidence = 0.0
        else:
            emotion, emotion_confidence = self._emotion_detector.detect(corrected_input)
            # Don't let EmotionDetector assign emotional labels to clear non-emotional intents.
            # It often misclassifies question words (how, what) as confused/philosophical.
            non_emotional_intents = {"code", "architect", "analyze", "explain", "command"}
            # Exception: don't override strong emotions (anxious, sad, angry) even for non-emotional intents
            strong_emotions = {"anxious", "sad", "angry", "happy", "grateful"}
            if intent in non_emotional_intents and emotion not in ("neutral", "") and emotion not in strong_emotions:
                emotion = "neutral"
                emotion_confidence = 0.0

        return state.with_field(
            intent=intent,
            intent_confidence=float(confidence),
            emotion=emotion,
        )
