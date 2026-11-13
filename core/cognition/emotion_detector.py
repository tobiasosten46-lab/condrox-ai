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
Emotion detector for conversational intelligence.

Detects the emotional tone of user input using keyword scoring.
Supports 10 emotional categories: happy, sad, angry, anxious, grateful,
confused, curious, philosophical, personal, and neutral.

Mathematical Foundation:
- Raw activation a_i = sum(keyword_matches) for emotion i.
- Emotion selected by argmax(a_i). Ties broken by priority order.
- Complexity: T(n) in O(e * k * n) for e emotions, k keywords, n text length.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# Emotion keyword sets — curated for high-precision detection
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "sad": [
        "lost", "lose", "died", "death", "passed away", "funeral",
        "depression", "depressed",
        "lonely", "alone", "miss", "missing", "cry", "crying",
        "tears", "heartbroken", "devastated", "broken",
        "failure", "failed", "disappointed", "hurt", "pain",
        "suffering", "suffer", "grief", "mourning", "sad",
        "unhappy", "miserable", "hopeless", "empty", "invisible",
        "divorce", "breakup", "ended", "over", "gone",
        "anniversary", "grave",
        "not happy", "don't feel good", "feeling down", "down today",
        "bad day", "rough day", "tough day", "struggling",
        "iam sad", "i am sad", "i'm sad", "feel sad",
        "i have cancer", "diagnosed with cancer",
        "my cancer", "has cancer", "got cancer",
    ],
    "angry": [
        "angry", "furious", "mad", "rage", "outraged",
        "frustrated", "annoyed", "irritated", "fed up", "sick of",
        "exploitation", "unacceptable", "robbery", "stupid",
        "hate", "hated", "lying", "lied", "liar",
        "credit", "stole", "stolen", "scratched", "ruined",
        "crashing", "crashed", "broken", "not fair", "wrong",
        "exploitation", "scam", "rip off", "ripoff",
        "ignored", "ignoring", "disrespect", "disrespected",
    ],
    "happy": [
        "happy", "excited", "thrilled", "joy", "joyful",
        "amazing", "wonderful", "fantastic", "great", "awesome",
        "promoted", "promotion", "passed", "finished", "completed",
        "accepted", "dream", "best day", "best feeling",
        "engaged", "engagement", "wedding", "married",
        "first steps", "graduated", "graduation", "won", "winner",
        "marathon", "accomplished", "achievement", "proud",
        "blooming", "beautiful", "love it", "love this",
        "can't believe it", "best day ever", "cute",
    ],
    "anxious": [
        "scared", "afraid", "fear", "worried", "worry",
        "anxious", "anxiety", "panic", "nervous", "terrified",
        "freaking out", "what if", "can't stop", "racing",
        "can't breathe", "tight chest", "can't sleep",
        "second-guessing", "uncertain", "uncertainty",
        "panic attack", "phobia", "dread", "impending",
    ],
    "grateful": [
        "thank", "thanks", "thankful", "grateful", "gratitude",
        "appreciate", "appreciated", "appreciation",
        "lucky", "blessed", "fortunate", "support",
        "supported", "believes in me", "believed in me",
        "means everything", "meant a lot", "made a difference",
        "rock", "there for me", "helped me",
    ],
    "confused": [
        "confused", "confusing", "don't understand", "doesn't make sense",
        "don't get it", "don't know", "lost", "unclear",
        "more confused", "overwhelming", "too complicated",
        "wait", "but wait", "not sure", "unsure",
    ],
    "curious": [
        "how", "what", "why", "wonder", "wondering",
        "curious", "question", "tell me about",
        "learn", "learning", "understand", "explain",
        "teach", "show me", "interesting", "fascinating",
        "did you know", "have you heard", "what if",
        "how come", "what's the difference",
    ],
    "philosophical": [
        "meaning of life", "free will", "consciousness",
        "purpose", "existence", "exist", "universe",
        "reality", "truth", "nature of", "inherent",
        "subjective", "objective", "morality", "moral",
        "right or wrong", "after we die", "beyond this life",
        "soul", "spirit", "essence", "being",
        "cosmic", "simulation", "determinism", "predetermined",
        "two sides", "same coin", "imposing order",
        "human invention", "truly understand",
    ],
    "personal": [
        "my life", "my career", "my relationship",
        "my partner", "my husband", "my wife", "my kids",
        "my children", "my family", "my mother", "my father",
        "my mom", "my dad", "my friend", "my health",
        "i'm thinking",
        "i've been", "i feel", "i don't know",
        "should i", "am i crazy", "what if i",
        "my body", "my mind", "my future",
        "i'm 50", "i'm 35",
        "starting my own", "quitting my job",
        "sober", "therapy", "therapist", "estranged",
    ],
}

# Priority order for tie-breaking (most specific emotions first)
EMOTION_PRIORITY: List[str] = [
    "sad", "angry", "anxious", "grateful", "personal",
    "confused", "happy", "philosophical", "curious", "neutral",
]


class EmotionDetector:
    """Detect emotional tone from text using keyword scoring.

    Complexity:
        detect: T(n) in O(e * k * n) for e emotions, k keywords, n text length.
                S(n) in O(e) for the score map.
    """

    __slots__ = ("_keywords", "_priority")

    def __init__(self) -> None:
        """Initialize with curated keyword sets."""
        self._keywords = EMOTION_KEYWORDS
        self._priority = EMOTION_PRIORITY

    def _score(self, text_lower: str, keywords: List[str]) -> int:
        """Count keyword matches in text.

        Uses word-boundary matching for single-word keywords to prevent
        false positives (e.g. "mad" matching "made"). Multi-word phrases
        use substring matching as before.

        Args:
            text_lower: Lowercase input text.
            keywords: Keywords for an emotion.

        Returns:
            Match count.

        Complexity: T(n) in O(k * n) for k keywords.
        """
        import re as _re
        score = 0
        for kw in keywords:
            if " " in kw:
                # Multi-word phrase: substring match
                if kw in text_lower:
                    score += 1
            else:
                # Single word: word-boundary match to avoid false positives
                # (e.g. "mad" in "made", "stole" in "stolen")
                if _re.search(r'\b' + _re.escape(kw) + r'\b', text_lower):
                    score += 1
        return score

    def detect(self, text: str) -> Tuple[str, float]:
        """Detect the dominant emotion in text.

        Args:
            text: Input text to analyze.

        Returns:
            Tuple of (emotion, confidence) where confidence is
            the proportion of the top emotion's score relative to
            all emotion scores.

        Complexity: T(n) in O(e * k * n), S(n) in O(e).
        """
        if not text or not text.strip():
            return "neutral", 0.0

        text_lower = text.lower()

        # Negation/context check: if user is asking for help to feel an emotion,
        # they are NOT feeling it — suppress that emotion's score
        _seeking_contexts = (
            "make me happy", "make me feel", "ideas for", "ide for",
            "help me feel", "want to be", "wish i was", "wish i were",
            "how to be", "how can i be", "i want to feel",
        )
        _suppressed_emotions: set[str] = set()
        for ctx in _seeking_contexts:
            if ctx in text_lower:
                _suppressed_emotions.add("happy")
                _suppressed_emotions.add("grateful")
                break

        scores: Dict[str, int] = {}

        for emotion, keywords in self._keywords.items():
            raw_score = self._score(text_lower, keywords)
            if emotion in _suppressed_emotions:
                raw_score = 0  # Suppress emotion that user is seeking, not feeling
            scores[emotion] = raw_score

        # Find best emotion by score, then by priority
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return "neutral", 0.0

        # Get all emotions with the max score
        candidates = [e for e, s in scores.items() if s == max_score]

        # Sort by priority (lower index = higher priority)
        best = min(candidates, key=lambda e: self._priority.index(e) if e in self._priority else len(self._priority))

        # Confidence = top score / total score
        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.0

        return best, confidence

    def __call__(self, text: str) -> Tuple[str, float]:
        """Alias for detect()."""
        return self.detect(text)
