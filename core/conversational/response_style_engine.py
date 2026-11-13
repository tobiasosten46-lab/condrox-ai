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
ResponseStyle Engine — adapts response style based on intent and emotional
context, eliminating robotic phrasing and ensuring natural conversation.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """
    Immutable style profile for response formatting.

    Attributes:
        name: Profile identifier.
        tone: Tone descriptor (e.g., "concise", "detailed", "friendly").
        formality: Formality level in [0.0, 1.0].
        max_preamble: Maximum preamble words.
        sentence_style: Sentence construction style.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    tone: str
    formality: float
    max_preamble: int
    sentence_style: str


# Default style profiles per intent.
_DEFAULT_PROFILES: Dict[str, StyleProfile] = {
    "code": StyleProfile(
        name="code",
        tone="concise_technical",
        formality=0.3,
        max_preamble=0,
        sentence_style="imperative",
    ),
    "explain": StyleProfile(
        name="explain",
        tone="educational",
        formality=0.5,
        max_preamble=3,
        sentence_style="declarative",
    ),
    "question": StyleProfile(
        name="question",
        tone="direct",
        formality=0.4,
        max_preamble=0,
        sentence_style="interrogative",
    ),
    "conversation": StyleProfile(
        name="conversation",
        tone="friendly",
        formality=0.2,
        max_preamble=2,
        sentence_style="mixed",
    ),
    "diagnostic": StyleProfile(
        name="diagnostic",
        tone="analytical",
        formality=0.6,
        max_preamble=1,
        sentence_style="declarative",
    ),
}

# Robotic phrases to detect and remove.
_ROBOTIC_PHRASES: List[str] = [
    "as an ai",
    "as a language model",
    "i don't have personal",
    "i cannot have",
    "i'm not able to",
    "it's important to note",
    "it's worth noting",
    "please note that",
    "keep in mind that",
    "i'd be happy to help",
    "let me help you with that",
    "certainly!",
    "of course!",
    "absolutely!",
    "sure thing!",
    "great question!",
    "that's a great question",
    "i apologize",
    "i'm sorry, but",
    "unfortunately, i",
]

# Preambles that add no value.
_PREAMBLE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^(sure|certainly|of course|absolutely|okay|ok)[,.!]\s*", re.IGNORECASE),
    re.compile(r"^(here(?:'s| is)|let me|allow me to)\s+", re.IGNORECASE),
    re.compile(r"^(i (?:can|will|would|could)\s+(?:help|assist|provide))\s+", re.IGNORECASE),
]


class ResponseStyleEngine:
    """
    Adapts response style based on intent and removes robotic phrasing.

    Complexity:
        apply: T(n) ∈ O(n) for n words, S(n) ∈ O(n).
        detect_robotic: T(n) ∈ O(p * n) for p phrases, n text length.
    """

    __slots__ = ("_profiles", "_robotic_phrases", "_preamble_patterns")

    def __init__(
        self,
        profiles: Optional[Dict[str, StyleProfile]] = None,
        robotic_phrases: Optional[List[str]] = None,
    ) -> None:
        """Initialize with optional custom profiles and phrase lists.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._profiles = profiles if profiles is not None else _DEFAULT_PROFILES
        self._robotic_phrases = robotic_phrases if robotic_phrases is not None else _ROBOTIC_PHRASES
        self._preamble_patterns = _PREAMBLE_PATTERNS

    def get_profile(self, intent: str) -> StyleProfile:
        """Get the style profile for an intent.

        Args:
            intent: Classified intent from Stage 02.

        Returns:
            StyleProfile for the intent, or a default profile.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        profile = self._profiles.get(intent)
        if profile is not None:
            return profile
        # Fallback: try "conversation", then any profile, then a default.
        profile = self._profiles.get("conversation")
        if profile is not None:
            return profile
        if self._profiles:
            return next(iter(self._profiles.values()))
        return StyleProfile(
            name="default",
            tone="neutral",
            formality=0.5,
            max_preamble=2,
            sentence_style="declarative",
        )

    def remove_robotic_phrases(self, text: str) -> Tuple[str, List[str]]:
        """Remove robotic phrases from text.

        Args:
            text: Response text.

        Returns:
            Tuple of (cleaned_text, list_of_removed_phrases).

        Complexity: T(n) ∈ O(p * n) for p phrases, n text length.
        """
        if not text:
            return text, []

        removed: List[str] = []
        cleaned = text

        for phrase in self._robotic_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            matches = pattern.findall(cleaned)
            if matches:
                removed.extend(matches)
                cleaned = pattern.sub("", cleaned)

        # Remove preamble patterns
        for pattern in self._preamble_patterns:
            cleaned = pattern.sub("", cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        cleaned = re.sub(r"^\s*[,;:]\s*", "", cleaned).strip()

        return cleaned, removed

    def apply(self, text: str, intent: str, emotion: str = "neutral") -> str:
        """Apply style transformation to response text.

        Args:
            text: Response text to style.
            intent: Classified intent.
            emotion: Detected emotion (affects tone).

        Returns:
            Styled response text.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        if not text:
            return text

        profile = self.get_profile(intent)

        # Step 1: Remove robotic phrases
        cleaned, _ = self.remove_robotic_phrases(text)

        # Step 2: Adjust formality
        if profile.formality < 0.4:
            # Casual: contract common phrases
            cleaned = self._casualize(cleaned)
        elif profile.formality >= 0.6:
            # Formal: expand contractions
            cleaned = self._formalize(cleaned)

        # Step 3: Enforce preamble limit
        if profile.max_preamble == 0:
            cleaned = self._strip_preamble(cleaned)

        # Step 4: Emotion-based adjustment
        if emotion in ("frustrated", "angry", "confused"):
            # Be more direct when user is frustrated
            cleaned = self._strip_preamble(cleaned)

        return cleaned.strip()

    def _casualize(self, text: str) -> str:
        """Apply casual contractions.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        replacements = [
            (r"\bdo not\b", "don't"),
            (r"\bdoes not\b", "doesn't"),
            (r"\bdid not\b", "didn't"),
            (r"\bis not\b", "isn't"),
            (r"\bare not\b", "aren't"),
            (r"\bcan not\b", "can't"),
            (r"\bcannot\b", "can't"),
            (r"\bwill not\b", "won't"),
            (r"\byou are\b", "you're"),
            (r"\bit is\b", "it's"),
            (r"\bthat is\b", "that's"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _formalize(self, text: str) -> str:
        """Expand contractions for formal tone.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        replacements = [
            (r"\bdon['']t\b", "do not"),
            (r"\bdoesn['']t\b", "does not"),
            (r"\bdidn['']t\b", "did not"),
            (r"\bisn['']t\b", "is not"),
            (r"\baren['']t\b", "are not"),
            (r"\bcan['']t\b", "cannot"),
            (r"\bwon['']t\b", "will not"),
            (r"\byou['']re\b", "you are"),
            (r"\bit['']s\b", "it is"),
            (r"\bthat['']s\b", "that is"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _strip_preamble(self, text: str) -> str:
        """Remove leading preamble phrases.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        for pattern in self._preamble_patterns:
            text = pattern.sub("", text)
        return text.strip()

    def detect_robotic(self, text: str) -> List[str]:
        """Detect robotic phrases without removing them.

        Args:
            text: Text to check.

        Returns:
            List of detected robotic phrases.

        Complexity: T(n) ∈ O(p * n) for p phrases, n text length.
        """
        if not text:
            return []

        detected: List[str] = []
        lower = text.lower()
        for phrase in self._robotic_phrases:
            if phrase in lower:
                detected.append(phrase)
        return detected
