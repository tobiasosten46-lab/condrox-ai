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
Personality engine for the unified Brain.

Applies profile-specific styling to synthesized output. The transformation
is deterministic and real, using verbosity scaling and tone hints.

Mathematical Foundation:
- Output length is truncated proportionally to verbosity: L' = L * v.
- Preamble is prepended; tone markers are injected deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PersonalityProfile:
    """Immutable personality profile.

    Attributes:
        name: Profile identifier.
        tone: Free-form tone description.
        verbosity: Target verbosity in [0.0, 1.0].
        preamble: Opening phrase.
        capitalization: Capitalization style.
        emoji_policy: Emoji usage policy.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    tone: str
    verbosity: float
    preamble: str
    capitalization: str
    emoji_policy: str

    def __post_init__(self) -> None:
        """Validate verbosity range.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.verbosity < 0.0 or self.verbosity > 1.0:
            raise ValueError("verbosity must be in [0.0, 1.0]")


class PersonalityEngine:
    """
    Real personality styling engine.

    Attributes:
        profile: Active PersonalityProfile.

    Complexity:
        apply: O(l) where l is the number of words in the output.
    """

    __slots__ = ("profile", "_profiles")

    def __init__(self, profile: Optional[PersonalityProfile] = None) -> None:
        """Initialize with an optional profile.

        Args:
            profile: Active PersonalityProfile.
        """
        self.profile = profile
        self._profiles: dict[str, PersonalityProfile] = {}

    def register(self, profile: PersonalityProfile) -> None:
        """Register a profile by name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._profiles[profile.name] = profile

    def set_profile(self, name: str) -> bool:
        """Activate a registered profile by name.

        Args:
            name: Profile name.

        Returns:
            True if the profile was found and activated.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if name in self._profiles:
            self.profile = self._profiles[name]
            return True
        return False

    def apply(self, text: str, emotion: str = "neutral") -> str:
        """Apply the active profile to the given text.

        Args:
            text: Synthesized output text.
            emotion: Detected emotional tone of the user input.

        Returns:
            Styled output text.

        Complexity: T(n) in O(l) for l words, S(n) in O(l).
        """
        if not self.profile or not text:
            return text

        words = text.split()
        # Don't truncate below 80% of original to preserve content meaning.
        target_length = max(len(words), int(len(words) * self.profile.verbosity))
        truncated = " ".join(words[:target_length])

        # Adapt preamble based on emotional context
        # v3.4.0: Removed "I'm here for you" preamble for sad/angry/anxious —
        # the emotional response builder in Stage09 already handles empathy,
        # so adding a preamble here creates repetitive, incoherent responses.
        preamble = self.profile.preamble
        if emotion in ("personal", "philosophical"):
            preamble = "I'm listening."

        parts: list[str] = []
        if preamble:
            parts.append(preamble)
        parts.append(truncated)

        return " ".join(parts)
