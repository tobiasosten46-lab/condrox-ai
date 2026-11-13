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
Output Stabilizer — removes repetition, loops, disclaimers, and robotic
language from final output.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True, slots=True)
class StabilizationResult:
    """
    Immutable result of output stabilization.

    Attributes:
        text: Stabilized output text.
        repetitions_removed: Number of repeated segments removed.
        disclaimers_removed: Number of disclaimers removed.
        loops_detected: Number of loops detected and broken.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    text: str
    repetitions_removed: int
    disclaimers_removed: int
    loops_detected: int


# Disclaimers to remove.
_DISCLAIMER_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"(?:always|please|note)\s+(?:check|verify|consult|refer to)\s+"
        r"(?:the\s+)?(?:official\s+)?(?:documentation|docs|manual|source)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:this\s+)?(?:information|response|answer)\s+(?:may|might|could)\s+"
        r"(?:not\s+be|be\s+(?:incomplete|outdated|inaccurate))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+)?(?:strongly\s+)?(?:recommend|suggest|advise)\s+"
        r"(?:that\s+)?you\s+(?:consult|verify|check|review)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:please\s+)?(?:note|keep\s+in\s+mind|be\s+aware)\s+that\s+"
        r"(?:this|the\s+above)\s+(?:information|response|answer)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:as\s+an\s+ai|as\s+a\s+(?:language\s+)?model|i\s+am\s+an\s+ai)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+(?:don't|do\s+not)\s+have\s+(?:personal|real|actual))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i\s+cannot\s+(?:have|experience|feel))",
        re.IGNORECASE,
    ),
]


class OutputStabilizer:
    """
    Stabilizes output by removing repetitions, loops, and disclaimers.

    Complexity:
        stabilize: T(n) ∈ O(n) for n characters, S(n) ∈ O(n).
    """

    __slots__ = ("_disclaimer_patterns", "_max_repeat_len", "_max_consecutive")

    def __init__(
        self,
        max_repeat_len: int = 50,
        max_consecutive: int = 2,
    ) -> None:
        """Initialize the stabilizer.

        Args:
            max_repeat_len: Maximum length of a repeated segment to detect.
            max_consecutive: Maximum allowed consecutive repetitions.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._disclaimer_patterns = _DISCLAIMER_PATTERNS
        self._max_repeat_len = max_repeat_len
        self._max_consecutive = max_consecutive

    def stabilize(self, text: str) -> StabilizationResult:
        """Stabilize output text.

        Args:
            text: Output text to stabilize.

        Returns:
            StabilizationResult with cleaned text and metrics.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        if not text:
            return StabilizationResult(
                text="",
                repetitions_removed=0,
                disclaimers_removed=0,
                loops_detected=0,
            )

        # Step 1: Remove disclaimers
        cleaned, disclaimers_removed = self._remove_disclaimers(text)

        # Step 2: Remove repeated sentences
        cleaned, repetitions_removed = self._remove_repeated_sentences(cleaned)

        # Step 3: Detect and break loops
        cleaned, loops_detected = self._break_loops(cleaned)

        # Step 4: Clean up whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        cleaned = cleaned.strip()

        return StabilizationResult(
            text=cleaned,
            repetitions_removed=repetitions_removed,
            disclaimers_removed=disclaimers_removed,
            loops_detected=loops_detected,
        )

    def _remove_disclaimers(self, text: str) -> Tuple[str, int]:
        """Remove disclaimer phrases.

        Complexity: T(n) ∈ O(p * n) for p patterns, n text length.
        """
        count = 0
        cleaned = text
        for pattern in self._disclaimer_patterns:
            matches = pattern.findall(cleaned)
            count += len(matches)
            cleaned = pattern.sub("", cleaned)
        # Clean up leftover punctuation
        cleaned = re.sub(r"\s*[,;:]\s*[,.!?]", ".", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(), count

    def _remove_repeated_sentences(self, text: str) -> Tuple[str, int]:
        """Remove repeated consecutive sentences.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return text, 0

        result: List[str] = []
        removed = 0
        consecutive_count = 1

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            if result and sentence.lower() == result[-1].lower():
                consecutive_count += 1
                if consecutive_count > self._max_consecutive:
                    removed += 1
                    continue
            else:
                consecutive_count = 1

            result.append(sentence)

        return " ".join(result), removed

    def _break_loops(self, text: str) -> Tuple[str, int]:
        """Detect and break repetitive loops in text.

        A loop is a pattern of 3+ repeated phrases.

        Complexity: T(n) ∈ O(n * r) for n characters, r max repeat length.
        """
        loops = 0
        cleaned = text

        # Check for repeated phrases of varying lengths
        words = cleaned.split()
        if len(words) < 6:
            return cleaned, 0

        result_words: List[str] = []
        i = 0
        while i < len(words):
            # Check if the next segment repeats
            found_loop = False
            for seg_len in range(2, min(self._max_repeat_len, len(words) - i)):
                segment = words[i : i + seg_len]
                # Check if this segment repeats immediately
                repeats = 0
                j = i + seg_len
                while j + seg_len <= len(words):
                    if words[j : j + seg_len] == segment:
                        repeats += 1
                        j += seg_len
                    else:
                        break

                if repeats >= 2:  # 3+ total occurrences = loop
                    # Keep only one instance
                    result_words.extend(segment)
                    i = j
                    loops += repeats
                    found_loop = True
                    break

            if not found_loop:
                result_words.append(words[i])
                i += 1

        return " ".join(result_words), loops

    def check_quality(self, text: str) -> List[str]:
        """Check output for quality issues without modifying.

        Args:
            text: Text to check.

        Returns:
            List of detected quality issues.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        issues: List[str] = []

        # Check for disclaimers
        for pattern in self._disclaimer_patterns:
            if pattern.search(text):
                issues.append("disclaimer_detected")

        # Check for repeated sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for i in range(1, len(sentences)):
            if sentences[i].strip().lower() == sentences[i - 1].strip().lower():
                issues.append("repeated_sentence")
                break

        # Check for word-level loops
        words = text.split()
        if len(words) >= 6:
            for i in range(len(words) - 5):
                if words[i] == words[i + 2] == words[i + 4]:
                    issues.append("word_loop")
                    break

        return issues
