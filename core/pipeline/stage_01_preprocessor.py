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
Stage 01: Preprocessor.

Normalizes raw input: trims whitespace, collapses repeated whitespace,
removes non-printable control characters, and validates length.

Mathematical Foundation:
- Normalization is a linear pass over the input: T(n) ∈ O(n), S(n) ∈ O(n).
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage01Preprocessor:
    """Real input preprocessor."""

    __slots__ = ("_spell_corrections",)

    # Language normalization rules — spell corrections and domain hints.
    # Applied as word-boundary substitutions before whitespace normalization.
    # T(n) ∈ O(r * n) for r rules, n text length.
    _SPELL_CORRECTIONS: tuple[tuple[str, str], ...] = (
        # Programming misspellings
        (r"\bpyhton\b", "python"),
        (r"\bpyton\b", "python"),
        (r"\bjavscript\b", "javascript"),
        (r"\bjavascrpt\b", "javascript"),
        (r"\bjavascrip\b", "javascript"),
        (r"\bprogrammign\b", "programming"),
        (r"\bprograming\b", "programming"),
        (r"\balgorithim\b", "algorithm"),
        (r"\balgoritm\b", "algorithm"),
        (r"\bfunciton\b", "function"),
        (r"\bfuntion\b", "function"),
        (r"\bvarible\b", "variable"),
        (r"\bvariabel\b", "variable"),
        # AI/ML misspellings
        (r"\bknowladge\b", "knowledge"),
        (r"\bknowlege\b", "knowledge"),
        (r"\bknwoledge\b", "knowledge"),
        (r"\binteligence\b", "intelligence"),
        (r"\bintellignce\b", "intelligence"),
        (r"\bneural\b", "neural"),
        (r"\bmachin\b", "machine"),
        (r"\blearnign\b", "learning"),
        # Security misspellings
        (r"\bsecuity\b", "security"),
        (r"\bsecurty\b", "security"),
        (r"\bencrytion\b", "encryption"),
        (r"\bauthenication\b", "authentication"),
        (r"\bvulnerabilty\b", "vulnerability"),
        (r"\bvulnurability\b", "vulnerability"),
        # Finance misspellings
        (r"\bbankign\b", "banking"),
        (r"\bacounting\b", "accounting"),
        (r"\baccoutning\b", "accounting"),
        (r"\bfinacne\b", "finance"),
        (r"\bfinnce\b", "finance"),
        # Contraction / chat-slang expansion (critical for emotion detection)
        (r"\biam\b", "i am"),
        (r"\bim\b", "i am"),
        (r"\bive\b", "i have"),
        (r"\bid\b", "i would"),
        (r"\bill\b", "i will"),
        (r"\bdont\b", "do not"),
        (r"\bdont\b", "do not"),
        (r"\bcant\b", "cannot"),
        (r"\bwont\b", "will not"),
        (r"\bshouldnt\b", "should not"),
        (r"\bcouldnt\b", "could not"),
        (r"\bwouldnt\b", "would not"),
        (r"\bisnt\b", "is not"),
        (r"\barent\b", "are not"),
        (r"\bwasnt\b", "was not"),
        (r"\bwerent\b", "were not"),
        (r"\bhasnt\b", "has not"),
        (r"\bhavent\b", "have not"),
        (r"\bhadnt\b", "had not"),
        (r"\bdoesnt\b", "does not"),
        (r"\bdidnt\b", "did not"),
        (r"\bcouldve\b", "could have"),
        (r"\bwouldve\b", "would have"),
        (r"\bshouldve\b", "should have"),
        (r"\byoure\b", "you are"),
        (r"\btheyre\b", "they are"),
        (r"\bweve\b", "we have"),
        (r"\bwhats\b", "what is"),
        (r"\bhows\b", "how is"),
        (r"\bthats\b", "that is"),
        (r"\bwheres\b", "where is"),
        (r"\bwhos\b", "who is"),
        (r"\bwhys\b", "why is"),
        (r"\bhows\b", "how is"),
        # Common abbreviations
        (r"\bu\b", "you"),
        (r"\bur\b", "your"),
        (r"\br\b", "are"),
        (r"\bn\b", "and"),
        (r"\bthru\b", "through"),
        (r"\btho\b", "though"),
        (r"\bpls\b", "please"),
        (r"\bplz\b", "please"),
        (r"\bthx\b", "thanks"),
        (r"\bty\b", "thank you"),
        (r"\bbtw\b", "by the way"),
        (r"\bimo\b", "in my opinion"),
        (r"\bidk\b", "i do not know"),
        (r"\bgonna\b", "going to"),
        (r"\bwanna\b", "want to"),
        (r"\bgotta\b", "got to"),
        (r"\bkinda\b", "kind of"),
        (r"\bsorta\b", "sort of"),
        (r"\byeah\b", "yes"),
        (r"\byep\b", "yes"),
        (r"\bnope\b", "no"),
        (r"\bok\b", "okay"),
    )

    def __init__(self) -> None:
        """Initialize with pre-compiled spell correction patterns."""
        self._spell_corrections = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self._SPELL_CORRECTIONS
        ]

    def normalize(self, text: str, max_length: int) -> str:
        """Normalize raw input text.

        Applies spell corrections, trims whitespace, collapses repeated
        whitespace, removes non-printable control characters, and validates
        length.

        Args:
            text: Raw input.
            max_length: Maximum allowed length.

        Returns:
            Normalized string.

        Complexity: T(n) ∈ O(r * n) for r spell rules, n text length.
        """
        text = text.strip()
        text = "".join(
            char for char in text if unicodedata.category(char)[0] != "C" or char in {"\n", "\t"}
        )
        # Apply language normalization (spell corrections)
        for pattern, replacement in self._spell_corrections:
            text = pattern.sub(replacement, text)
        text = re.sub(r"\s+", " ", text)
        return text[:max_length]

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run the preprocessor stage.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        if not state.raw_input:
            return state.add_error("empty_input")

        max_len = controller.config.max_input_length
        if len(state.raw_input) > max_len:
            return state.add_error(f"input_exceeds_max_length:{len(state.raw_input)}")

        processed = self.normalize(state.raw_input, max_len)
        return state.with_field(processed_input=processed)
