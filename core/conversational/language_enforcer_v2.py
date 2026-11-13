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
LanguageEnforcer v2 — enforces persona consistency, eliminates robotic
phrases, and maintains consistent tone across responses.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class PersonaRules:
    """
    Immutable persona language rules.

    Attributes:
        name: Persona identifier.
        forbidden_phrases: Phrases that must never appear.
        required_tone: Tone descriptors to maintain.
        contraction_policy: "expand", "contract", or "neutral".
        first_person: Whether first person is allowed.
        formality: Formality level in [0.0, 1.0].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    forbidden_phrases: tuple[str, ...]
    required_tone: tuple[str, ...]
    contraction_policy: str
    first_person: bool
    formality: float


# Default Condrox persona rules.
_DEFAULT_PERSONA = PersonaRules(
    name="condrox",
    forbidden_phrases=(
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
        "i am an ai",
        "i'm an ai",
        "as a model",
        "my training data",
        "i was trained",
    ),
    required_tone=("direct", "technical", "concise"),
    contraction_policy="contract",
    first_person=True,
    formality=0.3,
)


class LanguageEnforcerV2:
    """
    Enforces persona language rules on output text.

    Complexity:
        enforce: T(n) ∈ O(n) for n characters, S(n) ∈ O(n).
    """

    __slots__ = ("_persona", "_violation_log")

    def __init__(self, persona: Optional[PersonaRules] = None) -> None:
        """Initialize with a persona.

        Args:
            persona: Persona rules; defaults to Condrox persona.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._persona = persona if persona is not None else _DEFAULT_PERSONA
        self._violation_log: List[str] = []

    def enforce(self, text: str) -> Tuple[str, List[str]]:
        """Enforce persona rules on text.

        Args:
            text: Text to enforce rules on.

        Returns:
            Tuple of (enforced_text, list_of_violations_found).

        Complexity: T(n) ∈ O(n) for n characters, S(n) ∈ O(n).
        """
        if not text:
            return text, []

        violations: List[str] = []
        enforced = text

        # Step 1: Remove forbidden phrases
        enforced, phrase_violations = self._remove_forbidden_phrases(enforced)
        violations.extend(phrase_violations)

        # Step 2: Apply contraction policy
        if self._persona.contraction_policy == "contract":
            enforced = self._contract(enforced)
        elif self._persona.contraction_policy == "expand":
            enforced = self._expand(enforced)

        # Step 3: Enforce first-person policy
        if not self._persona.first_person:
            enforced = self._remove_first_person(enforced)

        # Step 4: Clean up whitespace and punctuation
        enforced = re.sub(r"\s{2,}", " ", enforced)
        enforced = re.sub(r"^\s*[,;:]\s*", "", enforced)
        enforced = re.sub(r"\s+([,.!?;:])", r"\1", enforced)
        enforced = enforced.strip()

        self._violation_log.extend(violations)
        return enforced, violations

    def check(self, text: str) -> List[str]:
        """Check text for violations without modifying.

        Args:
            text: Text to check.

        Returns:
            List of violations found.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        if not text:
            return []

        violations: List[str] = []
        lower = text.lower()

        for phrase in self._persona.forbidden_phrases:
            if phrase in lower:
                violations.append(f"forbidden_phrase: {phrase}")

        if not self._persona.first_person:
            if re.search(r"\b(i|i'm|i've|i'll|my|mine|myself)\b", text, re.IGNORECASE):
                violations.append("first_person_used")

        return violations

    def get_violation_log(self) -> List[str]:
        """Return all violations from the last enforce call.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return list(self._violation_log)

    def clear_log(self) -> None:
        """Clear the violation log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._violation_log.clear()

    def _remove_forbidden_phrases(self, text: str) -> Tuple[str, List[str]]:
        """Remove forbidden phrases from text.

        Complexity: T(n) ∈ O(p * n) for p phrases, n text length.
        """
        violations: List[str] = []
        cleaned = text

        for phrase in self._persona.forbidden_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            matches = pattern.findall(cleaned)
            if matches:
                violations.extend(matches)
                cleaned = pattern.sub("", cleaned)

        return cleaned, violations

    def _contract(self, text: str) -> str:
        """Apply contractions for casual tone.

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
            (r"\bthey are\b", "they're"),
            (r"\bwe are\b", "we're"),
            (r"\bhas not\b", "hasn't"),
            (r"\bhave not\b", "haven't"),
            (r"\bhad not\b", "hadn't"),
            (r"\bwould not\b", "wouldn't"),
            (r"\bshould not\b", "shouldn't"),
            (r"\bcould not\b", "couldn't"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _expand(self, text: str) -> str:
        """Expand contractions for formal tone.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        replacements = [
            (r"\bdon't\b", "do not"),
            (r"\bdoesn't\b", "does not"),
            (r"\bdidn't\b", "did not"),
            (r"\bisn't\b", "is not"),
            (r"\baren't\b", "are not"),
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\byou're\b", "you are"),
            (r"\bit's\b", "it is"),
            (r"\bthat's\b", "that is"),
            (r"\bthey're\b", "they are"),
            (r"\bwe're\b", "we are"),
            (r"\bhasn't\b", "has not"),
            (r"\bhaven't\b", "have not"),
            (r"\bhadn't\b", "had not"),
            (r"\bwouldn't\b", "would not"),
            (r"\bshouldn't\b", "should not"),
            (r"\bcouldn't\b", "could not"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _remove_first_person(self, text: str) -> str:
        """Remove first-person references.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        text = re.sub(r"\bi'm\b", "the system is", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi've\b", "the system has", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'll\b", "the system will", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\b", "the system", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmy\b", "the system's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmine\b", "the system's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmyself\b", "the system", text, flags=re.IGNORECASE)
        return text
