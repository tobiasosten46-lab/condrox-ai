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
StateQuery Handler — captures state verbs and converts them to structured
state-actions so downstream agents do not treat them as search queries.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class StateQueryResult:
    """
    Immutable result of state-query detection.

    Attributes:
        is_state_query: True if the input contains a state verb.
        action: The detected action class (e.g. "improve", "fix").
        target: The object of the state verb (e.g. "this code").
        confidence: Detection confidence in [0, 1].
        fallback_intent: Original intent if not a state query.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    is_state_query: bool
    action: str
    target: str
    confidence: float
    fallback_intent: str


# Default state verb map: action → list of trigger phrases.
_DEFAULT_STATE_VERBS: Dict[str, List[str]] = {
    "improve": ["improve", "enhance", "optimize", "make better", "refactor"],
    "fix": ["fix", "repair", "correct", "debug", "resolve"],
    "upgrade": ["upgrade", "update", "modernize", "migrate"],
    "confirm": ["confirm", "verify", "validate", "check", "test"],
    "diagnostic": ["diagnose", "analyze error", "what is wrong", "troubleshoot"],
}


class StateQueryHandler:
    """
    Detects state verbs in user input and returns a structured result.

    Complexity:
        handle: T(n) ∈ O(v * k * n) for v verb classes, k keywords, n text length.
                S(n) ∈ O(1).
    """

    __slots__ = ("_state_verbs", "_compiled")

    def __init__(self, state_verbs: Optional[Dict[str, List[str]]] = None) -> None:
        """Initialize with optional custom verb map.

        Args:
            state_verbs: Mapping from action name to list of trigger phrases.
                         Defaults are used if None.

        Complexity: T(n) ∈ O(v * k), S(n) ∈ O(v * k).
        """
        self._state_verbs = state_verbs if state_verbs is not None else _DEFAULT_STATE_VERBS
        # Pre-compile regex patterns for each trigger phrase.
        self._compiled: List[Tuple[str, re.Pattern[str], float]] = []
        for action, triggers in self._state_verbs.items():
            for i, trigger in enumerate(triggers):
                # Longer triggers get higher confidence.
                confidence = 0.5 + 0.5 * (len(triggers) - i) / len(triggers)
                pattern = re.compile(re.escape(trigger), re.IGNORECASE)
                self._compiled.append((action, pattern, confidence))

    def handle(self, processed_input: str, intent: str) -> StateQueryResult:
        """Detect state verbs and return a structured result.

        Args:
            processed_input: Preprocessed user text.
            intent: The classified intent from Stage 02.

        Returns:
            StateQueryResult with detection outcome.

        Complexity: T(n) ∈ O(v * k * n), S(n) ∈ O(1).
        """
        if not processed_input:
            return StateQueryResult(
                is_state_query=False,
                action="",
                target="",
                confidence=0.0,
                fallback_intent=intent,
            )

        best_action = ""
        best_confidence = 0.0
        match_start = -1
        match_end = -1

        for action, pattern, confidence in self._compiled:
            match = pattern.search(processed_input)
            if match and confidence > best_confidence:
                best_action = action
                best_confidence = confidence
                match_start = match.start()
                match_end = match.end()

        if not best_action:
            return StateQueryResult(
                is_state_query=False,
                action="",
                target="",
                confidence=0.0,
                fallback_intent=intent,
            )

        # Extract target: text after the verb phrase.
        target = processed_input[match_end:].strip().strip(".!?;,:")
        if not target:
            target = processed_input[:match_start].strip().strip(".!?;,:")
        if not target:
            target = "unspecified"

        return StateQueryResult(
            is_state_query=True,
            action=best_action,
            target=target[:200],  # cap target length
            confidence=best_confidence,
            fallback_intent=intent,
        )
