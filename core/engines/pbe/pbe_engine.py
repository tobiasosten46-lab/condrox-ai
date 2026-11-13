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
Pattern-Based Engine (PBE) for the unified Brain.

Provides real regex and literal pattern matching with extraction and
scoring. The PBE is deterministic and supports named capture groups.

Mathematical Foundation:
- Pattern activation uses linear matching over the input of length n.
- Match score is binary per pattern: 1.0 if matched, 0.0 otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Pattern:
    """Immutable pattern definition.

    Attributes:
        name: Pattern identifier.
        regex: Compiled or string regular expression.
        weight: Pattern weight (non-negative).

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    regex: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the regex compiles.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        try:
            re.compile(self.regex)
        except re.error as exc:
            raise ValueError(f"Invalid regex for {self.name}: {self.regex}") from exc

    def compiled(self) -> re.Pattern[str]:
        """Return a compiled regex pattern.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return re.compile(self.regex, re.IGNORECASE)


class PBEEngine:
    """
    Real pattern-based extraction and scoring engine.

    Attributes:
        _patterns: Registered patterns.

    Complexity:
        register: O(1)
        match: O(p * n) for p patterns and input length n.
        extract: O(p * n) plus extraction cost.
    """

    __slots__ = ("_patterns",)

    def __init__(self) -> None:
        """Initialize an empty PBE engine."""
        self._patterns: List[Pattern] = []

    def register(self, pattern: Pattern) -> None:
        """Register a pattern.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._patterns.append(pattern)

    def match(self, text: str) -> Tuple[float, Dict[str, bool]]:
        """Score the input against all patterns.

        Args:
            text: Input string.

        Returns:
            Tuple of (weighted_score, pattern_hit_dict).

        Complexity: T(n) ∈ O(p * n), S(n) ∈ O(p) for p patterns.
        """
        if not self._patterns or not text:
            return 0.0, {}

        total_weight = 0.0
        weighted_sum = 0.0
        hits: Dict[str, bool] = {}

        for pattern in self._patterns:
            matched = bool(pattern.compiled().search(text))
            hits[pattern.name] = matched
            total_weight += pattern.weight
            weighted_sum += pattern.weight * (1.0 if matched else 0.0)

        score = weighted_sum / total_weight if total_weight > 0.0 else 0.0
        return float(min(1.0, max(0.0, score))), hits

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract all named or positional groups from matching patterns.

        Args:
            text: Input string.

        Returns:
            Mapping from pattern name to list of matched strings.

        Complexity: T(n) ∈ O(p * n + g) where g is total group count.
        """
        result: Dict[str, List[str]] = {}
        if not text:
            return result

        for pattern in self._patterns:
            matches = pattern.compiled().findall(text)
            if matches:
                flat: List[str] = []
                for match in matches:
                    if isinstance(match, tuple):
                        flat.extend(str(part) for part in match if part)
                    else:
                        flat.append(str(match))
                if flat:
                    result[pattern.name] = flat
        return result

    def first_match(self, text: str, pattern_name: str) -> Optional[re.Match[str]]:
        """Return the first match of a named pattern.

        Args:
            text: Input string.
            pattern_name: Registered pattern name.

        Returns:
            The first match or None.

        Complexity: T(n) ∈ O(n) for the matching operation.
        """
        for pattern in self._patterns:
            if pattern.name == pattern_name:
                return pattern.compiled().search(text)
        return None
