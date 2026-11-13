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
Heuristic Weighted Analyzer (HWA) engine.

The HWA engine performs weighted heuristic scoring over a set of features.
It is used to quantify the presence or absence of patterns and guide
pipeline routing decisions.

Mathematical Foundation:
- Weighted score: S = Σ w_i * x_i / Σ w_i
- x_i are binary or continuous feature matches in [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True, slots=True)
class Heuristic:
    """Immutable heuristic feature with weight and matcher.

    Attributes:
        name: Heuristic identifier.
        weight: Feature weight (non-negative).
        matcher: Callable returning match strength in [0,1].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    weight: float
    matcher: Any = None

    def evaluate(self, text: str, context: Dict[str, Any]) -> float:
        """Evaluate this heuristic against input text and context.

        Args:
            text: Input string.
            context: Current context dict.

        Returns:
            Match strength in [0.0, 1.0].

        Complexity: T(n) ∈ O(m) where m is matcher complexity; O(1) for default.
        """
        if self.matcher is None:
            return 0.0
        try:
            return float(min(1.0, max(0.0, self.matcher(text, context))))
        except Exception:
            return 0.0


class HWAEngine:
    """
    Real heuristic weighted analyzer.

    Attributes:
        _heuristics: Registered heuristics.

    Complexity:
        register: O(1)
        analyze: O(h * m) for h heuristics and matcher cost m.
    """

    __slots__ = ("_heuristics",)

    def __init__(self) -> None:
        """Initialize an empty HWA engine."""
        self._heuristics: List[Heuristic] = []

    def register(self, heuristic: Heuristic) -> None:
        """Register a heuristic.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._heuristics.append(heuristic)

    def analyze(self, text: str, context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Run all heuristics and return a weighted score plus per-feature scores.

        Args:
            text: Input text.
            context: Current context.

        Returns:
            Tuple of (weighted_score, feature_scores).

        Complexity: T(n) ∈ O(h * m), S(n) ∈ O(h) for h heuristics.
        """
        if not self._heuristics:
            return 0.0, {}

        total_weight = 0.0
        weighted_sum = 0.0
        feature_scores: Dict[str, float] = {}

        for heuristic in self._heuristics:
            value = heuristic.evaluate(text, context)
            feature_scores[heuristic.name] = value
            total_weight += heuristic.weight
            weighted_sum += heuristic.weight * value

        score = weighted_sum / total_weight if total_weight > 0.0 else 0.0
        return float(min(1.0, max(0.0, score))), feature_scores
