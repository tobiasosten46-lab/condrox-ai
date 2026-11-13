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
Strategy manager for the unified Brain.

Selects processing strategy based on intent, mode, and domain confidence.
Uses a deterministic scoring function over registered strategies.

Mathematical Foundation:
- Strategy score: s = Σ w_i * f_i where f_i are feature matches in [0,1].
- Selection: argmax_s(s) with deterministic tie-breaking by strategy name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class Strategy:
    """Immutable processing strategy.

    Attributes:
        name: Unique strategy name.
        features: Feature weights dict (e.g., {"intent=code": 1.0}).
        pipeline_overrides: Optional stage overrides.

    Complexity: T(n) ∈ O(f) for f features, S(n) ∈ O(f).
    """

    name: str
    features: Dict[str, float]
    pipeline_overrides: Dict[str, Any] = field(default_factory=dict)

    def score(self, context: Dict[str, Any]) -> float:
        """Compute strategy score against a context.

        Args:
            context: Current context dictionary.

        Returns:
            Weighted score in [0.0, 1.0].

        Complexity: T(n) ∈ O(f) for f features, S(n) ∈ O(1).
        """
        if not self.features:
            return 0.0

        total = 0.0
        matched = 0.0
        for key, weight in self.features.items():
            total += weight
            value = context.get(key, 0.0)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                value = 1.0 if value else 0.0
            matched += weight * min(1.0, max(0.0, value))

        return matched / total if total > 0.0 else 0.0


class StrategyManager:
    """
    Manages and selects processing strategies.

    Attributes:
        _strategies: Registered strategies.

    Complexity:
        register: O(1)
        select: O(s * f) for s strategies and f features.
    """

    __slots__ = ("_strategies",)

    def __init__(self) -> None:
        """Initialize with default strategies."""
        self._strategies: List[Strategy] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in strategies.

        Complexity: O(1) because the set is fixed and small.
        """
        self.register(
            Strategy(
                "code",
                {"intent=code": 1.0, "mode=code": 1.0},
            )
        )
        self.register(
            Strategy(
                "analyze",
                {"intent=analyze": 1.0, "mode=analyze": 1.0},
            )
        )
        self.register(
            Strategy(
                "explain",
                {"intent=explain": 1.0, "mode=explain": 1.0},
            )
        )
        self.register(
            Strategy(
                "general",
                {"intent=question": 0.5, "intent=unknown": 0.5},
            )
        )

    def register(self, strategy: Strategy) -> None:
        """Register a strategy.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._strategies.append(strategy)

    def select(self, context: Dict[str, Any]) -> Optional[Strategy]:
        """Select the best strategy for the given context.

        Args:
            context: Current state context.

        Returns:
            The highest-scoring strategy or None if no strategies exist.

        Complexity: T(n) ∈ O(s * f), S(n) ∈ O(1).
        """
        if not self._strategies:
            return None
        scored = [(strategy, strategy.score(context)) for strategy in self._strategies]
        scored.sort(key=lambda x: (-x[1], x[0].name))
        return scored[0][0]

    def list_strategies(self) -> List[Strategy]:
        """Return all registered strategies.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s).
        """
        return list(self._strategies)
