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
Meta-learning engine for the unified Brain.

Tracks performance outcomes and adjusts strategy weights using a simple,
real exponential moving average (EMA) update. No external model is needed.

Mathematical Foundation:
- EMA update: w_t = α * reward + (1 - α) * w_{t-1}
- Strategy weight is clipped to [0.0, 1.0] and renormalized over active
  strategies so weights remain a probability distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True, slots=True)
class StrategyPerformance:
    """Immutable performance record for a strategy.

    Attributes:
        strategy_name: Strategy identifier.
        attempts: Number of invocations.
        successes: Number of successful outcomes.
        ema_score: Exponential moving average score.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    strategy_name: str
    attempts: int = 0
    successes: int = 0
    ema_score: float = 0.5

    def update(self, reward: float, alpha: float = 0.3) -> "StrategyPerformance":
        """Return a new record with an EMA update.

        Args:
            reward: Reward in [0.0, 1.0].
            alpha: Learning rate for the EMA.

        Returns:
            A new StrategyPerformance instance.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        new_attempts = self.attempts + 1
        new_successes = self.successes + (1 if reward >= 0.5 else 0)
        new_ema = alpha * reward + (1.0 - alpha) * self.ema_score
        return StrategyPerformance(
            strategy_name=self.strategy_name,
            attempts=new_attempts,
            successes=new_successes,
            ema_score=float(min(1.0, max(0.0, new_ema))),
        )


class MetaLearningEngine:
    """
    Real meta-learning engine that adapts strategy weights over time.

    Attributes:
        _records: Mapping from strategy name to performance record.
        _alpha: EMA learning rate.

    Complexity:
        record_outcome: O(1)
        get_weights: O(s) for s strategies.
    """

    __slots__ = ("_records", "_alpha")

    def __init__(self, alpha: float = 0.3) -> None:
        """Initialize the meta-learning engine.

        Args:
            alpha: EMA learning rate in (0.0, 1.0].
        """
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in (0.0, 1.0]")
        self._records: Dict[str, StrategyPerformance] = {}
        self._alpha = alpha

    def record_outcome(self, strategy_name: str, reward: float) -> StrategyPerformance:
        """Record an outcome reward for a strategy.

        Args:
            strategy_name: Strategy identifier.
            reward: Outcome reward in [0.0, 1.0].

        Returns:
            Updated StrategyPerformance.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        reward = float(min(1.0, max(0.0, reward)))
        current = self._records.get(strategy_name, StrategyPerformance(strategy_name))
        updated = current.update(reward, self._alpha)
        self._records[strategy_name] = updated
        return updated

    def get_weights(self) -> Dict[str, float]:
        """Return normalized strategy weights.

        Returns:
            Mapping from strategy name to weight in [0.0, 1.0] summing to 1.0.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s) for s strategies.
        """
        if not self._records:
            return {}
        scores = {name: rec.ema_score for name, rec in self._records.items()}
        total = sum(scores.values())
        if total == 0.0:
            count = len(scores)
            return {name: 1.0 / count for name in scores}
        return {name: score / total for name, score in scores.items()}

    def get_performance(self, strategy_name: str) -> StrategyPerformance:
        """Return the performance record for a strategy.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._records.get(
            strategy_name, StrategyPerformance(strategy_name)
        )

    def list_records(self) -> List[StrategyPerformance]:
        """Return all performance records.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s).
        """
        return list(self._records.values())
