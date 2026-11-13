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

"""Meta-Learning System — learns from its own performance to improve.

The meta-learning system tracks how well the AI performs on different types
of questions and adjusts its strategy accordingly. It learns:
- Which reasoning strategies work best for which intents
- Which knowledge domains need more learning
- Which response styles lead to better user engagement
- When to use deep reasoning vs quick responses
- How to improve knowledge extraction accuracy

This is NOT an LLM — it uses exponential moving averages (EMA) and
reinforcement-style updates to adapt over time.

Complexity:
    record_outcome: T(n) ∈ O(1)
    get_strategy_weights: T(n) ∈ O(s) for s strategies
    suggest_improvement: T(n) ∈ O(d) for d domains

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class LearningOutcome:
    """Record of a single learning outcome.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    strategy: str            # Which strategy was used
    intent: str              # What intent was being handled
    domain: str              # What knowledge domain
    reward: float            # How well it went [0.0, 1.0]
    response_time_ms: float  # How long it took
    user_continued: bool     # Did the user continue the conversation?
    timestamp: float         # When it happened


@dataclass(frozen=True, slots=True)
class StrategyWeight:
    """Weight for a strategy in a given context.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    strategy: str
    context: str             # Intent or domain context
    weight: float            # Current weight [0.0, 1.0]
    attempts: int            # Total attempts
    ema_reward: float        # Exponential moving average of reward
    last_updated: float      # Last update timestamp


@dataclass(slots=True)
class DomainLearningProgress:
    """Tracks learning progress for a knowledge domain.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    domain: str
    facts_learned: int = 0
    concepts_learned: int = 0
    avg_confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)
    needs_more_learning: bool = True


class MetaLearningSystem:
    """Meta-learning system that adapts strategy weights over time.

    Uses EMA-based updates to track which strategies work best:
    - w_t = α * reward + (1 - α) * w_{t-1}

    Where α (alpha) is the learning rate. Higher alpha = faster adaptation
    but more noise. Lower alpha = slower but more stable.

    Complexity:
        record_outcome: T(n) ∈ O(1)
        get_weights: T(n) ∈ O(s) for s strategies
        get_domain_progress: T(n) ∈ O(1)
    """

    __slots__ = (
        "_alpha",
        "_outcomes",
        "_strategy_weights",
        "_domain_progress",
        "_intent_performance",
        "_response_time_history",
        "_max_outcomes",
    )

    def __init__(
        self,
        alpha: float = 0.15,
        max_outcomes: int = 10000,
    ) -> None:
        """Initialize the meta-learning system.

        Args:
            alpha: EMA learning rate [0.0, 1.0]. Default 0.15.
            max_outcomes: Maximum outcomes to keep in memory.
        """
        self._alpha = alpha
        self._outcomes: List[LearningOutcome] = []
        self._strategy_weights: Dict[str, StrategyWeight] = {}
        self._domain_progress: Dict[str, DomainLearningProgress] = {}
        self._intent_performance: Dict[str, List[float]] = defaultdict(list)
        self._response_time_history: Dict[str, List[float]] = defaultdict(list)
        self._max_outcomes = max_outcomes

    def record_outcome(
        self,
        strategy: str,
        intent: str,
        domain: str,
        reward: float,
        response_time_ms: float = 0.0,
        user_continued: bool = False,
    ) -> None:
        """Record a learning outcome and update strategy weights.

        Args:
            strategy: The strategy that was used (e.g., "deep_reasoning", "quick_response").
            intent: The intent being handled.
            domain: The knowledge domain.
            reward: How well it went [0.0, 1.0].
            response_time_ms: Response time in milliseconds.
            user_continued: Whether the user continued the conversation.

        Complexity: T(n) ∈ O(1).
        """
        ts = time.time()
        outcome = LearningOutcome(
            strategy=strategy,
            intent=intent,
            domain=domain,
            reward=max(0.0, min(1.0, reward)),
            response_time_ms=response_time_ms,
            user_continued=user_continued,
            timestamp=ts,
        )
        self._outcomes.append(outcome)
        if len(self._outcomes) > self._max_outcomes:
            self._outcomes = self._outcomes[-self._max_outcomes:]

        # Update strategy weight with EMA
        context = f"{intent}:{domain}"
        key = f"{strategy}:{context}"
        existing = self._strategy_weights.get(key)
        if existing:
            new_ema = self._alpha * reward + (1 - self._alpha) * existing.ema_reward
            self._strategy_weights[key] = StrategyWeight(
                strategy=strategy,
                context=context,
                weight=new_ema,
                attempts=existing.attempts + 1,
                ema_reward=new_ema,
                last_updated=ts,
            )
        else:
            self._strategy_weights[key] = StrategyWeight(
                strategy=strategy,
                context=context,
                weight=reward,
                attempts=1,
                ema_reward=reward,
                last_updated=ts,
            )

        # Update intent performance
        self._intent_performance[intent].append(reward)
        if len(self._intent_performance[intent]) > 100:
            self._intent_performance[intent] = self._intent_performance[intent][-100:]

        # Update response time history
        if response_time_ms > 0:
            self._response_time_history[strategy].append(response_time_ms)
            if len(self._response_time_history[strategy]) > 100:
                self._response_time_history[strategy] = self._response_time_history[strategy][-100:]

    def record_learning(
        self,
        domain: str,
        facts_learned: int = 0,
        concepts_learned: int = 0,
        avg_confidence: float = 0.0,
    ) -> None:
        """Record that knowledge was learned in a domain.

        Complexity: T(n) ∈ O(1).
        """
        existing = self._domain_progress.get(domain)
        if existing:
            existing.facts_learned += facts_learned
            existing.concepts_learned += concepts_learned
            # Update average confidence with EMA
            if avg_confidence > 0:
                existing.avg_confidence = (
                    self._alpha * avg_confidence + (1 - self._alpha) * existing.avg_confidence
                )
            existing.last_updated = time.time()
            # A domain needs more learning if it has few facts or low confidence
            existing.needs_more_learning = (
                existing.facts_learned < 50 or existing.avg_confidence < 0.7
            )
        else:
            self._domain_progress[domain] = DomainLearningProgress(
                domain=domain,
                facts_learned=facts_learned,
                concepts_learned=concepts_learned,
                avg_confidence=avg_confidence,
                needs_more_learning=facts_learned < 50 or avg_confidence < 0.7,
            )

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights.

        Returns a mapping of "strategy:context" → weight.

        Complexity: T(n) ∈ O(s) for s strategies.
        """
        return {k: v.weight for k, v in self._strategy_weights.items()}

    def get_best_strategy(self, intent: str, domain: str) -> str:
        """Get the best strategy for a given intent and domain.

        Complexity: T(n) ∈ O(s) for s strategies.
        """
        context = f"{intent}:{domain}"
        best_strategy = "default"
        best_weight = -1.0

        for key, sw in self._strategy_weights.items():
            if sw.context == context and sw.weight > best_weight:
                best_weight = sw.weight
                best_strategy = sw.strategy

        # If no specific strategy found, try intent-only context
        if best_strategy == "default":
            for key, sw in self._strategy_weights.items():
                if sw.context.startswith(f"{intent}:") and sw.weight > best_weight:
                    best_weight = sw.weight
                    best_strategy = sw.strategy

        return best_strategy

    def get_domain_progress(self, domain: str) -> DomainLearningProgress:
        """Get learning progress for a domain.

        Complexity: T(n) ∈ O(1).
        """
        return self._domain_progress.get(domain, DomainLearningProgress(domain=domain))

    def get_weak_domains(self, threshold: int = 30) -> List[str]:
        """Get domains that need more learning.

        Complexity: T(n) ∈ O(d) for d domains.
        """
        return [
            d.domain for d in self._domain_progress.values()
            if d.needs_more_learning and d.facts_learned < threshold
        ]

    def suggest_improvement(self) -> str:
        """Suggest an improvement to the learning system.

        Complexity: T(n) ∈ O(d + s) for d domains, s strategies.
        """
        weak_domains = self.get_weak_domains()
        if weak_domains:
            return (
                f"Knowledge in domains {', '.join(weak_domains[:3])} is thin. "
                f"Focus learning on these areas to improve response quality."
            )

        # Find worst-performing strategy
        worst_strategy = None
        worst_weight = 1.0
        for key, sw in self._strategy_weights.items():
            if sw.attempts >= 3 and sw.weight < worst_weight:
                worst_weight = sw.weight
                worst_strategy = sw

        if worst_strategy and worst_weight < 0.4:
            return (
                f"Strategy '{worst_strategy.strategy}' for '{worst_strategy.context}' "
                f"is performing poorly (weight={worst_weight:.2f}). "
                f"Consider using a different strategy for this context."
            )

        # Find slow strategies
        for strategy, times in self._response_time_history.items():
            if len(times) >= 5:
                avg_time = sum(times) / len(times)
                if avg_time > 5000:  # > 5 seconds
                    return (
                        f"Strategy '{strategy}' is slow (avg {avg_time:.0f}ms). "
                        f"Consider optimizing or using a faster strategy."
                    )

        return "All strategies performing well. Continue current learning approach."

    def get_intent_performance(self, intent: str) -> float:
        """Get average performance for an intent.

        Complexity: T(n) ∈ O(1).
        """
        rewards = self._intent_performance.get(intent, [])
        if not rewards:
            return 0.5
        return sum(rewards) / len(rewards)

    @property
    def total_outcomes(self) -> int:
        return len(self._outcomes)

    @property
    def domains_tracked(self) -> int:
        return len(self._domain_progress)

    @property
    def strategies_tracked(self) -> int:
        return len(self._strategy_weights)
