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

"""Reward System — kognitiv verdi, ikke RL.

Reward oppstår når systemet:
- validerer en fact mange ganger → Stability Boost
- oppdager en ny relasjon → Insight Boost
- oppnår klarhet → Clarity Boost
- finner en bedre strategi → Strategy Boost
- gjør et gjennombrudd → Breakthrough Boost

Reward er ikke følelser. Reward er verdi i pipeline.
Reward bygger erfaring. Erfaring bygger mening.

Complexity:
    record_reward: T(n) ∈ O(1)
    get_total_reward: T(n) ∈ O(1)
    get_reward_by_type: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Reward:
    """A single cognitive reward.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    reward_type: str       # "stability", "insight", "clarity", "strategy", "breakthrough"
    value: float           # Reward magnitude [0.0, 1.0]
    reason: str            # Why the reward was given
    context: str           # What context triggered it
    timestamp: float       # When it happened
    fact_id: str = ""      # Related fact ID (if applicable)
    domain: str = "general"  # Knowledge domain


@dataclass(slots=True)
class FactValidationRecord:
    """Tracks how many times a fact has been validated.

    When a fact is validated enough times, it earns a Stability Boost.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    fact_id: str
    subject: str
    predicate: str
    object: str
    validation_count: int = 0
    last_validated: float = field(default_factory=time.time)
    total_confidence: float = 0.0
    domains: set = field(default_factory=set)
    last_rewarded: float = 0.0  # When stability boost was last given


class RewardSystem:
    """Cognitive reward system — ikke RL, ikke dopamin, men kognitiv verdi.

    The reward system tracks cognitive value across 5 dimensions:
    1. Stability — facts validated multiple times earn trust
    2. Insight — new relationships discovered earn novelty value
    3. Clarity — unclear topics resolved earn understanding value
    4. Strategy — working strategies earn consistency value
    5. Breakthrough — major discoveries earn transformation value

    Rewards feed back into the pipeline:
    - Stability boosts increase fact confidence
    - Insight boosts trigger knowledge integration
    - Clarity boosts trigger reflection completion
    - Strategy boosts increase strategy weights
    - Breakthrough boosts trigger model restructuring

    Thread-safety: All mutations guarded by a reentrant lock.

    Complexity:
        record_reward: T(n) ∈ O(1)
        record_validation: T(n) ∈ O(1)
        check_stability_boost: T(n) ∈ O(1)
        get_total_reward: T(n) ∈ O(1)
    """

    __slots__ = (
        "_rewards",
        "_fact_validations",
        "_total_reward",
        "_reward_by_type",
        "_lock",
        "_stability_threshold",
        "_stability_cooldown",
        "_breakthrough_count",
        "_last_breakthrough",
    )

    # Thresholds for reward triggers
    _DEFAULT_STABILITY_THRESHOLD: int = 3  # Validations needed for stability boost
    _DEFAULT_STABILITY_COOLDOWN: float = 3600.0  # 1 hour between stability boosts for same fact

    def __init__(
        self,
        stability_threshold: int = _DEFAULT_STABILITY_THRESHOLD,
        stability_cooldown: float = _DEFAULT_STABILITY_COOLDOWN,
    ) -> None:
        """Initialize the reward system.

        Args:
            stability_threshold: Number of validations before stability boost.
            stability_cooldown: Seconds between stability boosts for same fact.
        """
        self._rewards: List[Reward] = []
        self._fact_validations: Dict[str, FactValidationRecord] = {}
        self._total_reward: float = 0.0
        self._reward_by_type: Dict[str, float] = {
            "stability": 0.0,
            "insight": 0.0,
            "clarity": 0.0,
            "strategy": 0.0,
            "breakthrough": 0.0,
        }
        self._lock = threading.RLock()
        self._stability_threshold = stability_threshold
        self._stability_cooldown = stability_cooldown
        self._breakthrough_count: int = 0
        self._last_breakthrough: float = 0.0

    def record_validation(
        self,
        fact_id: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        domain: str = "general",
    ) -> Optional[Reward]:
        """Record a fact validation and check for stability boost.

        When a fact is validated enough times (stability_threshold),
        a Stability Boost reward is generated.

        Args:
            fact_id: Unique fact identifier.
            subject: Fact subject.
            predicate: Fact predicate.
            object: Fact object.
            confidence: Validation confidence [0.0, 1.0].
            domain: Knowledge domain.

        Returns:
            Reward if stability boost triggered, None otherwise.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            # Get or create validation record
            record = self._fact_validations.get(fact_id)
            if record is None:
                record = FactValidationRecord(
                    fact_id=fact_id,
                    subject=subject,
                    predicate=predicate,
                    object=object,
                )
                self._fact_validations[fact_id] = record

            record.validation_count += 1
            record.total_confidence += confidence
            record.last_validated = time.time()
            record.domains.add(domain)

            # Check for stability boost
            now = time.time()
            if (record.validation_count >= self._stability_threshold
                    and (now - record.last_rewarded) > self._stability_cooldown):
                avg_confidence = record.total_confidence / record.validation_count
                reward = Reward(
                    reward_type="stability",
                    value=min(1.0, avg_confidence * 0.8 + 0.2),
                    reason=f"Fact '{subject} {predicate} {object}' validated {record.validation_count} times",
                    context=f"domain={domain}, avg_confidence={avg_confidence:.2f}",
                    timestamp=now,
                    fact_id=fact_id,
                    domain=domain,
                )
                record.last_rewarded = now
                self._add_reward(reward)
                return reward

            return None

    def record_insight(
        self,
        subject: str,
        relation: str,
        object: str,
        domain: str = "general",
        novelty_score: float = 0.5,
    ) -> Reward:
        """Record an insight — a new relationship discovered.

        Args:
            subject: Source concept.
            relation: Relationship type.
            object: Target concept.
            domain: Knowledge domain.
            novelty_score: How novel this insight is [0.0, 1.0].

        Returns:
            Insight boost reward.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            reward = Reward(
                reward_type="insight",
                value=novelty_score,
                reason=f"New relationship: {subject} --{relation}--> {object}",
                context=f"domain={domain}",
                timestamp=time.time(),
                domain=domain,
            )
            self._add_reward(reward)
            return reward

    def record_clarity(
        self,
        topic: str,
        previous_clarity: float,
        new_clarity: float,
        domain: str = "general",
    ) -> Optional[Reward]:
        """Record clarity achievement — an unclear topic became clearer.

        Args:
            topic: The topic that became clearer.
            previous_clarity: Previous clarity level [0.0, 1.0].
            new_clarity: New clarity level [0.0, 1.0].
            domain: Knowledge domain.

        Returns:
            Clarity boost reward if clarity improved significantly, None otherwise.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            improvement = new_clarity - previous_clarity
            if improvement < 0.15:  # Need at least 15% improvement
                return None

            reward = Reward(
                reward_type="clarity",
                value=min(1.0, improvement * 2),
                reason=f"Clarity improved for '{topic}': {previous_clarity:.2f} → {new_clarity:.2f}",
                context=f"domain={domain}, improvement={improvement:.2f}",
                timestamp=time.time(),
                domain=domain,
            )
            self._add_reward(reward)
            return reward

    def record_strategy_success(
        self,
        strategy_name: str,
        intent: str,
        domain: str,
        success_score: float,
    ) -> Reward:
        """Record strategy success — a strategy worked well.

        Args:
            strategy_name: Name of the strategy.
            intent: Intent being handled.
            domain: Knowledge domain.
            success_score: How well it worked [0.0, 1.0].

        Returns:
            Strategy boost reward.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            reward = Reward(
                reward_type="strategy",
                value=success_score,
                reason=f"Strategy '{strategy_name}' succeeded for intent '{intent}'",
                context=f"domain={domain}, intent={intent}",
                timestamp=time.time(),
                domain=domain,
            )
            self._add_reward(reward)
            return reward

    def record_breakthrough(
        self,
        description: str,
        domain: str,
        impact_score: float = 0.8,
    ) -> Reward:
        """Record a breakthrough — a major discovery or paradigm shift.

        Breakthroughs are rare events that fundamentally change the
        system's understanding. They trigger model restructuring.

        Args:
            description: What the breakthrough was.
            domain: Knowledge domain.
            impact_score: How impactful [0.0, 1.0].

        Returns:
            Breakthrough boost reward.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            now = time.time()
            self._breakthrough_count += 1
            self._last_breakthrough = now

            reward = Reward(
                reward_type="breakthrough",
                value=impact_score,
                reason=f"Breakthrough: {description}",
                context=f"domain={domain}, breakthrough #{self._breakthrough_count}",
                timestamp=now,
                domain=domain,
            )
            self._add_reward(reward)
            return reward

    def _add_reward(self, reward: Reward) -> None:
        """Add a reward to the system.

        Complexity: T(n) ∈ O(1).
        """
        self._rewards.append(reward)
        if len(self._rewards) > 10000:
            self._rewards = self._rewards[-10000:]
        self._total_reward += reward.value
        self._reward_by_type[reward.reward_type] = (
            self._reward_by_type.get(reward.reward_type, 0.0) + reward.value
        )

    def get_total_reward(self) -> float:
        """Get total accumulated reward.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return self._total_reward

    def get_reward_by_type(self, reward_type: str) -> float:
        """Get accumulated reward for a specific type.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return self._reward_by_type.get(reward_type, 0.0)

    def get_recent_rewards(self, count: int = 10) -> List[Reward]:
        """Get the most recent rewards.

        Complexity: T(n) ∈ O(count).
        """
        with self._lock:
            return list(reversed(self._rewards[-count:]))

    def get_fact_validation_count(self, fact_id: str) -> int:
        """Get the number of times a fact has been validated.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            record = self._fact_validations.get(fact_id)
            return record.validation_count if record else 0

    def get_stable_facts(self) -> List[str]:
        """Get fact IDs that have earned stability boosts.

        Complexity: T(n) ∈ O(f) for f validated facts.
        """
        with self._lock:
            return [
                fid for fid, record in self._fact_validations.items()
                if record.validation_count >= self._stability_threshold
            ]

    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return {
                "total_reward": self._total_reward,
                "stability": self._reward_by_type.get("stability", 0.0),
                "insight": self._reward_by_type.get("insight", 0.0),
                "clarity": self._reward_by_type.get("clarity", 0.0),
                "strategy": self._reward_by_type.get("strategy", 0.0),
                "breakthrough": self._reward_by_type.get("breakthrough", 0.0),
                "total_rewards": len(self._rewards),
                "breakthrough_count": self._breakthrough_count,
                "validated_facts": len(self._fact_validations),
                "stable_facts": len(self.get_stable_facts()),
            }

    @property
    def breakthrough_count(self) -> int:
        """Get the number of breakthroughs achieved."""
        with self._lock:
            return self._breakthrough_count

    @property
    def total_rewards(self) -> int:
        """Get the total number of rewards given."""
        with self._lock:
            return len(self._rewards)
