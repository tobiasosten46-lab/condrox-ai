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
Conversational Priority Engine — assigns priority scores to conversation
elements to manage processing order and resource allocation.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PriorityTier(Enum):
    """Priority tiers for conversation elements."""

    CRITICAL = 0  # Safety, errors, aborts
    HIGH = 1      # Urgent questions, corrections
    NORMAL = 2    # Standard conversation
    LOW = 3       # Background tasks, context
    DEFERRED = 4  # Can be delayed


@dataclass(frozen=True, slots=True)
class PriorityScore:
    """
    Immutable priority score.

    Attributes:
        tier: Priority tier.
        score: Numeric score in [0, 100] (higher = more important).
        factors: List of factors that influenced the score.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    tier: PriorityTier
    score: int
    factors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrioritizedItem:
    """
    Immutable prioritized conversation item.

    Attributes:
        item_id: Unique identifier.
        content: Item content.
        priority: Priority score.
        timestamp: When the item was created.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    item_id: str
    content: str
    priority: PriorityScore
    timestamp: float


class ConversationalPriorityEngine:
    """
    Assigns priorities and manages a priority queue of conversation items.

    Complexity:
        score: T(n) ∈ O(p * n) for p patterns, n text length.
        enqueue: T(n) ∈ O(log q) for q queue items.
        dequeue: T(n) ∈ O(log q) for q queue items.
    """

    __slots__ = ("_queue", "_counter", "_urgency_patterns", "_error_patterns", "_safety_patterns")

    def __init__(self) -> None:
        """Initialize the priority engine.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._queue: List[PrioritizedItem] = []
        self._counter: int = 0
        self._urgency_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(urgent|asap|immediately|now|emergency)\b", re.IGNORECASE),
        ]
        self._error_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(error|crash|fail|broken|bug|exception)\b", re.IGNORECASE),
        ]
        self._safety_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(danger|threat|attack|vulnerability|breach)\b", re.IGNORECASE),
        ]

    def score(self, text: str) -> PriorityScore:
        """Score a text for priority.

        Args:
            text: Text to score.

        Returns:
            PriorityScore with tier and numeric score.

        Complexity: T(n) ∈ O(p * n) for p patterns, n text length.
        """
        if not text:
            return PriorityScore(
                tier=PriorityTier.LOW,
                score=0,
                factors=(),
            )

        factors: List[str] = []
        score_val = 50  # Base score

        # Check safety patterns (critical)
        for pattern in self._safety_patterns:
            if pattern.search(text):
                score_val = 100
                factors.append("safety_keyword")
                return PriorityScore(
                    tier=PriorityTier.CRITICAL,
                    score=score_val,
                    factors=tuple(factors),
                )

        # Check error patterns (high)
        for pattern in self._error_patterns:
            if pattern.search(text):
                score_val = max(score_val, 85)
                factors.append("error_keyword")
                break

        # Check urgency patterns (high)
        for pattern in self._urgency_patterns:
            if pattern.search(text):
                score_val = max(score_val, 80)
                factors.append("urgency_keyword")
                break

        # Check for questions (normal-high)
        if "?" in text:
            score_val = max(score_val, 65)
            factors.append("question")

        # Check for corrections
        if re.search(r"\b(no wait|actually|correction|wrong)\b", text, re.IGNORECASE):
            score_val = max(score_val, 75)
            factors.append("correction")

        # Check length — longer inputs may need more attention
        if len(text) > 500:
            score_val = min(score_val + 5, 95)
            factors.append("long_input")

        # Determine tier
        if score_val >= 90:
            tier = PriorityTier.CRITICAL
        elif score_val >= 70:
            tier = PriorityTier.HIGH
        elif score_val >= 40:
            tier = PriorityTier.NORMAL
        elif score_val >= 20:
            tier = PriorityTier.LOW
        else:
            tier = PriorityTier.DEFERRED

        if not factors:
            factors.append("default")

        return PriorityScore(
            tier=tier,
            score=score_val,
            factors=tuple(factors),
        )

    def enqueue(self, item_id: str, content: str) -> PrioritizedItem:
        """Add an item to the priority queue.

        Args:
            item_id: Unique identifier.
            content: Item content.

        Returns:
            The created PrioritizedItem.

        Complexity: T(n) ∈ O(log q) for q queue items.
        """
        self._counter += 1
        priority = self.score(content)
        item = PrioritizedItem(
            item_id=item_id,
            content=content,
            priority=priority,
            timestamp=time.time(),
        )
        self._queue.append(item)
        # Sort by score descending, then timestamp ascending
        self._queue.sort(key=lambda x: (-x.priority.score, x.timestamp))
        return item

    def dequeue(self) -> Optional[PrioritizedItem]:
        """Remove and return the highest-priority item.

        Returns:
            PrioritizedItem or None if queue is empty.

        Complexity: T(n) ∈ O(q) for q queue items (due to re-sort).
        """
        if not self._queue:
            return None
        return self._queue.pop(0)

    def peek(self) -> Optional[PrioritizedItem]:
        """Return the highest-priority item without removing it.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not self._queue:
            return None
        return self._queue[0]

    def remove(self, item_id: str) -> bool:
        """Remove a specific item from the queue.

        Returns:
            True if an item was removed.

        Complexity: T(n) ∈ O(q) for q queue items.
        """
        before = len(self._queue)
        self._queue = [item for item in self._queue if item.item_id != item_id]
        return len(self._queue) < before

    def queue_size(self) -> int:
        """Return the number of items in the queue.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._queue)

    def get_queue(self) -> List[PrioritizedItem]:
        """Return a copy of the queue.

        Complexity: T(n) ∈ O(q), S(n) ∈ O(q).
        """
        return list(self._queue)

    def clear(self) -> None:
        """Clear the queue.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._queue.clear()
        self._counter = 0

    def reprioritize(self, item_id: str, new_content: str) -> bool:
        """Update an item's content and recalculate its priority.

        Args:
            item_id: Item to update.
            new_content: New content text.

        Returns:
            True if the item was found and updated.

        Complexity: T(n) ∈ O(q log q) for q queue items.
        """
        for i, item in enumerate(self._queue):
            if item.item_id == item_id:
                new_priority = self.score(new_content)
                self._queue[i] = PrioritizedItem(
                    item_id=item_id,
                    content=new_content,
                    priority=new_priority,
                    timestamp=item.timestamp,
                )
                self._queue.sort(key=lambda x: (-x.priority.score, x.timestamp))
                return True
        return False
