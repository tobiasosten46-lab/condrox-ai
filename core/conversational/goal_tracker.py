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
Dialogue Goal Tracker — tracks conversation goals across turns and makes
them accessible to agents.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GoalStatus(Enum):
    """Goal lifecycle status."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """
    Mutable conversation goal.

    Attributes:
        goal_id: Unique identifier.
        description: Human-readable goal description.
        status: Current status.
        priority: Priority in [0, 10].
        created_at: Creation timestamp.
        completed_at: Completion timestamp (if completed).
        progress: Progress in [0.0, 1.0].
        sub_goals: List of sub-goal IDs.
        parent_goal: Optional parent goal ID.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    goal_id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    progress: float = 0.0
    sub_goals: List[str] = field(default_factory=list)
    parent_goal: Optional[str] = None


class DialogueGoalTracker:
    """
    Tracks conversation goals across turns.

    Complexity:
        add_goal: T(n) ∈ O(1), S(n) ∈ O(1).
        update_progress: T(n) ∈ O(1), S(n) ∈ O(1).
        get_active_goals: T(n) ∈ O(g) for g goals, S(n) ∈ O(g).
    """

    __slots__ = ("_goals", "_goal_counter", "_active_goal_id")

    def __init__(self) -> None:
        """Initialize the goal tracker.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._goals: Dict[str, Goal] = {}
        self._goal_counter: int = 0
        self._active_goal_id: Optional[str] = None

    def add_goal(
        self,
        description: str,
        priority: int = 5,
        parent_goal: Optional[str] = None,
    ) -> Goal:
        """Add a new goal.

        Args:
            description: Goal description.
            priority: Priority in [0, 10].
            parent_goal: Optional parent goal ID.

        Returns:
            The created Goal.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._goal_counter += 1
        goal_id = f"goal_{self._goal_counter:06d}"
        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=max(0, min(10, priority)),
            parent_goal=parent_goal,
        )
        self._goals[goal_id] = goal

        if parent_goal and parent_goal in self._goals:
            self._goals[parent_goal].sub_goals.append(goal_id)

        # Auto-activate first goal
        if self._active_goal_id is None:
            self._active_goal_id = goal_id
            goal.status = GoalStatus.ACTIVE

        return goal

    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Update goal progress.

        Args:
            goal_id: Goal to update.
            progress: Progress in [0.0, 1.0].

        Returns:
            True if the goal was updated.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False

        goal.progress = max(0.0, min(1.0, progress))

        if goal.progress >= 1.0 and goal.status != GoalStatus.COMPLETED:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            self._advance_active_goal()

        return True

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed.

        Returns:
            True if the goal was completed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False

        goal.status = GoalStatus.COMPLETED
        goal.progress = 1.0
        goal.completed_at = time.time()
        self._advance_active_goal()
        return True

    def abandon_goal(self, goal_id: str) -> bool:
        """Mark a goal as abandoned.

        Returns:
            True if the goal was abandoned.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False

        goal.status = GoalStatus.ABANDONED
        if self._active_goal_id == goal_id:
            self._advance_active_goal()
        return True

    def activate_goal(self, goal_id: str) -> bool:
        """Activate a specific goal.

        Returns:
            True if the goal was activated.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False

        if self._active_goal_id and self._active_goal_id in self._goals:
            self._goals[self._active_goal_id].status = GoalStatus.PENDING

        goal.status = GoalStatus.ACTIVE
        self._active_goal_id = goal_id
        return True

    def get_active_goal(self) -> Optional[Goal]:
        """Get the currently active goal.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self._active_goal_id is None:
            return None
        return self._goals.get(self._active_goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Get all active and pending goals.

        Complexity: T(n) ∈ O(g), S(n) ∈ O(g).
        """
        return [
            g for g in self._goals.values()
            if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING)
        ]

    def get_completed_goals(self) -> List[Goal]:
        """Get all completed goals.

        Complexity: T(n) ∈ O(g), S(n) ∈ O(g).
        """
        return [g for g in self._goals.values() if g.status == GoalStatus.COMPLETED]

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._goals.get(goal_id)

    def all_goals(self) -> List[Goal]:
        """Return all goals.

        Complexity: T(n) ∈ O(g), S(n) ∈ O(g).
        """
        return list(self._goals.values())

    def goal_count(self) -> int:
        """Return total goal count.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._goals)

    def clear(self) -> None:
        """Clear all goals.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._goals.clear()
        self._goal_counter = 0
        self._active_goal_id = None

    def _advance_active_goal(self) -> None:
        """Activate the next pending goal.

        Complexity: T(n) ∈ O(g), S(n) ∈ O(1).
        """
        self._active_goal_id = None
        for goal in self._goals.values():
            if goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.ACTIVE
                self._active_goal_id = goal.goal_id
                return
