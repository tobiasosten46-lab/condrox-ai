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

"""Tests for DialogueGoalTracker — Phase 2-E component."""

from __future__ import annotations

from core.conversational.goal_tracker import (
    DialogueGoalTracker,
    Goal,
    GoalStatus,
)


class TestDialogueGoalTracker:
    """Unit tests for DialogueGoalTracker."""

    def test_add_goal(self) -> None:
        """Goals can be added."""
        tracker = DialogueGoalTracker()
        goal = tracker.add_goal("Write a function")
        assert goal.goal_id is not None
        assert goal.description == "Write a function"
        assert tracker.goal_count() == 1

    def test_first_goal_auto_activated(self) -> None:
        """First goal is auto-activated."""
        tracker = DialogueGoalTracker()
        tracker.add_goal("Goal 1")
        active = tracker.get_active_goal()
        assert active is not None
        assert active.status == GoalStatus.ACTIVE

    def test_update_progress(self) -> None:
        """Progress can be updated."""
        tracker = DialogueGoalTracker()
        goal = tracker.add_goal("Goal 1")
        tracker.update_progress(goal.goal_id, 0.5)
        assert tracker.get_goal(goal.goal_id).progress == 0.5

    def test_complete_goal_on_full_progress(self) -> None:
        """Goal completes at 100% progress."""
        tracker = DialogueGoalTracker()
        goal = tracker.add_goal("Goal 1")
        tracker.add_goal("Goal 2")
        tracker.update_progress(goal.goal_id, 1.0)
        assert tracker.get_goal(goal.goal_id).status == GoalStatus.COMPLETED
        # Next goal should be activated
        active = tracker.get_active_goal()
        assert active is not None
        assert active.goal_id != goal.goal_id

    def test_complete_goal(self) -> None:
        """complete_goal marks goal as completed."""
        tracker = DialogueGoalTracker()
        goal = tracker.add_goal("Goal 1")
        tracker.add_goal("Goal 2")
        assert tracker.complete_goal(goal.goal_id) is True
        assert tracker.get_goal(goal.goal_id).status == GoalStatus.COMPLETED

    def test_abandon_goal(self) -> None:
        """abandon_goal marks goal as abandoned."""
        tracker = DialogueGoalTracker()
        goal = tracker.add_goal("Goal 1")
        tracker.add_goal("Goal 2")
        tracker.abandon_goal(goal.goal_id)
        assert tracker.get_goal(goal.goal_id).status == GoalStatus.ABANDONED

    def test_activate_goal(self) -> None:
        """activate_goal activates a specific goal."""
        tracker = DialogueGoalTracker()
        g1 = tracker.add_goal("Goal 1")
        g2 = tracker.add_goal("Goal 2")
        tracker.activate_goal(g1.goal_id)
        assert tracker.get_active_goal().goal_id == g1.goal_id

    def test_get_active_goals(self) -> None:
        """get_active_goals returns active and pending."""
        tracker = DialogueGoalTracker()
        tracker.add_goal("Goal 1")
        tracker.add_goal("Goal 2")
        tracker.add_goal("Goal 3")
        active = tracker.get_active_goals()
        assert len(active) == 3  # 1 active + 2 pending

    def test_get_completed_goals(self) -> None:
        """get_completed_goals returns only completed."""
        tracker = DialogueGoalTracker()
        g1 = tracker.add_goal("Goal 1")
        tracker.add_goal("Goal 2")
        tracker.complete_goal(g1.goal_id)
        completed = tracker.get_completed_goals()
        assert len(completed) == 1

    def test_sub_goals(self) -> None:
        """Sub-goals are linked to parent."""
        tracker = DialogueGoalTracker()
        parent = tracker.add_goal("Parent goal")
        child = tracker.add_goal("Child goal", parent_goal=parent.goal_id)
        assert child.parent_goal == parent.goal_id
        assert child.goal_id in tracker.get_goal(parent.goal_id).sub_goals

    def test_clear(self) -> None:
        """Clear removes all goals."""
        tracker = DialogueGoalTracker()
        tracker.add_goal("Goal 1")
        tracker.clear()
        assert tracker.goal_count() == 0
        assert tracker.get_active_goal() is None

    def test_get_missing_goal(self) -> None:
        """Missing goal returns None."""
        tracker = DialogueGoalTracker()
        assert tracker.get_goal("missing") is None

    def test_all_goals(self) -> None:
        """all_goals returns everything."""
        tracker = DialogueGoalTracker()
        tracker.add_goal("Goal 1")
        tracker.add_goal("Goal 2")
        assert len(tracker.all_goals()) == 2

    def test_priority_clamped(self) -> None:
        """Priority is clamped to [0, 10]."""
        tracker = DialogueGoalTracker()
        goal = tracker.add_goal("Goal", priority=99)
        assert goal.priority == 10
        goal2 = tracker.add_goal("Goal2", priority=-5)
        assert goal2.priority == 0
