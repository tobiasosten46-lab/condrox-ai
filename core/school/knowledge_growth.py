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
Knowledge Growth System (KGS) for the Condrox school system.

Tracks per-student levels, completed lessons, and spaced-repetition schedules.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from core.school.school_models import StudentProgress


@dataclass(frozen=True, slots=True)
class RepetitionItem:
    """
    Immutable repetition schedule entry.

    Attributes:
        lesson_id: Lesson to review.
        due_at: Unix timestamp when review is due.
        reason: spaced, error, context.

    Complexity: S(n) ∈ O(1).
    """

    lesson_id: str
    due_at: float
    reason: str


class KnowledgeGrowthSystem:
    """
    Tracks level progression and repetition schedules per student.

    Attributes:
        _progress: Mapping from user_id to StudentProgress.
        _repetitions: Mapping from user_id to list of RepetitionItem.

    Complexity:
        record_score: T(n) ∈ O(1).
        get_level: T(n) ∈ O(1).
        due_repetitions: T(n) ∈ O(r log r) for r items due to sorting.
    """

    __slots__ = ("_progress", "_repetitions")

    def __init__(self) -> None:
        """Initialize the KGS.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._progress: Dict[str, StudentProgress] = {}
        self._repetitions: Dict[str, List[RepetitionItem]] = {}

    def _ensure_progress(self, user_id: str) -> StudentProgress:
        """Return existing progress or create a blank one.

        Complexity: T(n) ∈ O(1).
        """
        if user_id not in self._progress:
            self._progress[user_id] = StudentProgress(
                user_id=user_id,
                levels=(),
                completed_lessons=(),
                exercise_scores=(),
            )
            self._repetitions[user_id] = []
        return self._progress[user_id]

    def get_progress(self, user_id: str) -> StudentProgress:
        """Retrieve or create progress for a user.

        Complexity: T(n) ∈ O(1).
        """
        return self._ensure_progress(user_id)

    def get_level(self, user_id: str, subject_id: str) -> int:
        """Return the user's level for a subject (default 1).

        Complexity: T(n) ∈ O(s) for s stored subject levels.
        """
        progress = self._ensure_progress(user_id)
        for subject, level in progress.levels:
            if subject == subject_id:
                return level
        return 1

    def record_score(
        self,
        user_id: str,
        lesson_id: str,
        subject_id: str,
        score: float,
        ready_to_advance: bool,
    ) -> StudentProgress:
        """Record an exercise/test score and update progress.

        Args:
            user_id: Student identifier.
            lesson_id: Lesson completed.
            subject_id: Subject the lesson belongs to.
            score: Score in [0.0, 1.0].
            ready_to_advance: Whether the student may advance a level.

        Returns:
            Updated StudentProgress.

        Complexity: T(n) ∈ O(s + l) for s subjects and l lessons.
        """
        progress = self._ensure_progress(user_id)

        # Update lesson completion.
        completed = set(progress.completed_lessons)
        completed.add(lesson_id)

        # Update best score for this lesson.
        score_dict: Dict[str, float] = dict(progress.exercise_scores)
        score_dict[lesson_id] = max(score_dict.get(lesson_id, 0.0), score)

        # Update level if ready.
        level_dict: Dict[str, int] = dict(progress.levels)
        if ready_to_advance:
            current = level_dict.get(subject_id, 1)
            level_dict[subject_id] = min(5, current + 1)

        new_progress = StudentProgress(
            user_id=user_id,
            levels=tuple(sorted(level_dict.items())),
            completed_lessons=tuple(sorted(completed)),
            exercise_scores=tuple(sorted(score_dict.items())),
        )
        self._progress[user_id] = new_progress

        # Schedule repetitions based on score.
        now = time.time()
        if score < 0.8:
            self._add_repetition(user_id, lesson_id, now + 300, "error")
        else:
            self._add_repetition(user_id, lesson_id, now + 86400, "spaced")

        return new_progress

    def _add_repetition(
        self, user_id: str, lesson_id: str, due_at: float, reason: str
    ) -> None:
        """Add a repetition schedule entry.

        Complexity: T(n) ∈ O(1).
        """
        self._repetitions.setdefault(user_id, []).append(
            RepetitionItem(lesson_id=lesson_id, due_at=due_at, reason=reason)
        )

    def due_repetitions(self, user_id: str, before: Optional[float] = None) -> List[RepetitionItem]:
        """Return repetition entries due before a timestamp.

        Args:
            user_id: Student identifier.
            before: Unix timestamp; defaults to now.

        Returns:
            Sorted list of due RepetitionItem.

        Complexity: T(n) ∈ O(r log r) for r items.
        """
        before = before if before is not None else time.time()
        items = [item for item in self._repetitions.get(user_id, []) if item.due_at <= before]
        items.sort(key=lambda item: item.due_at)
        return items
