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
Condrox School System — orchestrates curriculum, exercises, tests, and
knowledge growth for school-domain AI training.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import List, Optional

from core.school.curriculum import SchoolCurriculum
from core.school.exercise_engine import ExerciseEngine
from core.school.knowledge_growth import KnowledgeGrowthSystem
from core.school.school_models import Exercise, Lesson, StudentProgress, Subject
from core.school.test_engine import QuestionRunner, SchoolTestEngine, SchoolTestReport


class CondroxSchoolSystem:
    """
    Top-level school system facade.

    Coordinates curriculum, exercise generation, test administration, and
    student progress. It is deterministic and requires no external services.

    Attributes:
        curriculum: SchoolCurriculum instance.
        exercise_engine: ExerciseEngine instance.
        test_engine: TestEngine instance.
        growth: KnowledgeGrowthSystem instance.

    Complexity:
        generate_exercises: T(n) ∈ O(1) per lesson.
        test_lesson: T(n) ∈ O(m · e) for m exercises.
        record_result: T(n) ∈ O(1).
    """

    __slots__ = ("curriculum", "exercise_engine", "test_engine", "growth")

    def __init__(self) -> None:
        """Initialize the school system with a default curriculum.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.curriculum = SchoolCurriculum.default()
        self.exercise_engine = ExerciseEngine()
        self.test_engine = SchoolTestEngine(self.exercise_engine)
        self.growth = KnowledgeGrowthSystem()

    def list_subjects(self) -> List[str]:
        """Return all subject ids.

        Complexity: T(n) ∈ O(s) for s subjects.
        """
        return self.curriculum.list_subjects()

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Retrieve a subject.

        Complexity: T(n) ∈ O(1).
        """
        return self.curriculum.get_subject(subject_id)

    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Retrieve a lesson.

        Complexity: T(n) ∈ O(1).
        """
        return self.curriculum.get_lesson(lesson_id)

    def generate_exercises(self, lesson_id: str) -> List[Exercise]:
        """Generate exercises for a lesson.

        Args:
            lesson_id: Lesson identifier.

        Returns:
            List of Exercise objects.

        Complexity: T(n) ∈ O(1).
        """
        lesson = self.curriculum.get_lesson(lesson_id)
        if lesson is None:
            return []
        return self.exercise_engine.create_exercises_for_lesson(lesson)

    def test_lesson(self, lesson_id: str, runner: QuestionRunner) -> Optional[SchoolTestReport]:
        """Run all generated exercises for a lesson through a runner.

        Args:
            lesson_id: Lesson identifier.
            runner: Callable taking a question and returning an answer.

        Returns:
            TestReport or None if lesson not found.

        Complexity: T(n) ∈ O(m · e) for m exercises.
        """
        exercises = self.generate_exercises(lesson_id)
        if not exercises:
            return None
        return self.test_engine.run(lesson_id, exercises, runner)

    def record_result(
        self,
        user_id: str,
        lesson_id: str,
        report: SchoolTestReport,
    ) -> StudentProgress:
        """Record a test result and update progress.

        Args:
            user_id: Student identifier.
            lesson_id: Lesson identifier.
            report: TestReport from test_lesson.

        Returns:
            Updated StudentProgress.

        Complexity: T(n) ∈ O(1).
        """
        lesson = self.curriculum.get_lesson(lesson_id)
        subject_id = lesson.subject_id if lesson is not None else "unknown"
        return self.growth.record_score(
            user_id=user_id,
            lesson_id=lesson_id,
            subject_id=subject_id,
            score=report.score,
            ready_to_advance=report.ready_to_advance,
        )

    def get_progress(self, user_id: str) -> StudentProgress:
        """Retrieve progress for a user.

        Complexity: T(n) ∈ O(1).
        """
        return self.growth.get_progress(user_id)
