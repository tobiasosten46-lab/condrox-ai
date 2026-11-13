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

"""Tests for the Condrox School System."""

from __future__ import annotations

from core.school.curriculum import SchoolCurriculum
from core.school.exercise_engine import ExerciseEngine
from core.school.school_models import Lesson
from core.school.school_system import CondroxSchoolSystem
from core.school.test_engine import SchoolTestEngine, SchoolTestReport


class TestSchoolCurriculum:
    """Tests for the built-in school curriculum."""

    def test_default_curriculum_has_six_subjects(self) -> None:
        """Default curriculum contains six school subjects."""
        curriculum = SchoolCurriculum.default()
        assert len(curriculum.list_subjects()) == 6

    def test_get_lessons_for_subject(self) -> None:
        """Lessons can be retrieved by subject."""
        curriculum = SchoolCurriculum.default()
        lessons = curriculum.get_lessons_for_subject("math")
        assert len(lessons) >= 2
        assert all(isinstance(lesson, Lesson) for lesson in lessons)

    def test_curriculum_loaded_from_json(self) -> None:
        """Default curriculum loads from package JSON and contains expanded lessons."""
        curriculum = SchoolCurriculum.default()
        assert curriculum.get_lesson("math_percentages") is not None
        assert curriculum.get_lesson("it_security") is not None


class TestExerciseEngine:
    """Tests for exercise generation and evaluation."""

    def test_exercises_generated_from_lesson(self) -> None:
        """Engine generates exercises for a lesson."""
        curriculum = SchoolCurriculum.default()
        lesson = curriculum.get_lesson("math_fractions")
        assert lesson is not None
        engine = ExerciseEngine()
        exercises = engine.create_exercises_for_lesson(lesson)
        assert len(exercises) == 2

    def test_evaluate_good_answer_passes(self) -> None:
        """A correct, keyword-rich answer passes."""
        curriculum = SchoolCurriculum.default()
        lesson = curriculum.get_lesson("math_fractions")
        engine = ExerciseEngine()
        exercises = engine.create_exercises_for_lesson(lesson)
        answer = (
            "To add fractions, find a common denominator, convert each fraction, "
            "then add the numerators. For example 1/4 + 2/3 equals 11/12."
        )
        result = engine.evaluate(exercises[0], answer)
        assert result.passed

    def test_evaluate_bad_answer_fails(self) -> None:
        """An answer with forbidden patterns fails."""
        curriculum = SchoolCurriculum.default()
        lesson = curriculum.get_lesson("math_fractions")
        engine = ExerciseEngine()
        exercises = engine.create_exercises_for_lesson(lesson)
        result = engine.evaluate(exercises[0], "I think maybe you just add them.")
        assert not result.passed


class TestSchoolSystem:
    """Integration tests for CondroxSchoolSystem."""

    def test_system_generates_exercises(self) -> None:
        """Facade generates exercises for a known lesson."""
        system = CondroxSchoolSystem()
        exercises = system.generate_exercises("math_fractions")
        assert len(exercises) == 2

    def test_system_tests_lesson_with_good_runner(self) -> None:
        """Test a lesson with a runner that gives correct answers."""
        system = CondroxSchoolSystem()

        def runner(question: str) -> str:
            answers = {
                "Explain Adding Fractions.": (
                    "Find a common denominator, convert each fraction, add the "
                    "numerators, and keep the denominator."
                ),
                "Give an example of Adding Fractions.": (
                    "1/4 + 2/3 = 3/12 + 8/12 = 11/12, which is eleven twelfths, is an example."
                ),
            }
            return answers.get(question, "I do not know.")

        report = system.test_lesson("math_fractions", runner)
        assert report is not None
        assert report.ready_to_advance

    def test_progress_updates_after_result(self) -> None:
        """Progress reflects a completed lesson and level change."""
        system = CondroxSchoolSystem()
        report = SchoolTestReport(
            test_id="math_fractions",
            total=2,
            passed=2,
            score=1.0,
            ready_to_advance=True,
            results=(),
        )
        progress = system.record_result("alice", "math_fractions", report)
        assert "math_fractions" in progress.completed_lessons
        assert system.get_progress("alice").exercise_scores
