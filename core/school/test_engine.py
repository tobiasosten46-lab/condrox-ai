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
Test engine for the Condrox school system.

Runs a sequence of exercises as a test, computes a score, and determines
whether the student/AI is ready to advance a level.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from core.school.exercise_engine import ExerciseEngine
from core.school.school_models import Exercise, ExerciseResult


QuestionRunner = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class SchoolTestReport:
    """
    Immutable result of a school test.

    Attributes:
        test_id: Identifier.
        total: Total number of exercises.
        passed: Number passed.
        score: Average score in [0.0, 1.0].
        ready_to_advance: True if score meets threshold.
        results: Per-exercise results.

    Complexity: S(n) ∈ O(r) for r results.
    """

    test_id: str
    total: int
    passed: int
    score: float
    ready_to_advance: bool
    results: tuple[ExerciseResult, ...]


class SchoolTestEngine:
    """
    Engine that administers a test as a list of exercises.

    Attributes:
        _exercise_engine: ExerciseEngine used for evaluation.

    Complexity:
        run: T(n) ∈ O(m · e) for m exercises and e evaluation cost.
    """

    __slots__ = ("_exercise_engine", "_advance_threshold")

    def __init__(
        self,
        exercise_engine: Optional[ExerciseEngine] = None,
        advance_threshold: float = 0.8,
    ) -> None:
        """Initialize the test engine.

        Args:
            exercise_engine: Optional ExerciseEngine; defaults to new instance.
            advance_threshold: Minimum average score to advance.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._exercise_engine = exercise_engine or ExerciseEngine()
        self._advance_threshold = advance_threshold

    def run(
        self,
        test_id: str,
        exercises: List[Exercise],
        runner: QuestionRunner,
    ) -> SchoolTestReport:
        """Run a test by feeding each exercise to a runner.

        Args:
            test_id: Test identifier.
            exercises: List of exercises.
            runner: Callable that takes a question string and returns an answer.

        Returns:
            SchoolTestReport with aggregated results.

        Complexity: T(n) ∈ O(m · e) for m exercises.
        """
        results: List[ExerciseResult] = []
        for exercise in exercises:
            answer = runner(exercise.question)
            result = self._exercise_engine.evaluate(exercise, answer)
            results.append(result)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        score = sum(r.score for r in results) / max(1, total) if total else 0.0
        return SchoolTestReport(
            test_id=test_id,
            total=total,
            passed=passed,
            score=score,
            ready_to_advance=score >= self._advance_threshold,
            results=tuple(results),
        )
