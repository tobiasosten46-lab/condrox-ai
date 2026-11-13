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
Exercise engine for the Condrox school system.

Builds and evaluates exercises from lessons. Evaluation is deterministic:
keyword coverage, forbidden-pattern checks, and type-appropriate heuristics.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from typing import List, Optional

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.school.school_models import Exercise, ExerciseResult, Lesson


class ExerciseEngine:
    """
    Engine that generates exercises from lessons and evaluates answers.

    Attributes:
        _exercises: Mapping from exercise_id to Exercise.

    Complexity:
        create_exercises_for_lesson: T(n) ∈ O(1).
        evaluate: T(n) ∈ O(k + f + n) for k keywords, f forbidden patterns, n answer length.
    """

    __slots__ = ("_exercises",)

    def __init__(self) -> None:
        """Initialize an empty exercise engine.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._exercises: dict[str, Exercise] = {}

    def create_exercises_for_lesson(self, lesson: Lesson) -> List[Exercise]:
        """Generate a default set of exercises for a lesson.

        Args:
            lesson: Lesson to build exercises from.

        Returns:
            List of generated Exercise objects.

        Complexity: T(n) ∈ O(1) for fixed templates.
        """
        exercises: List[Exercise] = [
            Exercise(
                exercise_id=f"{lesson.lesson_id}_explain",
                lesson_id=lesson.lesson_id,
                exercise_type="explain",
                question=f"Explain {lesson.title}.",
                expected_keywords=lesson.concepts,
                forbidden_patterns=("i think", "maybe", "i guess"),
            ),
            Exercise(
                exercise_id=f"{lesson.lesson_id}_example",
                lesson_id=lesson.lesson_id,
                exercise_type="use_in_sentence",
                question=f"Give an example of {lesson.title}.",
                expected_keywords=lesson.examples,
                forbidden_patterns=("i think", "maybe"),
            ),
        ]
        for ex in exercises:
            self._exercises[ex.exercise_id] = ex
        return exercises

    def get(self, exercise_id: str) -> Optional[Exercise]:
        """Retrieve an exercise by id.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._exercises.get(exercise_id)

    def evaluate(self, exercise: Exercise, answer: str) -> ExerciseResult:
        """Evaluate a student answer against an exercise.

        Args:
            exercise: Exercise definition.
            answer: Student or AI answer.

        Returns:
            ExerciseResult with score and findings.

        Complexity: T(n) ∈ O(k + f + n).
        """
        if not answer or len(answer.strip()) < 10:
            return ExerciseResult(
                exercise_id=exercise.exercise_id,
                passed=False,
                score=0.0,
                findings=("Answer is empty or too short.",),
            )

        answer_lower = answer.lower()
        answer_tokens = set(KnowledgeBackend.tokenize(answer))
        findings: List[str] = []

        matched = 0
        for keyword in exercise.expected_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in answer_lower or keyword_lower in answer_tokens:
                matched += 1
            else:
                findings.append(f"Missing expected concept: {keyword}")

        score = matched / max(1, len(exercise.expected_keywords))

        for pattern in exercise.forbidden_patterns:
            if pattern in answer_lower:
                findings.append(f"Forbidden pattern: {pattern}")
                score = 0.0

        passed = score >= 0.8 and not findings
        return ExerciseResult(
            exercise_id=exercise.exercise_id,
            passed=passed,
            score=score,
            findings=tuple(findings),
        )
