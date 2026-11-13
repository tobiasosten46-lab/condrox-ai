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
School system models for the Condrox AI training curriculum.

Immutable value objects for subjects, lessons, exercises, tests, and student
progress.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Subject:
    """
    Immutable school subject.

    Attributes:
        subject_id: Unique identifier.
        name: Human-readable name.
        description: Short description.
        level_range: (min_level, max_level) in the knowledge growth system.

    Complexity: S(n) ∈ O(1).
    """

    subject_id: str
    name: str
    description: str
    level_range: Tuple[int, int]


@dataclass(frozen=True, slots=True)
class Lesson:
    """
    Immutable lesson within a subject.

    Attributes:
        lesson_id: Unique identifier.
        subject_id: Parent subject.
        title: Lesson title.
        concepts: Key concepts covered.
        explanation: Core explanation text.
        examples: Usage examples.
        context_rules: Rules for when/how this knowledge applies.
        level: Knowledge level (1-5).

    Complexity: S(n) ∈ O(c + e + r) for c concepts, e examples, r rules.
    """

    lesson_id: str
    subject_id: str
    title: str
    concepts: Tuple[str, ...]
    explanation: str
    examples: Tuple[str, ...]
    context_rules: Tuple[str, ...]
    level: int


@dataclass(frozen=True, slots=True)
class Exercise:
    """
    Immutable exercise for a lesson.

    Attributes:
        exercise_id: Unique identifier.
        lesson_id: Parent lesson.
        exercise_type: classify, explain, use_in_sentence, identify_error, solve.
        question: Prompt text.
        expected_keywords: Concepts the answer must contain.
        forbidden_patterns: Patterns that indicate a bad answer.

    Complexity: S(n) ∈ O(k + f) for k keywords and f forbidden patterns.
    """

    exercise_id: str
    lesson_id: str
    exercise_type: str
    question: str
    expected_keywords: Tuple[str, ...]
    forbidden_patterns: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExerciseResult:
    """
    Result of evaluating one exercise.

    Attributes:
        exercise_id: Identifier.
        passed: Whether the answer passed.
        score: Normalized score in [0.0, 1.0].
        findings: Human-readable feedback.

    Complexity: S(n) ∈ O(f) for f findings.
    """

    exercise_id: str
    passed: bool
    score: float
    findings: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudentProgress:
    """
    Immutable snapshot of a student's progress.

    Attributes:
        user_id: Student identifier.
        levels: Per-subject level (1-5).
        completed_lessons: Set of completed lesson ids.
        exercise_scores: Per-exercise best score.

    Complexity: S(n) ∈ O(l + s) for lessons and scores.
    """

    user_id: str
    levels: Tuple[Tuple[str, int], ...]
    completed_lessons: Tuple[str, ...]
    exercise_scores: Tuple[Tuple[str, float], ...]
