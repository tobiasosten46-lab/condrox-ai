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
School curriculum for Condrox AI.

Defines subjects and lessons for the school domain. The curriculum is
static and loaded at initialization; later phases may load from JSON.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from core.school.school_models import Lesson, Subject


class SchoolCurriculum:
    """
    In-memory registry of school subjects and lessons.

    Attributes:
        _subjects: Mapping from subject_id to Subject.
        _lessons: Mapping from lesson_id to Lesson.
        _subject_lessons: Mapping from subject_id to list of lesson ids.

    Complexity:
        add_subject: T(n) ∈ O(1).
        add_lesson: T(n) ∈ O(1).
        get_lessons_for_subject: T(n) ∈ O(l) for l lessons.
    """

    __slots__ = ("_subjects", "_lessons", "_subject_lessons")

    def __init__(self) -> None:
        """Initialize an empty curriculum.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._subjects: Dict[str, Subject] = {}
        self._lessons: Dict[str, Lesson] = {}
        self._subject_lessons: Dict[str, List[str]] = {}

    @classmethod
    def default(cls) -> "SchoolCurriculum":
        """Return the default K-12 curriculum loaded from package JSON.

        Falls back to a minimal built-in curriculum if the JSON file is missing.

        Complexity: T(n) ∈ O(n) for n lessons.
        """
        json_path = Path(__file__).with_suffix("").parent / "data" / "curriculum.json"
        if json_path.exists():
            return cls.from_json(json_path)
        return cls._minimal()

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SchoolCurriculum":
        """Load a curriculum from a JSON file.

        Args:
            path: Path to curriculum JSON.

        Returns:
            Populated SchoolCurriculum.

        Complexity: T(n) ∈ O(n) for n lessons.
        """
        curriculum = cls()
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        for subject_data in data.get("subjects", []):
            curriculum.add_subject(Subject(**subject_data))

        for lesson_data in data.get("lessons", []):
            # Convert list fields to tuples for immutable Lesson dataclass.
            lesson_data["concepts"] = tuple(lesson_data.get("concepts", []))
            lesson_data["examples"] = tuple(lesson_data.get("examples", []))
            lesson_data["context_rules"] = tuple(lesson_data.get("context_rules", []))
            curriculum.add_lesson(Lesson(**lesson_data))

        return curriculum

    @classmethod
    def _minimal(cls) -> "SchoolCurriculum":
        """Minimal built-in fallback curriculum.

        Complexity: T(n) ∈ O(1).
        """
        curriculum = cls()
        subjects = (
            Subject("math", "Mathematics", "Numbers, algebra, geometry, logic.", (1, 5)),
            Subject("norwegian", "Norwegian", "Language, grammar, literature.", (1, 5)),
            Subject("english", "English", "English language and literature.", (1, 5)),
            Subject("science", "Natural Science", "Biology, chemistry, physics.", (1, 5)),
            Subject("social", "Social Science", "Society, history, civics.", (1, 5)),
            Subject("it", "Information Technology", "Computers, programming, digital literacy.", (1, 5)),
        )
        for subject in subjects:
            curriculum.add_subject(subject)
        return curriculum

    def add_subject(self, subject: Subject) -> None:
        """Register a subject.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._subjects[subject.subject_id] = subject
        self._subject_lessons.setdefault(subject.subject_id, [])

    def add_lesson(self, lesson: Lesson) -> None:
        """Register a lesson under its subject.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._lessons[lesson.lesson_id] = lesson
        self._subject_lessons.setdefault(lesson.subject_id, []).append(lesson.lesson_id)

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Retrieve a subject by id.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._subjects.get(subject_id)

    def get_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Retrieve a lesson by id.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._lessons.get(lesson_id)

    def list_subjects(self) -> List[str]:
        """Return all subject ids.

        Complexity: T(n) ∈ O(s) for s subjects.
        """
        return list(self._subjects.keys())

    def get_lessons_for_subject(self, subject_id: str) -> List[Lesson]:
        """Return all lessons for a subject.

        Complexity: T(n) ∈ O(l) for l lessons.
        """
        return [self._lessons[lid] for lid in self._subject_lessons.get(subject_id, [])]
