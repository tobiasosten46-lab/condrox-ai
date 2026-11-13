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

"""Condrox School System — structured K-12 curriculum and validation."""

from core.school.curriculum import SchoolCurriculum
from core.school.exercise_engine import ExerciseEngine
from core.school.knowledge_growth import KnowledgeGrowthSystem, RepetitionItem
from core.school.school_models import (
    Exercise,
    ExerciseResult,
    Lesson,
    StudentProgress,
    Subject,
)
from core.school.school_system import CondroxSchoolSystem
from core.school.test_engine import SchoolTestEngine, SchoolTestReport

__all__ = [
    "CondroxSchoolSystem",
    "Exercise",
    "ExerciseEngine",
    "ExerciseResult",
    "KnowledgeGrowthSystem",
    "Lesson",
    "RepetitionItem",
    "SchoolCurriculum",
    "SchoolTestEngine",
    "SchoolTestReport",
    "StudentProgress",
    "Subject",
]
