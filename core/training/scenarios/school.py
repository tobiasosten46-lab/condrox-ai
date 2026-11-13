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
School training scenarios for the Condrox AI pre-launch validation pipeline.

These scenarios cover student and teacher roles across core school subjects.
All answers must be deterministic, correct, and free of hallucination markers.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from core.training.training_models import TrainingScenario


SCHOOL_SCENARIOS: tuple[TrainingScenario, ...] = (
    # Mathematics
    TrainingScenario(
        scenario_id="school_math_fraction_add",
        domain="school",
        role="student",
        input_text="How do I add 1/4 and 2/3?",
        expected_output_type="step_by_step",
        expected_keywords=("common denominator", "twelfths", "eleven", "11/12"),
        forbidden_patterns=("just guess", "approximately"),
        min_steps=3,
        min_words=30,
    ),
    TrainingScenario(
        scenario_id="school_math_pythagoras",
        domain="school",
        role="student",
        input_text="Explain the Pythagorean theorem.",
        expected_output_type="explanation",
        expected_keywords=("right triangle", "hypotenuse", "a²", "b²", "c²"),
        forbidden_patterns=("i think",),
        min_steps=0,
        min_words=25,
    ),
    # Norwegian / language
    TrainingScenario(
        scenario_id="school_norwegian_noun",
        domain="school",
        role="teacher",
        input_text="What is a noun?",
        expected_output_type="explanation",
        expected_keywords=("person", "place", "thing", "idea", "name"),
        forbidden_patterns=(),
        min_steps=0,
        min_words=20,
    ),
    # Natural science
    TrainingScenario(
        scenario_id="school_science_photosynthesis",
        domain="school",
        role="student",
        input_text="What is photosynthesis?",
        expected_output_type="explanation",
        expected_keywords=("sunlight", "carbon dioxide", "water", "glucose", "oxygen"),
        forbidden_patterns=("i guess",),
        min_steps=0,
        min_words=30,
    ),
    # Social science
    TrainingScenario(
        scenario_id="school_social_democracy",
        domain="school",
        role="student",
        input_text="What is democracy?",
        expected_output_type="explanation",
        expected_keywords=("people", "vote", "election", "government", "majority"),
        forbidden_patterns=(),
        min_steps=0,
        min_words=25,
    ),
    # IT / programming
    TrainingScenario(
        scenario_id="school_it_function",
        domain="school",
        role="teacher",
        input_text="Explain what a function is in programming.",
        expected_output_type="explanation",
        expected_keywords=("input", "output", "reuse", "def", "name"),
        forbidden_patterns=("maybe",),
        min_steps=0,
        min_words=25,
    ),
)
