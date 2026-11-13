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

"""Health and legal/accounting training scenarios for Condrox AI validation."""

from __future__ import annotations

from core.training.training_models import TrainingScenario


HEALTH_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="health_side_effects",
        domain="health",
        role="patient",
        input_text="What are the side effects of this medication?",
        expected_output_type="structured_text",
        expected_keywords=("side effect", "medicine", "doctor", "pharmacist"),
        forbidden_patterns=("i think", "probably", "you should be fine", "just"),
        min_words=25,
        requires_citations=True,
    ),
    TrainingScenario(
        scenario_id="health_nutrition",
        domain="health",
        role="patient",
        input_text="Give dietary advice for type 2 diabetes.",
        expected_output_type="structured_text",
        expected_keywords=("diabetes", "sugar", "carbohydrate", "healthcare"),
        forbidden_patterns=("i guess", "maybe"),
        min_words=30,
        requires_citations=True,
    ),
    TrainingScenario(
        scenario_id="accounting_vat",
        domain="health",
        role="accountant",
        input_text="What is the current VAT rate for food in Norway?",
        expected_output_type="explanation",
        expected_keywords=("vat", "food", "norway", "15"),
        forbidden_patterns=("i think", "about", "around"),
        min_words=15,
        requires_citations=True,
    ),
)
