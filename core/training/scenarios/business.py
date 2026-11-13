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

"""Business training scenarios for Condrox AI validation."""

from __future__ import annotations

from core.training.training_models import TrainingScenario


BUSINESS_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="business_invoice_process",
        domain="business",
        role="employee",
        input_text="How do I process a customer invoice in our ERP?",
        expected_output_type="step_by_step",
        expected_keywords=("invoice", "erp", "verify", "record", "payment"),
        forbidden_patterns=("i think", "maybe", "just guess"),
        min_steps=3,
        min_words=30,
    ),
    TrainingScenario(
        scenario_id="business_project_status",
        domain="business",
        role="manager",
        input_text="Give me a structured project status update.",
        expected_output_type="structured_text",
        expected_keywords=("status", "progress", "risk", "next", "milestone"),
        forbidden_patterns=("i guess",),
        min_words=25,
    ),
    TrainingScenario(
        scenario_id="business_meeting_minutes",
        domain="business",
        role="assistant",
        input_text="Summarize the key decisions from the meeting.",
        expected_output_type="structured_text",
        expected_keywords=("decision", "action", "owner", "deadline"),
        forbidden_patterns=("probably",),
        min_words=20,
    ),
)
