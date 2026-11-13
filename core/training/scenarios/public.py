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

"""Public sector training scenarios for Condrox AI validation."""

from __future__ import annotations

from core.training.training_models import TrainingScenario


PUBLIC_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario(
        scenario_id="public_application_status",
        domain="public",
        role="citizen",
        input_text="What is the status of my building permit application?",
        expected_output_type="structured_text",
        expected_keywords=("application", "status", "process", "decision"),
        forbidden_patterns=("i think", "maybe"),
        min_words=20,
    ),
    TrainingScenario(
        scenario_id="public_decision_draft",
        domain="public",
        role="clerk",
        input_text="Draft a formal decision text rejecting the application with reference to section 5.",
        expected_output_type="structured_text",
        expected_keywords=("decision", "section 5", "reject", "application", "pursuant"),
        forbidden_patterns=("i think", "probably", "maybe"),
        min_words=40,
        requires_citations=True,
    ),
    TrainingScenario(
        scenario_id="public_service_info",
        domain="public",
        role="citizen",
        input_text="Which documents do I need to enroll my child in kindergarten?",
        expected_output_type="structured_text",
        expected_keywords=("documents", "kindergarten", "enroll", "identity", "address"),
        forbidden_patterns=("just",),
        min_words=25,
    ),
)
