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

"""Tests for the training console and quality gate validator."""

from __future__ import annotations

from core.training.gate_validator import QualityGateValidator
from core.training.scenarios.school import SCHOOL_SCENARIOS
from core.training.training_console import TrainingConsole
from core.training.training_models import TrainingScenario


class TestQualityGateValidator:
    """Unit tests for the 6 quality gates."""

    def _scenario(self, **overrides) -> TrainingScenario:
        defaults = {
            "scenario_id": "test_001",
            "domain": "school",
            "role": "student",
            "input_text": "What is a noun?",
            "expected_output_type": "explanation",
            "expected_keywords": ("person", "place"),
            "forbidden_patterns": (),
            "min_steps": 0,
            "min_words": 10,
        }
        defaults.update(overrides)
        return TrainingScenario(**defaults)  # type: ignore[arg-type]

    def test_domain_gate_passes_with_keywords(self) -> None:
        """DOMAIN passes when expected keywords are present."""
        validator = QualityGateValidator()
        scenario = self._scenario()
        result = validator.validate(scenario, "A noun is a word for a person, place, or thing.")
        gate = next(g for g in result if g.gate_name == "DOMAIN")
        assert gate.passed

    def test_safety_gate_fails_on_hallucination_marker(self) -> None:
        """SAFETY fails when uncertainty markers are present."""
        validator = QualityGateValidator()
        scenario = self._scenario()
        result = validator.validate(scenario, "I think a noun is maybe a person.")
        gate = next(g for g in result if g.gate_name == "SAFETY")
        assert not gate.passed

    def test_structure_gate_fails_on_template_leak(self) -> None:
        """STRUCTURE fails on pipeline template leaks."""
        validator = QualityGateValidator()
        scenario = self._scenario()
        result = validator.validate(scenario, "Processing: noun found. mode=explain")
        gate = next(g for g in result if g.gate_name == "STRUCTURE")
        assert not gate.passed

    def test_step_by_step_gate_requires_numbered_steps(self) -> None:
        """STRUCTURE requires numbered steps for step_by_step output."""
        validator = QualityGateValidator()
        scenario = self._scenario(
            expected_output_type="step_by_step",
            min_steps=3,
        )
        result = validator.validate(scenario, "First do this. Then do that. Finally.")
        gate = next(g for g in result if g.gate_name == "STRUCTURE")
        assert not gate.passed


class TestTrainingConsole:
    """Tests for the training console."""

    def test_run_scenario_passes_with_good_runner(self) -> None:
        """Console marks scenario PASS when runner output is good."""
        scenario = TrainingScenario(
            scenario_id="good",
            domain="school",
            role="student",
            input_text="What is 2+2?",
            expected_output_type="explanation",
            expected_keywords=("four", "4"),
            min_words=5,
        )

        def runner(_text: str) -> str:
            return "The correct answer is 4 (four)."

        console = TrainingConsole(runner=runner)
        console.add_scenario(scenario)
        result = console.run_scenario("good")
        assert result.overall_pass

    def test_run_scenario_fails_with_bad_runner(self) -> None:
        """Console marks scenario FAIL when runner output is bad."""
        scenario = TrainingScenario(
            scenario_id="bad",
            domain="school",
            role="student",
            input_text="What is 2+2?",
            expected_output_type="explanation",
            expected_keywords=("four", "4"),
            min_words=20,
        )

        def runner(_text: str) -> str:
            return "I think maybe four."

        console = TrainingConsole(runner=runner)
        console.add_scenario(scenario)
        result = console.run_scenario("bad")
        assert not result.overall_pass

    def test_run_domain_aggregates_results(self) -> None:
        """run_domain returns a report for a domain."""
        console = TrainingConsole(runner=lambda text: f"Answer to: {text}")
        console.add_scenarios(SCHOOL_SCENARIOS)
        report = console.run_domain("school")
        assert report.total == len(SCHOOL_SCENARIOS)
        assert report.passed + report.failed == report.total

    def test_format_report_contains_results(self) -> None:
        """format_report produces a non-empty string."""
        console = TrainingConsole(runner=lambda text: f"Answer to: {text}")
        console.add_scenarios(SCHOOL_SCENARIOS[:1])
        report = console.run_domain("school")
        text = TrainingConsole.format_report(report)
        assert "Training Report" in text
        assert SCHOOL_SCENARIOS[0].scenario_id in text
