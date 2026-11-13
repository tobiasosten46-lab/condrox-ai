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
Training Console — deterministic validation harness for Condrox AI.

Runs scenarios through a runner (e.g. BrainController.process_request),
evaluates each response against the 6 quality gates, and produces a report.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from core.training.gate_validator import QualityGateValidator
from core.training.training_models import (
    TrainingReport,
    TrainingResult,
    TrainingRunner,
    TrainingScenario,
)


class TrainingConsole:
    """
    CLI/internal console for pre-launch AI training and validation.

    The console is runner-agnostic: it accepts any callable that maps an
    input string to an output string. This lets it validate the real
    BrainController pipeline or a mocked runner in tests.

    Attributes:
        _runner: Callable that produces AI output from input text.
        _validator: QualityGateValidator instance.
        _scenarios: In-memory scenario registry keyed by scenario_id.

    Complexity:
        add_scenario: T(n) ∈ O(1).
        run_scenario: T(n) ∈ O(r + v) for runner time r and validation v.
        run_domain: T(n) ∈ O(s · (r + v)) for s scenarios.
    """

    __slots__ = ("_runner", "_validator", "_scenarios")

    def __init__(
        self,
        runner: TrainingRunner,
        validator: Optional[QualityGateValidator] = None,
    ) -> None:
        """Initialize the training console.

        Args:
            runner: Callable taking input text and returning output text.
            validator: Optional QualityGateValidator; defaults to a new one.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._runner: TrainingRunner = runner
        self._validator: QualityGateValidator = validator or QualityGateValidator()
        self._scenarios: Dict[str, TrainingScenario] = {}

    def add_scenario(self, scenario: TrainingScenario) -> None:
        """Register a training scenario.

        Args:
            scenario: Scenario to register.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._scenarios[scenario.scenario_id] = scenario

    def add_scenarios(self, scenarios: Sequence[TrainingScenario]) -> int:
        """Register multiple scenarios.

        Args:
            scenarios: Sequence of scenarios.

        Returns:
            Number of scenarios registered.

        Complexity: T(n) ∈ O(s) for s scenarios.
        """
        for scenario in scenarios:
            self.add_scenario(scenario)
        return len(scenarios)

    def list_scenarios(self) -> List[str]:
        """Return all registered scenario ids.

        Complexity: T(n) ∈ O(s) for s scenarios.
        """
        return list(self._scenarios.keys())

    def run_scenario(self, scenario_id: str) -> TrainingResult:
        """Run a single scenario and validate the output.

        Args:
            scenario_id: Scenario identifier.

        Returns:
            TrainingResult with gate results.

        Raises:
            KeyError: If the scenario is not registered.

        Complexity: T(n) ∈ O(r + v) for runner time r and validation v.
        """
        scenario = self._scenarios[scenario_id]
        actual_output = self._runner(scenario.input_text)
        gate_results = self._validator.validate(scenario, actual_output)
        overall_pass = all(g.passed for g in gate_results)
        return TrainingResult(
            scenario_id=scenario_id,
            actual_output=actual_output,
            gate_results=gate_results,
            overall_pass=overall_pass,
        )

    def run_domain(self, domain: str) -> TrainingReport:
        """Run all scenarios in a domain.

        Args:
            domain: Domain label to run.

        Returns:
            TrainingReport with aggregated results.

        Complexity: T(n) ∈ O(s · (r + v)) for s scenarios in the domain.
        """
        results: List[TrainingResult] = []
        for scenario in self._scenarios.values():
            if scenario.domain == domain:
                results.append(self.run_scenario(scenario.scenario_id))

        passed = sum(1 for r in results if r.overall_pass)
        return TrainingReport(
            domain=domain,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            results=tuple(results),
        )

    def run_all(self) -> TrainingReport:
        """Run every registered scenario.

        Returns:
            TrainingReport across all domains.

        Complexity: T(n) ∈ O(s · (r + v)) for s scenarios.
        """
        results: List[TrainingResult] = []
        for scenario_id in self._scenarios:
            results.append(self.run_scenario(scenario_id))

        passed = sum(1 for r in results if r.overall_pass)
        return TrainingReport(
            domain="all",
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            results=tuple(results),
        )

    @staticmethod
    def format_report(report: TrainingReport) -> str:
        """Return a human-readable text report.

        Complexity: T(n) ∈ O(r) for r results.
        """
        lines: List[str] = [
            f"Training Report - {report.domain}",
            f"Total: {report.total}, Passed: {report.passed}, Failed: {report.failed}",
            "",
        ]
        for result in report.results:
            status = "PASS" if result.overall_pass else "FAIL"
            lines.append(f"[{status}] {result.scenario_id}")
            for gate in result.gate_results:
                gstatus = "PASS" if gate.passed else "FAIL"
                lines.append(f"  {gate.gate_name}: {gstatus} (score={gate.score:.2f})")
                for finding in gate.findings:
                    lines.append(f"    - {finding}")
        return "\n".join(lines)
