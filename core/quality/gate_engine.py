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
Gate Engine — runs all quality gates and enforces results.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import Tuple

from core.quality.gate_base import GateResult, GateSeverity, QualityGate
from core.quality.complexity_gate import ComplexityGate
from core.quality.safety_gate import SafetyGate
from core.quality.structure_gate import StructureGate
from core.quality.types_gate import TypesGate


class GateEngine:
    """
    Runs all quality gates and enforces results.

    Complexity: T(run_all) ∈ O(g×n) where g = gates, n = code size
    S(run_all) ∈ O(total_findings)
    Invariants: at least one gate registered
    Edge Cases: all gates pass → empty findings
    """

    def __init__(self) -> None:
        self._gates: list[QualityGate] = [
            TypesGate(),
            ComplexityGate(),
            SafetyGate(),
            StructureGate(),
        ]

    def register_gate(self, gate: QualityGate) -> None:
        """
        Register an additional gate.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1)
        Edge Cases: duplicate gate → replaced
        """
        self._gates.append(gate)

    async def run_all(
        self, code: str, file_path: str, language: str
    ) -> Tuple[GateResult, ...]:
        """
        Run all registered gates.

        Complexity: T(n) ∈ O(g×n) where g = gates, n = code size
        S(n) ∈ O(total_findings)
        Edge Cases: empty code → all gates pass
        """
        results: list[GateResult] = []
        for gate in self._gates:
            if gate.enabled:
                result = await gate.check(code, file_path, language)
                results.append(result)
        return tuple(results)

    def format_results(self, results: Tuple[GateResult, ...]) -> str:
        """
        Format gate results as readable text.

        Complexity: T(n) ∈ O(r×f) where r = results, f = findings
        S(n) ∈ O(output)
        Edge Cases: no results → empty string
        """
        lines: list[str] = []
        lines.append("── Condrox Quality Gates ──")
        lines.append("")

        all_passed: bool = True
        for r in results:
            status: str = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(f"  {status}  {r.gate_name:<12} (score: {r.score:.0%})")
            if not r.passed:
                all_passed = False
                for f in r.findings:
                    lines.append(f"         └─ [{f.severity.value}] {f.message}")
                    if f.suggested_fix:
                        lines.append(f"            Fix: {f.suggested_fix}")

        lines.append("")
        lines.append(f"  Overall: {'✅ ALL PASSED' if all_passed else '❌ ISSUES FOUND'}")
        return "\n".join(lines)

    def should_block(self, results: Tuple[GateResult, ...]) -> bool:
        """
        Check if any gate result has critical findings that should block output.

        Complexity: T(n) ∈ O(r) where r = results
        S(n) ∈ O(1)
        Edge Cases: no results → False
        """
        for r in results:
            if not r.passed and r.severity == GateSeverity.CRITICAL:
                return True
        return False
