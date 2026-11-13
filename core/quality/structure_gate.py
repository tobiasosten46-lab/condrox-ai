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
STRUCTURE Quality Gate — checks frozen=True, slots=True on dataclasses.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import ast

from core.quality.gate_base import (
    Finding,
    GateResult,
    GateSeverity,
    QualityGate,
)


class StructureGate(QualityGate):
    """
    STRUCTURE gate — verifies dataclass usage with frozen and slots.

    Complexity: T(check) ∈ O(n) where n = AST nodes, S(check) ∈ O(findings)
    Invariants: all @dataclass must have frozen=True and slots=True
    Edge Cases: non-Python → skip; no dataclasses → pass
    """

    @property
    def name(self) -> str:
        return "STRUCTURE"

    @property
    def enabled(self) -> bool:
        return True

    async def check(
        self, code: str, file_path: str, language: str
    ) -> GateResult:
        """
        Check dataclass structure compliance.

        Complexity: T(n) ∈ O(n) where n = AST nodes, S(n) ∈ O(findings)
        Edge Cases: non-Python → skip; empty → pass
        """
        if not code.strip() or language != "python":
            return GateResult(
                gate_name=self.name,
                passed=True,
                severity=GateSeverity.INFO,
                score=1.0,
            )

        findings: list[Finding] = []

        try:
            tree: ast.Module = ast.parse(code)
        except SyntaxError as e:
            return GateResult(
                gate_name=self.name,
                passed=False,
                severity=GateSeverity.CRITICAL,
                score=0.0,
                findings=(Finding(
                    rule_id="STRC-000",
                    message=f"Syntax error: {e.msg}",
                    file_path=file_path,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    severity=GateSeverity.CRITICAL,
                ),),
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_dataclass: bool = False
                frozen: bool = False
                slots: bool = False

                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                        has_dataclass = True
                    elif isinstance(decorator, ast.Call):
                        if (
                            isinstance(decorator.func, ast.Name)
                            and decorator.func.id == "dataclass"
                        ):
                            has_dataclass = True
                            for kw in decorator.keywords:
                                if kw.arg == "frozen" and isinstance(kw.value, ast.Constant):
                                    frozen = bool(kw.value.value)
                                if kw.arg == "slots" and isinstance(kw.value, ast.Constant):
                                    slots = bool(kw.value.value)

                if has_dataclass:
                    if not frozen:
                        findings.append(Finding(
                            rule_id="STRC-001",
                            message=f"Dataclass '{node.name}' missing frozen=True",
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            severity=GateSeverity.ERROR,
                            suggested_fix="Add frozen=True to @dataclass decorator",
                        ))

                    if not slots:
                        findings.append(Finding(
                            rule_id="STRC-002",
                            message=f"Dataclass '{node.name}' missing slots=True",
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            severity=GateSeverity.ERROR,
                            suggested_fix="Add slots=True to @dataclass decorator",
                        ))

        errors: int = sum(1 for f in findings if f.severity == GateSeverity.ERROR)
        score: float = max(0.0, 1.0 - (errors * 0.15))

        return GateResult(
            gate_name=self.name,
            passed=errors == 0,
            severity=GateSeverity.ERROR if errors > 0 else GateSeverity.INFO,
            score=score,
            findings=tuple(findings),
            auto_fixable=any(f.suggested_fix for f in findings),
        )
