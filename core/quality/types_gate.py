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
TYPES Quality Gate — checks Python type annotations via AST.

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


class TypesGate(QualityGate):
    """
    TYPES gate — verifies 100% type annotation coverage.

    Complexity: T(check) ∈ O(n) where n = AST nodes, S(check) ∈ O(findings)
    Invariants: all public functions must have return + parameter types
    Edge Cases: non-Python → skip; empty code → pass
    """

    @property
    def name(self) -> str:
        return "TYPES"

    @property
    def enabled(self) -> bool:
        return True

    async def check(
        self, code: str, file_path: str, language: str
    ) -> GateResult:
        """
        Check type annotation coverage.

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
                    rule_id="TYPES-000",
                    message=f"Syntax error: {e.msg}",
                    file_path=file_path,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    severity=GateSeverity.CRITICAL,
                ),),
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns is None:
                    findings.append(Finding(
                        rule_id="TYPES-002",
                        message=f"Function '{node.name}' missing return type annotation",
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        severity=GateSeverity.ERROR,
                        suggested_fix="Add return type annotation (e.g. '-> int')",
                    ))

                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != "self":
                        findings.append(Finding(
                            rule_id="TYPES-003",
                            message=f"Parameter '{arg.arg}' in '{node.name}' missing type annotation",
                            file_path=file_path,
                            line=arg.lineno or node.lineno,
                            column=arg.col_offset or node.col_offset,
                            severity=GateSeverity.ERROR,
                            suggested_fix=f"Add type annotation to '{arg.arg}'",
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
