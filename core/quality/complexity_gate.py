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
COMPLEXITY Quality Gate — checks cyclomatic complexity and O() notation.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import ast
import re

from core.quality.gate_base import (
    Finding,
    GateResult,
    GateSeverity,
    QualityGate,
)


class ComplexityGate(QualityGate):
    """
    COMPLEXITY gate — verifies complexity bounds and O() documentation.

    Complexity: T(check) ∈ O(n) where n = AST nodes, S(check) ∈ O(findings)
    Invariants: max_complexity default 10
    Edge Cases: empty code → pass; non-Python → skip
    """

    _COMPLEXITY_PATTERN: re.Pattern[str] = re.compile(
        r"[Tt]\(n\)\s*∈\s*O\(", re.MULTILINE
    )

    def __init__(self, max_complexity: int = 10) -> None:
        self._max: int = max_complexity

    @property
    def name(self) -> str:
        return "COMPLEXITY"

    @property
    def enabled(self) -> bool:
        return True

    async def check(
        self, code: str, file_path: str, language: str
    ) -> GateResult:
        """
        Check cyclomatic complexity and O() notation.

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
                    rule_id="CPLX-000",
                    message=f"Syntax error: {e.msg}",
                    file_path=file_path,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    severity=GateSeverity.CRITICAL,
                ),),
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity: int = self._cyclomatic_complexity(node)

                if complexity > self._max:
                    findings.append(Finding(
                        rule_id="CPLX-001",
                        message=(
                            f"Function '{node.name}' has cyclomatic complexity "
                            f"{complexity} (max {self._max})"
                        ),
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        severity=GateSeverity.ERROR,
                        suggested_fix="Break function into smaller functions",
                    ))

                docstring: str | None = ast.get_docstring(node)
                if docstring and not self._COMPLEXITY_PATTERN.search(docstring):
                    findings.append(Finding(
                        rule_id="CPLX-002",
                        message=f"Function '{node.name}' missing O() notation in docstring",
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        severity=GateSeverity.WARNING,
                        suggested_fix="Add 'Complexity: T(n) ∈ O(...)' to docstring",
                    ))

        errors: int = sum(1 for f in findings if f.severity == GateSeverity.ERROR)
        warnings: int = sum(1 for f in findings if f.severity == GateSeverity.WARNING)
        score: float = max(0.0, 1.0 - (errors * 0.15 + warnings * 0.05))

        return GateResult(
            gate_name=self.name,
            passed=errors == 0,
            severity=(
                GateSeverity.ERROR if errors > 0
                else GateSeverity.WARNING if warnings > 0
                else GateSeverity.INFO
            ),
            score=score,
            findings=tuple(findings),
            auto_fixable=any(f.suggested_fix for f in findings),
        )

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity of a function.

        Complexity: T(n) ∈ O(n) where n = AST nodes, S(n) ∈ O(1)
        Edge Cases: empty function → complexity 1
        """
        complexity: int = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1

        return complexity
