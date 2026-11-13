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
SAFETY Quality Gate — detects secrets, injection, and dangerous patterns.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re

from core.quality.gate_base import (
    Finding,
    GateResult,
    GateSeverity,
    QualityGate,
)


class SafetyGate(QualityGate):
    """
    SAFETY gate — detects hardcoded secrets and dangerous code patterns.

    Complexity: T(check) ∈ O(p×n) where p = patterns, n = code length
    Invariants: zero tolerance for secrets and injection
    Edge Cases: empty code → pass
    """

    _SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?:sk|pk|api_key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9]{20,}['\"]", re.IGNORECASE),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
        re.compile(r"gho_[A-Za-z0-9]{36}"),
    )

    _DANGER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("eval() usage", re.compile(r"\beval\s*\(")),
        ("exec() usage", re.compile(r"\bexec\s*\(")),
        ("os.system() usage", re.compile(r"\bos\.system\s*\(")),
        ("subprocess shell=True", re.compile(r"shell\s*=\s*True")),
        ("SQL string concatenation", re.compile(r"(?:SELECT|INSERT|UPDATE|DELETE).*?\+.*?(?:WHERE|VALUES|SET)", re.IGNORECASE | re.DOTALL)),
        ("pickle.loads() — deserialization risk", re.compile(r"\bpickle\.loads?\s*\(")),
        ("yaml.load() without SafeLoader", re.compile(r"\byaml\.load\s*\((?!.*SafeLoader)")),
        ("hardcoded password", re.compile(r"password\s*[=:]\s*['\"][^'\"]{3,}['\"]", re.IGNORECASE)),
    )

    @property
    def name(self) -> str:
        return "SAFETY"

    @property
    def enabled(self) -> bool:
        return True

    async def check(
        self, code: str, file_path: str, language: str
    ) -> GateResult:
        """
        Check for security issues.

        Complexity: T(n) ∈ O(p×n) where p = patterns, n = code length
        Edge Cases: empty code → pass
        """
        if not code.strip():
            return GateResult(
                gate_name=self.name,
                passed=True,
                severity=GateSeverity.INFO,
                score=1.0,
            )

        findings: list[Finding] = []

        for pattern in self._SECRET_PATTERNS:
            for match in pattern.finditer(code):
                line_num: int = code[:match.start()].count("\n") + 1
                findings.append(Finding(
                    rule_id="SAFE-001",
                    message=f"Potential hardcoded secret detected: {match.group()[:30]}...",
                    file_path=file_path,
                    line=line_num,
                    column=0,
                    severity=GateSeverity.CRITICAL,
                    suggested_fix="Use environment variables for secrets",
                ))

        for label, pattern in self._DANGER_PATTERNS:
            for match in pattern.finditer(code):
                line_num = code[:match.start()].count("\n") + 1
                findings.append(Finding(
                    rule_id="SAFE-002",
                    message=f"Security risk: {label}",
                    file_path=file_path,
                    line=line_num,
                    column=0,
                    severity=GateSeverity.ERROR,
                    suggested_fix=f"Avoid {label}",
                ))

        critical: int = sum(1 for f in findings if f.severity == GateSeverity.CRITICAL)
        errors: int = sum(1 for f in findings if f.severity == GateSeverity.ERROR)
        score: float = max(0.0, 1.0 - (critical * 0.5 + errors * 0.2))

        return GateResult(
            gate_name=self.name,
            passed=critical == 0 and errors == 0,
            severity=(
                GateSeverity.CRITICAL if critical > 0
                else GateSeverity.ERROR if errors > 0
                else GateSeverity.INFO
            ),
            score=score,
            findings=tuple(findings),
            auto_fixable=any(f.suggested_fix for f in findings),
        )
