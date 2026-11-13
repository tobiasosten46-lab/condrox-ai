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
Policy Enforcer — enforces rules.md policy rules on generated code.

Takes extracted PolicyRule objects and checks code against them. Returns
structured EnforcementResult with violations categorized by severity.

Complexity:
    enforce: T(n) in O(r * n) where r = rules, n = code length,
             S(n) in O(violations).

Invariants:
    - HARD_STOP violations -> blocked=True.
    - All matching rules generate violations.
    - Violations sorted by severity (HARD_STOP first).

Edge Cases:
    - No rules -> no violations, blocked=False.
    - Empty code -> no violations.
    - Rule with empty pattern -> skipped (no regex match).

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from agents.DeveloperAgent.src.policy.policy_rules import (
    PolicyRule,
    PolicyRuleExtractor,
    PolicyRuleType,
    PolicySeverity,
    PolicyViolation,
)


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EnforcementResult:
    """
    Immutable policy enforcement result.

    Complexity: T(n) in O(1), S(n) in O(violations).
    Invariants: hard_stop_count = violations with HARD_STOP severity.
    Edge Cases: no violations -> all counts zero, blocked=False.
    """

    violations: Tuple[PolicyViolation, ...]
    blocked: bool
    total_rules_checked: int
    hard_stop_count: int
    critical_count: int
    error_count: int
    warning_count: int
    elapsed_ms: int


# ─── Policy Enforcer ─────────────────────────────────────────────────────────


class PolicyEnforcer:
    """
    Enforces rules.md policy rules on generated code.

    Uses PolicyRuleExtractor to get rules, then checks code against
    each rule's regex pattern. HARD_STOP violations block output.

    Complexity:
        enforce: T(n) in O(r * n) where r = rules, n = code length,
                 S(n) in O(violations).
    Invariants:
        - HARD_STOP violations -> blocked=True.
        - Violations sorted by severity.
        - All rules with non-empty patterns checked.
    Edge Cases:
        - No rules -> no violations.
        - Empty code -> no violations.
        - Rule with empty pattern -> skipped.
    """

    __slots__ = ("_extractor", "_compiled_patterns")

    _SEVERITY_ORDER: Dict[PolicySeverity, int] = {
        PolicySeverity.HARD_STOP: 0,
        PolicySeverity.CRITICAL: 1,
        PolicySeverity.ERROR: 2,
        PolicySeverity.WARNING: 3,
        PolicySeverity.INFO: 4,
    }

    def __init__(
        self,
        extractor: PolicyRuleExtractor = None,
    ) -> None:
        """
        Initialize PolicyEnforcer with rule extractor.

        Complexity: T(n) in O(r) where r = rules, S(n) in O(patterns).
        Edge Cases: extractor=None -> default extractor created.
        """
        self._extractor: PolicyRuleExtractor = (
            extractor if extractor is not None else PolicyRuleExtractor()
        )
        self._compiled_patterns: Dict[str, re.Pattern[str]] = {}
        self._compile_patterns()

    # ─── Public API ──────────────────────────────────────────────────────────

    def enforce(
        self,
        code: str,
        file_path: str = "",
    ) -> EnforcementResult:
        """
        Enforce all policy rules on code.

        Complexity: T(n) in O(r * n) where r = rules, n = code length,
                    S(n) in O(violations).
        Invariants: HARD_STOP violations -> blocked=True.
        Edge Cases: empty code -> no violations.
        """
        start_time: float = time.perf_counter()

        if not code.strip():
            return EnforcementResult(
                violations=(),
                blocked=False,
                total_rules_checked=0,
                hard_stop_count=0,
                critical_count=0,
                error_count=0,
                warning_count=0,
                elapsed_ms=0,
            )

        rules: Tuple[PolicyRule, ...] = self._extractor.extract_all()
        violations: List[PolicyViolation] = []

        for rule in rules:
            if not rule.pattern:
                continue

            pattern: re.Pattern[str] = self._compiled_patterns.get(
                rule.rule_id
            )
            if pattern is None:
                continue

            for match in pattern.finditer(code):
                line_num: int = code[:match.start()].count("\n") + 1
                context: str = self._extract_context(code, match.start(), 50)
                violations.append(PolicyViolation(
                    rule_id=rule.rule_id,
                    rule_type=rule.rule_type,
                    severity=rule.severity,
                    message=f"{rule.name}: {rule.description}",
                    context=context,
                    line=line_num,
                ))

        violations.sort(
            key=lambda v: self._SEVERITY_ORDER.get(v.severity, 99)
        )

        hard_stops: int = sum(
            1 for v in violations
            if v.severity == PolicySeverity.HARD_STOP
        )
        criticals: int = sum(
            1 for v in violations
            if v.severity == PolicySeverity.CRITICAL
        )
        errors: int = sum(
            1 for v in violations
            if v.severity == PolicySeverity.ERROR
        )
        warnings: int = sum(
            1 for v in violations
            if v.severity == PolicySeverity.WARNING
        )

        elapsed_ms: int = int((time.perf_counter() - start_time) * 1000)

        return EnforcementResult(
            violations=tuple(violations),
            blocked=hard_stops > 0,
            total_rules_checked=len(rules),
            hard_stop_count=hard_stops,
            critical_count=criticals,
            error_count=errors,
            warning_count=warnings,
            elapsed_ms=elapsed_ms,
        )

    def enforce_type(
        self,
        code: str,
        rule_type: PolicyRuleType,
        file_path: str = "",
    ) -> EnforcementResult:
        """
        Enforce only rules of a specific type.

        Complexity: T(n) in O(r * n) where r = rules of type,
                    S(n) in O(violations).
        Edge Cases: no rules of type -> no violations.
        """
        start_time: float = time.perf_counter()

        rules: Tuple[PolicyRule, ...] = self._extractor.extract_type(
            rule_type
        )

        violations: List[PolicyViolation] = []

        for rule in rules:
            if not rule.pattern:
                continue

            pattern: re.Pattern[str] = self._compiled_patterns.get(
                rule.rule_id
            )
            if pattern is None:
                continue

            for match in pattern.finditer(code):
                line_num: int = code[:match.start()].count("\n") + 1
                context: str = self._extract_context(code, match.start(), 50)
                violations.append(PolicyViolation(
                    rule_id=rule.rule_id,
                    rule_type=rule.rule_type,
                    severity=rule.severity,
                    message=f"{rule.name}: {rule.description}",
                    context=context,
                    line=line_num,
                ))

        violations.sort(
            key=lambda v: self._SEVERITY_ORDER.get(v.severity, 99)
        )

        hard_stops: int = sum(
            1 for v in violations
            if v.severity == PolicySeverity.HARD_STOP
        )

        elapsed_ms: int = int((time.perf_counter() - start_time) * 1000)

        return EnforcementResult(
            violations=tuple(violations),
            blocked=hard_stops > 0,
            total_rules_checked=len(rules),
            hard_stop_count=hard_stops,
            critical_count=sum(1 for v in violations if v.severity == PolicySeverity.CRITICAL),
            error_count=sum(1 for v in violations if v.severity == PolicySeverity.ERROR),
            warning_count=sum(1 for v in violations if v.severity == PolicySeverity.WARNING),
            elapsed_ms=elapsed_ms,
        )

    def format_report(self, result: EnforcementResult) -> str:
        """
        Format enforcement result as human-readable report.

        Complexity: T(n) in O(v) where v = violations, S(n) in O(report).
        Edge Cases: no violations -> "Policy: CLEAN".
        """
        if not result.violations:
            return (
                f"Policy: CLEAN — {result.total_rules_checked} rules checked, "
                f"0 violations ({result.elapsed_ms}ms)"
            )

        lines: List[str] = [
            f"Policy Report: {len(result.violations)} violations "
            f"({result.total_rules_checked} rules checked, "
            f"{result.elapsed_ms}ms)",
            f"  HARD_STOP: {result.hard_stop_count}",
            f"  CRITICAL:  {result.critical_count}",
            f"  ERROR:     {result.error_count}",
            f"  WARNING:   {result.warning_count}",
            f"  BLOCKED:   {'YES' if result.blocked else 'NO'}",
            "",
        ]

        for v in result.violations[:20]:
            lines.append(
                f"  [{v.severity.value.upper()}] Line {v.line}: "
                f"{v.rule_id} — {v.message}"
            )

        if len(result.violations) > 20:
            lines.append(
                f"  ... and {len(result.violations) - 20} more"
            )

        return "\n".join(lines)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _compile_patterns(self) -> None:
        """
        Pre-compile all rule regex patterns for O(1) matching.

        Complexity: T(n) in O(r) where r = rules, S(n) in O(patterns).
        Edge Cases: invalid pattern -> skipped (no crash).
        """
        rules: Tuple[PolicyRule, ...] = self._extractor.extract_all()
        for rule in rules:
            if not rule.pattern:
                continue
            try:
                self._compiled_patterns[rule.rule_id] = re.compile(
                    rule.pattern
                )
            except re.error:
                continue

    @staticmethod
    def _extract_context(
        code: str, pos: int, radius: int
    ) -> str:
        """
        Extract context around a position in code.

        Complexity: T(n) in O(1), S(n) in O(context).
        Edge Cases: pos near start/end -> shorter context.
        """
        start: int = max(0, pos - radius)
        end: int = min(len(code), pos + radius)
        return code[start:end].replace("\n", "\\n")
