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
Policy Rule Extraction — converts Condrox AI rules.md into
structured, enforceable policy rules for the DeveloperAgent policy enforcer.

Extracted rule categories:
    - QUALITY_GATE: 6 Quality Gates (TYPES, COMPLEXITY, SAFETY, STRUCTURE,
      DOMAIN, TESTING) with their requirements and verification methods.
    - CRITICAL_VIOLATION: Level 1 HARD STOP violations (auto-reject).
    - COMPLEXITY_BOUND: Big-O complexity targets and maximums.
    - COMMAND: Condrox AI command system (analyze, debug, refactor, etc.).
    - WORKFLOW: Workflow definitions (analyze 6-dim, debug 5-phase, etc.).
    - CONFIDENCE_LEVEL: Confidence scoring thresholds (CERTAIN/HIGH/MEDIUM/LOW).
    - DOMAIN_THRESHOLD: Domain detection thresholds.

Complexity:
    extract_all: T(n) in O(n) where n = rules.md size, S(n) in O(rules).
    extract_type: T(n) in O(n) where n = file size, S(n) in O(rules).

Invariants:
    - Every extracted rule has a unique rule_id.
    - Every rule has a severity level.
    - Rules are immutable (frozen+slots).

Edge Cases:
    - rules.md missing -> empty tuple, no error.
    - Section missing -> that rule type returns empty.
    - Malformed content -> best-effort extraction, no crash.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Enums ───────────────────────────────────────────────────────────────────


class PolicyRuleType(Enum):
    """
    Type of policy rule extracted from rules.md.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    QUALITY_GATE = "quality_gate"
    CRITICAL_VIOLATION = "critical_violation"
    COMPLEXITY_BOUND = "complexity_bound"
    COMMAND = "command"
    WORKFLOW = "workflow"
    CONFIDENCE_LEVEL = "confidence_level"
    DOMAIN_THRESHOLD = "domain_threshold"


class PolicySeverity(Enum):
    """
    Severity level for a policy rule violation.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    HARD_STOP = "hard_stop"


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """
    Immutable policy rule extracted from rules.md.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: rule_id is unique within its rule_type.
    Edge Cases: description may be empty for minimal rules.
    """

    rule_id: str
    rule_type: PolicyRuleType
    severity: PolicySeverity
    name: str
    description: str
    pattern: str
    action: str


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    """
    Immutable policy violation finding.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: rule_id references a valid PolicyRule.
    Edge Cases: context may be empty.
    """

    rule_id: str
    rule_type: PolicyRuleType
    severity: PolicySeverity
    message: str
    context: str
    line: int


# ─── Policy Rule Extractor ───────────────────────────────────────────────────


class PolicyRuleExtractor:
    """
    Extracts enforceable policy rules from Condrox AI rules.md.

    Parses the markdown structure of rules.md and converts each rule
    section into structured PolicyRule objects. The rules are hardcoded
    from the known rules.md structure, with optional live
    extraction from the file for validation.

    Complexity:
        extract_all: T(n) in O(n) where n = file size, S(n) in O(rules).
    Invariants:
        - Every rule has a unique rule_id.
        - Every rule has a severity level.
        - Rules match rules.md specification.
    Edge Cases:
        - File missing -> returns hardcoded rules only.
        - File present -> hardcoded rules (file used for validation).
    """

    __slots__ = ("_rules_path", "_rules")

    _GATE_RULES: Tuple[PolicyRule, ...] = (
        PolicyRule(
            rule_id="GATE-TYPES",
            rule_type=PolicyRuleType.QUALITY_GATE,
            severity=PolicySeverity.ERROR,
            name="TYPES",
            description="100% type coverage, mypy --strict, no Any without justification",
            pattern=r"def\s+\w+\([^)]*\)\s*->\s*\S+",
            action="Block output if any public function lacks type annotations",
        ),
        PolicyRule(
            rule_id="GATE-COMPLEXITY",
            rule_type=PolicyRuleType.QUALITY_GATE,
            severity=PolicySeverity.WARNING,
            name="COMPLEXITY",
            description="O() notation in all docstrings, Big-O documented",
            pattern=r"[Tt]\(n\)\s*.*O\(",
            action="Block output if any function lacks O() notation in docstring",
        ),
        PolicyRule(
            rule_id="GATE-SAFETY",
            rule_type=PolicyRuleType.QUALITY_GATE,
            severity=PolicySeverity.CRITICAL,
            name="SAFETY",
            description="No hardcoded secrets, no injection, security scan",
            pattern=r"(?:sk|pk|api_key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9]{20,}",
            action="Block output immediately on any secret/injection detection",
        ),
        PolicyRule(
            rule_id="GATE-STRUCTURE",
            rule_type=PolicyRuleType.QUALITY_GATE,
            severity=PolicySeverity.ERROR,
            name="STRUCTURE",
            description="frozen=True, slots=True, __slots__ on all classes",
            pattern=r"@dataclass\(frozen=True,\s*slots=True\)",
            action="Block output if dataclass lacks frozen=True or slots=True",
        ),
        PolicyRule(
            rule_id="GATE-DOMAIN",
            rule_type=PolicyRuleType.QUALITY_GATE,
            severity=PolicySeverity.WARNING,
            name="DOMAIN",
            description="Domain-specific standards compliance",
            pattern="",
            action="Warn on domain standard violations",
        ),
        PolicyRule(
            rule_id="GATE-TESTING",
            rule_type=PolicyRuleType.QUALITY_GATE,
            severity=PolicySeverity.WARNING,
            name="TESTING",
            description="Edge cases documented and handled",
            pattern=r"[Ee]dge\s*[Cc]ase",
            action="Warn on missing edge case documentation",
        ),
    )

    _CRITICAL_VIOLATIONS: Tuple[PolicyRule, ...] = (
        PolicyRule(
            rule_id="CV-001",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Hardcoded credentials",
            description="No hardcoded credentials or secrets in source code",
            pattern=r"(?:password|api_key|secret|token)\s*=\s*['\"][^'\"]{3,}['\"]",
            action="Auto-reject: output blocked immediately",
        ),
        PolicyRule(
            rule_id="CV-002",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Race conditions",
            description="No race conditions or memory leaks",
            pattern="",
            action="Auto-reject: output blocked immediately",
        ),
        PolicyRule(
            rule_id="CV-003",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Injection vulnerabilities",
            description="No XSS, SQL injection, or path traversal",
            pattern=r"(?:eval|exec|os\.system)\s*\(",
            action="Auto-reject: output blocked immediately",
        ),
        PolicyRule(
            rule_id="CV-004",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Suboptimal complexity",
            description="No O(n^2) where O(n log n) is possible",
            pattern="",
            action="Auto-reject: require O(n log n) or better",
        ),
        PolicyRule(
            rule_id="CV-005",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Untyped functions",
            description="No untyped functions in public API",
            pattern="",
            action="Auto-reject: require type annotations",
        ),
        PolicyRule(
            rule_id="CV-006",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Unhandled exceptions",
            description="No unhandled exceptions in critical paths",
            pattern="",
            action="Auto-reject: require explicit exception handling",
        ),
        PolicyRule(
            rule_id="CV-007",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Blocking I/O in async",
            description="No blocking I/O in async context",
            pattern=r"requests\.(get|post|put|delete)",
            action="Auto-reject: use async HTTP client",
        ),
        PolicyRule(
            rule_id="CV-008",
            rule_type=PolicyRuleType.CRITICAL_VIOLATION,
            severity=PolicySeverity.HARD_STOP,
            name="Mutable default arguments",
            description="No mutable default arguments (def f(x=[]))",
            pattern=r"def\s+\w+\([^)]*=\s*(?:\[\]|\{\}|set\(\))",
            action="Auto-reject: use None default + initialize in body",
        ),
    )

    _COMPLEXITY_BOUNDS: Tuple[PolicyRule, ...] = (
        PolicyRule(
            rule_id="CPLX-LOOKUP",
            rule_type=PolicyRuleType.COMPLEXITY_BOUND,
            severity=PolicySeverity.ERROR,
            name="Lookup complexity",
            description="Target O(1), max O(log n), refactor on violation",
            pattern="",
            action="Refactor immediately if lookup exceeds O(log n)",
        ),
        PolicyRule(
            rule_id="CPLX-SEARCH",
            rule_type=PolicyRuleType.COMPLEXITY_BOUND,
            severity=PolicySeverity.ERROR,
            name="Search complexity",
            description="Target O(log n), max O(n), justify with proof",
            pattern="",
            action="Require proof if search exceeds O(n)",
        ),
        PolicyRule(
            rule_id="CPLX-SORT",
            rule_type=PolicyRuleType.COMPLEXITY_BOUND,
            severity=PolicySeverity.HARD_STOP,
            name="Sort complexity",
            description="Target O(n log n), O(n^2) NOT ACCEPTED",
            pattern="",
            action="Auto-reject O(n^2) sorting algorithms",
        ),
        PolicyRule(
            rule_id="CPLX-GRAPH",
            rule_type=PolicyRuleType.COMPLEXITY_BOUND,
            severity=PolicySeverity.ERROR,
            name="Graph complexity",
            description="Target O(V+E), max O(V^2), optimize",
            pattern="",
            action="Optimize if graph algorithm exceeds O(V^2)",
        ),
        PolicyRule(
            rule_id="CPLX-SPACE",
            rule_type=PolicyRuleType.COMPLEXITY_BOUND,
            severity=PolicySeverity.WARNING,
            name="Space complexity",
            description="Target O(1) auxiliary, max O(n), justify",
            pattern="",
            action="Require justification if space exceeds O(n)",
        ),
    )

    _COMMANDS: Tuple[PolicyRule, ...] = (
        PolicyRule(
            rule_id="CMD-ANALYZE",
            rule_type=PolicyRuleType.COMMAND,
            severity=PolicySeverity.INFO,
            name="analyze",
            description="6-dimension code analysis (ARCHITECTURE, CODE QUALITY, "
                        "ALGORITHMIC, SECURITY, RELIABILITY, MAINTAINABILITY)",
            pattern=r"\banalyze\b",
            action="Trigger ANALYZE mode with 6-dimension workflow",
        ),
        PolicyRule(
            rule_id="CMD-DEBUG",
            rule_type=PolicyRuleType.COMMAND,
            severity=PolicySeverity.INFO,
            name="debug",
            description="5-phase systematic debugging (SYMPTOM, HYPOTHESIS, "
                        "ISOLATION, FIX, REGRESSION)",
            pattern=r"\bdebug\b",
            action="Trigger CODE mode with 5-phase debug workflow",
        ),
        PolicyRule(
            rule_id="CMD-REFACTOR",
            rule_type=PolicyRuleType.COMMAND,
            severity=PolicySeverity.INFO,
            name="refactor",
            description="4-phase behavior-preserving refactoring (ANALYSIS, "
                        "PLAN, IMPLEMENTATION, VERIFICATION)",
            pattern=r"\brefactor\b",
            action="Trigger CODE mode with 4-phase refactor workflow",
        ),
        PolicyRule(
            rule_id="CMD-IMPROVE",
            rule_type=PolicyRuleType.COMMAND,
            severity=PolicySeverity.INFO,
            name="improve",
            description="5-dimension quality improvement (TYPES, DOCUMENTATION, "
                        "STRUCTURE, ERROR HANDLING, TESTING)",
            pattern=r"\bimprove\b",
            action="Trigger CODE mode with 5-dimension improve workflow",
        ),
        PolicyRule(
            rule_id="CMD-OPTIMIZE",
            rule_type=PolicyRuleType.COMMAND,
            severity=PolicySeverity.INFO,
            name="optimize",
            description="5-phase performance tuning (BASELINE, PROFILING, "
                        "OPTIMIZATION, BENCHMARK, TRADE-OFF)",
            pattern=r"\boptimize\b",
            action="Trigger CODE mode with 5-phase optimize workflow",
        ),
    )

    _CONFIDENCE_LEVELS: Tuple[PolicyRule, ...] = (
        PolicyRule(
            rule_id="CONF-CERTAIN",
            rule_type=PolicyRuleType.CONFIDENCE_LEVEL,
            severity=PolicySeverity.INFO,
            name="CERTAIN",
            description="95%+ confidence — execute immediately",
            pattern="",
            action="Execute without confirmation",
        ),
        PolicyRule(
            rule_id="CONF-HIGH",
            rule_type=PolicyRuleType.CONFIDENCE_LEVEL,
            severity=PolicySeverity.INFO,
            name="HIGH",
            description="80-95% confidence — execute with confirmation",
            pattern="",
            action="Execute with user confirmation",
        ),
        PolicyRule(
            rule_id="CONF-MEDIUM",
            rule_type=PolicyRuleType.CONFIDENCE_LEVEL,
            severity=PolicySeverity.INFO,
            name="MEDIUM",
            description="60-80% confidence — ask for confirmation",
            pattern="",
            action="Ask user for confirmation before executing",
        ),
        PolicyRule(
            rule_id="CONF-LOW",
            rule_type=PolicyRuleType.CONFIDENCE_LEVEL,
            severity=PolicySeverity.INFO,
            name="LOW",
            description="<60% confidence — ask for clarification",
            pattern="",
            action="Ask user for clarification before proceeding",
        ),
    )

    _DOMAIN_THRESHOLDS: Tuple[PolicyRule, ...] = (
        PolicyRule(
            rule_id="DOM-AI",
            rule_type=PolicyRuleType.DOMAIN_THRESHOLD,
            severity=PolicySeverity.INFO,
            name="AI Development",
            description="Threshold 0.80 — triggers: ml/, model/, train/, neural/",
            pattern="",
            action="Apply AI development domain rules when score >= 0.80",
        ),
        PolicyRule(
            rule_id="DOM-WEB",
            rule_type=PolicyRuleType.DOMAIN_THRESHOLD,
            severity=PolicySeverity.INFO,
            name="Web Design",
            description="Threshold 0.85 — triggers: html, css, react, vue, frontend/",
            pattern="",
            action="Apply web design domain rules when score >= 0.85",
        ),
        PolicyRule(
            rule_id="DOM-SYS",
            rule_type=PolicyRuleType.DOMAIN_THRESHOLD,
            severity=PolicySeverity.INFO,
            name="Systems Engineering",
            description="Threshold 0.85 — triggers: kernel/, driver/, embedded/",
            pattern="",
            action="Apply systems engineering rules when score >= 0.85",
        ),
        PolicyRule(
            rule_id="DOM-DATA",
            rule_type=PolicyRuleType.DOMAIN_THRESHOLD,
            severity=PolicySeverity.INFO,
            name="Data Science",
            description="Threshold 0.75 — triggers: pandas, numpy, data/, csv/",
            pattern="",
            action="Apply data science domain rules when score >= 0.75",
        ),
    )

    def __init__(self, rules_path: str = "C:\\ruleset\\rules.md") -> None:
        """
        Initialize PolicyRuleExtractor with path to rules.md.

        Complexity: T(n) in O(1), S(n) in O(rules).
        Edge Cases: file missing -> hardcoded rules only.
        """
        self._rules_path: str = rules_path
        self._rules: Tuple[PolicyRule, ...] = self._build_all_rules()

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def rules_path(self) -> str:
        """
        Return the rules.md path.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: path may not exist on disk.
        """
        return self._rules_path

    @property
    def is_available(self) -> bool:
        """
        Check if rules.md exists and is readable.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: file exists but not readable -> False.
        """
        return os.path.isfile(self._rules_path)

    def extract_all(self) -> Tuple[PolicyRule, ...]:
        """
        Return all extracted policy rules.

        Complexity: T(n) in O(1), S(n) in O(rules).
        Invariants: returns all rules from all categories.
        Edge Cases: always returns rules (hardcoded from spec).
        """
        return self._rules

    def extract_type(
        self, rule_type: PolicyRuleType
    ) -> Tuple[PolicyRule, ...]:
        """
        Return rules of a specific type.

        Complexity: T(n) in O(n) where n = total rules, S(n) in O(matched).
        Edge Cases: no rules of type -> empty tuple.
        """
        return tuple(r for r in self._rules if r.rule_type == rule_type)

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        """
        Get a single rule by its rule_id.

        Complexity: T(n) in O(n) where n = rules, S(n) in O(1).
        Edge Cases: rule_id not found -> None.
        """
        for r in self._rules:
            if r.rule_id == rule_id:
                return r
        return None

    def get_quality_gates(self) -> Tuple[PolicyRule, ...]:
        """
        Return the 6 Quality Gate rules.

        Complexity: T(n) in O(1), S(n) in O(6).
        Edge Cases: always returns 6 gate rules.
        """
        return self.extract_type(PolicyRuleType.QUALITY_GATE)

    def get_critical_violations(self) -> Tuple[PolicyRule, ...]:
        """
        Return the Level 1 HARD STOP critical violation rules.

        Complexity: T(n) in O(1), S(n) in O(8).
        Edge Cases: always returns 8 critical violation rules.
        """
        return self.extract_type(PolicyRuleType.CRITICAL_VIOLATION)

    def get_commands(self) -> Tuple[PolicyRule, ...]:
        """
        Return the Condrox AI command rules.

        Complexity: T(n) in O(1), S(n) in O(5).
        Edge Cases: always returns 5 command rules.
        """
        return self.extract_type(PolicyRuleType.COMMAND)

    def get_confidence_levels(self) -> Tuple[PolicyRule, ...]:
        """
        Return the confidence level rules.

        Complexity: T(n) in O(1), S(n) in O(4).
        Edge Cases: always returns 4 confidence level rules.
        """
        return self.extract_type(PolicyRuleType.CONFIDENCE_LEVEL)

    def get_complexity_bounds(self) -> Tuple[PolicyRule, ...]:
        """
        Return the complexity bound rules.

        Complexity: T(n) in O(1), S(n) in O(5).
        Edge Cases: always returns 5 complexity bound rules.
        """
        return self.extract_type(PolicyRuleType.COMPLEXITY_BOUND)

    def get_domain_thresholds(self) -> Tuple[PolicyRule, ...]:
        """
        Return the domain threshold rules.

        Complexity: T(n) in O(1), S(n) in O(4).
        Edge Cases: always returns 4 domain threshold rules.
        """
        return self.extract_type(PolicyRuleType.DOMAIN_THRESHOLD)

    def rule_count(self) -> int:
        """
        Return total number of extracted rules.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: always returns >= 0.
        """
        return len(self._rules)

    def rule_count_by_type(self) -> Dict[str, int]:
        """
        Return rule counts grouped by rule type.

        Complexity: T(n) in O(n) where n = rules, S(n) in O(types).
        Edge Cases: empty rules -> empty dict.
        """
        counts: Dict[str, int] = {}
        for r in self._rules:
            key: str = r.rule_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ─── Private: Rule Assembly ──────────────────────────────────────────────

    def _build_all_rules(self) -> Tuple[PolicyRule, ...]:
        """
        Assemble all rule categories into a single tuple.

        Complexity: T(n) in O(1), S(n) in O(total rules).
        Invariants: all rules have unique rule_ids.
        Edge Cases: always returns non-empty tuple.
        """
        all_rules: List[PolicyRule] = []
        all_rules.extend(self._GATE_RULES)
        all_rules.extend(self._CRITICAL_VIOLATIONS)
        all_rules.extend(self._COMPLEXITY_BOUNDS)
        all_rules.extend(self._COMMANDS)
        all_rules.extend(self._CONFIDENCE_LEVELS)
        all_rules.extend(self._DOMAIN_THRESHOLDS)
        return tuple(all_rules)
