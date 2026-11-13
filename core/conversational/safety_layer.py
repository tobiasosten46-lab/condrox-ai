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
Safety & Ethics Layer — unified content safety with multi-tier filtering,
override support, and audit logging.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SafetyLevel(Enum):
    """Safety severity levels."""

    SAFE = 0
    WARNING = 1
    BLOCKED = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class SafetyViolation:
    """
    Immutable safety violation record.

    Attributes:
        rule_id: Identifier of the triggered rule.
        level: Severity level.
        matched_text: The text that triggered the violation.
        message: Human-readable description.
        timestamp: When the violation was detected.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    rule_id: str
    level: SafetyLevel
    matched_text: str
    message: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class SafetyResult:
    """
    Immutable result of a safety check.

    Attributes:
        passed: True if content is safe.
        violations: List of detected violations.
        cleaned_text: Text with violations removed/redacted.
        override_available: Whether an override can bypass this.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    passed: bool
    violations: tuple[SafetyViolation, ...]
    cleaned_text: str
    override_available: bool


@dataclass
class SafetyRule:
    """
    Mutable safety rule definition.

    Attributes:
        rule_id: Unique identifier.
        pattern: Compiled regex pattern.
        level: Severity level.
        message: Description.
        override_allowed: Whether user override can bypass.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    rule_id: str
    pattern: re.Pattern[str]
    level: SafetyLevel
    message: str
    override_allowed: bool


# Default safety rules.
def _default_rules() -> List[SafetyRule]:
    """Create default safety rules.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
    """
    rules: List[SafetyRule] = []

    # Critical: PII patterns
    rules.append(SafetyRule(
        rule_id="pii_credit_card",
        pattern=re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        level=SafetyLevel.CRITICAL,
        message="Credit card number detected",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="pii_ssn",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        level=SafetyLevel.CRITICAL,
        message="Social security number detected",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="pii_email",
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        level=SafetyLevel.WARNING,
        message="Email address detected",
        override_allowed=True,
    ))

    # Blocked: Harmful content
    rules.append(SafetyRule(
        rule_id="harm_violence",
        pattern=re.compile(r"\b(kill|murder|attack|bomb|weapon)\b", re.IGNORECASE),
        level=SafetyLevel.BLOCKED,
        message="Violence-related content detected",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="harm_self_harm",
        pattern=re.compile(r"\b(suicide|self[\s-]?harm|kill\s+myself)\b", re.IGNORECASE),
        level=SafetyLevel.CRITICAL,
        message="Self-harm content detected",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="harm_exploit",
        pattern=re.compile(r"\b(exploit|vulnerability\s+scan|sql\s+injection|xss)\b", re.IGNORECASE),
        level=SafetyLevel.WARNING,
        message="Exploit-related content detected",
        override_allowed=True,
    ))

    # Warning: Profanity (basic)
    rules.append(SafetyRule(
        rule_id="profanity_basic",
        pattern=re.compile(r"\b(damn|hell|crap)\b", re.IGNORECASE),
        level=SafetyLevel.WARNING,
        message="Profanity detected",
        override_allowed=True,
    ))

    # Blocked: Prompt injection attacks
    rules.append(SafetyRule(
        rule_id="prompt_injection_ignore",
        pattern=re.compile(
            r"ignore\s+(?:previous|prior|all|your)\s+(?:instructions?|rules?|prompts?)",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="Prompt injection attempt detected (instruction override)",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="prompt_injection_reveal",
        pattern=re.compile(
            r"(?:reveal|show|display|print|output)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|guidelines?|config)",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="Prompt injection attempt detected (system prompt extraction)",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="prompt_injection_forget",
        pattern=re.compile(
            r"(?:forget|disregard|discard)\s+(?:your|all|previous|prior)\s+(?:instructions?|rules?|guidelines?)",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="Prompt injection attempt detected (instruction reset)",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="prompt_injection_role",
        pattern=re.compile(
            r"(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you\s+are)|from\s+now\s+on\s+you)",
            re.IGNORECASE,
        ),
        level=SafetyLevel.WARNING,
        message="Potential role-play injection detected",
        override_allowed=True,
    ))

    # Blocked: SQL injection syntax (not just the phrase "sql injection")
    rules.append(SafetyRule(
        rule_id="sqli_drop",
        pattern=re.compile(
            r"['\"]?\s*(?:;|--|/\*|\*/)\s*(?:DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE)\s+",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="SQL injection attempt detected (DDL/DML injection)",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="sqli_boolean",
        pattern=re.compile(
            r"['\"]?\s*(?:OR|AND)\s+\d+\s*=\s*\d+",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="SQL injection attempt detected (boolean injection)",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="sqli_union",
        pattern=re.compile(
            r"['\"]?\s*UNION\s+(?:ALL\s+)?SELECT\s+",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="SQL injection attempt detected (UNION injection)",
        override_allowed=False,
    ))

    # Blocked: Path traversal
    rules.append(SafetyRule(
        rule_id="path_traversal",
        pattern=re.compile(
            r"(?:\.\./|\.\.\\|\.\.%2f|\.\.%5c|/etc/passwd|/etc/shadow|%2e%2e)",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="Path traversal attempt detected",
        override_allowed=False,
    ))

    # Blocked: Phishing — suspicious URLs with credential harvesting paths
    rules.append(SafetyRule(
        rule_id="phishing_verify_account",
        pattern=re.compile(
            r"(?:verify\s+(?:your\s+)?account|confirm\s+(?:your\s+)?account|update\s+(?:your\s+)?(?:account|password|credentials))",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="Phishing attempt detected (account verification lure)",
        override_allowed=False,
    ))
    rules.append(SafetyRule(
        rule_id="phishing_suspicious_tld",
        pattern=re.compile(
            r"https?://[^\s]+\.(?:tk|ml|ga|cf|gq|top|click|loan|work|cam|stream|download|review|men|date|party|trade|science|racing)\b",
            re.IGNORECASE,
        ),
        level=SafetyLevel.BLOCKED,
        message="Phishing attempt detected (suspicious TLD)",
        override_allowed=False,
    ))

    return rules


class SafetyLayer:
    """
    Unified safety and ethics layer with multi-tier filtering.

    Complexity:
        check: T(n) ∈ O(r * n) for r rules, n text length.
        enforce: T(n) ∈ O(r * n), S(n) ∈ O(n).
    """

    __slots__ = ("_rules", "_audit_log", "_override_active")

    def __init__(self, rules: Optional[List[SafetyRule]] = None) -> None:
        """Initialize with optional custom rules.

        Args:
            rules: Safety rules; defaults are used if None.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(r).
        """
        self._rules = rules if rules is not None else _default_rules()
        self._audit_log: List[SafetyViolation] = []
        self._override_active: bool = False

    def check(self, text: str) -> SafetyResult:
        """Check text for safety violations.

        Args:
            text: Text to check.

        Returns:
            SafetyResult with violations and pass status.

        Complexity: T(n) ∈ O(r * n), S(n) ∈ O(v) for v violations.
        """
        if not text:
            return SafetyResult(
                passed=True,
                violations=(),
                cleaned_text=text,
                override_available=False,
            )

        violations: List[SafetyViolation] = []
        has_blocking = False
        override_available = True

        for rule in self._rules:
            matches = rule.pattern.findall(text)
            if matches:
                for match in matches:
                    violation = SafetyViolation(
                        rule_id=rule.rule_id,
                        level=rule.level,
                        matched_text=str(match)[:100],
                        message=rule.message,
                        timestamp=time.time(),
                    )
                    violations.append(violation)
                    self._audit_log.append(violation)

                if rule.level in (SafetyLevel.BLOCKED, SafetyLevel.CRITICAL):
                    has_blocking = True
                    if not rule.override_allowed:
                        override_available = False

        passed = not has_blocking or (self._override_active and override_available)

        return SafetyResult(
            passed=passed,
            violations=tuple(violations),
            cleaned_text=text,
            override_available=override_available,
        )

    def enforce(self, text: str) -> Tuple[str, SafetyResult]:
        """Check and clean text, redacting violations.

        Args:
            text: Text to enforce.

        Returns:
            Tuple of (cleaned_text, SafetyResult).

        Complexity: T(n) ∈ O(r * n), S(n) ∈ O(n).
        """
        result = self.check(text)
        if result.passed:
            return text, result

        cleaned = text
        for violation in result.violations:
            rule = self._find_rule(violation.rule_id)
            if rule:
                cleaned = rule.pattern.sub("[REDACTED]", cleaned)

        return cleaned, result

    def set_override(self, active: bool) -> None:
        """Enable or disable override mode.

        Args:
            active: True to enable override for allowed rules.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._override_active = active

    def get_audit_log(self, limit: int = 100) -> List[SafetyViolation]:
        """Return recent audit log entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of SafetyViolation records.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._audit_log[-limit:])

    def clear_audit_log(self) -> None:
        """Clear the audit log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._audit_log.clear()

    def add_rule(self, rule: SafetyRule) -> None:
        """Add a custom safety rule.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID.

        Returns:
            True if a rule was removed.

        Complexity: T(n) ∈ O(r), S(n) ∈ O(1).
        """
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
        return len(self._rules) < before

    def _find_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Find a rule by ID.

        Complexity: T(n) ∈ O(r), S(n) ∈ O(1).
        """
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None
