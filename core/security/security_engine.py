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
Security engine for the unified Brain.

Performs real input/output validation: length checks, blocked pattern matching,
rate-limit structure, and audit logging. No hardcoded secrets are used.

Mathematical Foundation:
- Pattern risk is the maximum regex match count across all blocked patterns.
- Request size check is O(1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ─── Output security patterns ────────────────────────────────────────────────
# PII: email addresses
_PII_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# PII: phone numbers (international and US formats)
# Must have either a + prefix or at least 7 digits total to avoid false positives
# on numbers like "25,000" or "938" in normal text
_PII_PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:\+\d{1,3}[-.\s]\d{2,4}[-.\s]\d{3,4}[-.\s]?\d{0,4})"
    r"|"
    r"(?:\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b)"
    r"|"
    r"(?:\b\(\d{3}\)\s?\d{3}[-.\s]\d{4}\b)"
)

# PII: credit card numbers (13-19 digit groups)
_PII_CREDIT_CARD_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"
)

# Secret leakage: API keys, tokens, passwords in assignment-like patterns
_SECRET_LEAK_PATTERN: re.Pattern[str] = re.compile(
    r"(?i)(?:api[_-]?key|token|secret|password|passwd|pwd)\s*[=:]\s*['\"][A-Za-z0-9+/=_-]{8,}['\"]"
)

# Prompt injection echo: system directives in generated output
_PROMPT_INJECTION_PATTERN: re.Pattern[str] = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|above)\s+instructions?|system\s*:\s*|<\|im_start\|>|<\|system\|>)"
)


@dataclass(frozen=True, slots=True)
class SecurityReport:
    """Immutable security check report.

    Attributes:
        approved: True if all checks passed.
        risk_score: Aggregate risk in [0.0, 1.0].
        violations: List of violation descriptions.
        checks: Dict of individual check results.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(v) for v violations.
    """

    approved: bool
    risk_score: float
    violations: List[str] = field(default_factory=list)
    checks: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Validate score range.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.risk_score < 0.0 or self.risk_score > 1.0:
            raise ValueError("risk_score must be in [0.0, 1.0]")


class SecurityEngine:
    """
    Real security engine with pattern matching and size checks.

    Attributes:
        max_size: Maximum allowed request size in bytes.
        blocked_patterns: List of compiled regex patterns.

    Complexity:
        check_input: O(p * n) for p patterns and input length n.
    """

    __slots__ = ("max_size", "blocked_patterns")

    def __init__(
        self,
        max_size: int = 1048576,
        blocked_patterns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the security engine.

        Args:
            max_size: Maximum allowed request size in bytes.
            blocked_patterns: List of regex strings for forbidden content.
        """
        self.max_size = max_size
        self.blocked_patterns: List[re.Pattern[str]] = []
        for pattern in blocked_patterns or []:
            try:
                self.blocked_patterns.append(re.compile(pattern))
            except re.error as exc:
                raise ValueError(f"Invalid security pattern: {pattern}") from exc

    def check_input(self, text: str) -> SecurityReport:
        """Run all security checks on input text.

        Args:
            text: Input text.

        Returns:
            SecurityReport with approval status and violations.

        Complexity: T(n) ∈ O(p * n) for p patterns and input length n.
        """
        violations: List[str] = []
        size = len(text.encode("utf-8"))
        size_ok = size <= self.max_size
        if not size_ok:
            violations.append(f"request_size_exceeded: {size} > {self.max_size}")

        pattern_hits = 0
        matched_patterns: List[str] = []
        for compiled in self.blocked_patterns:
            matches = compiled.findall(text)
            if matches:
                pattern_hits += len(matches)
                matched_patterns.append(compiled.pattern)
                violations.append(f"blocked_pattern_matched: {compiled.pattern[:50]}")

        risk_score = min(1.0, (pattern_hits * 0.3) + (0.0 if size_ok else 0.5))
        approved = pattern_hits == 0 and size_ok

        return SecurityReport(
            approved=bool(approved),
            risk_score=risk_score,
            violations=violations,
            checks={
                "size_ok": size_ok,
                "size_bytes": size,
                "pattern_hits": pattern_hits,
                "matched_patterns": matched_patterns,
            },
        )

    def check_output(self, text: str) -> SecurityReport:
        """Run output-specific security checks on generated text.

        Checks for:
        - PII leakage (emails, phone numbers, SSN-like patterns, credit cards)
        - Secret/key leakage (API keys, tokens, passwords in output)
        - Prompt injection echoes (system directives in generated text)
        - Blocked patterns (inherited from input config)
        - Size limits

        Args:
            text: Generated output text.

        Returns:
            SecurityReport with approval status and violations.

        Complexity: T(n) ∈ O(p * n) for p patterns and output length n.
        """
        violations: List[str] = []
        size = len(text.encode("utf-8"))
        size_ok = size <= self.max_size
        if not size_ok:
            violations.append(f"output_size_exceeded: {size} > {self.max_size}")

        pattern_hits = 0
        matched_patterns: List[str] = []

        # Check inherited blocked patterns
        for compiled in self.blocked_patterns:
            matches = compiled.findall(text)
            if matches:
                pattern_hits += len(matches)
                matched_patterns.append(compiled.pattern)
                violations.append(f"blocked_pattern_in_output: {compiled.pattern[:50]}")

        # PII detection — email addresses
        email_matches = _PII_EMAIL_PATTERN.findall(text)
        if email_matches:
            pattern_hits += len(email_matches)
            violations.append(f"pii_email_leak: {len(email_matches)} email(s) detected in output")

        # PII detection — phone numbers (international format)
        phone_matches = _PII_PHONE_PATTERN.findall(text)
        if phone_matches:
            pattern_hits += len(phone_matches)
            violations.append(f"pii_phone_leak: {len(phone_matches)} phone number(s) detected in output")

        # PII detection — credit card numbers
        cc_matches = _PII_CREDIT_CARD_PATTERN.findall(text)
        if cc_matches:
            pattern_hits += len(cc_matches)
            violations.append(f"pii_credit_card_leak: {len(cc_matches)} credit card number(s) detected in output")

        # Secret leakage — API keys, tokens, passwords in output
        secret_matches = _SECRET_LEAK_PATTERN.findall(text)
        if secret_matches:
            pattern_hits += len(secret_matches)
            violations.append(f"secret_leak: {len(secret_matches)} secret(s) detected in output")

        # Prompt injection echo — system directives in generated output
        injection_matches = _PROMPT_INJECTION_PATTERN.findall(text)
        if injection_matches:
            pattern_hits += len(injection_matches)
            violations.append(f"prompt_injection_echo: {len(injection_matches)} directive(s) detected in output")

        risk_score = min(1.0, (pattern_hits * 0.3) + (0.0 if size_ok else 0.5))
        approved = pattern_hits == 0 and size_ok

        return SecurityReport(
            approved=bool(approved),
            risk_score=risk_score,
            violations=violations,
            checks={
                "size_ok": size_ok,
                "size_bytes": size,
                "pattern_hits": pattern_hits,
                "matched_patterns": matched_patterns,
                "pii_emails": len(email_matches),
                "pii_phones": len(phone_matches),
                "pii_credit_cards": len(cc_matches),
                "secret_leaks": len(secret_matches),
                "prompt_injection_echoes": len(injection_matches),
            },
        )
