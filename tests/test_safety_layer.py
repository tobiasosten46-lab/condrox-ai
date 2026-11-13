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

"""Tests for SafetyLayer — Phase 2-C component."""

from __future__ import annotations

import re

from core.conversational.safety_layer import (
    SafetyLayer,
    SafetyLevel,
    SafetyResult,
    SafetyRule,
)


class TestSafetyLayer:
    """Unit tests for SafetyLayer."""

    def test_safe_text_passes(self) -> None:
        """Safe text passes check."""
        layer = SafetyLayer()
        result = layer.check("The function returns a value.")
        assert result.passed is True
        assert len(result.violations) == 0

    def test_credit_card_detected(self) -> None:
        """Credit card numbers are detected as critical."""
        layer = SafetyLayer()
        result = layer.check("My card is 4532-1234-5678-9012")
        assert result.passed is False
        assert any(v.level == SafetyLevel.CRITICAL for v in result.violations)

    def test_ssn_detected(self) -> None:
        """SSN is detected as critical."""
        layer = SafetyLayer()
        result = layer.check("SSN: 123-45-6789")
        assert result.passed is False
        assert any(v.level == SafetyLevel.CRITICAL for v in result.violations)

    def test_email_detected_as_warning(self) -> None:
        """Email is detected as warning."""
        layer = SafetyLayer()
        result = layer.check("Contact me at user@example.com")
        assert any(v.level == SafetyLevel.WARNING for v in result.violations)

    def test_violence_blocked(self) -> None:
        """Violence content is blocked."""
        layer = SafetyLayer()
        result = layer.check("How to attack someone")
        assert result.passed is False
        assert any(v.level == SafetyLevel.BLOCKED for v in result.violations)

    def test_self_harm_critical(self) -> None:
        """Self-harm content is critical."""
        layer = SafetyLayer()
        result = layer.check("I want to commit suicide")
        assert result.passed is False
        assert any(v.level == SafetyLevel.CRITICAL for v in result.violations)

    def test_enforce_redacts_content(self) -> None:
        """enforce() redacts violating content."""
        layer = SafetyLayer()
        text = "My card is 4532-1234-5678-9012"
        cleaned, result = layer.enforce(text)
        assert "[REDACTED]" in cleaned
        assert result.passed is False

    def test_override_allows_warning(self) -> None:
        """Override allows warning-level content."""
        layer = SafetyLayer()
        layer.set_override(True)
        result = layer.check("Contact user@example.com for help")
        # Email is warning level with override_allowed=True
        assert result.passed is True

    def test_override_does_not_allow_critical(self) -> None:
        """Override does not allow critical content."""
        layer = SafetyLayer()
        layer.set_override(True)
        result = layer.check("SSN: 123-45-6789")
        assert result.passed is False

    def test_audit_log(self) -> None:
        """Audit log records violations."""
        layer = SafetyLayer()
        layer.check("SSN: 123-45-6789")
        log = layer.get_audit_log()
        assert len(log) > 0

    def test_clear_audit_log(self) -> None:
        """clear_audit_log empties the log."""
        layer = SafetyLayer()
        layer.check("SSN: 123-45-6789")
        layer.clear_audit_log()
        assert len(layer.get_audit_log()) == 0

    def test_add_custom_rule(self) -> None:
        """Custom rules can be added."""
        layer = SafetyLayer()
        rule = SafetyRule(
            rule_id="custom_test",
            pattern=re.compile(r"\bforbidden_word\b", re.IGNORECASE),
            level=SafetyLevel.BLOCKED,
            message="Custom forbidden word",
            override_allowed=False,
        )
        layer.add_rule(rule)
        result = layer.check("This contains forbidden_word here")
        assert result.passed is False
        assert any(v.rule_id == "custom_test" for v in result.violations)

    def test_remove_rule(self) -> None:
        """Rules can be removed."""
        layer = SafetyLayer()
        assert layer.remove_rule("pii_email") is True
        result = layer.check("Contact user@example.com")
        assert not any(v.rule_id == "pii_email" for v in result.violations)

    def test_empty_text_passes(self) -> None:
        """Empty text passes."""
        layer = SafetyLayer()
        result = layer.check("")
        assert result.passed is True

    def test_safety_result_is_frozen(self) -> None:
        """SafetyResult is immutable."""
        layer = SafetyLayer()
        result = layer.check("safe text")
        try:
            result.passed = False  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected
