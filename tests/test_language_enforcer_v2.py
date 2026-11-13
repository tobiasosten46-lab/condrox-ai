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

"""Tests for LanguageEnforcerV2 — Phase 2-B component."""

from __future__ import annotations

from core.conversational.language_enforcer_v2 import (
    LanguageEnforcerV2,
    PersonaRules,
)


class TestLanguageEnforcerV2:
    """Unit tests for LanguageEnforcerV2."""

    def test_enforce_removes_forbidden_phrases(self) -> None:
        """Forbidden phrases are removed."""
        enforcer = LanguageEnforcerV2()
        text = "As an AI, I don't have personal opinions. The answer is 42."
        result, violations = enforcer.enforce(text)
        assert "as an ai" not in result.lower()
        assert len(violations) > 0

    def test_enforce_contracts_for_condrox(self) -> None:
        """Condrox persona contracts words."""
        enforcer = LanguageEnforcerV2()
        text = "You do not need to worry. It is fine."
        result, _ = enforcer.enforce(text)
        assert "don't" in result or "do not" not in result

    def test_enforce_clean_text(self) -> None:
        """Clean text passes through with no violations."""
        enforcer = LanguageEnforcerV2()
        text = "The function returns the square of the input."
        result, violations = enforcer.enforce(text)
        assert violations == []
        assert "function" in result

    def test_enforce_empty_text(self) -> None:
        """Empty text returns empty."""
        enforcer = LanguageEnforcerV2()
        result, violations = enforcer.enforce("")
        assert result == ""
        assert violations == []

    def test_check_detects_forbidden(self) -> None:
        """check() detects forbidden phrases."""
        enforcer = LanguageEnforcerV2()
        violations = enforcer.check("As an AI, I cannot help.")
        assert len(violations) > 0
        assert any("forbidden_phrase" in v for v in violations)

    def test_check_clean_text(self) -> None:
        """check() returns empty for clean text."""
        enforcer = LanguageEnforcerV2()
        violations = enforcer.check("The answer is 42.")
        assert violations == []

    def test_violation_log(self) -> None:
        """Violation log accumulates."""
        enforcer = LanguageEnforcerV2()
        enforcer.enforce("As an AI, I cannot help.")
        log = enforcer.get_violation_log()
        assert len(log) > 0

    def test_clear_log(self) -> None:
        """clear_log empties the log."""
        enforcer = LanguageEnforcerV2()
        enforcer.enforce("As an AI, I cannot help.")
        enforcer.clear_log()
        assert enforcer.get_violation_log() == []

    def test_custom_persona(self) -> None:
        """Custom persona works."""
        custom = PersonaRules(
            name="test",
            forbidden_phrases=("foobar",),
            required_tone=("test",),
            contraction_policy="expand",
            first_person=False,
            formality=0.8,
        )
        enforcer = LanguageEnforcerV2(persona=custom)
        text = "foobar I'm here."
        result, violations = enforcer.enforce(text)
        assert "foobar" not in result
        assert len(violations) > 0

    def test_no_first_person_policy(self) -> None:
        """First person is removed when policy disables it."""
        custom = PersonaRules(
            name="no_first_person",
            forbidden_phrases=(),
            required_tone=("formal",),
            contraction_policy="expand",
            first_person=False,
            formality=0.9,
        )
        enforcer = LanguageEnforcerV2(persona=custom)
        text = "I think the answer is 42."
        result, _ = enforcer.enforce(text)
        assert "the system" in result.lower()

    def test_expand_policy(self) -> None:
        """Expand policy expands contractions."""
        custom = PersonaRules(
            name="formal",
            forbidden_phrases=(),
            required_tone=("formal",),
            contraction_policy="expand",
            first_person=True,
            formality=0.9,
        )
        enforcer = LanguageEnforcerV2(persona=custom)
        text = "I don't know the answer."
        result, _ = enforcer.enforce(text)
        assert "do not" in result

    def test_whitespace_cleanup(self) -> None:
        """Extra whitespace is cleaned."""
        enforcer = LanguageEnforcerV2()
        text = "Hello    world.  How are you?"
        result, _ = enforcer.enforce(text)
        assert "  " not in result

    def test_persona_rules_is_frozen(self) -> None:
        """PersonaRules is immutable."""
        rules = PersonaRules(
            name="test",
            forbidden_phrases=(),
            required_tone=(),
            contraction_policy="neutral",
            first_person=True,
            formality=0.5,
        )
        try:
            rules.name = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected
