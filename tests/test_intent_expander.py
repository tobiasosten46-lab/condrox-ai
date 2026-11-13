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

"""Tests for SemanticIntentExpander — Phase 2-E component."""

from __future__ import annotations

from core.conversational.intent_expander import (
    ExpandedIntent,
    SemanticIntentExpander,
)


class TestSemanticIntentExpander:
    """Unit tests for SemanticIntentExpander."""

    def test_detect_code_intent(self) -> None:
        """Code-related text detects code intent."""
        expander = SemanticIntentExpander()
        intent, conf = expander.detect("Write a function to sort a list")
        assert intent == "code"
        assert conf > 0

    def test_detect_explain_intent(self) -> None:
        """Explain-related text detects explain intent."""
        expander = SemanticIntentExpander()
        intent, _ = expander.detect("Explain how Python decorators work")
        assert intent == "explain"

    def test_detect_question_intent(self) -> None:
        """Question text detects question intent."""
        expander = SemanticIntentExpander()
        intent, _ = expander.detect("What is a closure?")
        assert intent == "question"

    def test_detect_fix_intent(self) -> None:
        """Fix-related text detects fix intent."""
        expander = SemanticIntentExpander()
        intent, _ = expander.detect("Fix the bug in my code")
        assert intent == "fix"

    def test_detect_unknown_intent(self) -> None:
        """Unknown text returns unknown."""
        expander = SemanticIntentExpander()
        intent, _ = expander.detect("xyzzy foobar")
        assert intent == "unknown"

    def test_detect_empty_text(self) -> None:
        """Empty text returns unknown."""
        expander = SemanticIntentExpander()
        intent, conf = expander.detect("")
        assert intent == "unknown"
        assert conf == 0.0

    def test_expand_code_intent(self) -> None:
        """Code intent expands into sub-intents."""
        expander = SemanticIntentExpander()
        result = expander.expand("Write a function to sort a list")
        assert result.primary == "code"
        assert len(result.sub_intents) > 0
        assert result.expansion_depth > 0

    def test_expand_with_explicit_intent(self) -> None:
        """Explicit intent is used when provided."""
        expander = SemanticIntentExpander()
        result = expander.expand("some text", intent="explain")
        assert result.primary == "explain"

    def test_expand_unknown_intent(self) -> None:
        """Unknown intent returns empty sub-intents."""
        expander = SemanticIntentExpander()
        result = expander.expand("xyzzy foobar")
        assert result.primary == "unknown"
        assert result.sub_intents == ()

    def test_expand_chain(self) -> None:
        """expand_chain returns sub-intent list."""
        expander = SemanticIntentExpander()
        chain = expander.expand_chain("code")
        assert len(chain) > 0
        assert "analyze" in chain

    def test_register_custom_expansion(self) -> None:
        """Custom expansions can be registered."""
        expander = SemanticIntentExpander()
        expander.register_expansion("custom_intent", ["step1", "step2"])
        chain = expander.expand_chain("custom_intent")
        assert "step1" in chain
        assert "step2" in chain

    def test_max_depth_limits_expansion(self) -> None:
        """Max depth limits expansion."""
        expander = SemanticIntentExpander(max_depth=1)
        result = expander.expand("Write a function", intent="code")
        assert result.expansion_depth <= 1

    def test_expanded_intent_is_frozen(self) -> None:
        """ExpandedIntent is immutable."""
        expander = SemanticIntentExpander()
        result = expander.expand("test", intent="code")
        try:
            result.primary = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass
