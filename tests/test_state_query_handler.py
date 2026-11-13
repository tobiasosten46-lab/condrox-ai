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

"""Tests for StateQueryHandler — Phase 2-A critical component."""

from __future__ import annotations

from core.conversational.state_query_handler import StateQueryHandler, StateQueryResult


class TestStateQueryHandler:
    """Unit tests for StateQueryHandler."""

    def test_detects_improve_verb(self) -> None:
        """improve verb is detected with correct action."""
        handler = StateQueryHandler()
        result = handler.handle("Please improve this code", "code")
        assert result.is_state_query is True
        assert result.action == "improve"
        assert result.confidence > 0.0

    def test_detects_fix_verb(self) -> None:
        """fix verb is detected."""
        handler = StateQueryHandler()
        result = handler.handle("Fix the bug in main.py", "code")
        assert result.is_state_query is True
        assert result.action == "fix"

    def test_detects_upgrade_verb(self) -> None:
        """upgrade verb is detected."""
        handler = StateQueryHandler()
        result = handler.handle("Upgrade the database schema", "code")
        assert result.is_state_query is True
        assert result.action == "upgrade"

    def test_detects_confirm_verb(self) -> None:
        """confirm verb is detected."""
        handler = StateQueryHandler()
        result = handler.handle("Verify the test results", "question")
        assert result.is_state_query is True
        assert result.action == "confirm"

    def test_detects_diagnostic_verb(self) -> None:
        """diagnostic verb is detected."""
        handler = StateQueryHandler()
        result = handler.handle("Diagnose what is wrong with the server", "question")
        assert result.is_state_query is True
        assert result.action == "diagnostic"

    def test_non_state_query_returns_false(self) -> None:
        """Normal queries are not state queries."""
        handler = StateQueryHandler()
        result = handler.handle("Explain Python functions.", "explain")
        assert result.is_state_query is False
        assert result.action == ""
        assert result.fallback_intent == "explain"

    def test_empty_input_returns_false(self) -> None:
        """Empty input is not a state query."""
        handler = StateQueryHandler()
        result = handler.handle("", "unknown")
        assert result.is_state_query is False

    def test_target_extraction(self) -> None:
        """Target is extracted from text after the verb."""
        handler = StateQueryHandler()
        result = handler.handle("Improve the authentication module", "code")
        assert result.is_state_query is True
        assert "authentication" in result.target or "module" in result.target

    def test_fallback_intent_preserved(self) -> None:
        """Fallback intent is preserved when not a state query."""
        handler = StateQueryHandler()
        result = handler.handle("What is Python?", "question")
        assert result.is_state_query is False
        assert result.fallback_intent == "question"

    def test_result_is_frozen(self) -> None:
        """StateQueryResult is immutable."""
        handler = StateQueryHandler()
        result = handler.handle("Fix this", "code")
        try:
            result.action = "hacked"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except Exception:
            pass  # expected

    def test_custom_verbs(self) -> None:
        """Custom verb map works."""
        custom = {"deploy": ["deploy", "ship", "release"]}
        handler = StateQueryHandler(state_verbs=custom)
        result = handler.handle("Deploy to production", "code")
        assert result.is_state_query is True
        assert result.action == "deploy"

    def test_case_insensitive(self) -> None:
        """State verbs are detected case-insensitively."""
        handler = StateQueryHandler()
        result = handler.handle("IMPROVE the code", "code")
        assert result.is_state_query is True
        assert result.action == "improve"
