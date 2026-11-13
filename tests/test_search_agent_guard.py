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

"""Tests for SearchAgentGuard — Phase 2-A critical component."""

from __future__ import annotations

from core.conversational.search_agent_guard import GuardResult, SearchAgentGuard


class TestSearchAgentGuard:
    """Unit tests for SearchAgentGuard."""

    def test_none_input_rejected(self) -> None:
        """None input is invalid."""
        guard = SearchAgentGuard()
        valid, reason = guard.validate(None)
        assert valid is False
        assert "None" in reason

    def test_empty_string_rejected(self) -> None:
        """Empty string is invalid."""
        guard = SearchAgentGuard()
        valid, reason = guard.validate("")
        assert valid is False
        assert "empty" in reason

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only string is invalid."""
        guard = SearchAgentGuard()
        valid, reason = guard.validate("   \n\t  ")
        assert valid is False
        assert "empty" in reason or "whitespace" in reason

    def test_valid_query_accepted(self) -> None:
        """Valid query passes."""
        guard = SearchAgentGuard()
        valid, reason = guard.validate("Python functions")
        assert valid is True
        assert reason == "ok"

    def test_guard_returns_sentinel_for_none(self) -> None:
        """guard() returns sentinel for None."""
        guard = SearchAgentGuard()
        result = guard.guard(None)
        assert result == "<no-query>"

    def test_guard_returns_sentinel_for_empty(self) -> None:
        """guard() returns sentinel for empty string."""
        guard = SearchAgentGuard()
        result = guard.guard("")
        assert result == "<empty-query>"

    def test_guard_returns_stripped_query_for_valid(self) -> None:
        """guard() returns stripped query for valid input."""
        guard = SearchAgentGuard()
        result = guard.guard("  Python functions  ")
        assert result == "Python functions"

    def test_guard_never_returns_none(self) -> None:
        """guard() never returns None."""
        guard = SearchAgentGuard()
        assert guard.guard(None) is not None
        assert guard.guard("") is not None
        assert guard.guard("valid") is not None

    def test_check_returns_structured_result(self) -> None:
        """check() returns a GuardResult."""
        guard = SearchAgentGuard()
        result = guard.check("valid query")
        assert isinstance(result, GuardResult)
        assert result.valid is True
        assert result.query == "valid query"
        assert result.reason == "ok"

    def test_check_for_none(self) -> None:
        """check() for None returns invalid result."""
        guard = SearchAgentGuard()
        result = guard.check(None)
        assert result.valid is False
        assert result.query == "<no-query>"

    def test_max_length_exceeded(self) -> None:
        """Query over max length is rejected."""
        guard = SearchAgentGuard(max_length=10)
        valid, reason = guard.validate("a" * 20)
        assert valid is False
        assert "max length" in reason

    def test_non_string_rejected(self) -> None:
        """Non-string input is rejected."""
        guard = SearchAgentGuard()
        valid, reason = guard.validate(123)  # type: ignore[arg-type]
        assert valid is False
        assert "str" in reason

    def test_guard_result_is_frozen(self) -> None:
        """GuardResult is immutable."""
        guard = SearchAgentGuard()
        result = guard.check("test")
        try:
            result.valid = False  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected
