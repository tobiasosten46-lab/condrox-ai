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

"""Tests for MultiTurnReasoningEngine — Phase 2-E component."""

from __future__ import annotations

from core.conversational.multi_turn_reasoning import (
    MultiTurnReasoningEngine,
    ReasoningContext,
    TurnRecord,
)


class TestMultiTurnReasoningEngine:
    """Unit tests for MultiTurnReasoningEngine."""

    def test_record_and_get_turns(self) -> None:
        """Turns are recorded and retrievable."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Hello", "Hi there", "conversation")
        engine.record_turn("Write code", "def foo(): pass", "code")
        turns = engine.get_turns()
        assert len(turns) == 2
        assert turns[0].user_input == "Hello"
        assert turns[1].intent == "code"

    def test_get_context_empty(self) -> None:
        """Empty history returns empty context."""
        engine = MultiTurnReasoningEngine()
        ctx = engine.get_context()
        assert ctx.turn_count == 0
        assert ctx.dominant_intent == "unknown"

    def test_get_context_with_turns(self) -> None:
        """Context is derived from turns."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Write a function", "def foo(): pass", "code")
        engine.record_turn("Write another function", "def bar(): pass", "code")
        ctx = engine.get_context()
        assert ctx.turn_count == 2
        assert ctx.dominant_intent == "code"
        assert ctx.suggested_focus != ""

    def test_topic_continuity(self) -> None:
        """Topic continuity is computed."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Python function code", "def foo(): pass", "code")
        engine.record_turn("Python function code", "def bar(): pass", "code")
        ctx = engine.get_context()
        assert ctx.topic_continuity > 0

    def test_topic_continuity_low_for_different_topics(self) -> None:
        """Different topics have low continuity."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Python programming", "def foo(): pass", "code")
        engine.record_turn("Recipe for pancakes", "Mix flour and eggs", "conversation")
        ctx = engine.get_context()
        assert ctx.topic_continuity < 0.5

    def test_history_summary(self) -> None:
        """History summary contains turn info."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Hello", "Hi", "conversation")
        ctx = engine.get_context()
        assert "Turn 1" in ctx.history_summary

    def test_get_turn_by_id(self) -> None:
        """Turn can be retrieved by ID."""
        engine = MultiTurnReasoningEngine()
        turn = engine.record_turn("Hello", "Hi", "conversation")
        retrieved = engine.get_turn(turn.turn_id)
        assert retrieved is not None
        assert retrieved.turn_id == turn.turn_id

    def test_get_missing_turn(self) -> None:
        """Missing turn returns None."""
        engine = MultiTurnReasoningEngine()
        assert engine.get_turn(999) is None

    def test_max_history_limit(self) -> None:
        """History is limited to max_history."""
        engine = MultiTurnReasoningEngine(max_history=3)
        for i in range(5):
            engine.record_turn(f"input_{i}", f"output_{i}", "code")
        assert engine.turn_count() == 3

    def test_clear(self) -> None:
        """Clear empties all history."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Hello", "Hi", "conversation")
        engine.clear()
        assert engine.turn_count() == 0

    def test_turn_count(self) -> None:
        """turn_count returns correct count."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("a", "b", "code")
        engine.record_turn("c", "d", "code")
        assert engine.turn_count() == 2

    def test_turn_record_is_frozen(self) -> None:
        """TurnRecord is immutable."""
        engine = MultiTurnReasoningEngine()
        turn = engine.record_turn("Hello", "Hi", "conversation")
        try:
            turn.user_input = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_reasoning_context_is_frozen(self) -> None:
        """ReasoningContext is immutable."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Hello", "Hi", "conversation")
        ctx = engine.get_context()
        try:
            ctx.turn_count = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_suggested_focus_for_code(self) -> None:
        """Code intent suggests code focus."""
        engine = MultiTurnReasoningEngine()
        engine.record_turn("Write code", "def foo(): pass", "code")
        ctx = engine.get_context()
        assert "code" in ctx.suggested_focus.lower()
