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

"""Tests for SemanticInterruptHandler — Phase 2-F component."""

from __future__ import annotations

from core.conversational.semantic_interrupt import (
    InterruptResult,
    InterruptType,
    SemanticInterruptHandler,
)


class TestSemanticInterruptHandler:
    """Unit tests for SemanticInterruptHandler."""

    def test_no_interrupt_normal_input(self) -> None:
        """Normal input is not an interrupt."""
        handler = SemanticInterruptHandler()
        result = handler.detect("Tell me about Python")
        assert result.is_interrupt is False
        assert result.interrupt_type == InterruptType.NONE

    def test_abort_interrupt(self) -> None:
        """Abort keywords trigger abort."""
        handler = SemanticInterruptHandler()
        result = handler.detect("cancel that")
        assert result.is_interrupt is True
        assert result.interrupt_type == InterruptType.ABORT
        assert result.should_abort is True

    def test_urgency_interrupt(self) -> None:
        """Urgency keywords trigger urgency."""
        handler = SemanticInterruptHandler()
        result = handler.detect("stop! I need help now")
        assert result.is_interrupt is True
        assert result.interrupt_type == InterruptType.URGENCY
        assert result.should_abort is True

    def test_correction_interrupt(self) -> None:
        """Correction keywords trigger correction."""
        handler = SemanticInterruptHandler()
        result = handler.detect("actually, I meant Python not Java")
        assert result.is_interrupt is True
        assert result.interrupt_type == InterruptType.CORRECTION
        assert result.should_abort is False

    def test_question_interrupt(self) -> None:
        """Questions trigger question interrupt."""
        handler = SemanticInterruptHandler()
        result = handler.detect("What is a closure?")
        assert result.is_interrupt is True
        assert result.interrupt_type == InterruptType.QUESTION

    def test_topic_change_interrupt(self) -> None:
        """Topic change is detected."""
        handler = SemanticInterruptHandler()
        result = handler.detect("Recipe for pancakes", current_topic="Python programming language")
        assert result.is_interrupt is True
        assert result.interrupt_type == InterruptType.TOPIC_CHANGE

    def test_no_topic_change_for_same_topic(self) -> None:
        """Same topic doesn't trigger interrupt."""
        handler = SemanticInterruptHandler()
        result = handler.detect("Python function example", current_topic="Python programming language")
        assert result.interrupt_type != InterruptType.TOPIC_CHANGE

    def test_empty_input(self) -> None:
        """Empty input is not an interrupt."""
        handler = SemanticInterruptHandler()
        result = handler.detect("")
        assert result.is_interrupt is False

    def test_handoff_message_present(self) -> None:
        """Interrupts have handoff messages."""
        handler = SemanticInterruptHandler()
        result = handler.detect("cancel that")
        assert result.handoff_message != ""

    def test_never_mind_variations(self) -> None:
        """Various abort phrases work."""
        handler = SemanticInterruptHandler()
        for phrase in ["never mind", "forget it", "nvm", "abort"]:
            result = handler.detect(phrase)
            assert result.interrupt_type == InterruptType.ABORT

    def test_interrupt_result_is_frozen(self) -> None:
        """InterruptResult is immutable."""
        handler = SemanticInterruptHandler()
        result = handler.detect("cancel")
        try:
            result.is_interrupt = False  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_abort_priority_over_question(self) -> None:
        """Abort takes priority over question."""
        handler = SemanticInterruptHandler()
        result = handler.detect("cancel what I asked?")
        assert result.interrupt_type == InterruptType.ABORT
