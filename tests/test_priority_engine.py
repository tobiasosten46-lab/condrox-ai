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

"""Tests for ConversationalPriorityEngine — Phase 2-F component."""

from __future__ import annotations

from core.conversational.priority_engine import (
    ConversationalPriorityEngine,
    PrioritizedItem,
    PriorityScore,
    PriorityTier,
)


class TestConversationalPriorityEngine:
    """Unit tests for ConversationalPriorityEngine."""

    def test_score_normal_text(self) -> None:
        """Normal text gets normal tier."""
        engine = ConversationalPriorityEngine()
        score = engine.score("Tell me about Python")
        assert score.tier == PriorityTier.NORMAL
        assert 40 <= score.score <= 70

    def test_score_safety_critical(self) -> None:
        """Safety keywords get critical tier."""
        engine = ConversationalPriorityEngine()
        score = engine.score("There is a security breach in the system")
        assert score.tier == PriorityTier.CRITICAL
        assert score.score == 100

    def test_score_error_high(self) -> None:
        """Error keywords get high tier."""
        engine = ConversationalPriorityEngine()
        score = engine.score("The application crashed with an error")
        assert score.tier == PriorityTier.HIGH
        assert score.score >= 70

    def test_score_urgency_high(self) -> None:
        """Urgency keywords get high tier."""
        engine = ConversationalPriorityEngine()
        score = engine.score("I need this ASAP please")
        assert score.tier == PriorityTier.HIGH

    def test_score_question(self) -> None:
        """Questions get a boost."""
        engine = ConversationalPriorityEngine()
        score = engine.score("What is the meaning of life?")
        assert "question" in score.factors
        assert score.score >= 65

    def test_score_empty_text(self) -> None:
        """Empty text gets low tier."""
        engine = ConversationalPriorityEngine()
        score = engine.score("")
        assert score.tier == PriorityTier.LOW

    def test_enqueue_and_dequeue(self) -> None:
        """Items are enqueued and dequeued by priority."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("low", "Tell me a joke")
        engine.enqueue("critical", "Security breach detected")
        engine.enqueue("normal", "What is Python?")
        first = engine.dequeue()
        assert first.item_id == "critical"
        second = engine.dequeue()
        assert second.item_id == "normal"

    def test_dequeue_empty(self) -> None:
        """Dequeue from empty queue returns None."""
        engine = ConversationalPriorityEngine()
        assert engine.dequeue() is None

    def test_peek(self) -> None:
        """Peek returns highest without removing."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("low", "joke")
        engine.enqueue("high", "error crash")
        item = engine.peek()
        assert item.item_id == "high"
        assert engine.queue_size() == 2

    def test_remove(self) -> None:
        """Remove deletes a specific item."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("a", "test a")
        engine.enqueue("b", "test b")
        assert engine.remove("a") is True
        assert engine.queue_size() == 1

    def test_remove_missing(self) -> None:
        """Remove missing item returns False."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("a", "test")
        assert engine.remove("missing") is False

    def test_queue_size(self) -> None:
        """queue_size returns correct count."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("a", "test")
        engine.enqueue("b", "test")
        assert engine.queue_size() == 2

    def test_clear(self) -> None:
        """Clear empties the queue."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("a", "test")
        engine.clear()
        assert engine.queue_size() == 0

    def test_reprioritize(self) -> None:
        """Reprioritize updates an item's priority."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("item", "normal text")
        engine.reprioritize("item", "security breach detected")
        item = engine.peek()
        assert item.priority.tier == PriorityTier.CRITICAL

    def test_reprioritize_missing(self) -> None:
        """Reprioritize missing item returns False."""
        engine = ConversationalPriorityEngine()
        assert engine.reprioritize("missing", "text") is False

    def test_get_queue(self) -> None:
        """get_queue returns a copy."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("a", "test")
        queue = engine.get_queue()
        assert len(queue) == 1
        queue.clear()
        assert engine.queue_size() == 1  # Original unchanged

    def test_priority_score_is_frozen(self) -> None:
        """PriorityScore is immutable."""
        engine = ConversationalPriorityEngine()
        score = engine.score("test")
        try:
            score.score = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_prioritized_item_is_frozen(self) -> None:
        """PrioritizedItem is immutable."""
        engine = ConversationalPriorityEngine()
        engine.enqueue("a", "test")
        item = engine.dequeue()
        try:
            item.content = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_correction_factor(self) -> None:
        """Correction keywords boost priority."""
        engine = ConversationalPriorityEngine()
        score = engine.score("no wait, I meant something else")
        assert "correction" in score.factors
        assert score.score >= 75
