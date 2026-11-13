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

"""Tests for ConsistencyEngine — Phase 2-E component."""

from __future__ import annotations

from core.conversational.consistency_engine import (
    ConsistencyEngine,
    ConsistencyReport,
)


class TestConsistencyEngine:
    """Unit tests for ConsistencyEngine."""

    def test_consistent_response(self) -> None:
        """Consistent response passes."""
        engine = ConsistencyEngine()
        engine.add_statement("Python is a programming language")
        report = engine.check("Python is a great programming language")
        assert report.consistent is True
        assert len(report.contradictions) == 0

    def test_negation_contradiction(self) -> None:
        """Negation contradiction is detected."""
        engine = ConsistencyEngine()
        engine.add_statement("The feature is supported")
        report = engine.check("The feature is not supported")
        assert report.consistent is False
        assert len(report.contradictions) > 0

    def test_no_contradiction_without_statements(self) -> None:
        """No statements means no contradictions."""
        engine = ConsistencyEngine()
        report = engine.check("Any response")
        assert report.consistent is True

    def test_empty_response(self) -> None:
        """Empty response is consistent."""
        engine = ConsistencyEngine()
        engine.add_statement("test")
        report = engine.check("")
        assert report.consistent is True

    def test_quantifier_contradiction(self) -> None:
        """Quantifier flip is detected."""
        engine = ConsistencyEngine()
        engine.add_statement("The system always works correctly")
        report = engine.check("The system never works correctly")
        assert report.consistent is False

    def test_confidence_decreases_with_contradictions(self) -> None:
        """More contradictions lower confidence."""
        engine = ConsistencyEngine()
        engine.add_statement("The value is true")
        engine.add_statement("The result is correct")
        report = engine.check("The value is false and the result is incorrect")
        assert report.confidence < 1.0

    def test_contradiction_log(self) -> None:
        """Contradictions are logged."""
        engine = ConsistencyEngine()
        engine.add_statement("The feature is supported")
        engine.check("The feature is not supported")
        log = engine.get_contradiction_log()
        assert len(log) > 0

    def test_clear_contradiction_log(self) -> None:
        """clear_contradiction_log empties the log."""
        engine = ConsistencyEngine()
        engine.add_statement("The feature is supported")
        engine.check("The feature is not supported")
        engine.clear_contradiction_log()
        assert len(engine.get_contradiction_log()) == 0

    def test_clear_statements(self) -> None:
        """clear_statements removes all statements."""
        engine = ConsistencyEngine()
        engine.add_statement("test")
        engine.clear_statements()
        assert engine.statement_count() == 0

    def test_statement_count(self) -> None:
        """statement_count returns correct count."""
        engine = ConsistencyEngine()
        engine.add_statement("s1")
        engine.add_statement("s2")
        assert engine.statement_count() == 2

    def test_consistency_report_is_frozen(self) -> None:
        """ConsistencyReport is immutable."""
        engine = ConsistencyEngine()
        report = engine.check("test")
        try:
            report.consistent = False  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_no_false_positive_on_different_subjects(self) -> None:
        """Different subjects don't trigger contradictions."""
        engine = ConsistencyEngine()
        engine.add_statement("The database is valid")
        report = engine.check("The certificate is invalid")
        # Different subjects (database vs certificate) should not contradict
        assert report.consistent is True
