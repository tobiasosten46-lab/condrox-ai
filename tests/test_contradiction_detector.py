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

"""Tests for ContradictionDetector — Phase 2-E component."""

from __future__ import annotations

from core.conversational.contradiction_detector import (
    ContradictionDetector,
    ContradictionFinding,
    ContradictionReport,
)


class TestContradictionDetector:
    """Unit tests for ContradictionDetector."""

    def test_no_contradiction_without_facts(self) -> None:
        """No facts means no contradictions."""
        detector = ContradictionDetector()
        report = detector.check("Any statement")
        assert report.has_contradictions is False

    def test_no_contradiction_consistent(self) -> None:
        """Consistent statement has no contradictions."""
        detector = ContradictionDetector()
        detector.add_fact("Python is a programming language")
        report = detector.check("Python is a great programming language")
        assert report.has_contradictions is False

    def test_negation_contradiction(self) -> None:
        """Negation contradiction is detected."""
        detector = ContradictionDetector()
        detector.add_fact("The feature is supported")
        report = detector.check("The feature is not supported")
        assert report.has_contradictions is True
        assert any(f.contradiction_type == "negation" for f in report.findings)

    def test_quantifier_flip(self) -> None:
        """Quantifier flip is detected."""
        detector = ContradictionDetector()
        detector.add_fact("The system always works correctly")
        report = detector.check("The system never works correctly")
        assert report.has_contradictions is True
        assert any(f.contradiction_type == "quantifier_flip" for f in report.findings)

    def test_numeric_conflict(self) -> None:
        """Numeric conflict is detected."""
        detector = ContradictionDetector()
        detector.add_fact("The value is 42")
        report = detector.check("The value is 10")
        assert report.has_contradictions is True
        assert any(f.contradiction_type == "numeric_conflict" for f in report.findings)

    def test_empty_statement(self) -> None:
        """Empty statement has no contradictions."""
        detector = ContradictionDetector()
        detector.add_fact("test")
        report = detector.check("")
        assert report.has_contradictions is False

    def test_fact_count(self) -> None:
        """fact_count returns correct count."""
        detector = ContradictionDetector()
        detector.add_fact("fact 1")
        detector.add_fact("fact 2")
        assert detector.fact_count() == 2

    def test_clear_facts(self) -> None:
        """clear_facts removes all facts."""
        detector = ContradictionDetector()
        detector.add_fact("test")
        detector.clear_facts()
        assert detector.fact_count() == 0

    def test_total_checked(self) -> None:
        """total_checked reflects number of facts."""
        detector = ContradictionDetector()
        detector.add_fact("fact 1")
        detector.add_fact("fact 2")
        report = detector.check("some statement")
        assert report.total_checked == 2

    def test_no_false_positive_different_subjects(self) -> None:
        """Different subjects don't trigger contradictions."""
        detector = ContradictionDetector()
        detector.add_fact("The database is valid")
        report = detector.check("The certificate is invalid")
        assert report.has_contradictions is False

    def test_contradiction_finding_is_frozen(self) -> None:
        """ContradictionFinding is immutable."""
        detector = ContradictionDetector()
        detector.add_fact("The feature is supported")
        report = detector.check("The feature is not supported")
        if report.findings:
            finding = report.findings[0]
            try:
                finding.confidence = 0.0  # type: ignore[misc]
                assert False, "Should have raised"
            except Exception:
                pass

    def test_contradiction_report_is_frozen(self) -> None:
        """ContradictionReport is immutable."""
        detector = ContradictionDetector()
        report = detector.check("test")
        try:
            report.has_contradictions = True  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_multiple_contradictions(self) -> None:
        """Multiple contradictions are all found."""
        detector = ContradictionDetector()
        detector.add_fact("The feature is supported")
        detector.add_fact("The result is correct")
        report = detector.check("The feature is not supported and the result is incorrect")
        assert len(report.findings) >= 2
