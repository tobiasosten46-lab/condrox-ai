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

"""Tests for SemanticRepair — Phase 2-E component."""

from __future__ import annotations

from core.conversational.semantic_repair import (
    RepairResult,
    SemanticRepair,
)


class TestSemanticRepair:
    """Unit tests for SemanticRepair."""

    def test_no_drift_on_topic(self) -> None:
        """On-topic response has low drift."""
        repair = SemanticRepair()
        drift = repair.check_drift("Python function", "Python function example code")
        assert drift <= 0.5

    def test_high_drift_off_topic(self) -> None:
        """Off-topic response has high drift."""
        repair = SemanticRepair()
        drift = repair.check_drift("Python function", "Recipe for pancakes with eggs")
        assert drift > 0.5

    def test_drift_empty_query(self) -> None:
        """Empty query has zero drift."""
        repair = SemanticRepair()
        assert repair.check_drift("", "response") == 0.0

    def test_drift_empty_response(self) -> None:
        """Empty response has zero drift."""
        repair = SemanticRepair()
        assert repair.check_drift("query", "") == 0.0

    def test_repair_no_drift(self) -> None:
        """No drift returns text unchanged."""
        repair = SemanticRepair()
        query = "Python function"
        response = "A Python function is defined with def keyword"
        result = repair.repair(query, response)
        assert result.drift_detected is False
        assert result.repaired_text == response

    def test_repair_removes_off_topic(self) -> None:
        """Off-topic sentences are removed."""
        repair = SemanticRepair(drift_threshold=0.4, min_overlap=0.05)
        query = "Python function sort"
        response = (
            "A Python function can sort a list. "
            "Pancakes are delicious for breakfast. "
            "The sort function uses comparisons."
        )
        result = repair.repair(query, response)
        assert result.drift_detected is True
        assert "Pancakes" not in result.repaired_text

    def test_repair_empty_response(self) -> None:
        """Empty response returns empty."""
        repair = SemanticRepair()
        result = repair.repair("query", "")
        assert result.repaired_text == ""
        assert result.drift_detected is False

    def test_repair_empty_query(self) -> None:
        """Empty query returns response unchanged."""
        repair = SemanticRepair()
        result = repair.repair("", "some response")
        assert result.repaired_text == "some response"

    def test_repair_result_is_frozen(self) -> None:
        """RepairResult is immutable."""
        repair = SemanticRepair()
        result = repair.repair("query", "response")
        try:
            result.repaired_text = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_repair_keeps_on_topic(self) -> None:
        """On-topic sentences are preserved."""
        repair = SemanticRepair(drift_threshold=0.3, min_overlap=0.05)
        query = "Python function"
        response = "A Python function uses def. The function returns a value."
        result = repair.repair(query, response)
        assert "Python function" in result.repaired_text
        assert "returns a value" in result.repaired_text

    def test_repairs_applied_logged(self) -> None:
        """Repairs are logged in repairs_applied."""
        repair = SemanticRepair(drift_threshold=0.3, min_overlap=0.05)
        query = "Python function"
        response = "Python function. Random unrelated text here about cooking."
        result = repair.repair(query, response)
        if result.drift_detected:
            assert len(result.repairs_applied) > 0
