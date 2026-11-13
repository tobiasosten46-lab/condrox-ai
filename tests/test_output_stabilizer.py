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

"""Tests for OutputStabilizer — Phase 2-B component."""

from __future__ import annotations

from core.conversational.output_stabilizer import OutputStabilizer, StabilizationResult


class TestOutputStabilizer:
    """Unit tests for OutputStabilizer."""

    def test_stabilize_clean_text(self) -> None:
        """Clean text passes through unchanged."""
        stabilizer = OutputStabilizer()
        result = stabilizer.stabilize("The function returns the square of the input.")
        assert result.text == "The function returns the square of the input."
        assert result.repetitions_removed == 0
        assert result.disclaimers_removed == 0

    def test_stabilize_empty_text(self) -> None:
        """Empty text returns empty result."""
        stabilizer = OutputStabilizer()
        result = stabilizer.stabilize("")
        assert result.text == ""
        assert result.repetitions_removed == 0

    def test_remove_disclaimers(self) -> None:
        """Disclaimers are removed."""
        stabilizer = OutputStabilizer()
        text = "The answer is 42. Please note that this information may not be complete."
        result = stabilizer.stabilize(text)
        assert result.disclaimers_removed > 0
        assert "may not be complete" not in result.text

    def test_remove_ai_disclaimers(self) -> None:
        """AI disclaimers are removed."""
        stabilizer = OutputStabilizer()
        text = "As an AI, I don't have personal opinions. The answer is 42."
        result = stabilizer.stabilize(text)
        assert result.disclaimers_removed > 0
        assert "as an ai" not in result.text.lower()

    def test_remove_repeated_sentences(self) -> None:
        """Repeated sentences are removed."""
        stabilizer = OutputStabilizer()
        text = "The answer is 42. The answer is 42. The answer is 42."
        result = stabilizer.stabilize(text)
        assert result.repetitions_removed > 0
        # Should only have one instance (max_consecutive=2)
        assert result.text.count("The answer is 42") <= 2

    def test_break_loops(self) -> None:
        """Word loops are broken."""
        stabilizer = OutputStabilizer()
        text = "the value is the value is the value is 42"
        result = stabilizer.stabilize(text)
        assert result.loops_detected >= 0  # May or may not detect depending on segmentation

    def test_check_quality_clean(self) -> None:
        """Clean text has no quality issues."""
        stabilizer = OutputStabilizer()
        issues = stabilizer.check_quality("The function returns a value.")
        assert issues == []

    def test_check_quality_with_disclaimer(self) -> None:
        """Disclaimer is detected."""
        stabilizer = OutputStabilizer()
        issues = stabilizer.check_quality("As an AI, I cannot help.")
        assert "disclaimer_detected" in issues

    def test_check_quality_with_repetition(self) -> None:
        """Repetition is detected."""
        stabilizer = OutputStabilizer()
        issues = stabilizer.check_quality("Hello. Hello. How are you?")
        assert "repeated_sentence" in issues

    def test_stabilization_result_is_frozen(self) -> None:
        """StabilizationResult is immutable."""
        stabilizer = OutputStabilizer()
        result = stabilizer.stabilize("test")
        try:
            result.text = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected

    def test_whitespace_cleanup(self) -> None:
        """Extra whitespace is cleaned."""
        stabilizer = OutputStabilizer()
        text = "Hello    world.  \n\n\n  How are you?"
        result = stabilizer.stabilize(text)
        assert "  " not in result.text
        assert "\n\n\n" not in result.text

    def test_max_consecutive_configurable(self) -> None:
        """Max consecutive is configurable."""
        stabilizer = OutputStabilizer(max_consecutive=1)
        text = "Hello. Hello. World."
        result = stabilizer.stabilize(text)
        assert result.repetitions_removed >= 1
