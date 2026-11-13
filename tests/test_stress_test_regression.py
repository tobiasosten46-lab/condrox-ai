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

"""
Regression tests for bugs discovered by the conversation stress test.

These tests ensure the fixes for the 4 bugs found during stress testing
remain fixed:

BUG #1: Output was identical for all inputs (9.1% uniqueness)
  - Root cause: Synthesis stage produced "Mode: X | Domain: Y" only
  - Fix: Synthesis now includes the processed input text and mode-specific headers

BUG #2: Empty input caused ValueError exception
  - Root cause: process_request raises ValueError on empty input
  - Fix: Stress test skips empty inputs (API layer handles 400 response)

BUG #3: Mode 'clarify' dominated (42.9% of all turns)
  - Root cause: Mode selector returned 'clarify' when confidence < 0.4
  - Fix: Only return 'clarify' when confidence < 0.25 AND intent is 'unknown'

BUG #4: Domain 'general' dominated (54.5% of all turns)
  - Root cause: Domain router used Jaccard similarity with tiny keyword sets
  - Fix: Expanded keyword sets and switched to intersection count scoring

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

from core import BrainController


@pytest.fixture
def controller() -> BrainController:
    """Provide a booted BrainController."""
    bc = BrainController()
    bc.boot_system()
    return bc


# ─── BUG #1: Output uniqueness ───────────────────────────────────────────────


class TestOutputUniqueness:
    """Regression tests for BUG #1: outputs must differ for different inputs."""

    def test_different_inputs_produce_different_outputs(
        self, controller: BrainController
    ) -> None:
        """Two different inputs must produce different final outputs."""
        s1 = controller.process_request("Write a Python function that sorts a list.")
        s2 = controller.process_request("Explain what a closure is in JavaScript.")
        assert s1.final_output != s2.final_output

    def test_output_contains_input_text(self, controller: BrainController) -> None:
        """Output should reference the user's input text."""
        state = controller.process_request("Explain what a decorator is in Python.")
        # The output should reference at least one key term from the input
        combined = state.final_output + state.synthesized_output
        combined_lower = combined.lower()
        assert any(w in combined_lower for w in ["decorator", "python", "function", "syntax"])

    def test_output_not_just_preamble(self, controller: BrainController) -> None:
        """Output must contain substantive content beyond any preamble."""
        state = controller.process_request("Write a Python function that caches data.")
        # With empty preamble, output should be substantive content directly
        assert len(state.final_output.strip()) > 20
        # Should contain the user's topic or mode-specific framing
        assert any(w in state.final_output.lower() for w in ["cache", "code", "analyzing", "processing"])

    def test_ten_different_inputs_produce_ten_different_outputs(
        self, controller: BrainController
    ) -> None:
        """10 different inputs should produce at least 8 unique outputs."""
        inputs = [
            "Write a Python function that sorts a list.",
            "Explain what a closure is in JavaScript.",
            "Analyze the performance of my API.",
            "Design a system architecture for a chat platform.",
            "Fix this bug: my function returns None.",
            "What is a decorator in Python?",
            "Create a Go class for a queue.",
            "Refactor my code to be more modular.",
            "Teach me about async/await patterns.",
            "Audit my code for security issues.",
        ]
        outputs = []
        for inp in inputs:
            state = controller.process_request(inp)
            outputs.append(state.final_output)
        unique = len(set(outputs))
        assert unique >= 8, f"Only {unique}/10 outputs were unique"


# ─── BUG #2: Empty input handling ────────────────────────────────────────────


class TestEmptyInputHandling:
    """Regression tests for BUG #2: empty input handling."""

    def test_empty_input_raises_value_error(self, controller: BrainController) -> None:
        """Empty input raises ValueError (by design, API catches it)."""
        with pytest.raises(ValueError, match="non-empty"):
            controller.process_request("")

    def test_whitespace_input_raises_value_error(self, controller: BrainController) -> None:
        """Whitespace-only input raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            controller.process_request("   ")

    def test_single_char_input_works(self, controller: BrainController) -> None:
        """Single character input is processed successfully."""
        state = controller.process_request("a")
        assert state.final_output
        assert state.security_approved is True


# ─── BUG #3: Mode distribution ───────────────────────────────────────────────


class TestModeDistribution:
    """Regression tests for BUG #3: mode should not over-clarify."""

    def test_explain_intent_gets_explain_mode(self, controller: BrainController) -> None:
        """'Explain' requests should get 'explain' mode, not 'clarify'."""
        state = controller.process_request("Explain what a closure is in JavaScript.")
        assert state.mode == "explain"

    def test_code_intent_gets_code_mode(self, controller: BrainController) -> None:
        """'Write' requests should get 'code' mode, not 'clarify'."""
        state = controller.process_request("Write a Python function that sorts a list.")
        assert state.mode == "code"

    def test_analyze_intent_gets_analyze_mode(self, controller: BrainController) -> None:
        """'Analyze' requests should get 'analyze' mode, not 'clarify'."""
        state = controller.process_request("Analyze the performance of my system.")
        assert state.mode == "analyze"

    def test_question_intent_gets_explain_mode(self, controller: BrainController) -> None:
        """'What is' requests should get 'explain' mode, not 'clarify'."""
        state = controller.process_request("What is a decorator in Python?")
        assert state.mode == "explain"

    def test_clarify_only_for_unknown_low_confidence(
        self, controller: BrainController
    ) -> None:
        """'clarify' mode should only appear for truly unknown/ambiguous inputs."""
        # These should NOT be clarify
        clear_inputs = [
            "Write a Python function that sorts a list.",
            "Explain what a closure is in JavaScript.",
            "Analyze the performance of my API.",
            "Fix this bug: my function returns None.",
            "Create a Go class for a queue.",
            "Refactor my code to be more modular.",
            "Teach me about async/await patterns.",
            "Audit my code for security issues.",
            "Design a system architecture for a chat platform.",
            "Implement a binary search in Python.",
        ]
        clarify_count = 0
        for inp in clear_inputs:
            state = controller.process_request(inp)
            if state.mode == "clarify":
                clarify_count += 1
        # At most 2 out of 10 clear inputs should be clarify
        assert clarify_count <= 2, f"{clarify_count}/10 clear inputs got 'clarify' mode"


# ─── BUG #4: Domain distribution ─────────────────────────────────────────────


class TestDomainDistribution:
    """Regression tests for BUG #4: domain should not default to 'general'."""

    def test_code_request_gets_programming_domain(
        self, controller: BrainController
    ) -> None:
        """Code requests should route to 'programming' domain."""
        state = controller.process_request("Write a Python function that sorts a list.")
        assert state.domain == "programming"

    def test_architecture_request_gets_architecture_domain(
        self, controller: BrainController
    ) -> None:
        """Architecture requests should route to 'architecture' domain."""
        state = controller.process_request("Design a system architecture for a chat platform.")
        assert state.domain == "architecture"

    def test_security_request_gets_security_domain(
        self, controller: BrainController
    ) -> None:
        """Security requests should route to 'security' domain."""
        state = controller.process_request("Audit my code for security vulnerabilities.")
        assert state.domain == "security"

    def test_data_request_gets_data_domain(
        self, controller: BrainController
    ) -> None:
        """Data-related requests should route to 'data' domain."""
        state = controller.process_request("Analyze the dataset statistics and metrics.")
        assert state.domain == "data"

    def test_general_not_dominant(self, controller: BrainController) -> None:
        """'general' domain should not dominate for specific requests."""
        specific_inputs = [
            "Write a Python function that sorts a list.",
            "Explain what a closure is in JavaScript.",
            "Create a Go class for a queue.",
            "Fix this bug: my function returns None.",
            "Implement a binary search in Python.",
            "Refactor my code to be more modular.",
            "Build a Rust module that encrypts data.",
            "Code a Swift script to parse files.",
            "Design a microservice architecture.",
            "Audit my authentication system for security.",
        ]
        general_count = 0
        for inp in specific_inputs:
            state = controller.process_request(inp)
            if state.domain == "general":
                general_count += 1
        # At most 3 out of 10 specific inputs should be 'general'
        assert general_count <= 3, f"{general_count}/10 specific inputs got 'general' domain"


# ─── Multi-turn conversation regression ──────────────────────────────────────


class TestMultiTurnConversationRegression:
    """Regression tests for multi-turn conversation quality."""

    def test_five_turn_conversation_has_varied_modes(
        self, controller: BrainController
    ) -> None:
        """A 5-turn conversation with varied inputs should have >1 mode."""
        inputs = [
            "Write a Python function that sorts a list.",
            "Explain what a closure is in JavaScript.",
            "Analyze the performance of my API.",
            "Fix this bug: my function returns None.",
            "Design a system architecture for a chat platform.",
        ]
        modes = []
        for inp in inputs:
            state = controller.process_request(inp)
            modes.append(state.mode)
        assert len(set(modes)) >= 3, f"Only {len(set(modes))} unique modes in 5 turns: {modes}"

    def test_five_turn_conversation_has_varied_domains(
        self, controller: BrainController
    ) -> None:
        """A 5-turn conversation with varied inputs should have >1 domain."""
        inputs = [
            "Write a Python function that sorts a list.",
            "Design a system architecture for a chat platform.",
            "Audit my code for security vulnerabilities.",
            "Analyze the dataset statistics and metrics.",
            "Explain what a closure is in JavaScript.",
        ]
        domains = []
        for inp in inputs:
            state = controller.process_request(inp)
            domains.append(state.domain)
        assert len(set(domains)) >= 3, f"Only {len(set(domains))} unique domains in 5 turns: {domains}"

    def test_conversation_outputs_are_unique(
        self, controller: BrainController
    ) -> None:
        """A 5-turn conversation should produce mostly unique outputs."""
        inputs = [
            "Write a Python function that sorts a list.",
            "Explain what a closure is in JavaScript.",
            "Analyze the performance of my API.",
            "Fix this bug: my function returns None.",
            "Design a system architecture for a chat platform.",
        ]
        outputs = []
        for inp in inputs:
            state = controller.process_request(inp)
            outputs.append(state.final_output)
        unique = len(set(outputs))
        assert unique >= 4, f"Only {unique}/5 outputs were unique"
