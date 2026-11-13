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
Agent integration tests for the pipeline (W-19).

Tests cover:
- process_request with "Write a Python function" triggers CodeAgent and mode="code"
- process_request with "Explain Python" triggers ExplainAgent and mode="explain"
- process_request with "Analyze this code" triggers AnalyzeAgent and mode="analyze"
- Agent metadata appears in state.context (agent_id, agent_confidence)
- SearchAgentGuard doesn't crash the pipeline on valid input
- The pipeline has 18 stages (including the new 02b, 05b, 08b, 09b)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

from core.brain.brain_controller import BrainController
from core.brain.brain_state import BrainState


# â”€â”€â”€ Fixtures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@pytest.fixture
def controller() -> BrainController:
    """Provide a fresh, booted BrainController for each test."""
    bc = BrainController()
    bc.boot_system()
    return bc


# â”€â”€â”€ Code Agent Integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCodeAgentIntegration:
    """Tests that code-related requests trigger the CodeAgent."""

    def test_code_agent_integration(self, controller: BrainController) -> None:
        """A 'Write a Python function' request triggers CodeAgent and mode='code'."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert state.mode == "code"
        assert "agent_id" in state.context
        assert state.context["agent_id"] == "code_agent"

    def test_code_agent_intent(self, controller: BrainController) -> None:
        """A code request classifies intent as 'code'."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert state.intent == "code"

    def test_code_agent_returns_state(self, controller: BrainController) -> None:
        """process_request returns a BrainState instance."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert isinstance(state, BrainState)

    def test_code_agent_has_final_output(self, controller: BrainController) -> None:
        """A code request produces a final output."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert state.final_output


# â”€â”€â”€ Explain Agent Integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestExplainAgentIntegration:
    """Tests that explain-related requests trigger the ExplainAgent."""

    def test_explain_agent_integration(self, controller: BrainController) -> None:
        """An 'Explain Python' request triggers ExplainAgent and mode='explain'."""
        state = controller.process_request(
            "Explain Python decorators step by step."
        )
        assert state.mode == "explain"
        assert "agent_id" in state.context
        assert state.context["agent_id"] == "explain_agent"

    def test_explain_agent_intent(self, controller: BrainController) -> None:
        """An explain request classifies intent as 'explain'."""
        state = controller.process_request(
            "Explain Python decorators step by step."
        )
        assert state.intent == "explain"

    def test_explain_agent_has_final_output(self, controller: BrainController) -> None:
        """An explain request produces a final output."""
        state = controller.process_request(
            "Explain Python decorators step by step."
        )
        assert state.final_output


# â”€â”€â”€ Analyze Agent Integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAnalyzeAgentIntegration:
    """Tests that analyze-related requests trigger the AnalyzeAgent."""

    def test_analyze_agent_integration(self, controller: BrainController) -> None:
        """An 'Analyze this code' request triggers AnalyzeAgent and mode='analyze'."""
        state = controller.process_request(
            "Analyze and review the performance of this code."
        )
        assert state.mode == "analyze"
        assert "agent_id" in state.context
        assert state.context["agent_id"] == "analyze_agent"

    def test_analyze_agent_intent(self, controller: BrainController) -> None:
        """An analyze request classifies intent as 'analyze'."""
        state = controller.process_request(
            "Analyze and review the performance of this code."
        )
        assert state.intent == "analyze"

    def test_analyze_agent_has_final_output(self, controller: BrainController) -> None:
        """An analyze request produces a final output."""
        state = controller.process_request(
            "Analyze and review the performance of this code."
        )
        assert state.final_output


# â”€â”€â”€ Agent Metadata in Context â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestAgentMetadata:
    """Tests that agent metadata appears in state.context after pipeline."""

    def test_agent_id_in_context(self, controller: BrainController) -> None:
        """agent_id is present in state.context after processing."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert "agent_id" in state.context
        assert isinstance(state.context["agent_id"], str)
        assert len(state.context["agent_id"]) > 0

    def test_agent_confidence_in_context(self, controller: BrainController) -> None:
        """agent_confidence is present in state.context after processing."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert "agent_confidence" in state.context
        confidence = state.context["agent_confidence"]
        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0

    def test_agent_mode_in_context(self, controller: BrainController) -> None:
        """agent_mode is present in state.context and matches state.mode."""
        state = controller.process_request("Write a Python function to sort a list.")
        assert "agent_mode" in state.context
        assert state.context["agent_mode"] == state.mode

    def test_agent_metadata_consistent_across_modes(
        self, controller: BrainController
    ) -> None:
        """Agent metadata is present for all three agent modes."""
        inputs_and_modes = [
            ("Write a Python function to sort a list.", "code", "code_agent"),
            ("Explain Python decorators step by step.", "explain", "explain_agent"),
            (
                "Analyze and review the performance of this code.",
                "analyze",
                "analyze_agent",
            ),
        ]
        for text, expected_mode, expected_agent_id in inputs_and_modes:
            state = controller.process_request(text)
            assert state.mode == expected_mode, f"Failed for input: {text}"
            assert state.context.get("agent_id") == expected_agent_id, (
                f"Failed for input: {text}"
            )
            assert "agent_confidence" in state.context, f"Failed for input: {text}"


# â”€â”€â”€ SearchAgentGuard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSearchAgentGuard:
    """Tests that SearchAgentGuard doesn't crash the pipeline on valid input."""

    def test_guard_validates_valid_input(self, controller: BrainController) -> None:
        """SearchAgentGuard.validate returns (True, 'ok') for valid input."""
        valid, reason = controller.search_agent_guard.validate(
            "Write a Python function to sort a list."
        )
        assert valid is True
        assert reason == "ok"

    def test_guard_does_not_crash_pipeline(self, controller: BrainController) -> None:
        """The pipeline completes without guard-related crashes on valid input."""
        state = controller.process_request("Write a Python function to sort a list.")
        # The pipeline should complete and produce output
        assert state.final_output
        # No guard-related errors should be present
        guard_errors = [e for e in state.errors if "guard" in e.lower()]
        assert guard_errors == []

    def test_guard_rejects_empty_input(self, controller: BrainController) -> None:
        """SearchAgentGuard.validate rejects empty/whitespace input."""
        valid, reason = controller.search_agent_guard.validate("   ")
        assert valid is False

    def test_guard_rejects_none_input(self, controller: BrainController) -> None:
        """SearchAgentGuard.validate rejects None input."""
        valid, reason = controller.search_agent_guard.validate(None)
        assert valid is False


# â”€â”€â”€ Pipeline Stage Count â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestPipelineStageCount:
    """Tests that the pipeline has 20 stages (including 02b, 05b, 08b, 09b, 09c, 09d)."""

    def test_pipeline_has_20_stages(self, controller: BrainController) -> None:
        """The pipeline orchestrator has exactly 20 stages."""
        assert controller.pipeline.stage_count() == 20

    def test_pipeline_has_new_enrichment_stages(
        self, controller: BrainController
    ) -> None:
        """The pipeline includes the new 02b, 05b, 08b, 09b, 09c, 09d stages."""
        stage_names = [type(s).__name__ for s in controller.pipeline._stages]
        assert "Stage02bIntentEnrichment" in stage_names
        assert "Stage05bContextEnrichment" in stage_names
        assert "Stage08bAgentExecution" in stage_names
        assert "Stage09bResponseQuality" in stage_names
        assert "Stage09cPatternLearning" in stage_names
        assert "Stage09dMetacognition" in stage_names

    def test_pipeline_stage_order(self, controller: BrainController) -> None:
        """The 20 stages are in the correct order."""
        stage_names = [type(s).__name__ for s in controller.pipeline._stages]
        expected_order = [
            "Stage01Preprocessor",
            "Stage02Intent",
            "Stage02bIntentEnrichment",
            "Stage03ModeSelector",
            "Stage04DomainRouter",
            "Stage05ContextBuilder",
            "Stage05bContextEnrichment",
            "Stage06KnowledgeFetch",
            "Stage07MemoryFetch",
            "Stage08ReasoningCore",
            "Stage08bAgentExecution",
            "Stage09Synthesis",
            "Stage09bResponseQuality",
            "Stage09cPatternLearning",
            "Stage09dMetacognition",
            "Stage10Personality",
            "Stage11Ethics",
            "Stage12Security",
            "Stage13OutputBuilder",
            "Stage14Delivery",
        ]
        assert stage_names == expected_order

    def test_status_reports_20_stages(self, controller: BrainController) -> None:
        """The status snapshot reports 20 pipeline stages."""
        status = controller.status()
        assert status["pipeline_stages"] == 20
