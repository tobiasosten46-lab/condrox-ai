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
Unit tests for the Phase 3 Agent system.

NOTE: This test file is DISABLED because core.agents does not exist.
The actual agents are in agents/DeveloperAgent and agents/PersonalAgent.
See test_condrox_agent_coder.py for DeveloperAgent tests.

Tests cover:
- AgentBase lifecycle (prepare, execute, finalize, error handling)
- AgentContext and AgentResult immutability
- AgentRegistry registration, lookup, fallback
- CodeAgent sub-intent detection and execution
- AnalyzeAgent analysis type detection and execution
- ArchitectAgent scope detection and execution
- ExplainAgent depth detection and execution
- DefaultAgent fallback execution
- BrainController agent integration

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

# DISABLED: core.agents does not exist
# from core.agents.agent_base import (
#     AgentBase,
#     AgentContext,
#     AgentMode,
#     AgentRegistry,
#     AgentResult,
#     AgentState,
# )
# from core.agents.analyze_agent import AnalyzeAgent
# from core.agents.architect_agent import ArchitectAgent
# from core.agents.code_agent import CodeAgent
# from core.agents.default_agent import DefaultAgent
# from core.agents.explain_agent import ExplainAgent
# from core.brain.brain_controller import BrainController
# from core.brain.brain_state import BrainState

pytest.skip("core.agents does not exist - agents are in agents/DeveloperAgent and agents/PersonalAgent", allow_module_level=True)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def controller() -> BrainController:
    """Provide a booted BrainController."""
    bc = BrainController()
    bc.boot_system()
    return bc


@pytest.fixture
def context() -> AgentContext:
    """Provide a basic AgentContext."""
    return AgentContext(
        request_id="test-001",
        input_text="Write a Python function that sorts a list.",
        mode="code",
        domain="programming",
    )


# ─── AgentContext tests ──────────────────────────────────────────────────────


class TestAgentContext:
    """Tests for AgentContext immutability and construction."""

    def test_context_is_frozen(self) -> None:
        """AgentContext is immutable."""
        ctx = AgentContext(
            request_id="r1", input_text="test", mode="code", domain="general"
        )
        with pytest.raises((AttributeError, Exception)):
            ctx.input_text = "changed"  # type: ignore[misc]

    def test_context_from_state(self) -> None:
        """AgentContext.from_state extracts fields from BrainState."""
        state = BrainState(
            request_id="r2",
            raw_input="Explain Python.",
            processed_input="Explain Python.",
            mode="explain",
            domain="general",
        )
        ctx = AgentContext.from_state(state)
        assert ctx.request_id == "r2"
        assert ctx.input_text == "Explain Python."
        assert ctx.mode == "explain"
        assert ctx.domain == "general"

    def test_context_defaults(self) -> None:
        """AgentContext has correct defaults."""
        ctx = AgentContext(
            request_id="r1", input_text="test", mode="code", domain="general"
        )
        assert ctx.retrieved_knowledge == ()
        assert ctx.retrieved_memories == ()
        assert ctx.context == {}
        assert ctx.metadata == {}


# ─── AgentResult tests ───────────────────────────────────────────────────────


class TestAgentResult:
    """Tests for AgentResult immutability and failure constructor."""

    def test_result_is_frozen(self) -> None:
        """AgentResult is immutable."""
        result = AgentResult(agent_id="a1", mode="code")
        with pytest.raises((AttributeError, Exception)):
            result.confidence = 0.9  # type: ignore[misc]

    def test_failure_result(self) -> None:
        """AgentResult.failure creates a proper failure result."""
        result = AgentResult.failure("a1", "code", "something went wrong")
        assert result.agent_id == "a1"
        assert result.mode == "code"
        assert result.confidence == 0.0
        assert "something went wrong" in result.errors


# ─── AgentBase tests ─────────────────────────────────────────────────────────


class TestAgentBase:
    """Tests for the AgentBase lifecycle."""

    def test_agent_lifecycle_states(self, controller: BrainController, context: AgentContext) -> None:
        """Agent transitions through lifecycle states."""

        class TestAgent(AgentBase):
            def _execute(self, ctx, ctrl):
                return AgentResult(
                    agent_id=self._agent_id,
                    mode=self._mode.value,
                    synthesized_output="test output",
                    confidence=0.8,
                )

        agent = TestAgent("test_agent", AgentMode.CODE)
        assert agent.state == AgentState.IDLE

        result = agent.run(context, controller)
        assert agent.state == AgentState.COMPLETED
        assert result.synthesized_output == "test output"
        assert result.confidence == 0.8
        assert agent.execution_count == 1
        assert agent.error_count == 0

    def test_agent_prepare_error(self, controller: BrainController) -> None:
        """Agent handles prepare errors gracefully."""

        class FailPrepareAgent(AgentBase):
            def _prepare(self, ctx, ctrl):
                raise RuntimeError("prepare failed")

            def _execute(self, ctx, ctrl):
                return AgentResult(agent_id=self._agent_id, mode=self._mode.value)

        agent = FailPrepareAgent("fail_prep", AgentMode.CODE)
        ctx = AgentContext(
            request_id="r1", input_text="test", mode="code", domain="general"
        )
        result = agent.run(ctx, controller)
        assert agent.state == AgentState.FAILED
        assert agent.error_count == 1
        assert "prepare error" in result.errors[0]

    def test_agent_execute_error(self, controller: BrainController, context: AgentContext) -> None:
        """Agent handles execute errors gracefully."""

        class FailExecuteAgent(AgentBase):
            def _execute(self, ctx, ctrl):
                raise RuntimeError("execute failed")

        agent = FailExecuteAgent("fail_exec", AgentMode.CODE)
        result = agent.run(context, controller)
        assert agent.state == AgentState.FAILED
        assert agent.error_count == 1
        assert "execute error" in result.errors[0]

    def test_agent_empty_input_raises(self, controller: BrainController) -> None:
        """Agent rejects empty input in prepare."""
        agent = CodeAgent()
        ctx = AgentContext(
            request_id="r1", input_text="", mode="code", domain="general"
        )
        result = agent.run(ctx, controller)
        assert result.confidence == 0.0
        assert "prepare error" in result.errors[0]

    def test_agent_reset(self, controller: BrainController, context: AgentContext) -> None:
        """Agent reset returns to idle."""
        agent = CodeAgent()
        agent.run(context, controller)
        assert agent.state == AgentState.COMPLETED
        agent.reset()
        assert agent.state == AgentState.IDLE

    def test_agent_success_rate(self, controller: BrainController, context: AgentContext) -> None:
        """Success rate is computed correctly."""
        agent = CodeAgent()
        agent.run(context, controller)
        agent.run(context, controller)
        assert agent.execution_count == 2
        assert agent.error_count == 0
        assert agent.success_rate == 1.0

    def test_agent_status(self, controller: BrainController, context: AgentContext) -> None:
        """Status returns a valid snapshot."""
        agent = CodeAgent()
        agent.run(context, controller)
        status = agent.status()
        assert status["agent_id"] == "code_agent"
        assert status["mode"] == "code"
        assert status["state"] == "completed"
        assert status["execution_count"] == 1


# ─── AgentRegistry tests ─────────────────────────────────────────────────────


class TestAgentRegistry:
    """Tests for the AgentRegistry."""

    def test_register_and_get(self) -> None:
        """Registry registers and retrieves agents."""
        registry = AgentRegistry()
        agent = CodeAgent()
        registry.register(agent)
        assert registry.get("code") is agent

    def test_register_duplicate_raises(self) -> None:
        """Registry rejects duplicate mode registration."""
        registry = AgentRegistry()
        registry.register(CodeAgent())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CodeAgent())

    def test_get_or_default_fallback(self) -> None:
        """Registry falls back to default agent."""
        registry = AgentRegistry()
        registry.register(CodeAgent())
        registry.register(DefaultAgent())
        agent = registry.get_or_default("nonexistent")
        assert agent.mode == AgentMode.DEFAULT

    def test_get_or_default_no_default_raises(self) -> None:
        """Registry raises if neither mode nor default exists."""
        registry = AgentRegistry()
        registry.register(CodeAgent())
        with pytest.raises(KeyError):
            registry.get_or_default("nonexistent")

    def test_unregister(self) -> None:
        """Registry unregisters agents."""
        registry = AgentRegistry()
        registry.register(CodeAgent())
        assert registry.unregister("code") is True
        assert registry.get("code") is None
        assert registry.unregister("code") is False

    def test_modes(self) -> None:
        """Registry returns all modes."""
        registry = AgentRegistry()
        registry.register(CodeAgent())
        registry.register(ExplainAgent())
        modes = registry.modes()
        assert "code" in modes
        assert "explain" in modes

    def test_count(self) -> None:
        """Registry count is correct."""
        registry = AgentRegistry()
        assert registry.count() == 0
        registry.register(CodeAgent())
        assert registry.count() == 1

    def test_status(self) -> None:
        """Registry status returns valid snapshot."""
        registry = AgentRegistry()
        registry.register(CodeAgent())
        status = registry.status()
        assert status["agent_count"] == 1
        assert "code" in status["modes"]
        assert "code_agent" in status["agents"]


# ─── CodeAgent tests ─────────────────────────────────────────────────────────


class TestCodeAgent:
    """Tests for the CodeAgent."""

    def test_detect_fix_intent(self) -> None:
        """CodeAgent detects fix sub-intent."""
        agent = CodeAgent()
        assert agent._detect_sub_intent("Fix this bug in my code") == "fix"
        assert agent._detect_sub_intent("There's an error in my function") == "fix"

    def test_detect_review_intent(self) -> None:
        """CodeAgent detects review sub-intent."""
        agent = CodeAgent()
        assert agent._detect_sub_intent("Review my code") == "review"
        assert agent._detect_sub_intent("Audit this module") == "review"

    def test_detect_refactor_intent(self) -> None:
        """CodeAgent detects refactor sub-intent."""
        agent = CodeAgent()
        assert agent._detect_sub_intent("Refactor this class") == "refactor"

    def test_detect_generate_intent(self) -> None:
        """CodeAgent detects generate sub-intent."""
        agent = CodeAgent()
        assert agent._detect_sub_intent("Write a Python function") == "generate"
        assert agent._detect_sub_intent("Create a new module") == "generate"

    def test_detect_general_intent(self) -> None:
        """CodeAgent falls back to general."""
        agent = CodeAgent()
        assert agent._detect_sub_intent("Python is a language") == "general"

    def test_execute_produces_result(
        self, controller: BrainController, context: AgentContext
    ) -> None:
        """CodeAgent produces a valid result."""
        agent = CodeAgent()
        result = agent.run(context, controller)
        assert result.agent_id == "code_agent"
        assert result.mode == "code"
        assert result.confidence > 0.0
        assert "Analyzing" in result.synthesized_output or "code" in result.synthesized_output.lower()
        assert "generate" in result.metadata["sub_intent"]

    def test_execute_fix_confidence(
        self, controller: BrainController
    ) -> None:
        """CodeAgent has high confidence for fix intent."""
        agent = CodeAgent()
        ctx = AgentContext(
            request_id="r1",
            input_text="Fix the bug in my sorting function.",
            mode="code",
            domain="programming",
        )
        result = agent.run(ctx, controller)
        assert result.confidence == 0.9
        assert result.metadata["sub_intent"] == "fix"


# ─── AnalyzeAgent tests ──────────────────────────────────────────────────────


class TestAnalyzeAgent:
    """Tests for the AnalyzeAgent."""

    def test_detect_performance(self) -> None:
        """AnalyzeAgent detects performance analysis."""
        agent = AnalyzeAgent()
        assert agent._detect_analysis_type("Analyze the performance of this system") == "performance"

    def test_detect_security(self) -> None:
        """AnalyzeAgent detects security analysis."""
        agent = AnalyzeAgent()
        assert agent._detect_analysis_type("Security analysis of the API") == "security"

    def test_detect_architecture(self) -> None:
        """AnalyzeAgent detects architecture analysis."""
        agent = AnalyzeAgent()
        assert agent._detect_analysis_type("Analyze the architecture design") == "architecture"

    def test_detect_data(self) -> None:
        """AnalyzeAgent detects data analysis."""
        agent = AnalyzeAgent()
        assert agent._detect_analysis_type("Analyze the dataset statistics") == "data"

    def test_execute_produces_result(
        self, controller: BrainController
    ) -> None:
        """AnalyzeAgent produces a valid result."""
        agent = AnalyzeAgent()
        ctx = AgentContext(
            request_id="r1",
            input_text="Analyze the performance of this system.",
            mode="analyze",
            domain="general",
        )
        result = agent.run(ctx, controller)
        assert result.agent_id == "analyze_agent"
        assert result.mode == "analyze"
        assert "Analysis" in result.synthesized_output or "analysis" in result.synthesized_output.lower()
        assert "performance" in result.metadata["analysis_type"]
        assert len(result.metadata["dimensions"]) == 6


# ─── ArchitectAgent tests ────────────────────────────────────────────────────


class TestArchitectAgent:
    """Tests for the ArchitectAgent."""

    def test_detect_system_scope(self) -> None:
        """ArchitectAgent detects system scope."""
        agent = ArchitectAgent()
        assert agent._detect_architecture_scope("Design the system architecture") == "system"

    def test_detect_component_scope(self) -> None:
        """ArchitectAgent detects component scope."""
        agent = ArchitectAgent()
        assert agent._detect_architecture_scope("Design a microservice component") == "component"

    def test_detect_integration_scope(self) -> None:
        """ArchitectAgent detects integration scope."""
        agent = ArchitectAgent()
        assert agent._detect_architecture_scope("Integrate the API with the database") == "integration"

    def test_detect_migration_scope(self) -> None:
        """ArchitectAgent detects migration scope."""
        agent = ArchitectAgent()
        assert agent._detect_architecture_scope("Migrate to a new infrastructure") == "migration"

    def test_execute_produces_result(
        self, controller: BrainController
    ) -> None:
        """ArchitectAgent produces a valid result."""
        agent = ArchitectAgent()
        ctx = AgentContext(
            request_id="r1",
            input_text="Design the system architecture for a chat platform.",
            mode="architect",
            domain="programming",
        )
        result = agent.run(ctx, controller)
        assert result.agent_id == "architect_agent"
        assert result.mode == "architect"
        assert "Architecture" in result.synthesized_output or "architecture" in result.synthesized_output.lower()
        assert "system" in result.metadata["scope"]
        assert len(result.metadata["concerns"]) == 6
        assert len(result.metadata["trade_offs"]) == 3


# ─── ExplainAgent tests ──────────────────────────────────────────────────────


class TestExplainAgent:
    """Tests for the ExplainAgent."""

    def test_detect_brief_depth(self) -> None:
        """ExplainAgent detects brief depth."""
        agent = ExplainAgent()
        assert agent._detect_explanation_depth("Give a brief summary of Python") == "brief"

    def test_detect_tutorial_depth(self) -> None:
        """ExplainAgent detects tutorial depth."""
        agent = ExplainAgent()
        assert agent._detect_explanation_depth("Step by step guide to Python") == "tutorial"

    def test_detect_detailed_depth(self) -> None:
        """ExplainAgent detects detailed depth."""
        agent = ExplainAgent()
        assert agent._detect_explanation_depth("Give a detailed explanation of closures") == "detailed"

    def test_execute_produces_result(
        self, controller: BrainController
    ) -> None:
        """ExplainAgent produces a valid result."""
        agent = ExplainAgent()
        ctx = AgentContext(
            request_id="r1",
            input_text="Explain what a closure is in JavaScript.",
            mode="explain",
            domain="general",
        )
        result = agent.run(ctx, controller)
        assert result.agent_id == "explain_agent"
        assert result.mode == "explain"
        assert "Explaining" in result.synthesized_output or "explanation" in result.synthesized_output.lower()
        assert len(result.metadata["stages"]) >= 2

    def test_execute_brief_has_fewer_stages(
        self, controller: BrainController
    ) -> None:
        """Brief explanation has fewer stages than general."""
        agent = ExplainAgent()
        ctx = AgentContext(
            request_id="r1",
            input_text="Brief summary of Python decorators.",
            mode="explain",
            domain="general",
        )
        result = agent.run(ctx, controller)
        assert result.metadata["depth"] == "brief"
        assert len(result.metadata["stages"]) == 2  # define, summarize


# ─── DefaultAgent tests ──────────────────────────────────────────────────────


class TestDefaultAgent:
    """Tests for the DefaultAgent."""

    def test_execute_produces_result(
        self, controller: BrainController
    ) -> None:
        """DefaultAgent produces a valid result."""
        agent = DefaultAgent()
        ctx = AgentContext(
            request_id="r1",
            input_text="Hello, how are you?",
            mode="default",
            domain="general",
        )
        result = agent.run(ctx, controller)
        assert result.agent_id == "default_agent"
        assert result.mode == "default"
        assert "Processing" in result.synthesized_output or "processing" in result.synthesized_output.lower()
        assert result.confidence == 0.5


# ─── BrainController integration tests ───────────────────────────────────────


class TestBrainControllerAgentIntegration:
    """Tests for BrainController + agent integration."""

    def test_controller_has_agent_registry(self, controller: BrainController) -> None:
        """BrainController has an agent registry."""
        assert controller.agent_registry is not None
        assert controller.agent_registry.count() == 5

    def test_controller_agent_modes(self, controller: BrainController) -> None:
        """BrainController has all 5 agent modes."""
        modes = controller.agent_registry.modes()
        assert "code" in modes
        assert "analyze" in modes
        assert "architect" in modes
        assert "explain" in modes
        assert "default" in modes

    def test_controller_run_agent_code(
        self, controller: BrainController
    ) -> None:
        """BrainController.run_agent executes the code agent."""
        state = BrainState(
            request_id="r1",
            raw_input="Write a Python function.",
            processed_input="Write a Python function.",
            mode="code",
            domain="programming",
        )
        result = controller.run_agent("code", state)
        assert result.agent_id == "code_agent"
        assert result.mode == "code"
        assert result.confidence > 0.0

    def test_controller_run_agent_falls_back(
        self, controller: BrainController
    ) -> None:
        """BrainController.run_agent falls back to default for unknown mode."""
        state = BrainState(
            request_id="r1",
            raw_input="Hello",
            processed_input="Hello",
            mode="unknown_mode",
            domain="general",
        )
        result = controller.run_agent("unknown_mode", state)
        assert result.agent_id == "default_agent"

    def test_controller_status_includes_agents(
        self, controller: BrainController
    ) -> None:
        """BrainController.status includes agent information."""
        status = controller.status()
        assert status["agent_registry"] is True
        assert status["agent_count"] == 5
        assert "code" in status["agent_modes"]
        assert "agent_status" in status
