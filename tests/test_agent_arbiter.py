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

"""Tests for AgentArbiter — Phase 2-D component."""

from __future__ import annotations

from core.conversational.agent_arbiter import (
    AgentArbiter,
    AgentCapability,
    AgentDescriptor,
    ArbitrationResult,
)


def _make_agent(
    agent_id: str,
    caps: tuple[AgentCapability, ...],
    priority: int = 5,
    max_concurrent: int = 10,
) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        name=f"Agent {agent_id}",
        capabilities=caps,
        priority=priority,
        max_concurrent=max_concurrent,
    )


class TestAgentArbiter:
    """Unit tests for AgentArbiter."""

    def test_register_and_select(self) -> None:
        """Agent is registered and selected for matching capability."""
        arbiter = AgentArbiter()
        agent = _make_agent("search_1", (AgentCapability.SEARCH,))
        arbiter.register(agent)
        result = arbiter.select(AgentCapability.SEARCH, "task_1")
        assert result.selected_agent is not None
        assert result.selected_agent.agent_id == "search_1"

    def test_no_agent_available(self) -> None:
        """No agent returns None."""
        arbiter = AgentArbiter()
        result = arbiter.select(AgentCapability.SEARCH, "task_1")
        assert result.selected_agent is None

    def test_priority_selection(self) -> None:
        """Higher priority agent is selected."""
        arbiter = AgentArbiter()
        low = _make_agent("low", (AgentCapability.SEARCH,), priority=1)
        high = _make_agent("high", (AgentCapability.SEARCH,), priority=9)
        arbiter.register(low)
        arbiter.register(high)
        result = arbiter.select(AgentCapability.SEARCH, "task_1")
        assert result.selected_agent.agent_id == "high"

    def test_capability_filter(self) -> None:
        """Only agents with matching capability are selected."""
        arbiter = AgentArbiter()
        search_agent = _make_agent("search", (AgentCapability.SEARCH,))
        code_agent = _make_agent("code", (AgentCapability.CODE,))
        arbiter.register(search_agent)
        arbiter.register(code_agent)
        result = arbiter.select(AgentCapability.CODE, "task_1")
        assert result.selected_agent.agent_id == "code"

    def test_general_fallback(self) -> None:
        """General-purpose agent is used as fallback."""
        arbiter = AgentArbiter()
        general = _make_agent("general", (AgentCapability.GENERAL,))
        arbiter.register(general)
        result = arbiter.select(AgentCapability.SEARCH, "task_1")
        assert result.selected_agent is not None
        assert result.selected_agent.agent_id == "general"

    def test_exclude_agents(self) -> None:
        """Excluded agents are not selected."""
        arbiter = AgentArbiter()
        a1 = _make_agent("a1", (AgentCapability.SEARCH,), priority=9)
        a2 = _make_agent("a2", (AgentCapability.SEARCH,), priority=5)
        arbiter.register(a1)
        arbiter.register(a2)
        result = arbiter.select(AgentCapability.SEARCH, "task_1", exclude=("a1",))
        assert result.selected_agent.agent_id == "a2"

    def test_handoff_detection(self) -> None:
        """Handoff is detected when a different agent is assigned."""
        arbiter = AgentArbiter()
        a1 = _make_agent("a1", (AgentCapability.SEARCH,), priority=5)
        a2 = _make_agent("a2", (AgentCapability.CODE,), priority=9)
        arbiter.register(a1)
        arbiter.register(a2)
        # First assign to search agent
        arbiter.select(AgentCapability.SEARCH, "task_1")
        # Then select for code — should trigger handoff
        result = arbiter.select(AgentCapability.CODE, "task_1")
        assert result.handoff_required is True

    def test_release_assignment(self) -> None:
        """Release clears the assignment."""
        arbiter = AgentArbiter()
        agent = _make_agent("a1", (AgentCapability.SEARCH,))
        arbiter.register(agent)
        arbiter.select(AgentCapability.SEARCH, "task_1")
        arbiter.release("task_1")
        assert "task_1" not in arbiter.get_active_assignments()

    def test_unregister(self) -> None:
        """Unregister removes an agent."""
        arbiter = AgentArbiter()
        agent = _make_agent("a1", (AgentCapability.SEARCH,))
        arbiter.register(agent)
        assert arbiter.unregister("a1") is True
        assert arbiter.agent_count() == 0

    def test_concurrency_limit(self) -> None:
        """Concurrency limit prevents over-assignment."""
        arbiter = AgentArbiter()
        agent = _make_agent("a1", (AgentCapability.SEARCH,), max_concurrent=1)
        arbiter.register(agent)
        r1 = arbiter.select(AgentCapability.SEARCH, "task_1")
        assert r1.selected_agent is not None
        r2 = arbiter.select(AgentCapability.SEARCH, "task_2")
        # Should fail — max concurrent reached
        assert r2.selected_agent is None

    def test_handoff_log(self) -> None:
        """Handoff log records handoffs."""
        arbiter = AgentArbiter()
        a1 = _make_agent("a1", (AgentCapability.SEARCH,), priority=5)
        a2 = _make_agent("a2", (AgentCapability.CODE,), priority=9)
        arbiter.register(a1)
        arbiter.register(a2)
        arbiter.select(AgentCapability.SEARCH, "task_1")
        arbiter.select(AgentCapability.CODE, "task_1")
        log = arbiter.get_handoff_log()
        assert len(log) > 0

    def test_arbitration_result_is_frozen(self) -> None:
        """ArbitrationResult is immutable."""
        arbiter = AgentArbiter()
        agent = _make_agent("a1", (AgentCapability.SEARCH,))
        arbiter.register(agent)
        result = arbiter.select(AgentCapability.SEARCH, "task_1")
        try:
            result.reason = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_all_agents(self) -> None:
        """all_agents returns all registered agents."""
        arbiter = AgentArbiter()
        arbiter.register(_make_agent("a1", (AgentCapability.SEARCH,)))
        arbiter.register(_make_agent("a2", (AgentCapability.CODE,)))
        assert len(arbiter.all_agents()) == 2
