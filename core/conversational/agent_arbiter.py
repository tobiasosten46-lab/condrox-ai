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
Agent Arbitration Layer — selects the correct agent for a given intent
and manages handoffs between agents.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class AgentCapability(Enum):
    """Agent capability categories."""

    SEARCH = "search"
    REASONING = "reasoning"
    CODE = "code"
    EXPLANATION = "explanation"
    DIALOG = "dialog"
    DIAGNOSTIC = "diagnostic"
    MEMORY = "memory"
    GENERAL = "general"


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """
    Immutable agent descriptor.

    Attributes:
        agent_id: Unique identifier.
        name: Human-readable name.
        capabilities: Tuple of capabilities this agent supports.
        priority: Priority in [0, 10] (higher = preferred).
        max_concurrent: Maximum concurrent tasks.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    agent_id: str
    name: str
    capabilities: tuple[AgentCapability, ...]
    priority: int
    max_concurrent: int


@dataclass(frozen=True, slots=True)
class ArbitrationResult:
    """
    Immutable result of agent arbitration.

    Attributes:
        selected_agent: The chosen agent descriptor.
        reason: Why this agent was selected.
        alternatives: Other agents that could handle the task.
        handoff_required: Whether a handoff from a different agent is needed.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    selected_agent: Optional[AgentDescriptor]
    reason: str
    alternatives: tuple[str, ...]
    handoff_required: bool


class AgentArbiter:
    """
    Selects the best agent for a task based on capabilities and priority.

    Complexity:
        select: T(n) ∈ O(a) for a agents, S(n) ∈ O(a).
        register: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_agents", "_active_assignments", "_handoff_log")

    def __init__(self) -> None:
        """Initialize the arbiter with no agents.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._agents: Dict[str, AgentDescriptor] = {}
        self._active_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self._handoff_log: List[Tuple[str, str, str]] = []  # (task, from, to)

    def register(self, agent: AgentDescriptor) -> None:
        """Register an agent.

        Args:
            agent: Agent descriptor to register.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> bool:
        """Unregister an agent.

        Returns:
            True if an agent was removed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._agents.pop(agent_id, None) is not None

    def select(
        self,
        required_capability: AgentCapability,
        task_id: str,
        exclude: Optional[Tuple[str, ...]] = None,
    ) -> ArbitrationResult:
        """Select the best agent for a required capability.

        Args:
            required_capability: The capability needed for this task.
            task_id: Unique task identifier.
            exclude: Agent IDs to exclude from selection.

        Returns:
            ArbitrationResult with the selected agent.

        Complexity: T(n) ∈ O(a) for a agents, S(n) ∈ O(a).
        """
        exclude_set = set(exclude or ())

        # Find agents with the required capability
        candidates: List[Tuple[int, AgentDescriptor]] = []
        for agent in self._agents.values():
            if agent.agent_id in exclude_set:
                continue
            if required_capability in agent.capabilities:
                # Check concurrency limit
                active_count = sum(
                    1 for aid in self._active_assignments.values()
                    if aid == agent.agent_id
                )
                if active_count >= agent.max_concurrent:
                    continue
                candidates.append((agent.priority, agent))

        if not candidates:
            # Try general-purpose agents as fallback
            for agent in self._agents.values():
                if agent.agent_id in exclude_set:
                    continue
                if AgentCapability.GENERAL in agent.capabilities:
                    active_count = sum(
                        1 for aid in self._active_assignments.values()
                        if aid == agent.agent_id
                    )
                    if active_count >= agent.max_concurrent:
                        continue
                    candidates.append((agent.priority, agent))

        if not candidates:
            return ArbitrationResult(
                selected_agent=None,
                reason=f"No agent available for capability {required_capability.value}",
                alternatives=(),
                handoff_required=False,
            )

        # Sort by priority descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]
        alternatives = tuple(c[1].agent_id for c in candidates[1:])

        # Check if handoff is needed
        current_agent_id = self._active_assignments.get(task_id)
        handoff_required = current_agent_id is not None and current_agent_id != selected.agent_id

        if handoff_required and current_agent_id:
            self._handoff_log.append((task_id, current_agent_id, selected.agent_id))

        # Record assignment
        self._active_assignments[task_id] = selected.agent_id

        return ArbitrationResult(
            selected_agent=selected,
            reason=f"Selected {selected.name} (priority={selected.priority})",
            alternatives=alternatives,
            handoff_required=handoff_required,
        )

    def release(self, task_id: str) -> None:
        """Release an active assignment.

        Args:
            task_id: Task to release.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._active_assignments.pop(task_id, None)

    def get_active_assignments(self) -> Dict[str, str]:
        """Return a copy of active assignments.

        Complexity: T(n) ∈ O(a), S(n) ∈ O(a).
        """
        return dict(self._active_assignments)

    def get_handoff_log(self, limit: int = 50) -> List[Tuple[str, str, str]]:
        """Return recent handoff log entries.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._handoff_log[-limit:])

    def get_agent(self, agent_id: str) -> Optional[AgentDescriptor]:
        """Get an agent descriptor by ID.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._agents.get(agent_id)

    def all_agents(self) -> List[AgentDescriptor]:
        """Return all registered agents.

        Complexity: T(n) ∈ O(a), S(n) ∈ O(a).
        """
        return list(self._agents.values())

    def agent_count(self) -> int:
        """Return the number of registered agents.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._agents)
