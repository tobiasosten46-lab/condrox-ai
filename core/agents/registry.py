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

"""Agent registry for BrainController and admin API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from core.agents.base import BaseAgent


@dataclass
class AgentRegistry:
    """Registry of mode-specific agents.

    Provides the interface expected by admin_routes.py and BrainController.

    Complexity:
        register: T(n) ∈ O(1)
        get:      T(n) ∈ O(1)
        count:    T(n) ∈ O(1)
    """

    _agents: Dict[str, BaseAgent] = field(default_factory=dict)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent by its mode."""
        self._agents[agent.mode] = agent

    def get(self, mode: str) -> Optional[BaseAgent]:
        """Retrieve an agent for the given mode."""
        return self._agents.get(mode)

    def unregister(self, mode: str) -> Optional[BaseAgent]:
        """Remove and return an registered agent by mode.

        Complexity: T(n) ∈ O(1).
        """
        return self._agents.pop(mode, None)

    def count(self) -> int:
        """Return the number of registered agents."""
        return len(self._agents)

    def modes(self) -> Iterable[str]:
        """Return all registered modes."""
        return self._agents.keys()

    def status(self) -> List[dict[str, Any]]:
        """Return serializable status for all agents."""
        return [agent.status() for agent in self._agents.values()]

    def all_agents(self) -> List[BaseAgent]:
        """Return all registered agent instances."""
        return list(self._agents.values())
