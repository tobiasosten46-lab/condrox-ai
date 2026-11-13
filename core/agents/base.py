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

"""Base agent interface for Condrox core agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController
    from core.brain.brain_state import BrainState


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Output produced by a mode-specific agent.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1) for fixed metadata.
    """

    agent_id: str
    mode: str
    confidence: float
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)
    synthesized_output: str = ""
    errors: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base for all Condrox mode agents."""

    __slots__ = ("agent_id", "mode", "description")

    def __init__(self, agent_id: str, mode: str, description: str = "") -> None:
        self.agent_id: str = agent_id
        self.mode: str = mode
        self.description: str = description

    @abstractmethod
    def run(
        self, controller: "BrainController", state: "BrainState"
    ) -> AgentResult:
        """Execute the agent and return a result."""
        ...

    def status(self) -> dict[str, Any]:
        """Return serializable agent status."""
        return {
            "agent_id": self.agent_id,
            "mode": self.mode,
            "description": self.description,
            "ready": True,
        }
