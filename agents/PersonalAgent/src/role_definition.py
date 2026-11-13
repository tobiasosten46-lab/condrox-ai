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
Foundation Layer — Role Definition for Personal Assistant Agent.

Defines the agent's identity, role, and responsibility area following Condrox architecture:
- Main AI is the brain; agent is only an extension
- Agent has no own intelligence, pipeline, identity, reasoning, or context
- Agent provides role-specific tools, tone, and workflows

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class PersonaMode(Enum):
    """
    Agent persona mode - indicates this is an extension, not a separate AI.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    ASSISTANT = "assistant"  # PA-mode: extension of Main AI


class ToneProfile(Enum):
    """
    Tone profile for Personal Assistant role.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    PROFESSIONAL = "professional"  # Professional, organized, polite, efficient
    CASUAL = "casual"  # Friendly, relaxed
    FORMAL = "formal"  # Very formal, structured


@dataclass(frozen=True, slots=True)
class DomainTags:
    """
    Domain tags for Personal Assistant role.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    schedule: bool = True
    reminder: bool = True
    task: bool = True
    info: bool = True
    planning: bool = True

    def to_list(self) -> list[str]:
        """Convert to list of active tags.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n) for n tags.
        """
        tags = []
        if self.schedule:
            tags.append("schedule")
        if self.reminder:
            tags.append("reminder")
        if self.task:
            tags.append("task")
        if self.info:
            tags.append("info")
        if self.planning:
            tags.append("planning")
        return tags


@dataclass(frozen=True, slots=True)
class RoleDefinition:
    """
    Role definition for Personal Assistant agent.

    Defines the agent's identity, role, and responsibility area.
    This is NOT a separate AI - it's an extension of Main AI.

    Attributes:
        role: Agent role (Personal Assistant)
        persona_mode: Assistant-mode (extension of Main AI)
        tone_profile: Professional, organized, polite, efficient
        domain_tags: Domain tags for PA role
        description: Role description

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    role: str = "Personal Assistant"
    persona_mode: PersonaMode = PersonaMode.ASSISTANT
    tone_profile: ToneProfile = ToneProfile.PROFESSIONAL
    domain_tags: DomainTags = field(default_factory=DomainTags)
    description: str = (
        "Personal Assistant agent that extends Main AI with PA-specific tools, "
        "workflows, and tone. Main AI is the brain; this agent is only an extension."
    )

    def __post_init__(self):
        """Validate role definition.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.persona_mode != PersonaMode.ASSISTANT:
            raise ValueError("Personal Agent must use ASSISTANT persona mode")

    def get_greeting(self) -> str:
        """Get role-appropriate greeting.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.tone_profile == ToneProfile.PROFESSIONAL:
            return "Hello. I'm your Personal Assistant. How can I help you today?"
        elif self.tone_profile == ToneProfile.CASUAL:
            return "Hi there! I'm your Personal Assistant. What can I help you with?"
        elif self.tone_profile == ToneProfile.FORMAL:
            return "Good day. I am your Personal Assistant. How may I assist you?"
        return "Hello. I'm your Personal Assistant."

    def get_farewell(self) -> str:
        """Get role-appropriate farewell.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.tone_profile == ToneProfile.PROFESSIONAL:
            return "Thank you. Have a productive day."
        elif self.tone_profile == ToneProfile.CASUAL:
            return "Thanks! Have a great day!"
        elif self.tone_profile == ToneProfile.FORMAL:
            return "Thank you for your time. Good day."
        return "Thank you. Goodbye."
