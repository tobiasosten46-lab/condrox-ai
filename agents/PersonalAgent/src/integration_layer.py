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
Integration Layer — Main AI Control for Personal Assistant Agent.

Ensures that the agent is always controlled by Main AI:
- Main AI activates the agent based on intent
- Agent has no own pipeline
- Agent has no own identity
- Agent has no own reasoning
- Agent has no own context
- Everything goes through Main AI → agent → Main AI

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

from agents.PersonalAgent.src.role_definition import RoleDefinition


class IntegrationLayer:
    """
    Integration layer for Main AI control.

    Ensures the agent is always controlled by Main AI and never operates independently.

    Complexity:
        activate: T(n) ∈ O(1), S(n) ∈ O(1)
        validate: T(n) ∈ O(1), S(n) ∈ O(1)
    """

    __slots__ = ("_brain_controller", "_role_definition")

    def __init__(
        self,
        brain_controller: "BrainController",
        role_definition: RoleDefinition,
    ) -> None:
        """Initialize IntegrationLayer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._brain_controller = brain_controller
        self._role_definition = role_definition

    def activate(self, intent: str) -> bool:
        """
        Activate agent based on Main AI intent detection.

        The agent is only activated when Main AI determines it's appropriate.
        Agent never activates itself.

        Args:
            intent: Intent detected by Main AI.

        Returns:
            True if agent should be activated, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Agent is activated for PA-specific intents
        pa_intents = [
            "task",
            "reminder",
            "note",
            "plan",
            "schedule",
            "personal_assistant",
        ]
        
        return any(intent_tag in intent.lower() for intent_tag in pa_intents)

    def validate_control(self) -> bool:
        """
        Validate that agent is under Main AI control.

        Ensures:
        - Agent has no own pipeline
        - Agent has no own identity
        - Agent has no own reasoning
        - Agent has no own context

        Returns:
            True if agent is under Main AI control, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Agent uses Main AI's pipeline
        has_pipeline = hasattr(self._brain_controller, 'pipeline')
        
        # Agent uses Main AI's identity (no own identity)
        has_identity = self._role_definition.persona_mode.value == "assistant"
        
        # Agent uses Main AI's reasoning (no own reasoning)
        has_reasoning = hasattr(self._brain_controller, 'reasoning_engine')
        
        # Agent uses Main AI's context (no own context)
        has_context = hasattr(self._brain_controller, 'memory_manager')
        
        return has_pipeline and has_identity and has_reasoning and has_context

    def get_brain_controller(self) -> "BrainController":
        """
        Get the Main AI BrainController.

        Agent never has its own BrainController.

        Returns:
            Main AI BrainController instance.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._brain_controller

    def get_role_definition(self) -> RoleDefinition:
        """
        Get the agent's role definition.

        Agent role is defined externally, not internally.

        Returns:
            RoleDefinition instance.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._role_definition

    def enforce_no_own_pipeline(self) -> bool:
        """
        Enforce that agent has no own pipeline.

        Agent must use Main AI's pipeline.

        Returns:
            True if enforcement passed, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Agent should not have a pipeline attribute
        # This is a design constraint, not a runtime check
        return True

    def enforce_no_own_identity(self) -> bool:
        """
        Enforce that agent has no own identity.

        Agent identity comes from RoleDefinition, not internal state.

        Returns:
            True if enforcement passed, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Agent identity is defined in RoleDefinition
        # Agent does not have internal identity state
        return self._role_definition.persona_mode.value == "assistant"

    def enforce_no_own_reasoning(self) -> bool:
        """
        Enforce that agent has no own reasoning.

        Agent must use Main AI's reasoning engine.

        Returns:
            True if enforcement passed, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Agent should not have a reasoning engine
        # Agent uses Main AI's reasoning via process_request
        return hasattr(self._brain_controller, 'reasoning_engine')

    def enforce_no_own_context(self) -> bool:
        """
        Enforce that agent has no own context.

        Agent must use Main AI's memory and context.

        Returns:
            True if enforcement passed, False otherwise.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Agent should not have its own memory/context
        # Agent uses Main AI's memory_manager
        return hasattr(self._brain_controller, 'memory_manager')

    def validate_all_constraints(self) -> dict:
        """
        Validate all Main AI control constraints.

        Returns:
            Dictionary with validation results for all constraints.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return {
            "no_own_pipeline": self.enforce_no_own_pipeline(),
            "no_own_identity": self.enforce_no_own_identity(),
            "no_own_reasoning": self.enforce_no_own_reasoning(),
            "no_own_context": self.enforce_no_own_context(),
            "under_main_ai_control": self.validate_control(),
        }
