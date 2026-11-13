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

"""Adapter agents for PersonalAgent and DeveloperAgent (CondroxAgentCoder).

Wraps the standalone agent classes from agents/PersonalAgent and
agents/DeveloperAgent so they can be registered in the BrainController's
AgentRegistry and invoked via Stage08b (Agent Execution).

Complexity:
    run: T(n) ∈ O(wrapped agent), S(n) ∈ O(result).

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController
    from core.brain.brain_state import BrainState

logger = logging.getLogger(__name__)


class PersonalAgentAdapter(BaseAgent):
    """Adapter that wraps PersonalAgent as a BaseAgent for the registry.

    Mode: "personal" — activated for conversational, emotional, and
    personal-assistant intents (reminders, notes, tasks, daily brief).

    Complexity: T(n) ∈ O(pipeline + workflow), S(n) ∈ O(result).
    """

    __slots__ = ("_pa",)

    def __init__(self) -> None:
        super().__init__(
            agent_id="personal_agent",
            mode="personal",
            description="Personal assistant agent — conversational, tasks, reminders, notes.",
        )
        self._pa: Any = None

    def _ensure_pa(self, controller: "BrainController") -> Any:
        """Lazily initialize the PersonalAgent with the brain controller.

        Complexity: T(n) ∈ O(1) after first call.
        """
        if self._pa is None:
            try:
                from agents.PersonalAgent.personal_agent import PersonalAgent

                self._pa = PersonalAgent(controller)
                logger.info("PersonalAgentAdapter: PersonalAgent initialized.")
            except Exception as exc:
                logger.warning("PersonalAgentAdapter: failed to init PersonalAgent: %s", exc)
        return self._pa

    def run(self, controller: "BrainController", state: "BrainState") -> AgentResult:
        """Execute the PersonalAgent and return an AgentResult.

        Args:
            controller: BrainController instance (Main AI).
            state: Current BrainState with processed input and context.

        Returns:
            AgentResult with the personal assistant's response.

        Complexity: T(n) ∈ O(pipeline + workflow).
        """
        pa = self._ensure_pa(controller)
        if pa is None:
            return AgentResult(
                agent_id=self.agent_id,
                mode=self.mode,
                confidence=0.0,
                errors=["PersonalAgent initialization failed"],
            )

        try:
            context_str = ""
            if state.context_manager is not None:
                ctx = state.context_manager.current()
                context_str = ctx.get("conversation_history", "")
            elif isinstance(state.context, dict):
                context_str = state.context.get("conversation_history", "")

            result = pa.process(
                user_input=state.processed_input,
                context=context_str,
            )

            return AgentResult(
                agent_id=self.agent_id,
                mode=self.mode,
                confidence=result.confidence,
                reasoning=result.response,
                synthesized_output=result.response,
                evidence=[],
                errors=[],
            )
        except Exception as exc:
            logger.error("PersonalAgentAdapter: execution error: %s", exc)
            return AgentResult(
                agent_id=self.agent_id,
                mode=self.mode,
                confidence=0.0,
                errors=[f"PersonalAgent error: {exc}"],
            )


class DeveloperAgentAdapter(BaseAgent):
    """Adapter that wraps CondroxAgentCoder as a BaseAgent for the registry.

    Mode: "developer" — activated for code generation, debugging,
    refactoring, and syntax checking tasks.

    Complexity: T(n) ∈ O(pipeline + gates + rag), S(n) ∈ O(result).
    """

    __slots__ = ("_dev",)

    def __init__(self) -> None:
        super().__init__(
            agent_id="developer_agent",
            mode="developer",
            description="Developer coding agent — code generation, debugging, refactoring.",
        )
        self._dev: Any = None

    def _ensure_dev(self, controller: "BrainController") -> Any:
        """Lazily initialize the CondroxAgentCoder.

        Complexity: T(n) ∈ O(1) after first call.
        """
        if self._dev is None:
            try:
                from agents.DeveloperAgent.condrox_agent_coder import CondroxAgentCoder

                self._dev = CondroxAgentCoder()
                logger.info("DeveloperAgentAdapter: CondroxAgentCoder initialized.")
            except Exception as exc:
                logger.warning("DeveloperAgentAdapter: failed to init CondroxAgentCoder: %s", exc)
        return self._dev

    def run(self, controller: "BrainController", state: "BrainState") -> AgentResult:
        """Execute the CondroxAgentCoder and return an AgentResult.

        Args:
            controller: BrainController instance (Main AI).
            state: Current BrainState with processed input and context.

        Returns:
            AgentResult with the developer agent's code output.

        Complexity: T(n) ∈ O(pipeline + gates + rag).
        """
        dev = self._ensure_dev(controller)
        if dev is None:
            return AgentResult(
                agent_id=self.agent_id,
                mode=self.mode,
                confidence=0.0,
                errors=["CondroxAgentCoder initialization failed"],
            )

        try:
            import asyncio

            context_str = ""
            if state.context_manager is not None:
                ctx = state.context_manager.current()
                context_str = ctx.get("conversation_history", "")
            elif isinstance(state.context, dict):
                context_str = state.context.get("conversation_history", "")

            # Detect language from domain or default to python
            language = "python"
            if state.domain == "programming":
                query_lower = state.processed_input.lower()
                if "javascript" in query_lower or "js" in query_lower:
                    language = "javascript"
                elif "typescript" in query_lower:
                    language = "typescript"
                elif "java" in query_lower and "javascript" not in query_lower:
                    language = "java"
                elif "go " in query_lower or "golang" in query_lower:
                    language = "go"
                elif "rust" in query_lower:
                    language = "rust"

            result = asyncio.run(dev.execute(
                user_input=state.processed_input,
                brain_controller=controller,
                language=language,
                context=context_str,
            ))

            output_parts = []
            if hasattr(result, "output") and result.output:
                output_parts.append(result.output)
            if hasattr(result, "verification") and result.verification:
                output_parts.append(f"Verification: {result.verification}")
            if hasattr(result, "findings") and result.findings:
                output_parts.append(f"Findings: {'; '.join(result.findings)}")

            synthesized = "\n\n".join(output_parts) if output_parts else ""

            return AgentResult(
                agent_id=self.agent_id,
                mode=self.mode,
                confidence=getattr(result, "confidence", 0.8),
                reasoning=synthesized,
                synthesized_output=synthesized,
                evidence=[],
                errors=[],
            )
        except Exception as exc:
            logger.error("DeveloperAgentAdapter: execution error: %s", exc)
            return AgentResult(
                agent_id=self.agent_id,
                mode=self.mode,
                confidence=0.0,
                errors=[f"DeveloperAgent error: {exc}"],
            )
