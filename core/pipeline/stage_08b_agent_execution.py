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
Stage 08.5: Agent Execution.

Runs the mode-specific agent between Stage08 (Reasoning) and Stage09 (Synthesis).
The agent enriches BrainState with mode-specific reasoning, evidence, and a
draft synthesized output that Stage09 can incorporate.

Mathematical Foundation:
- Agent selection is O(1) via registry lookup by mode string.
- Agent execution is a pure function: result = f(context, controller).
- State merge is O(1) for fixed-size fields.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage08bAgentExecution:
    """Agent execution stage — runs mode-specific agent and merges result.

    Inserts between Stage08 (Reasoning) and Stage09 (Synthesis).
    The agent provides mode-specific reasoning and a draft output that
    enriches the state before synthesis.
    """

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run the mode-specific agent and merge result into state.

        Uses SearchAgentGuard to validate input before agent execution.

        Args:
            controller: BrainController with agent registry.
            state: Current BrainState after Stage08 reasoning.

        Returns:
            New BrainState with agent reasoning and draft output merged.

        Complexity: T(n) ∈ O(p + e + f) for agent lifecycle phases.
        """
        # Validate input with SearchAgentGuard before agent execution
        try:
            valid, reason = controller.search_agent_guard.validate(
                state.processed_input
            )
            if not valid:
                return state.add_error(
                    f"AgentExecution: input rejected by guard: {reason}"
                )
        except Exception:
            pass  # Guard is optional — continue if it fails

        try:
            result = controller.run_agent(state.mode, state)
            # Record agent selection in arbiter for tracking
            try:
                controller.agent_arbiter.select(
                    task_id=state.request_id,
                    intent=state.intent,
                    mode=state.mode,
                )
            except Exception:
                pass
        except Exception as exc:
            return state.add_error(f"AgentExecution: {exc}")

        # Merge agent result into state
        # Use agent's reasoning if it's richer than existing
        reasoning = result.reasoning if result.reasoning else state.reasoning

        # Merge evidence (agent evidence + existing)
        merged_evidence = list(state.reasoning_evidence)
        if result.evidence:
            merged_evidence.extend(result.evidence)

        # Use agent's draft output as synthesized_output seed for Stage09
        synthesized = result.synthesized_output if result.synthesized_output else ""

        # Store agent metadata in context for Stage09
        if state.context_manager is not None:
            state.context_manager.set("agent_id", result.agent_id)
            state.context_manager.set("agent_confidence", result.confidence)
            state.context_manager.set("agent_mode", result.mode)
            agent_meta = state.context_manager.current()
        else:
            agent_meta = dict(state.context)
            agent_meta["agent_id"] = result.agent_id
            agent_meta["agent_confidence"] = result.confidence
            agent_meta["agent_mode"] = result.mode

        # Record any agent errors
        new_state = state.with_field(
            reasoning=reasoning,
            reasoning_evidence=merged_evidence,
            synthesized_output=synthesized,
            context=agent_meta,
        )

        if result.errors:
            for err in result.errors:
                new_state = new_state.add_error(f"Agent[{result.agent_id}]: {err}")

        return new_state
