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
Pipeline orchestrator for the unified Brain.

Coordinates the 14-stage cognitive pipeline. Each stage is a callable object
that receives the current BrainState and the BrainController, and returns a
new immutable BrainState.

Mathematical Foundation:
- Pipeline is a fold/reduce over stages: S_n = f_n(...f_2(f_1(S_0))...).
- Complexity of a full run is the sum of stage complexities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

StageType = Callable[["BrainController", BrainState], BrainState]


class PipelineOrchestrator:
    """
    Real pipeline orchestrator with ordered stage execution.

    Attributes:
        _stages: Ordered list of pipeline stage callables.

    Complexity:
        add_stage: O(1)
        run: O(Σ s_i) where s_i is stage i complexity.
    """

    __slots__ = ("_stages",)

    def __init__(self) -> None:
        """Initialize an empty pipeline."""
        self._stages: List[StageType] = []

    def add_stage(self, stage: StageType) -> None:
        """Append a stage to the pipeline.

        Args:
            stage: Callable (controller, state) -> state.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._stages.append(stage)

    def run(self, controller: "BrainController", initial: BrainState) -> BrainState:
        """Execute all stages in order with error handling.

        If a stage raises an exception, the error is recorded in the state
        and the pipeline continues with the last valid state. This prevents
        a single stage failure from crashing the entire pipeline.

        At key transition points, state snapshots are published to the AI-OS
        data flow channels (input → intent → context → reasoning → output)
        when the kernel is active.

        Args:
            controller: BrainController with subsystems.
            initial: Initial BrainState.

        Returns:
            Final BrainState after all stages (with any errors recorded).

        Complexity: T(n) ∈ O(Σ s_i), S(n) ∈ O(max stage memory).
        """
        state = initial

        # Resolve data flow channels if AI-OS kernel is active
        df = None
        if controller.aios_kernel is not None and controller.aios_kernel.is_booted:
            df = controller.aios_kernel.data_flow

        # Publish to input channel at pipeline entry
        if df is not None:
            df.put_nowait("input", {
                "request_id": state.request_id,
                "raw_input": state.raw_input,
                "timestamp": state.metadata.get("started_at", 0.0),
            })

        # Stage index → channel mapping for transition snapshots
        # After index 3 (Stage03ModeSelector) → intent
        # After index 8 (Stage07MemoryFetch) → context
        # After index 10 (Stage08bAgentExecution) → reasoning
        # After last stage → output
        _INTENT_IDX = 3
        _CONTEXT_IDX = 8
        _REASONING_IDX = 10

        for i, stage in enumerate(self._stages):
            try:
                state = stage(controller, state)
            except Exception as exc:
                stage_name = type(stage).__name__
                state = state.add_error(f"{stage_name}: {exc}")

            # Publish state snapshots at transition points
            if df is not None:
                if i == _INTENT_IDX:
                    df.put_nowait("intent", {
                        "request_id": state.request_id,
                        "intent": state.intent,
                        "mode": state.mode,
                        "domain": state.domain,
                        "confidence": state.intent_confidence,
                    })
                elif i == _CONTEXT_IDX:
                    df.put_nowait("context", {
                        "request_id": state.request_id,
                        "intent": state.intent,
                        "mode": state.mode,
                        "domain": state.domain,
                        "context_keys": list(state.metadata.keys()),
                    })
                elif i == _REASONING_IDX:
                    df.put_nowait("reasoning", {
                        "request_id": state.request_id,
                        "intent": state.intent,
                        "mode": state.mode,
                        "reasoning_available": bool(state.reasoning),
                    })

        # Publish to output channel at pipeline exit
        if df is not None:
            df.put_nowait("output", {
                "request_id": state.request_id,
                "final_output": state.final_output,
                "intent": state.intent,
                "mode": state.mode,
                "errors": state.errors,
            })

        return state

    def stage_count(self) -> int:
        """Return the number of stages.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._stages)
