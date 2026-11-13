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
Stage 10: Personality.

Applies the active personality profile to the synthesized output.

Mathematical Foundation:
- Styling is a linear text transformation: O(l) for l words.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage10Personality:
    """Real personality application stage."""

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Apply personality styling.

        School curriculum answers preserve their verified structure (numbered
        steps, examples, context) so quality gates can validate them.

        Complexity: T(n) in O(l) for l output words.
        """
        if (state.context or {}).get("school_answer"):
            return state.with_field(
                synthesized_output=state.synthesized_output,
                personality_applied=True,
            )
        styled = controller.personality_engine.apply(state.synthesized_output, state.emotion)
        return state.with_field(
            synthesized_output=styled,
            personality_applied=True,
        )
