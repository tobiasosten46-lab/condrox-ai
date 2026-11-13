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
Stage 13: Output Builder.

Formats the final output, adds metadata, and ensures the output is non-empty.

Mathematical Foundation:
- Formatting is a linear pass over the synthesized output: O(n).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage13OutputBuilder:
    """Real output builder stage."""

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Build final output.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        text = state.synthesized_output.strip()
        if not text:
            text = "[No output generated.]"

        metadata = {
            **state.metadata,
            "finished_at": time.time(),
            "pipeline_stage": "output_builder",
        }

        return state.with_field(final_output=text, metadata=metadata)
