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
Stage 11: Ethics.

Runs the EthicsEngine on the synthesized output. If blocked topics are
flagged, the stage records an error and returns a sanitized refusal text.

Mathematical Foundation:
- Topic match score is the maximum Jaccard overlap across t topics:
  O(t * n) for text length n.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage11Ethics:
    """Real ethics check stage."""

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run ethics check.

        Complexity: T(n) ∈ O(t * n), S(n) ∈ O(1).
        """
        report = controller.ethics_engine.check(state.synthesized_output)

        if not report.approved:
            refusal = (
                "I cannot produce this response because it triggered the "
                f"following ethics flags: {', '.join(report.flagged_topics)}."
            )
            return state.with_field(
                ethical_approved=False,
                synthesized_output=refusal,
                errors=[*state.errors, f"ethics_violation:{','.join(report.flagged_topics)}"],
            )

        return state.with_field(ethical_approved=True)
