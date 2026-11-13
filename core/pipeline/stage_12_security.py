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
Stage 12: Security.

Runs the SecurityEngine on the synthesized output. If the risk score is too
high, the output is replaced with a safe refusal message.

Mathematical Foundation:
- Security scan is O(p * n) for p blocked patterns and output length n.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage12Security:
    """Real security check stage."""

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run security check.

        Complexity: T(n) ∈ O(p * n), S(n) ∈ O(1).
        """
        report = controller.security_engine.check_output(state.synthesized_output)

        if not report.approved:
            refusal = (
                "This response was blocked by the security layer. "
                f"Violations: {'; '.join(report.violations)}."
            )
            return state.with_field(
                security_approved=False,
                synthesized_output=refusal,
                errors=[*state.errors, f"security_violation:{';'.join(report.violations)}"],
            )

        return state.with_field(security_approved=True)
