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
Stage 14: Delivery.

Finalizes the response for delivery, applies a final metadata stamp, and
performs any post-processing such as updating meta-learning based on the
successful outcome.

Mathematical Foundation:
- Finalization is O(1) plus any learning update O(1).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage14Delivery:
    """Real delivery stage."""

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Finalize response with latency compensation.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        delivered_at = time.time()
        started_at = state.metadata.get("started_at", delivered_at)
        actual_latency_ms = (delivered_at - started_at) * 1000.0

        # Record latency for compensation prediction
        try:
            controller.latency_compensation.record("pipeline", actual_latency_ms)
        except Exception:
            pass

        metadata = {
            **state.metadata,
            "delivered_at": delivered_at,
            "latency_ms": round(actual_latency_ms, 3),
            "mode": state.mode,
            "domain": state.domain,
            "intent": state.intent,
            "intent_confidence": round(state.intent_confidence, 4),
        }

        # Update meta-learning reward for the active mode
        reward = 1.0 if not state.errors else 0.7
        if controller.config.enable_learning:
            controller.meta_learning_engine.record_outcome(state.mode, reward)

        return state.with_field(metadata=metadata)
