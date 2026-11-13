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
Stage 09c: User Pattern Learning.

Analyzes each completed interaction to extract and persist behavioral,
emotional, and conversational patterns. This is the "learning from every
chat" stage — the AI observes how the user communicates, what emotions
they express, what topics they prefer, and stores these patterns for
future personalization.

This stage runs AFTER synthesis (Stage 09) so it has access to the final
response, detected emotion, and full pipeline state.

Complexity: T(n) ∈ O(n) for input analysis, S(n) ∈ O(p) for p patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState
from core.cognition.user_pattern_learner import UserPatternLearner

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage09cPatternLearning:
    """Learn user patterns from each interaction.

    Extracts emotional, stylistic, topical, and behavioral patterns from
    the current interaction and stores them as tagged memories.

    Complexity: T(n) ∈ O(n) for input length n, S(n) ∈ O(p) for p patterns.
    """

    __slots__ = ("_learner",)

    def __init__(self) -> None:
        """Initialize with a shared pattern learner instance.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._learner = UserPatternLearner()

    def __call__(
        self, controller: "BrainController", state: BrainState
    ) -> BrainState:
        """Analyze interaction and store learned patterns.

        Args:
            controller: BrainController with memory_manager.
            state: Current BrainState after synthesis.

        Returns:
            Unchanged BrainState (this stage is side-effect only).

        Complexity: T(n) ∈ O(n) for input analysis, S(n) ∈ O(p).
        """
        # Skip pattern learning if store_memory is disabled
        # (PreReleaseAgent public demo mode)
        store_memory = state.metadata.get("store_memory", True)
        if not store_memory:
            return state

        # Skip if no meaningful input
        if not state.processed_input or len(state.processed_input.strip()) < 3:
            return state

        try:
            emotion_conf = state.metadata.get("emotion_confidence", 0.0)
            self._learner.learn(
                controller=controller,
                user_input=state.processed_input,
                emotion=state.emotion,
                emotion_confidence=emotion_conf,
                intent=state.intent,
                mode=state.mode,
                domain=state.domain,
                final_output=state.final_output or state.synthesized_output or "",
                store_memory=store_memory,
            )
        except Exception:
            pass  # Pattern learning failure is non-critical

        return state
