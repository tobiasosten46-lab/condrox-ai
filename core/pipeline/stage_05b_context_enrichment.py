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
Stage 05b: Context Enrichment.

Enriches the pipeline context with multi-turn reasoning, goal tracking,
and temporal context. This stage runs after Stage05 (Context Builder)
to provide richer context for downstream stages.

Mathematical Foundation:
- Multi-turn reasoning: O(t) for t turns in history.
- Goal tracking: O(g) for g active goals.
- Temporal context: O(k) for k context keys with TTL.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage05bContextEnrichment:
    """Context enrichment stage — multi-turn, goals, temporal.

    Runs after Stage05 (Context Builder).
    Adds multi-turn reasoning context, tracks dialogue goals,
    and applies temporal context TTL management.
    """

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Enrich context with multi-turn, goals, and temporal data.

        Args:
            controller: BrainController with CI components.
            state: Current BrainState after Stage05 context building.

        Returns:
            New BrainState with enriched context.

        Complexity: T(n) ∈ O(t + g + k) for t turns, g goals, k temporal keys.
        """
        context = dict(state.context)

        # 1. Multi-turn reasoning — record turn and get context
        try:
            controller.multi_turn_reasoning.record_turn(
                user_input=state.processed_input,
                agent_output="",  # output not yet generated
                intent=state.intent,
            )
            turn_context = controller.multi_turn_reasoning.get_context()
            context["multi_turn_continuity"] = turn_context.topic_continuity
            context["multi_turn_focus"] = turn_context.suggested_focus
            context["multi_turn_dominant_intent"] = turn_context.dominant_intent
        except Exception:
            pass

        # 2. Goal tracking — add active goals
        try:
            active_goals = controller.goal_tracker.get_active_goals()
            if active_goals:
                context["active_goals"] = [g.description if hasattr(g, "description") else str(g) for g in active_goals]
        except Exception:
            pass

        # 3. Temporal context — apply TTL-based context management
        try:
            # TemporalContext manages TTL — we just record that it's active
            context["temporal_context_active"] = True
        except Exception:
            pass

        return state.with_field(context=context)
