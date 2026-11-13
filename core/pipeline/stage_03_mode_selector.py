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
Stage 03: Mode Selector.

Maps the classified intent to a processing mode. If intent confidence is
too low, defaults to a safe clarification mode.

Mathematical Foundation:
- Mode mapping is a deterministic lookup with O(1) complexity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage03ModeSelector:
    """Real mode selection stage."""

    __slots__ = ("_mapping",)

    def __init__(self) -> None:
        """Initialize intent-to-mode mapping."""
        self._mapping = {
            "code": "code",
            "analyze": "analyze",
            "explain": "explain",
            "question": "explain",
            "identity": "explain",
            "command": "code",
            "greet": "personal",
            "architect": "architect",
            "clarify": "clarify",
            "emotional": "personal",
            "personal": "personal",
            "developer": "developer",
            "unknown": "default",
        }

    def select(self, intent: str, confidence: float) -> str:
        """Select mode from intent and confidence.

        Args:
            intent: Classified intent.
            confidence: Intent confidence.

        Returns:
            Selected mode.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        # Only fall back to clarify when confidence is very low AND
        # the intent is unknown. A recognized intent with moderate
        # confidence should still use its mapped mode.
        if confidence < 0.25 and intent == "unknown":
            return "clarify"
        return self._mapping.get(intent, "default")

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run mode selection with StrategyManager enrichment.

        Uses intent-based mapping first, then enriches with StrategyManager
        if it selects a different (higher-scoring) strategy.

        After selecting a mode, verifies that an agent is actually registered
        for that mode. If no agent is registered (e.g., task agents were
        unregistered in PreReleaseAgent), falls back to 'default'.

        Complexity: T(n) ∈ O(1 + s * f), S(n) ∈ O(1).
        """
        mode = self.select(state.intent, state.intent_confidence)

        # Enrich with StrategyManager — may override mode if strategy scores higher
        try:
            strategy_context = {
                f"intent={state.intent}": state.intent_confidence,
                "mode": mode,
                "domain": state.domain,
            }
            # Incorporate meta-learning weights into strategy context
            try:
                weights = controller.meta_learning_engine.get_weights()
                for strat_name, weight in weights.items():
                    strategy_context[f"learned_{strat_name}"] = weight
            except Exception:
                pass

            strategy = controller.strategy_manager.select(strategy_context)
            if strategy and strategy.name in {"code", "analyze", "explain", "general", "personal"}:
                # Only override if intent confidence is very low and strategy differs
                if strategy.name != mode and state.intent_confidence < 0.4 and mode == "default":
                    mode = strategy.name
        except Exception:
            pass

        # Verify that an agent is actually registered for the selected mode.
        # If task agents were unregistered (e.g., PreReleaseAgent), fall back
        # to 'default' so the DefaultAgent handles the response.
        if mode != "default":
            try:
                agent = controller.agent_registry.get(mode)
                if agent is None:
                    mode = "default"
            except Exception:
                mode = "default"

        return state.with_field(mode=mode)
