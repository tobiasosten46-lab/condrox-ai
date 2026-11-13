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
Stage 02b: Intent Enrichment.

Enriches the classified intent with sub-intents from IntentExpander and
detects semantic interrupts. This stage runs between Stage02 (Intent) and
Stage03 (Mode Selector) to provide richer intent information for mode selection.

Mathematical Foundation:
- Intent expansion is O(1) via lookup table.
- Semantic interrupt detection is O(p * n) for p patterns, n text length.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage02bIntentEnrichment:
    """Intent enrichment stage — expands intent and detects interrupts.

    Runs between Stage02 (Intent) and Stage03 (Mode Selector).
    Uses IntentExpander to predict sub-intents and SemanticInterrupt
    to detect conversation interrupts (topic changes, urgency, corrections).
    """

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Enrich intent with sub-intents and detect interrupts.

        Args:
            controller: BrainController with CI components.
            state: Current BrainState after Stage02 intent classification.

        Returns:
            New BrainState with enriched intent metadata.

        Complexity: T(n) ∈ O(1 + p * n) for intent expansion + interrupt detection.
        """
        # 1. Expand intent using IntentExpander
        try:
            expanded = controller.intent_expander.expand(
                state.processed_input, state.intent
            )
            # Store sub-intents in context for downstream stages
            context = dict(state.context)
            context["sub_intents"] = list(expanded.sub_intents)
            context["intent_expansion_depth"] = expanded.expansion_depth
        except Exception:
            context = state.context

        # 2. Detect semantic interrupts
        try:
            interrupt = controller.semantic_interrupt.detect(state.processed_input)
            if interrupt.is_interrupt:
                context = dict(context)
                context["semantic_interrupt"] = {
                    "type": interrupt.interrupt_type,
                    "confidence": interrupt.confidence,
                }
        except Exception:
            pass

        return state.with_field(context=context)
