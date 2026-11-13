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
Stage 08: Reasoning Core.

Runs the symbolic ReasoningEngine over facts derived from intent, mode,
domain, memory, and knowledge presence.

Mathematical Foundation:
- Forward chaining over r rules with a antecedents and f facts is
  O(r * a * f). Here a and f are bounded by small constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage08ReasoningCore:
    """Real reasoning stage."""

    __slots__ = ()

    def _build_facts(self, state: BrainState) -> dict[str, float]:
        """Derive facts from the current state.

        Args:
            state: Current BrainState.

        Returns:
            Mapping from fact string to confidence.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        facts: dict[str, float] = {}
        facts[f"intent={state.intent}"] = state.intent_confidence
        facts[f"mode={state.mode}"] = 1.0
        facts[f"domain={state.domain}"] = 1.0
        if state.retrieved_memories:
            facts["has_memory"] = min(1.0, len(state.retrieved_memories) / 5.0)
        if state.retrieved_knowledge:
            facts["has_knowledge"] = min(1.0, len(state.retrieved_knowledge) / 5.0)
        return facts

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run reasoning with claim ranking.

        Complexity: T(n) ∈ O(r * a * f + c * n), S(n) ∈ O(r + c).
        """
        facts = self._build_facts(state)
        consequent, confidence, evidence = controller.reasoning_engine.reason(facts, top_k=5)
        reasoning = f"[Reasoning] {consequent} (confidence={confidence:.3f})"

        # Rank knowledge claims using ClaimRankerV2
        try:
            from core.conversational.claim_ranker_v2 import Claim
            cm = state.context_manager if state.context_manager is not None else controller.context_manager
            knowledge_texts = cm.get("knowledge_texts", [])
            if knowledge_texts:
                claims = [
                    Claim(
                        text=text[:200],
                        source="knowledge",
                        evidence_score=0.7,
                        recency_score=0.5,
                        relevance_score=0.6,
                        confidence_score=confidence,
                    )
                    for text in knowledge_texts[:5]
                ]
                ranked = controller.claim_ranker_v2.rank(claims)
                # Add top-ranked claim to evidence
                if ranked:
                    top_claim = ranked[0]
                    evidence = list(evidence) + [f"top_claim_score={top_claim.composite_score:.3f}"]
        except Exception:
            pass

        return state.with_field(
            reasoning=reasoning,
            reasoning_evidence=evidence,
        )
