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
Stage 09d: Metacognitive Reflection.

The AI reflects on its own response, judges its quality across cognitive
dimensions, and revises if the self-assessment is poor.

This stage runs AFTER Stage09b (Response Quality) and AFTER Stage09c
(Pattern Learning), but BEFORE Stage10 (Personality). It is the AI's
"inner critic" — the self-intelligence layer.

Pipeline flow:
  Stage09 (Synthesis) → Stage09b (Quality) → Stage09c (Pattern Learning)
  → Stage09d (Metacognition) → Stage10 (Personality)

Complexity: T(n) ∈ O(n + k) for response length n, knowledge items k.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState
from core.cognition.metacognition_engine import CognitiveAssessment, MetacognitionEngine

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage09dMetacognition:
    """Metacognitive reflection and self-revision stage.

    Evaluates the AI's own response and revises it if the self-assessment
    indicates poor quality (generic templates, missing knowledge, low
    relevance, lack of self-awareness for identity queries).

    Complexity: T(n) ∈ O(n + k), S(n) ∈ O(n).
    """

    __slots__ = ("_engine",)

    def __init__(self) -> None:
        """Initialize with a shared metacognition engine.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._engine = MetacognitionEngine()

    def __call__(
        self, controller: "BrainController", state: BrainState
    ) -> BrainState:
        """Reflect on and potentially revise the synthesized output.

        Args:
            controller: BrainController.
            state: Current BrainState with synthesized_output.

        Returns:
            BrainState with potentially revised output and assessment metadata.

        Complexity: T(n) ∈ O(n + k), S(n) ∈ O(n).
        """
        output = state.synthesized_output
        if not output or len(output.strip()) < 10:
            return state

        # Skip metacognitive revision for conversational/emotional intents —
        # these responses are intentionally empathetic and don't need knowledge grounding
        if state.intent in ("greet", "clarify", "emotional", "grateful", "identity"):
            return state

        # Skip for short conversational responses (yes, no, ok, thanks, etc.)
        # — these are contextual follow-ups, not knowledge queries
        input_lower = state.processed_input.lower().strip().rstrip(".!?")
        _short_conv = frozenset({
            "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please do",
            "go ahead", "please", "i agree", "correct", "right", "exactly",
            "that's right", "thats right", "indeed", "absolutely", "definitely",
            "no", "nope", "nah", "not really", "i disagree", "wrong",
            "that's wrong", "thats wrong", "incorrect", "not at all",
            "alright", "fine", "i see", "got it", "understood",
            "makes sense", "i understand", "thanks", "thank you", "thx",
        })
        if len(input_lower) <= 30 and input_lower in _short_conv:
            return state

        cm = state.context_manager if state.context_manager is not None else controller.context_manager
        knowledge_texts = cm.get("knowledge_texts", [])
        memory_texts = cm.get("memory_texts", [])

        # ─── 1. Self-evaluation ────────────────────────────────────────
        assessment = self._engine.evaluate(state, knowledge_texts, memory_texts)

        # Store assessment in context for observability
        cm.set("metacognition_assessment", {
            "relevance": assessment.relevance,
            "grounding": assessment.grounding,
            "coherence": assessment.coherence,
            "completeness": assessment.completeness,
            "specificity": assessment.specificity,
            "self_awareness": assessment.self_awareness,
            "overall": assessment.overall,
            "issues": list(assessment.issues),
            "should_revise": assessment.should_revise,
        })

        # ─── 2. Self-revision if needed ────────────────────────────────
        if assessment.should_revise:
            revised = self._engine.revise(
                state, assessment, knowledge_texts, memory_texts
            )
            if revised and revised != output:
                output = revised

        return state.with_field(
            synthesized_output=output,
            context_manager=cm,
            context=cm.current(),
        )
