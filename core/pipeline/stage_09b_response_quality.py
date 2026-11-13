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
Stage 09b: Response Quality.

Runs quality checks on the synthesized output after Stage09 (Synthesis).
Uses ConsistencyEngine, SemanticRepair, and ContradictionDetector to
ensure response quality before personality and ethics stages.

Mathematical Foundation:
- Consistency check: O(c * n) for c consistency pairs, n output length.
- Semantic repair: O(n) for drift detection and trimming.
- Contradiction detection: O(s * n) for s strategies, n output length.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage09bResponseQuality:
    """Response quality stage — consistency, repair, contradictions.

    Runs after Stage09 (Synthesis) and before Stage10 (Personality).
    Checks response for consistency, semantic drift, and contradictions.
    """

    __slots__ = ()

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run quality checks on synthesized output.

        Args:
            controller: BrainController with CI components.
            state: Current BrainState after Stage09 synthesis.

        Returns:
            New BrainState with quality-checked output.

        Complexity: T(n) ∈ O(c * n + n + s * n) for c pairs, s strategies.
        """
        output = state.synthesized_output
        if not output:
            return state

        errors = list(state.errors)

        # 1. Consistency check — detect internal contradictions
        try:
            report = controller.consistency_engine.check(output)
            if not report.consistent:
                for contra in report.contradictions:
                    errors.append(f"ConsistencyIssue: {contra}")
        except Exception:
            pass

        # 2. Semantic repair — skip for conversational/emotional/short/school/technical responses
        # v3.4.0 PATCH: Also skip for code/analyze/architect/explain modes. The semantic
        # repair uses sentence-level word overlap with the query to detect "drift", but
        # structured technical responses (code analysis, architecture plans, step-by-step
        # explanations) naturally contain sentences that don't echo query words but are
        # still on-topic (e.g., "Therefore, the algorithm is correct because..."). This
        # was causing Step 4/5 of code analysis and Phases 2-4 of architecture plans to
        # be stripped, truncating responses from ~1200 chars to ~580 chars and dropping
        # StepByStepStructure/ExplanationQuality scores.
        is_emotional = state.intent in ("emotional", "unknown") and state.emotion != "neutral"
        is_school_answer = (state.context or {}).get("school_answer", False)
        is_technical = state.mode in ("code", "analyze", "architect", "explain")
        if (state.intent not in ("greet", "clarify", "emotional")
                and not is_emotional
                and not is_school_answer
                and not is_technical
                and len(output) > 100):
            try:
                repair_result = controller.semantic_repair.repair(
                    state.processed_input, output
                )
                output = repair_result.repaired_text
            except Exception:
                pass

        # 3. Contradiction detection — check output against known facts
        try:
            contra_report = controller.contradiction_detector.check(output)
            if contra_report.has_contradictions:
                errors.append("ContradictionDetected: output contradicts known facts")
        except Exception:
            pass

        return state.with_field(
            synthesized_output=output,
            errors=errors,
        )
