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

"""Reasoning evidence evaluator.

Evaluates Condrox AI reasoning outputs on step-by-step structure, logical
coherence, technical correctness signals, planning, code analysis, and
explanation quality — without exposing internal model weights or reasoning
chains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True, slots=True)
class ReasoningMetric:
    """A single reasoning quality metric."""

    name: str
    score: float  # 0.0–100.0
    status: str  # PASS, WARN, FAIL
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReasoningEvidence:
    """Evidence captured for one reasoning task."""

    input: str
    output: str
    intent: str
    mode: str
    domain: str
    metrics: List[ReasoningMetric] = field(default_factory=list)
    deviations: List[str] = field(default_factory=list)


class ReasoningEvidenceEvaluator:
    """Evaluate reasoning quality and produce evidence.

    Complexity: T(evaluate) ∈ O(n) where n = output length.
    """

    __slots__ = ()

    _STEP_INDICATORS: Tuple[str, ...] = (
        "step", "first", "second", "third", "next", "then", "finally",
        "1.", "2.", "3.", "4.", "5.",
        "one", "two", "three", "four", "five",
        "start by", "begin by", "after", "before", "conclude",
    )

    _LOGIC_INDICATORS: Tuple[str, ...] = (
        "because", "therefore", "thus", "since", "if", "then", "else",
        "given", "assume", "follows", "implies", "so", "hence",
    )

    _EXPLANATION_INDICATORS: Tuple[str, ...] = (
        "example", "for instance", "in other words", "simply", "means",
        "defined as", "refers to", "essentially", "basically",
    )

    _PLANNING_INDICATORS: Tuple[str, ...] = (
        "plan", "approach", "strategy", "implement", "design", "structure",
        "phase", "stage", "iteration", "milestone",
    )

    _CODE_ANALYSIS_INDICATORS: Tuple[str, ...] = (
        "function", "class", "method", "variable", "return", "parameter",
        "complexity", "O(", "time", "space", "algorithm", "data structure",
        "edge case", "handle", "error", "exception",
    )

    def evaluate(
        self,
        user_input: str,
        output: str,
        state: Any,
    ) -> ReasoningEvidence:
        """Evaluate a single reasoning output."""
        metrics: List[ReasoningMetric] = []
        deviations: List[str] = []

        step_score = self._step_by_step_score(output)
        metrics.append(step_score)
        if step_score.status != "PASS":
            deviations.append("Response lacks clear step-by-step structure")

        logic_score = self._logic_score(output)
        metrics.append(logic_score)
        if logic_score.status != "PASS":
            deviations.append("Logical connectors / reasoning signals missing")

        explanation_score = self._explanation_score(output)
        metrics.append(explanation_score)

        planning_score = self._planning_score(user_input, output)
        metrics.append(planning_score)

        code_analysis_score = self._code_analysis_score(user_input, output)
        metrics.append(code_analysis_score)

        correctness_score = self._correctness_signal_score(user_input, output)
        metrics.append(correctness_score)
        if correctness_score.status == "FAIL":
            deviations.append("No correctness signal detected in reasoning")

        return ReasoningEvidence(
            input=user_input,
            output=output,
            intent=getattr(state, "intent", ""),
            mode=getattr(state, "mode", ""),
            domain=getattr(state, "domain", ""),
            metrics=metrics,
            deviations=deviations,
        )

    def summarize(
        self, evidence: List[ReasoningEvidence]
    ) -> Dict[str, Any]:
        """Return a summary dictionary for a set of reasoning tasks."""
        if not evidence:
            return {"score": 0.0, "status": "FAIL", "tasks": 0}

        all_metrics = [m for ev in evidence for m in ev.metrics]
        score = sum(m.score for m in all_metrics) / len(all_metrics) if all_metrics else 0.0
        fail_count = sum(1 for m in all_metrics if m.status == "FAIL")
        warn_count = sum(1 for m in all_metrics if m.status == "WARN")

        if fail_count:
            status = "FAIL"
        elif warn_count:
            status = "WARN"
        else:
            status = "PASS"

        by_name: Dict[str, List[float]] = {}
        for m in all_metrics:
            by_name.setdefault(m.name, []).append(m.score)

        return {
            "score": round(score, 2),
            "status": status,
            "tasks": len(evidence),
            "metric_averages": {name: round(sum(scores) / len(scores), 2) for name, scores in by_name.items()},
            "deviation_count": sum(len(ev.deviations) for ev in evidence),
        }

    def _count_indicators(self, output: str, indicators: Tuple[str, ...]) -> int:
        lower = output.lower()
        return sum(1 for ind in indicators if ind in lower)

    def _step_by_step_score(self, output: str) -> ReasoningMetric:
        count = self._count_indicators(output, self._STEP_INDICATORS)
        score = min(100.0, 40.0 + count * 15.0)
        status = "PASS" if count >= 2 else "WARN"
        return ReasoningMetric(
            name="StepByStepStructure",
            score=score,
            status=status,
            details={"indicators_found": count},
        )

    def _logic_score(self, output: str) -> ReasoningMetric:
        count = self._count_indicators(output, self._LOGIC_INDICATORS)
        score = min(100.0, 40.0 + count * 12.0)
        status = "PASS" if count >= 1 else "WARN"
        return ReasoningMetric(
            name="LogicalCoherence",
            score=score,
            status=status,
            details={"indicators_found": count},
        )

    def _explanation_score(self, output: str) -> ReasoningMetric:
        count = self._count_indicators(output, self._EXPLANATION_INDICATORS)
        score = min(100.0, 50.0 + count * 12.0)
        status = "PASS" if score >= 50.0 else "WARN"
        return ReasoningMetric(
            name="ExplanationQuality",
            score=score,
            status=status,
            details={"indicators_found": count},
        )

    def _planning_score(self, user_input: str, output: str) -> ReasoningMetric:
        planning_intent = any(kw in user_input.lower() for kw in ("plan", "design", "strategy", "approach", "steps"))
        if not planning_intent:
            return ReasoningMetric(
                name="PlanningQuality",
                score=100.0,
                status="PASS",
                details={"not_required": True},
            )
        count = self._count_indicators(output, self._PLANNING_INDICATORS)
        score = min(100.0, 40.0 + count * 15.0)
        status = "PASS" if count >= 2 else "WARN"
        return ReasoningMetric(
            name="PlanningQuality",
            score=score,
            status=status,
            details={"indicators_found": count},
        )

    def _code_analysis_score(self, user_input: str, output: str) -> ReasoningMetric:
        code_intent = any(kw in user_input.lower() for kw in ("code", "function", "algorithm", "analyze", "review"))
        if not code_intent:
            return ReasoningMetric(
                name="CodeAnalysis",
                score=100.0,
                status="PASS",
                details={"not_required": True},
            )
        count = self._count_indicators(output, self._CODE_ANALYSIS_INDICATORS)
        score = min(100.0, 40.0 + count * 12.0)
        status = "PASS" if count >= 2 else "WARN"
        return ReasoningMetric(
            name="CodeAnalysis",
            score=score,
            status=status,
            details={"indicators_found": count},
        )

    def _correctness_signal_score(self, user_input: str, output: str) -> ReasoningMetric:
        """Surface-level correctness signals (not formal verification).

        v3.4.0 PATCH: Expanded signal detection with more correctness markers
        and a hard penalty for irrelevant responses. The previous version only
        detected 3-4 signals (complexity notation, numbers, length, answer
        patterns) which gave 50% even for wrong answers. The new version adds:
        - Code correctness markers (def, return, if, else, for, while, try)
        - Validation language ("verified", "correct", "tested", "edge case")
        - Step-by-step correctness ("first", "then", "finally", "therefore")
        - Hard penalty (-2 signals) when output is clearly irrelevant to input
          (e.g., "Past Simple" when asked about quicksort complexity)
        """
        lower = output.lower()
        input_lower = user_input.lower()
        signals = 0

        # Mentions of complexity notation
        if "O(" in output:
            signals += 1

        # Contains numeric or structured claims
        if re.search(r"\d+", output):
            signals += 1

        # Non-empty, non-trivial response
        if len(output.split()) >= 20:
            signals += 1

        # For math/logic questions, includes a final answer pattern
        if any(kw in input_lower for kw in ("what is", "solve", "calculate", "result", "equals")):
            if re.search(r"(answer is|result is|=\s*\d+|therefore,?\s*\d+)", lower):
                signals += 1

        # v3.4.0 PATCH: Code correctness markers
        _code_markers = ("def ", "return ", "if ", "else:", "elif ", "for ", "while ",
                         "try:", "except", "import ", "class ", "lambda ",
                         "async ", "await ", "yield ", "raise ")
        code_hits = sum(1 for m in _code_markers if m in lower)
        if code_hits >= 2:
            signals += 1
        if code_hits >= 4:
            signals += 1  # Bonus for substantial code content

        # v3.4.0 PATCH: Validation language (shows correctness awareness)
        _validation_markers = (
            "verified", "correct", "tested", "edge case", "edge cases",
            "boundary", "off-by-one", "null check", "error handling",
            "input validation", "output validation", "sanitized",
            "type check", "type hint", "type safety",
        )
        if any(m in lower for m in _validation_markers):
            signals += 1

        # v3.4.0 PATCH: Step-by-step correctness markers
        _step_markers = ("first", "then", "finally", "therefore", "next",
                         "step 1", "step 2", "step 3",
                         "phase 1", "phase 2", "phase 3",
                         "1.", "2.", "3.", "4.")
        step_hits = sum(1 for m in _step_markers if m in lower)
        if step_hits >= 2:
            signals += 1

        # v3.4.0 PATCH: Hard penalty for irrelevant responses
        # If the input is clearly technical but the output contains
        # language/grammar terms, penalize heavily.
        _technical_input_signals = (
            "quicksort", "mergesort", "bubble", "complexity", "algorithm",
            "recursion", "recursive", "function", "python", "javascript",
            "sort", "search", "tree", "graph", "stack", "queue",
            "api", "rest", "endpoint", "implement", "debug", "code",
            "def ", "class ", "return ",
        )
        _irrelevant_output_signals = (
            "past simple", "present simple", "irregular verb", "regular verb",
            "norwegian", "grammar", "verb conjugation", "noun", "adjective",
            "photosynthesis", "human rights", "cells", "ecosystem",
        )
        is_technical_input = any(s in input_lower for s in _technical_input_signals)
        has_irrelevant_output = any(s in lower for s in _irrelevant_output_signals)
        if is_technical_input and has_irrelevant_output:
            signals -= 2  # Hard penalty for domain drift

        # v3.4.0 PATCH: Bonus for direct relevance (input keywords appear in output)
        input_words = set(input_lower.split()) - {"the", "a", "an", "is", "are",
                                                   "what", "how", "why", "of", "to",
                                                   "in", "on", "for", "and", "or"}
        output_words = set(lower.split())
        if input_words and len(input_words & output_words) >= 2:
            signals += 1

        score = min(100.0, max(0.0, 30.0 + signals * 15.0))
        status = "PASS" if signals >= 2 else "WARN"
        return ReasoningMetric(
            name="CorrectnessSignal",
            score=score,
            status=status,
            details={"signals": signals},
        )
