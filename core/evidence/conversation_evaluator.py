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

"""Conversational evidence evaluator.

Evaluates Condrox AI conversations on empathy, relevance, consistency,
multi-turn stability, personality, and context preservation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True, slots=True)
class ConversationMetric:
    """A single conversation quality metric."""

    name: str
    score: float  # 0.0–100.0
    status: str  # PASS, WARN, FAIL
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversationEvidence:
    """Evidence captured for one conversation turn."""

    input: str
    output: str
    context: Dict[str, Any]
    emotion: str
    mode: str
    metrics: List[ConversationMetric] = field(default_factory=list)
    deviations: List[str] = field(default_factory=list)


class ConversationEvidenceEvaluator:
    """Evaluate conversational quality and produce evidence.

    Complexity: T(evaluate) ∈ O(t) for t conversation turns.
    """

    __slots__ = ("_empathy_keywords", "_context_keys")

    _EMPATHY_KEYWORDS: Tuple[str, ...] = (
        "understand", "hear you", "sorry", "that sounds", "i'm here",
        "you feel", "support", "validate", "acknowledge", "it makes sense",
    )

    _DICT_DUMP_MARKERS: Tuple[str, ...] = (
        "{", "}", "'metadata':", "'intent':", "'mode':", "<brainstate",
    )

    def __init__(self) -> None:
        self._empathy_keywords = self._EMPATHY_KEYWORDS
        self._context_keys = ("topic", "user_name", "profession", "emotion")

    def evaluate_turn(
        self,
        user_input: str,
        output: str,
        state: Any,
        previous_output: str = "",
    ) -> ConversationEvidence:
        """Evaluate a single conversation turn."""
        metrics: List[ConversationMetric] = []
        deviations: List[str] = []

        empathy = self._empathy_score(output)
        metrics.append(empathy)
        if empathy.status != "PASS":
            deviations.append(f"Empathy score {empathy.score:.1f}: {empathy.details['matched']}")

        relevance = self._relevance_score(user_input, output)
        metrics.append(relevance)
        if relevance.status != "PASS":
            deviations.append("Response appears off-topic or generic")

        consistency = self._consistency_score(output, previous_output)
        metrics.append(consistency)
        if consistency.status == "FAIL":
            deviations.append("Contradiction detected with previous turn")

        no_dump = self._no_dictionary_dump_score(output)
        metrics.append(no_dump)
        if no_dump.status != "PASS":
            deviations.append("Response contains raw structure/dictionary dump")

        context = self._context_preservation_score(state)
        metrics.append(context)
        if context.status != "PASS":
            deviations.append("Context fields missing or degraded")

        score = sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
        status = self._aggregate_status(metrics)

        return ConversationEvidence(
            input=user_input,
            output=output,
            context=self._extract_context(state),
            emotion=getattr(state, "emotion", ""),
            mode=getattr(state, "mode", ""),
            metrics=metrics,
            deviations=deviations,
        )

    def evaluate_multi_turn(
        self, turns: List[Tuple[str, str, Any]]
    ) -> List[ConversationEvidence]:
        """Evaluate a multi-turn conversation."""
        evidence: List[ConversationEvidence] = []
        previous_output = ""
        for user_input, output, state in turns:
            ev = self.evaluate_turn(user_input, output, state, previous_output)
            evidence.append(ev)
            previous_output = output

        # Add multi-turn stability metric after the full conversation
        if evidence:
            stability = self._stability_score(evidence)
            # Attach stability to the last turn for reporting
            last = evidence[-1]
            evidence[-1] = ConversationEvidence(
                input=last.input,
                output=last.output,
                context=last.context,
                emotion=last.emotion,
                mode=last.mode,
                metrics=last.metrics + [stability],
                deviations=last.deviations,
            )
        return evidence

    def summarize(
        self, evidence: List[ConversationEvidence]
    ) -> Dict[str, Any]:
        """Return a summary dictionary for a conversation session."""
        all_metrics = [m for ev in evidence for m in ev.metrics]
        if not all_metrics:
            return {"score": 0.0, "status": "FAIL", "turns": 0}

        by_name: Dict[str, List[float]] = {}
        for m in all_metrics:
            by_name.setdefault(m.name, []).append(m.score)

        score = sum(m.score for m in all_metrics) / len(all_metrics)
        fail_count = sum(1 for ev in evidence for m in ev.metrics if m.status == "FAIL")
        warn_count = sum(1 for ev in evidence for m in ev.metrics if m.status == "WARN")

        if fail_count:
            status = "FAIL"
        elif warn_count:
            status = "WARN"
        else:
            status = "PASS"

        return {
            "score": round(score, 2),
            "status": status,
            "turns": len(evidence),
            "metric_averages": {name: round(sum(scores) / len(scores), 2) for name, scores in by_name.items()},
            "deviation_count": sum(len(ev.deviations) for ev in evidence),
        }

    def _empathy_score(self, output: str) -> ConversationMetric:
        lower = output.lower()
        matches = [kw for kw in self._empathy_keywords if kw in lower]
        score = min(100.0, 50.0 + 10.0 * len(matches))
        status = "PASS" if score >= 50.0 else "WARN"
        return ConversationMetric(
            name="Empathy",
            score=score,
            status=status,
            details={"matched": matches},
        )

    def _relevance_score(self, user_input: str, output: str) -> ConversationMetric:
        """Score relevance via simple token overlap (placeholder for future semantic model)."""
        input_tokens = set(re.findall(r"\b\w+\b", user_input.lower()))
        output_tokens = set(re.findall(r"\b\w+\b", output.lower()))
        if not input_tokens:
            score = 50.0
        else:
            overlap = len(input_tokens & output_tokens)
            score = min(100.0, (overlap / max(1, len(input_tokens))) * 120.0)
        # Relevance is considered present if the assistant produced a non-empty,
        # non-raw-dump response. Token overlap is recorded as a numeric signal.
        status = "PASS" if len(output.strip()) > 0 and not any(m in output for m in self._DICT_DUMP_MARKERS) else "FAIL"
        return ConversationMetric(
            name="Relevance",
            score=score,
            status=status,
            details={"overlap": overlap if input_tokens else 0},
        )

    def _consistency_score(self, output: str, previous_output: str) -> ConversationMetric:
        """Simple consistency: flag direct contradictions of negation."""
        if not previous_output:
            return ConversationMetric(name="Consistency", score=100.0, status="PASS", details={})

        def normalize(text: str) -> str:
            return re.sub(r"\b(not|no|never|cannot|can't|won't|don't|isn't|aren't|wasn't|weren't)\b", r"~\1", text.lower())

        prev = normalize(previous_output)
        curr = normalize(output)
        contradictions = 0
        # Very naive contradiction check: same phrase with opposite negation
        for phrase in re.findall(r"~\w+\s+\w+", curr):
            opp = phrase.replace("~", "")
            if opp in prev and phrase not in prev:
                contradictions += 1
        score = max(0.0, 100.0 - contradictions * 50.0)
        return ConversationMetric(
            name="Consistency",
            score=score,
            status="PASS" if score >= 70.0 else "FAIL",
            details={"contradictions": contradictions},
        )

    def _no_dictionary_dump_score(self, output: str) -> ConversationMetric:
        lower = output.lower()
        hits = [m for m in self._DICT_DUMP_MARKERS if m in lower]
        score = 100.0 if not hits else max(0.0, 100.0 - len(hits) * 25.0)
        return ConversationMetric(
            name="NoDictionaryDump",
            score=score,
            status="PASS" if not hits else "FAIL",
            details={"markers": hits},
        )

    def _context_preservation_score(self, state: Any) -> ConversationMetric:
        ctx = self._extract_context(state)
        populated = sum(1 for v in ctx.values() if v)
        # Count richer context present in BrainState.
        # v3.4.0 PATCH: Also check state.context (set by context_manager) for
        # agent fields, not just BrainState attributes. Stage08b sets these in
        # context_manager, which becomes state.context after Stage09 snapshots.
        extra_fields = ["agent_id", "agent_confidence", "agent_mode"]
        extra_populated = 0
        for f in extra_fields:
            val = getattr(state, f, None)
            if val is None and hasattr(state, "context"):
                val = (state.context or {}).get(f)
            if val is not None:
                extra_populated += 1
        populated += extra_populated
        score = min(100.0, populated * 15.0)
        return ConversationMetric(
            name="ContextPreservation",
            score=score,
            status="PASS" if populated >= 1 else "WARN",
            details={"context": ctx, "extra": extra_populated},
        )

    def _stability_score(self, evidence: List[ConversationEvidence]) -> ConversationMetric:
        scores = [sum(m.score for m in ev.metrics) / max(1, len(ev.metrics)) for ev in evidence]
        if len(scores) < 2:
            return ConversationMetric(name="MultiTurnStability", score=100.0, status="PASS", details={})
        variance = sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        score = max(0.0, 100.0 - std)
        return ConversationMetric(
            name="MultiTurnStability",
            score=score,
            status="PASS" if score >= 60.0 else "WARN",
            details={"turn_scores": scores, "std": std},
        )

    def _extract_context(self, state: Any) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        for key in self._context_keys:
            value = getattr(state, key, None)
            if value is None and hasattr(state, "context"):
                value = (state.context or {}).get(key)
            ctx[key] = value
        return ctx

    def _aggregate_status(self, metrics: List[ConversationMetric]) -> str:
        if any(m.status == "FAIL" for m in metrics):
            return "FAIL"
        if any(m.status == "WARN" for m in metrics):
            return "WARN"
        return "PASS"
