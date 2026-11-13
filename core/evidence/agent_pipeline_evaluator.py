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

"""Agent and pipeline evidence evaluator.

Evaluates whether Condrox actually uses agents, pipeline stages, metadata,
RAG, personality engine, and memory manager during request processing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class AgentPipelineMetric:
    """A single agent/pipeline quality metric."""

    name: str
    score: float  # 0.0–100.0
    status: str  # PASS, WARN, FAIL
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentPipelineEvidence:
    """Evidence captured for one agent/pipeline invocation."""

    input: str
    agent_id: str
    agent_mode: str
    intent: str
    mode: str
    domain: str
    metrics: List[AgentPipelineMetric] = field(default_factory=list)
    deviations: List[str] = field(default_factory=list)


class AgentPipelineEvaluator:
    """Evaluate agent and pipeline utilization and produce evidence.

    Complexity: T(evaluate) ∈ O(1) per request.
    """

    __slots__ = ()

    def evaluate(
        self,
        user_input: str,
        state: Any,
    ) -> AgentPipelineEvidence:
        """Evaluate a single request for agent/pipeline utilization."""
        metrics: List[AgentPipelineMetric] = []
        deviations: List[str] = []

        agent_id = self._extract_agent_id(state)
        agent_mode = self._extract_agent_mode(state)
        intent = getattr(state, "intent", "")
        mode = getattr(state, "mode", "")
        domain = getattr(state, "domain", "")

        agent_usage = self._agent_usage_metric(agent_id, agent_mode, mode)
        metrics.append(agent_usage)
        if agent_usage.status != "PASS":
            deviations.append(f"Agent usage issue: {agent_usage.details}")

        metadata = self._metadata_metric(state)
        metrics.append(metadata)
        if metadata.status != "PASS":
            deviations.append("Agent metadata not present or incomplete")

        rag_usage = self._rag_usage_metric(state)
        metrics.append(rag_usage)
        if rag_usage.status == "FAIL":
            deviations.append("RAG was expected but not used")

        memory_usage = self._memory_usage_metric(state)
        metrics.append(memory_usage)
        if memory_usage.status == "FAIL":
            deviations.append("Memory manager was expected but not used")

        personality_usage = self._personality_usage_metric(state)
        metrics.append(personality_usage)

        return AgentPipelineEvidence(
            input=user_input,
            agent_id=agent_id,
            agent_mode=agent_mode,
            intent=intent,
            mode=mode,
            domain=domain,
            metrics=metrics,
            deviations=deviations,
        )

    def evaluate_pipeline_trace(
        self,
        stage_times: List[Tuple[str, float, float]],
    ) -> Dict[str, Any]:
        """Evaluate a pipeline stage timing trace."""
        if not stage_times:
            return {"score": 0.0, "status": "FAIL", "total_ms": 0.0, "stages": 0}

        total_ms = sum(end - start for _, start, end in stage_times) * 1000.0
        avg_ms = total_ms / len(stage_times)
        score = min(100.0, max(0.0, 100.0 - (total_ms / 1000.0)))
        status = "PASS" if score >= 50.0 else "WARN"
        return {
            "score": round(score, 2),
            "status": status,
            "total_ms": round(total_ms, 2),
            "stages": len(stage_times),
            "average_ms": round(avg_ms, 2),
        }

    def summarize(
        self,
        evidence: List[AgentPipelineEvidence],
        pipeline_traces: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return a summary dictionary for agent/pipeline evidence."""
        if not evidence:
            return {"score": 0.0, "status": "FAIL", "requests": 0}

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

        pipeline_summary = {"traces": len(pipeline_traces or [])}
        if pipeline_traces:
            pipeline_summary["avg_score"] = round(
                sum(t.get("score", 0.0) for t in pipeline_traces) / len(pipeline_traces), 2
            )

        return {
            "score": round(score, 2),
            "status": status,
            "requests": len(evidence),
            "metric_averages": {name: round(sum(scores) / len(scores), 2) for name, scores in by_name.items()},
            "deviation_count": sum(len(ev.deviations) for ev in evidence),
            "pipeline_summary": pipeline_summary,
        }

    def _extract_agent_id(self, state: Any) -> str:
        if hasattr(state, "context") and state.context:
            return str(state.context.get("agent_id", "") or "")
        return ""

    def _extract_agent_mode(self, state: Any) -> str:
        if hasattr(state, "context") and state.context:
            return str(state.context.get("agent_mode", "") or "")
        return ""

    def _agent_usage_metric(self, agent_id: str, agent_mode: str, mode: str) -> AgentPipelineMetric:
        """Check that an appropriate agent was selected and used."""
        has_agent = bool(agent_id)
        mode_match = bool(agent_mode) and agent_mode == mode
        score = 0.0
        if has_agent:
            score += 50.0
        if mode_match:
            score += 50.0
        status = "PASS" if score >= 80.0 else ("WARN" if score >= 50.0 else "FAIL")
        return AgentPipelineMetric(
            name="AgentUsage",
            score=score,
            status=status,
            details={"agent_id": agent_id, "agent_mode": agent_mode, "mode": mode},
        )

    def _metadata_metric(self, state: Any) -> AgentPipelineMetric:
        """Check that agent metadata is present in state context."""
        context = getattr(state, "context", {}) or {}
        required_keys = ("agent_id", "agent_mode", "agent_confidence")
        populated = sum(1 for k in required_keys if context.get(k))
        score = min(100.0, populated * (100.0 / len(required_keys)))
        status = "PASS" if populated >= 2 else "WARN"
        return AgentPipelineMetric(
            name="AgentMetadata",
            score=score,
            status=status,
            details={"populated": populated, "context_keys": list(context.keys())[:10]},
        )

    def _rag_usage_metric(self, state: Any) -> AgentPipelineMetric:
        """Check whether RAG was invoked when knowledge was requested."""
        context = getattr(state, "context", {}) or {}
        knowledge_requested = bool(context.get("knowledge_requested") or context.get("knowledge_count"))
        has_knowledge = bool(context.get("has_knowledge") or context.get("knowledge_count", 0) > 0)
        if not knowledge_requested:
            score = 100.0
            status = "PASS"
        else:
            score = 100.0 if has_knowledge else 0.0
            status = "PASS" if has_knowledge else "FAIL"
        return AgentPipelineMetric(
            name="RAGUsage",
            score=score,
            status=status,
            details={"knowledge_requested": knowledge_requested, "has_knowledge": has_knowledge},
        )

    def _memory_usage_metric(self, state: Any) -> AgentPipelineMetric:
        """Check whether memory manager was consulted."""
        context = getattr(state, "context", {}) or {}
        memory_count = context.get("memory_count", 0) or 0
        has_memory = bool(context.get("has_memory") or memory_count > 0)
        score = 100.0 if has_memory else 50.0
        status = "PASS"
        return AgentPipelineMetric(
            name="MemoryUsage",
            score=score,
            status=status,
            details={"has_memory": has_memory, "memory_count": memory_count},
        )

    def _personality_usage_metric(self, state: Any) -> AgentPipelineMetric:
        """Check whether personality engine shaped the output."""
        emotion = getattr(state, "emotion", "") or ""
        tone = getattr(state, "tone", "") or ""
        has_personality = bool(emotion) or bool(tone)
        score = 100.0 if has_personality else 60.0
        status = "PASS" if has_personality else "WARN"
        return AgentPipelineMetric(
            name="PersonalityUsage",
            score=score,
            status=status,
            details={"emotion": emotion, "tone": tone},
        )
