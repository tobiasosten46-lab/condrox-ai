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

"""Full-system integration evidence evaluator.

Runs a realistic end-to-end scenario that exercises conversation, reasoning,
agent execution, RAG, memory, personality, pipeline flow, and (optionally)
API health — producing a single integration score.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.brain.brain_controller import BrainController
from core.evidence.agent_pipeline_evaluator import AgentPipelineEvaluator
from core.evidence.conversation_evaluator import ConversationEvidenceEvaluator
from core.evidence.reasoning_evaluator import ReasoningEvidenceEvaluator


@dataclass(frozen=True, slots=True)
class IntegrationCheck:
    """A single integration check result."""

    name: str
    status: str  # PASS, WARN, FAIL
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FullSystemEvidence:
    """Evidence from a full end-to-end system run."""

    name: str
    checks: List[IntegrationCheck]
    summary: Dict[str, Any]
    deviations: List[str] = field(default_factory=list)


class FullSystemEvaluator:
    """Evaluate a complete Condrox AI session from end to end.

    Complexity: T(evaluate) ∈ O(s) for s sub-evaluations.
    """

    __slots__ = ("_api_host", "_api_port")

    def __init__(self, api_host: str = "localhost", api_port: int = 8000) -> None:
        self._api_host = api_host
        self._api_port = api_port

    def evaluate(
        self,
        controller: BrainController,
        steps: List[Dict[str, Any]],
    ) -> FullSystemEvidence:
        """Run a full-system scenario and produce evidence.

        Each step is a dict with:
          - "input" (str): user message
          - "category" (str): "conversation", "reasoning", "agent", "rag"
        """
        conversation_eval = ConversationEvidenceEvaluator()
        reasoning_eval = ReasoningEvidenceEvaluator()
        agent_eval = AgentPipelineEvaluator()

        checks: List[IntegrationCheck] = []
        deviations: List[str] = []
        previous_output = ""
        conversation_turns = 0
        reasoning_tasks = 0
        agent_invocations = 0
        rag_hits = 0

        for step in steps:
            user_input = str(step.get("input", ""))
            category = str(step.get("category", "conversation"))
            state = controller.process_request(user_input)
            output = str(getattr(state, "final_output", "") or "")

            if category == "conversation":
                ev = conversation_eval.evaluate_turn(user_input, output, state, previous_output)
                status = self._aggregate_status(ev.metrics)
                checks.append(
                    IntegrationCheck(
                        name=f"conversation_{conversation_turns + 1}",
                        status=status,
                        score=sum(m.score for m in ev.metrics) / max(1, len(ev.metrics)),
                        details={"input": user_input, "metrics": [(m.name, m.status) for m in ev.metrics]},
                    )
                )
                conversation_turns += 1
                if ev.deviations:
                    deviations.extend(ev.deviations)
            elif category == "reasoning":
                ev = reasoning_eval.evaluate(user_input, output, state)
                status = self._aggregate_status(ev.metrics)
                checks.append(
                    IntegrationCheck(
                        name=f"reasoning_{reasoning_tasks + 1}",
                        status=status,
                        score=sum(m.score for m in ev.metrics) / max(1, len(ev.metrics)),
                        details={"input": user_input, "metrics": [(m.name, m.status) for m in ev.metrics]},
                    )
                )
                reasoning_tasks += 1
                if ev.deviations:
                    deviations.extend(ev.deviations)
            elif category in ("agent", "rag"):
                ev = agent_eval.evaluate(user_input, state)
                status = self._aggregate_status(ev.metrics)
                checks.append(
                    IntegrationCheck(
                        name=f"agent_{agent_invocations + 1}",
                        status=status,
                        score=sum(m.score for m in ev.metrics) / max(1, len(ev.metrics)),
                        details={"input": user_input, "agent_id": ev.agent_id, "metrics": [(m.name, m.status) for m in ev.metrics]},
                    )
                )
                agent_invocations += 1
                if ev.deviations:
                    deviations.extend(ev.deviations)
                if category == "rag" and any(m.name == "RAGUsage" and m.status == "PASS" for m in ev.metrics):
                    rag_hits += 1

            previous_output = output

        # API health check
        api_status = self._check_api_health()
        checks.append(api_status)
        if api_status.status != "PASS":
            deviations.append(api_status.details.get("message", "API health check failed"))

        scores = [c.score for c in checks]
        score = sum(scores) / len(scores) if scores else 0.0
        fail_count = sum(1 for c in checks if c.status == "FAIL")
        warn_count = sum(1 for c in checks if c.status == "WARN")
        status = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")
        critical = fail_count

        summary = {
            "score": round(score, 2),
            "status": status,
            "checks": len(checks),
            "conversation_turns": conversation_turns,
            "reasoning_tasks": reasoning_tasks,
            "agent_invocations": agent_invocations,
            "rag_hits": rag_hits,
            "critical_issues": critical,
            "release_status": "READY" if status == "PASS" and critical == 0 else "NOT_READY",
        }

        return FullSystemEvidence(
            name="full_system_integration",
            checks=checks,
            summary=summary,
            deviations=deviations,
        )

    def _check_api_health(self) -> IntegrationCheck:
        url = f"http://{self._api_host}:{self._api_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                data = response.read().decode("utf-8")
                healthy = '"healthy"' in data
                return IntegrationCheck(
                    name="APIHealth",
                    status="PASS" if healthy else "WARN",
                    score=100.0 if healthy else 50.0,
                    details={"url": url, "response": data[:200]},
                )
        except urllib.error.URLError as exc:
            return IntegrationCheck(
                name="APIHealth",
                status="WARN",
                score=50.0,
                details={"url": url, "message": str(exc)},
            )
        except Exception as exc:
            return IntegrationCheck(
                name="APIHealth",
                status="FAIL",
                score=0.0,
                details={"url": url, "message": str(exc)},
            )

    @staticmethod
    def _aggregate_status(metrics: List[Any]) -> str:
        if any(m.status == "FAIL" for m in metrics):
            return "FAIL"
        if any(m.status == "WARN" for m in metrics):
            return "WARN"
        return "PASS"
