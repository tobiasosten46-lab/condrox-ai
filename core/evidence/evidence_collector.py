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

"""Evidence collector — runs verification scenarios and captures evidence."""

from __future__ import annotations

import os
import platform
import socket
import time
from typing import Any, Dict, List, Optional

from core.brain.brain_controller import BrainController
from core.evidence.agent_pipeline_evaluator import AgentPipelineEvaluator
from core.evidence.conversation_evaluator import ConversationEvidenceEvaluator
from core.evidence.evidence_report import (
    AgentPipelineSession,
    ConversationSession,
    EvidenceReport,
    EvidenceRun,
    FullSystemSession,
    ReasoningSession,
)
from core.evidence.full_system_evaluator import FullSystemEvaluator
from core.evidence.reasoning_evaluator import ReasoningEvidenceEvaluator
from core.verification import SystemProbe
from core.verification.report import VerificationReport


class EvidenceCollector:
    """Collect evidence from system probes and BrainController runs.

    Complexity: T(collect) ∈ O(p + s) for p probes and s scenarios.
    """

    __slots__ = ("_brain_controller", "_own_controller", "_api_host", "_api_port")

    def __init__(
        self,
        brain_controller: Optional[BrainController] = None,
        api_host: str = "localhost",
        api_port: int = 8000,
    ) -> None:
        os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")
        self._brain_controller: Optional[BrainController] = brain_controller
        self._own_controller: bool = brain_controller is None
        self._api_host = api_host
        self._api_port = api_port

    def _ensure_controller(self) -> BrainController:
        """Return a booted BrainController, creating one if needed."""
        if self._brain_controller is None:
            self._brain_controller = BrainController()
            self._brain_controller.boot_system()
        return self._brain_controller

    def close(self) -> None:
        """Release resources if the collector created the controller."""
        if self._own_controller and self._brain_controller is not None:
            self._brain_controller.close()
            self._brain_controller = None

    def collect_system_evidence(self) -> VerificationReport:
        """Run system probes and return a verification report."""
        bc = self._ensure_controller()
        probe = SystemProbe(
            api_host=self._api_host,
            api_port=self._api_port,
            brain_controller=bc,
        )
        results = list(probe.probe_all().values())
        pass_count = sum(1 for r in results if r.status == "PASS")
        fail_count = sum(1 for r in results if r.status == "FAIL")
        total = len(results)
        score = (pass_count / total) * 100.0 if total else 0.0
        overall = "FAIL" if fail_count else ("WARN" if any(r.status == "WARN" for r in results) else "OK")
        return VerificationReport(
            overall=overall,
            score=score,
            checks=results,
            metadata={"api_host": self._api_host, "api_port": self._api_port},
        )

    def collect_run(
        self,
        name: str,
        category: str,
        input_text: str,
        expected_intent: str = "",
        expected_mode: str = "",
    ) -> EvidenceRun:
        """Run a single scenario through BrainController and capture evidence."""
        bc = self._ensure_controller()
        start = time.perf_counter()
        state = bc.process_request(input_text)
        duration_ms = (time.perf_counter() - start) * 1000

        status = "PASS"
        deviations: List[str] = []
        if expected_intent and state.intent != expected_intent:
            status = "WARN"
            deviations.append(
                f"Expected intent '{expected_intent}', got '{state.intent}'"
            )
        if expected_mode and state.mode != expected_mode:
            status = "WARN"
            deviations.append(
                f"Expected mode '{expected_mode}', got '{state.mode}'"
            )

        score = 100.0 if status == "PASS" else 50.0
        return EvidenceRun(
            name=name,
            category=category,
            input_data={"text": input_text, "expected_intent": expected_intent, "expected_mode": expected_mode},
            output_data={
                "final_output": state.final_output,
                "intent": state.intent,
                "mode": state.mode,
                "domain": state.domain,
                "emotion": state.emotion,
                "context": dict(state.context or {}),
            },
            status=status,
            score=score,
            metrics={"duration_ms": duration_ms},
            deviations=deviations,
        )

    def collect_conversation(
        self,
        name: str,
        turns: List[Dict[str, Any]],
    ) -> ConversationSession:
        """Run a multi-turn conversation through BrainController and evaluate it.

        Each turn is a dict with key "input" (str). The controller is invoked
        sequentially and previous assistant output is passed to the evaluator.
        """
        bc = self._ensure_controller()
        evaluator = ConversationEvidenceEvaluator()
        turn_data: List[Any] = []
        previous_output = ""
        for turn in turns:
            user_input = str(turn.get("input", ""))
            state = bc.process_request(user_input)
            output = str(getattr(state, "final_output", "") or "")
            turn_data.append((user_input, output, state, previous_output))
            previous_output = output

        # Evaluate each turn against the previous assistant output.
        evidence = []
        for user_input, output, state, prev in turn_data:
            ev = evaluator.evaluate_turn(user_input, output, state, prev)
            evidence.append(ev)

        # Attach multi-turn stability summary.
        summary = evaluator.summarize(evidence)
        return ConversationSession(
            name=name,
            turns=[
                {
                    "input": ev.input,
                    "output": ev.output,
                    "emotion": ev.emotion,
                    "mode": ev.mode,
                    "metrics": [
                        {"name": m.name, "score": m.score, "status": m.status, "details": m.details}
                        for m in ev.metrics
                    ],
                    "deviations": ev.deviations,
                }
                for ev in evidence
            ],
            summary=summary,
        )

    def collect_reasoning(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
    ) -> ReasoningSession:
        """Run a set of reasoning tasks through BrainController and evaluate them."""
        bc = self._ensure_controller()
        evaluator = ReasoningEvidenceEvaluator()
        evidence_tasks: List[Any] = []
        for task in tasks:
            user_input = str(task.get("input", ""))
            state = bc.process_request(user_input)
            output = str(getattr(state, "final_output", "") or "")
            ev = evaluator.evaluate(user_input, output, state)
            evidence_tasks.append(ev)

        summary = evaluator.summarize(evidence_tasks)
        return ReasoningSession(
            name=name,
            tasks=[
                {
                    "input": ev.input,
                    "output": ev.output,
                    "intent": ev.intent,
                    "mode": ev.mode,
                    "domain": ev.domain,
                    "metrics": [
                        {"name": m.name, "score": m.score, "status": m.status, "details": m.details}
                        for m in ev.metrics
                    ],
                    "deviations": ev.deviations,
                }
                for ev in evidence_tasks
            ],
            summary=summary,
        )

    def collect_agent_pipeline(
        self,
        name: str,
        requests: List[Dict[str, Any]],
    ) -> AgentPipelineSession:
        """Run a set of requests through BrainController and evaluate agent/pipeline usage."""
        bc = self._ensure_controller()
        evaluator = AgentPipelineEvaluator()
        evidence_requests: List[Any] = []
        pipeline_traces: List[Dict[str, Any]] = []
        for req in requests:
            user_input = str(req.get("input", ""))
            start = time.perf_counter()
            state = bc.process_request(user_input)
            duration_ms = (time.perf_counter() - start) * 1000
            ev = evaluator.evaluate(user_input, state)
            evidence_requests.append(ev)
            pipeline_traces.append({
                "input": user_input,
                "stages": bc.pipeline.stage_count(),
                "duration_ms": duration_ms,
            })

        summary = evaluator.summarize(evidence_requests, pipeline_traces=pipeline_traces)
        return AgentPipelineSession(
            name=name,
            requests=[
                {
                    "input": ev.input,
                    "agent_id": ev.agent_id,
                    "agent_mode": ev.agent_mode,
                    "intent": ev.intent,
                    "mode": ev.mode,
                    "domain": ev.domain,
                    "metrics": [
                        {"name": m.name, "score": m.score, "status": m.status, "details": m.details}
                        for m in ev.metrics
                    ],
                    "deviations": ev.deviations,
                }
                for ev in evidence_requests
            ],
            pipeline_traces=pipeline_traces,
            summary=summary,
        )

    def collect_full_system(
        self,
        name: str,
        steps: List[Dict[str, Any]],
    ) -> FullSystemSession:
        """Run an end-to-end full-system integration scenario and capture evidence."""
        bc = self._ensure_controller()
        evaluator = FullSystemEvaluator(
            api_host=self._api_host,
            api_port=self._api_port,
        )
        evidence = evaluator.evaluate(bc, steps)
        return FullSystemSession(
            name=name,
            checks=[
                {
                    "name": c.name,
                    "status": c.status,
                    "score": c.score,
                    "details": c.details,
                }
                for c in evidence.checks
            ],
            summary=evidence.summary,
            deviations=evidence.deviations,
        )

    def collect_all(
        self,
        title: str = "Condrox AI Evidence Report",
        scenarios: Optional[List[Dict[str, Any]]] = None,
        conversations: Optional[List[Dict[str, Any]]] = None,
        reasoning: Optional[List[Dict[str, Any]]] = None,
        agent_pipeline: Optional[List[Dict[str, Any]]] = None,
        full_system: Optional[List[Dict[str, Any]]] = None,
    ) -> EvidenceReport:
        """Run all evidence collection modes and produce an EvidenceReport."""
        bc = self._ensure_controller()
        version = bc.config.version

        system_report = self.collect_system_evidence()
        runs: List[EvidenceRun] = []
        for scenario in scenarios or []:
            runs.append(
                self.collect_run(
                    name=str(scenario.get("name", "unnamed")),
                    category=str(scenario.get("category", "general")),
                    input_text=str(scenario.get("input_text", scenario.get("input", ""))),
                    expected_intent=str(scenario.get("expected_intent", "")),
                    expected_mode=str(scenario.get("expected_mode", "")),
                )
            )

        conversation_sessions: List[ConversationSession] = []
        for conversation in conversations or []:
            conversation_sessions.append(
                self.collect_conversation(
                    name=str(conversation.get("name", "unnamed")),
                    turns=list(conversation.get("turns", [])),
                )
            )

        reasoning_sessions: List[ReasoningSession] = []
        for session in reasoning or []:
            reasoning_sessions.append(
                self.collect_reasoning(
                    name=str(session.get("name", "unnamed")),
                    tasks=list(session.get("tasks", [])),
                )
            )

        agent_pipeline_sessions: List[AgentPipelineSession] = []
        for session in agent_pipeline or []:
            agent_pipeline_sessions.append(
                self.collect_agent_pipeline(
                    name=str(session.get("name", "unnamed")),
                    requests=list(session.get("requests", [])),
                )
            )

        full_system_sessions: List[FullSystemSession] = []
        for session in full_system or []:
            full_system_sessions.append(
                self.collect_full_system(
                    name=str(session.get("name", "unnamed")),
                    steps=list(session.get("steps", [])),
                )
            )

        all_scores = [system_report.score]
        all_scores += [r.score for r in runs]
        all_scores += [s.summary["score"] for s in conversation_sessions]
        all_scores += [s.summary["score"] for s in reasoning_sessions]
        all_scores += [s.summary["score"] for s in agent_pipeline_sessions]
        all_scores += [s.summary["score"] for s in full_system_sessions]
        score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        sessions = conversation_sessions + reasoning_sessions + agent_pipeline_sessions + full_system_sessions
        any_fail = (
            system_report.overall == "FAIL"
            or any(r.status == "FAIL" for r in runs)
            or any(s.summary.get("status", "FAIL") == "FAIL" for s in sessions)
        )
        any_warn = (
            system_report.overall == "WARN"
            or any(r.status == "WARN" for r in runs)
            or any(s.summary.get("status", "FAIL") == "WARN" for s in sessions)
        )
        overall = "FAIL" if any_fail else ("WARN" if any_warn else "PASS")

        return EvidenceReport(
            title=title,
            overall=overall,
            score=score,
            version=version,
            environment=self._environment(),
            metadata={
                "scenario_count": len(runs),
                "conversation_count": len(conversation_sessions),
                "reasoning_session_count": len(reasoning_sessions),
                "agent_pipeline_session_count": len(agent_pipeline_sessions),
                "full_system_session_count": len(full_system_sessions),
                "system_checks": len(system_report.checks),
            },
            runs=runs,
            conversation_sessions=conversation_sessions,
            reasoning_sessions=reasoning_sessions,
            agent_pipeline_sessions=agent_pipeline_sessions,
            full_system_sessions=full_system_sessions,
            deviations=[
                d for r in runs for d in r.deviations
            ] + [
                f"{c.name}: {c.message}" for c in system_report.checks if c.status in ("FAIL", "WARN")
            ],
        )

    @staticmethod
    def _environment() -> Dict[str, Any]:
        """Capture runtime environment metadata."""
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hostname": socket.gethostname(),
            "processor": platform.processor(),
        }
