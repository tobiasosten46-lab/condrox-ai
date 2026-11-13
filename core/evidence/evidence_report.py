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

"""Data structures for academic-grade evidence reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class EvidenceRun:
    """A single verification run captured as evidence.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(fields).
    """

    name: str
    category: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    status: str  # PASS, WARN, FAIL
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    deviations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass(frozen=True, slots=True)
class ConversationSession:
    """A named conversation session with its evidence."""

    name: str
    turns: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReasoningSession:
    """A named reasoning evaluation session with its evidence."""

    name: str
    tasks: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentPipelineSession:
    """A named agent/pipeline evaluation session with its evidence."""

    name: str
    requests: List[Dict[str, Any]]
    pipeline_traces: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class FullSystemSession:
    """A named full-system integration session with its evidence."""

    name: str
    checks: List[Dict[str, Any]]
    summary: Dict[str, Any]
    deviations: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvidenceReport:
    """Aggregated evidence report for a verification session.

    Complexity: T(n) ∈ O(r + c) for r runs and c conversation sessions, S(n) ∈ O(r + c).
    """

    title: str
    overall: str  # PASS, WARN, FAIL
    score: float
    version: str
    environment: Dict[str, Any]
    metadata: Dict[str, Any]
    runs: List[EvidenceRun] = field(default_factory=list)
    conversation_sessions: List[ConversationSession] = field(default_factory=list)
    reasoning_sessions: List[ReasoningSession] = field(default_factory=list)
    agent_pipeline_sessions: List[AgentPipelineSession] = field(default_factory=list)
    full_system_sessions: List[FullSystemSession] = field(default_factory=list)
    deviations: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to a dictionary."""
        return {
            "title": self.title,
            "overall": self.overall,
            "score": self.score,
            "version": self.version,
            "environment": self.environment,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
            "deviations": self.deviations,
            "runs": [
                {
                    "name": r.name,
                    "category": r.category,
                    "input": r.input_data,
                    "output": r.output_data,
                    "status": r.status,
                    "score": r.score,
                    "metrics": r.metrics,
                    "deviations": r.deviations,
                    "timestamp": r.timestamp,
                }
                for r in self.runs
            ],
            "conversation_sessions": [
                {
                    "name": s.name,
                    "summary": s.summary,
                    "turns": s.turns,
                }
                for s in self.conversation_sessions
            ],
            "reasoning_sessions": [
                {
                    "name": s.name,
                    "summary": s.summary,
                    "tasks": s.tasks,
                }
                for s in self.reasoning_sessions
            ],
            "agent_pipeline_sessions": [
                {
                    "name": s.name,
                    "summary": s.summary,
                    "requests": s.requests,
                    "pipeline_traces": s.pipeline_traces,
                }
                for s in self.agent_pipeline_sessions
            ],
            "full_system_sessions": [
                {
                    "name": s.name,
                    "summary": s.summary,
                    "checks": s.checks,
                    "deviations": s.deviations,
                }
                for s in self.full_system_sessions
            ],
        }
