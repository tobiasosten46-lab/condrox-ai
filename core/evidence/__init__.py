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

"""Condrox Evidence Engine.

Collects, stores, and renders academic-grade evidence for Condrox AI
verification runs.
"""

from __future__ import annotations

from core.evidence.academic_report_generator import AcademicReportConfig, AcademicReportGenerator
from core.evidence.agent_pipeline_evaluator import AgentPipelineEvaluator
from core.evidence.conversation_evaluator import ConversationEvidenceEvaluator
from core.evidence.evidence_collector import EvidenceCollector
from core.evidence.evidence_report import (
    AgentPipelineSession,
    ConversationSession,
    EvidenceReport,
    EvidenceRun,
    FullSystemSession,
    ReasoningSession,
)
from core.evidence.evidence_store import EvidenceStore
from core.evidence.full_system_evaluator import FullSystemEvaluator, FullSystemEvidence
from core.evidence.portal_generator import EvidencePortalGenerator
from core.evidence.reasoning_evaluator import ReasoningEvidenceEvaluator
from core.evidence.report_generator import ReportGenerator

__all__ = [
    "AcademicReportConfig",
    "AcademicReportGenerator",
    "AgentPipelineEvaluator",
    "AgentPipelineSession",
    "ConversationEvidenceEvaluator",
    "ConversationSession",
    "EvidenceCollector",
    "EvidencePortalGenerator",
    "EvidenceReport",
    "EvidenceRun",
    "EvidenceStore",
    "FullSystemEvaluator",
    "FullSystemEvidence",
    "FullSystemSession",
    "ReasoningEvidenceEvaluator",
    "ReasoningSession",
    "ReportGenerator",
]
