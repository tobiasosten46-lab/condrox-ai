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

"""Tests for Condrox Evidence Engine (Phase 1 + 2)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from core.brain.brain_controller import BrainController
from core.evidence import (
    AcademicReportConfig,
    AcademicReportGenerator,
    AgentPipelineEvaluator,
    AgentPipelineSession,
    ConversationEvidenceEvaluator,
    ConversationSession,
    EvidenceCollector,
    EvidencePortalGenerator,
    EvidenceReport,
    EvidenceRun,
    EvidenceStore,
    FullSystemEvaluator,
    FullSystemEvidence,
    FullSystemSession,
    ReasoningEvidenceEvaluator,
    ReasoningSession,
    ReportGenerator,
)


@pytest.fixture(scope="module")
def booted_controller() -> BrainController:
    """Boot a single BrainController shared by evidence tests."""
    os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")
    bc = BrainController()
    bc.boot_system()
    yield bc
    bc.close()


@pytest.fixture
def collector(booted_controller: BrainController) -> EvidenceCollector:
    """Provide an EvidenceCollector that reuses the shared controller."""
    return EvidenceCollector(
        brain_controller=booted_controller,
    )


class TestConversationEvidenceEvaluator:
    """Tests for conversation evidence evaluation."""

    def test_evaluate_turn_returns_metrics(self) -> None:
        class FakeState:
            emotion = "neutral"
            mode = "explain"
            context = {"topic": "work"}

        evaluator = ConversationEvidenceEvaluator()
        ev = evaluator.evaluate_turn(
            "I'm feeling stressed.",
            "I hear you. Stress at work is really hard.",
            FakeState(),
        )
        assert ev.input == "I'm feeling stressed."
        assert "hear you" in ev.output
        metric_names = {m.name for m in ev.metrics}
        assert metric_names >= {"Empathy", "Relevance", "Consistency", "NoDictionaryDump", "ContextPreservation"}

    def test_multi_turn_stability(self) -> None:
        class FakeState:
            emotion = "neutral"
            mode = "explain"
            context = {}

        evaluator = ConversationEvidenceEvaluator()
        turns = [
            ("Hi", "Hello! How can I help?", FakeState()),
            ("I'm sad", "I'm sorry to hear that.", FakeState()),
        ]
        evidence = evaluator.evaluate_multi_turn(turns)
        assert len(evidence) == 2
        summary = evaluator.summarize(evidence)
        assert "score" in summary
        assert "status" in summary


class TestReasoningEvidenceEvaluator:
    """Tests for reasoning evidence evaluation."""

    def test_evaluate_reasoning_task(self) -> None:
        class FakeState:
            intent = "explain"
            mode = "explain"
            domain = "general"

        evaluator = ReasoningEvidenceEvaluator()
        output = (
            "1. Pick a pivot. 2. Partition the array. 3. Recurse on both halves. "
            "Therefore the average time complexity is O(n log n)."
        )
        ev = evaluator.evaluate("Explain quicksort.", output, FakeState())
        assert ev.input == "Explain quicksort."
        metric_names = {m.name for m in ev.metrics}
        assert metric_names >= {
            "StepByStepStructure",
            "LogicalCoherence",
            "ExplanationQuality",
            "PlanningQuality",
            "CodeAnalysis",
            "CorrectnessSignal",
        }

    def test_summarize_reasoning(self) -> None:
        class FakeState:
            intent = "explain"
            mode = "explain"
            domain = "general"

        evaluator = ReasoningEvidenceEvaluator()
        ev = evaluator.evaluate("What is 2+2?", "The answer is 4.", FakeState())
        summary = evaluator.summarize([ev])
        assert "score" in summary
        assert "status" in summary
        assert summary["tasks"] == 1


class TestAgentPipelineEvaluator:
    """Tests for agent/pipeline evidence evaluation."""

    def test_evaluate_agent_usage(self) -> None:
        class FakeState:
            intent = "code"
            mode = "code"
            domain = "ai_development"
            emotion = "neutral"
            context = {
                "agent_id": "code_agent",
                "agent_mode": "code",
                "agent_confidence": 0.95,
            }

        evaluator = AgentPipelineEvaluator()
        ev = evaluator.evaluate("Write a Python function to sort a list.", FakeState())
        assert ev.agent_id == "code_agent"
        assert ev.agent_mode == "code"
        metric_names = {m.name for m in ev.metrics}
        assert metric_names >= {"AgentUsage", "AgentMetadata", "RAGUsage", "MemoryUsage", "PersonalityUsage"}

    def test_pipeline_trace_summary(self) -> None:
        evaluator = AgentPipelineEvaluator()
        trace = evaluator.evaluate_pipeline_trace([
            ("Stage01", 0.0, 0.05),
            ("Stage02", 0.05, 0.10),
            ("Stage03", 0.10, 0.15),
        ])
        assert "score" in trace
        assert "status" in trace
        assert trace["stages"] == 3


class TestFullSystemEvaluator:
    """Tests for full-system integration evidence evaluation."""

    def test_evaluate_full_system_session(self, booted_controller: BrainController) -> None:
        evaluator = FullSystemEvaluator()
        steps = [
            {"input": "Hi, what can you do?", "category": "conversation"},
            {"input": "What is 2+2?", "category": "reasoning"},
            {"input": "Write a Python function to add two numbers.", "category": "agent"},
        ]
        evidence = evaluator.evaluate(booted_controller, steps)
        assert isinstance(evidence, FullSystemEvidence)
        assert evidence.name == "full_system_integration"
        assert len(evidence.checks) == len(steps) + 1  # steps + API health
        assert "score" in evidence.summary
        assert "status" in evidence.summary


class TestEvidenceCollector:
    """Tests for evidence collection."""

    def test_collect_system_evidence(self, collector: EvidenceCollector) -> None:
        report = collector.collect_system_evidence()
        assert 0.0 <= report.score <= 100.0
        assert any(c.name == "BrainController Boot" for c in report.checks)

    def test_collect_run(self, collector: EvidenceCollector) -> None:
        run = collector.collect_run(
            name="code_agent",
            category="agents",
            input_text="Write a Python function to sort a list.",
            expected_intent="code",
            expected_mode="code",
        )
        assert isinstance(run, EvidenceRun)
        assert run.status == "PASS"
        assert run.score == 100.0
        assert run.output_data["intent"] == "code"
        assert run.output_data["mode"] == "code"

    def test_collect_conversation(self, collector: EvidenceCollector) -> None:
        session = collector.collect_conversation(
            name="test_convo",
            turns=[{"input": "Hello"}, {"input": "How are you?"}],
        )
        assert isinstance(session, ConversationSession)
        assert session.name == "test_convo"
        assert len(session.turns) == 2
        assert hasattr(session, "summary")

    def test_collect_reasoning(self, collector: EvidenceCollector) -> None:
        session = collector.collect_reasoning(
            name="test_reasoning",
            tasks=[{"input": "What is 2+2?"}],
        )
        assert isinstance(session, ReasoningSession)
        assert session.name == "test_reasoning"
        assert len(session.tasks) == 1
        assert hasattr(session, "summary")

    def test_collect_agent_pipeline(self, collector: EvidenceCollector) -> None:
        session = collector.collect_agent_pipeline(
            name="test_agent_pipeline",
            requests=[{"input": "Write a Python function to sort a list."}],
        )
        assert isinstance(session, AgentPipelineSession)
        assert session.name == "test_agent_pipeline"
        assert len(session.requests) == 1
        assert hasattr(session, "summary")

    def test_collect_full_system(self, collector: EvidenceCollector) -> None:
        session = collector.collect_full_system(
            name="test_full_system",
            steps=[
                {"input": "Hello", "category": "conversation"},
                {"input": "What is 2+2?", "category": "reasoning"},
            ],
        )
        assert isinstance(session, FullSystemSession)
        assert session.name == "test_full_system"
        assert len(session.checks) >= 2
        assert hasattr(session, "summary")

    def test_collect_all(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(
            scenarios=[{
                "name": "test",
                "category": "test",
                "input_text": "Hello",
                "expected_intent": "greet",
            }],
            conversations=[{
                "name": "test_convo",
                "turns": [{"input": "Hello"}],
            }],
            reasoning=[{
                "name": "test_reasoning",
                "tasks": [{"input": "What is 2+2?"}],
            }],
            agent_pipeline=[{
                "name": "test_agent_pipeline",
                "requests": [{"input": "What is 2+2?"}],
            }],
            full_system=[{
                "name": "test_full_system",
                "steps": [{"input": "Hello", "category": "conversation"}],
            }],
        )
        assert isinstance(report, EvidenceReport)
        assert report.title.startswith("Condrox")
        assert 0.0 <= report.score <= 100.0
        assert len(report.runs) == 1
        assert len(report.conversation_sessions) == 1
        assert len(report.reasoning_sessions) == 1
        assert len(report.agent_pipeline_sessions) == 1
        assert len(report.full_system_sessions) == 1


class TestAcademicReportGenerator:
    """Tests for academic/lab-grade HTML report generation."""

    def test_generate_academic_html(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[])
        config = AcademicReportConfig(
            title="Test Academic Report",
            subtitle="Unit test edition",
            edition="Test Edition",
        )
        generator = AcademicReportGenerator()
        html = generator.generate(report, config)
        assert "<!DOCTYPE html>" in html
        assert config.title in html
        assert "Methodology" in html
        assert "References" in html
        assert "Signature and Attestation" in html
        assert "Reproducibility" in html

    def test_academic_report_writes_to_disk(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[])
        config = AcademicReportConfig(
            title="Disk Test",
            subtitle="Write test",
            edition="Disk Edition",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.html"
            generator = AcademicReportGenerator()
            generator.generate(report, config, path=path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert config.title in content


class TestEvidencePortalGenerator:
    """Tests for the static Evidence Portal generator."""

    def test_generates_portal_from_reports(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[])
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            store = EvidenceStore(base_dir=str(report_dir))
            store.save(report, filename="report1.json")

            generator = EvidencePortalGenerator()
            portal_path = generator.generate(report_dir=report_dir)
            assert portal_path.exists()
            content = portal_path.read_text(encoding="utf-8")
            assert "Condrox Evidence Portal" in content
            assert "Score History" in content
            assert "Evidence Reports" in content

    def test_portal_lists_multiple_reports(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[])
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            store = EvidenceStore(base_dir=str(report_dir))
            store.save(report, filename="report1.json")
            store.save(report, filename="report2.json")

            generator = EvidencePortalGenerator()
            portal_path = generator.generate(report_dir=report_dir)
            content = portal_path.read_text(encoding="utf-8")
            assert "report1.json" in content
            assert "report2.json" in content


class TestEvidenceStore:
    """Tests for evidence persistence."""

    def test_save_and_load(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[], conversations=[])
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(base_dir=tmp)
            path = store.save(report, filename="test.json")
            assert path.exists()
            data = store.load("test.json")
            assert data["title"] == report.title
            assert "runs" in data
            assert "conversation_sessions" in data


class TestReportGenerator:
    """Tests for JSON and HTML report generation."""

    def test_generate_json(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[], conversations=[])
        generator = ReportGenerator()
        text = generator.generate_json(report)
        assert '"title"' in text
        assert report.title in text

    def test_generate_html(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(scenarios=[], conversations=[])
        generator = ReportGenerator()
        html = generator.generate_html(report)
        assert "<!DOCTYPE html>" in html
        assert report.title in html

    def test_generate_html_with_conversations(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(
            scenarios=[],
            conversations=[{"name": "test", "turns": [{"input": "Hi"}]}],
        )
        generator = ReportGenerator()
        html = generator.generate_html(report)
        assert "Conversation Sessions" in html

    def test_generate_html_with_reasoning(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(
            scenarios=[],
            reasoning=[{"name": "test", "tasks": [{"input": "What is 2+2?"}]}],
        )
        generator = ReportGenerator()
        html = generator.generate_html(report)
        assert "Reasoning Sessions" in html

    def test_generate_html_with_agent_pipeline(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(
            scenarios=[],
            agent_pipeline=[{"name": "test", "requests": [{"input": "What is 2+2?"}]}],
        )
        generator = ReportGenerator()
        html = generator.generate_html(report)
        assert "Agent & Pipeline Sessions" in html

    def test_generate_html_with_full_system(self, collector: EvidenceCollector) -> None:
        report = collector.collect_all(
            scenarios=[],
            full_system=[{"name": "test", "steps": [{"input": "Hello", "category": "conversation"}]}],
        )
        generator = ReportGenerator()
        html = generator.generate_html(report)
        assert "Full System Integration Sessions" in html
