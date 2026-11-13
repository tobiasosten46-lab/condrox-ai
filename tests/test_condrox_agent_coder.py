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
Comprehensive pytest test suite for DeveloperAgent v2.0.

Tests cover:
- 13 CodeIntent classifiers
- State machine transitions
- Domain detection edge cases
- Confidence scoring boundaries
- Auto-repair patterns
- Integration with Condrox brain

Complexity: T(n) in O(tests), S(n) in O(fixtures).
"""

import pytest
from typing import Any
from unittest.mock import Mock, AsyncMock, MagicMock

from agents.DeveloperAgent.condrox_agent_coder import (
    AgentState,
    CodeIntent,
    ConfidenceLevel,
    DomainType,
    CodeTask,
    VerificationResult,
    AgentCodeResult,
    DomainDetection,
    DomainDetector,
    IntentClassifier,
    ConfidenceScorer,
    PlanBuilder,
    AutoRepair,
    CondroxAgentCoder,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_brain_controller() -> Mock:
    """Mock Condrox BrainController for testing."""
    controller = Mock()
    controller.process_request = Mock(return_value=Mock(
        final_output="Generated code",
        emotion="neutral",
        intent="explain",
        mode="explain",
        domain="general",
        reasoning_output={},
    ))
    return controller


@pytest.fixture
def agent() -> CondroxAgentCoder:
    """CondroxAgentCoder instance for testing."""
    return CondroxAgentCoder(root_dir=".")


@pytest.fixture
def domain_detector() -> DomainDetector:
    """DomainDetector instance for testing."""
    return DomainDetector()


@pytest.fixture
def intent_classifier() -> IntentClassifier:
    """IntentClassifier instance for testing."""
    return IntentClassifier(word_engine=None)


@pytest.fixture
def confidence_scorer() -> ConfidenceScorer:
    """ConfidenceScorer instance for testing."""
    return ConfidenceScorer()


@pytest.fixture
def plan_builder() -> PlanBuilder:
    """PlanBuilder instance for testing."""
    return PlanBuilder()


@pytest.fixture
def auto_repair() -> AutoRepair:
    """AutoRepair instance for testing."""
    return AutoRepair()


# ─── Domain Detection Tests ───────────────────────────────────────────────────


class TestDomainDetection:
    """Test domain auto-detection edge cases."""

    def test_ai_dev_detection(self, domain_detector: DomainDetector) -> None:
        """Test AI Development domain detection."""
        result = domain_detector.detect(
            "ml/model.py",
            "import tensorflow\nimport pytorch\nneural network",
            "python"
        )
        assert result.domain == DomainType.AI_DEV
        assert result.score >= 0.80

    def test_web_design_detection(self, domain_detector: DomainDetector) -> None:
        """Test Web Design domain detection."""
        result = domain_detector.detect(
            "frontend/App.tsx",
            "import react\nimport tailwind\ncomponent",
            "typescript"
        )
        assert result.domain == DomainType.WEB_DESIGN
        assert result.score >= 0.85

    def test_systems_eng_detection(self, domain_detector: DomainDetector) -> None:
        """Test Systems Engineering domain detection."""
        result = domain_detector.detect(
            "kernel/driver.rs",
            "memory safe\nconcurrency\nsyscall",
            "rust"
        )
        assert result.domain == DomainType.SYSTEMS_ENG
        assert result.score >= 0.85

    def test_data_science_detection(self, domain_detector: DomainDetector) -> None:
        """Test Data Science domain detection."""
        result = domain_detector.detect(
            "data/analysis.py",
            "import pandas\nimport numpy\ndataframe",
            "python"
        )
        assert result.domain == DomainType.DATA_SCIENCE
        assert result.score >= 0.75

    def test_general_domain_fallback(self, domain_detector: DomainDetector) -> None:
        """Test GENERAL domain fallback when no signals."""
        result = domain_detector.detect(
            "random.py",
            "print('hello')",
            "python"
        )
        assert result.domain == DomainType.GENERAL
        assert result.score == 0.0

    def test_empty_input_handling(self, domain_detector: DomainDetector) -> None:
        """Test empty input handling."""
        result = domain_detector.detect("", "", "")
        assert result.domain == DomainType.GENERAL
        assert result.score == 0.0


# ─── Intent Classification Tests ───────────────────────────────────────────────


class TestIntentClassification:
    """Test 13 CodeIntent classifiers."""

    @pytest.mark.parametrize("input_text,expected_intent", [
        ("fix this bug", CodeIntent.FIX),
        ("generate a function", CodeIntent.GENERATE),
        ("review my code", CodeIntent.REVIEW),
        ("refactor this class", CodeIntent.REFACTOR),
        ("write tests", CodeIntent.TEST),
        ("document this", CodeIntent.DOCUMENT),
        ("analyze this code", CodeIntent.ANALYZE),
        ("optimize performance", CodeIntent.OPTIMIZE),
        ("security audit", CodeIntent.SECURITY_AUDIT),
        ("explain how this works", CodeIntent.EXPLAIN),
        ("deploy to kubernetes", CodeIntent.DEPLOY),
        ("migrate to new version", CodeIntent.MIGRATE),
        ("general request", CodeIntent.GENERAL),
    ])
    def test_intent_classification(
        self,
        intent_classifier: IntentClassifier,
        input_text: str,
        expected_intent: CodeIntent
    ) -> None:
        """Test intent classification for all 13 intents."""
        intent, confidence = intent_classifier.classify(input_text)
        assert intent == expected_intent
        assert 0.0 <= confidence <= 1.0

    def test_empty_input_classification(self, intent_classifier: IntentClassifier) -> None:
        """Test empty input classification."""
        intent, confidence = intent_classifier.classify("")
        assert intent == CodeIntent.GENERAL
        assert confidence == 0.0

    def test_whitespace_input_classification(self, intent_classifier: IntentClassifier) -> None:
        """Test whitespace-only input classification."""
        intent, confidence = intent_classifier.classify("   ")
        assert intent == CodeIntent.GENERAL
        assert confidence == 0.0

    def test_norwegian_keywords(self, intent_classifier: IntentClassifier) -> None:
        """Test Norwegian keyword support."""
        intent, confidence = intent_classifier.classify("fikse denne feilen")
        assert intent == CodeIntent.FIX
        assert confidence > 0.0

    def test_confidence_calculation(self, intent_classifier: IntentClassifier) -> None:
        """Test confidence calculation based on keyword matches."""
        # More keywords = higher confidence
        intent1, conf1 = intent_classifier.classify("fix bug error broken")
        intent2, conf2 = intent_classifier.classify("fix")
        assert intent1 == intent2 == CodeIntent.FIX
        assert conf1 > conf2


# ─── Confidence Scoring Tests ───────────────────────────────────────────────


class TestConfidenceScoring:
    """Test confidence scoring boundaries."""

    def test_certain_level(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test CERTAIN level (95%+)."""
        assert confidence_scorer.to_level(0.95) == ConfidenceLevel.CERTAIN
        assert confidence_scorer.to_level(1.0) == ConfidenceLevel.CERTAIN

    def test_high_level(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test HIGH level (80-95%)."""
        assert confidence_scorer.to_level(0.80) == ConfidenceLevel.HIGH
        assert confidence_scorer.to_level(0.94) == ConfidenceLevel.HIGH

    def test_medium_level(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test MEDIUM level (60-80%)."""
        assert confidence_scorer.to_level(0.60) == ConfidenceLevel.MEDIUM
        assert confidence_scorer.to_level(0.79) == ConfidenceLevel.MEDIUM

    def test_low_level(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test LOW level (<60%)."""
        assert confidence_scorer.to_level(0.0) == ConfidenceLevel.LOW
        assert confidence_scorer.to_level(0.59) == ConfidenceLevel.LOW

    def test_out_of_range_clamping(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test out-of-range clamping."""
        assert confidence_scorer.to_level(-0.1) == ConfidenceLevel.LOW
        assert confidence_scorer.to_level(1.5) == ConfidenceLevel.CERTAIN

    def test_from_verification_calculation(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test final confidence from verification + intent confidence."""
        verification = VerificationResult(
            passed=True,
            blocked=False,
            score=0.9,
            gate_results=(),
            formatted_report="",
            critical_count=0,
            error_count=0,
            warning_count=0,
        )
        final = confidence_scorer.from_verification(verification, 0.8)
        # Formula: 0.5 * intent + 0.5 * verification
        assert final == pytest.approx(0.85, abs=0.01)

    def test_critical_penalty(self, confidence_scorer: ConfidenceScorer) -> None:
        """Test critical finding penalty."""
        verification = VerificationResult(
            passed=False,
            blocked=True,
            score=0.9,
            gate_results=(),
            formatted_report="",
            critical_count=1,
            error_count=0,
            warning_count=0,
        )
        final = confidence_scorer.from_verification(verification, 0.8)
        # Penalty: 1 * 0.3 = 0.3
        assert final < 0.85


# ─── Plan Builder Tests ─────────────────────────────────────────────────────


class TestPlanBuilder:
    """Test plan templates for all intents."""

    def test_generate_plan(self, plan_builder: PlanBuilder) -> None:
        """Test GENERATE plan template."""
        plan = plan_builder.build(CodeIntent.GENERATE)
        assert "Understand requirements" in plan
        assert "Design data structures" in plan
        assert "Quality gates" in plan

    def test_fix_plan(self, plan_builder: PlanBuilder) -> None:
        """Test FIX plan template (debug workflow)."""
        plan = plan_builder.build(CodeIntent.FIX)
        assert "SYMPTOM ANALYSIS" in plan
        assert "HYPOTHESIS GENERATION" in plan
        assert "ISOLATION" in plan
        assert "FIX IMPLEMENTATION" in plan
        assert "REGRESSION TEST" in plan

    def test_analyze_plan(self, plan_builder: PlanBuilder) -> None:
        """Test ANALYZE plan template (6 dimensions)."""
        plan = plan_builder.build(CodeIntent.ANALYZE)
        assert "ARCHITECTURE" in plan
        assert "CODE QUALITY" in plan
        assert "ALGORITHMIC" in plan
        assert "SECURITY" in plan
        assert "RELIABILITY" in plan
        assert "MAINTAINABILITY" in plan

    def test_refactor_plan(self, plan_builder: PlanBuilder) -> None:
        """Test REFACTOR plan template (4 phases)."""
        plan = plan_builder.build(CodeIntent.REFACTOR)
        assert "ANALYSIS" in plan
        assert "PLAN" in plan
        assert "IMPLEMENTATION" in plan
        assert "VERIFICATION" in plan

    def test_optimize_plan(self, plan_builder: PlanBuilder) -> None:
        """Test OPTIMIZE plan template (5 phases)."""
        plan = plan_builder.build(CodeIntent.OPTIMIZE)
        assert "BASELINE" in plan
        assert "PROFILING" in plan
        assert "OPTIMIZATION" in plan
        assert "BENCHMARK" in plan
        assert "TRADE-OFF" in plan

    def test_all_intents_have_plans(self, plan_builder: PlanBuilder) -> None:
        """Test that all 13 intents have plan templates."""
        for intent in CodeIntent:
            plan = plan_builder.build(intent)
            assert len(plan) > 0
            assert isinstance(plan, str)


# ─── Auto Repair Tests ───────────────────────────────────────────────────────


class TestAutoRepair:
    """Test auto-repair pattern fixes."""

    def test_no_fix_when_passed(self, auto_repair: AutoRepair) -> None:
        """Test no fix when verification passed."""
        code = "def foo(): return 1"
        verification = VerificationResult(
            passed=True,
            blocked=False,
            score=1.0,
            gate_results=(),
            formatted_report="",
            critical_count=0,
            error_count=0,
            warning_count=0,
        )
        fixed = auto_repair.attempt_fix(code, verification)
        assert fixed == code

    def test_fix_missing_return_type(self, auto_repair: AutoRepair) -> None:
        """Test fixing missing return type annotation."""
        code = "def foo(): return 1"
        verification = VerificationResult(
            passed=False,
            blocked=False,
            score=0.5,
            gate_results=(Mock(
                gate_name="TYPES",
                passed=False,
                findings=[Mock(rule_id="TYPES-002", message="foo")],
            ),),
            formatted_report="",
            critical_count=0,
            error_count=1,
            warning_count=0,
        )
        fixed = auto_repair.attempt_fix(code, verification)
        # Should add -> str placeholder
        assert "-> str" in fixed or "def foo() -> str:" in fixed

    def test_fix_missing_frozen_slots(self, auto_repair: AutoRepair) -> None:
        """Test fixing missing frozen=True, slots=True."""
        code = "@dataclass\nclass Foo:\n    x: int"
        verification = VerificationResult(
            passed=False,
            blocked=False,
            score=0.5,
            gate_results=(Mock(
                gate_name="STRUCTURE",
                passed=False,
                findings=[Mock(rule_id="STRUCT-001", message="Foo")],
            ),),
            formatted_report="",
            critical_count=0,
            error_count=1,
            warning_count=0,
        )
        fixed = auto_repair.attempt_fix(code, verification)
        # Should add frozen=True, slots=True
        assert "frozen=True" in fixed or "slots=True" in fixed

    def test_fix_missing_o_notation(self, auto_repair: AutoRepair) -> None:
        """Test fixing missing O() notation in docstrings."""
        code = 'def foo():\n    """Do something."""\n    return 1'
        verification = VerificationResult(
            passed=False,
            blocked=False,
            score=0.5,
            gate_results=(Mock(
                gate_name="COMPLEXITY",
                passed=False,
                findings=[Mock(rule_id="CPLX-001", message="foo")],
            ),),
            formatted_report="",
            critical_count=0,
            error_count=1,
            warning_count=0,
        )
        fixed = auto_repair.attempt_fix(code, verification)
        # Should add O() notation
        assert "O(" in fixed

    def test_no_fixable_findings(self, auto_repair: AutoRepair) -> None:
        """Test no fix when no fixable findings."""
        code = "def foo(): return 1"
        verification = VerificationResult(
            passed=False,
            blocked=False,
            score=0.5,
            gate_results=(Mock(
                gate_name="UNKNOWN",
                passed=False,
                findings=[],
            ),),
            formatted_report="",
            critical_count=0,
            error_count=1,
            warning_count=0,
        )
        fixed = auto_repair.attempt_fix(code, verification)
        assert fixed == code


# ─── State Machine Tests ─────────────────────────────────────────────────────


class TestStateMachine:
    """Test state machine transitions."""

    def test_initial_state(self, agent: CondroxAgentCoder) -> None:
        """Test initial agent state is IDLE."""
        assert agent.state == AgentState.IDLE

    def test_rag_index_size(self, agent: CondroxAgentCoder) -> None:
        """Test RAG index size property."""
        assert agent.rag_index_size == 0  # No files indexed yet

    def test_index_project(self, agent: CondroxAgentCoder) -> None:
        """Test project file indexing."""
        agent.index_project("test.py", "def foo(): return 1", "python")
        assert agent.rag_index_size > 0

    def test_empty_input_raises_value_error(self, agent: CondroxAgentCoder) -> None:
        """Test empty input raises ValueError."""
        import asyncio
        with pytest.raises(ValueError):
            asyncio.run(agent.execute("", Mock()))

    def test_whitespace_input_raises_value_error(self, agent: CondroxAgentCoder) -> None:
        """Test whitespace input raises ValueError."""
        import asyncio
        with pytest.raises(ValueError):
            asyncio.run(agent.execute("   ", Mock()))


# ─── Integration Tests ───────────────────────────────────────────────────────


class TestIntegration:
    """Test end-to-end integration with Condrox brain."""

    @pytest.mark.asyncio
    async def test_full_execute_flow(
        self,
        agent: CondroxAgentCoder,
        mock_brain_controller: Mock
    ) -> None:
        """Test full execute flow through state machine."""
        result = await agent.execute(
            "Generate a function",
            mock_brain_controller,
            language="python",
            file_path="output.py",
        )
        assert isinstance(result, AgentCodeResult)
        assert result.task_id != ""
        assert result.intent == CodeIntent.GENERATE
        assert result.output != ""
        assert isinstance(result.verification, VerificationResult)
        assert result.state in (AgentState.COMPLETED, AgentState.BLOCKED, AgentState.FAILED)

    @pytest.mark.asyncio
    async def test_execute_with_context(
        self,
        agent: CondroxAgentCoder,
        mock_brain_controller: Mock
    ) -> None:
        """Test execute with additional context."""
        result = await agent.execute(
            "Fix this bug",
            mock_brain_controller,
            context="This is in the payment module",
        )
        assert result.output != ""
        assert result.intent == CodeIntent.FIX

    @pytest.mark.asyncio
    async def test_execute_with_language(
        self,
        agent: CondroxAgentCoder,
        mock_brain_controller: Mock
    ) -> None:
        """Test execute with different language."""
        result = await agent.execute(
            "Generate a function",
            mock_brain_controller,
            language="typescript",
        )
        assert result.output != ""


# ─── Edge Case Tests ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_input(self, intent_classifier: IntentClassifier) -> None:
        """Test very long input handling."""
        long_input = "fix " * 1000
        intent, confidence = intent_classifier.classify(long_input)
        assert intent == CodeIntent.FIX
        assert confidence <= 1.0

    def test_special_characters(self, intent_classifier: IntentClassifier) -> None:
        """Test special characters in input."""
        intent, confidence = intent_classifier.classify("fix @#$%^&*()")
        assert intent == CodeIntent.FIX

    def test_unicode_characters(self, intent_classifier: IntentClassifier) -> None:
        """Test Unicode characters in input."""
        intent, confidence = intent_classifier.classify("fix 🐛 bug")
        assert intent == CodeIntent.FIX

    def test_mixed_case(self, intent_classifier: IntentClassifier) -> None:
        """Test mixed case input."""
        intent, confidence = intent_classifier.classify("FiX ThIs BuG")
        assert intent == CodeIntent.FIX

    def test_domain_threshold_boundary(self, domain_detector: DomainDetector) -> None:
        """Test domain threshold boundary."""
        # Score exactly at threshold
        result = domain_detector.detect("ml/model.py", "import tensorflow", "python")
        # Should detect AI_DEV if score >= 0.80
        assert result.domain in (DomainType.AI_DEV, DomainType.GENERAL)


# ─── Data Class Tests ───────────────────────────────────────────────────────


class TestDataClasses:
    """Test immutable data containers."""

    def test_code_task_immutability(self) -> None:
        """Test CodeTask is frozen."""
        task = CodeTask(
            task_id="test",
            intent=CodeIntent.GENERATE,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            domain=DomainType.GENERAL,
            user_input="test",
            language="python",
            file_path="test.py",
            context="",
            created_at=0.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            task.task_id = "changed"

    def test_verification_result_immutability(self) -> None:
        """Test VerificationResult is frozen."""
        result = VerificationResult(
            passed=True,
            blocked=False,
            score=1.0,
            gate_results=(),
            formatted_report="",
            critical_count=0,
            error_count=0,
            warning_count=0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.passed = False

    def test_agent_code_result_immutability(self) -> None:
        """Test AgentCodeResult is frozen."""
        result = AgentCodeResult(
            task_id="test",
            intent=CodeIntent.GENERATE,
            output="",
            verification=VerificationResult(
                passed=True,
                blocked=False,
                score=1.0,
                gate_results=(),
                formatted_report="",
                critical_count=0,
                error_count=0,
                warning_count=0,
            ),
            rag_context="",
            state=AgentState.COMPLETED,
            confidence=1.0,
            confidence_level=ConfidenceLevel.CERTAIN,
            domain=DomainType.GENERAL,
            latency_ms=100,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.state = AgentState.FAILED

    def test_domain_detection_immutability(self) -> None:
        """Test DomainDetection is frozen."""
        detection = DomainDetection(
            domain=DomainType.GENERAL,
            score=0.0,
            signals=(),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            detection.score = 0.5
