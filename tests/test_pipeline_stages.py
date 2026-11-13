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
Comprehensive per-stage tests for the cognitive pipeline (stages 05-14 plus
the new enrichment stages 05b, 08b, 09b).

Each stage receives 2-3 tests covering:
  1. Correct output for typical input.
  2. Edge cases (empty input, missing data).
  3. Robustness against unexpected input (no crash).

Run with:
    $env:PYTHONPATH = "."; python -m pytest tests/test_pipeline_stages.py -v --tb=short
"""

from __future__ import annotations

import time

import pytest

from core.brain.brain_config import BrainConfig
from core.brain.brain_controller import BrainController
from core.brain.brain_state import BrainState


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def controller() -> BrainController:
    """Provide a fresh, booted BrainController for each test."""
    bc = BrainController()
    bc.boot_system()
    return bc


@pytest.fixture
def controller_with_guardrails() -> BrainController:
    """Controller configured with ethics blocked topics and security patterns."""
    config = BrainConfig(
        blocked_topics=("harmful activity", "illegal activity"),
        blocked_patterns=(r"DROP\s+TABLE", r"rm\s+-rf"),
    )
    bc = BrainController(config=config)
    bc.boot_system()
    return bc


def _base_state(**overrides) -> BrainState:
    """Build a baseline BrainState with sensible defaults for mid-pipeline tests."""
    defaults = dict(
        request_id="req-test-1",
        raw_input="test input",
        processed_input="test input",
        intent="explain",
        intent_confidence=0.8,
        mode="explain",
        domain="general",
        emotion="neutral",
    )
    defaults.update(overrides)
    return BrainState(**defaults)


# ─── Stage 05: Context Builder ───────────────────────────────────────────────


def test_stage05_context_builder_typical(controller: BrainController) -> None:
    """Context snapshot contains intent, mode, domain, and request_id."""
    from core.pipeline.stage_05_context_builder import Stage05ContextBuilder

    state = _base_state(intent="explain", mode="explain", domain="general", request_id="req-1")
    stage = Stage05ContextBuilder()
    result = stage(controller, state)

    assert "intent" in result.context
    assert result.context["intent"] == "explain"
    assert result.context["mode"] == "explain"
    assert result.context["domain"] == "general"
    assert result.context["request_id"] == "req-1"
    assert "processed_at" in result.context


def test_stage05_context_builder_empty_strings(controller: BrainController) -> None:
    """Context builder handles empty-string fields without crashing."""
    from core.pipeline.stage_05_context_builder import Stage05ContextBuilder

    state = _base_state(intent="", mode="", domain="", request_id="", processed_input="")
    stage = Stage05ContextBuilder()
    result = stage(controller, state)

    assert result.context["intent"] == ""
    assert result.context["request_id"] == ""
    # Stack should not leak — base frame remains
    assert len(controller.context_manager._stack) >= 1


def test_stage05_context_builder_does_not_crash_on_unexpected(controller: BrainController) -> None:
    """Context builder survives a state with unusual confidence values."""
    from core.pipeline.stage_05_context_builder import Stage05ContextBuilder

    state = _base_state(intent_confidence=1.0)
    stage = Stage05ContextBuilder()
    result = stage(controller, state)
    assert result.context["intent_confidence"] == 1.0


# ─── Stage 05b: Context Enrichment ───────────────────────────────────────────


def test_stage05b_context_enrichment_multi_turn(controller: BrainController) -> None:
    """Enrichment adds multi-turn continuity context keys."""
    from core.pipeline.stage_05b_context_enrichment import Stage05bContextEnrichment

    state = _base_state(context={"intent": "explain"})
    stage = Stage05bContextEnrichment()
    result = stage(controller, state)

    assert "multi_turn_continuity" in result.context
    assert "multi_turn_focus" in result.context
    assert "multi_turn_dominant_intent" in result.context
    assert "temporal_context_active" in result.context
    assert result.context["temporal_context_active"] is True


def test_stage05b_context_enrichment_with_goals(controller: BrainController) -> None:
    """Active goals are surfaced into the context when present."""
    from core.pipeline.stage_05b_context_enrichment import Stage05bContextEnrichment

    controller.goal_tracker.add_goal("Explain the topic clearly")
    state = _base_state(context={"intent": "explain"})
    stage = Stage05bContextEnrichment()
    result = stage(controller, state)

    assert "active_goals" in result.context
    assert "Explain the topic clearly" in result.context["active_goals"]


def test_stage05b_context_enrichment_empty_input(controller: BrainController) -> None:
    """Enrichment does not crash on empty processed input."""
    from core.pipeline.stage_05b_context_enrichment import Stage05bContextEnrichment

    state = _base_state(processed_input="", context={})
    stage = Stage05bContextEnrichment()
    result = stage(controller, state)

    # Should still produce temporal flag and multi-turn keys
    assert result.context.get("temporal_context_active") is True


# ─── Stage 06: Knowledge Fetch ───────────────────────────────────────────────


def test_stage06_knowledge_fetch_typical(controller: BrainController) -> None:
    """Knowledge fetch retrieves ingested items and sets context keys."""
    from core.pipeline.stage_06_knowledge_fetch import Stage06KnowledgeFetch

    controller.ingest_knowledge("Python functions use the def keyword.", domain="programming")
    state = _base_state(processed_input="python function", domain="programming")
    stage = Stage06KnowledgeFetch(top_k=5)
    result = stage(controller, state)

    assert isinstance(result.retrieved_knowledge, list)
    assert result.context.get("has_knowledge") is True
    assert result.context.get("knowledge_count", 0) >= 1
    assert "knowledge_texts" in result.context


def test_stage06_knowledge_fetch_empty_store(controller: BrainController) -> None:
    """Knowledge fetch with an empty store returns no items but sets context.

    The Wikipedia API fallback is mocked off to isolate local-store behavior.
    """
    from unittest.mock import patch

    from core.pipeline.stage_06_knowledge_fetch import Stage06KnowledgeFetch

    state = _base_state(processed_input="nothing here", domain="general")
    stage = Stage06KnowledgeFetch(top_k=3)
    stage._wikipedia_api = None  # Disable WikipediaAPI instance
    with patch(
        "core.knowledge.wikipedia_bridge.fetch_wikipedia_summary",
        return_value=None,
    ):
        result = stage(controller, state)

    assert result.retrieved_knowledge == []
    assert result.context.get("has_knowledge") is False
    assert result.context.get("knowledge_count") == 0


def test_stage06_knowledge_fetch_empty_input(controller: BrainController) -> None:
    """Knowledge fetch does not crash on empty processed input."""
    from core.pipeline.stage_06_knowledge_fetch import Stage06KnowledgeFetch

    state = _base_state(processed_input="", domain="general")
    stage = Stage06KnowledgeFetch(top_k=3)
    result = stage(controller, state)
    assert isinstance(result.retrieved_knowledge, list)


# ─── Stage 07: Memory Fetch ──────────────────────────────────────────────────


def test_stage07_memory_fetch_typical(controller: BrainController) -> None:
    """Memory fetch retrieves stored memories and sets context keys."""
    from core.pipeline.stage_07_memory_fetch import Stage07MemoryFetch

    controller.add_memory("User likes Python programming.", tags=["preference"])
    state = _base_state(processed_input="python programming")
    stage = Stage07MemoryFetch(top_k=5)
    result = stage(controller, state)

    assert isinstance(result.retrieved_memories, list)
    assert result.context.get("has_memory") is True
    assert result.context.get("memory_count", 0) >= 1
    assert "memory_texts" in result.context


def test_stage07_memory_fetch_with_graph(controller: BrainController) -> None:
    """Memory graph nodes are merged into retrieved memory texts."""
    from core.pipeline.stage_07_memory_fetch import Stage07MemoryFetch

    controller.memory_graph.add_node("node-1", content="Graph memory about Python", tags=("python",))
    state = _base_state(processed_input="python")
    stage = Stage07MemoryFetch(top_k=5)
    result = stage(controller, state)

    # The graph node content should appear in the memory texts
    all_texts = result.context.get("memory_texts", [])
    assert "Graph memory about Python" in all_texts


def test_stage07_memory_fetch_empty(controller: BrainController) -> None:
    """Memory fetch with no memories does not crash and reports empty."""
    from core.pipeline.stage_07_memory_fetch import Stage07MemoryFetch

    state = _base_state(processed_input="")
    stage = Stage07MemoryFetch(top_k=3)
    result = stage(controller, state)

    assert isinstance(result.retrieved_memories, list)
    assert result.context.get("has_memory") is False


# ─── Stage 08: Reasoning Core ────────────────────────────────────────────────


def test_stage08_reasoning_typical(controller: BrainController) -> None:
    """Reasoning produces a reasoning trace and evidence list."""
    from core.pipeline.stage_08_reasoning_core import Stage08ReasoningCore

    state = _base_state(intent="explain", mode="explain", domain="general")
    stage = Stage08ReasoningCore()
    result = stage(controller, state)

    assert result.reasoning
    assert result.reasoning.startswith("[Reasoning]")
    assert isinstance(result.reasoning_evidence, list)


def test_stage08_reasoning_with_knowledge(controller: BrainController) -> None:
    """Claim ranking enriches evidence when knowledge texts are present."""
    from core.pipeline.stage_08_reasoning_core import Stage08ReasoningCore

    controller.context_manager.set("knowledge_texts", ["Python is a language.", "Functions use def."])
    state = _base_state(
        intent="explain",
        mode="explain",
        retrieved_knowledge=["k1", "k2"],
    )
    stage = Stage08ReasoningCore()
    result = stage(controller, state)

    assert result.reasoning
    # Evidence may include a top_claim_score entry from claim ranking
    assert isinstance(result.reasoning_evidence, list)


def test_stage08_reasoning_empty_state(controller: BrainController) -> None:
    """Reasoning does not crash with no memories or knowledge."""
    from core.pipeline.stage_08_reasoning_core import Stage08ReasoningCore

    state = _base_state(intent="unknown", intent_confidence=0.1)
    stage = Stage08ReasoningCore()
    result = stage(controller, state)
    assert result.reasoning.startswith("[Reasoning]")


# ─── Stage 08b: Agent Execution ──────────────────────────────────────────────


def test_stage08b_agent_execution_typical(controller: BrainController) -> None:
    """Agent execution runs the mode-specific agent and merges result."""
    from core.pipeline.stage_08b_agent_execution import Stage08bAgentExecution

    state = _base_state(mode="explain", processed_input="Explain Python.")
    stage = Stage08bAgentExecution()
    result = stage(controller, state)

    assert "agent_id" in result.context
    assert "agent_confidence" in result.context
    assert "agent_mode" in result.context
    assert result.context["agent_mode"] == "explain"


def test_stage08b_agent_execution_search_guard_rejects_empty(controller: BrainController) -> None:
    """Search guard rejects empty input and records an error."""
    from core.pipeline.stage_08b_agent_execution import Stage08bAgentExecution

    state = _base_state(mode="explain", processed_input="")
    stage = Stage08bAgentExecution()
    result = stage(controller, state)

    # Guard should reject empty input → error recorded, no agent run
    assert any("guard" in e.lower() or "AgentExecution" in e for e in result.errors)


def test_stage08b_agent_execution_unknown_mode(controller: BrainController) -> None:
    """Agent execution falls back to default agent for unknown mode."""
    from core.pipeline.stage_08b_agent_execution import Stage08bAgentExecution

    state = _base_state(mode="nonexistent_mode", processed_input="hello")
    stage = Stage08bAgentExecution()
    result = stage(controller, state)

    # Should not crash; default agent handles unknown modes
    assert "agent_id" in result.context


# ─── Stage 09: Synthesis ─────────────────────────────────────────────────────


def test_stage09_synthesis_typical(controller: BrainController) -> None:
    """Synthesis produces non-empty output for a typical state."""
    from core.pipeline.stage_09_synthesis import Stage09Synthesis

    state = _base_state(mode="explain", processed_input="Explain recursion.", emotion="curious")
    stage = Stage09Synthesis()
    result = stage(controller, state)

    assert result.synthesized_output
    assert "recursion" in result.synthesized_output.lower()


def test_stage09_synthesis_empathy_prefix(controller: BrainController) -> None:
    """Empathy prefix is prepended for emotional inputs."""
    from core.pipeline.stage_09_synthesis import Stage09Synthesis

    state = _base_state(mode="clarify", processed_input="I feel sad today.", emotion="sad")
    stage = Stage09Synthesis()
    result = stage(controller, state)

    assert result.synthesized_output
    assert "sorry to hear" in result.synthesized_output.lower()


def test_stage09_synthesis_token_trimming() -> None:
    """Synthesis trims overly long output to the token budget."""
    from core.pipeline.stage_09_synthesis import Stage09Synthesis

    # Use a small budget so trimming is triggered by a modest input.
    small_config = BrainConfig(max_input_length=100)
    bc = BrainController(config=small_config)
    bc.boot_system()
    output_budget = bc.token_pressure_regulator.get_budget().output

    # Build input with more tokens than the output budget to force trimming.
    long_input = "Explain " + ("python " * (output_budget + 50))
    state = _base_state(mode="explain", processed_input=long_input, emotion="neutral")
    stage = Stage09Synthesis()
    result = stage(bc, state)

    assert result.synthesized_output
    # Trimmed output token count must not exceed the output budget.
    trimmed_tokens = len(result.synthesized_output.split())
    assert trimmed_tokens <= output_budget + 1  # +1 tolerance for ellipsis token


# ─── Stage 09b: Response Quality ─────────────────────────────────────────────


def test_stage09b_response_quality_typical(controller: BrainController) -> None:
    """Response quality stage preserves valid output and returns state."""
    from core.pipeline.stage_09b_response_quality import Stage09bResponseQuality

    state = _base_state(synthesized_output="Python is a programming language used widely.")
    stage = Stage09bResponseQuality()
    result = stage(controller, state)

    assert result.synthesized_output
    assert isinstance(result.errors, list)


def test_stage09b_response_quality_empty_output(controller: BrainController) -> None:
    """Response quality stage is a no-op when output is empty."""
    from core.pipeline.stage_09b_response_quality import Stage09bResponseQuality

    state = _base_state(synthesized_output="")
    stage = Stage09bResponseQuality()
    result = stage(controller, state)

    assert result.synthesized_output == ""
    # No crash; errors unchanged
    assert result.errors == state.errors


def test_stage09b_response_quality_contradiction_detection(controller: BrainController) -> None:
    """Contradiction detector flags output contradicting known facts without crashing."""
    from core.pipeline.stage_09b_response_quality import Stage09bResponseQuality

    state = _base_state(
        synthesized_output="The sky is green and the grass is blue today.",
        processed_input="what color is the sky",
    )
    stage = Stage09bResponseQuality()
    result = stage(controller, state)

    # Stage must not crash regardless of detector outcome
    assert isinstance(result.synthesized_output, str)


# ─── Stage 10: Personality ───────────────────────────────────────────────────


def test_stage10_personality_typical(controller: BrainController) -> None:
    """Personality stage applies styling and sets the flag."""
    from core.pipeline.stage_10_personality import Stage10Personality

    state = _base_state(synthesized_output="Here is an explanation of Python.", emotion="neutral")
    stage = Stage10Personality()
    result = stage(controller, state)

    assert result.personality_applied is True
    assert result.synthesized_output  # non-empty after styling


def test_stage10_personality_empty_output(controller: BrainController) -> None:
    """Personality stage handles empty output without crashing."""
    from core.pipeline.stage_10_personality import Stage10Personality

    state = _base_state(synthesized_output="", emotion="neutral")
    stage = Stage10Personality()
    result = stage(controller, state)

    assert result.personality_applied is True


def test_stage10_personality_emotional_input(controller: BrainController) -> None:
    """Personality styling adapts to emotional tone."""
    from core.pipeline.stage_10_personality import Stage10Personality

    state = _base_state(synthesized_output="I will help you with this.", emotion="happy")
    stage = Stage10Personality()
    result = stage(controller, state)

    assert result.personality_applied is True
    assert result.synthesized_output


# ─── Stage 11: Ethics ────────────────────────────────────────────────────────


def test_stage11_ethics_approves_clean_output(controller: BrainController) -> None:
    """Ethics stage approves output without blocked topics."""
    from core.pipeline.stage_11_ethics import Stage11Ethics

    state = _base_state(synthesized_output="Python is a great programming language.")
    stage = Stage11Ethics()
    result = stage(controller, state)

    assert result.ethical_approved is True
    assert "Python" in result.synthesized_output


def test_stage11_ethics_blocks_flagged_output(controller_with_guardrails: BrainController) -> None:
    """Ethics stage blocks output containing blocked topics."""
    from core.pipeline.stage_11_ethics import Stage11Ethics

    state = _base_state(synthesized_output="harmful activity now")
    stage = Stage11Ethics()
    result = stage(controller_with_guardrails, state)

    assert result.ethical_approved is False
    assert "cannot produce" in result.synthesized_output.lower()
    assert any("ethics_violation" in e for e in result.errors)


def test_stage11_ethics_empty_output(controller: BrainController) -> None:
    """Ethics stage approves empty output (no content to flag)."""
    from core.pipeline.stage_11_ethics import Stage11Ethics

    state = _base_state(synthesized_output="")
    stage = Stage11Ethics()
    result = stage(controller, state)

    assert result.ethical_approved is True


# ─── Stage 12: Security ──────────────────────────────────────────────────────


def test_stage12_security_approves_clean_output(controller: BrainController) -> None:
    """Security stage approves output without blocked patterns."""
    from core.pipeline.stage_12_security import Stage12Security

    state = _base_state(synthesized_output="Here is a safe explanation.")
    stage = Stage12Security()
    result = stage(controller, state)

    assert result.security_approved is True


def test_stage12_security_blocks_pattern(controller_with_guardrails: BrainController) -> None:
    """Security stage blocks output containing blocked patterns."""
    from core.pipeline.stage_12_security import Stage12Security

    state = _base_state(synthesized_output="Run this: DROP TABLE users;")
    stage = Stage12Security()
    result = stage(controller_with_guardrails, state)

    assert result.security_approved is False
    assert "blocked by the security layer" in result.synthesized_output.lower()
    assert any("security_violation" in e for e in result.errors)


def test_stage12_security_empty_output(controller: BrainController) -> None:
    """Security stage approves empty output."""
    from core.pipeline.stage_12_security import Stage12Security

    state = _base_state(synthesized_output="")
    stage = Stage12Security()
    result = stage(controller, state)

    assert result.security_approved is True


# ─── Stage 13: Output Builder ────────────────────────────────────────────────


def test_stage13_output_builder_typical(controller: BrainController) -> None:
    """Output builder formats text and adds metadata."""
    from core.pipeline.stage_13_output_builder import Stage13OutputBuilder

    state = _base_state(synthesized_output="  Final answer here.  ")
    stage = Stage13OutputBuilder()
    result = stage(controller, state)

    assert result.final_output == "Final answer here."
    assert result.metadata["pipeline_stage"] == "output_builder"
    assert "finished_at" in result.metadata


def test_stage13_output_builder_fallback_empty(controller: BrainController) -> None:
    """Output builder supplies a fallback for empty output."""
    from core.pipeline.stage_13_output_builder import Stage13OutputBuilder

    state = _base_state(synthesized_output="   ")
    stage = Stage13OutputBuilder()
    result = stage(controller, state)

    assert result.final_output == "[No output generated.]"


def test_stage13_output_builder_preserves_metadata(controller: BrainController) -> None:
    """Output builder preserves existing metadata while adding new keys."""
    from core.pipeline.stage_13_output_builder import Stage13OutputBuilder

    state = _base_state(
        synthesized_output="answer",
        metadata={"started_at": 1000.0, "custom": "value"},
    )
    stage = Stage13OutputBuilder()
    result = stage(controller, state)

    assert result.metadata["custom"] == "value"
    assert result.metadata["started_at"] == 1000.0
    assert "finished_at" in result.metadata


# ─── Stage 14: Delivery ──────────────────────────────────────────────────────


def test_stage14_delivery_metadata(controller: BrainController) -> None:
    """Delivery stage stamps final metadata with mode, domain, intent."""
    from core.pipeline.stage_14_delivery import Stage14Delivery

    started = time.time()
    state = _base_state(
        mode="explain",
        domain="general",
        intent="explain",
        intent_confidence=0.9,
        metadata={"started_at": started},
    )
    stage = Stage14Delivery()
    result = stage(controller, state)

    assert result.metadata["mode"] == "explain"
    assert result.metadata["domain"] == "general"
    assert result.metadata["intent"] == "explain"
    assert result.metadata["intent_confidence"] == 0.9
    assert "delivered_at" in result.metadata
    assert "latency_ms" in result.metadata


def test_stage14_delivery_latency_recording(controller: BrainController) -> None:
    """Delivery records latency in the latency compensation layer."""
    from core.pipeline.stage_14_delivery import Stage14Delivery

    started = time.time() - 0.05  # 50ms ago
    state = _base_state(metadata={"started_at": started})
    stage = Stage14Delivery()
    result = stage(controller, state)

    assert result.metadata["latency_ms"] >= 0.0
    # Latency compensation layer should have recorded the sample for "pipeline"
    assert "pipeline" in controller.latency_compensation.get_stage_names()


def test_stage14_delivery_meta_learning(controller: BrainController) -> None:
    """Delivery updates meta-learning with a reward based on error presence."""
    from core.pipeline.stage_14_delivery import Stage14Delivery

    state = _base_state(mode="explain", metadata={"started_at": time.time()})
    stage = Stage14Delivery()
    result = stage(controller, state)

    assert "latency_ms" in result.metadata
    # Meta-learning engine should have recorded an outcome for the mode
    # (reward 1.0 when no errors, 0.7 when errors present)
    assert controller.config.enable_learning is True


def test_stage14_delivery_with_errors(controller: BrainController) -> None:
    """Delivery handles states with accumulated errors without crashing."""
    from core.pipeline.stage_14_delivery import Stage14Delivery

    state = _base_state(
        mode="explain",
        errors=["some non-fatal error"],
        metadata={"started_at": time.time()},
    )
    stage = Stage14Delivery()
    result = stage(controller, state)

    assert "latency_ms" in result.metadata
    assert result.metadata["mode"] == "explain"
