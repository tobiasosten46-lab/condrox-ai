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
Tests for the unified BrainController.

Covers boot, health, request processing, knowledge ingestion, memory
management, and subsystem wiring.
"""

from __future__ import annotations

import pytest

from core import BrainConfig, BrainController, BrainState


@pytest.fixture
def controller() -> BrainController:
    """Provide a fresh BrainController for each test."""
    return BrainController()


def test_brain_config_default() -> None:
    """BrainConfig defaults are valid."""
    config = BrainConfig.default()
    assert config.system_name == "Condrox AI"
    assert config.max_input_length > 0


def test_brain_controller_instantiation(controller: BrainController) -> None:
    """BrainController wires all subsystems on creation."""
    assert controller.memory_manager is not None
    assert controller.knowledge_router is not None
    assert controller.reasoning_engine is not None
    assert controller.pipeline.stage_count() == 20


def test_boot_system(controller: BrainController) -> None:
    """Boot checks run and report success."""
    report = controller.boot_system()
    assert report.success
    assert "numpy" in [c.split(":")[0] for c in report.checks]


def test_health_aggregate(controller: BrainController) -> None:
    """Aggregate health is a valid score."""
    health = controller.health()
    assert 0.0 <= health.score <= 1.0


def test_ingest_knowledge(controller: BrainController) -> None:
    """Knowledge can be ingested and retrieved."""
    item_id = controller.ingest_knowledge("Python is a language.", domain="programming")
    assert item_id
    assert controller.knowledge_router.backend.get(item_id) is not None


def test_add_memory(controller: BrainController) -> None:
    """Memory can be added and searched."""
    item_id = controller.add_memory("User likes Python.", tags=["preference"])
    assert item_id
    results = controller.memory_manager.search("Python", top_k=3)
    assert any(item.item_id == item_id for item in results)


def test_process_request_code(controller: BrainController) -> None:
    """Pipeline processes a code request end-to-end."""
    controller.ingest_knowledge("Functions in Python use the def keyword.", domain="programming")
    state = controller.process_request("Write a Python function.")
    assert isinstance(state, BrainState)
    assert state.final_output
    assert state.mode == "code"
    assert state.security_approved
    assert state.ethical_approved


def test_process_request_explain(controller: BrainController) -> None:
    """Pipeline processes an explain request end-to-end."""
    state = controller.process_request("Explain Python.")
    assert state.final_output
    assert state.mode == "explain"
    assert state.intent == "explain"


def test_empty_input_raises(controller: BrainController) -> None:
    """Empty input raises ValueError."""
    with pytest.raises(ValueError):
        controller.process_request("")


def test_status_snapshot(controller: BrainController) -> None:
    """Status snapshot reflects subsystem state."""
    status = controller.status()
    assert status["pipeline_stages"] == 20
    assert status["version"] == "3.4.0"


def test_lazy_creative_stack(controller: BrainController) -> None:
    """Creative stack components are lazy-initialized."""
    assert controller.emotion_palette_mapper is None
    assert controller.music_prompt_builder is None
    # quality_guardrail is eager (hot path)
    assert controller.quality_guardrail is not None
    # Trigger lazy init
    controller._ensure_creative()
    assert controller.emotion_palette_mapper is not None
    assert controller.music_prompt_builder is not None


def test_lazy_engines(controller: BrainController) -> None:
    """HWA and PBE engines are lazy-initialized."""
    assert controller.hwa_engine is None
    assert controller.pbe_engine is None
    # Trigger lazy init
    controller._ensure_engines()
    assert controller.hwa_engine is not None
    assert controller.pbe_engine is not None
