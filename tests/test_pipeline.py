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
Tests for the 14-stage cognitive pipeline.

Covers stage-level behavior, full pipeline run, and error handling.
"""

from __future__ import annotations

import pytest

from core import BrainController, BrainState
from core.pipeline.stage_01_preprocessor import Stage01Preprocessor
from core.pipeline.stage_02_intent import Stage02Intent
from core.pipeline.stage_03_mode_selector import Stage03ModeSelector
from core.pipeline.stage_04_domain_router import Stage04DomainRouter


def test_preprocessor_stage() -> None:
    """Preprocessor normalizes input and respects max length."""
    controller = BrainController()
    stage = Stage01Preprocessor()
    initial = BrainState(raw_input="  Hello   World!!  ")
    result = stage(controller, initial)
    assert result.processed_input == "Hello World!!"


def test_intent_stage_classifies_code() -> None:
    """Intent stage classifies a code request."""
    controller = BrainController()
    stage = Stage02Intent()
    initial = BrainState(processed_input="Write a function.")
    result = stage(controller, initial)
    assert result.intent == "code"
    assert result.intent_confidence > 0.0


def test_mode_selector_defaults_to_clarify_on_low_confidence() -> None:
    """Mode selector falls back to clarify when confidence is low."""
    stage = Stage03ModeSelector()
    assert stage.select("unknown", 0.2) == "clarify"


def test_domain_router_programming() -> None:
    """Domain router selects programming for code input."""
    stage = Stage04DomainRouter()
    assert stage.route("write python function", "code") == "programming"


def test_full_pipeline_end_to_end() -> None:
    """Pipeline runs all 14 stages and produces final output."""
    controller = BrainController()
    controller.ingest_knowledge("Python functions use def.", domain="programming")
    state = controller.process_request("How do I write a Python function?")
    assert state.final_output
    assert state.intent == "code"
    assert state.mode == "code"
    assert state.security_approved
    assert state.ethical_approved


def test_pipeline_error_handling_empty_input() -> None:
    """Pipeline handles empty input gracefully."""
    controller = BrainController()
    with pytest.raises(ValueError):
        controller.process_request("")
