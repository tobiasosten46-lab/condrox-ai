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
Tests for Condrox AI Full System Verification (Phase 1).

Covers:
- SystemProbe checks return valid CheckResult objects
- ScenarioEngine loads scenario files
- run_verification produces a report with expected keys
"""

from __future__ import annotations

import os

import pytest

from core.brain.brain_controller import BrainController
from core.verification import ScenarioEngine, SystemProbe
from core.verification.report import CheckResult, VerificationReport
from scripts.verify_system import run_verification


@pytest.fixture(scope="module")
def booted_controller() -> BrainController:
    """Boot a single BrainController shared by all verification tests."""
    os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")
    bc = BrainController()
    bc.boot_system()
    yield bc
    bc.close()


@pytest.fixture
def probe(booted_controller: BrainController) -> SystemProbe:
    """Provide a SystemProbe that reuses the shared controller."""
    return SystemProbe(
        api_host="localhost",
        api_port=8000,
        brain_controller=booted_controller,
    )


class TestSystemProbe:
    """Tests for individual system probes."""

    def test_wiring_check_passes(self, probe: SystemProbe) -> None:
        result = probe.probe_all()["wiring"]
        assert isinstance(result, CheckResult)
        assert result.status == "PASS"
        assert "Core imports load" in result.message

    def test_brain_controller_boot_passes(self, probe: SystemProbe) -> None:
        result = probe.probe_all()["brain_controller"]
        assert isinstance(result, CheckResult)
        assert result.status == "PASS"
        assert result.details["success"] is True

    def test_pipeline_stages_check_passes(self, probe: SystemProbe) -> None:
        result = probe.probe_all()["pipeline_stages"]
        assert isinstance(result, CheckResult)
        assert result.status == "PASS"
        assert result.details["stage_count"] == 20

    def test_agent_registry_check_passes(self, probe: SystemProbe) -> None:
        result = probe.probe_all()["agent_registry"]
        assert isinstance(result, CheckResult)
        assert result.status == "PASS"
        assert result.details["count"] >= 5


class TestScenarioEngine:
    """Tests for the scenario engine."""

    def test_loads_scenarios(self) -> None:
        engine = ScenarioEngine()
        assert engine.count() > 0
        scenarios = engine.list_scenarios()
        categories = {s.category for s in scenarios}
        assert categories >= {"conversation", "intelligence", "agents"}


class TestVerificationRunner:
    """Tests for the end-to-end verification runner."""

    def test_run_verification_report(self, monkeypatch) -> None:
        monkeypatch.setenv("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")
        report = run_verification("localhost", 8000)
        assert isinstance(report, VerificationReport)
        assert 0.0 <= report.score <= 100.0
        check_names = {c.name for c in report.checks}
        expected = {
            "System Wiring",
            "API Health",
            "BrainController Boot",
            "Pipeline Stages",
            "Agent Registry",
            "Scenario Registry",
        }
        assert expected <= check_names
