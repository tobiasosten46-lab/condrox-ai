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
Tests for AI-OS integration with BrainController pipeline and admin API endpoints.

Covers:
- Pipeline data flow channel writes during process_request
- /admin/security/events endpoint logic
- /admin/errors endpoint logic

Uses a shared session-scoped BrainController to avoid ~12s boot per test.

Complexity: T(n) ∈ O(p) for p pipeline stages per test.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Generator


def _run(coro):
    """Run an async coroutine from a synchronous pytest test."""
    return asyncio.run(coro)
from unittest.mock import patch

import pytest


# ─── Shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def controller() -> Generator["BrainController", None, None]:
    """Shared BrainController with AI-OS booted (one boot for all tests).

    Complexity: T(n) ∈ O(boot) once per module.
    """
    from core.brain.brain_controller import BrainController

    bc = BrainController()
    bc.boot_system()
    yield bc
    bc.close()


# ─── Pipeline data flow channel tests ──────────────────────────────────────────


class TestPipelineDataFlowIntegration:
    """Tests that pipeline stages publish to data flow channels."""

    @pytest.fixture(autouse=True)
    def _stop_data_flow_router(self, controller: "BrainController") -> None:
        """Stop the background router so channel items are not consumed."""
        controller.aios_kernel.data_flow.stop_router()

    def test_input_channel_receives_data(self, controller: "BrainController") -> None:
        """process_request publishes to 'input' channel when AI-OS is booted.

        Complexity: T(n) ∈ O(Σ s_i) for pipeline stages.
        """
        controller.process_request("Hello, what is 2+2?")
        df = controller.aios_kernel.data_flow
        item = df.get_nowait("input")
        assert item is not None
        assert "request_id" in item
        assert "raw_input" in item

    def test_output_channel_receives_data(self, controller: "BrainController") -> None:
        """process_request publishes to 'output' channel after pipeline.

        Complexity: T(n) ∈ O(Σ s_i) for pipeline stages.
        """
        controller.process_request("What is reasoning?")
        df = controller.aios_kernel.data_flow
        item = df.get_nowait("output")
        assert item is not None
        assert "final_output" in item
        assert "request_id" in item

    def test_intent_channel_receives_data(self, controller: "BrainController") -> None:
        """Pipeline publishes intent snapshot after Stage03.

        Complexity: T(n) ∈ O(Σ s_i) for pipeline stages.
        """
        controller.process_request("Explain machine learning")
        df = controller.aios_kernel.data_flow
        item = df.get_nowait("intent")
        assert item is not None
        assert "intent" in item
        assert "mode" in item

    def test_aios_kernel_is_booted(self, controller: "BrainController") -> None:
        """AI-OS kernel is booted and active after boot_system.

        Complexity: T(n) ∈ O(1).
        """
        assert controller.aios_kernel is not None
        assert controller.aios_kernel.is_booted is True

    def test_data_flow_has_all_channels(self, controller: "BrainController") -> None:
        """All 5 data flow channels exist: input, intent, context, reasoning, output.

        Complexity: T(n) ∈ O(1).
        """
        df = controller.aios_kernel.data_flow
        for ch in ("input", "intent", "context", "reasoning", "output"):
            assert ch in df._channels, f"Channel '{ch}' missing from data flow"


# ─── Admin security events endpoint tests ──────────────────────────────────────


class TestAdminSecurityEvents:
    """Tests for /admin/security/events endpoint logic."""

    def test_returns_empty_when_no_violations(self, controller: "BrainController") -> None:
        """Endpoint returns empty events list and zero stats when no violations.

        Complexity: T(n) ∈ O(1).
        """
        # Clear audit log to ensure clean state
        controller.safety_layer._audit_log.clear()

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_security_events

            result = _run(list_security_events())
        assert result["count"] == 0
        assert result["events"] == []
        assert result["stats"] == {"pii": 0, "secret": 0, "injection": 0, "ethics": 0}

    def test_returns_events_after_violation(self, controller: "BrainController") -> None:
        """Endpoint returns events after a safety violation is triggered.

        Complexity: T(n) ∈ O(n) for n audit log entries.
        """
        controller.safety_layer._audit_log.clear()

        # Trigger violence rule (harm_violence pattern: bomb)
        controller.safety_layer.check("this is a bomb threat")
        violations = controller.safety_layer.get_audit_log(limit=10)
        assert len(violations) > 0

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_security_events

            result = _run(list_security_events())
        assert result["count"] >= 1
        assert len(result["events"]) >= 1
        assert "stats" in result
        # The violation should be classified as injection (harm_violence → injection)
        evt = result["events"][0]
        assert evt["type"] in ("pii", "secret", "injection", "ethics")
        assert evt["action"] in ("blocked", "warning", "approved")

    def test_filter_by_type(self, controller: "BrainController") -> None:
        """Event type filter works correctly.

        Complexity: T(n) ∈ O(n) for n audit log entries.
        """
        controller.safety_layer._audit_log.clear()
        # Trigger email PII rule (pii_email)
        controller.safety_layer.check("contact me at test@example.com")

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_security_events

            result = _run(list_security_events(event_type="pii"))
        assert result["count"] >= 1
        for evt in result["events"]:
            assert evt["type"] == "pii"


# ─── Admin errors endpoint tests ───────────────────────────────────────────────


class TestAdminErrors:
    """Tests for /admin/errors endpoint logic."""

    def test_returns_empty_when_no_errors(self, controller: "BrainController") -> None:
        """Endpoint returns empty errors list when no errors logged.

        Complexity: T(n) ∈ O(1).
        """
        controller.error_recovery._error_log.clear()

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_errors

            result = _run(list_errors())
        assert result["count"] == 0
        assert result["errors"] == []
        assert result["recovered"] == 0

    def test_returns_errors_after_log_entry(self, controller: "BrainController") -> None:
        """Endpoint returns errors after an error is logged.

        Complexity: T(n) ∈ O(n) for n error log entries.
        """
        from core.conversational.error_recovery import ErrorRecord

        controller.error_recovery._error_log.clear()
        controller.error_recovery._error_log.append(
            ErrorRecord(
                error_id="test-err-001",
                stage_name="Stage05Reasoning",
                error_type="RuntimeError",
                error_message="Test error in reasoning stage",
                timestamp=12345.0,
            )
        )

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_errors

            result = _run(list_errors())
        assert result["count"] == 1
        assert result["errors"][0]["stage"] == "Stage05Reasoning"
        assert result["errors"][0]["exception"] == "Test error in reasoning stage"
        assert result["errors"][0]["severity"] == "INFO"
        assert result["errors"][0]["recovery_status"] == "none"

    def test_severity_classification_critical(self, controller: "BrainController") -> None:
        """Failed recovery → CRITICAL severity.

        Complexity: T(n) ∈ O(1).
        """
        from core.conversational.error_recovery import (
            ErrorRecord,
            RecoveryResult,
            RecoveryType,
        )

        controller.error_recovery._error_log.clear()
        controller.error_recovery._error_log.append(
            ErrorRecord(
                error_id="test-err-crit",
                stage_name="Stage07Memory",
                error_type="FatalError",
                error_message="Memory corruption",
                timestamp=99999.0,
                recovery_attempted=True,
                recovery_result=RecoveryResult(
                    success=False,
                    recovery_type=RecoveryType.HARD,
                    message="Recovery failed",
                    retry_recommended=False,
                ),
            )
        )

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_errors

            result = _run(list_errors())
        assert result["count"] == 1
        assert result["errors"][0]["severity"] == "CRITICAL"
        assert result["errors"][0]["recovery_status"] == "failed"
        assert result["stats"]["CRITICAL"] == 1

    def test_filter_by_stage(self, controller: "BrainController") -> None:
        """Stage filter works correctly.

        Complexity: T(n) ∈ O(n) for n error log entries.
        """
        from core.conversational.error_recovery import ErrorRecord

        controller.error_recovery._error_log.clear()
        controller.error_recovery._error_log.append(
            ErrorRecord(
                error_id="err-a", stage_name="StageA", error_type="E", error_message="A", timestamp=1.0,
            )
        )
        controller.error_recovery._error_log.append(
            ErrorRecord(
                error_id="err-b", stage_name="StageB", error_type="E", error_message="B", timestamp=2.0,
            )
        )

        with patch("api.main.current_controller", controller):
            from api.admin_routes import list_errors

            result = _run(list_errors(stage="StageA"))
        assert result["count"] == 1
        assert result["errors"][0]["stage"] == "StageA"
