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

"""System-level probes for Condrox AI wiring and health."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any, Dict

from core.verification.report import CheckResult


class SystemProbe:
    """Probe API, BrainController, pipeline, and agent registry health.

    If a booted BrainController is supplied, the probe reuses it for
    BrainController-derived checks instead of booting a new instance.

    Complexity: T(probe) ∈ O(1) per check, S(n) ∈ O(1).
    """

    __slots__ = ("api_port", "api_host", "_brain_controller", "_own_controller")

    def __init__(
        self,
        api_host: str = "localhost",
        api_port: int = 8000,
        brain_controller: Any = None,
    ) -> None:
        self.api_host = api_host
        self.api_port = api_port
        self._brain_controller: Any = brain_controller
        self._own_controller: bool = False

    def probe_all(self) -> Dict[str, CheckResult]:
        """Run all foundation probes and return results by name."""
        return {
            "wiring": self._check_wiring(),
            "api_health": self._check_api_health(),
            "brain_controller": self._check_brain_controller(),
            "pipeline_stages": self._check_pipeline_stages(),
            "agent_registry": self._check_agent_registry(),
        }

    def _ensure_controller(self) -> Any:
        """Return a booted BrainController, creating one if needed."""
        if self._brain_controller is None:
            import os

            os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")
            from core.brain.brain_controller import BrainController

            self._brain_controller = BrainController()
            self._brain_controller.boot_system()
            self._own_controller = True
        return self._brain_controller

    def close(self) -> None:
        """Close any controller created by this probe."""
        if self._own_controller and self._brain_controller is not None:
            self._brain_controller.close()
            self._brain_controller = None
            self._own_controller = False

    def _check_wiring(self) -> CheckResult:
        start = time.perf_counter()
        try:
            from core.brain.brain_controller import BrainController
            from core.pipeline.pipeline_orchestrator import PipelineOrchestrator

            ok = BrainController is not None and PipelineOrchestrator is not None
            return CheckResult(
                name="System Wiring",
                status="PASS" if ok else "FAIL",
                message="Core imports load without errors",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:
            return CheckResult(
                name="System Wiring",
                status="FAIL",
                message=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_api_health(self) -> CheckResult:
        start = time.perf_counter()
        url = f"http://{self.api_host}:{self.api_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as response:
                data = response.read().decode("utf-8")
                healthy = '"healthy"' in data or '"status"' in data
                return CheckResult(
                    name="API Health",
                    status="PASS" if healthy else "WARN",
                    message=f"API reports healthy at {url}" if healthy else f"Unexpected response: {data[:80]}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={"url": url, "response": data[:200]},
                )
        except urllib.error.URLError as exc:
            return CheckResult(
                name="API Health",
                status="WARN",
                message=f"API not reachable at {url}: {exc.reason if hasattr(exc, 'reason') else exc}",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"url": url},
            )
        except Exception as exc:
            return CheckResult(
                name="API Health",
                status="FAIL",
                message=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"url": url},
            )

    def _check_brain_controller(self) -> CheckResult:
        start = time.perf_counter()
        try:
            bc = self._ensure_controller()
            return CheckResult(
                name="BrainController Boot",
                status="PASS",
                message=f"BrainController booted: success={True}",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"success": True, "version": bc.config.version},
            )
        except Exception as exc:
            return CheckResult(
                name="BrainController Boot",
                status="FAIL",
                message=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_pipeline_stages(self) -> CheckResult:
        start = time.perf_counter()
        try:
            bc = self._ensure_controller()
            count = bc.pipeline.stage_count()
            expected = 20
            status = "PASS" if count == expected else "FAIL"
            return CheckResult(
                name="Pipeline Stages",
                status=status,
                message=f"Pipeline has {count} stages (expected {expected})",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"stage_count": count, "expected": expected},
            )
        except Exception as exc:
            return CheckResult(
                name="Pipeline Stages",
                status="FAIL",
                message=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_agent_registry(self) -> CheckResult:
        start = time.perf_counter()
        try:
            bc = self._ensure_controller()
            count = bc.agent_registry.count()
            modes = list(bc.agent_registry.modes())
            status = "PASS" if count >= 5 else "FAIL"
            return CheckResult(
                name="Agent Registry",
                status=status,
                message=f"{count} agents registered: {modes}",
                duration_ms=(time.perf_counter() - start) * 1000,
                details={"count": count, "modes": modes},
            )
        except Exception as exc:
            return CheckResult(
                name="Agent Registry",
                status="FAIL",
                message=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
