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
verify_system.py — Condrox AI Full System Verification CLI

Phase 1 entry point:
  - Runs system probes (wiring, API, BrainController, pipeline, agents)
  - Loads verification scenarios
  - Prints a simple PASS/WARN/FAIL report

Usage:
    python scripts/verify_system.py
    python scripts/verify_system.py --api-port 8000 --format json

Complexity: T(n) ∈ O(p) for p probes, O(s) for s scenarios.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.verification import ScenarioEngine, SystemProbe
from core.verification.report import CheckResult, VerificationReport


def run_verification(api_host: str, api_port: int) -> VerificationReport:
    """Run all Phase 1 system verification probes and build a report."""
    os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")

    probe = SystemProbe(api_host=api_host, api_port=api_port)
    results = list(probe.probe_all().values())

    # Load scenario registry (Phase 1 only counts available scenarios)
    start = time.perf_counter()
    engine = ScenarioEngine()
    scenario_count = engine.count()
    results.append(
        CheckResult(
            name="Scenario Registry",
            status="PASS" if scenario_count > 0 else "WARN",
            message=f"{scenario_count} verification scenarios loaded",
            duration_ms=(time.perf_counter() - start) * 1000,
            details={"count": scenario_count},
        )
    )

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    total = len(results)
    score = (pass_count / total) * 100.0 if total else 0.0
    overall = "FAIL" if fail_count else ("WARN" if any(r.status == "WARN" for r in results) else "OK")

    return VerificationReport(
        overall=overall,
        score=score,
        checks=results,
        metadata={"api_host": api_host, "api_port": api_port},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Condrox AI System Verification")
    parser.add_argument("--api-host", default="localhost", help="API host")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    args = parser.parse_args()

    report = run_verification(args.api_host, args.api_port)

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_text())

    return 0 if report.overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
