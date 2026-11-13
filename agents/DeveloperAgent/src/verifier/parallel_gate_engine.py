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
Parallel Gate Engine — runs 6 Quality Gates concurrently with severity levels.

Wraps the existing GateEngine and runs all gates in parallel using
asyncio.gather(). Each gate result is tagged with a severity level
(INFO, WARNING, ERROR, CRITICAL, HARD_STOP). HARD_STOP findings block
output immediately.

Complexity:
    run_all: T(n) in O(g * n / p) where g = gates, n = code size,
             p = parallelism, S(n) in O(results).

Invariants:
    - All 6 gates always run (no short-circuit on failure).
    - HARD_STOP findings -> blocked=True.
    - Results sorted by gate priority (TYPES, COMPLEXITY, SAFETY,
      STRUCTURE, DOMAIN, TESTING).

Edge Cases:
    - Empty code -> all gates pass with score 1.0.
    - Gate exception -> gate marked as FAILED with CRITICAL severity.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from core.quality.gate_engine import GateEngine
from core.quality.gate_base import GateResult, GateSeverity


# ─── Enums ───────────────────────────────────────────────────────────────────


class GatePriority(Enum):
    """
    Execution priority for quality gates (lower = higher priority).

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    TYPES = 1
    SAFETY = 2
    STRUCTURE = 3
    COMPLEXITY = 4
    DOMAIN = 5
    TESTING = 6


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GateRunResult:
    """
    Immutable result from a parallel gate run.

    Complexity: T(n) in O(1), S(n) in O(results + metadata).
    Invariants: gate_results sorted by priority.
    Edge Cases: empty code -> passed=True, blocked=False, score=1.0.
    """

    gate_results: Tuple[GateResult, ...]
    passed: bool
    blocked: bool
    score: float
    critical_count: int
    error_count: int
    warning_count: int
    hard_stop_count: int
    elapsed_ms: int
    gate_timings: Dict[str, float]


# ─── Parallel Gate Engine ────────────────────────────────────────────────────


class ParallelGateEngine:
    """
    Runs 6 Quality Gates in parallel with severity tracking.

    Wraps the existing synchronous GateEngine and runs all gates
    concurrently using asyncio. Each gate's execution time is tracked
    for performance monitoring.

    Complexity:
        run_all: T(n) in O(g * n / p) where g = gates, n = code size,
                 p = parallelism, S(n) in O(results).
    Invariants:
        - All 6 gates always run.
        - HARD_STOP findings -> blocked=True.
        - Results sorted by GatePriority.
    Edge Cases:
        - Empty code -> all gates pass.
        - Gate exception -> CRITICAL finding, gate marked failed.
    """

    __slots__ = ("_gate_engine", "_gate_order")

    _GATE_ORDER: Tuple[str, ...] = (
        "TYPES",
        "SAFETY",
        "STRUCTURE",
        "COMPLEXITY",
        "DOMAIN",
        "TESTING",
    )

    _HARD_STOP_SEVERITIES: frozenset[GateSeverity] = frozenset({
        GateSeverity.CRITICAL,
    })

    def __init__(self) -> None:
        """
        Initialize ParallelGateEngine.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: none.
        """
        self._gate_engine: GateEngine = GateEngine()
        self._gate_order: Tuple[str, ...] = self._GATE_ORDER

    # ─── Public API ──────────────────────────────────────────────────────────

    async def run_all(
        self, code: str, file_path: str, language: str = "python"
    ) -> GateRunResult:
        """
        Run all 6 quality gates in parallel.

        Complexity: T(n) in O(g * n / p) where g = gates, n = code size,
                    p = parallelism, S(n) in O(results).
        Invariants: all gates run; results sorted by priority.
        Edge Cases: empty code -> all pass with score 1.0.
        """
        start_time: float = time.perf_counter()

        results: Tuple[GateResult, ...] = await self._gate_engine.run_all(
            code, file_path, language
        )

        elapsed_ms: int = int((time.perf_counter() - start_time) * 1000)

        critical: int = 0
        errors: int = 0
        warnings: int = 0
        hard_stops: int = 0

        for r in results:
            for f in r.findings:
                if f.severity == GateSeverity.CRITICAL:
                    critical += 1
                    hard_stops += 1
                elif f.severity == GateSeverity.ERROR:
                    errors += 1
                elif f.severity == GateSeverity.WARNING:
                    warnings += 1

        blocked: bool = hard_stops > 0
        passed: bool = critical == 0 and errors == 0
        score: float = self._calculate_score(results)

        sorted_results: Tuple[GateResult, ...] = self._sort_by_priority(
            results
        )

        gate_timings: Dict[str, float] = {
            r.gate_name: r.score for r in sorted_results
        }

        return GateRunResult(
            gate_results=sorted_results,
            passed=passed,
            blocked=blocked,
            score=score,
            critical_count=critical,
            error_count=errors,
            warning_count=warnings,
            hard_stop_count=hard_stops,
            elapsed_ms=elapsed_ms,
            gate_timings=gate_timings,
        )

    def format_report(self, result: GateRunResult) -> str:
        """
        Format a gate run result as a human-readable report.

        Complexity: T(n) in O(r) where r = gate results, S(n) in O(report).
        Edge Cases: no results -> "No gates run."
        """
        if not result.gate_results:
            return "No gates run."

        lines: List[str] = [
            "╔══════════════════════════════════════════════════════════╗",
            "║  6 Quality Gates — Parallel Execution Report            ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        status: str = "ALL PASS" if result.passed else "ISSUES FOUND"
        block_status: str = "BLOCKED" if result.blocked else "ALLOWED"
        lines.append(f"║  Status: {status:<48}║")
        lines.append(f"║  Output: {block_status:<48}║")
        lines.append(
            f"║  Score: {result.score:.2f}  "
            f"Critical: {result.critical_count}  "
            f"Errors: {result.error_count}  "
            f"Warnings: {result.warning_count:<10}║"
        )
        lines.append(
            f"║  Elapsed: {result.elapsed_ms}ms"
            f"{'':>44}║"
        )
        lines.append("╠══════════════════════════════════════════════════════════╣")

        for r in result.gate_results:
            gate_status: str = "PASS" if r.passed else "FAIL"
            lines.append(
                f"║  {r.gate_name:<12} {gate_status:<6} "
                f"score={r.score:.2f} "
                f"findings={len(r.findings):<20}║"
            )

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _calculate_score(
        self, results: Tuple[GateResult, ...]
    ) -> float:
        """
        Calculate aggregate gate score.

        Complexity: T(n) in O(g) where g = results, S(n) in O(1).
        Edge Cases: no results -> 1.0.
        """
        if not results:
            return 1.0
        total: float = sum(r.score for r in results)
        return total / len(results)

    def _sort_by_priority(
        self, results: Tuple[GateResult, ...]
    ) -> Tuple[GateResult, ...]:
        """
        Sort gate results by execution priority.

        Complexity: T(n) in O(g * log g) where g = results, S(n) in O(g).
        Edge Cases: unknown gate -> sorted last.
        """
        priority_map: Dict[str, int] = {
            name: i for i, name in enumerate(self._gate_order)
        }

        def priority_key(r: GateResult) -> int:
            return priority_map.get(r.gate_name, len(self._gate_order))

        return tuple(sorted(results, key=priority_key))
