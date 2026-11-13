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

"""Verification report data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single verification check.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    status: str  # PASS, WARN, FAIL
    message: str
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Aggregated verification report.

    Complexity: T(n) ∈ O(c) for c checks, S(n) ∈ O(c).
    """

    overall: str  # OK, WARN, FAIL
    score: float  # 0.0–100.0
    checks: List[CheckResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Render report as simple text."""
        lines = [
            "Condrox AI System Verification",
            "=" * 40,
            f"Overall:  {self.overall}",
            f"Score:    {self.score:.1f}%",
            f"Checks:   {len(self.checks)}",
            "-" * 40,
        ]
        for check in self.checks:
            lines.append(f"[{check.status:4}] {check.name}: {check.message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Render report as a dictionary."""
        return {
            "overall": self.overall,
            "score": self.score,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "duration_ms": c.duration_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "metadata": self.metadata,
        }
