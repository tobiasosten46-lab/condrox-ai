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
Abstract Quality Gate interface for Condrox AI.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class GateSeverity(Enum):
    """Severity levels for gate findings.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class Finding:
    """
    Single finding from a quality gate check.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    rule_id: str
    message: str
    file_path: str
    line: int
    column: int
    severity: GateSeverity
    suggested_fix: str = ""


@dataclass(frozen=True, slots=True)
class GateResult:
    """
    Result from running a quality gate.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(findings).
    """

    gate_name: str
    passed: bool
    severity: GateSeverity
    score: float
    findings: Tuple[Finding, ...] = ()
    auto_fixable: bool = False


class QualityGate(ABC):
    """
    Abstract base class for all quality gates.

    Complexity: T(check) ∈ O(n) where n = code size.
    Invariants: name() and enabled() always return valid values.
    Edge Cases: empty code → passed=True.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Gate name (e.g. 'TYPES')."""
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this gate is active."""
        ...

    @abstractmethod
    async def check(
        self, code: str, file_path: str, language: str
    ) -> GateResult:
        """
        Run the gate check on code.

        Args:
            code: Source code to check.
            file_path: Path to the file (for reporting).
            language: Programming language ('python', 'javascript', etc.).

        Returns:
            GateResult with findings and pass/fail status.

        Complexity: T(n) ∈ O(n) where n = code length.
        Edge Cases: empty code → passed=True.
        """
        ...
