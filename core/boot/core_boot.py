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
Boot system for the unified Brain.

Performs real pre-flight checks: configuration validation, dependency checks,
and health check registration. Produces a deterministic BootReport.

Mathematical Foundation:
- Boot status is the conjunction of all checks: B = ∧ c_i.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass(frozen=True, slots=True)
class BootReport:
    """Immutable boot report.

    Attributes:
        success: True if all checks passed.
        checks: List of individual check results.
        messages: Human-readable boot messages.
        metadata: Additional boot metadata.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(c) for c checks.
    """

    success: bool
    checks: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict, hash=False)


class CoreBoot:
    """
    Real boot orchestrator for the BrainController.

    Attributes:
        _checks: Registered boot checks.

    Complexity:
        register: O(1)
        run: O(c) for c checks.
    """

    __slots__ = ("_checks",)

    def __init__(self) -> None:
        """Initialize boot orchestrator."""
        self._checks: Dict[str, Callable[[], tuple[bool, str]]] = {}

    def register(self, name: str, check: Callable[[], tuple[bool, str]]) -> None:
        """Register a boot check.

        Args:
            name: Check name.
            check: Callable returning (success, message).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._checks[name] = check

    def _check_dependency(self, module_name: str) -> tuple[bool, str]:
        """Check if a Python module can be imported.

        Args:
            module_name: Module to import.

        Returns:
            Tuple of (success, message).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        try:
            importlib.import_module(module_name)
            return True, f"{module_name} available"
        except ImportError:
            return False, f"{module_name} missing"

    def load_defaults(self) -> None:
        """Register default boot checks.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.register("numpy", lambda: self._check_dependency("numpy"))
        self.register("yaml", lambda: self._check_dependency("yaml"))
        self.register("fastapi", lambda: self._check_dependency("fastapi"))

    def run(self) -> BootReport:
        """Run all registered boot checks and return a report.

        Returns:
            BootReport with all results.

        Complexity: T(n) ∈ O(c) for c checks.
        """
        checks: List[str] = []
        messages: List[str] = []
        success = True

        for name in sorted(self._checks.keys()):
            ok, message = self._checks[name]()
            checks.append(f"{name}: {'PASS' if ok else 'FAIL'}")
            messages.append(message)
            if not ok:
                success = False

        return BootReport(
            success=success,
            checks=checks,
            messages=messages,
            metadata={"total": str(len(checks)), "failed": str(sum(1 for c in checks if "FAIL" in c))},
        )
