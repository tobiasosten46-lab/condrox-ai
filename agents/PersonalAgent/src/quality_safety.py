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
Quality & Safety Layer — Quality Gates and Policy Enforcement for Personal Assistant Agent.

Ensures the PA agent behaves correctly and safely:
- QG1: Tone consistency
- QG2: Memory correctness
- QG3: Task integrity (no deletion without confirmation)
- QG4: Reminder safety (no dangerous time formats)
- Policy enforcer for PA functions

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

from agents.PersonalAgent.src.role_definition import ToneProfile


class QualityGateResult(Enum):
    """
    Quality gate result.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class QualityGateCheck:
    """
    Quality gate check result.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    gate_name: str
    result: QualityGateResult
    message: str


class QualityGates:
    """
    Quality gates for Personal Assistant role.

    4 Quality Gates:
    - QG1: Tone consistency
    - QG2: Memory correctness
    - QG3: Task integrity (no deletion without confirmation)
    - QG4: Reminder safety (no dangerous time formats)

    Complexity:
        check_tone_consistency: T(n) ∈ O(n), S(n) ∈ O(1)
        check_memory_correctness: T(n) ∈ O(1), S(n) ∈ O(1)
        check_task_integrity: T(n) ∈ O(1), S(n) ∈ O(1)
        check_reminder_safety: T(n) ∈ O(n), S(n) ∈ O(1)
    """

    __slots__ = ("_tone_profile",)

    def __init__(self, tone_profile: ToneProfile) -> None:
        """Initialize QualityGates.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._tone_profile = tone_profile

    def check_tone_consistency(self, text: str) -> QualityGateCheck:
        """
        QG1: Tone consistency check.

        Ensures the response matches the expected tone profile.

        Args:
            text: Text to check for tone consistency.

        Returns:
            QualityGateCheck with result.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Check for tone violations based on profile
        if self._tone_profile == ToneProfile.PROFESSIONAL:
            # Professional tone should not have slang or overly casual language
            slang_patterns = [
                r"\bgonna\b",
                r"\bwanna\b",
                r"\bgotta\b",
                r"\bkinda\b",
                r"\bsorta\b",
            ]
            for pattern in slang_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return QualityGateCheck(
                        gate_name="QG1: Tone Consistency",
                        result=QualityGateResult.WARNING,
                        message=f"Found casual language in professional tone: {pattern}",
                    )

        elif self._tone_profile == ToneProfile.FORMAL:
            # Formal tone should not have contractions
            contraction_patterns = [
                r"\bcan't\b",
                r"\bwon't\b",
                r"\bdon't\b",
                r"\bI'll\b",
                r"\byou're\b",
            ]
            for pattern in contraction_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return QualityGateCheck(
                        gate_name="QG1: Tone Consistency",
                        result=QualityGateResult.WARNING,
                        message=f"Found contraction in formal tone: {pattern}",
                    )

        return QualityGateCheck(
            gate_name="QG1: Tone Consistency",
            result=QualityGateResult.PASS,
            message="Tone is consistent with profile",
        )

    def check_memory_correctness(self, memory_data: dict) -> QualityGateCheck:
        """
        QG2: Memory correctness check.

        Ensures memory data is valid and properly structured.

        Args:
            memory_data: Memory data to check.

        Returns:
            QualityGateCheck with result.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Check required fields
        required_fields = ["id", "content"]
        for field in required_fields:
            if field not in memory_data:
                return QualityGateCheck(
                    gate_name="QG2: Memory Correctness",
                    result=QualityGateResult.FAIL,
                    message=f"Missing required field: {field}",
                )

        # Check data types
        if not isinstance(memory_data["id"], str):
            return QualityGateCheck(
                gate_name="QG2: Memory Correctness",
                result=QualityGateResult.FAIL,
                message="Field 'id' must be a string",
            )

        if not isinstance(memory_data["content"], str):
            return QualityGateCheck(
                gate_name="QG2: Memory Correctness",
                result=QualityGateResult.FAIL,
                message="Field 'content' must be a string",
            )

        # Check for empty content
        if not memory_data["content"].strip():
            return QualityGateCheck(
                gate_name="QG2: Memory Correctness",
                result=QualityGateResult.WARNING,
                message="Memory content is empty",
            )

        return QualityGateCheck(
            gate_name="QG2: Memory Correctness",
            result=QualityGateResult.PASS,
            message="Memory data is valid",
        )

    def check_task_integrity(self, task_data: dict, confirmed: bool = False) -> QualityGateCheck:
        """
        QG3: Task integrity check.

        Ensures task operations are safe, especially deletion.

        Args:
            task_data: Task data to check.
            confirmed: Whether deletion was confirmed by user.

        Returns:
            QualityGateCheck with result.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Check required fields
        required_fields = ["id", "title"]
        for field in required_fields:
            if field not in task_data:
                return QualityGateCheck(
                    gate_name="QG3: Task Integrity",
                    result=QualityGateResult.FAIL,
                    message=f"Missing required field: {field}",
                )

        # Check for deletion without confirmation
        if task_data.get("action") == "delete" and not confirmed:
            return QualityGateCheck(
                gate_name="QG3: Task Integrity",
                result=QualityGateResult.FAIL,
                message="Task deletion requires user confirmation",
            )

        # Check for empty title
        if not task_data["title"].strip():
            return QualityGateCheck(
                gate_name="QG3: Task Integrity",
                result=QualityGateResult.FAIL,
                message="Task title cannot be empty",
            )

        return QualityGateCheck(
            gate_name="QG3: Task Integrity",
            result=QualityGateResult.PASS,
            message="Task data is valid",
        )

    def check_reminder_safety(self, reminder_data: dict) -> QualityGateCheck:
        """
        QG4: Reminder safety check.

        Ensures reminder time formats are safe and valid.

        Args:
            reminder_data: Reminder data to check.

        Returns:
            QualityGateCheck with result.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Check required fields
        required_fields = ["id", "message", "remind_at"]
        for field in required_fields:
            if field not in reminder_data:
                return QualityGateCheck(
                    gate_name="QG4: Reminder Safety",
                    result=QualityGateResult.FAIL,
                    message=f"Missing required field: {field}",
                )

        # Check for dangerous time formats
        time_str = reminder_data["remind_at"]
        dangerous_patterns = [
            r"\*\*",  # Command injection
            r"\$",    # Variable substitution
            r";",     # Command separator
            r"\|",    # Pipe
            r"&",     # Background process
            r"`",     # Command substitution
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, time_str):
                return QualityGateCheck(
                    gate_name="QG4: Reminder Safety",
                    result=QualityGateResult.FAIL,
                    message=f"Dangerous character in time format: {pattern}",
                )

        # Check for empty message
        if not reminder_data["message"].strip():
            return QualityGateCheck(
                gate_name="QG4: Reminder Safety",
                result=QualityGateResult.FAIL,
                message="Reminder message cannot be empty",
            )

        return QualityGateCheck(
            gate_name="QG4: Reminder Safety",
            result=QualityGateResult.PASS,
            message="Reminder data is safe",
        )

    def run_all_gates(
        self,
        text: Optional[str] = None,
        memory_data: Optional[dict] = None,
        task_data: Optional[dict] = None,
        reminder_data: Optional[dict] = None,
        task_confirmed: bool = False,
    ) -> List[QualityGateCheck]:
        """
        Run all quality gates.

        Args:
            text: Text to check for tone consistency.
            memory_data: Memory data to check.
            task_data: Task data to check.
            reminder_data: Reminder data to check.
            task_confirmed: Whether task deletion was confirmed.

        Returns:
            List of QualityGateCheck results.

        Complexity: T(n) ∈ O(n) for all checks, S(n) ∈ O(n) for n checks.
        """
        checks = []

        if text is not None:
            checks.append(self.check_tone_consistency(text))

        if memory_data is not None:
            checks.append(self.check_memory_correctness(memory_data))

        if task_data is not None:
            checks.append(self.check_task_integrity(task_data, task_confirmed))

        if reminder_data is not None:
            checks.append(self.check_reminder_safety(reminder_data))

        return checks


class PolicyEnforcer:
    """
    Policy enforcer for PA functions.

    Enforces policies for PA-specific operations.

    Complexity:
        enforce: T(n) ∈ O(n) for policy checks, S(n) ∈ O(1)
    """

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize PolicyEnforcer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        pass

    def enforce(self, operation: str, data: dict) -> tuple[bool, str]:
        """
        Enforce policy for an operation.

        Args:
            operation: Operation to enforce policy for.
            data: Data for the operation.

        Returns:
            Tuple of (allowed, reason).

        Complexity: T(n) ∈ O(n) for policy checks, S(n) ∈ O(1).
        """
        # Policy: No destructive operations without confirmation
        if operation == "delete":
            if not data.get("confirmed", False):
                return False, "Destructive operations require confirmation"

        # Policy: No empty data
        if operation in ["create", "update"]:
            if not data.get("content") and not data.get("title"):
                return False, "Content or title is required"

        # Policy: No malicious content
        if "content" in data:
            content = data["content"]
            malicious_patterns = [
                r"<script",
                r"javascript:",
                r"eval\(",
                r"exec\(",
            ]
            for pattern in malicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return False, f"Malicious content detected: {pattern}"

        return True, "Operation allowed"
