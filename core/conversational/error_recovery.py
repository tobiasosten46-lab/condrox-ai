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
Error Recovery Layer — provides soft and hard recovery mechanisms for
pipeline failures.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class RecoveryType(Enum):
    """Recovery strategy types."""

    SOFT = "soft"  # Restore state and retry
    HARD = "hard"  # Reset pipeline and restart
    ABORT = "abort"  # Give up and return error


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """
    Immutable result of a recovery attempt.

    Attributes:
        success: Whether recovery succeeded.
        recovery_type: Strategy used.
        message: Description of what happened.
        retry_recommended: Whether a retry should be attempted.
        recovered_state: Optional recovered state data.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    success: bool
    recovery_type: RecoveryType
    message: str
    retry_recommended: bool
    recovered_state: Optional[Dict[str, Any]] = None


@dataclass
class ErrorRecord:
    """
    Mutable error record for tracking failures.

    Attributes:
        error_id: Unique identifier.
        stage_name: Where the error occurred.
        error_type: Type of error.
        error_message: Error message.
        timestamp: When the error occurred.
        recovery_attempted: Whether recovery was attempted.
        recovery_result: Result of recovery if attempted.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    error_id: str
    stage_name: str
    error_type: str
    error_message: str
    timestamp: float
    recovery_attempted: bool = False
    recovery_result: Optional[RecoveryResult] = None


class ErrorRecoveryLayer:
    """
    Provides soft and hard recovery for pipeline failures.

    Soft recovery: attempts to restore state from a checkpoint and retry
    the failed stage.

    Hard recovery: resets the pipeline to initial state and restarts from
    the beginning.

    Complexity:
        handle_error: T(n) ∈ O(1), S(n) ∈ O(1).
        checkpoint: T(n) ∈ O(s), S(n) ∈ O(s) for s state items.
    """

    __slots__ = (
        "_checkpoints",
        "_error_log",
        "_max_retries",
        "_retry_counts",
        "_recovery_handlers",
    )

    def __init__(self, max_retries: int = 3) -> None:
        """Initialize the error recovery layer.

        Args:
            max_retries: Maximum retries per stage before aborting.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._error_log: List[ErrorRecord] = []
        self._max_retries = max_retries
        self._retry_counts: Dict[str, int] = {}
        self._recovery_handlers: Dict[str, Callable[[ErrorRecord], RecoveryType]] = {}

    def checkpoint(self, request_id: str, state: Dict[str, Any]) -> None:
        """Save a state checkpoint for recovery.

        Args:
            request_id: Request identifier.
            state: State to checkpoint.

        Complexity: T(n) ∈ O(s), S(n) ∈ O(s) for s state items.
        """
        self._checkpoints[request_id] = dict(state)

    def restore_checkpoint(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Restore a saved checkpoint.

        Args:
            request_id: Request identifier.

        Returns:
            Saved state, or None if no checkpoint exists.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._checkpoints.get(request_id)

    def clear_checkpoint(self, request_id: str) -> None:
        """Clear a checkpoint.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._checkpoints.pop(request_id, None)

    def register_handler(
        self,
        error_type: str,
        handler: Callable[[ErrorRecord], RecoveryType],
    ) -> None:
        """Register a custom recovery handler for an error type.

        Args:
            error_type: Type of error to handle.
            handler: Function that returns the recovery strategy.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._recovery_handlers[error_type] = handler

    def handle_error(
        self,
        request_id: str,
        stage_name: str,
        error: Exception,
    ) -> RecoveryResult:
        """Handle a pipeline error with appropriate recovery strategy.

        Args:
            request_id: Request identifier.
            stage_name: Name of the failed stage.
            error: The exception that occurred.

        Returns:
            RecoveryResult with recovery outcome.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        error_record = ErrorRecord(
            error_id=f"{request_id}:{stage_name}:{int(time.time())}",
            stage_name=stage_name,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=time.time(),
        )

        # Check retry count
        retry_key = f"{request_id}:{stage_name}"
        current_retries = self._retry_counts.get(retry_key, 0)

        if current_retries >= self._max_retries:
            result = RecoveryResult(
                success=False,
                recovery_type=RecoveryType.ABORT,
                message=f"Max retries ({self._max_retries}) exceeded for {stage_name}",
                retry_recommended=False,
            )
            error_record.recovery_attempted = True
            error_record.recovery_result = result
            self._error_log.append(error_record)
            return result

        # Determine recovery strategy
        recovery_type = self._determine_recovery(error_record)

        # Execute recovery
        if recovery_type == RecoveryType.SOFT:
            result = self._soft_recover(request_id, stage_name, error_record)
        elif recovery_type == RecoveryType.HARD:
            result = self._hard_recover(request_id, stage_name, error_record)
        else:
            result = RecoveryResult(
                success=False,
                recovery_type=RecoveryType.ABORT,
                message=f"Aborting: {error_record.error_message}",
                retry_recommended=False,
            )

        # Update retry count
        self._retry_counts[retry_key] = current_retries + 1

        error_record.recovery_attempted = True
        error_record.recovery_result = result
        self._error_log.append(error_record)

        return result

    def _determine_recovery(self, error_record: ErrorRecord) -> RecoveryType:
        """Determine the recovery strategy for an error.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Check for custom handler
        handler = self._recovery_handlers.get(error_record.error_type)
        if handler:
            return handler(error_record)

        # Default strategies by error type
        if error_record.error_type in ("ValueError", "KeyError", "TypeError"):
            return RecoveryType.SOFT
        if error_record.error_type in ("RuntimeError", "TimeoutError"):
            return RecoveryType.HARD
        if error_record.error_type in ("PermissionError", "AuthenticationError"):
            return RecoveryType.ABORT

        # Default: soft recovery
        return RecoveryType.SOFT

    def _soft_recover(
        self,
        request_id: str,
        stage_name: str,
        error_record: ErrorRecord,
    ) -> RecoveryResult:
        """Perform soft recovery: restore checkpoint and retry.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        checkpoint = self._checkpoints.get(request_id)
        if checkpoint is None:
            return RecoveryResult(
                success=False,
                recovery_type=RecoveryType.SOFT,
                message=f"No checkpoint available for {request_id}",
                retry_recommended=False,
            )

        return RecoveryResult(
            success=True,
            recovery_type=RecoveryType.SOFT,
            message=f"Restored checkpoint for {stage_name}, retry recommended",
            retry_recommended=True,
            recovered_state=checkpoint,
        )

    def _hard_recover(
        self,
        request_id: str,
        stage_name: str,
        error_record: ErrorRecord,
    ) -> RecoveryResult:
        """Perform hard recovery: clear state and restart.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._checkpoints.pop(request_id, None)
        self._retry_counts.pop(f"{request_id}:{stage_name}", None)

        return RecoveryResult(
            success=True,
            recovery_type=RecoveryType.HARD,
            message=f"Reset pipeline state for {request_id}, full restart recommended",
            retry_recommended=True,
            recovered_state=None,
        )

    def get_error_log(self, limit: int = 100) -> List[ErrorRecord]:
        """Return recent error log entries.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._error_log[-limit:])

    def clear_error_log(self) -> None:
        """Clear the error log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._error_log.clear()

    def reset_retries(self, request_id: str) -> None:
        """Reset retry counts for a request.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        keys_to_remove = [k for k in self._retry_counts if k.startswith(f"{request_id}:")]
        for key in keys_to_remove:
            del self._retry_counts[key]
