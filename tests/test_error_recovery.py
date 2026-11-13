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

"""Tests for ErrorRecoveryLayer — Phase 2-C component."""

from __future__ import annotations

from core.conversational.error_recovery import (
    ErrorRecoveryLayer,
    ErrorRecord,
    RecoveryResult,
    RecoveryType,
)


class TestErrorRecoveryLayer:
    """Unit tests for ErrorRecoveryLayer."""

    def test_checkpoint_and_restore(self) -> None:
        """Checkpoint save and restore works."""
        layer = ErrorRecoveryLayer()
        state = {"stage": 5, "data": "test"}
        layer.checkpoint("req_001", state)
        restored = layer.restore_checkpoint("req_001")
        assert restored == state

    def test_restore_missing_checkpoint(self) -> None:
        """Missing checkpoint returns None."""
        layer = ErrorRecoveryLayer()
        assert layer.restore_checkpoint("missing") is None

    def test_clear_checkpoint(self) -> None:
        """clear_checkpoint removes checkpoint."""
        layer = ErrorRecoveryLayer()
        layer.checkpoint("req_001", {"data": "test"})
        layer.clear_checkpoint("req_001")
        assert layer.restore_checkpoint("req_001") is None

    def test_soft_recovery_with_checkpoint(self) -> None:
        """Soft recovery restores checkpoint."""
        layer = ErrorRecoveryLayer()
        layer.checkpoint("req_001", {"stage": 3})
        result = layer.handle_error("req_001", "stage_05", ValueError("test error"))
        assert result.success is True
        assert result.recovery_type == RecoveryType.SOFT
        assert result.retry_recommended is True
        assert result.recovered_state == {"stage": 3}

    def test_soft_recovery_without_checkpoint(self) -> None:
        """Soft recovery fails without checkpoint."""
        layer = ErrorRecoveryLayer()
        result = layer.handle_error("req_001", "stage_05", ValueError("test"))
        assert result.success is False
        assert result.recovery_type == RecoveryType.SOFT

    def test_hard_recovery(self) -> None:
        """Hard recovery resets state."""
        layer = ErrorRecoveryLayer()
        layer.checkpoint("req_001", {"stage": 3})
        result = layer.handle_error("req_001", "stage_05", RuntimeError("crash"))
        assert result.success is True
        assert result.recovery_type == RecoveryType.HARD
        assert layer.restore_checkpoint("req_001") is None

    def test_max_retries_exceeded(self) -> None:
        """Max retries triggers abort."""
        layer = ErrorRecoveryLayer(max_retries=2)
        for _ in range(2):
            layer.handle_error("req_001", "stage_05", ValueError("test"))
        result = layer.handle_error("req_001", "stage_05", ValueError("test"))
        assert result.recovery_type == RecoveryType.ABORT
        assert result.retry_recommended is False

    def test_custom_handler(self) -> None:
        """Custom recovery handlers work."""
        layer = ErrorRecoveryLayer()

        def custom_handler(record: ErrorRecord) -> RecoveryType:
            return RecoveryType.ABORT

        layer.register_handler("CustomError", custom_handler)
        result = layer.handle_error("req_001", "stage_05", CustomError("custom"))
        assert result.recovery_type == RecoveryType.ABORT

    def test_error_log(self) -> None:
        """Error log records failures."""
        layer = ErrorRecoveryLayer()
        layer.handle_error("req_001", "stage_05", ValueError("test"))
        log = layer.get_error_log()
        assert len(log) > 0
        assert log[0].error_type == "ValueError"

    def test_clear_error_log(self) -> None:
        """clear_error_log empties the log."""
        layer = ErrorRecoveryLayer()
        layer.handle_error("req_001", "stage_05", ValueError("test"))
        layer.clear_error_log()
        assert len(layer.get_error_log()) == 0

    def test_reset_retries(self) -> None:
        """reset_retries clears retry counts."""
        layer = ErrorRecoveryLayer(max_retries=2)
        layer.handle_error("req_001", "stage_05", ValueError("test"))
        layer.handle_error("req_001", "stage_05", ValueError("test"))
        layer.reset_retries("req_001")
        # Should be able to retry again
        result = layer.handle_error("req_001", "stage_05", ValueError("test"))
        assert result.recovery_type != RecoveryType.ABORT

    def test_recovery_result_is_frozen(self) -> None:
        """RecoveryResult is immutable."""
        layer = ErrorRecoveryLayer()
        result = layer.handle_error("req_001", "stage_05", ValueError("test"))
        try:
            result.success = False  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected

    def test_permission_error_aborts(self) -> None:
        """PermissionError triggers abort."""
        layer = ErrorRecoveryLayer()
        result = layer.handle_error("req_001", "stage_05", PermissionError("denied"))
        assert result.recovery_type == RecoveryType.ABORT


class CustomError(Exception):
    """Test custom error."""
    pass
