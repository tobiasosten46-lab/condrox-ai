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

"""Tests for TokenPressureRegulator — Phase 2-F component."""

from __future__ import annotations

from core.conversational.token_pressure_regulator import (
    PressureReport,
    TokenBudget,
    TokenPressureRegulator,
)


class TestTokenPressureRegulator:
    """Unit tests for TokenPressureRegulator."""

    def test_measure_text(self) -> None:
        """measure estimates token count."""
        count = TokenPressureRegulator.measure("hello world foo bar")
        assert count == 4

    def test_measure_empty_text(self) -> None:
        """Empty text has zero tokens."""
        assert TokenPressureRegulator.measure("") == 0

    def test_set_usage_and_check_pressure(self) -> None:
        """Usage and pressure are tracked."""
        reg = TokenPressureRegulator(total_budget=1000)
        reg.set_usage(500)
        report = reg.check_pressure()
        assert report.pressure_level == 0.5
        assert report.action == "trim"

    def test_pressure_ok(self) -> None:
        """Low usage is ok."""
        reg = TokenPressureRegulator(total_budget=1000)
        reg.set_usage(100)
        report = reg.check_pressure()
        assert report.action == "ok"

    def test_pressure_warning(self) -> None:
        """High usage triggers compress."""
        reg = TokenPressureRegulator(total_budget=1000)
        reg.set_usage(800)
        report = reg.check_pressure()
        assert report.action == "compress"

    def test_pressure_critical(self) -> None:
        """Critical usage triggers block."""
        reg = TokenPressureRegulator(total_budget=1000)
        reg.set_usage(950)
        report = reg.check_pressure()
        assert report.action == "block"

    def test_add_usage(self) -> None:
        """add_usage accumulates."""
        reg = TokenPressureRegulator(total_budget=1000)
        reg.set_usage(100)
        reg.add_usage(50)
        assert reg.get_usage() == 150

    def test_trim_to_budget(self) -> None:
        """Text is trimmed to budget."""
        reg = TokenPressureRegulator(total_budget=1000)
        text = " ".join(f"word{i}" for i in range(100))
        trimmed = reg.trim_to_budget(text, max_tokens=10)
        assert trimmed.endswith("...")
        assert len(trimmed.split()) <= 11  # 10 words + ellipsis

    def test_trim_within_budget(self) -> None:
        """Text within budget is unchanged."""
        reg = TokenPressureRegulator(total_budget=1000)
        text = "hello world"
        trimmed = reg.trim_to_budget(text, max_tokens=10)
        assert trimmed == text

    def test_compress_context(self) -> None:
        """Context items are compressed to fit budget."""
        reg = TokenPressureRegulator(total_budget=1000)
        items = [f"item {i} " * 10 for i in range(20)]
        compressed = reg.compress_context(items, max_tokens=50)
        assert len(compressed) < len(items)

    def test_get_budget(self) -> None:
        """Budget is accessible."""
        reg = TokenPressureRegulator(total_budget=1000)
        budget = reg.get_budget()
        assert budget.total == 1000

    def test_reset_usage(self) -> None:
        """reset_usage sets usage to zero."""
        reg = TokenPressureRegulator(total_budget=1000)
        reg.set_usage(500)
        reg.reset_usage()
        assert reg.get_usage() == 0

    def test_token_budget_is_frozen(self) -> None:
        """TokenBudget is immutable."""
        reg = TokenPressureRegulator(total_budget=1000)
        budget = reg.get_budget()
        try:
            budget.total = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_pressure_report_is_frozen(self) -> None:
        """PressureReport is immutable."""
        reg = TokenPressureRegulator(total_budget=1000)
        report = reg.check_pressure()
        try:
            report.action = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_budget_ratios(self) -> None:
        """Budget is split according to ratios."""
        reg = TokenPressureRegulator(
            total_budget=1000,
            input_ratio=0.5,
            output_ratio=0.3,
            context_ratio=0.15,
            reserve_ratio=0.05,
        )
        budget = reg.get_budget()
        assert budget.input == 500
        assert budget.output == 300
        assert budget.context == 150
        assert budget.reserve == 50
