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
Token Pressure Regulator — monitors and regulates token usage to prevent
context window overflow and manage computational budget.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """
    Immutable token budget allocation.

    Attributes:
        total: Total token budget.
        input: Input token allocation.
        output: Output token allocation.
        context: Context window token allocation.
        reserve: Reserve tokens for safety margin.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    total: int
    input: int
    output: int
    context: int
    reserve: int


@dataclass(frozen=True, slots=True)
class PressureReport:
    """
    Immutable token pressure report.

    Attributes:
        current_usage: Current token count.
        budget: Active token budget.
        pressure_level: Pressure in [0.0, 1.0] (1.0 = at budget limit).
        action: Recommended action ("ok", "trim", "compress", "block").

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    current_usage: int
    budget: TokenBudget
    pressure_level: float
    action: str


class TokenPressureRegulator:
    """
    Monitors and regulates token usage to prevent overflow.

    Complexity:
        measure: T(n) ∈ O(n) for n characters.
        check_pressure: T(n) ∈ O(1), S(n) ∈ O(1).
        trim_to_budget: T(n) ∈ O(n), S(n) ∈ O(n).
    """

    __slots__ = ("_budget", "_current_usage", "_warning_threshold", "_critical_threshold")

    def __init__(
        self,
        total_budget: int = 8192,
        input_ratio: float = 0.4,
        output_ratio: float = 0.4,
        context_ratio: float = 0.15,
        reserve_ratio: float = 0.05,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.9,
    ) -> None:
        """Initialize the token pressure regulator.

        Args:
            total_budget: Total token budget.
            input_ratio: Fraction of budget for input.
            output_ratio: Fraction of budget for output.
            context_ratio: Fraction of budget for context.
            reserve_ratio: Fraction of budget for reserve.
            warning_threshold: Pressure level for warning.
            critical_threshold: Pressure level for critical.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._budget = TokenBudget(
            total=total_budget,
            input=int(total_budget * input_ratio),
            output=int(total_budget * output_ratio),
            context=int(total_budget * context_ratio),
            reserve=int(total_budget * reserve_ratio),
        )
        self._current_usage: int = 0
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold

    @staticmethod
    def measure(text: str) -> int:
        """Estimate token count for text.

        Uses a heuristic of ~4 characters per token.

        Args:
            text: Text to measure.

        Returns:
            Estimated token count.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        if not text:
            return 0
        # Split on whitespace and punctuation for a rough estimate
        tokens = re.findall(r"\S+", text)
        return max(1, len(tokens))

    def set_usage(self, token_count: int) -> None:
        """Set the current token usage.

        Args:
            token_count: Current token usage.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._current_usage = max(0, token_count)

    def add_usage(self, token_count: int) -> None:
        """Add to the current token usage.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._current_usage += max(0, token_count)

    def check_pressure(self) -> PressureReport:
        """Check current token pressure level.

        Returns:
            PressureReport with current status and recommended action.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        pressure = self._current_usage / self._budget.total if self._budget.total > 0 else 0.0
        pressure = min(1.0, pressure)

        if pressure >= self._critical_threshold:
            action = "block"
        elif pressure >= self._warning_threshold:
            action = "compress"
        elif pressure >= 0.5:
            action = "trim"
        else:
            action = "ok"

        return PressureReport(
            current_usage=self._current_usage,
            budget=self._budget,
            pressure_level=pressure,
            action=action,
        )

    def trim_to_budget(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Trim text to fit within a token budget.

        Args:
            text: Text to trim.
            max_tokens: Maximum tokens; defaults to output budget.

        Returns:
            Trimmed text.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        if not text:
            return text

        limit = max_tokens if max_tokens is not None else self._budget.output
        tokens = text.split()

        if len(tokens) <= limit:
            return text

        # Trim to limit and add ellipsis
        trimmed = tokens[:limit]
        return " ".join(trimmed) + "..."

    def compress_context(self, context_items: List[str], max_tokens: Optional[int] = None) -> List[str]:
        """Compress context items to fit within a token budget.

        Args:
            context_items: List of context text items.
            max_tokens: Maximum tokens; defaults to context budget.

        Returns:
            Compressed list of context items.

        Complexity: T(n) ∈ O(n * m) for n items, m avg length.
        """
        limit = max_tokens if max_tokens is not None else self._budget.context
        result: List[str] = []
        used = 0

        for item in context_items:
            tokens = self.measure(item)
            if used + tokens > limit:
                # Try to fit a trimmed version
                remaining = limit - used
                if remaining > 5:
                    trimmed = self.trim_to_budget(item, remaining)
                    result.append(trimmed)
                break
            result.append(item)
            used += tokens

        return result

    def get_budget(self) -> TokenBudget:
        """Return the current token budget.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._budget

    def get_usage(self) -> int:
        """Return current token usage.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._current_usage

    def reset_usage(self) -> None:
        """Reset token usage to zero.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._current_usage = 0
