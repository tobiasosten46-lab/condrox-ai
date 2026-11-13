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
SearchAgent Argument Guard — rejects null or empty input before SearchAgent
runs, preventing null-call errors and fallback router invocations without
a query.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class GuardResult:
    """
    Immutable result of a guard check.

    Attributes:
        valid: True if the query is safe to pass to SearchAgent.
        query: The validated query string, or a safe sentinel.
        reason: Explanation if the query was rejected.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    valid: bool
    query: str
    reason: str


class SearchAgentGuard:
    """
    Validates search agent input before execution.

    Rules:
    - None → invalid, sentinel "<no-query>"
    - Empty string → invalid, sentinel "<empty-query>"
    - Whitespace-only → invalid, sentinel "<empty-query>"
    - Valid string → valid, query passed through

    Complexity: T(n) ∈ O(n) where n is query length, S(n) ∈ O(1).
    """

    __slots__ = ("_sentinel_none", "_sentinel_empty", "_max_length")

    def __init__(
        self,
        sentinel_none: str = "<no-query>",
        sentinel_empty: str = "<empty-query>",
        max_length: int = 4096,
    ) -> None:
        """Initialize with sentinel values and max query length.

        Args:
            sentinel_none: Returned for None input.
            sentinel_empty: Returned for empty/whitespace input.
            max_length: Maximum allowed query length.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._sentinel_none = sentinel_none
        self._sentinel_empty = sentinel_empty
        self._max_length = max_length

    def validate(self, query: str | None) -> Tuple[bool, str]:
        """Check if a query is valid.

        Args:
            query: The query string to validate (may be None).

        Returns:
            Tuple of (valid, reason).

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        if query is None:
            return False, "query is None"
        if not isinstance(query, str):
            return False, f"query must be str, got {type(query).__name__}"
        if not query.strip():
            return False, "query is empty or whitespace-only"
        if len(query) > self._max_length:
            return False, f"query exceeds max length {self._max_length}"
        return True, "ok"

    def guard(self, query: str | None) -> str:
        """Return a safe query string, substituting sentinels for invalid input.

        Args:
            query: The query string to guard (may be None).

        Returns:
            The validated query, or a sentinel string.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        valid, reason = self.validate(query)
        if valid:
            return query.strip()  # type: ignore[union-attr]
        if query is None:
            return self._sentinel_none
        return self._sentinel_empty

    def check(self, query: str | None) -> GuardResult:
        """Full guard check returning a structured result.

        Args:
            query: The query string to check (may be None).

        Returns:
            GuardResult with validity, safe query, and reason.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        valid, reason = self.validate(query)
        safe_query = self.guard(query)
        return GuardResult(valid=valid, query=safe_query, reason=reason)
