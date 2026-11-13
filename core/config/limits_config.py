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
Runtime limits and safety constants for the unified Brain system.

These limits are intentionally conservative and can be overridden by
BrainConfig where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """Immutable runtime limits.

    Attributes:
        max_input_length: Maximum input characters per request.
        max_request_size_bytes: Maximum request payload size.
        max_memory_items: Soft cap on memory store size.
        max_knowledge_items: Soft cap on knowledge backend size.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    max_input_length: int = 4096
    max_request_size_bytes: int = 1048576
    max_memory_items: int = 10000
    max_knowledge_items: int = 10000

    def __post_init__(self) -> None:
        """Validate all limits are positive.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if value <= 0:
                raise ValueError(f"{field.name} must be positive, got {value}")


DEFAULT_LIMITS = LimitsConfig()
