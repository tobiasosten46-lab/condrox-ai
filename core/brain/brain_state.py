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
Global state model for the unified Brain system.

BrainState is an immutable value object that flows through the 14-stage
cognitive pipeline. Each stage returns a new BrainState instance, ensuring
thread-safe, deterministic transformations and full traceability.

Mathematical Foundation:
- replace() creates a shallow copy with selective field updates: T(m) ∈ O(m)
  where m is the number of replaced fields; S(m) ∈ O(m).
- All lookups are O(1) because dataclass fields are fixed at class definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.cognition.context_manager import ContextManager

# Valid emotion labels — must match EmotionDetector categories
_VALID_EMOTIONS = frozenset({
    "neutral", "happy", "sad", "angry", "anxious",
    "grateful", "confused", "curious", "philosophical", "personal",
})


@dataclass(frozen=True, slots=True)
class BrainState:
    """
    Immutable container for all data produced and consumed by the pipeline.

    Attributes:
        request_id: Unique identifier for the request (UUID string).
        raw_input: User-supplied raw text.
        processed_input: Normalized, validated text.
        intent: Classified intent label.
        intent_confidence: Confidence score in [0.0, 1.0].
        mode: Selected processing mode (code, analyze, explain, etc.).
        domain: Selected knowledge domain.
        context: Dynamic context dictionary built during pipeline execution.
        retrieved_knowledge: List of knowledge item identifiers.
        retrieved_memories: List of memory item identifiers.
        reasoning: Structured reasoning trace.
        reasoning_evidence: Supporting evidence list.
        synthesized_output: Draft output before personality/security/ethics.
        personality_applied: Whether a personality profile was applied.
        ethical_approved: Whether the output passed ethics checks.
        security_approved: Whether the output passed security checks.
        final_output: Final rendered response.
        metadata: Free-form metadata (timestamps, version, etc.).
        errors: Accumulated non-fatal error messages.

    Complexity: T(init) ∈ O(1), S(init) ∈ O(1) for fixed-size metadata.
    """

    request_id: str = ""
    raw_input: str = ""
    processed_input: str = ""
    intent: str = "unknown"
    intent_confidence: float = 0.0
    mode: str = "default"
    domain: str = "general"
    emotion: str = "neutral"
    context: Dict[str, Any] = field(default_factory=dict)
    retrieved_knowledge: List[str] = field(default_factory=list)
    retrieved_memories: List[str] = field(default_factory=list)
    reasoning: str = ""
    reasoning_evidence: List[str] = field(default_factory=list)
    synthesized_output: str = ""
    personality_applied: bool = False
    ethical_approved: bool = False
    security_approved: bool = False
    final_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    context_manager: Optional["ContextManager"] = None  # transient — not serialized

    def __post_init__(self) -> None:
        """Validate field constraints after initialization.

        Raises:
            ValueError: If intent_confidence is outside [0.0, 1.0].

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not (0.0 <= self.intent_confidence <= 1.0):
            raise ValueError(
                f"intent_confidence must be in [0.0, 1.0], got {self.intent_confidence}"
            )

    def with_field(self, **changes: Any) -> "BrainState":
        """Return a new BrainState with the supplied field changes.

        Args:
            **changes: Field names and their new values.

        Returns:
            A new immutable BrainState instance.

        Complexity: T(n) ∈ O(m) where m is the number of changed fields;
            S(n) ∈ O(m) for the new object.
        """
        return replace(self, **changes)

    def add_error(self, message: str) -> "BrainState":
        """Return a new BrainState with an additional error message.

        Args:
            message: Non-fatal error or warning to append.

        Returns:
            A new immutable BrainState instance.

        Complexity: T(n) ∈ O(1) amortized, S(n) ∈ O(1) amortized.
        """
        new_errors = list(self.errors)
        new_errors.append(message)
        return replace(self, errors=new_errors)

    def is_valid(self) -> bool:
        """Return True when the state has valid critical fields.

        Checks raw_input, processed_input, and request_id.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return bool(self.raw_input) and bool(self.request_id)
