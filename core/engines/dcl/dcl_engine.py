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
Dynamic Context Layer (DCL) engine.

Implements a deterministic finite-state machine for tracking conversational
context state. Transitions are triggered by predicates evaluated against
current context facts.

Mathematical Foundation:
- DCL is a deterministic finite automaton (DFA) with states Q, alphabet Σ
  (events), and transition function δ: Q × Σ → Q.
- Each transition uses a predicate P: C → {0,1} over context variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class DCLState:
    """Immutable DCL state descriptor.

    Attributes:
        name: Unique state name.
        description: Human-readable state description.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    name: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class DCLTransition:
    """Immutable DCL transition with predicate.

    Attributes:
        source: Source state name.
        target: Target state name.
        event: Triggering event label.
        predicate: Function accepting context and returning bool.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    source: str
    target: str
    event: str
    predicate: Any = field(default=None)


class DCLEngine:
    """
    Real deterministic finite-state context engine.

    Attributes:
        _states: Mapping of state names to DCLState.
        _transitions: List of transitions.
        _current: Current state name.

    Complexity:
        add_state: O(1)
        add_transition: O(1)
        step: O(t) for t transitions from the current state.
    """

    __slots__ = ("_states", "_transitions", "_current")

    def __init__(self, initial_state: str = "idle") -> None:
        """Initialize the DCL engine.

        Args:
            initial_state: Starting state name.
        """
        self._states: Dict[str, DCLState] = {}
        self._transitions: List[DCLTransition] = []
        self._current: str = initial_state

    def add_state(self, state: DCLState) -> None:
        """Register a state.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._states[state.name] = state

    def add_transition(self, transition: DCLTransition) -> None:
        """Register a transition.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._transitions.append(transition)

    @property
    def current_state(self) -> str:
        """Return the current state name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._current

    def step(self, event: str, context: Dict[str, Any]) -> DCLState:
        """Process an event and possibly transition to a new state.

        Args:
            event: Incoming event label.
            context: Current context dictionary.

        Returns:
            The resulting DCLState.

        Complexity: T(n) ∈ O(t) where t is transitions from current state.
        """
        for transition in self._transitions:
            if transition.source != self._current or transition.event != event:
                continue
            predicate = transition.predicate
            if predicate is None or predicate(context):
                self._current = transition.target
                break
        return self._states.get(self._current, DCLState(self._current))

    def load_defaults(self) -> None:
        """Load a minimal default state machine.

        Complexity: O(1) because the graph is fixed and small.
        """
        self.add_state(DCLState("idle", "Waiting for user input"))
        self.add_state(DCLState("processing", "Processing a request"))
        self.add_state(DCLState("coding", "Generating code"))
        self.add_state(DCLState("analyzing", "Performing analysis"))
        self.add_state(DCLState("explaining", "Providing explanation"))

        self.add_transition(DCLTransition("idle", "processing", "request"))
        self.add_transition(
            DCLTransition(
                "processing",
                "coding",
                "mode_code",
                predicate=lambda ctx: ctx.get("mode") == "code",
            )
        )
        self.add_transition(
            DCLTransition(
                "processing",
                "analyzing",
                "mode_analyze",
                predicate=lambda ctx: ctx.get("mode") == "analyze",
            )
        )
        self.add_transition(
            DCLTransition(
                "processing",
                "explaining",
                "mode_explain",
                predicate=lambda ctx: ctx.get("mode") == "explain",
            )
        )
        self.add_transition(DCLTransition("coding", "idle", "complete"))
        self.add_transition(DCLTransition("analyzing", "idle", "complete"))
        self.add_transition(DCLTransition("explaining", "idle", "complete"))
