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
Context manager for the unified Brain.

Maintains a stack-structured context dictionary with scope-based push/pop
operations and deterministic key lookup.

Mathematical Foundation:
- Context lookup is a linear search over a stack of frames: O(s) where s
  is the number of scopes. In practice s is bounded by a small constant.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


class ContextManager:
    """
    Hierarchical context manager with immutable frame copies.

    Attributes:
        _stack: List of context frames. The top frame shadows lower keys.

    Complexity:
        get: O(s) where s is scope depth.
        set: O(1) for top scope.
        push/pop: O(1) for reference, O(s) for deep copy on push.
    """

    __slots__ = ("_stack",)

    def __init__(self, initial: Optional[Dict[str, Any]] = None) -> None:
        """Initialize context with an optional initial frame.

        Args:
            initial: Initial key-value frame.
        """
        self._stack: List[Dict[str, Any]] = [dict(initial) if initial else {}]

    def push(self, frame: Optional[Dict[str, Any]] = None) -> None:
        """Push a new context frame.

        Args:
            frame: Optional initial frame values.

        Complexity: T(n) ∈ O(f) for f frame items, S(n) ∈ O(f).
        """
        self._stack.append(dict(frame) if frame else {})

    def pop(self) -> Dict[str, Any]:
        """Pop the top context frame.

        Returns:
            The removed frame.

        Raises:
            IndexError: If popping the only remaining frame.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if len(self._stack) <= 1:
            raise IndexError("Cannot pop the root context frame")
        return self._stack.pop()

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a key across all scopes from top to bottom.

        Args:
            key: Key to look up.
            default: Value returned when key is missing.

        Returns:
            The first matching value or default.

        Complexity: T(n) ∈ O(s) where s is scope depth, S(n) ∈ O(1).
        """
        for frame in reversed(self._stack):
            if key in frame:
                return frame[key]
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a key in the top context frame.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._stack[-1][key] = value

    def current(self) -> Dict[str, Any]:
        """Return a deep copy of the merged active context.

        Complexity: T(n) ∈ O(s * f) where f is average frame size.
        """
        merged: Dict[str, Any] = {}
        for frame in self._stack:
            merged.update(frame)
        return deepcopy(merged)

    def snapshot(self) -> Dict[str, Any]:
        """Alias for current().

        Complexity: T(n) ∈ O(s * f).
        """
        return self.current()
