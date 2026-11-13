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
Core configuration for the unified Brain system.

This module re-exports and extends the immutable BrainConfig.
"""

from __future__ import annotations

from core.brain.brain_config import BrainConfig

__all__ = ["BrainConfig", "CoreConfig"]


class CoreConfig:
    """
    Compatibility wrapper around BrainConfig.

    Allows other modules to import configuration from `core.config` while
    the canonical implementation lives in `core.brain`.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_config",)

    def __init__(self, config: BrainConfig | None = None) -> None:
        """Initialize with a BrainConfig instance.

        Args:
            config: Optional BrainConfig; defaults are used if None.
        """
        self._config = config if config is not None else BrainConfig.default()

    @property
    def brain(self) -> BrainConfig:
        """Return the underlying BrainConfig.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._config
