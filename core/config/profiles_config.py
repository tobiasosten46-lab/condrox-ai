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
Profile configuration for the unified Brain system.

Provides a registry of available personality profiles.
"""

from __future__ import annotations

from core.personality.personality_engine import PersonalityProfile
from core.personality.personality_profiles import condrox_profile, analyst_profile


class ProfilesConfig:
    """
    Real profile registry.

    Attributes:
        _profiles: Mapping from profile name to PersonalityProfile.

    Complexity:
        register: O(1)
        get: O(1)
    """

    __slots__ = ("_profiles",)

    def __init__(self) -> None:
        """Initialize with default profiles."""
        self._profiles: dict[str, PersonalityProfile] = {
            "condrox": condrox_profile(),
            "analyst": analyst_profile(),
        }

    def register(self, name: str, profile: PersonalityProfile) -> None:
        """Register a profile.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._profiles[name] = profile

    def get(self, name: str) -> PersonalityProfile | None:
        """Return a profile by name.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._profiles.get(name)

    def list_profiles(self) -> list[str]:
        """Return all registered profile names.

        Complexity: T(n) ∈ O(p), S(n) ∈ O(p).
        """
        return list(self._profiles.keys())
