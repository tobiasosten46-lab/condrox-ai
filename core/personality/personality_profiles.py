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
Built-in personality profiles for the unified Brain.

Each profile is an immutable dataclass specifying tone, verbosity,
preamble, and transformation rules.
"""

from __future__ import annotations

from core.personality.personality_engine import PersonalityProfile


def condrox_profile() -> PersonalityProfile:
    """Return the Condrox engineering profile.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    return PersonalityProfile(
        name="condrox",
        tone="precise, technical, direct",
        verbosity=0.8,
        preamble="",
        capitalization="sentence",
        emoji_policy="none",
    )


def analyst_profile() -> PersonalityProfile:
    """Return the analyst profile.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    return PersonalityProfile(
        name="analyst",
        tone="analytical, structured, concise",
        verbosity=0.5,
        preamble="Analyst mode online. State your objective.",
        capitalization="sentence",
        emoji_policy="none",
    )
