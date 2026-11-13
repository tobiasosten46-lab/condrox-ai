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

"""Tests for Phase B user memory and personalization stores."""

from __future__ import annotations

import pytest

from core.memory.long_term_memory import EpisodicMemory, LongTermMemory
from core.memory.skill_memory import ProceduralSkill, SkillMemory
from core.memory.user_profile_store import UserProfile, UserProfileStore


class TestUserProfileStore:
    """Tests for UserProfileStore."""

    def test_create_and_get(self) -> None:
        """Creating and retrieving a profile works."""
        store = UserProfileStore()
        profile = store.upsert(
            user_id="alice",
            preferences={"language": "no", "theme": "dark"},
            summary="Likes Python and AI.",
        )

        assert isinstance(profile, UserProfile)
        assert profile.user_id == "alice"
        assert dict(profile.preferences)["language"] == "no"

        fetched = store.get("alice")
        assert fetched is not None
        assert fetched.summary == "Likes Python and AI."

    def test_upsert_merges_preferences(self) -> None:
        """Upsert merges new preferences into existing profile."""
        store = UserProfileStore()
        store.upsert("alice", preferences={"language": "no"}, summary="A")
        profile = store.upsert("alice", preferences={"theme": "dark"})

        assert dict(profile.preferences) == {"language": "no", "theme": "dark"}

    def test_profile_is_immutable(self) -> None:
        """UserProfile preferences cannot be mutated after creation."""
        store = UserProfileStore()
        profile = store.upsert("alice", preferences={"language": "no"})

        with pytest.raises(TypeError):
            profile.preferences[0] = ("language", "en")  # type: ignore

    def test_list_users(self) -> None:
        """list_users returns all user ids."""
        store = UserProfileStore()
        store.upsert("alice", preferences={})
        store.upsert("bob", preferences={})
        assert sorted(store.list_users()) == ["alice", "bob"]


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    def test_add_and_recall(self) -> None:
        """Adding a memory and recalling by query works."""
        ltm = LongTermMemory()
        ltm.add(
            user_id="alice",
            summary="Alice prefers Python for backend work.",
            session_id="sess_1",
            entities=["Python", "backend"],
            emotion="neutral",
        )

        results = ltm.recall("alice", "python backend", top_k=3)
        assert len(results) == 1
        assert results[0].user_id == "alice"
        assert "Python" in results[0].summary

    def test_recall_filters_by_user(self) -> None:
        """Recall only returns memories for the requested user."""
        ltm = LongTermMemory()
        ltm.add(user_id="alice", summary="Alice likes Python.")
        ltm.add(user_id="bob", summary="Bob likes JavaScript.")

        results = ltm.recall("alice", "python", top_k=3)
        assert len(results) == 1
        assert results[0].user_id == "alice"

    def test_no_match_returns_empty(self) -> None:
        """A query with no overlap returns empty."""
        ltm = LongTermMemory()
        ltm.add(user_id="alice", summary="Alice likes Python.")
        assert ltm.recall("alice", "quantum physics", top_k=3) == []


class TestSkillMemory:
    """Tests for SkillMemory."""

    def test_add_and_match(self) -> None:
        """Adding a skill and matching by trigger phrase works."""
        skills = SkillMemory()
        skills.add(
            name="deploy",
            trigger_phrases=["deploy app", "push to production"],
            steps=["run tests", "build image", "deploy"],
        )

        results = skills.match("deploy the app", top_k=3)
        assert len(results) == 1
        assert results[0].name == "deploy"

    def test_execute_returns_steps(self) -> None:
        """execute returns the ordered steps."""
        skills = SkillMemory()
        skill = skills.add(
            name="backup",
            trigger_phrases=["backup database"],
            steps=["dump db", "compress", "upload"],
        )
        assert skills.execute(skill.skill_id) == ["dump db", "compress", "upload"]

    def test_no_match_returns_empty(self) -> None:
        """A query with no trigger overlap returns empty."""
        skills = SkillMemory()
        skills.add(
            name="deploy",
            trigger_phrases=["deploy app"],
            steps=["run tests", "deploy"],
        )
        assert skills.match("order pizza", top_k=3) == []
