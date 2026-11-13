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

"""Tests for ResponseStyleEngine — Phase 2-B component."""

from __future__ import annotations

from core.conversational.response_style_engine import (
    ResponseStyleEngine,
    StyleProfile,
)


class TestResponseStyleEngine:
    """Unit tests for ResponseStyleEngine."""

    def test_get_profile_for_code_intent(self) -> None:
        """Code intent gets concise technical profile."""
        engine = ResponseStyleEngine()
        profile = engine.get_profile("code")
        assert profile.name == "code"
        assert profile.tone == "concise_technical"

    def test_get_profile_for_explain_intent(self) -> None:
        """Explain intent gets educational profile."""
        engine = ResponseStyleEngine()
        profile = engine.get_profile("explain")
        assert profile.name == "explain"

    def test_get_profile_defaults_to_conversation(self) -> None:
        """Unknown intent defaults to conversation."""
        engine = ResponseStyleEngine()
        profile = engine.get_profile("unknown_intent")
        assert profile.name == "conversation"

    def test_remove_robotic_phrases(self) -> None:
        """Robotic phrases are removed."""
        engine = ResponseStyleEngine()
        text = "As an AI, I don't have personal opinions. The function returns a value."
        cleaned, removed = engine.remove_robotic_phrases(text)
        assert "as an ai" not in cleaned.lower()
        assert len(removed) > 0

    def test_apply_removes_robotic_language(self) -> None:
        """apply() removes robotic language."""
        engine = ResponseStyleEngine()
        text = "Certainly! Here is the solution to your problem."
        result = engine.apply(text, "code")
        assert "certainly" not in result.lower()
        assert "here is" not in result.lower() or len(result) < len(text)

    def test_apply_casualizes_for_low_formality(self) -> None:
        """Low formality profile contracts words."""
        engine = ResponseStyleEngine()
        text = "You do not need to worry about this. It is fine."
        result = engine.apply(text, "code")  # code has formality 0.3
        assert "don't" in result or "do not" not in result

    def test_apply_formalizes_for_high_formality(self) -> None:
        """High formality profile expands contractions."""
        engine = ResponseStyleEngine()
        text = "You don't need to worry. It's fine."
        # diagnostic has formality 0.6
        result = engine.apply(text, "diagnostic")
        assert "do not" in result or "don't" not in result

    def test_apply_strips_preamble_for_code(self) -> None:
        """Code profile strips preambles."""
        engine = ResponseStyleEngine()
        text = "Sure, the answer is 42."
        result = engine.apply(text, "code")
        assert not result.lower().startswith("sure")

    def test_apply_strips_preamble_for_frustrated_emotion(self) -> None:
        """Frustrated emotion strips preambles."""
        engine = ResponseStyleEngine()
        text = "Certainly! Here is what you need to do."
        result = engine.apply(text, "explain", emotion="frustrated")
        assert not result.lower().startswith("certainly")

    def test_detect_robotic(self) -> None:
        """detect_robotic finds phrases without removing."""
        engine = ResponseStyleEngine()
        text = "As an AI, I cannot help with that."
        detected = engine.detect_robotic(text)
        assert len(detected) > 0
        assert "as an ai" in detected

    def test_detect_robotic_clean_text(self) -> None:
        """Clean text has no robotic phrases."""
        engine = ResponseStyleEngine()
        text = "The function returns the square of the input."
        detected = engine.detect_robotic(text)
        assert detected == []

    def test_empty_text(self) -> None:
        """Empty text returns empty."""
        engine = ResponseStyleEngine()
        assert engine.apply("", "code") == ""
        assert engine.remove_robotic_phrases("")[0] == ""

    def test_custom_profiles(self) -> None:
        """Custom profiles work."""
        custom = {
            "custom": StyleProfile(
                name="custom",
                tone="test",
                formality=0.5,
                max_preamble=0,
                sentence_style="declarative",
            )
        }
        engine = ResponseStyleEngine(profiles=custom)
        profile = engine.get_profile("custom")
        assert profile.name == "custom"
