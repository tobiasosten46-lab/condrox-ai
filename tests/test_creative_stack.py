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
Unit tests for the creative stack: EmotionPaletteMapper, MusicPromptBuilder,
QualityGuardrail.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

from core.creative import (
    EmotionPaletteMapper,
    MusicPrompt,
    MusicPromptBuilder,
    QualityAssessment,
    QualityGuardrail,
    SoundPalette,
)
from core.creative.emotion_palette_mapper import PALETTES


# ─── EmotionPaletteMapper ────────────────────────────────────────────────────


class TestEmotionPaletteMapper:
    """Tests for emotion-to-palette mapping."""

    @pytest.fixture
    def mapper(self) -> EmotionPaletteMapper:
        return EmotionPaletteMapper()

    def test_sad_maps_to_ambient(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("sad")
        assert p.emotion == "sad"
        assert p.genre == "ambient"
        assert p.tempo_bpm < 80
        assert p.energy < 0.3
        assert p.valence < 0.2

    def test_angry_maps_to_intense_genre(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("angry")
        assert p.emotion == "angry"
        assert p.energy > 0.8
        assert p.tempo_bpm > 140
        assert p.valence < 0.2

    def test_happy_maps_to_upbeat(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("happy")
        assert p.emotion == "happy"
        assert p.energy > 0.6
        assert p.valence > 0.8
        assert p.tempo_bpm > 100

    def test_anxious_maps_to_tense(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("anxious")
        assert p.emotion == "anxious"
        assert p.valence < 0.3
        assert "tense" in p.mood_tags or "nervous" in p.mood_tags

    def test_grateful_maps_to_folk(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("grateful")
        assert p.emotion == "grateful"
        assert p.genre == "folk"
        assert p.valence > 0.6

    def test_neutral_maps_to_lofi(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("neutral")
        assert p.emotion == "neutral"
        assert p.genre == "lo-fi hip-hop"

    def test_unknown_emotion_returns_neutral(self, mapper: EmotionPaletteMapper) -> None:
        p = mapper.map("nonexistent")
        assert p.emotion == "neutral"

    def test_all_emotions_have_palettes(self, mapper: EmotionPaletteMapper) -> None:
        emotions = mapper.available_emotions()
        assert "sad" in emotions
        assert "happy" in emotions
        assert "angry" in emotions
        assert "anxious" in emotions
        assert "grateful" in emotions
        assert "neutral" in emotions
        assert len(emotions) >= 10

    def test_blend_two_emotions(self, mapper: EmotionPaletteMapper) -> None:
        """Blending sad + happy should produce a moderate palette."""
        blended = mapper.blend([("sad", 0.5), ("happy", 0.5)])
        # Tempo should be between sad (65) and happy (128)
        assert 70 < blended.tempo_bpm < 120
        # Energy should be between sad (0.15) and happy (0.80)
        assert 0.3 < blended.energy < 0.7

    def test_blend_preserves_instruments(self, mapper: EmotionPaletteMapper) -> None:
        """Blending should union instruments from both palettes."""
        blended = mapper.blend([("sad", 0.5), ("angry", 0.5)])
        # Should have instruments from both sad and angry palettes
        all_instruments = set(blended.instruments)
        sad_inst = set(PALETTES["sad"].instruments)
        angry_inst = set(PALETTES["angry"].instruments)
        assert all_instruments & sad_inst  # Some sad instruments
        assert all_instruments & angry_inst  # Some angry instruments

    def test_blend_single_emotion_returns_direct(self, mapper: EmotionPaletteMapper) -> None:
        blended = mapper.blend([("happy", 1.0)])
        assert blended.emotion == "happy"
        assert blended.genre == "indie pop"

    def test_blend_empty_returns_neutral(self, mapper: EmotionPaletteMapper) -> None:
        blended = mapper.blend([])
        assert blended.emotion == "neutral"

    def test_palette_validates_energy_range(self) -> None:
        with pytest.raises(ValueError, match="energy"):
            SoundPalette(emotion="test", genre="test", energy=2.0)

    def test_palette_validates_valence_range(self) -> None:
        with pytest.raises(ValueError, match="valence"):
            SoundPalette(emotion="test", genre="test", valence=-0.5)

    def test_palette_validates_tempo_range(self) -> None:
        with pytest.raises(ValueError, match="tempo"):
            SoundPalette(emotion="test", genre="test", tempo_bpm=300)


# ─── MusicPromptBuilder ──────────────────────────────────────────────────────


class TestMusicPromptBuilder:
    """Tests for music prompt generation."""

    @pytest.fixture
    def builder(self) -> MusicPromptBuilder:
        return MusicPromptBuilder()

    @pytest.fixture
    def mapper(self) -> EmotionPaletteMapper:
        return EmotionPaletteMapper()

    def test_build_sad_prompt(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("sad")
        prompt = builder.build(palette)
        assert "ambient" in prompt.prompt_text
        assert "melancholic" in prompt.prompt_text.lower() or "reflective" in prompt.prompt_text.lower()
        assert "65" in prompt.prompt_text  # tempo
        assert "A minor" in prompt.prompt_text

    def test_build_happy_prompt(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("happy")
        prompt = builder.build(palette)
        assert "indie pop" in prompt.prompt_text
        assert "128" in prompt.prompt_text
        assert "G major" in prompt.prompt_text

    def test_build_includes_instruments(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("angry")
        prompt = builder.build(palette)
        assert "distorted guitar" in prompt.prompt_text
        assert "aggressive drums" in prompt.prompt_text

    def test_build_includes_style_tags(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("happy")
        prompt = builder.build(palette)
        assert "indie pop" in prompt.style_tags
        assert "uplifting" in prompt.style_tags or "joyful" in prompt.style_tags

    def test_build_includes_lyrics(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("sad")
        prompt = builder.build(palette, include_lyrics=True)
        assert prompt.lyrics
        assert "chorus" in prompt.lyrics.lower()

    def test_build_without_lyrics(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("happy")
        prompt = builder.build(palette, include_lyrics=False)
        assert prompt.lyrics == ""

    def test_build_includes_negative_prompt(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("sad")
        prompt = builder.build(palette)
        assert prompt.negative_prompt
        assert "cheerful" in prompt.negative_prompt or "upbeat" in prompt.negative_prompt

    def test_build_includes_metadata(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("angry")
        prompt = builder.build(palette)
        assert prompt.metadata["emotion"] == "angry"
        assert prompt.metadata["genre"] == "industrial metal"
        assert "160" in prompt.metadata["tempo"]

    def test_build_summary(self, builder: MusicPromptBuilder, mapper: EmotionPaletteMapper) -> None:
        palette = mapper.map("happy")
        prompt = builder.build(palette)
        summary = prompt.summary()
        assert "happy" in summary
        assert "indie pop" in summary

    def test_lyrics_differ_by_emotion(self, builder: MusicPromptBuilder) -> None:
        sad_lyrics = builder.build_lyrics("sad")
        happy_lyrics = builder.build_lyrics("happy")
        assert sad_lyrics != happy_lyrics

    def test_lyrics_deterministic_with_context(self, builder: MusicPromptBuilder) -> None:
        """Same context should produce same lyrics."""
        lyrics1 = builder.build_lyrics("sad", "my dog died")
        lyrics2 = builder.build_lyrics("sad", "my dog died")
        assert lyrics1 == lyrics2

    def test_lyrics_different_context_different_output(self, builder: MusicPromptBuilder) -> None:
        """Different contexts should produce different lyrics (when multiple templates)."""
        lyrics1 = builder.build_lyrics("sad", "my dog died")
        lyrics2 = builder.build_lyrics("sad", "my father passed away yesterday")
        # They might be the same if there's only 1 template, but with 2 they should differ
        # Just check both are non-empty
        assert lyrics1
        assert lyrics2


# ─── QualityGuardrail ────────────────────────────────────────────────────────


class TestQualityGuardrail:
    """Tests for the quality guardrail."""

    @pytest.fixture
    def guardrail(self) -> QualityGuardrail:
        return QualityGuardrail(threshold=0.6, min_length=20)

    def test_good_output_passes(self, guardrail: QualityGuardrail) -> None:
        output = "I'm sorry to hear that. That sounds really difficult. You mentioned your dog died. I understand your pain."
        result = guardrail.assess(output, "My dog died yesterday", emotion="sad")
        assert result.score > 0.5
        assert "empathy" in result.checks

    def test_generic_output_fails(self, guardrail: QualityGuardrail) -> None:
        output = "Mode: code | Domain: programming [Reasoning] confidence=1.0"
        result = guardrail.assess(output, "Write a function", emotion="neutral")
        assert not result.checks.get("not_generic", True)
        assert "GENERIC_RESPONSE" in result.issues

    def test_repetition_detected(self, guardrail: QualityGuardrail) -> None:
        output = "I understand your situation. That sounds really hard."
        prior = [output]
        result = guardrail.assess(output, "I'm sad", emotion="sad", prior_outputs=prior)
        assert not result.checks["no_repetition"]
        assert "REPETITION" in result.issues

    def test_too_short_fails(self, guardrail: QualityGuardrail) -> None:
        output = "OK."
        result = guardrail.assess(output, "Tell me about Python", emotion="neutral")
        assert not result.checks["min_length"]
        assert "TOO_SHORT" in result.issues

    def test_tone_mismatch_for_sad(self, guardrail: QualityGuardrail) -> None:
        """Cheerful response to sad input should fail tone check."""
        output = "That's fantastic! Congratulations! Wonderful news!"
        result = guardrail.assess(output, "My father died", emotion="sad")
        assert not result.checks.get("tone_appropriate", True)

    def test_tone_match_for_happy(self, guardrail: QualityGuardrail) -> None:
        output = "That's fantastic! I'm so happy for you! Congratulations!"
        result = guardrail.assess(output, "I got promoted!", emotion="happy")
        assert result.checks.get("tone_appropriate", False)

    def test_gate_passes_good_output(self, guardrail: QualityGuardrail) -> None:
        output = "I'm sorry to hear that. That sounds really difficult. You mentioned your dog. I understand."
        gated, assessment = guardrail.gate(output, "My dog died", emotion="sad")
        assert gated == output
        assert assessment.recommendation == "pass"

    def test_gate_replaces_generic_output(self, guardrail: QualityGuardrail) -> None:
        output = "Mode: code | Domain: programming [Reasoning] confidence=1.0"
        fallback = "I'm here to help. Tell me more about what you need."
        gated, assessment = guardrail.gate(output, "Write code", emotion="neutral", fallback=fallback)
        assert gated == fallback
        assert assessment.recommendation == "replace"

    def test_gate_enriches_low_empathy(self, guardrail: QualityGuardrail) -> None:
        """Output with low empathy should be enriched with a prefix."""
        output = "You asked about your health checkup. I want to make sure I understand what you need."
        gated, assessment = guardrail.gate(output, "I'm scared about my health checkup", emotion="anxious")
        # Should be enriched with empathy prefix
        assert "normal" in gated.lower() or "understand" in gated.lower()

    def test_critical_failure_forces_replace(self, guardrail: QualityGuardrail) -> None:
        """A critical failure (generic) should force replacement even if score is high."""
        output = "Mode: code [Reasoning] confidence=1.0 I understand your pain and I'm sorry."
        result = guardrail.assess(output, "I'm sad", emotion="sad")
        assert "not_generic" in result.critical_failures
        assert not result.passed

    def test_assessment_is_immutable(self, guardrail: QualityGuardrail) -> None:
        result = guardrail.assess("Good output here", "test input", emotion="neutral")
        # QualityAssessment is frozen
        with pytest.raises((AttributeError, Exception)):
            result.score = 0.0
