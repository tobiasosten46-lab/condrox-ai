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
Integration tests for the emotion -> music creative pipeline.

Tests the full flow: user input -> pipeline -> emotion detection ->
sound palette -> music prompt -> quality guardrail.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

from core import BrainController
from core.creative import EmotionPaletteMapper, MusicPromptBuilder, QualityGuardrail


@pytest.fixture
def controller() -> BrainController:
    bc = BrainController()
    bc.boot_system()
    return bc


# ─── Full Pipeline Integration ───────────────────────────────────────────────


class TestCreativePipelineIntegration:
    """Integration tests for the full creative pipeline."""

    def test_generate_creative_output_sad(self, controller: BrainController) -> None:
        """Sad input produces ambient palette with low energy."""
        result = controller.generate_creative_output("I lost my dog yesterday. I'm devastated.")
        assert result["state"].emotion == "sad"
        palette = result["palette"]
        assert palette.genre == "ambient"
        assert palette.energy < 0.3
        assert palette.valence < 0.2
        music = result["music_prompt"]
        assert "ambient" in music.prompt_text
        assert music.lyrics  # Should have lyrics
        assert "melancholic" in music.style_tags or "reflective" in music.style_tags

    def test_generate_creative_output_happy(self, controller: BrainController) -> None:
        """Happy input produces upbeat palette with high valence."""
        result = controller.generate_creative_output("I just got promoted! I'm so happy!")
        assert result["state"].emotion == "happy"
        palette = result["palette"]
        assert palette.valence > 0.8
        assert palette.energy > 0.6
        music = result["music_prompt"]
        assert "indie pop" in music.prompt_text

    def test_generate_creative_output_angry(self, controller: BrainController) -> None:
        """Angry input produces intense palette."""
        result = controller.generate_creative_output("I'm furious! Someone stole my work!")
        assert result["state"].emotion == "angry"
        palette = result["palette"]
        assert palette.energy > 0.8
        assert palette.tempo_bpm > 140
        music = result["music_prompt"]
        assert "industrial" in music.prompt_text.lower()

    def test_generate_creative_output_includes_quality(self, controller: BrainController) -> None:
        """Creative output includes a quality assessment."""
        result = controller.generate_creative_output("I'm grateful for your help. Thank you.")
        assert result["quality"] is not None
        assert result["quality"].score > 0.0

    def test_generate_creative_output_without_lyrics(self, controller: BrainController) -> None:
        """Can generate without lyrics."""
        result = controller.generate_creative_output("What is the meaning of life?", include_lyrics=False)
        assert result["music_prompt"].lyrics == ""

    def test_get_sound_palette_directly(self, controller: BrainController) -> None:
        """get_sound_palette returns palette for emotion."""
        palette = controller.get_sound_palette("sad")
        assert palette.emotion == "sad"
        assert palette.genre == "ambient"

    def test_build_music_prompt_directly(self, controller: BrainController) -> None:
        """build_music_prompt generates prompt from emotion."""
        prompt = controller.build_music_prompt("happy", user_context="I graduated!")
        assert "indie pop" in prompt.prompt_text
        assert prompt.lyrics  # Should have lyrics

    def test_creative_status_in_system_status(self, controller: BrainController) -> None:
        """Creative stack appears in system status."""
        status = controller.status()
        assert status["emotion_palette_mapper"] is True
        assert status["music_prompt_builder"] is True
        assert status["quality_guardrail"] is True
        assert "sad" in status["palette_emotions"]
        assert "happy" in status["palette_emotions"]

    def test_five_emotions_produce_five_different_palettes(self, controller: BrainController) -> None:
        """Five different emotions should produce 5 different genres."""
        emotions = ["sad", "happy", "angry", "anxious", "grateful"]
        genres = []
        for e in emotions:
            p = controller.get_sound_palette(e)
            genres.append(p.genre)
        assert len(set(genres)) == 5

    def test_five_emotions_produce_five_different_prompts(self, controller: BrainController) -> None:
        """Five different emotions should produce 5 different music prompts."""
        emotions = ["sad", "happy", "angry", "anxious", "grateful"]
        prompts = []
        for e in emotions:
            p = controller.build_music_prompt(e)
            prompts.append(p.prompt_text)
        assert len(set(prompts)) == 5


# ─── Quality Guardrail Integration ───────────────────────────────────────────


class TestQualityGuardrailIntegration:
    """Integration tests for quality guardrail with real pipeline output."""

    def test_pipeline_output_passes_guardrail(self, controller: BrainController) -> None:
        """Real pipeline output should pass the quality guardrail."""
        state = controller.process_request("I'm sad. My cat died today.")
        guardrail = controller.quality_guardrail
        result = guardrail.assess(
            state.final_output,
            "I'm sad. My cat died today.",
            emotion=state.emotion,
        )
        assert result.checks.get("not_generic", False), "Output should not be generic"
        assert result.checks.get("no_repetition", False), "Output should not repeat"

    def test_guardrail_catches_generic_output(self, controller: BrainController) -> None:
        """Guardrail should catch generic metadata in output."""
        guardrail = controller.quality_guardrail
        fake_output = "Mode: code | Domain: programming [Reasoning] confidence=1.0"
        result = guardrail.assess(fake_output, "Write code", emotion="neutral")
        assert not result.checks["not_generic"]
        assert not result.passed

    def test_guardrail_gate_with_real_pipeline(self, controller: BrainController) -> None:
        """Gate should pass real pipeline output."""
        state = controller.process_request("I'm happy! I just graduated!")
        guardrail = controller.quality_guardrail
        gated, assessment = guardrail.gate(
            state.final_output,
            "I'm happy! I just graduated!",
            emotion=state.emotion,
        )
        # Should pass or at least not be replaced
        assert assessment.recommendation in ("pass", "enrich")


# ─── End-to-End Creative Flow ────────────────────────────────────────────────


class TestEndToEndCreativeFlow:
    """End-to-end tests: user input -> emotion -> palette -> prompt -> lyrics."""

    def test_sad_flow_produces_coherent_package(self, controller: BrainController) -> None:
        """Sad input produces a coherent creative package."""
        result = controller.generate_creative_output(
            "I lost my father. He was my best friend. I'm heartbroken."
        )
        # Emotion detected
        assert result["state"].emotion == "sad"
        # Palette matches
        palette = result["palette"]
        assert palette.valence < 0.2
        assert "piano" in palette.instruments or "strings" in palette.instruments
        # Prompt is well-formed
        music = result["music_prompt"]
        assert "Genre:" in music.prompt_text
        assert "Tempo:" in music.prompt_text
        assert "Key:" in music.prompt_text
        assert "Instruments:" in music.prompt_text
        # Lyrics are present and emotional
        assert music.lyrics
        assert "chorus" in music.lyrics.lower()
        # Negative prompt avoids wrong mood
        assert "cheerful" in music.negative_prompt or "upbeat" in music.negative_prompt

    def test_happy_flow_produces_coherent_package(self, controller: BrainController) -> None:
        """Happy input produces a coherent creative package."""
        result = controller.generate_creative_output(
            "I just got engaged! This is the best day of my life!"
        )
        assert result["state"].emotion == "happy"
        palette = result["palette"]
        assert palette.valence > 0.8
        music = result["music_prompt"]
        assert "uplifting" in music.style_tags or "joyful" in music.style_tags
        assert "dark" in music.negative_prompt or "sad" in music.negative_prompt

    def test_angry_flow_produces_coherent_package(self, controller: BrainController) -> None:
        """Angry input produces a coherent creative package."""
        result = controller.generate_creative_output(
            "I'm furious! My coworker stole my work and took credit for it!"
        )
        assert result["state"].emotion == "angry"
        palette = result["palette"]
        assert palette.energy > 0.8
        music = result["music_prompt"]
        assert "aggressive" in music.style_tags or "intense" in music.style_tags

    def test_philosophical_flow_produces_coherent_package(self, controller: BrainController) -> None:
        """Philosophical input produces a coherent creative package."""
        result = controller.generate_creative_output(
            "What is the meaning of life? Do we have free will?"
        )
        assert result["state"].emotion == "philosophical"
        palette = result["palette"]
        assert palette.genre == "post-rock"
        music = result["music_prompt"]
        assert "contemplative" in music.style_tags or "expansive" in music.style_tags

    def test_conversation_to_music_chain(self, controller: BrainController) -> None:
        """A multi-turn conversation can chain into music generation."""
        # Turn 1: sad
        r1 = controller.generate_creative_output("My dog died. I'm so sad.")
        assert r1["state"].emotion == "sad"
        assert r1["palette"].genre == "ambient"

        # Turn 2: happy (mood shift)
        r2 = controller.generate_creative_output("But then I adopted a new puppy! I'm so happy!")
        assert r2["state"].emotion == "happy"
        assert r2["palette"].genre == "indie pop"

        # The two prompts should be different
        assert r1["music_prompt"].prompt_text != r2["music_prompt"].prompt_text
