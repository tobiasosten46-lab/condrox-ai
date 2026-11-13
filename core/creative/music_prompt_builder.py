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
Music Prompt Builder.

Generates AI music generation prompts (for Suno, Udio, or custom models)
from a SoundPalette and PersonalityProfile. The builder creates structured
prompt text with genre, mood, instrumentation, tempo, and style directives.

Also generates lyrics snippets when the conversation context provides
emotional narrative material.

Mathematical Foundation:
- Prompt assembly is linear concatenation of palette fields: O(p + t)
  for p palette fields and t text tokens.
- Lyric generation is template-based with keyword injection: O(k + t)
  for k keywords and t template tokens.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.creative.emotion_palette_mapper import SoundPalette
from core.personality.personality_engine import PersonalityProfile


@dataclass(frozen=True, slots=True)
class MusicPrompt:
    """Immutable music generation prompt.

    Attributes:
        prompt_text: Full prompt for the music AI (genre, mood, instruments, etc.).
        style_tags: Comma-separated style tags for tag-based systems.
        lyrics: Optional lyrics snippet (verse + chorus).
        negative_prompt: Things to avoid (for systems that support it).
        metadata: Generation metadata (emotion, tempo, key, energy, valence).

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    prompt_text: str
    style_tags: str
    lyrics: str = ""
    negative_prompt: str = ""
    metadata: Dict[str, str] = field(default_factory=dict, hash=False)

    def summary(self) -> str:
        """Return a one-line summary of the prompt.

        Complexity: T(n) in O(1).
        """
        return (
            f"[{self.metadata.get('emotion', '?')}] "
            f"{self.metadata.get('genre', '?')} "
            f"{self.metadata.get('tempo', '?')}BPM "
            f"{self.metadata.get('key', '?')} "
            f"energy={self.metadata.get('energy', '?')} "
            f"valence={self.metadata.get('valence', '?')}"
        )


# ─── Lyric Templates by Emotion ──────────────────────────────────────────────

LYRIC_TEMPLATES: dict[str, list[str]] = {
    "sad": [
        "The room is quiet now\nYour shadow fades away\nI hold the memory\nOf everything we were\n\n(chorus)\nAnd I miss you\nLike the rain misses the sky\nFalling down\nWith nowhere to land",
        "Fourteen years of footsteps\nEcho on the floor\nI can't believe you're gone\nI can't open the door\n\n(chorus)\nEvery silence is a song\nEvery song is you\nAnd I'm still here\nLearning how to lose",
    ],
    "happy": [
        "The sun is calling out my name\nI feel it in my veins\nEverything I wanted\nIs finally here today\n\n(chorus)\nOh, I'm alive\nAnd the world is on my side\nOh, I'm alive\nAnd I'm running with the tide",
        "We did it, we made it through\nEvery wall came down\nI'm standing in the light\nNo more shadows now\n\n(chorus)\nThis is the day\nWe were waiting for\nOpen the door\nThere's so much more",
    ],
    "angry": [
        "You took what was mine\nAnd you smiled while you did it\nI'm not gonna be quiet\nI'm not gonna forget it\n\n(chorus)\nNo more, no more\nI won't be your floor\nNo more, no more\nI'm walking out the door",
        "Three weeks of silence\nThree weeks of lies\nYou think I'm nothing\nOpen your eyes\n\n(chorus)\nI'm not your tool\nI'm not your fool\nBurn the bridge\nI'm done with you",
    ],
    "anxious": [
        "My chest is tight\nMy mind won't stop\nThe clock is ticking\nAnd I can't turn it off\n\n(chorus)\nWhat if, what if\nThe spiral never ends\nWhat if, what if\nI can't pretend",
        "Tomorrow is coming\nAnd I'm not ready\nEveryone's watching\nMy hands are shaking\n\n(chorus)\nBreathe in, breathe out\nThey say it's all in my head\nBreathe in, breathe out\nBut the fear is in my chest",
    ],
    "grateful": [
        "You stood by me\nWhen I couldn't stand\nYou believed in me\nWhen I had no plan\n\n(chorus)\nThank you, thank you\nFor being my light\nThank you, thank you\nFor making it right",
        "I look around\nAnd I see what I have\nIt's more than I dreamed\nIt's more than I asked\n\n(chorus)\nI'm grateful, I'm grateful\nFor every single day\nI'm grateful, I'm grateful\nIn every single way",
    ],
    "philosophical": [
        "We build our categories\nWe draw our lines\nBut the universe\nDoesn't care about our designs\n\n(chorus)\nIs it real\nOr just a name we gave\nIs it beauty\nOr just the way we're made",
        "The tree falls\nIn the empty wood\nDoes it make a sound\nOr just a should\n\n(chorus)\nWe are the meaning\nWe are the question\nWe are the silence\nAnd the suggestion",
    ],
    "personal": [
        "I'm standing at the edge\nOf everything I know\nFifty years behind me\nAnd I'm scared to go\n\n(chorus)\nBut it's my life\nAnd it's my time\nI'll take the leap\nI'll make it mine",
        "They say I'm too old\nThey say it's too late\nBut my heart is young\nAnd I can't wait\n\n(chorus)\nI'm starting over\nI'm starting now\nI don't know how\nBut I'll find a way somehow",
    ],
    "neutral": [
        "The day is moving slow\nCoffee in my hand\nNothing to prove\nNo demands\n\n(chorus)\nJust breathing\nJust being\nThe world turns\nAnd I'm fine with it",
    ],
}

# Negative prompts — things to avoid in generation
NEGATIVE_PROMPTS: dict[str, str] = {
    "sad": "upbeat, cheerful, dance, party, aggressive, distorted",
    "angry": "calm, peaceful, relaxing, lullaby, cheerful, upbeat",
    "happy": "sad, dark, depressing, heavy, aggressive, angry",
    "anxious": "relaxing, calm, peaceful, cheerful, upbeat, party",
    "grateful": "angry, aggressive, dark, heavy, distorted, sad",
    "confused": "straightforward, simple, pop, generic, predictable",
    "curious": "boring, repetitive, generic, predictable, formulaic",
    "philosophical": "shallow, generic, pop, dance, party, superficial",
    "personal": "distant, cold, electronic, synthetic, industrial",
    "neutral": "aggressive, intense, overwhelming, chaotic, noisy",
}


class MusicPromptBuilder:
    """Build AI music generation prompts from palettes and profiles.

    Complexity:
        build: T(n) in O(p + t) for p palette fields, t text tokens.
        build_lyrics: T(n) in O(k + t) for k keywords, t template tokens.
    """

    __slots__ = ()

    def build(
        self,
        palette: SoundPalette,
        profile: Optional[PersonalityProfile] = None,
        include_lyrics: bool = True,
        user_context: str = "",
    ) -> MusicPrompt:
        """Build a complete music generation prompt.

        Args:
            palette: Sound palette from EmotionPaletteMapper.
            profile: Active personality profile (affects prompt tone).
            include_lyrics: Whether to include generated lyrics.
            user_context: Optional user input text for lyric inspiration.

        Returns:
            MusicPrompt with all fields populated.

        Complexity: T(n) in O(p + t).
        """
        # Build the main prompt text
        parts: List[str] = []

        # Genre and sub-genres
        genre_line = palette.genre
        if palette.sub_genres:
            genre_line += " / " + " / ".join(palette.sub_genres[:3])
        parts.append(f"Genre: {genre_line}")

        # Mood tags
        if palette.mood_tags:
            parts.append(f"Mood: {', '.join(palette.mood_tags[:6])}")

        # Tempo and key
        parts.append(f"Tempo: {palette.tempo_bpm} BPM")
        parts.append(f"Key: {palette.key}")

        # Energy and valence descriptors
        energy_desc = self._energy_descriptor(palette.energy)
        valence_desc = self._valence_descriptor(palette.valence)
        parts.append(f"Energy: {energy_desc}")
        parts.append(f"Emotional tone: {valence_desc}")

        # Instruments
        if palette.instruments:
            parts.append(f"Instruments: {', '.join(palette.instruments[:6])}")

        # Production style
        parts.append(f"Production: {palette.production_style}")

        # Reference artists
        if palette.reference_artists:
            parts.append(f"Style reference: {', '.join(palette.reference_artists[:3])}")

        # Personality influence
        if profile:
            parts.append(f"Persona: {profile.name} — {profile.tone}")

        # Build style tags (comma-separated for tag-based systems)
        style_tags_parts: List[str] = [palette.genre]
        style_tags_parts.extend(palette.sub_genres[:3])
        style_tags_parts.extend(palette.mood_tags[:5])
        style_tags_parts.append(f"{palette.tempo_bpm}bpm")
        style_tags_parts.append(palette.key)
        style_tags_parts.append(energy_desc)
        style_tags_parts.append(valence_desc)
        style_tags = ", ".join(style_tags_parts)

        # Build lyrics
        lyrics = ""
        if include_lyrics:
            lyrics = self.build_lyrics(palette.emotion, user_context)

        # Negative prompt
        negative = NEGATIVE_PROMPTS.get(
            palette.emotion.split("+")[0],
            "generic, formulaic, repetitive, low quality"
        )

        # Metadata
        metadata = {
            "emotion": palette.emotion,
            "genre": palette.genre,
            "tempo": str(palette.tempo_bpm),
            "key": palette.key,
            "energy": f"{palette.energy:.2f}",
            "valence": f"{palette.valence:.2f}",
        }

        return MusicPrompt(
            prompt_text="\n".join(parts),
            style_tags=style_tags,
            lyrics=lyrics,
            negative_prompt=negative,
            metadata=metadata,
        )

    def build_lyrics(self, emotion: str, user_context: str = "") -> str:
        """Generate lyrics for an emotion, optionally inspired by user context.

        Args:
            emotion: Emotional tone.
            user_context: Optional user input for inspiration.

        Returns:
            Lyrics text (verse + chorus).

        Complexity: T(n) in O(1) — template lookup.
        """
        # Get primary emotion (in case of blended like "sad+angry")
        primary = emotion.split("+")[0]
        templates = LYRIC_TEMPLATES.get(primary, LYRIC_TEMPLATES["neutral"])
        # Use hash of user_context to pick a template deterministically
        if user_context:
            idx = hash(user_context) % len(templates)
        else:
            idx = 0
        return templates[idx]

    def _energy_descriptor(self, energy: float) -> str:
        """Convert energy value to text descriptor.

        Complexity: T(n) in O(1).
        """
        if energy < 0.2:
            return "very calm"
        elif energy < 0.4:
            return "mellow"
        elif energy < 0.6:
            return "moderate"
        elif energy < 0.8:
            return "energetic"
        else:
            return "intense"

    def _valence_descriptor(self, valence: float) -> str:
        """Convert valence value to text descriptor.

        Complexity: T(n) in O(1).
        """
        if valence < 0.2:
            return "dark, melancholic"
        elif valence < 0.4:
            return "somber, reflective"
        elif valence < 0.6:
            return "balanced, bittersweet"
        elif valence < 0.8:
            return "warm, hopeful"
        else:
            return "bright, joyful"

    def __call__(
        self,
        palette: SoundPalette,
        profile: Optional[PersonalityProfile] = None,
        include_lyrics: bool = True,
        user_context: str = "",
    ) -> MusicPrompt:
        """Alias for build()."""
        return self.build(palette, profile, include_lyrics, user_context)
