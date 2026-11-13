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
Creative stack for Condrox AI.

Connects conversational intelligence to music/creative generation:
  - EmotionPaletteMapper: emotion -> sound palette (genre, tempo, key, instruments)
  - MusicPromptBuilder: palette + personality -> Suno/Udio prompt text + lyrics
  - QualityGuardrail: conversation quality gate before live user delivery

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from core.creative.emotion_palette_mapper import EmotionPaletteMapper, SoundPalette, PALETTES
from core.creative.music_prompt_builder import MusicPrompt, MusicPromptBuilder, LYRIC_TEMPLATES
from core.creative.quality_guardrail import QualityAssessment, QualityGuardrail

__all__ = [
    "EmotionPaletteMapper",
    "SoundPalette",
    "PALETTES",
    "MusicPrompt",
    "MusicPromptBuilder",
    "LYRIC_TEMPLATES",
    "QualityAssessment",
    "QualityGuardrail",
]
