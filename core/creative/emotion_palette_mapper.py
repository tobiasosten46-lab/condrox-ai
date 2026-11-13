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
Emotion-to-Sound-Palette Mapper.

Maps detected emotional tones to musical sound palettes: genre, tempo,
key, instruments, energy level, and mood descriptors. These palettes
feed into the MusicPromptBuilder which generates prompts for Suno /
Udio / custom music AI systems.

Mathematical Foundation:
- Mapping is a deterministic O(1) lookup from emotion string to palette.
- Palette blending (for mixed emotions) is a weighted average over
  continuous fields (tempo, energy) and union over discrete fields
  (instruments, tags). T(n) in O(p) for p palettes.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class SoundPalette:
    """Immutable sound palette for a given emotional tone.

    Attributes:
        emotion: Source emotion label.
        genre: Primary musical genre.
        sub_genres: Secondary genre tags for blending.
        tempo_bpm: Target tempo in beats per minute.
        key: Musical key (e.g. 'A minor', 'C major').
        energy: Energy level in [0.0, 1.0] (0=calm, 1=intense).
        valence: Emotional valence in [0.0, 1.0] (0=sad/dark, 1=happy/bright).
        instruments: Core instrument list.
        mood_tags: Descriptive mood tags for prompt generation.
        production_style: Production aesthetic descriptor.
        reference_artists: Reference artists for style guidance.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    emotion: str
    genre: str
    sub_genres: Tuple[str, ...] = ()
    tempo_bpm: int = 120
    key: str = "C major"
    energy: float = 0.5
    valence: float = 0.5
    instruments: Tuple[str, ...] = ()
    mood_tags: Tuple[str, ...] = ()
    production_style: str = "clean"
    reference_artists: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate ranges.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        if not 0.0 <= self.energy <= 1.0:
            raise ValueError(f"energy must be in [0.0, 1.0], got {self.energy}")
        if not 0.0 <= self.valence <= 1.0:
            raise ValueError(f"valence must be in [0.0, 1.0], got {self.valence}")
        if not 40 <= self.tempo_bpm <= 240:
            raise ValueError(f"tempo_bpm must be in [40, 240], got {self.tempo_bpm}")


# ─── Curated Emotion → Palette Mapping ───────────────────────────────────────

PALETTES: Dict[str, SoundPalette] = {
    "sad": SoundPalette(
        emotion="sad",
        genre="ambient",
        sub_genres=("melancholic", "cinematic", "neo-classical"),
        tempo_bpm=65,
        key="A minor",
        energy=0.15,
        valence=0.10,
        instruments=("piano", "cello", "strings", "soft pad", "rain ambience"),
        mood_tags=("melancholic", "reflective", "mourning", "tender", "weeping",
                   "sparse", "intimate", "lonely", "nostalgic"),
        production_style="warm, reverb-heavy, close-miked",
        reference_artists=("Max Richter", "Nils Frahm", "Ludovico Einaudi", "Olafur Arnalds"),
    ),
    "angry": SoundPalette(
        emotion="angry",
        genre="industrial metal",
        sub_genres=("noise rock", "post-hardcore", "dark electronic"),
        tempo_bpm=160,
        key="D minor",
        energy=0.95,
        valence=0.08,
        instruments=("distorted guitar", "bass", "aggressive drums", "synth",
                     "industrial noise", "screaming vocals"),
        mood_tags=("aggressive", "intense", "confrontational", "raw", "visceral",
                   "heavy", "driving", "furious", "explosive"),
        production_style="raw, distorted, punchy, in-your-face",
        reference_artists=("Nine Inch Nails", "Rage Against the Machine", "Health", "Godflesh"),
    ),
    "happy": SoundPalette(
        emotion="happy",
        genre="indie pop",
        sub_genres=("sunshine pop", "synth-pop", "tropical house"),
        tempo_bpm=128,
        key="G major",
        energy=0.80,
        valence=0.92,
        instruments=("acoustic guitar", "upbeat synth", "bass", "drums",
                     "handclaps", "shimmering keys", "brass stabs"),
        mood_tags=("uplifting", "joyful", "bright", "celebratory", "sunny",
                   "energetic", "feel-good", "euphoric", "warm"),
        production_style="polished, wide, colorful, sparkling",
        reference_artists=("Tegan and Sara", "Phoenix", "M83", "The Naked and Famous"),
    ),
    "anxious": SoundPalette(
        emotion="anxious",
        genre="dark ambient",
        sub_genres=("drone", "experimental electronic", "cinematic tension"),
        tempo_bpm=90,
        key="E minor",
        energy=0.35,
        valence=0.15,
        instruments=("sub-bass drone", "anxious strings", "heartbeat percussion",
                     "dissonant pad", "ticking clock", "whisper synth"),
        mood_tags=("tense", "unsettling", "nervous", "suspenseful", "dread",
                   "pulsing", "claustrophobic", "on-edge", "breathing"),
        production_style="close, claustrophobic, pulsing, sub-bass heavy",
        reference_artists=("Ben Frost", "Tim Hecker", "Oneohtrix Point Never", "Forest Swords"),
    ),
    "grateful": SoundPalette(
        emotion="grateful",
        genre="folk",
        sub_genres=("indie folk", "acoustic", "chamber folk"),
        tempo_bpm=95,
        key="D major",
        energy=0.45,
        valence=0.80,
        instruments=("acoustic guitar", "warm piano", "soft strings", "mandolin",
                     "upright bass", "gentle drums", "harmonium"),
        mood_tags=("heartfelt", "warm", "thankful", "tender", "grateful",
                   "sincere", "grounded", "peaceful", "loving"),
        production_style="organic, warm, room sound, intimate",
        reference_artists=("Bon Iver", "Fleet Foxes", "The Lumineers", "Sufjan Stevens"),
    ),
    "confused": SoundPalette(
        emotion="confused",
        genre="experimental",
        sub_genres=("glitch", "IDM", "avant-garde electronic"),
        tempo_bpm=110,
        key="B minor",
        energy=0.40,
        valence=0.30,
        instruments=("glitchy synth", "broken beats", "modular patches",
                     "detuned samples", "reverse reverb", "stutter effects"),
        mood_tags=("uncertain", "searching", "fragmented", "curious", "lost",
                   "exploratory", "off-kilter", "dreamlike", "shifting"),
        production_style="glitchy, unpredictable, textured, stereo-playful",
        reference_artists=("Aphex Twin", "Boards of Canada", "Autechre", "Flying Lotus"),
    ),
    "curious": SoundPalette(
        emotion="curious",
        genre="electronic",
        sub_genres=("downtempo", "trip-hop", "lo-fi electronic"),
        tempo_bpm=100,
        key="F major",
        energy=0.50,
        valence=0.60,
        instruments=("warm synth", "lo-fi drums", "bass", "vinyl crackle",
                     "rhodes", "field recordings", "soft percussion"),
        mood_tags=("inquisitive", "exploratory", "wondering", "engaging",
                   "thoughtful", "gentle", "flowing", "discovering"),
        production_style="warm, textured, lo-fi, layered",
        reference_artists=("Bonobo", "Thievery Corporation", "Röyksopp", "Tycho"),
    ),
    "philosophical": SoundPalette(
        emotion="philosophical",
        genre="post-rock",
        sub_genres=("cinematic", "neo-classical", "ambient rock"),
        tempo_bpm=85,
        key="C# minor",
        energy=0.45,
        valence=0.35,
        instruments=("electric guitar", "bass", "drums", "strings",
                     "piano", "synth pad", "glockenspiel"),
        mood_tags=("contemplative", "expansive", "profound", "searching",
                   "existential", "cathartic", "building", "epic"),
        production_style="wide, cinematic, building dynamics, reverb-drenched",
        reference_artists=("Explosions in the Sky", "Mogwai", "Sigur Ros", "This Will Destroy You"),
    ),
    "personal": SoundPalette(
        emotion="personal",
        genre="singer-songwriter",
        sub_genres=("indie acoustic", "chamber pop", "confessional"),
        tempo_bpm=80,
        key="E major",
        energy=0.35,
        valence=0.50,
        instruments=("acoustic guitar", "piano", "soft strings", "upright bass",
                     "brushed drums", "vocals", "harmonica"),
        mood_tags=("intimate", "honest", "vulnerable", "confessional", "raw",
                   "personal", "storytelling", "warm", "close"),
        production_style="intimate, dry, close-miked, room sound",
        reference_artists=("Sufjan Stevens", "Phoebe Bridgers", "José González", "Iron & Wine"),
    ),
    "neutral": SoundPalette(
        emotion="neutral",
        genre="lo-fi hip-hop",
        sub_genres=("chillhop", "jazz-hop", "ambient hip-hop"),
        tempo_bpm=85,
        key="A major",
        energy=0.40,
        valence=0.55,
        instruments=("lo-fi piano", "jazz guitar samples", "boom-bap drums",
                     "bass", "vinyl crackle", "soft synth"),
        mood_tags=("relaxed", "focused", "calm", "studious", "mellow",
                   "easygoing", "smooth", "unobtrusive"),
        production_style="lo-fi, warm, vinyl warmth, gentle compression",
        reference_artists=("Nujabes", "J Dilla", "Tomppabeats", "Idealism"),
    ),
}


class EmotionPaletteMapper:
    """Map emotional tones to musical sound palettes.

    Complexity:
        map: T(n) in O(1) for single emotion, O(p) for blended.
        blend: T(n) in O(p) for p palettes.
    """

    __slots__ = ("_palettes",)

    def __init__(self, palettes: Optional[Dict[str, SoundPalette]] = None) -> None:
        """Initialize with curated palettes or custom set.

        Args:
            palettes: Custom palette dict; defaults to PALETTES.
        """
        self._palettes = palettes if palettes is not None else PALETTES

    def map(self, emotion: str) -> SoundPalette:
        """Map a single emotion to its sound palette.

        Args:
            emotion: Emotion label (sad, happy, angry, etc.).

        Returns:
            SoundPalette for the emotion, or neutral palette if unknown.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        return self._palettes.get(emotion, self._palettes["neutral"])

    def blend(self, emotions: List[Tuple[str, float]]) -> SoundPalette:
        """Blend multiple emotions into a single palette.

        Used when the EmotionDetector returns secondary emotions with
        confidence scores. Continuous fields (tempo, energy, valence)
        are weighted-averaged. Discrete fields (instruments, tags) are
        unioned with deduplication.

        Args:
            emotions: List of (emotion, weight) tuples. Weights should
                sum to 1.0 for a proper average.

        Returns:
            Blended SoundPalette.

        Complexity: T(n) in O(p * i) for p emotions, i instruments.
        """
        if not emotions:
            return self._palettes["neutral"]

        if len(emotions) == 1:
            return self.map(emotions[0][0])

        # Normalize weights
        total_weight = sum(w for _, w in emotions)
        if total_weight == 0:
            return self._palettes["neutral"]
        normalized = [(e, w / total_weight) for e, w in emotions]

        # Weighted average for continuous fields
        tempo = sum(self.map(e).tempo_bpm * w for e, w in normalized)
        energy = sum(self.map(e).energy * w for e, w in normalized)
        valence = sum(self.map(e).valence * w for e, w in normalized)

        # Union for discrete fields (instruments, tags, sub_genres, references)
        all_instruments: List[str] = []
        all_tags: List[str] = []
        all_sub_genres: List[str] = []
        all_references: List[str] = []
        seen_inst: set = set()
        seen_tag: set = set()
        seen_sub: set = set()
        seen_ref: set = set()

        primary_genre = self.map(normalized[0][0]).genre
        primary_key = self.map(normalized[0][0]).key
        primary_production = self.map(normalized[0][0]).production_style

        for emotion, _ in normalized:
            p = self.map(emotion)
            for inst in p.instruments:
                if inst not in seen_inst:
                    all_instruments.append(inst)
                    seen_inst.add(inst)
            for tag in p.mood_tags:
                if tag not in seen_tag:
                    all_tags.append(tag)
                    seen_tag.add(tag)
            for sg in p.sub_genres:
                if sg not in seen_sub:
                    all_sub_genres.append(sg)
                    seen_sub.add(sg)
            for ref in p.reference_artists:
                if ref not in seen_ref:
                    all_references.append(ref)
                    seen_ref.add(ref)

        return SoundPalette(
            emotion="+".join(e for e, _ in normalized),
            genre=primary_genre,
            sub_genres=tuple(all_sub_genres[:5]),
            tempo_bpm=int(round(tempo)),
            key=primary_key,
            energy=round(energy, 3),
            valence=round(valence, 3),
            instruments=tuple(all_instruments[:8]),
            mood_tags=tuple(all_tags[:10]),
            production_style=primary_production,
            reference_artists=tuple(all_references[:4]),
        )

    def available_emotions(self) -> List[str]:
        """Return list of emotions with defined palettes.

        Complexity: T(n) in O(e) for e emotions.
        """
        return list(self._palettes.keys())

    def __call__(self, emotion: str) -> SoundPalette:
        """Alias for map()."""
        return self.map(emotion)
