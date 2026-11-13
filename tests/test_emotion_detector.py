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
Comprehensive tests for EmotionDetector (core/cognition/emotion_detector.py).

Tests cover:
- Each of the 10 emotion categories (sad, angry, happy, anxious, grateful,
  confused, curious, philosophical, personal, neutral)
- Confidence scores are in [0.0, 1.0]
- Priority-based tie-breaking
- Empty input returns neutral
- Mixed emotions detect one dominant emotion
- The "terrified" keyword maps to "anxious" not "personal"

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

from core.cognition.emotion_detector import EmotionDetector


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def detector() -> EmotionDetector:
    """Provide a fresh EmotionDetector instance."""
    return EmotionDetector()


# ─── Individual Emotion Detection ────────────────────────────────────────────


class TestEmotionDetection:
    """Tests that each of the 10 emotions is correctly detected."""

    def test_detect_sad(self, detector: EmotionDetector) -> None:
        """Sad emotion is detected from grief/loss keywords."""
        emotion, confidence = detector.detect(
            "I feel so sad and heartbroken after the funeral"
        )
        assert emotion == "sad"
        assert 0.0 <= confidence <= 1.0

    def test_detect_angry(self, detector: EmotionDetector) -> None:
        """Angry emotion is detected from fury/outrage keywords."""
        emotion, confidence = detector.detect(
            "I am furious and outraged at this unacceptable behavior"
        )
        assert emotion == "angry"
        assert 0.0 <= confidence <= 1.0

    def test_detect_happy(self, detector: EmotionDetector) -> None:
        """Happy emotion is detected from joy/excitement keywords."""
        emotion, confidence = detector.detect(
            "I am so happy and excited, this is amazing and wonderful"
        )
        assert emotion == "happy"
        assert 0.0 <= confidence <= 1.0

    def test_detect_anxious(self, detector: EmotionDetector) -> None:
        """Anxious emotion is detected from fear/worry keywords."""
        emotion, confidence = detector.detect(
            "I am scared and worried, feeling anxious about everything"
        )
        assert emotion == "anxious"
        assert 0.0 <= confidence <= 1.0

    def test_detect_grateful(self, detector: EmotionDetector) -> None:
        """Grateful emotion is detected from thankfulness keywords."""
        emotion, confidence = detector.detect(
            "I am so thankful and grateful for all the support"
        )
        assert emotion == "grateful"
        assert 0.0 <= confidence <= 1.0

    def test_detect_confused(self, detector: EmotionDetector) -> None:
        """Confused emotion is detected from bewilderment keywords."""
        emotion, confidence = detector.detect(
            "I am confused, this doesn't make sense and I don't understand"
        )
        assert emotion == "confused"
        assert 0.0 <= confidence <= 1.0

    def test_detect_curious(self, detector: EmotionDetector) -> None:
        """Curious emotion is detected from inquiry keywords."""
        emotion, confidence = detector.detect(
            "I am curious, tell me about how this works, it's fascinating"
        )
        assert emotion == "curious"
        assert 0.0 <= confidence <= 1.0

    def test_detect_philosophical(self, detector: EmotionDetector) -> None:
        """Philosophical emotion is detected from existential keywords."""
        emotion, confidence = detector.detect(
            "What is the meaning of life and consciousness"
        )
        assert emotion == "philosophical"
        assert 0.0 <= confidence <= 1.0

    def test_detect_personal(self, detector: EmotionDetector) -> None:
        """Personal emotion is detected from self-referential keywords."""
        emotion, confidence = detector.detect(
            "I'm thinking about my career and my future"
        )
        assert emotion == "personal"
        assert 0.0 <= confidence <= 1.0

    def test_detect_neutral_no_keywords(self, detector: EmotionDetector) -> None:
        """Neutral is returned when no emotion keywords match."""
        emotion, confidence = detector.detect("The sky is blue today")
        assert emotion == "neutral"
        assert confidence == 0.0


# ─── Confidence Range ────────────────────────────────────────────────────────


class TestConfidenceRange:
    """Tests that confidence scores are always in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        "text",
        [
            "I am so happy and excited",
            "I feel sad and heartbroken",
            "I am furious and outraged",
            "I am scared and worried",
            "I am thankful and grateful",
            "I am confused and don't understand",
            "I am curious, tell me more",
            "What is the meaning of life",
            "I'm thinking about my career",
            "The sky is blue",
        ],
    )
    def test_confidence_in_range(self, detector: EmotionDetector, text: str) -> None:
        """Confidence is always in [0.0, 1.0] for any input."""
        _, confidence = detector.detect(text)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_is_proportion(self, detector: EmotionDetector) -> None:
        """Confidence equals top_score / total_score."""
        emotion, confidence = detector.detect("happy happy happy")
        # All keywords belong to happy, so confidence should be 1.0
        assert emotion == "happy"
        assert confidence == pytest.approx(1.0)

    def test_confidence_partial_when_mixed(self, detector: EmotionDetector) -> None:
        """Confidence is less than 1.0 when multiple emotions score."""
        emotion, confidence = detector.detect("I am happy but also sad")
        assert emotion in ("happy", "sad")
        # Two emotions each score 1, so confidence = 1/2 = 0.5
        assert confidence == pytest.approx(0.5)


# ─── Priority-Based Tie-Breaking ─────────────────────────────────────────────


class TestPriorityTieBreaking:
    """Tests for priority-based tie-breaking when scores are equal."""

    def test_sad_beats_happy_on_tie(self, detector: EmotionDetector) -> None:
        """Sad has higher priority than happy, so it wins on a tie."""
        emotion, confidence = detector.detect("I am happy but also sad")
        # Both "happy" and "sad" score 1; sad is first in priority list
        assert emotion == "sad"

    def test_anxious_beats_happy_on_tie(self, detector: EmotionDetector) -> None:
        """Anxious has higher priority than happy, so it wins on a tie."""
        emotion, confidence = detector.detect("I'm happy but also worried")
        # "happy" scores 1, "worried" scores 1 for anxious
        # Priority: anxious (index 2) < happy (index 8)
        assert emotion == "anxious"

    def test_angry_beats_anxious_on_tie(self, detector: EmotionDetector) -> None:
        """Angry has higher priority than anxious, so it wins on a tie."""
        emotion, confidence = detector.detect("I am angry and scared")
        # "angry" scores 1 for angry, "scared" scores 1 for anxious
        # Priority: angry (index 1) < anxious (index 2)
        assert emotion == "angry"

    def test_priority_order_is_deterministic(self, detector: EmotionDetector) -> None:
        """The same tied input always produces the same emotion."""
        text = "I am happy but also sad"
        results = [detector.detect(text) for _ in range(5)]
        first_emotion = results[0][0]
        for emotion, _ in results:
            assert emotion == first_emotion


# ─── Empty Input ─────────────────────────────────────────────────────────────


class TestEmptyInput:
    """Tests for empty and whitespace-only input."""

    def test_empty_string_returns_neutral(self, detector: EmotionDetector) -> None:
        """Empty string returns ('neutral', 0.0)."""
        emotion, confidence = detector.detect("")
        assert emotion == "neutral"
        assert confidence == 0.0

    def test_whitespace_only_returns_neutral(self, detector: EmotionDetector) -> None:
        """Whitespace-only string returns ('neutral', 0.0)."""
        emotion, confidence = detector.detect("   \t\n  ")
        assert emotion == "neutral"
        assert confidence == 0.0

    def test_none_input_returns_neutral(self, detector: EmotionDetector) -> None:
        """None input returns ('neutral', 0.0) — detect handles falsy input."""
        emotion, confidence = detector.detect(None)  # type: ignore[arg-type]
        assert emotion == "neutral"
        assert confidence == 0.0


# ─── Mixed Emotions ──────────────────────────────────────────────────────────


class TestMixedEmotions:
    """Tests for mixed-emotion inputs detecting one dominant emotion."""

    def test_happy_but_worried_detects_dominant(self, detector: EmotionDetector) -> None:
        """'I'm happy but also worried' detects one dominant emotion."""
        emotion, confidence = detector.detect("I'm happy but also worried")
        assert emotion != "neutral"
        assert confidence > 0.0
        # Only one emotion should be returned (the dominant one)
        assert isinstance(emotion, str)

    def test_sad_but_grateful_detects_dominant(self, detector: EmotionDetector) -> None:
        """'I'm sad but also grateful' detects one dominant emotion."""
        emotion, confidence = detector.detect(
            "I'm sad and heartbroken but also grateful and thankful"
        )
        assert emotion != "neutral"
        assert confidence > 0.0

    def test_mixed_returns_single_emotion(self, detector: EmotionDetector) -> None:
        """Mixed input always returns exactly one emotion string."""
        emotion, _ = detector.detect(
            "I am happy and excited but also scared and worried"
        )
        assert isinstance(emotion, str)
        assert len(emotion) > 0


# ─── Terrified Maps to Anxious ───────────────────────────────────────────────


class TestTerrifiedMapping:
    """Tests that 'terrified' maps to 'anxious', not 'personal'.

    This is a regression test for a previously fixed bug where 'terrified'
    was incorrectly mapped to the 'personal' emotion category.
    """

    def test_terrified_alone_maps_to_anxious(self, detector: EmotionDetector) -> None:
        """The word 'terrified' alone maps to 'anxious'."""
        emotion, confidence = detector.detect("I am terrified")
        assert emotion == "anxious"
        assert emotion != "personal"
        assert confidence > 0.0

    def test_terrified_not_personal(self, detector: EmotionDetector) -> None:
        """'terrified' must never produce 'personal'."""
        emotion, _ = detector.detect("I am terrified")
        assert emotion != "personal"

    def test_terrified_with_personal_context_still_anxious(
        self, detector: EmotionDetector
    ) -> None:
        """Even with personal context words, 'terrified' keeps anxious dominant."""
        emotion, confidence = detector.detect("I am terrified about my future")
        # "terrified" → anxious (1), "my future" → personal (1)
        # Priority: anxious (index 2) < personal (index 4)
        assert emotion == "anxious"
        assert emotion != "personal"


# ─── Call Alias ──────────────────────────────────────────────────────────────


class TestCallAlias:
    """Tests that __call__ is an alias for detect()."""

    def test_call_matches_detect(self, detector: EmotionDetector) -> None:
        """__call__ returns the same result as detect()."""
        text = "I am so happy and excited"
        detect_result = detector.detect(text)
        call_result = detector(text)
        assert detect_result == call_result
