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
Regression tests for conversational intelligence quality.

Tests that the system responds with appropriate empathy, acknowledgment,
tone matching, and non-generic output across emotional contexts.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import pytest

from core import BrainController
from core.cognition.emotion_detector import EmotionDetector


@pytest.fixture
def controller() -> BrainController:
    """Provide a booted BrainController."""
    bc = BrainController()
    bc.boot_system()
    return bc


@pytest.fixture
def detector() -> EmotionDetector:
    """Provide an EmotionDetector instance."""
    return EmotionDetector()


# ─── Emotion Detection ───────────────────────────────────────────────────────


class TestEmotionDetection:
    """Tests for the emotion detector."""

    def test_detects_sad(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("I lost my dog yesterday. I'm so sad.")
        assert emotion == "sad"
        assert conf > 0.0

    def test_detects_happy(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("I just got promoted! I'm so happy and excited!")
        assert emotion == "happy"
        assert conf > 0.0

    def test_detects_angry(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("I'm furious! This is unacceptable and I'm angry!")
        assert emotion == "angry"
        assert conf > 0.0

    def test_detects_anxious(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("I'm scared and worried. What if something goes wrong?")
        assert emotion == "anxious"
        assert conf > 0.0

    def test_detects_grateful(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("Thank you so much. I'm really grateful for your help.")
        assert emotion == "grateful"
        assert conf > 0.0

    def test_detects_curious(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("How does that work? I'm curious to learn more.")
        assert emotion == "curious"
        assert conf > 0.0

    def test_detects_philosophical(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("What is the meaning of life? Do we have free will?")
        assert emotion == "philosophical"
        assert conf > 0.0

    def test_detects_personal(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("I'm thinking about quitting my job. My career feels stuck.")
        assert emotion == "personal"
        assert conf > 0.0

    def test_neutral_for_technical(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("Write a Python function that sorts a list.")
        assert emotion == "neutral"

    def test_empty_input(self, detector: EmotionDetector) -> None:
        emotion, conf = detector.detect("")
        assert emotion == "neutral"
        assert conf == 0.0


# ─── Emotion in Pipeline ─────────────────────────────────────────────────────


class TestEmotionInPipeline:
    """Tests that emotion is detected and stored in BrainState."""

    def test_sad_input_gets_sad_emotion(self, controller: BrainController) -> None:
        state = controller.process_request("I lost my father. I'm devastated.")
        assert state.emotion == "sad"

    def test_happy_input_gets_happy_emotion(self, controller: BrainController) -> None:
        state = controller.process_request("I just graduated! Best day ever!")
        assert state.emotion == "happy"

    def test_angry_input_gets_angry_emotion(self, controller: BrainController) -> None:
        state = controller.process_request("I'm furious! This company is exploiting us!")
        assert state.emotion == "angry"

    def test_technical_input_gets_neutral_emotion(self, controller: BrainController) -> None:
        state = controller.process_request("Write a Python function that sorts a list.")
        assert state.emotion == "neutral"


# ─── Empathy in Responses ────────────────────────────────────────────────────


class TestEmpathyInResponses:
    """Tests that responses contain empathy for emotional inputs."""

    def test_sad_response_has_empathy(self, controller: BrainController) -> None:
        state = controller.process_request("I lost my dog yesterday. He was my best friend.")
        output = state.final_output.lower()
        assert any(w in output for w in ["sorry", "difficult", "hear", "understand"])

    def test_angry_response_validates_feelings(self, controller: BrainController) -> None:
        state = controller.process_request("I'm so angry! My coworker stole my work!")
        output = state.final_output.lower()
        assert any(w in output for w in ["frustration", "unfair", "understand", "hear"])

    def test_anxious_response_reassures(self, controller: BrainController) -> None:
        state = controller.process_request("I'm terrified about my job interview tomorrow.")
        output = state.final_output.lower()
        assert any(w in output for w in ["normal", "hear", "understand", "worried"])

    def test_happy_response_celebrates(self, controller: BrainController) -> None:
        state = controller.process_request("I just got promoted! I'm so happy!")
        output = state.final_output.lower()
        assert any(w in output for w in ["fantastic", "happy", "great", "wonderful", "excited"])


# ─── No Generic Metadata in Output ───────────────────────────────────────────


class TestNoGenericMetadata:
    """Tests that internal metadata doesn't leak into user-facing output."""

    def test_no_mode_label(self, controller: BrainController) -> None:
        state = controller.process_request("Write a Python function that sorts a list.")
        assert "Mode:" not in state.final_output

    def test_no_domain_label(self, controller: BrainController) -> None:
        state = controller.process_request("Explain what a closure is.")
        assert "Domain:" not in state.final_output

    def test_no_intent_label(self, controller: BrainController) -> None:
        state = controller.process_request("What is a decorator?")
        assert "Intent:" not in state.final_output

    def test_no_reasoning_label(self, controller: BrainController) -> None:
        state = controller.process_request("Analyze my code for bugs.")
        assert "[Reasoning]" not in state.final_output

    def test_no_confidence_value(self, controller: BrainController) -> None:
        state = controller.process_request("Explain closures.")
        assert "confidence=" not in state.final_output


# ─── Tone Adaptation ─────────────────────────────────────────────────────────


class TestToneAdaptation:
    """Tests that the personality engine adapts tone based on emotion."""

    def test_sad_gets_soft_greeting(self, controller: BrainController) -> None:
        state = controller.process_request("My mother passed away. I'm heartbroken.")
        # Should NOT start with a cheerful greeting
        assert not state.final_output.startswith("Hei!")
        # Should have an empathetic, soft tone (not cheerful)
        output_lower = state.final_output.lower()
        assert any(w in output_lower for w in ["sorry", "here for you", "courage", "struggling", "heartbroken", "difficult"])

    def test_happy_keeps_cheerful_greeting(self, controller: BrainController) -> None:
        state = controller.process_request("I just got engaged! So excited!")
        # Empathy prefix for happy emotion
        assert any(w in state.final_output.lower() for w in ["great", "happy", "fantastic", "wonderful", "excited"])

    def test_personal_gets_listening_greeting(self, controller: BrainController) -> None:
        state = controller.process_request("I'm thinking about quitting my job. I don't know what to do.")
        assert state.final_output.startswith("I'm listening.") or state.final_output.startswith("I'm here")


# ─── Acknowledgment ──────────────────────────────────────────────────────────


class TestAcknowledgment:
    """Tests that responses acknowledge the user's input."""

    def test_response_references_user_input(self, controller: BrainController) -> None:
        state = controller.process_request("I'm scared about my health checkup tomorrow.")
        # The response should reference keywords from the input
        output_lower = state.final_output.lower()
        assert "health" in output_lower or "checkup" in output_lower or "scared" in output_lower

    def test_response_quotes_user_input(self, controller: BrainController) -> None:
        state = controller.process_request("I just adopted a puppy and I'm so happy!")
        # The response should acknowledge the happy event (celebratory or referencing input)
        output_lower = state.final_output.lower()
        assert any(w in output_lower for w in ["puppy", "adopted", "wonderful", "happy", "celebrat", "exciting", "great news"])


# ─── Multi-Turn Emotional Conversation ───────────────────────────────────────


class TestMultiTurnEmotionalConversation:
    """Tests for emotional coherence across multi-turn conversations."""

    def test_emotion_shifts_with_input(self, controller: BrainController) -> None:
        """Emotion should change as conversation shifts from sad to happy."""
        s1 = controller.process_request("I lost my friend. I'm so sad.")
        assert s1.emotion == "sad"

        s2 = controller.process_request("But then I got a new job! I'm so happy!")
        assert s2.emotion == "happy"

    def test_responses_differ_across_emotions(self, controller: BrainController) -> None:
        """Responses for different emotions should be different."""
        s1 = controller.process_request("I'm devastated. My dog died.")
        s2 = controller.process_request("I'm thrilled! I won the lottery!")
        assert s1.final_output != s2.final_output

    def test_five_emotional_turns_all_unique(self, controller: BrainController) -> None:
        """Five emotionally different inputs should produce unique outputs."""
        inputs = [
            "I'm so happy! I just graduated!",
            "I'm devastated. My cat died today.",
            "I'm furious! Someone stole my car!",
            "I'm anxious about my surgery tomorrow.",
            "I'm grateful for my family's support.",
        ]
        outputs = []
        for inp in inputs:
            state = controller.process_request(inp)
            outputs.append(state.final_output)
        assert len(set(outputs)) == 5, f"Only {len(set(outputs))}/5 outputs were unique"
