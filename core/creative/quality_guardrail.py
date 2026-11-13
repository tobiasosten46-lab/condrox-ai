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
Quality Guardrail.

Evaluates conversation quality before output reaches a live user.
Acts as a gate: if quality is below threshold, the output is either
enriched, replaced with a fallback, or flagged for review.

Uses the same metrics as the conversation intelligence simulation:
  - Empathy (emotional awareness)
  - Acknowledgment (references user input)
  - Relevance (keyword overlap with user input)
  - Tone appropriateness (emotion-aware tone matching)
  - Not generic (no internal metadata leakage)
  - No repetition (not identical to prior outputs)
  - Minimum length

Mathematical Foundation:
- Quality score q = sum(check_passes) / total_checks. O(c) for c checks.
- Gate decision: pass if q >= threshold AND no critical failures.
- Complexity: T(n) in O(n + c) for n output tokens, c checks.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Reuse indicators from the simulation (kept in sync)
EMPATHY_INDICATORS: List[str] = [
    "sorry", "understand", "feel", "must be", "that's", "difficult",
    "hard", "tough", "challenging", "congratulations", "wonderful",
    "amazing", "great", "happy for you", "proud", "support",
    "care", "important", "valid", "normal", "okay", "alright",
    "hear you", "with you", "there for", "strength", "courage",
    "brave", "inspiring", "heartbreaking", "devastating", "exciting",
    "thrilled", "concerning", "worried", "anxious", "fear",
    "love", "hope", "grateful", "thank", "appreciate",
    "fantastic", "unfair", "frustration", "listening", "here for you",
]

ACKNOWLEDGMENT_INDICATORS: List[str] = [
    "i hear", "i understand", "i see", "that's", "it sounds",
    "you mentioned", "you said", "you're feeling", "you feel",
    "your", "this is about", "regarding", "about your",
    "you asked", "you want", "you'd like", "you're thinking",
    "i hear you", "talking about", "you just",
]

GENERIC_INDICATORS: List[str] = [
    "mode:", "domain:", "intent:", "reasoning]", "confidence=",
    "[no output", "content blocked", "[code]", "[explain]",
    "[analyze]", "[architect]", "[clarify]", "[default]",
]

# Stopwords for relevance scoring
STOPWORDS: set = {
    "the", "a", "an", "is", "was", "i", "my", "me", "to", "and",
    "but", "so", "it", "for", "on", "in", "at", "of", "with",
    "that", "this", "just", "really", "very", "about", "like",
    "what", "how", "why", "when", "where", "do", "does", "did",
    "can", "could", "would", "should", "will", "have", "has",
}


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Immutable quality assessment result.

    Attributes:
        score: Overall quality score in [0.0, 1.0].
        passed: Whether the output passed the guardrail gate.
        checks: Individual check results.
        issues: List of issue labels for failed checks.
        critical_failures: Checks that must not fail (empty = OK).
        recommendation: Suggested action (pass, enrich, replace, review).

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    score: float
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict, hash=False)
    issues: Tuple[str, ...] = ()
    critical_failures: Tuple[str, ...] = ()
    recommendation: str = "pass"


class QualityGuardrail:
    """Evaluate conversation quality before live user delivery.

    Complexity:
        assess: T(n) in O(n + c) for n output tokens, c checks.
    """

    __slots__ = ("_threshold", "_min_length", "_critical_checks")

    def __init__(
        self,
        threshold: float = 0.45,
        min_length: int = 20,
        critical_checks: Optional[List[str]] = None,
    ) -> None:
        """Initialize the guardrail.

        Args:
            threshold: Minimum quality score to pass (default 0.6).
            min_length: Minimum output length in characters.
            critical_checks: Checks that must pass regardless of score.
        """
        self._threshold = threshold
        self._min_length = min_length
        self._critical_checks = critical_checks or [
            "not_generic",
            "no_repetition",
            "min_length",
        ]

    def assess(
        self,
        output: str,
        user_input: str,
        emotion: str = "neutral",
        prior_outputs: Optional[List[str]] = None,
    ) -> QualityAssessment:
        """Assess the quality of a generated output.

        Args:
            output: The generated response text.
            user_input: The user's original input.
            emotion: Detected emotional tone.
            prior_outputs: Previous outputs in this conversation.

        Returns:
            QualityAssessment with score, checks, and recommendation.

        Complexity: T(n) in O(n + c) for n tokens, c checks.
        """
        output_lower = output.lower()
        user_lower = user_input.lower()
        prior = prior_outputs or []

        checks: Dict[str, bool] = {}
        issues: List[str] = []

        # 1. Empathy — response contains emotional awareness
        checks["empathy"] = any(ind in output_lower for ind in EMPATHY_INDICATORS)
        if not checks["empathy"]:
            issues.append("NO_EMPATHY")

        # 2. Acknowledgment — response acknowledges user input
        checks["acknowledgment"] = any(ind in output_lower for ind in ACKNOWLEDGMENT_INDICATORS)
        if not checks["acknowledgment"]:
            issues.append("NO_ACKNOWLEDGMENT")

        # 3. Relevance — at least 2 content-word overlaps
        user_words = set(user_lower.split()) - STOPWORDS
        output_words = set(output_lower.split())
        overlap = user_words & output_words
        checks["relevance"] = len(overlap) >= 2
        if not checks["relevance"]:
            issues.append("NOT_RELEVANT")

        # 4. Tone appropriate — emotion-aware tone check
        checks["tone_appropriate"] = self._check_tone(output_lower, emotion)
        if not checks["tone_appropriate"]:
            issues.append("TONE_MISMATCH")

        # 5. Not generic — no internal metadata
        checks["not_generic"] = not any(ind in output_lower for ind in GENERIC_INDICATORS)
        if not checks["not_generic"]:
            issues.append("GENERIC_RESPONSE")

        # 6. No repetition — not identical to any prior output
        checks["no_repetition"] = output not in prior
        if not checks["no_repetition"]:
            issues.append("REPETITION")

        # 7. Minimum length
        checks["min_length"] = len(output) >= self._min_length
        if not checks["min_length"]:
            issues.append("TOO_SHORT")

        # Calculate score
        score = sum(checks.values()) / len(checks) if checks else 0.0

        # Identify critical failures
        critical_fails: List[str] = []
        for cc in self._critical_checks:
            if cc in checks and not checks[cc]:
                critical_fails.append(cc)

        # Determine pass/fail
        passed = score >= self._threshold and len(critical_fails) == 0

        # Recommendation
        if passed:
            recommendation = "pass"
        elif len(critical_fails) > 0:
            recommendation = "replace"
        elif score >= self._threshold - 0.15:
            recommendation = "enrich"
        else:
            recommendation = "replace"

        return QualityAssessment(
            score=round(score, 3),
            passed=passed,
            checks=dict(checks),
            issues=tuple(issues),
            critical_failures=tuple(critical_fails),
            recommendation=recommendation,
        )

    def _check_tone(self, output_lower: str, emotion: str) -> bool:
        """Check if response tone is appropriate for the emotion.

        Args:
            output_lower: Lowercase output text.
            emotion: Detected emotion.

        Returns:
            True if tone is appropriate.

        Complexity: T(n) in O(1) — keyword checks.
        """
        if emotion in ("sad", "angry", "anxious", "personal"):
            # Should have empathetic/validating language
            negative_words = [
                "sorry", "understand", "difficult", "hard", "challenging",
                "concerning", "unfair", "frustration", "hear", "worried",
                "normal", "here for you", "listening",
            ]
            return any(w in output_lower for w in negative_words)
        elif emotion in ("happy", "grateful"):
            # Should have positive language
            positive_words = [
                "fantastic", "wonderful", "amazing", "congratulations",
                "happy", "exciting", "great", "love", "proud", "thankful",
                "grateful", "appreciate",
            ]
            return any(w in output_lower for w in positive_words)
        elif emotion in ("curious", "confused"):
            # Should be informative or acknowledging
            return ("help" in output_lower or "clarif" in output_lower
                    or "understand" in output_lower or "let me" in output_lower
                    or "here's" in output_lower)
        elif emotion in ("philosophical",):
            # Should engage with the question
            return ("think" in output_lower or "question" in output_lower
                    or "thoughts" in output_lower or "deep" in output_lower
                    or "share" in output_lower)
        # neutral or unknown — always pass
        return True

    def gate(
        self,
        output: str,
        user_input: str,
        emotion: str = "neutral",
        prior_outputs: Optional[List[str]] = None,
        fallback: str = "",
    ) -> Tuple[str, QualityAssessment]:
        """Gate an output: pass, enrich, or replace.

        Args:
            output: Generated response.
            user_input: User's input.
            emotion: Detected emotion.
            prior_outputs: Previous outputs.
            fallback: Fallback text if output is rejected.

        Returns:
            Tuple of (final_output, assessment).

        Complexity: T(n) in O(n + c).
        """
        assessment = self.assess(output, user_input, emotion, prior_outputs)

        if assessment.recommendation == "pass":
            return output, assessment
        elif assessment.recommendation == "enrich":
            # Try to enrich by prepending an empathy prefix
            enriched = self._enrich(output, emotion)
            re_assessment = self.assess(enriched, user_input, emotion, prior_outputs)
            if re_assessment.passed:
                return enriched, re_assessment
            # Fall through to replace if enrichment didn't help
        # Replace with fallback
        if fallback:
            return fallback, assessment
        # Last resort: return original with warning
        return output, assessment

    def _enrich(self, output: str, emotion: str) -> str:
        """Enrich output with empathy prefix.

        Args:
            output: Original output.
            emotion: Detected emotion.

        Returns:
            Enriched output.

        Complexity: T(n) in O(1).
        """
        empathy_prefixes = {
            "sad": "I'm sorry you're going through this. ",
            "angry": "I understand your frustration. ",
            "anxious": "It's completely normal to feel this way. ",
            "personal": "Thank you for sharing. ",
            "happy": "That's wonderful! ",
            "grateful": "That's beautiful. ",
            "philosophical": "That's a profound thought. ",
            "confused": "",
            "curious": "Great question! ",
        }
        prefix = empathy_prefixes.get(emotion, "")
        return prefix + output
