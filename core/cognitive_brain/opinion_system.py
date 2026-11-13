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

"""Opinion System — erfaringens vurdering.

Mening er ikke subjektivitet.
Mening er erfaringens vurdering:
- dette temaet er viktig
- dette mønsteret er interessant
- denne strategien fungerer
- dette må reflekteres mer over
- dette er et brudd som må undersøkes

Mening = prioritet basert på erfaring.

Complexity:
    form_opinion: T(n) ∈ O(1)
    get_opinion: T(n) ∈ O(1)
    get_opinions_by_domain: T(n) ∈ O(o) for o opinions

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(slots=True)
class Opinion:
    """A cognitive opinion — an experience-based assessment.

    Opinions are not subjective. They are assessments based on
    accumulated experience with a topic, pattern, or strategy.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    topic: str               # What the opinion is about
    assessment: str          # The assessment (e.g., "important", "unclear", "working")
    confidence: float        # How confident [0.0, 1.0]
    domain: str              # Knowledge domain
    evidence_count: int = 0  # How many experiences support this
    last_updated: float = field(default_factory=time.time)
    supporting_evidence: List[str] = field(default_factory=list)  # Brief evidence descriptions

    def add_evidence(self, evidence: str, confidence_delta: float = 0.05) -> None:
        """Add supporting evidence to this opinion.

        Complexity: T(n) ∈ O(1).
        """
        self.evidence_count += 1
        self.confidence = min(1.0, self.confidence + confidence_delta)
        self.last_updated = time.time()
        self.supporting_evidence.append(evidence)
        if len(self.supporting_evidence) > 10:
            self.supporting_evidence = self.supporting_evidence[-10:]


class OpinionSystem:
    """Cognitive opinion system — erfaringens vurdering.

    The opinion system forms assessments based on accumulated experience.
    These are not subjective opinions — they are experience-based
    priorities that shape how the system allocates attention.

    Opinion types:
    - "important" — This topic is valuable and should be prioritized
    - "interesting" — This pattern is worth investigating
    - "unclear" — This topic needs more understanding
    - "working" — This strategy is effective
    - "not_working" — This strategy is ineffective
    - "breakthrough" — This is a significant discovery
    - "needs_reflection" — This needs more thought
    - "solid" — This knowledge is well-established

    Opinions influence:
    - Response depth (important topics get deeper responses)
    - Learning priority (unclear topics get more learning effort)
    - Strategy selection (working strategies are preferred)
    - Follow-up questions (interesting topics get exploratory follow-ups)

    Thread-safety: All mutations guarded by a reentrant lock.

    Complexity:
        form_opinion: T(n) ∈ O(1)
        get_opinion: T(n) ∈ O(1)
        get_opinions_by_domain: T(n) ∈ O(o) for o opinions
    """

    __slots__ = ("_opinions", "_lock")

    # Valid assessment types
    _VALID_ASSESSMENTS = {
        "important", "interesting", "unclear", "working", "not_working",
        "breakthrough", "needs_reflection", "solid", "emerging", "deprecated",
    }

    def __init__(self) -> None:
        """Initialize the opinion system."""
        self._opinions: Dict[str, Opinion] = {}
        self._lock = threading.RLock()

    def form_opinion(
        self,
        topic: str,
        assessment: str,
        domain: str = "general",
        confidence: float = 0.5,
        evidence: str = "",
    ) -> Opinion:
        """Form or update an opinion about a topic.

        Args:
            topic: What the opinion is about.
            assessment: The assessment type.
            domain: Knowledge domain.
            confidence: Initial confidence [0.0, 1.0].
            evidence: Supporting evidence description.

        Returns:
            The formed/updated Opinion.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            topic_key = topic.lower().strip()
            existing = self._opinions.get(topic_key)

            if existing:
                # Update existing opinion
                if existing.assessment != assessment:
                    # Assessment changed — reset confidence somewhat
                    existing.assessment = assessment
                    existing.confidence = max(0.3, confidence)
                if evidence:
                    existing.add_evidence(evidence)
                return existing
            else:
                # Create new opinion
                opinion = Opinion(
                    topic=topic,
                    assessment=assessment,
                    confidence=max(0.0, min(1.0, confidence)),
                    domain=domain,
                    evidence_count=1 if evidence else 0,
                    supporting_evidence=[evidence] if evidence else [],
                )
                self._opinions[topic_key] = opinion
                return opinion

    def reinforce_opinion(
        self,
        topic: str,
        evidence: str,
        confidence_delta: float = 0.05,
    ) -> Optional[Opinion]:
        """Reinforce an existing opinion with new evidence.

        Args:
            topic: The opinion topic.
            evidence: New supporting evidence.
            confidence_delta: How much to increase confidence.

        Returns:
            Updated opinion if found, None otherwise.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            topic_key = topic.lower().strip()
            opinion = self._opinions.get(topic_key)
            if opinion is None:
                return None
            opinion.add_evidence(evidence, confidence_delta)
            return opinion

    def get_opinion(self, topic: str) -> Optional[Opinion]:
        """Get the opinion about a topic.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return self._opinions.get(topic.lower().strip())

    def get_opinions_by_domain(self, domain: str) -> List[Opinion]:
        """Get all opinions for a knowledge domain.

        Complexity: T(n) ∈ O(o) for o opinions.
        """
        with self._lock:
            return [o for o in self._opinions.values() if o.domain == domain]

    def get_opinions_by_assessment(self, assessment: str) -> List[Opinion]:
        """Get all opinions with a specific assessment.

        Complexity: T(n) ∈ O(o) for o opinions.
        """
        with self._lock:
            return [o for o in self._opinions.values() if o.assessment == assessment]

    def get_important_topics(self, limit: int = 10) -> List[Opinion]:
        """Get topics assessed as important, sorted by confidence.

        Complexity: T(n) ∈ O(o log o) for o opinions.
        """
        with self._lock:
            important = [
                o for o in self._opinions.values()
                if o.assessment in ("important", "breakthrough", "solid")
            ]
            important.sort(key=lambda o: o.confidence, reverse=True)
            return important[:limit]

    def get_unclear_topics(self, limit: int = 10) -> List[Opinion]:
        """Get topics assessed as unclear or needing reflection.

        Complexity: T(n) ∈ O(o log o) for o opinions.
        """
        with self._lock:
            unclear = [
                o for o in self._opinions.values()
                if o.assessment in ("unclear", "needs_reflection")
            ]
            unclear.sort(key=lambda o: o.confidence, reverse=True)
            return unclear[:limit]

    def get_working_strategies(self) -> List[Opinion]:
        """Get strategies assessed as working.

        Complexity: T(n) ∈ O(o) for o opinions.
        """
        with self._lock:
            return [
                o for o in self._opinions.values()
                if o.assessment == "working"
            ]

    def get_opinion_summary(self, topic: str) -> str:
        """Get a human-readable summary of the opinion about a topic.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            opinion = self._opinions.get(topic.lower().strip())
            if opinion is None:
                return f"No formed opinion about '{topic}' yet."
            return (
                f"I assess '{topic}' as {opinion.assessment} "
                f"(confidence: {opinion.confidence:.0%}, "
                f"based on {opinion.evidence_count} experiences)."
            )

    def get_stats(self) -> Dict[str, float]:
        """Get opinion system statistics.

        Complexity: T(n) ∈ O(o) for o opinions.
        """
        with self._lock:
            by_assessment: Dict[str, int] = {}
            for o in self._opinions.values():
                by_assessment[o.assessment] = by_assessment.get(o.assessment, 0) + 1
            return {
                "total_opinions": len(self._opinions),
                "important": by_assessment.get("important", 0),
                "unclear": by_assessment.get("unclear", 0),
                "working": by_assessment.get("working", 0),
                "breakthrough": by_assessment.get("breakthrough", 0),
                "solid": by_assessment.get("solid", 0),
            }

    @property
    def opinion_count(self) -> int:
        """Get total number of opinions."""
        with self._lock:
            return len(self._opinions)
