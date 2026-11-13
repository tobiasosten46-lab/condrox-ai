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

"""Experience Memory — syntetisk erfaring.

Systemet lagrer:
- alt det lærer
- alt det opplever
- alt det reflekterer over
- alt det analyserer
- alle mønstre
- alle brudd
- alle strategier
- alle relasjoner

Erfaring er ikke "data". Erfaring er kognitiv struktur.

Complexity:
    record_experience: T(n) ∈ O(1)
    get_relevant_experiences: T(n) ∈ O(e) for e experiences
    get_experience_by_domain: T(n) ∈ O(e) for e experiences

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Experience:
    """A single experience — a unit of cognitive memory.

    Experiences are not just data points. They are structured
    memories that include what happened, what was learned,
    and how it connects to other experiences.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    experience_id: str
    timestamp: float
    experience_type: str    # "conversation", "learning", "reflection", "breakthrough", "pattern"
    description: str        # What happened
    what_was_learned: str   # What was learned
    domain: str             # Knowledge domain
    intent: str             # Associated intent
    emotion: str            # Associated emotion
    reward_value: float     # Cognitive reward received
    related_experiences: Tuple[str, ...] = ()  # IDs of related experiences
    topics: Tuple[str, ...] = ()  # Topics involved
    confidence: float = 0.5  # How confident the system is about this experience


class ExperienceMemory:
    """Synthetic experience memory — kognitiv struktur.

    The experience memory stores everything the system experiences
    as structured cognitive memories. These are not just data —
    they are the system's accumulated experience that shapes
    future thinking.

    Experience types:
    - "conversation" — A conversation interaction
    - "learning" — Knowledge was learned or validated
    - "reflection" — The system reflected on something
    - "breakthrough" — A major discovery or insight
    - "pattern" — A pattern was recognized
    - "strategy" — A strategy was used and evaluated
    - "clarification" — Something unclear became clear
    - "connection" — Two concepts were connected

    Experiences form a network — each experience can be related
    to others, creating a web of cognitive structure.

    Thread-safety: All mutations guarded by a reentrant lock.

    Complexity:
        record_experience: T(n) ∈ O(1)
        get_relevant_experiences: T(n) ∈ O(e) for e experiences
        get_experience_by_domain: T(n) ∈ O(e) for e experiences
    """

    __slots__ = ("_experiences", "_lock", "_counter", "_by_domain", "_by_type", "_by_topic", "_max_experiences")

    def __init__(self, max_experiences: int = 50000) -> None:
        """Initialize the experience memory.

        Args:
            max_experiences: Maximum experiences to keep in memory.
        """
        self._experiences: Dict[str, Experience] = {}
        self._lock = threading.RLock()
        self._counter: int = 0
        self._by_domain: Dict[str, List[str]] = {}  # domain → experience IDs
        self._by_type: Dict[str, List[str]] = {}    # type → experience IDs
        self._by_topic: Dict[str, List[str]] = {}   # topic → experience IDs
        self._max_experiences = max_experiences

    def record_experience(
        self,
        experience_type: str,
        description: str,
        what_was_learned: str = "",
        domain: str = "general",
        intent: str = "unknown",
        emotion: str = "neutral",
        reward_value: float = 0.0,
        topics: Optional[List[str]] = None,
        confidence: float = 0.5,
        related_experiences: Optional[List[str]] = None,
    ) -> Experience:
        """Record a new experience.

        Args:
            experience_type: Type of experience.
            description: What happened.
            what_was_learned: What was learned.
            domain: Knowledge domain.
            intent: Associated intent.
            emotion: Associated emotion.
            reward_value: Cognitive reward received.
            topics: Topics involved.
            confidence: Confidence in this experience.
            related_experiences: IDs of related experiences.

        Returns:
            The recorded Experience.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._counter += 1
            exp_id = f"exp_{self._counter}"
            experience = Experience(
                experience_id=exp_id,
                timestamp=time.time(),
                experience_type=experience_type,
                description=description,
                what_was_learned=what_was_learned,
                domain=domain,
                intent=intent,
                emotion=emotion,
                reward_value=reward_value,
                topics=tuple(topics or []),
                confidence=confidence,
                related_experiences=tuple(related_experiences or []),
            )
            self._experiences[exp_id] = experience

            # Index by domain
            if domain not in self._by_domain:
                self._by_domain[domain] = []
            self._by_domain[domain].append(exp_id)

            # Index by type
            if experience_type not in self._by_type:
                self._by_type[experience_type] = []
            self._by_type[experience_type].append(exp_id)

            # Index by topics
            for topic in (topics or []):
                topic_lower = topic.lower()
                if topic_lower not in self._by_topic:
                    self._by_topic[topic_lower] = []
                self._by_topic[topic_lower].append(exp_id)

            # Prune if over limit
            if len(self._experiences) > self._max_experiences:
                self._prune()

            return experience

    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """Get a specific experience by ID.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return self._experiences.get(experience_id)

    def get_relevant_experiences(
        self,
        topics: List[str],
        domain: str = "general",
        limit: int = 5,
    ) -> List[Experience]:
        """Get experiences relevant to given topics and domain.

        Args:
            topics: Topics to search for.
            domain: Knowledge domain to filter by.
            limit: Maximum number of experiences to return.

        Returns:
            List of relevant experiences, sorted by relevance.

        Complexity: T(n) ∈ O(e) for e experiences.
        """
        with self._lock:
            relevant: List[Tuple[Experience, float]] = []

            for exp in self._experiences.values():
                score = 0.0

                # Domain match
                if exp.domain == domain:
                    score += 0.3

                # Topic overlap
                exp_topics_lower = {t.lower() for t in exp.topics}
                query_topics_lower = {t.lower() for t in topics}
                overlap = len(exp_topics_lower & query_topics_lower)
                if overlap > 0:
                    score += overlap * 0.2

                # Reward value (higher reward = more relevant)
                score += exp.reward_value * 0.1

                # Recency (more recent = slightly more relevant)
                age = time.time() - exp.timestamp
                recency_score = max(0.0, 1.0 - age / 86400)  # Decay over 24 hours
                score += recency_score * 0.1

                if score > 0.1:
                    relevant.append((exp, score))

            relevant.sort(key=lambda x: x[1], reverse=True)
            return [exp for exp, _ in relevant[:limit]]

    def get_experience_by_domain(self, domain: str, limit: int = 10) -> List[Experience]:
        """Get experiences for a specific domain.

        Complexity: T(n) ∈ O(e) for e experiences in domain.
        """
        with self._lock:
            ids = self._by_domain.get(domain, [])
            experiences = [self._experiences[eid] for eid in ids if eid in self._experiences]
            experiences.sort(key=lambda e: e.timestamp, reverse=True)
            return experiences[:limit]

    def get_experience_by_type(self, experience_type: str, limit: int = 10) -> List[Experience]:
        """Get experiences of a specific type.

        Complexity: T(n) ∈ O(e) for e experiences of type.
        """
        with self._lock:
            ids = self._by_type.get(experience_type, [])
            experiences = [self._experiences[eid] for eid in ids if eid in self._experiences]
            experiences.sort(key=lambda e: e.timestamp, reverse=True)
            return experiences[:limit]

    def get_breakthroughs(self, limit: int = 5) -> List[Experience]:
        """Get breakthrough experiences.

        Complexity: T(n) ∈ O(e) for e breakthrough experiences.
        """
        with self._lock:
            return self.get_experience_by_type("breakthrough", limit)

    def get_reflections(self, limit: int = 10) -> List[Experience]:
        """Get reflection experiences.

        Complexity: T(n) ∈ O(e) for e reflection experiences.
        """
        with self._lock:
            return self.get_experience_by_type("reflection", limit)

    def get_total_reward(self) -> float:
        """Get total reward accumulated across all experiences.

        Complexity: T(n) ∈ O(e) for e experiences.
        """
        with self._lock:
            return sum(e.reward_value for e in self._experiences.values())

    def get_experience_network(self, experience_id: str, depth: int = 2) -> List[Experience]:
        """Get the network of experiences related to a given experience.

        Traverses the related_experiences links to build a network.

        Args:
            experience_id: Starting experience ID.
            depth: How deep to traverse.

        Returns:
            List of related experiences.

        Complexity: T(n) ∈ O(d * r) for d depth, r relations per experience.
        """
        with self._lock:
            visited: Set[str] = set()
            result: List[Experience] = []
            self._traverse_network(experience_id, depth, visited, result)
            return result

    def _traverse_network(
        self,
        exp_id: str,
        depth: int,
        visited: Set[str],
        result: List[Experience],
    ) -> None:
        """Traverse the experience network recursively.

        Complexity: T(n) ∈ O(d * r) for d depth, r relations.
        """
        if depth <= 0 or exp_id in visited:
            return
        visited.add(exp_id)
        exp = self._experiences.get(exp_id)
        if exp is None:
            return
        result.append(exp)
        for related_id in exp.related_experiences:
            self._traverse_network(related_id, depth - 1, visited, result)

    def get_stats(self) -> Dict[str, float]:
        """Get experience memory statistics.

        Complexity: T(n) ∈ O(e) for e experiences.
        """
        with self._lock:
            type_counts: Dict[str, int] = {}
            domain_counts: Dict[str, int] = {}
            total_reward = 0.0

            for exp in self._experiences.values():
                type_counts[exp.experience_type] = type_counts.get(exp.experience_type, 0) + 1
                domain_counts[exp.domain] = domain_counts.get(exp.domain, 0) + 1
                total_reward += exp.reward_value

            return {
                "total_experiences": len(self._experiences),
                "conversations": type_counts.get("conversation", 0),
                "learnings": type_counts.get("learning", 0),
                "reflections": type_counts.get("reflection", 0),
                "breakthroughs": type_counts.get("breakthrough", 0),
                "patterns": type_counts.get("pattern", 0),
                "total_reward": total_reward,
                "domains": len(domain_counts),
            }

    def _prune(self) -> None:
        """Remove oldest experiences when over limit.

        Complexity: T(n) ∈ O(e log e) for e experiences.
        """
        # Sort by timestamp and remove oldest
        sorted_ids = sorted(
            self._experiences.keys(),
            key=lambda eid: self._experiences[eid].timestamp,
        )
        to_remove = sorted_ids[:len(sorted_ids) - self._max_experiences]
        for eid in to_remove:
            exp = self._experiences.pop(eid)
            # Clean up indices
            if exp.domain in self._by_domain:
                try:
                    self._by_domain[exp.domain].remove(eid)
                except ValueError:
                    pass
            if exp.experience_type in self._by_type:
                try:
                    self._by_type[exp.experience_type].remove(eid)
                except ValueError:
                    pass
            for topic in exp.topics:
                topic_lower = topic.lower()
                if topic_lower in self._by_topic:
                    try:
                        self._by_topic[topic_lower].remove(eid)
                    except ValueError:
                        pass

    @property
    def experience_count(self) -> int:
        """Get total number of experiences."""
        with self._lock:
            return len(self._experiences)
