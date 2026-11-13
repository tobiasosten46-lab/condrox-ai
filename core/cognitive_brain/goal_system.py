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

"""Goal System — kognitiv retning, ikke "mål".

Goal er ikke en modul. Goal er hele pipeline-ens retning.
Systemet har retning basert på:
- hva som er viktig
- hva som er uklart
- hva som må reflekteres mer over
- hva som gir verdi
- hva som bygger erfaring

Goal = kognitiv retning, ikke fri vilje.

Complexity:
    set_goal: T(n) ∈ O(1)
    get_active_goals: T(n) ∈ O(g) for g goals
    update_goal_progress: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(slots=True)
class Goal:
    """A cognitive goal — a direction the system is moving toward.

    Goals are not tasks. Goals are directions that shape how the
    pipeline processes information.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    goal_id: str
    description: str           # What the goal is
    goal_type: str             # "understand", "improve", "clarify", "connect", "master"
    domain: str                # Knowledge domain
    priority: float            # [0.0, 1.0] — how important
    progress: float = 0.0      # [0.0, 1.0] — how far along
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    completed: bool = False
    related_topics: List[str] = field(default_factory=list)
    sub_goals: List[str] = field(default_factory=list)  # Goal IDs of sub-goals

    def update_progress(self, delta: float) -> None:
        """Update goal progress.

        Complexity: T(n) ∈ O(1).
        """
        self.progress = max(0.0, min(1.0, self.progress + delta))
        self.last_updated = time.time()
        if self.progress >= 0.95:
            self.completed = True


class GoalSystem:
    """Cognitive goal system — retning i pipeline.

    The goal system maintains active goals that shape how the pipeline
    processes information. Goals are derived from:
    - Knowledge gaps (topics with low confidence)
    - User interests (topics the user asks about)
    - Pattern observations (recurring themes)
    - Reward signals (what gives cognitive value)
    - Vision directives (system identity)

    Goals influence:
    - Knowledge retrieval priority
    - Reasoning depth
    - Response focus
    - Learning emphasis

    Thread-safety: All mutations guarded by a reentrant lock.

    Complexity:
        set_goal: T(n) ∈ O(1)
        get_active_goals: T(n) ∈ O(g log g) for g goals
        update_goal_progress: T(n) ∈ O(1)
        derive_goals_from_gaps: T(n) ∈ O(d) for d domains
    """

    __slots__ = ("_goals", "_lock", "_goal_counter", "_max_active_goals")

    def __init__(self, max_active_goals: int = 20) -> None:
        """Initialize the goal system.

        Args:
            max_active_goals: Maximum number of active goals.
        """
        self._goals: Dict[str, Goal] = {}
        self._lock = threading.RLock()
        self._goal_counter: int = 0
        self._max_active_goals = max_active_goals

    def set_goal(
        self,
        description: str,
        goal_type: str = "understand",
        domain: str = "general",
        priority: float = 0.5,
        related_topics: Optional[List[str]] = None,
    ) -> Goal:
        """Set a new cognitive goal.

        Args:
            description: What the goal is.
            goal_type: Type of goal (understand, improve, clarify, connect, master).
            domain: Knowledge domain.
            priority: How important [0.0, 1.0].
            related_topics: Topics related to this goal.

        Returns:
            The created Goal.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._goal_counter += 1
            goal_id = f"goal_{self._goal_counter}"
            goal = Goal(
                goal_id=goal_id,
                description=description,
                goal_type=goal_type,
                domain=domain,
                priority=max(0.0, min(1.0, priority)),
                related_topics=related_topics or [],
            )
            self._goals[goal_id] = goal

            # If over max, remove lowest-priority completed or low-priority goals
            if len(self._goals) > self._max_active_goals:
                self._prune_goals()

            return goal

    def update_goal_progress(self, goal_id: str, delta: float) -> bool:
        """Update progress on a goal.

        Args:
            goal_id: The goal to update.
            delta: Progress change (positive or negative).

        Returns:
            True if goal was found and updated.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            goal.update_progress(delta)
            return True

    def get_active_goals(self, limit: int = 10) -> List[Goal]:
        """Get active goals sorted by priority.

        Args:
            limit: Maximum number of goals to return.

        Returns:
            List of active (non-completed) goals sorted by priority.

        Complexity: T(n) ∈ O(g log g) for g goals.
        """
        with self._lock:
            active = [g for g in self._goals.values() if not g.completed]
            active.sort(key=lambda g: (g.priority, g.progress), reverse=True)
            return active[:limit]

    def get_goals_for_domain(self, domain: str) -> List[Goal]:
        """Get goals for a specific knowledge domain.

        Complexity: T(n) ∈ O(g) for g goals.
        """
        with self._lock:
            return [
                g for g in self._goals.values()
                if g.domain == domain and not g.completed
            ]

    def get_goals_for_topic(self, topic: str) -> List[Goal]:
        """Get goals related to a specific topic.

        Complexity: T(n) ∈ O(g * t) for g goals, t topics per goal.
        """
        with self._lock:
            topic_lower = topic.lower()
            return [
                g for g in self._goals.values()
                if not g.completed and any(topic_lower in t.lower() for t in g.related_topics)
            ]

    def derive_goals_from_gaps(
        self,
        weak_domains: List[str],
        low_confidence_topics: List[Tuple[str, float]],
    ) -> List[Goal]:
        """Derive new goals from knowledge gaps.

        When the system identifies weak domains or low-confidence topics,
        it creates goals to improve understanding in those areas.

        Args:
            weak_domains: Domains with insufficient knowledge.
            low_confidence_topics: (topic, confidence) pairs below threshold.

        Returns:
            List of created goals.

        Complexity: T(n) ∈ O(d + t) for d domains, t topics.
        """
        with self._lock:
            created: List[Goal] = []

            # Create goals for weak domains
            for domain in weak_domains:
                # Check if we already have a goal for this domain
                existing = self.get_goals_for_domain(domain)
                if not existing:
                    goal = self.set_goal(
                        description=f"Build knowledge in {domain} domain",
                        goal_type="understand",
                        domain=domain,
                        priority=0.7,
                    )
                    created.append(goal)

            # Create goals for low-confidence topics
            for topic, confidence in low_confidence_topics:
                if confidence < 0.4:
                    existing = self.get_goals_for_topic(topic)
                    if not existing:
                        goal = self.set_goal(
                            description=f"Clarify understanding of {topic}",
                            goal_type="clarify",
                            domain="general",
                            priority=0.6,
                            related_topics=[topic],
                        )
                        created.append(goal)

            return created

    def derive_goals_from_user_interests(
        self,
        user_topics: List[str],
        user_domains: List[str],
    ) -> List[Goal]:
        """Derive goals from user interests.

        When users repeatedly ask about certain topics, the system
        creates goals to better serve those interests.

        Args:
            user_topics: Topics the user is interested in.
            user_domains: Domains the user engages with.

        Returns:
            List of created goals.

        Complexity: T(n) ∈ O(t + d) for t topics, d domains.
        """
        with self._lock:
            created: List[Goal] = []

            for topic in user_topics[:3]:  # Top 3 topics
                existing = self.get_goals_for_topic(topic)
                if not existing:
                    goal = self.set_goal(
                        description=f"Deepen understanding of {topic} for user engagement",
                        goal_type="master",
                        domain="general",
                        priority=0.5,
                        related_topics=[topic],
                    )
                    created.append(goal)

            return created

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            goal.completed = True
            goal.progress = 1.0
            goal.last_updated = time.time()
            return True

    def get_top_priority(self) -> Optional[Goal]:
        """Get the highest-priority active goal.

        Complexity: T(n) ∈ O(g) for g goals.
        """
        with self._lock:
            active = [g for g in self._goals.values() if not g.completed]
            if not active:
                return None
            return max(active, key=lambda g: g.priority)

    def get_stats(self) -> Dict[str, float]:
        """Get goal system statistics.

        Complexity: T(n) ∈ O(g) for g goals.
        """
        with self._lock:
            active = [g for g in self._goals.values() if not g.completed]
            completed = [g for g in self._goals.values() if g.completed]
            avg_progress = (
                sum(g.progress for g in active) / len(active) if active else 0.0
            )
            return {
                "total_goals": len(self._goals),
                "active_goals": len(active),
                "completed_goals": len(completed),
                "avg_progress": avg_progress,
            }

    def _prune_goals(self) -> None:
        """Remove completed and lowest-priority goals.

        Complexity: T(n) ∈ O(g log g) for g goals.
        """
        # First remove completed goals
        completed_ids = [gid for gid, g in self._goals.items() if g.completed]
        for gid in completed_ids:
            del self._goals[gid]

        # If still over limit, remove lowest-priority
        if len(self._goals) > self._max_active_goals:
            sorted_goals = sorted(
                self._goals.items(),
                key=lambda x: (x[1].priority, x[1].progress),
            )
            while len(self._goals) > self._max_active_goals:
                gid, _ = sorted_goals.pop(0)
                del self._goals[gid]
