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

"""Cognitive Brain Core — den unified orchestrator.

Dette er den syntetiske hjernen. Alt henger sammen her:
- reward → bygger erfaring
- erfaring → bygger mening
- mening → bygger prioritet
- prioritet → bygger mål
- mål → bygger visjon
- visjon → styrer pipeline
- pipeline → styrer hele hjernen

Dette er én hjerne, ikke moduler.

Complexity:
    think: T(n) ∈ O(p + r + g + v + o + e) for all subsystems
    learn: T(n) ∈ O(e + r + g + o + m)
    get_state: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.cognitive_brain.reward_system import RewardSystem, Reward
from core.cognitive_brain.goal_system import GoalSystem, Goal
from core.cognitive_brain.vision_system import VisionSystem, AlignmentResult
from core.cognitive_brain.opinion_system import OpinionSystem, Opinion
from core.cognitive_brain.cognitive_phases import CognitivePhases, CognitiveFlowResult
from core.cognitive_brain.experience_memory import ExperienceMemory, Experience

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BrainState:
    """Snapshot of the brain's current state.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    total_reward: float
    active_goals: int
    alignment_trend: float
    total_opinions: int
    total_experiences: int
    breakthroughs: int
    stable_facts: int
    vision: str
    top_goal: str
    overall_confidence: float


@dataclass(slots=True)
class ThoughtResult:
    """Result of a single thought cycle.

    A thought cycle is the brain processing input through all
    cognitive phases, forming opinions, checking goals, evaluating
    alignment, and producing a unified cognitive output.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(p) for p phases.
    """
    cognitive_flow: CognitiveFlowResult
    alignment: AlignmentResult
    opinions_formed: List[Opinion]
    goals_updated: List[str]
    rewards_earned: List[Reward]
    experience_recorded: Experience
    unified_insight: str
    overall_confidence: float


class CognitiveBrainCore:
    """Den syntetiske hjernen — unified cognitive system.

    Dette er ikke en modul. Dette er en hjerne.
    Alt henger sammen:

    1. Input kommer inn
    2. Cognitive phases behandler det (reasoning, pattern, analysis, reverse, structure)
    3. Vision evaluerer om responsen er aligned med identitet
    4. Opinions dannes basert på erfaring
    5. Goals oppdateres basert på hva som er viktig
    6. Rewards gis basert på kognitiv verdi
    7. Experience lagres som kognitiv struktur
    8. Alt feedes tilbake til pipeline

    Ingen simulering. Ingen moduler. Én hjerne.

    Thread-safety: All state mutations guarded by a reentrant lock.
    Error recovery: Every subsystem wrapped in try/except.

    Complexity:
        think: T(n) ∈ O(p + r + g + v + o + e) for all subsystems
        learn: T(n) ∈ O(e + r + g + o + m)
        get_state: T(n) ∈ O(1)
    """

    __slots__ = (
        "_reward", "_goals", "_vision", "_opinions",
        "_phases", "_memory", "_lock",
        "_total_thoughts", "_last_thought_time",
    )

    def __init__(self) -> None:
        """Initialize the cognitive brain.

        All subsystems are created internally and wired together
        as one unified system.
        """
        self._reward = RewardSystem()
        self._goals = GoalSystem()
        self._vision = VisionSystem()
        self._opinions = OpinionSystem()
        self._phases = CognitivePhases()
        self._memory = ExperienceMemory()
        self._lock = threading.RLock()
        self._total_thoughts: int = 0
        self._last_thought_time: float = 0.0

    def think(
        self,
        input_text: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        domain: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> ThoughtResult:
        """Run a complete thought cycle.

        This is the brain's main function. It processes input through
        all cognitive phases, forms opinions, updates goals, evaluates
        alignment, gives rewards, and records experience.

        Args:
            input_text: The input to think about.
            intent: Classified intent.
            emotion: Detected emotion.
            domain: Knowledge domain.
            context: Additional context.

        Returns:
            ThoughtResult with all cognitive outputs.

        Complexity: T(n) ∈ O(p + r + g + v + o + e) for all subsystems.
        """
        with self._lock:
            self._total_thoughts += 1
            self._last_thought_time = time.time()
            ctx = context or {}

            # 1. Run cognitive phases (reasoning, pattern, analysis, reverse, structure)
            try:
                cognitive_flow = self._phases.process(input_text, ctx)
            except Exception as exc:
                logger.error("BrainCore: cognitive phases failed: %s", exc)
                cognitive_flow = CognitiveFlowResult(
                    phases=(),
                    unified_insight="Cognitive phases failed.",
                    overall_confidence=0.0,
                    connections=(),
                )

            # 2. Form opinions based on cognitive flow
            opinions_formed: List[Opinion] = []
            try:
                opinions_formed = self._form_opinions_from_flow(
                    cognitive_flow, input_text, domain, intent
                )
            except Exception as exc:
                logger.error("BrainCore: opinion formation failed: %s", exc)

            # 3. Update goals based on cognitive flow
            goals_updated: List[str] = []
            try:
                goals_updated = self._update_goals_from_flow(
                    cognitive_flow, domain, intent
                )
            except Exception as exc:
                logger.error("BrainCore: goal update failed: %s", exc)

            # 4. Evaluate alignment with vision
            try:
                alignment = self._vision.evaluate_alignment(
                    response=cognitive_flow.unified_insight,
                    intent=intent,
                    used_knowledge=any(p.evidence for p in cognitive_flow.phases),
                    used_templates=False,
                    acknowledged_limitation=cognitive_flow.overall_confidence < 0.3,
                    connected_to_previous=bool(ctx.get("previous_topics")),
                )
            except Exception as exc:
                logger.error("BrainCore: vision alignment failed: %s", exc)
                alignment = AlignmentResult(
                    alignment_score=0.5,
                    aligned_directives=(),
                    misaligned_directives=(),
                    recommendation="Vision evaluation failed.",
                )

            # 5. Give rewards based on cognitive value
            rewards_earned: List[Reward] = []
            try:
                rewards_earned = self._give_rewards_from_flow(
                    cognitive_flow, alignment, domain, intent
                )
            except Exception as exc:
                logger.error("BrainCore: reward system failed: %s", exc)

            # 6. Record experience
            experience: Experience = self._memory.record_experience(
                experience_type="conversation",
                description=input_text[:200],
                what_was_learned=cognitive_flow.unified_insight[:200],
                domain=domain,
                intent=intent,
                emotion=emotion,
                reward_value=sum(r.value for r in rewards_earned),
                topics=list(ctx.get("topics", [])),
                confidence=cognitive_flow.overall_confidence,
            )

            # 7. Build unified insight
            unified_insight = self._build_unified_insight(
                cognitive_flow, alignment, opinions_formed
            )

            return ThoughtResult(
                cognitive_flow=cognitive_flow,
                alignment=alignment,
                opinions_formed=opinions_formed,
                goals_updated=goals_updated,
                rewards_earned=rewards_earned,
                experience_recorded=experience,
                unified_insight=unified_insight,
                overall_confidence=cognitive_flow.overall_confidence,
            )

    def learn(
        self,
        input_text: str,
        ai_response: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        domain: str = "general",
        user_continued: bool = False,
        topics: Optional[List[str]] = None,
    ) -> ThoughtResult:
        """Learn from a conversation turn.

        This is called AFTER the AI has responded. The brain processes
        the entire conversation turn — what the user said, what the AI
        responded, and what was learned.

        Args:
            input_text: What the user said.
            ai_response: What the AI responded.
            intent: Classified intent.
            emotion: Detected emotion.
            domain: Knowledge domain.
            user_continued: Whether the user continued after this turn.
            topics: Topics in the conversation.

        Returns:
            ThoughtResult with cognitive outputs.

        Complexity: T(n) ∈ O(p + r + g + v + o + e) for all subsystems.
        """
        with self._lock:
            # Combine input and response for full context
            full_text = f"User: {input_text}\nAI: {ai_response}"

            # Run the thought cycle
            result = self.think(
                input_text=full_text,
                intent=intent,
                emotion=emotion,
                domain=domain,
                context={"topics": topics or [], "user_message": input_text, "ai_response": ai_response},
            )

            # Additional learning-specific processing

            # Check for insight (new relationship discovered)
            if any(p.phase_name == "analysis" and p.confidence > 0.6 for p in result.cognitive_flow.phases):
                try:
                    insight_reward = self._reward.record_insight(
                        subject=input_text[:50],
                        relation="relates to",
                        object=ai_response[:50],
                        domain=domain,
                        novelty_score=result.overall_confidence,
                    )
                    result.rewards_earned.append(insight_reward)
                except Exception as exc:
                    logger.error("BrainCore: insight reward failed: %s", exc)

            # Check for strategy success
            if user_continued:
                try:
                    strategy_reward = self._reward.record_strategy_success(
                        strategy_name=f"{intent}_response",
                        intent=intent,
                        domain=domain,
                        success_score=0.7,
                    )
                    result.rewards_earned.append(strategy_reward)
                except Exception as exc:
                    logger.error("BrainCore: strategy reward failed: %s", exc)

            # Record learning experience
            try:
                self._memory.record_experience(
                    experience_type="learning",
                    description=f"Learned from conversation about {', '.join((topics or [])[:3])}",
                    what_was_learned=result.unified_insight[:200],
                    domain=domain,
                    intent=intent,
                    emotion=emotion,
                    reward_value=sum(r.value for r in result.rewards_earned),
                    topics=topics or [],
                    confidence=result.overall_confidence,
                )
            except Exception as exc:
                logger.error("BrainCore: learning experience record failed: %s", exc)

            return result

    def _form_opinions_from_flow(
        self,
        flow: CognitiveFlowResult,
        input_text: str,
        domain: str,
        intent: str,
    ) -> List[Opinion]:
        """Form opinions based on cognitive flow results.

        Complexity: T(n) ∈ O(p) for p phases.
        """
        opinions: List[Opinion] = []

        # Extract topics from input
        words = [w.strip(".,!?;:") for w in input_text.split() if len(w) > 4]
        key_topics = list(set(words))[:3]

        for topic in key_topics:
            # Determine assessment based on cognitive flow
            assessment = "interesting"
            confidence = flow.overall_confidence

            # If pattern phase detected repetitions, topic is important
            for phase in flow.phases:
                if phase.phase_name == "pattern":
                    if any("Repetition" in i for i in phase.insights):
                        assessment = "important"
                        confidence = max(confidence, phase.confidence)

                # If analysis phase found strong relationships, topic is solid
                if phase.phase_name == "analysis":
                    if any("relationship" in i.lower() for i in phase.insights):
                        assessment = "solid"
                        confidence = max(confidence, phase.confidence)

                # If reasoning phase found low confidence, topic is unclear
                if phase.phase_name == "reasoning":
                    if phase.confidence < 0.4:
                        assessment = "unclear"
                        confidence = phase.confidence

            opinion = self._opinions.form_opinion(
                topic=topic,
                assessment=assessment,
                domain=domain,
                confidence=confidence,
                evidence=f"Cognitive flow: {flow.unified_insight[:80]}",
            )
            opinions.append(opinion)

        return opinions

    def _update_goals_from_flow(
        self,
        flow: CognitiveFlowResult,
        domain: str,
        intent: str,
    ) -> List[str]:
        """Update goals based on cognitive flow.

        Complexity: T(n) ∈ O(g) for g goals.
        """
        updated: List[str] = []

        # If cognitive confidence is low, create a clarification goal
        if flow.overall_confidence < 0.4:
            goal = self._goals.set_goal(
                description=f"Clarify understanding in {domain} domain",
                goal_type="clarify",
                domain=domain,
                priority=0.7,
            )
            updated.append(goal.goal_id)

        # If pattern phase detected strong patterns, update mastery goal
        for phase in flow.phases:
            if phase.phase_name == "pattern" and phase.confidence > 0.7:
                active_goals = self._goals.get_goals_for_domain(domain)
                for g in active_goals:
                    if g.goal_type == "master":
                        self._goals.update_goal_progress(g.goal_id, 0.1)
                        updated.append(g.goal_id)

        # If analysis found strong relationships, update understanding goal
        for phase in flow.phases:
            if phase.phase_name == "analysis" and phase.confidence > 0.6:
                active_goals = self._goals.get_goals_for_domain(domain)
                for g in active_goals:
                    if g.goal_type == "understand":
                        self._goals.update_goal_progress(g.goal_id, 0.05)
                        updated.append(g.goal_id)

        return updated

    def _give_rewards_from_flow(
        self,
        flow: CognitiveFlowResult,
        alignment: AlignmentResult,
        domain: str,
        intent: str,
    ) -> List[Reward]:
        """Give rewards based on cognitive value.

        Complexity: T(n) ∈ O(1).
        """
        rewards: List[Reward] = []

        # Clarity boost — if cognitive flow has high confidence
        if flow.overall_confidence > 0.7:
            reward = self._reward.record_clarity(
                topic=domain,
                previous_clarity=0.3,
                new_clarity=flow.overall_confidence,
                domain=domain,
            )
            if reward:
                rewards.append(reward)

        # Strategy boost — if alignment is high
        if alignment.alignment_score > 0.8:
            reward = self._reward.record_strategy_success(
                strategy_name="cognitive_flow",
                intent=intent,
                domain=domain,
                success_score=alignment.alignment_score,
            )
            rewards.append(reward)

        # Breakthrough — if overall confidence is very high AND alignment is very high
        if flow.overall_confidence > 0.85 and alignment.alignment_score > 0.9:
            reward = self._reward.record_breakthrough(
                description=f"High-confidence cognitive flow in {domain}",
                domain=domain,
                impact_score=flow.overall_confidence,
            )
            rewards.append(reward)

        return rewards

    def _build_unified_insight(
        self,
        flow: CognitiveFlowResult,
        alignment: AlignmentResult,
        opinions: List[Opinion],
    ) -> str:
        """Build a unified insight from all cognitive outputs.

        Complexity: T(n) ∈ O(p + o) for p phases, o opinions.
        """
        parts: List[str] = []

        # Cognitive flow insight
        if flow.unified_insight and flow.unified_insight != "No unified insight — insufficient cognitive activity.":
            parts.append(flow.unified_insight)

        # Alignment insight
        if alignment.alignment_score > 0.8:
            parts.append("Well-aligned with vision")
        elif alignment.misaligned_directives:
            parts.append(f"Needs improvement: {', '.join(alignment.misaligned_directives[:2])}")

        # Opinion insights
        if opinions:
            top_opinion = opinions[0]
            parts.append(f"Assessment: {top_opinion.topic} is {top_opinion.assessment}")

        if not parts:
            return "Cognitive processing complete. No significant insights."

        return " | ".join(parts[:4])

    def get_state(self) -> BrainState:
        """Get a snapshot of the brain's current state.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            reward_stats = self._reward.get_stats()
            goal_stats = self._goals.get_stats()
            vision_stats = self._vision.get_stats()
            opinion_stats = self._opinions.get_stats()
            memory_stats = self._memory.get_stats()

            top_goal = self._goals.get_top_priority()
            top_goal_desc = top_goal.description if top_goal else "No active goals"

            return BrainState(
                total_reward=reward_stats["total_reward"],
                active_goals=int(goal_stats["active_goals"]),
                alignment_trend=vision_stats["alignment_trend"],
                total_opinions=int(opinion_stats["total_opinions"]),
                total_experiences=int(memory_stats["total_experiences"]),
                breakthroughs=int(reward_stats["breakthrough_count"]),
                stable_facts=int(reward_stats["stable_facts"]),
                vision=self._vision.get_vision()[:100],
                top_goal=top_goal_desc,
                overall_confidence=vision_stats["alignment_trend"],
            )

    def get_vision(self) -> str:
        """Get the system's vision statement.

        Complexity: T(n) ∈ O(1).
        """
        return self._vision.get_vision()

    def get_identity(self) -> str:
        """Get the system's identity statement.

        Complexity: T(n) ∈ O(1).
        """
        return self._vision.get_identity_statement()

    def get_active_goals(self, limit: int = 5) -> List[Goal]:
        """Get active goals.

        Complexity: T(n) ∈ O(g log g) for g goals.
        """
        with self._lock:
            return self._goals.get_active_goals(limit)

    def get_important_topics(self) -> List[Opinion]:
        """Get topics assessed as important.

        Complexity: T(n) ∈ O(o log o) for o opinions.
        """
        with self._lock:
            return self._opinions.get_important_topics()

    def get_recent_experiences(self, limit: int = 5) -> List[Experience]:
        """Get recent experiences.

        Complexity: T(n) ∈ O(e) for e experiences.
        """
        with self._lock:
            return list(reversed(list(self._memory._experiences.values())))[:limit]

    def get_recent_rewards(self, limit: int = 5) -> List[Reward]:
        """Get recent rewards.

        Complexity: T(n) ∈ O(r) for r rewards.
        """
        with self._lock:
            return self._reward.get_recent_rewards(limit)

    def get_relevant_experiences(self, topics: List[str], domain: str = "general") -> List[Experience]:
        """Get experiences relevant to topics.

        Complexity: T(n) ∈ O(e) for e experiences.
        """
        with self._lock:
            return self._memory.get_relevant_experiences(topics, domain)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return {
                "total_thoughts": self._total_thoughts,
                "reward": self._reward.get_stats(),
                "goals": self._goals.get_stats(),
                "vision": self._vision.get_stats(),
                "opinions": self._opinions.get_stats(),
                "memory": self._memory.get_stats(),
            }

    @property
    def reward_system(self) -> RewardSystem:
        """Direct access to reward system."""
        return self._reward

    @property
    def goal_system(self) -> GoalSystem:
        """Direct access to goal system."""
        return self._goals

    @property
    def vision_system(self) -> VisionSystem:
        """Direct access to vision system."""
        return self._vision

    @property
    def opinion_system(self) -> OpinionSystem:
        """Direct access to opinion system."""
        return self._opinions

    @property
    def cognitive_phases(self) -> CognitivePhases:
        """Direct access to cognitive phases."""
        return self._phases

    @property
    def experience_memory(self) -> ExperienceMemory:
        """Direct access to experience memory."""
        return self._memory
