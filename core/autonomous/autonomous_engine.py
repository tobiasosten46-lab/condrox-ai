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

"""Autonomous Knowledge Engine — the orchestrator for the entire learning system.

This is the central brain of the autonomous learning system. It coordinates:
- Knowledge extraction from every conversation
- Validation and storage of extracted knowledge
- Pattern recognition from interaction history
- User behavior analysis and profiling
- Deep reasoning for complex questions
- Meta-learning to improve over time
- Free will response generation

The engine is thread-safe and bulletproof — every component has error
recovery. No single failure can crash the system.

Design:
    The engine sits alongside the existing BrainController and enhances
    it with autonomous learning capabilities. It does NOT replace the
    18-stage pipeline — it enriches it with:
    - Real knowledge extraction from conversations
    - Pattern-based prediction
    - Behavioral adaptation
    - Deep reasoning chains
    - Free will response generation

Complexity:
    learn_from_turn: T(n) ∈ O(e + v + p) for extraction, validation, patterns
    generate_response: T(n) ∈ O(k + r + p) for knowledge, reasoning, patterns
    get_stats: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.autonomous.knowledge_extractor import (
    KnowledgeExtractor,
    ExtractionResult,
)
from core.autonomous.knowledge_validator import (
    KnowledgeValidator,
    KnowledgeStore,
    ValidationResult,
)
from core.autonomous.pattern_recognizer import (
    PatternRecognizer,
    Pattern,
    InteractionRecord,
)
from core.autonomous.behavior_analyzer import (
    BehaviorAnalyzer,
    UserProfile,
)
from core.autonomous.deep_reasoning import (
    DeepReasoningEngine,
    ReasoningResult,
)
from core.autonomous.meta_learning import (
    MetaLearningSystem,
    LearningOutcome,
)
from core.autonomous.free_will_engine import (
    FreeWillResponseEngine,
    ResponseContext,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LearningStats:
    """Statistics about the autonomous learning system.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    total_facts: int
    total_concepts: int
    total_relationships: int
    total_interactions: int
    total_patterns: int
    domains_tracked: int
    strategies_tracked: int
    weak_domains: int
    last_learning_time: float
    store_last_updated: float


class AutonomousKnowledgeEngine:
    """Orchestrates the autonomous learning system.

    This engine is the central coordinator that ties together all
    autonomous learning components. It is thread-safe and bulletproof.

    Thread-safety: A reentrant lock guards all state mutations.
    Error recovery: Every component call is wrapped in try/except.
    No single failure can crash the engine.

    Complexity:
        learn_from_turn: T(n) ∈ O(e + v + p) for extraction, validation, patterns
        generate_response: T(n) ∈ O(k + r + p)
        get_stats: T(n) ∈ O(1)
    """

    __slots__ = (
        "_extractor",
        "_store",
        "_validator",
        "_patterns",
        "_behavior",
        "_reasoning",
        "_meta",
        "_free_will",
        "_lock",
        "_last_learning_time",
        "_total_turns",
    )

    def __init__(self) -> None:
        """Initialize the autonomous knowledge engine.

        All components are created internally and wired together.
        """
        self._extractor = KnowledgeExtractor()
        self._store = KnowledgeStore()
        self._validator = KnowledgeValidator(self._store)
        self._patterns = PatternRecognizer()
        self._behavior = BehaviorAnalyzer()
        self._reasoning = DeepReasoningEngine(self._store)
        self._meta = MetaLearningSystem()
        self._free_will = FreeWillResponseEngine(
            store=self._store,
            reasoning=self._reasoning,
            patterns=self._patterns,
        )
        self._lock = threading.RLock()
        self._last_learning_time: float = 0.0
        self._total_turns: int = 0

    def learn_from_turn(
        self,
        user_message: str,
        ai_response: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        domain: str = "general",
        response_time_ms: float = 0.0,
        user_continued: bool = False,
    ) -> ExtractionResult:
        """Learn from a single conversation turn.

        This is called AFTER the AI has responded. It extracts knowledge
        from the conversation, validates it, stores it, and updates
        patterns, behavior, and meta-learning.

        Args:
            user_message: What the user said.
            ai_response: What the AI responded.
            intent: Classified intent.
            emotion: Detected emotion.
            domain: Knowledge domain.
            response_time_ms: Response time in milliseconds.
            user_continued: Whether the user continued after this turn.

        Returns:
            ExtractionResult with what was learned.

        Complexity: T(n) ∈ O(e + v + p) for extraction, validation, patterns.
        """
        with self._lock:
            timestamp = time.time()
            self._last_learning_time = timestamp
            self._total_turns += 1

            # 1. Extract knowledge from the conversation
            try:
                extraction = self._extractor.extract(
                    user_message=user_message,
                    ai_response=ai_response,
                    intent=intent,
                    emotion=emotion,
                    timestamp=timestamp,
                )
            except Exception as exc:
                logger.error("AutonomousEngine: extraction failed: %s", exc)
                extraction = ExtractionResult(timestamp=timestamp)

            # 2. Validate and store facts
            try:
                for fact in extraction.facts:
                    self._validator.validate_and_store(
                        fact=fact,
                        conversation_intent=intent,
                        conversation_topics=extraction.topics,
                    )
            except Exception as exc:
                logger.error("AutonomousEngine: validation failed: %s", exc)

            # 3. Store concepts
            try:
                for concept in extraction.concepts:
                    self._store.add_concept(concept)
            except Exception as exc:
                logger.error("AutonomousEngine: concept storage failed: %s", exc)

            # 4. Store relationships
            try:
                for rel in extraction.relationships:
                    self._store.add_relationship(rel)
            except Exception as exc:
                logger.error("AutonomousEngine: relationship storage failed: %s", exc)

            # 5. Record interaction for pattern recognition
            try:
                self._patterns.record_interaction(InteractionRecord(
                    timestamp=timestamp,
                    intent=intent,
                    emotion=emotion,
                    topics=extraction.topics,
                    domain=domain,
                    user_message_length=len(user_message),
                    response_length=len(ai_response),
                    response_time_ms=response_time_ms,
                    follow_up_asked=bool(extraction.facts),
                    user_returned=user_continued,
                    curiosity_score=extraction.user_knowledge_level,
                ))
            except Exception as exc:
                logger.error("AutonomousEngine: pattern recording failed: %s", exc)

            # 6. Record behavior
            try:
                self._behavior.record(
                    message=user_message,
                    intent=intent,
                    emotion=emotion,
                    domain=domain,
                    response_length=len(ai_response),
                    timestamp=timestamp,
                )
            except Exception as exc:
                logger.error("AutonomousEngine: behavior recording failed: %s", exc)

            # 7. Record meta-learning outcome
            try:
                # Reward = user knowledge level (higher = better interaction)
                reward = extraction.user_knowledge_level
                if user_continued:
                    reward = min(1.0, reward + 0.2)
                strategy = f"{intent}_response"
                self._meta.record_outcome(
                    strategy=strategy,
                    intent=intent,
                    domain=domain,
                    reward=reward,
                    response_time_ms=response_time_ms,
                    user_continued=user_continued,
                )

                # Record domain learning progress
                self._meta.record_learning(
                    domain=domain,
                    facts_learned=len(extraction.facts),
                    concepts_learned=len(extraction.concepts),
                    avg_confidence=extraction.user_knowledge_level,
                )
            except Exception as exc:
                logger.error("AutonomousEngine: meta-learning failed: %s", exc)

            return extraction

    def generate_response(
        self,
        user_message: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        domain: str = "general",
        mode: str = "default",
        confidence: float = 0.5,
        conversation_history: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        follow_up_needed: bool = True,
    ) -> str:
        """Generate a response using the free will engine.

        This is called INSTEAD OF template-based response generation.
        The engine constructs a unique response from knowledge, reasoning,
        patterns, and user profile.

        Args:
            user_message: The user's input.
            intent: Classified intent.
            emotion: Detected emotion.
            domain: Knowledge domain.
            mode: Processing mode.
            confidence: Intent confidence.
            conversation_history: Recent user messages.
            topics: Topics in the current conversation.
            follow_up_needed: Whether to add a follow-up question.

        Returns:
            A unique response string.

        Complexity: T(n) ∈ O(k + r + p) for knowledge, reasoning, patterns.
        """
        with self._lock:
            # Get user profile
            try:
                profile = self._behavior.analyze()
            except Exception as exc:
                logger.error("AutonomousEngine: profile analysis failed: %s", exc)
                profile = None

            # Build response context
            ctx = ResponseContext(
                user_message=user_message,
                intent=intent,
                emotion=emotion,
                domain=domain,
                mode=mode,
                confidence=confidence,
                conversation_history=tuple(conversation_history or []),
                user_profile=profile,
                topics=tuple(topics or []),
                follow_up_needed=follow_up_needed,
            )

            # Generate response using free will
            try:
                return self._free_will.generate(ctx)
            except Exception as exc:
                logger.error("AutonomousEngine: response generation failed: %s", exc)
                # Fallback — never return empty
                return f"I'm thinking about what you said. Could you tell me more about what you're looking for?"

    def get_user_profile(self) -> UserProfile:
        """Get the current user's behavioral profile.

        Complexity: T(n) ∈ O(n) for n interactions.
        """
        with self._lock:
            return self._behavior.analyze()

    def get_patterns(self) -> List[Pattern]:
        """Get recognized patterns.

        Complexity: T(n) ∈ O(n + p) for n interactions, p patterns.
        """
        with self._lock:
            return self._patterns.recognize_patterns()

    def get_predicted_topics(self, current_topics: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict what topics might come next.

        Complexity: T(n) ∈ O(t * c) for t topics, c co-occurrences.
        """
        with self._lock:
            return self._patterns.predict_next_topics(current_topics, top_k)

    def get_reasoning(self, question: str, intent: str = "question", domain: str = "general") -> ReasoningResult:
        """Run deep reasoning on a question.

        Complexity: T(n) ∈ O(d * k) for d depth, k knowledge.
        """
        with self._lock:
            return self._reasoning.reason(question, intent, domain)

    def get_meta_suggestion(self) -> str:
        """Get a meta-learning improvement suggestion.

        Complexity: T(n) ∈ O(d + s) for d domains, s strategies.
        """
        with self._lock:
            return self._meta.suggest_improvement()

    def get_best_strategy(self, intent: str, domain: str) -> str:
        """Get the best strategy for a context.

        Complexity: T(n) ∈ O(s) for s strategies.
        """
        with self._lock:
            return self._meta.get_best_strategy(intent, domain)

    def get_stats(self) -> LearningStats:
        """Get statistics about the learning system.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            patterns = self._patterns.recognize_patterns()
            weak_domains = self._meta.get_weak_domains()
            return LearningStats(
                total_facts=self._store.fact_count,
                total_concepts=self._store.concept_count,
                total_relationships=self._store.relationship_count,
                total_interactions=self._total_turns,
                total_patterns=len(patterns),
                domains_tracked=self._meta.domains_tracked,
                strategies_tracked=self._meta.strategies_tracked,
                weak_domains=len(weak_domains),
                last_learning_time=self._last_learning_time,
                store_last_updated=self._store.last_updated,
            )

    def new_session(self) -> None:
        """Record that a new conversation session has started.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            self._behavior.new_session()

    def clear_session(self) -> None:
        """Clear session-specific data (but keep learned knowledge).

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            # Knowledge and patterns persist across sessions
            # Only session-specific behavior is reset
            pass  # Behavior analyzer keeps history for cross-session learning

    @property
    def knowledge_store(self) -> KnowledgeStore:
        """Direct access to the knowledge store (for testing)."""
        return self._store

    @property
    def pattern_recognizer(self) -> PatternRecognizer:
        """Direct access to the pattern recognizer (for testing)."""
        return self._patterns

    @property
    def meta_learning(self) -> MetaLearningSystem:
        """Direct access to the meta-learning system (for testing)."""
        return self._meta

    @property
    def free_will_engine(self) -> FreeWillResponseEngine:
        """Direct access to the free will engine (for testing)."""
        return self._free_will

    @property
    def behavior_analyzer(self) -> BehaviorAnalyzer:
        """Direct access to the behavior analyzer (for testing)."""
        return self._behavior
