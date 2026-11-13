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

"""Unified Cognitive Pipeline — én data flow i korrekt logisk rekkefølge.

Dette er ikke tre separate systemer. Dette er ÉN pipeline der alt henger
sammen i en sammenhengende strøm. Hver stage feeder into den neste.

Data flow (korrekt logisk rekkefølge):

    1. VISION CONTEXT — Vision setter kontekst: hvem er jeg, hva er active directives
    2. GOAL CHECK — Goals sjekker: hva prøver jeg å oppnå? Former kunnskapshenting
    3. OPINION CHECK — Opinions: hva tenker jeg allerede om dette? Former respons
    4. COGNITIVE PHASES — Reasoning → Pattern → Analysis → Reverse → Structure
    5. KNOWLEDGE RETRIEVAL — Formet av goals, opinions, cognitive phases
    6. RESPONSE GENERATION — Free will, informert av ALT over
    7. VISION ALIGNMENT — Er responsen aligned med identitet?
    8. REWARD — Kognitiv verdi assessment
    9. EXPERIENCE RECORDING — Lagre som kognitiv struktur
    10. GOAL PROGRESS — Førte dette nærmere et mål?
    11. OPINION UPDATE — Endret dette noen meninger?
    12. META-LEARNING — Forbedret dette strategier?

Ingen parallellitet. Én strøm. Alt i rekkefølge.

Complexity:
    process: T(n) ∈ O(v + g + o + p + k + r + a + e + m) for all stages

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
from core.cognitive_brain.vision_system import VisionSystem, AlignmentResult, VisionDirective
from core.cognitive_brain.opinion_system import OpinionSystem, Opinion
from core.cognitive_brain.cognitive_phases import (
    CognitivePhases,
    CognitiveFlowResult,
    PhaseResult,
)
from core.cognitive_brain.experience_memory import ExperienceMemory, Experience
from core.autonomous.knowledge_extractor import KnowledgeExtractor, ExtractionResult
from core.autonomous.knowledge_validator import KnowledgeValidator, KnowledgeStore
from core.autonomous.pattern_recognizer import PatternRecognizer, InteractionRecord
from core.autonomous.behavior_analyzer import BehaviorAnalyzer, UserProfile
from core.autonomous.deep_reasoning import DeepReasoningEngine
from core.autonomous.meta_learning import MetaLearningSystem
from core.autonomous.free_will_engine import FreeWillResponseEngine, ResponseContext

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineContext:
    """Context that flows through the entire pipeline.

    Each stage enriches this context. The next stage reads from it.
    This is the data flow — one stream, not parallel systems.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(c) for accumulated context.
    """
    # Input
    user_message: str
    intent: str = "unknown"
    emotion: str = "neutral"
    domain: str = "general"
    mode: str = "default"
    confidence: float = 0.5
    conversation_history: List[str] = field(default_factory=list)
    session_id: str = ""

    # Stage 1: Vision context
    active_directives: List[VisionDirective] = field(default_factory=list)
    vision_statement: str = ""

    # Stage 2: Goal context
    active_goals: List[Goal] = field(default_factory=list)
    relevant_goals: List[Goal] = field(default_factory=list)

    # Stage 3: Opinion context
    existing_opinions: List[Opinion] = field(default_factory=list)
    topic_assessments: Dict[str, str] = field(default_factory=dict)  # topic → assessment

    # Stage 4: Cognitive phases
    cognitive_flow: Optional[CognitiveFlowResult] = None
    reasoning_insights: List[str] = field(default_factory=list)
    pattern_insights: List[str] = field(default_factory=list)
    analysis_insights: List[str] = field(default_factory=list)
    reverse_insights: List[str] = field(default_factory=list)
    structure_insights: List[str] = field(default_factory=list)

    # Stage 5: Knowledge retrieval
    retrieved_facts: List[Tuple[Any, float, float]] = field(default_factory=list)  # (fact, quality, importance)
    reasoning_result: Optional[Any] = None
    predicted_topics: List[Tuple[str, float]] = field(default_factory=list)

    # Stage 6: Response generation
    generated_response: str = ""

    # Stage 7: Vision alignment
    alignment: Optional[AlignmentResult] = None

    # Stage 8: Reward
    rewards_earned: List[Reward] = field(default_factory=list)

    # Stage 9: Experience
    experience_recorded: Optional[Experience] = None

    # Stage 10: Goal progress
    goals_updated: List[str] = field(default_factory=list)

    # Stage 11: Opinion updates
    opinions_formed: List[Opinion] = field(default_factory=list)

    # Stage 12: Meta-learning
    strategy_used: str = ""
    strategy_reward: float = 0.0

    # User profile (from behavior analyzer)
    user_profile: Optional[UserProfile] = None

    # Extracted knowledge
    extraction: Optional[ExtractionResult] = None

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0


class UnifiedCognitivePipeline:
    """Én unified kognitiv pipeline — ikke separate systemer.

    Alt henger sammen i én data flow:
    1. Vision → 2. Goals → 3. Opinions → 4. Cognitive Phases →
    5. Knowledge → 6. Response → 7. Alignment → 8. Reward →
    9. Experience → 10. Goal Progress → 11. Opinion Update → 12. Meta-Learning

    Hver stage leser fra PipelineContext og skriver tilbake til den.
    Neste stage leser det forrige skrev. Én strøm.

    Thread-safety: RLock guards all state.
    Error recovery: Every stage wrapped in try/except.

    Complexity:
        process: T(n) ∈ O(v + g + o + p + k + r + a + e + m)
    """

    __slots__ = (
        # Cognitive brain components
        "_reward", "_goals", "_vision", "_opinions",
        "_phases", "_memory",
        # Autonomous learning components (merged INTO this pipeline)
        "_extractor", "_store", "_validator",
        "_patterns", "_behavior", "_reasoning", "_meta",
        "_free_will",
        # Lock
        "_lock",
        "_total_cycles",
    )

    def __init__(self) -> None:
        """Initialize the unified cognitive pipeline.

        All components are created internally. There are no separate
        systems — everything is part of one pipeline.
        """
        # Cognitive brain components
        self._reward = RewardSystem()
        self._goals = GoalSystem()
        self._vision = VisionSystem()
        self._opinions = OpinionSystem()
        self._phases = CognitivePhases()
        self._memory = ExperienceMemory()

        # Autonomous learning components — merged INTO this pipeline
        self._extractor = KnowledgeExtractor()
        self._store = KnowledgeStore()
        self._validator = KnowledgeValidator(self._store)
        self._patterns = PatternRecognizer()
        self._behavior = BehaviorAnalyzer()
        self._reasoning = DeepReasoningEngine(self._store)
        self._meta = MetaLearningSystem()

        # Free will engine — uses the same knowledge store and reasoning
        self._free_will = FreeWillResponseEngine(
            store=self._store,
            reasoning=self._reasoning,
            patterns=self._patterns,
        )

        self._lock = threading.RLock()
        self._total_cycles = 0

    def process(
        self,
        user_message: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        domain: str = "general",
        mode: str = "default",
        confidence: float = 0.5,
        conversation_history: Optional[List[str]] = None,
        session_id: str = "",
        pipeline_output: str = "",  # Output from the 18-stage pipeline (if available)
    ) -> PipelineContext:
        """Run the complete unified cognitive pipeline.

        This is ONE data flow. Each stage runs in order and feeds
        into the next. The context accumulates as it flows.

        Args:
            user_message: The user's input.
            intent: Classified intent from the 18-stage pipeline.
            emotion: Detected emotion.
            domain: Knowledge domain.
            mode: Processing mode.
            confidence: Intent confidence.
            conversation_history: Recent user messages.
            session_id: Session identifier.
            pipeline_output: Output from the 18-stage pipeline (agent draft).

        Returns:
            PipelineContext with all stages completed.

        Complexity: T(n) ∈ O(v + g + o + p + k + r + a + e + m)
        """
        with self._lock:
            self._total_cycles += 1

            # Initialize context
            ctx = PipelineContext(
                user_message=user_message,
                intent=intent,
                emotion=emotion,
                domain=domain,
                mode=mode,
                confidence=confidence,
                conversation_history=conversation_history or [],
                session_id=session_id,
            )

            # ─── STAGE 1: VISION CONTEXT ─────────────────────────────
            # Vision setter kontekst: hvem er jeg, hva er active directives
            self._stage_1_vision(ctx)

            # ─── STAGE 2: GOAL CHECK ──────────────────────────────────
            # Goals sjekker: hva prøver jeg å oppnå? Former kunnskapshenting
            self._stage_2_goals(ctx)

            # ─── STAGE 3: OPINION CHECK ───────────────────────────────
            # Opinions: hva tenker jeg allerede om dette? Former respons
            self._stage_3_opinions(ctx)

            # ─── STAGE 4: COGNITIVE PHASES ────────────────────────────
            # Reasoning → Pattern → Analysis → Reverse → Structure
            self._stage_4_cognitive_phases(ctx)

            # ─── STAGE 5: KNOWLEDGE RETRIEVAL ─────────────────────────
            # Formet av goals, opinions, cognitive phases
            self._stage_5_knowledge_retrieval(ctx)

            # ─── STAGE 6: RESPONSE GENERATION ─────────────────────────
            # Free will, informert av ALT over
            self._stage_6_response_generation(ctx, pipeline_output)

            # ─── STAGE 7: VISION ALIGNMENT ────────────────────────────
            # Er responsen aligned med identitet?
            self._stage_7_vision_alignment(ctx)

            # ─── STAGE 8: REWARD ──────────────────────────────────────
            # Kognitiv verdi assessment
            self._stage_8_reward(ctx)

            # ─── STAGE 9: EXPERIENCE RECORDING ────────────────────────
            # Lagre som kognitiv struktur
            self._stage_9_experience_recording(ctx)

            # ─── STAGE 10: GOAL PROGRESS ──────────────────────────────
            # Førte dette nærmere et mål?
            self._stage_10_goal_progress(ctx)

            # ─── STAGE 11: OPINION UPDATE ─────────────────────────────
            # Endret dette noen meninger?
            self._stage_11_opinion_update(ctx)

            # ─── STAGE 12: META-LEARNING ──────────────────────────────
            # Forbedret dette strategier?
            self._stage_12_meta_learning(ctx)

            ctx.end_time = time.time()
            return ctx

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: VISION CONTEXT
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_1_vision(self, ctx: PipelineContext) -> None:
        """Vision setter kontekst — hvem er jeg, hva er active directives.

        Complexity: T(n) ∈ O(d) for d directives.
        """
        try:
            ctx.active_directives = self._vision.get_active_directives()
            ctx.vision_statement = self._vision.get_vision()
        except Exception as exc:
            logger.error("Stage 1 Vision failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: GOAL CHECK
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_2_goals(self, ctx: PipelineContext) -> None:
        """Goals sjekker — hva prøver jeg å oppnå?

        Active goals shape what knowledge to retrieve and how deep
        to reason. If there's a goal to "clarify X" and the user
        asks about X, we reason deeper.

        Complexity: T(n) ∈ O(g) for g goals.
        """
        try:
            ctx.active_goals = self._goals.get_active_goals(limit=10)

            # Find goals relevant to this conversation
            # Extract key words from user message
            words = [w.strip(".,!?;:") for w in ctx.user_message.split() if len(w) > 3]
            for goal in ctx.active_goals:
                # Check if any goal topics match user message words
                for topic in goal.related_topics:
                    if topic.lower() in ctx.user_message.lower():
                        ctx.relevant_goals.append(goal)
                        break
                # Check if goal domain matches
                if goal.domain == ctx.domain and goal not in ctx.relevant_goals:
                    ctx.relevant_goals.append(goal)
        except Exception as exc:
            logger.error("Stage 2 Goals failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: OPINION CHECK
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_3_opinions(self, ctx: PipelineContext) -> None:
        """Opinions — hva tenker jeg allerede om dette?

        Existing opinions shape how to respond. If I think a topic
        is "important", I respond with more depth. If "unclear",
        I acknowledge the difficulty.

        Complexity: T(n) ∈ O(w) for w words in message.
        """
        try:
            words = [w.strip(".,!?;:") for w in ctx.user_message.split() if len(w) > 3]
            for word in words:
                opinion = self._opinions.get_opinion(word)
                if opinion:
                    ctx.existing_opinions.append(opinion)
                    ctx.topic_assessments[word] = opinion.assessment
        except Exception as exc:
            logger.error("Stage 3 Opinions failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 4: COGNITIVE PHASES
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_4_cognitive_phases(self, ctx: PipelineContext) -> None:
        """Cognitive phases — reasoning, pattern, analysis, reverse, structure.

        Each phase runs in sequence and feeds into the next.
        Results are stored in context for use by later stages.

        Complexity: T(n) ∈ O(p * n) for p phases, n input size.
        """
        try:
            # Build context from previous stages
            phase_context = {
                "active_directives": [d.directive_id for d in ctx.active_directives],
                "relevant_goals": [g.description for g in ctx.relevant_goals],
                "existing_opinions": ctx.topic_assessments,
                "domain": ctx.domain,
                "intent": ctx.intent,
            }

            ctx.cognitive_flow = self._phases.process(ctx.user_message, phase_context)

            # Extract insights per phase for later stages
            for phase in ctx.cognitive_flow.phases:
                if phase.phase_name == "reasoning":
                    ctx.reasoning_insights = list(phase.insights)
                elif phase.phase_name == "pattern":
                    ctx.pattern_insights = list(phase.insights)
                elif phase.phase_name == "analysis":
                    ctx.analysis_insights = list(phase.insights)
                elif phase.phase_name == "reverse":
                    ctx.reverse_insights = list(phase.insights)
                elif phase.phase_name == "structure":
                    ctx.structure_insights = list(phase.insights)

        except Exception as exc:
            logger.error("Stage 4 Cognitive Phases failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 5: KNOWLEDGE RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_5_knowledge_retrieval(self, ctx: PipelineContext) -> None:
        """Knowledge retrieval — formet av goals, opinions, cognitive phases.

        Knowledge retrieval is SHAPED by:
        - Goals: if there's a goal to understand X, prioritize X knowledge
        - Opinions: if I think X is important, retrieve more about X
        - Cognitive phases: if pattern phase detected repetitions, retrieve related

        Complexity: T(n) ∈ O(k + r + p) for knowledge, reasoning, patterns.
        """
        try:
            # Extract topics from cognitive phases
            topics: List[str] = []
            if ctx.cognitive_flow:
                # Use pattern phase insights to identify topics
                for insight in ctx.pattern_insights:
                    if "Pattern" in insight:
                        # Extract pattern name as topic
                        words = insight.split("'")
                        if len(words) > 1:
                            topics.append(words[1])

            # Add topics from user message
            msg_words = [w.strip(".,!?;:") for w in ctx.user_message.split() if len(w) > 4]
            topics.extend(msg_words[:3])

            # Retrieve knowledge from the knowledge store
            for topic in topics[:3]:
                facts = self._store.find_by_subject(topic)
                for fact, quality, importance in facts[:2]:
                    if quality > 0.3:
                        ctx.retrieved_facts.append((fact, quality, importance))

            # If not enough facts, try domain
            if len(ctx.retrieved_facts) < 3 and ctx.domain != "general":
                domain_facts = self._store.find_by_domain(ctx.domain, limit=5)
                for fact, quality, importance in domain_facts:
                    if quality > 0.3:
                        ctx.retrieved_facts.append((fact, quality, importance))

            # Run deep reasoning — shaped by cognitive phase insights
            ctx.reasoning_result = self._reasoning.reason(
                question=ctx.user_message,
                intent=ctx.intent,
                domain=ctx.domain,
                max_depth=4,
            )

            # Predict next topics based on patterns
            ctx.predicted_topics = self._patterns.predict_next_topics(topics, top_k=3)

        except Exception as exc:
            logger.error("Stage 5 Knowledge Retrieval failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 6: RESPONSE GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_6_response_generation(self, ctx: PipelineContext, pipeline_output: str) -> None:
        """Response generation — free will, informert av ALT over.

        The response is generated using:
        - Vision directives (be authentic, be useful, be honest)
        - Active goals (respond with more depth for goal-relevant topics)
        - Existing opinions (acknowledge if topic is "unclear")
        - Cognitive phase insights (use reasoning/analysis insights)
        - Retrieved knowledge (facts from knowledge store)
        - Reasoning results (deep reasoning conclusions)
        - Predicted topics (for proactive follow-ups)

        For conversational intents: use free will engine
        For task intents: use pipeline output (CodeAgent, etc.)
        For question intents: use cognitive flow + knowledge + reasoning

        Complexity: T(n) ∈ O(k + r + p) for knowledge, reasoning, patterns.
        """
        try:
            _FREE_WILL_INTENTS = {"greet", "identity", "emotional", "grateful", "clarify", "unknown"}
            _TASK_INTENTS = {"code", "analyze", "architect"}

            if ctx.intent in _FREE_WILL_INTENTS:
                # Use free will engine — it has access to knowledge store,
                # reasoning, and patterns (all shared in this pipeline)
                topics = list(ctx.topic_assessments.keys())[:5]
                response_ctx = ResponseContext(
                    user_message=ctx.user_message,
                    intent=ctx.intent,
                    emotion=ctx.emotion,
                    domain=ctx.domain,
                    mode=ctx.mode,
                    confidence=ctx.confidence,
                    conversation_history=tuple(ctx.conversation_history),
                    user_profile=ctx.user_profile,
                    topics=tuple(topics),
                    follow_up_needed=True,
                )
                ctx.generated_response = self._free_will.generate(response_ctx)
                ctx.strategy_used = f"free_will_{ctx.intent}"

            elif ctx.intent in _TASK_INTENTS:
                # Use pipeline output (CodeAgent, AnalyzeAgent, ArchitectAgent)
                ctx.generated_response = pipeline_output
                ctx.strategy_used = f"pipeline_{ctx.intent}"

            else:
                # For question/explain intents: build from cognitive flow + knowledge
                ctx.generated_response = self._build_question_response(ctx, pipeline_output)
                ctx.strategy_used = f"cognitive_{ctx.intent}"

            # If response is empty or too short, fall back to pipeline output
            if not ctx.generated_response or len(ctx.generated_response) < 20:
                ctx.generated_response = pipeline_output or "I'm thinking about what you said. Could you tell me more?"

        except Exception as exc:
            logger.error("Stage 6 Response Generation failed: %s", exc)
            ctx.generated_response = pipeline_output or "I'm processing your input. What would you like to know?"

    def _build_question_response(self, ctx: PipelineContext, pipeline_output: str) -> str:
        """Build a response for question/explain intents.

        Uses cognitive flow insights + retrieved knowledge + reasoning.
        Falls back to pipeline output if no cognitive content available.

        Complexity: T(n) ∈ O(k + r + i) for knowledge, reasoning, insights.
        """
        parts: List[str] = []

        # If pipeline output is good and relevant, use it as the base
        if pipeline_output and len(pipeline_output) > 50:
            # Check if pipeline output is relevant (not irrelevant knowledge)
            if self._is_pipeline_output_relevant(ctx, pipeline_output):
                parts.append(pipeline_output)

        # Add knowledge-based content
        if ctx.retrieved_facts and not parts:
            for fact, quality, importance in ctx.retrieved_facts[:2]:
                # Convert fact to sentence
                predicate_map = {
                    "is-a": "is", "means": "means", "refers-to": "refers to",
                    "defined-as": "is defined as", "causes": "causes",
                    "leads-to": "leads to", "uses": "uses", "requires": "requires",
                }
                predicate = predicate_map.get(fact.predicate, fact.predicate)
                subject = fact.subject.replace("_", " ")
                obj = fact.object.replace("_", " ")
                if quality > 0.5:
                    parts.append(f"{subject.capitalize()} {predicate} {obj}.")

        # Add reasoning conclusion if available
        if ctx.reasoning_result and ctx.reasoning_result.final_conclusion:
            conclusion = ctx.reasoning_result.final_conclusion
            if conclusion != "The available evidence is insufficient for a confident conclusion.":
                parts.append(conclusion)

        # Add cognitive flow insights if no other content
        if not parts and ctx.cognitive_flow:
            # Use the structure phase to format the response
            for insight in ctx.analysis_insights[:2]:
                parts.append(insight)

        if not parts:
            # Last resort: use pipeline output even if uncertain
            return pipeline_output if pipeline_output else (
                f"That's an interesting question about {ctx.domain}. "
                f"Let me think about this more. What specific aspect interests you?"
            )

        # Add follow-up based on goals and opinions
        follow_up = self._build_follow_up(ctx)
        response = " ".join(parts)
        if follow_up:
            response = f"{response} {follow_up}"

        return response

    def _is_pipeline_output_relevant(self, ctx: PipelineContext, output: str) -> bool:
        """Check if pipeline output is relevant to the user's question.

        Complexity: T(n) ∈ O(w) for w words in message.
        """
        # Extract key terms from user message
        stop = {"what", "is", "are", "the", "a", "an", "how", "does", "do",
                "why", "when", "where", "tell", "me", "about", "explain",
                "can", "you", "should", "i", "to", "of", "in", "on", "at"}
        msg_terms = set(
            w.lower().strip("?,.!") for w in ctx.user_message.split()
            if w.lower().strip("?,.!") not in stop and len(w) > 2
        )
        if not msg_terms:
            return True  # Can't determine — assume relevant

        output_lower = output.lower()
        overlap = sum(1 for term in msg_terms if term in output_lower)
        # Require at least 20% of message terms to appear in output
        min_overlap = max(1, len(msg_terms) * 0.2)
        return overlap >= min_overlap

    def _build_follow_up(self, ctx: PipelineContext) -> str:
        """Build a follow-up question based on goals and opinions.

        Complexity: T(n) ∈ O(1).
        """
        # If there's a clarification goal, ask about it
        clarify_goals = [g for g in ctx.relevant_goals if g.goal_type == "clarify"]
        if clarify_goals:
            return "Would you like me to go deeper on any part of this?"

        # If there are unclear opinions, offer to clarify
        unclear = [t for t, a in ctx.topic_assessments.items() if a == "unclear"]
        if unclear:
            return f"Would you like me to clarify anything about {unclear[0]}?"

        # If there are predicted topics, offer to explore
        if ctx.predicted_topics and ctx.predicted_topics[0][1] > 0.3:
            return f"Would you like to explore {ctx.predicted_topics[0][0]} as well?"

        return ""

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 7: VISION ALIGNMENT
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_7_vision_alignment(self, ctx: PipelineContext) -> None:
        """Vision alignment — er responsen aligned med identitet?

        Complexity: T(n) ∈ O(d) for d directives.
        """
        try:
            ctx.alignment = self._vision.evaluate_alignment(
                response=ctx.generated_response,
                intent=ctx.intent,
                used_knowledge=bool(ctx.retrieved_facts),
                used_templates=False,
                acknowledged_limitation=ctx.cognitive_flow is not None and ctx.cognitive_flow.overall_confidence < 0.3,
                connected_to_previous=len(ctx.conversation_history) > 0,
            )
        except Exception as exc:
            logger.error("Stage 7 Vision Alignment failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 8: REWARD
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_8_reward(self, ctx: PipelineContext) -> None:
        """Reward — kognitiv verdi assessment.

        Rewards are given based on:
        - Clarity boost: if cognitive flow has high confidence
        - Strategy boost: if alignment is high
        - Insight boost: if new knowledge was extracted
        - Breakthrough: if both confidence and alignment are very high

        Complexity: T(n) ∈ O(1).
        """
        try:
            # Clarity boost
            if ctx.cognitive_flow and ctx.cognitive_flow.overall_confidence > 0.7:
                reward = self._reward.record_clarity(
                    topic=ctx.domain,
                    previous_clarity=0.3,
                    new_clarity=ctx.cognitive_flow.overall_confidence,
                    domain=ctx.domain,
                )
                if reward:
                    ctx.rewards_earned.append(reward)

            # Strategy boost
            if ctx.alignment and ctx.alignment.alignment_score > 0.8:
                reward = self._reward.record_strategy_success(
                    strategy_name=ctx.strategy_used or "unified_pipeline",
                    intent=ctx.intent,
                    domain=ctx.domain,
                    success_score=ctx.alignment.alignment_score,
                )
                ctx.rewards_earned.append(reward)
                ctx.strategy_reward = reward.value

            # Breakthrough
            if (ctx.cognitive_flow and ctx.cognitive_flow.overall_confidence > 0.85
                    and ctx.alignment and ctx.alignment.alignment_score > 0.9):
                reward = self._reward.record_breakthrough(
                    description=f"High-confidence cognitive flow in {ctx.domain}",
                    domain=ctx.domain,
                    impact_score=ctx.cognitive_flow.overall_confidence,
                )
                ctx.rewards_earned.append(reward)

        except Exception as exc:
            logger.error("Stage 8 Reward failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 9: EXPERIENCE RECORDING
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_9_experience_recording(self, ctx: PipelineContext) -> None:
        """Experience recording — lagre som kognitiv struktur.

        Also extracts and validates knowledge from the conversation.

        Complexity: T(n) ∈ O(e + v) for extraction and validation.
        """
        try:
            # Extract knowledge from the conversation
            ctx.extraction = self._extractor.extract(
                user_message=ctx.user_message,
                ai_response=ctx.generated_response,
                intent=ctx.intent,
                emotion=ctx.emotion,
            )

            # Validate and store facts
            for fact in ctx.extraction.facts:
                self._validator.validate_and_store(
                    fact=fact,
                    conversation_intent=ctx.intent,
                    conversation_topics=ctx.extraction.topics,
                )

            # Store concepts
            for concept in ctx.extraction.concepts:
                self._store.add_concept(concept)

            # Store relationships
            for rel in ctx.extraction.relationships:
                self._store.add_relationship(rel)

            # Record experience in memory
            reward_value = sum(r.value for r in ctx.rewards_earned)
            ctx.experience_recorded = self._memory.record_experience(
                experience_type="conversation",
                description=ctx.user_message[:200],
                what_was_learned=ctx.extraction.topics[:3] if ctx.extraction.topics else ctx.domain,
                domain=ctx.domain,
                intent=ctx.intent,
                emotion=ctx.emotion,
                reward_value=reward_value,
                topics=list(ctx.extraction.topics) if ctx.extraction.topics else [],
                confidence=ctx.cognitive_flow.overall_confidence if ctx.cognitive_flow else 0.5,
            )

            # Record interaction for pattern recognition
            self._patterns.record_interaction(InteractionRecord(
                timestamp=time.time(),
                intent=ctx.intent,
                emotion=ctx.emotion,
                topics=tuple(ctx.extraction.topics) if ctx.extraction.topics else (),
                domain=ctx.domain,
                user_message_length=len(ctx.user_message),
                response_length=len(ctx.generated_response),
                response_time_ms=0.0,
                follow_up_asked=bool(ctx.retrieved_facts),
                user_returned=False,
                curiosity_score=ctx.extraction.user_knowledge_level if ctx.extraction else 0.5,
            ))

            # Record behavior
            self._behavior.record(
                message=ctx.user_message,
                intent=ctx.intent,
                emotion=ctx.emotion,
                domain=ctx.domain,
                response_length=len(ctx.generated_response),
            )

        except Exception as exc:
            logger.error("Stage 9 Experience Recording failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 10: GOAL PROGRESS
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_10_goal_progress(self, ctx: PipelineContext) -> None:
        """Goal progress — førte dette nærmere et mål?

        Complexity: T(n) ∈ O(g) for g goals.
        """
        try:
            # If cognitive confidence is low, create a clarification goal
            if ctx.cognitive_flow and ctx.cognitive_flow.overall_confidence < 0.4:
                goal = self._goals.set_goal(
                    description=f"Clarify understanding in {ctx.domain} domain",
                    goal_type="clarify",
                    domain=ctx.domain,
                    priority=0.7,
                )
                ctx.goals_updated.append(goal.goal_id)

            # Update progress on relevant goals
            for goal in ctx.relevant_goals:
                if ctx.cognitive_flow:
                    if goal.goal_type == "understand":
                        self._goals.update_goal_progress(goal.goal_id, 0.05)
                        ctx.goals_updated.append(goal.goal_id)
                    elif goal.goal_type == "clarify" and ctx.cognitive_flow.overall_confidence > 0.6:
                        self._goals.update_goal_progress(goal.goal_id, 0.1)
                        ctx.goals_updated.append(goal.goal_id)
                    elif goal.goal_type == "master" and ctx.cognitive_flow.overall_confidence > 0.7:
                        self._goals.update_goal_progress(goal.goal_id, 0.1)
                        ctx.goals_updated.append(goal.goal_id)

            # Derive new goals from knowledge gaps
            if ctx.extraction and ctx.extraction.topics:
                weak_domains = [ctx.domain] if ctx.cognitive_flow and ctx.cognitive_flow.overall_confidence < 0.4 else []
                low_confidence_topics = [
                    (t, ctx.cognitive_flow.overall_confidence)
                    for t in ctx.extraction.topics[:3]
                ] if ctx.cognitive_flow else []
                self._goals.derive_goals_from_gaps(weak_domains, low_confidence_topics)

        except Exception as exc:
            logger.error("Stage 10 Goal Progress failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 11: OPINION UPDATE
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_11_opinion_update(self, ctx: PipelineContext) -> None:
        """Opinion update — endret dette noen meninger?

        Complexity: T(n) ∈ O(p) for p phases.
        """
        try:
            if not ctx.extraction:
                return

            # Form opinions based on cognitive flow
            for topic in ctx.extraction.topics[:3]:
                assessment = "interesting"
                confidence = ctx.cognitive_flow.overall_confidence if ctx.cognitive_flow else 0.5

                # Pattern phase detected repetitions → important
                if any("Repetition" in i for i in ctx.pattern_insights):
                    assessment = "important"
                    confidence = max(confidence, 0.6)

                # Analysis phase found relationships → solid
                if any("relationship" in i.lower() for i in ctx.analysis_insights):
                    assessment = "solid"
                    confidence = max(confidence, 0.6)

                # Low confidence → unclear
                if ctx.cognitive_flow and ctx.cognitive_flow.overall_confidence < 0.4:
                    assessment = "unclear"

                # High reward → important
                if sum(r.value for r in ctx.rewards_earned) > 1.0:
                    assessment = "important"

                opinion = self._opinions.form_opinion(
                    topic=topic,
                    assessment=assessment,
                    domain=ctx.domain,
                    confidence=confidence,
                    evidence=f"Cognitive flow: {ctx.cognitive_flow.unified_insight[:80]}" if ctx.cognitive_flow else "",
                )
                ctx.opinions_formed.append(opinion)

        except Exception as exc:
            logger.error("Stage 11 Opinion Update failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 12: META-LEARNING
    # ═══════════════════════════════════════════════════════════════════════

    def _stage_12_meta_learning(self, ctx: PipelineContext) -> None:
        """Meta-learning — forbedret dette strategier?

        Complexity: T(n) ∈ O(1).
        """
        try:
            # Record outcome for meta-learning
            reward = ctx.strategy_reward
            if ctx.alignment:
                reward = max(reward, ctx.alignment.alignment_score)

            self._meta.record_outcome(
                strategy=ctx.strategy_used or "unified_pipeline",
                intent=ctx.intent,
                domain=ctx.domain,
                reward=reward,
                response_time_ms=(ctx.end_time - ctx.start_time) * 1000 if ctx.end_time > 0 else 0,
                user_continued=False,
            )

            # Record domain learning progress
            if ctx.extraction:
                self._meta.record_learning(
                    domain=ctx.domain,
                    facts_learned=len(ctx.extraction.facts),
                    concepts_learned=len(ctx.extraction.concepts),
                    avg_confidence=ctx.cognitive_flow.overall_confidence if ctx.cognitive_flow else 0.5,
                )

        except Exception as exc:
            logger.error("Stage 12 Meta-Learning failed: %s", exc)

    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def get_response(self, ctx: PipelineContext) -> str:
        """Get the generated response from a completed pipeline context.

        Complexity: T(n) ∈ O(1).
        """
        return ctx.generated_response

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all systems.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            return {
                "total_cycles": self._total_cycles,
                "reward": self._reward.get_stats(),
                "goals": self._goals.get_stats(),
                "vision": self._vision.get_stats(),
                "opinions": self._opinions.get_stats(),
                "memory": self._memory.get_stats(),
                "knowledge_store": {
                    "facts": self._store.fact_count,
                    "concepts": self._store.concept_count,
                    "relationships": self._store.relationship_count,
                },
                "patterns": len(self._patterns.recognize_patterns()),
                "meta": {
                    "outcomes": self._meta.total_outcomes,
                    "domains": self._meta.domains_tracked,
                    "strategies": self._meta.strategies_tracked,
                },
            }

    def get_state(self) -> Dict[str, Any]:
        """Get a snapshot of the brain's current state.

        Complexity: T(n) ∈ O(1).
        """
        with self._lock:
            stats = self.get_stats()
            top_goal = self._goals.get_top_priority()
            return {
                "total_cycles": self._total_cycles,
                "total_reward": stats["reward"]["total_reward"],
                "breakthroughs": stats["reward"]["breakthrough_count"],
                "stable_facts": stats["reward"]["stable_facts"],
                "active_goals": stats["goals"]["active_goals"],
                "alignment_trend": stats["vision"]["alignment_trend"],
                "total_opinions": stats["opinions"]["total_opinions"],
                "total_experiences": stats["memory"]["total_experiences"],
                "knowledge_facts": stats["knowledge_store"]["facts"],
                "knowledge_concepts": stats["knowledge_store"]["concepts"],
                "top_goal": top_goal.description if top_goal else "No active goals",
                "vision": self._vision.get_vision()[:100],
            }

    def get_vision(self) -> str:
        """Get the system's vision statement."""
        return self._vision.get_vision()

    def get_identity(self) -> str:
        """Get the system's identity statement."""
        return self._vision.get_identity_statement()

    # Direct access for testing/debugging
    @property
    def reward_system(self) -> RewardSystem:
        return self._reward

    @property
    def goal_system(self) -> GoalSystem:
        return self._goals

    @property
    def vision_system(self) -> VisionSystem:
        return self._vision

    @property
    def opinion_system(self) -> OpinionSystem:
        return self._opinions

    @property
    def experience_memory(self) -> ExperienceMemory:
        return self._memory

    @property
    def knowledge_store(self) -> KnowledgeStore:
        return self._store

    @property
    def pattern_recognizer(self) -> PatternRecognizer:
        return self._patterns

    @property
    def meta_learning(self) -> MetaLearningSystem:
        return self._meta
