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
PersonalAgent — Extended Personal Assistant Agent using Condrox AI.

Uses Condrox's BrainController.process_request() for intelligence.
No external LLM — all reasoning done by Condrox pipeline.

Extended with PA-specific tools and workflows:
- Role Definition (PA persona, tone, domain tags)
- Capability Layer (TaskManager, ReminderEngine, InfoLookup, NoteStore, QuickPlan, ToneController)
- Workflow Layer (behavioral logic for PA tasks)
- Memory Integration (TaskMemory, ReminderMemory, NoteMemory, ContextLinking)
- Extended Tools (TimeParser, PriorityClassifier, SmartSuggest, AutoPlan, DailyBrief)
- Integration Layer (Main AI control enforcement)
- Quality & Safety Layer (4 Quality Gates, policy enforcer)

Architecture:
- Takes BrainController as input (Main AI)
- Uses BrainController.process_request() for all intelligence
- Extends with PA-specific tools and workflows
- Has no own pipeline, identity, reasoning, or context
- Everything goes through Main AI → agent → Main AI

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController

from agents.PersonalAgent.src.role_definition import RoleDefinition, ToneProfile
from agents.PersonalAgent.src.capability_layer import (
    TaskManager,
    ReminderEngine,
    NoteStore,
    QuickPlan,
    ToneController,
    InfoLookup,
)
from agents.PersonalAgent.src.workflow_layer import WorkflowLayer, WorkflowResult
from agents.PersonalAgent.src.memory_integration import MemoryIntegration
from agents.PersonalAgent.src.extended_tools import (
    TimeParser,
    PriorityClassifier,
    SmartSuggest,
    AutoPlan,
    DailyBrief,
)
from agents.PersonalAgent.src.integration_layer import IntegrationLayer
from agents.PersonalAgent.src.quality_safety import QualityGates, PolicyEnforcer


class PersonalAgentState(Enum):
    """
    PersonalAgent state machine states.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PersonalAgentResult:
    """
    Result from PersonalAgent processing.

    Attributes:
        request_id: Unique request identifier.
        response: Natural conversational response.
        emotion: Detected emotion (if any).
        context_used: Whether conversation context was used.
        confidence: Confidence score in [0.0, 1.0].
        metadata: Additional metadata.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    request_id: str
    response: str
    emotion: str = ""
    context_used: bool = False
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class PersonalAgent:
    """
    Extended Personal Assistant using Condrox's BrainController.

    Uses BrainController.process_request() for all intelligence.
    Extends with PA-specific tools, workflows, and quality gates.

    Architecture:
    - Main AI is the brain (BrainController)
    - Agent is an extension with PA-specific tools
    - Agent has no own pipeline, identity, reasoning, or context
    - Everything goes through Main AI → agent → Main AI

    Complexity:
        process: T(n) ∈ O(pipeline + workflow) where pipeline = 18 stages.
    """

    __slots__ = (
        "_brain_controller",
        "_role_definition",
        "_integration_layer",
        "_workflow_layer",
        "_memory_integration",
        "_quality_gates",
        "_policy_enforcer",
        "_time_parser",
        "_priority_classifier",
        "_smart_suggest",
        "_auto_plan",
        "_daily_brief",
        "_state",
        "_request_count",
    )

    def __init__(
        self,
        brain_controller: "BrainController",
        tone_profile: ToneProfile = ToneProfile.PROFESSIONAL,
    ) -> None:
        """Initialize the Personal Assistant agent.

        Args:
            brain_controller: Condrox BrainController instance (Main AI).
            tone_profile: Tone profile for PA role.
        """
        self._brain_controller = brain_controller
        self._role_definition = RoleDefinition(tone_profile=tone_profile)
        self._state: PersonalAgentState = PersonalAgentState.IDLE
        self._request_count: int = 0

        # Layer 6: Integration Layer - Main AI Control
        self._integration_layer = IntegrationLayer(brain_controller, self._role_definition)

        # Layer 2: Capability Layer - Core Tools
        self._workflow_layer = WorkflowLayer(brain_controller, self._role_definition)
        self._memory_integration = MemoryIntegration(brain_controller)

        # Layer 5: Extended Tools
        self._time_parser = TimeParser()
        self._priority_classifier = PriorityClassifier()
        self._smart_suggest = SmartSuggest()
        self._auto_plan = AutoPlan(brain_controller)
        self._daily_brief = DailyBrief(
            self._workflow_layer._task_manager,
            self._workflow_layer._note_store,
        )

        # Layer 7: Quality & Safety Layer
        self._quality_gates = QualityGates(tone_profile=tone_profile)
        self._policy_enforcer = PolicyEnforcer()

    @property
    def state(self) -> PersonalAgentState:
        """Return the current state.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._state

    @property
    def request_count(self) -> int:
        """Return the total number of requests processed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._request_count

    def process(
        self,
        user_input: str,
        context: str = "",
    ) -> PersonalAgentResult:
        """
        Process user input and return conversational response.

        Uses BrainController.process_request() for all intelligence.
        The BrainController provides:
        - Multi-turn reasoning for context
        - Emotion detection and empathy
        - Knowledge retrieval (local + Wikidata API)
        - Memory graph for conversation context
        - Response style engine for natural language

        Args:
            user_input: User's message or question.
            context: Additional context string (default: empty).

        Returns:
            PersonalAgentResult with conversational response.

        Complexity: T(n) ∈ O(pipeline) where pipeline = 18 stages.
        """
        if not user_input.strip():
            return PersonalAgentResult(
                request_id=uuid.uuid4().hex[:16],
                response=self._role_definition.get_greeting(),
                confidence=1.0,
            )

        self._state = PersonalAgentState.PROCESSING
        request_id = uuid.uuid4().hex[:16]
        start_time = time.perf_counter()

        try:
            # Check if this is a PA-specific workflow
            workflow_result = self._workflow_layer.execute(user_input)

            if workflow_result.success:
                # PA workflow executed successfully
                response = workflow_result.message
                if workflow_result.follow_up:
                    response += f"\n{workflow_result.follow_up}"

                # Quality gate check
                gate_check = self._quality_gates.check_tone_consistency(response)
                if gate_check.result.value == "fail":
                    response = workflow_result.message  # Use original if tone check fails

                return PersonalAgentResult(
                    request_id=request_id,
                    response=response,
                    confidence=0.9,
                    metadata={
                        "processing_time": time.perf_counter() - start_time,
                        "workflow": "pa_specific",
                        "data": workflow_result.data,
                    },
                )

            # Fall back to Main AI for general conversation
            state = self._brain_controller.process_request(user_input)

            # Extract conversational intelligence from BrainState
            emotion = ""
            context_used = False

            if hasattr(state, "metadata"):
                emotion = state.metadata.get("emotion", "")
                context_used = state.metadata.get("context_used", False)

            # Try to get emotion from EmotionDetector if available
            if not emotion and hasattr(self._brain_controller, "emotion_detector"):
                try:
                    emotion_result = self._brain_controller.emotion_detector.detect(
                        user_input
                    )
                    emotion = emotion_result.get("emotion", "")
                except Exception:
                    pass

            # Build result
            result = PersonalAgentResult(
                request_id=request_id,
                response=state.final_output,
                emotion=emotion,
                context_used=context_used,
                confidence=state.confidence if hasattr(state, "confidence") else 0.8,
                metadata={
                    "processing_time": time.perf_counter() - start_time,
                    "mode": state.mode if hasattr(state, "mode") else "conversational",
                    "domain": state.domain if hasattr(state, "domain") else "general",
                    "workflow": "main_ai",
                },
            )

            self._state = PersonalAgentState.COMPLETED
            self._request_count += 1
            return result

        except Exception as exc:
            self._state = PersonalAgentState.FAILED
            return PersonalAgentResult(
                request_id=request_id,
                response=f"I encountered an error: {str(exc)}",
                confidence=0.0,
                metadata={"error": str(exc)},
            )

    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context from BrainController.

        Returns:
            Dictionary with conversation context.
        """
        try:
            if hasattr(self._brain_controller, 'memory_graph'):
                return self._brain_controller.memory_graph.get_context()
        except Exception:
            pass
        return {}

    def clear_conversation_context(self) -> None:
        """Clear conversation context for fresh start."""
        try:
            if hasattr(self._brain_controller, 'memory_manager'):
                self._brain_controller.memory_manager.store._items.clear()
        except Exception:
            pass

    def get_daily_brief(self) -> str:
        """Get daily brief summary of tasks and notes.

        Returns:
            Formatted daily brief.

        Complexity: T(n) ∈ O(n) for n tasks and notes, S(n) ∈ O(n).
        """
        return self._daily_brief.generate()

    def validate_main_ai_control(self) -> Dict[str, bool]:
        """Validate that agent is under Main AI control.

        Returns:
            Dictionary with validation results.
        """
        return self._integration_layer.validate_all_constraints()

    def status(self) -> Dict[str, Any]:
        """Return a status snapshot.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return {
            "state": self._state.value,
            "request_count": self._request_count,
            "brain_controller": self._brain_controller is not None,
            "role": self._role_definition.role,
            "tone_profile": self._role_definition.tone_profile.value,
            "domain_tags": self._role_definition.domain_tags.to_list(),
            "main_ai_control": self.validate_main_ai_control(),
        }
