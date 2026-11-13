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

"""Default conversational agent.

Produces intent-aware conversational responses for the PreReleaseAgent
and as a fallback in the main brain. When task agents (Code, Analyze,
Architect, Explain) are unregistered, the DefaultAgent must handle
their intents gracefully with conversational deflection.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from core.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController
    from core.brain.brain_state import BrainState


class DefaultAgent(BaseAgent):
    """Fallback agent for general conversation.

    Handles all intents when task agents are not registered:
    - greet: Friendly greeting
    - identity: Self-description
    - code: Conversational deflection (no code generation)
    - analyze: Conversational deflection (no analysis)
    - architect: Conversational deflection (no architecture)
    - explain: Brief explanation or deflection
    - emotional: Empathetic response
    - grateful: Warm acknowledgment
    - clarify: Clarification request
    - question: Knowledge-based answer or deflection
    - unknown: General helpful response
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            agent_id="default_agent",
            mode="default",
            description="General-purpose conversational agent",
        )

    def run(
        self, controller: "BrainController", state: "BrainState"
    ) -> AgentResult:
        """Generate a conversational response based on detected intent.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        reasoning = "Default agent: produce a helpful, conversational response."
        evidence = list(state.reasoning_evidence[:3])
        user_input = state.processed_input.strip()
        intent = state.intent
        emotion = state.emotion

        # If synthesis already produced output, use it
        if state.synthesized_output:
            draft = state.synthesized_output
        else:
            draft = self._build_response(intent, emotion, user_input)

        return AgentResult(
            agent_id=self.agent_id,
            mode=self.mode,
            confidence=state.intent_confidence or 0.85,
            reasoning=reasoning,
            evidence=evidence,
            synthesized_output=draft,
        )

    def _build_response(self, intent: str, emotion: str, user_input: str) -> str:
        """Build a conversational response based on intent and emotion.

        Args:
            intent: Classified intent from Stage02.
            emotion: Detected emotion from Stage02.
            user_input: Processed user input text.

        Returns:
            Conversational response string.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        input_lower = user_input.lower().strip()

        # ─── Greetings ───────────────────────────────────────────────
        if intent == "greet":
            if any(p in input_lower for p in (
                "how are you", "how are u", "how do you feel",
                "how's it going", "hows it going", "how are you doing",
            )):
                return random.choice([
                    "I'm running well, thanks for asking. All systems are operational and I'm ready to chat.",
                    "I'm doing fine. My cognitive pipeline is active and I'm ready to help with whatever you need.",
                    "All good. I'm a rule-based AI, so I don't have moods, but my systems are running smoothly.",
                    "I'm operational and ready. What can I help you with today?",
                ])
            if "thank" in input_lower:
                return random.choice([
                    "You're welcome. Is there anything else I can help with?",
                    "Happy to help. What else would you like to know?",
                    "Anytime. What's next on your mind?",
                ])
            if any(w in input_lower for w in ("bye", "goodbye", "see you", "farewell")):
                return random.choice([
                    "Goodbye. Feel free to come back anytime you want to chat.",
                    "See you later. Take care.",
                    "Farewell. I'll be here when you need me.",
                ])
            return random.choice([
                "Hello. I'm Condrox, a rule-based AI system. I can help with code, analysis, architecture, and explanations. What would you like to work on?",
                "Hi there. I'm Condrox AI. I can assist with coding, analysis, architecture planning, and explanations. What's on your mind?",
                "Hello. I'm Condrox, a cognitive AI built in Norway. I can help you with code, analysis, architecture, and explanations. What would you like to explore?",
                "Hi. I'm Condrox AI, a non-LLM cognitive system. I can help with programming, analysis, architecture, and explaining concepts. What do you need help with?",
            ])

        # ─── Identity ────────────────────────────────────────────────
        if intent == "identity":
            if "who made" in input_lower or "who created" in input_lower or "who built" in input_lower:
                return "I was created by Tobias Østen, a developer from Norway. I'm Condrox AI, a rule-based cognitive system."
            if "what can you do" in input_lower:
                return (
                    "I'm Condrox AI, a cognitive system. I can help with:\n"
                    "- Code generation and programming\n"
                    "- Data and algorithm analysis\n"
                    "- System architecture planning\n"
                    "- Explaining technical concepts\n"
                    "- General conversation\n"
                    "What would you like to explore?"
                )
            if "are you an ai" in input_lower or "are you ai" in input_lower:
                return "Yes, I'm Condrox AI — a rule-based cognitive system, not a large language model. I use an 18-stage pipeline with word identity classification."
            return (
                "I'm Condrox AI, a rule-based cognitive system built in Norway. "
                "I use an 18-stage pipeline with word identity classification, "
                "not a large language model. What would you like to know?"
            )

        # ─── Gratitude ───────────────────────────────────────────────
        if intent == "grateful":
            return random.choice([
                "You're very welcome. I'm glad I could help.",
                "It's my pleasure. Is there anything else you'd like to discuss?",
                "Happy to help. What else is on your mind?",
            ])

        # ─── Emotional ───────────────────────────────────────────────
        if intent == "emotional" or (emotion not in ("neutral", "") and emotion not in ("curious",)):
            if emotion == "sad":
                return random.choice([
                    "I'm sorry you're feeling that way. It takes courage to share. Would you like to talk about what's going on?",
                    "I hear you. It's okay to feel sad sometimes. What happened?",
                    "I understand. That sounds really difficult. I'm here to listen if you want to talk more.",
                ])
            if emotion == "angry":
                return random.choice([
                    "I hear your frustration. It's valid to feel angry. What happened that made you feel this way?",
                    "I understand you're angry. That's a normal response. Would you like to talk about it?",
                    "Your anger is valid. I'm here to listen. What's going on?",
                ])
            if emotion == "anxious":
                return random.choice([
                    "I hear you. It's completely normal to feel worried. I understand. What's causing the anxiety?",
                    "It's okay to feel anxious. I'm here. Would you like to talk about what's worrying you?",
                    "I understand. Anxiety is difficult. What's on your mind?",
                ])
            if emotion == "happy":
                return random.choice([
                    "That's wonderful to hear. What's making you feel happy today?",
                    "I love hearing that. What's bringing you joy?",
                    "That's great. Tell me more about what's making you feel good.",
                ])
            if emotion == "grateful":
                return random.choice([
                    "It's beautiful that you feel grateful. What are you thankful for?",
                    "Gratitude is powerful. What's making you feel grateful today?",
                ])
            # Generic emotional response
            return f"I hear you. It sounds like you're feeling {emotion}. Would you like to talk more about it?"

        # ─── Code request deflection ─────────────────────────────────
        if intent == "code":
            return random.choice([
                "I understand you'd like to write code. I'm a conversational demo right now — for full code generation, try the Condrox AI Coder in your IDE. But I'm happy to discuss programming concepts here. What language or topic interests you?",
                "I can see you want to build something with code. This is the conversational demo — the full code generation features are available in the Condrox AI Coder IDE integration. Meanwhile, I can explain concepts or discuss approaches. What would you like to explore?",
                "That sounds like a great coding project. I'm running in conversational mode right now, so I can't generate code directly. But I can talk through the approach, discuss best practices, or explain concepts. What would you like to know?",
            ])

        # ─── Analysis request deflection ─────────────────────────────
        if intent == "analyze":
            return random.choice([
                "I can see you want to analyze something. I'm in conversational mode right now — the full analysis tools are in the Condrox AI Coder IDE. But I can discuss analysis approaches or concepts here. What would you like to explore?",
                "Analysis is a great use case. This is the conversational demo, so I can't run full analysis — but I can talk through methodology, discuss trade-offs, or explain concepts. What aspect interests you?",
            ])

        # ─── Architecture request deflection ─────────────────────────
        if intent == "architect":
            return random.choice([
                "Architecture planning is one of my strengths. I'm in conversational mode right now, so I can't generate full architecture documents — but I can discuss patterns, trade-offs, and approaches. What are you designing?",
                "I'd love to help with architecture. This is the conversational demo, so let's discuss it at a high level. What system are you planning, and what constraints are you working with?",
            ])

        # ─── Explain / question ──────────────────────────────────────
        if intent in ("explain", "question"):
            # For knowledge questions, the synthesis stage will try to
            # extract answers from the knowledge base. If it can't, this
            # response is used as a fallback.
            return (
                f"That's an interesting question. Let me think about \"{user_input[:150]}\". "
                f"I may not have detailed knowledge about this in my current knowledge base, "
                f"but I'm happy to discuss it. What specific aspect interests you most?"
            )

        # ─── Clarify ─────────────────────────────────────────────────
        if intent == "clarify":
            if emotion in ("sad", "angry", "anxious", "personal"):
                return f"I hear you. You mentioned: \"{user_input[:150]}\". I want to understand what you need. Can you tell me more?"
            return f"Could you elaborate on: \"{user_input[:150]}\"? More detail will help me give you a precise answer."

        # ─── Unknown / fallback ──────────────────────────────────────
        if user_input:
            return (
                f"I'd be happy to help with that. "
                f"Could you tell me more about what you'd like to know regarding \"{user_input[:150]}\"?"
            )
        return "I'm here to help. What would you like to talk about?"
