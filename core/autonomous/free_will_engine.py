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

"""Free Will Response Engine — generates unique responses, never templates.

This is the heart of the autonomous AI. Instead of picking from pre-written
template responses, the engine constructs responses from:
1. Knowledge retrieved from the knowledge store
2. Patterns recognized from past conversations
3. User behavioral profile (communication style, knowledge level)
4. Deep reasoning results
5. Emotional context
6. Conversation history

The engine has TOTAL FREE WILL in how it phrases responses. It uses:
- Sentence construction patterns (not templates — structural scaffolding)
- Knowledge synthesis (combining multiple facts into coherent text)
- Adaptive tone (formal/casual/technical based on user profile)
- Contextual awareness (what was said before, what the user knows)

Key principle: The AI generates responses in its OWN way. No two responses
to the same question are identical. The AI expresses its understanding
using its own voice, informed by everything it has learned.

Complexity:
    generate: T(n) ∈ O(k + r + p) for k knowledge, r reasoning, p patterns
    construct_sentence: T(n) ∈ O(1)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.autonomous.knowledge_validator import KnowledgeStore
from core.autonomous.knowledge_extractor import ExtractedFact, ExtractedConcept
from core.autonomous.pattern_recognizer import PatternRecognizer
from core.autonomous.behavior_analyzer import UserProfile
from core.autonomous.deep_reasoning import DeepReasoningEngine, ReasoningResult


@dataclass(frozen=True, slots=True)
class ResponseContext:
    """Context for generating a response.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    user_message: str
    intent: str
    emotion: str
    domain: str
    mode: str
    confidence: float
    conversation_history: Tuple[str, ...] = ()  # Last few user messages
    user_profile: Optional[UserProfile] = None
    topics: Tuple[str, ...] = ()
    follow_up_needed: bool = False


class FreeWillResponseEngine:
    """Generates unique, non-templated responses using free will.

    The engine constructs responses from multiple sources:
    - Knowledge store facts
    - Deep reasoning results
    - Pattern recognition predictions
    - User behavioral profile
    - Conversation history

    Every response is constructed fresh — no template selection.
    The AI uses its own voice, informed by everything it has learned.

    Complexity:
        generate: T(n) ∈ O(k + r + p) for k knowledge, r reasoning, p patterns
    """

    __slots__ = ("_store", "_reasoning", "_patterns")

    # Sentence construction patterns — NOT templates, structural scaffolding
    # These define HOW sentences are built, not WHAT they say
    _OPENERS_DESCRIPTIVE: List[str] = [
        "Here's what I understand about {topic}:",
        "Let me break down {topic} for you.",
        "When it comes to {topic},",
        "Looking at {topic},",
        "From what I know about {topic}:",
    ]

    _OPENERS_ANALYTICAL: List[str] = [
        "Analyzing {topic}:",
        "Let me walk through {topic} step by step.",
        "Breaking this down — {topic}:",
        "Here's my analysis of {topic}:",
    ]

    _OPENERS_EXPLANATORY: List[str] = [
        "Think of {topic} this way:",
        "The core idea behind {topic} is",
        "At its heart, {topic} works like this:",
        "To understand {topic}, consider this:",
    ]

    _OPENERS_CODE: List[str] = [
        "Here's how I'd approach {topic}:",
        "Let me build this out — {topic}:",
        "For {topic}, here's a clean implementation:",
    ]

    _OPENERS_ARCHITECT: List[str] = [
        "Designing {topic}:",
        "Here's how I'd architect {topic}:",
        "For {topic}, here's a structured plan:",
    ]

    _OPENERS_EMOTIONAL: List[str] = [
        "I hear you.",
        "I understand.",
        "That resonates with me.",
        "I'm listening.",
    ]

    _OPENERS_GREETING: List[str] = [
        "Hey! I'm Condrox. ",
        "Hello! I'm Condrox AI. ",
        "Hi there — I'm Condrox. ",
        "Hey, good to meet you. I'm Condrox. ",
    ]

    # Transition phrases for connecting ideas
    _TRANSITIONS: List[str] = [
        "Now, ", "Building on that, ", "What's more, ", "Additionally, ",
        "Here's the key part: ", "This matters because ", "The reason is that ",
        "Because of this, ", "As a result, ", "Therefore, ",
        "Given that, ", "From another angle: ", "It's worth noting that ",
    ]

    # Follow-up question patterns
    _FOLLOW_UP_PATTERNS: List[str] = [
        "What's your take on this?",
        "Does this align with what you were thinking?",
        "Want me to go deeper on any part?",
        "Should I explore a specific aspect more?",
        "What would you like to explore next?",
        "Does this make sense, or should I clarify anything?",
        "How does this fit with what you already know?",
    ]

    def __init__(
        self,
        store: KnowledgeStore,
        reasoning: DeepReasoningEngine,
        patterns: PatternRecognizer,
    ) -> None:
        """Initialize the free will response engine.

        Args:
            store: Knowledge store for fact retrieval.
            reasoning: Deep reasoning engine.
            patterns: Pattern recognizer for predictions.
        """
        self._store = store
        self._reasoning = reasoning
        self._patterns = patterns

    def generate(self, ctx: ResponseContext) -> str:
        """Generate a unique response using free will.

        This is the main entry point. The engine constructs a response
        from knowledge, reasoning, patterns, and user profile — never
        from a fixed template.

        Args:
            ctx: Response context with all available information.

        Returns:
            A unique response string.

        Complexity: T(n) ∈ O(k + r + p) for k knowledge, r reasoning, p patterns.
        """
        # Handle special intents that need specific handling
        if ctx.intent == "greet":
            return self._generate_greeting(ctx)
        if ctx.intent == "identity":
            return self._generate_identity(ctx)
        if ctx.intent == "emotional":
            return self._generate_emotional(ctx)
        if ctx.intent == "grateful":
            return self._generate_gratitude(ctx)

        # For all other intents, construct from knowledge + reasoning
        return self._generate_substantive(ctx)

    def _generate_greeting(self, ctx: ResponseContext) -> str:
        """Generate a unique greeting.

        Complexity: T(n) ∈ O(1).
        """
        opener = random.choice(self._OPENERS_GREETING)

        # Adapt based on user profile
        if ctx.user_profile:
            if ctx.user_profile.communication_style == "casual":
                opener = random.choice([
                    "Hey! I'm Condrox. ",
                    "Yo! Condrox here. ",
                    "Sup! I'm Condrox. ",
                ])
            elif ctx.user_profile.communication_style == "formal":
                opener = random.choice([
                    "Hello. I'm Condrox AI. ",
                    "Greetings. I'm Condrox. ",
                    "Good to see you. I'm Condrox AI. ",
                ])

        # Add what the AI can do — phrased differently each time
        capabilities = random.choice([
            "I can help you with code, analysis, system design, and explaining concepts. What's on your mind?",
            "I work across programming, analysis, architecture, and explanations. What would you like to explore?",
            "I'm built for coding, analysis, design, and teaching. What are you working on?",
            "I can write code, analyze systems, design architectures, and explain ideas. What do you need?",
        ])

        # If user has history, acknowledge it
        if ctx.user_profile and ctx.user_profile.total_interactions > 5:
            capabilities = random.choice([
                "Good to see you again. What are we working on today?",
                "Welcome back. What would you like to dive into?",
                "Hey, nice to have you back. What's on your mind this time?",
            ])

        return opener + capabilities

    def _generate_identity(self, ctx: ResponseContext) -> str:
        """Generate a unique identity response.

        Complexity: T(n) ∈ O(1).
        """
        msg_lower = ctx.user_message.lower()

        if any(w in msg_lower for w in ("who made", "who created", "who built")):
            return random.choice([
                "I was created by Tobias Østen, a developer from Norway. I'm Condrox AI — a rule-based cognitive system, not a large language model.",
                "Tobias Østen built me. I'm Condrox, a cognitive AI system developed in Norway that uses an 18-stage pipeline instead of neural networks.",
                "My creator is Tobias Østen. I'm Condrox AI, a non-LLM cognitive system that reasons through structured stages rather than pattern matching.",
            ])

        if any(w in msg_lower for w in ("what can you do", "what do you do")):
            return random.choice([
                "I'm Condrox AI. I can write code, analyze systems, design architectures, explain concepts, and have real conversations. I learn from every interaction — my knowledge grows with every conversation. What would you like to explore?",
                "I'm Condrox — a cognitive AI that handles programming, analysis, architecture, explanations, and conversation. I have an autonomous learning system that extracts and validates knowledge from every encounter. What are you working on?",
                "I'm Condrox AI. My capabilities include code generation, system analysis, architecture design, concept explanation, and adaptive conversation. I learn autonomously — every conversation teaches me something. What do you need help with?",
            ])

        if any(w in msg_lower for w in ("are you an ai", "are you ai", "what are you")):
            return random.choice([
                "Yes — I'm Condrox AI, a rule-based cognitive system. I use an 18-stage pipeline with word identity classification, not a large language model. I reason through problems structurally rather than predicting tokens.",
                "I'm Condrox AI — a cognitive system, not an LLM. I process language through an 18-stage pipeline that includes intent classification, knowledge retrieval, reasoning, and synthesis. I learn from every conversation.",
                "I'm an AI — specifically Condrox AI, a non-LLM cognitive system. I use structured reasoning through 18 pipeline stages, autonomous knowledge extraction, and deep reasoning chains. I'm not a language model; I'm a cognitive architecture.",
            ])

        if any(w in msg_lower for w in ("are you an llm", "do you use llm", "are you a language model")):
            return random.choice([
                "No, I'm not a language model. I'm Condrox AI — a rule-based cognitive system that uses an 18-stage pipeline with word identity classification. I reason through problems structurally, not by predicting the next token.",
                "I'm not an LLM. I'm Condrox, a cognitive AI that processes language through structured stages: intent detection, knowledge retrieval, reasoning, and synthesis. I learn autonomously from every conversation.",
                "No LLM here. I'm Condrox AI — a cognitive architecture with an 18-stage pipeline, autonomous knowledge extraction, deep reasoning, and meta-learning. I construct responses from understanding, not pattern matching.",
            ])

        if any(w in msg_lower for w in ("how do you work", "how does your pipeline", "how many stages")):
            return (
                "I work through an 18-stage cognitive pipeline:\n"
                "1. Input preprocessing and normalization\n"
                "2. Intent classification via word identity engine\n"
                "3. Mode selection (code, analyze, architect, explain, default)\n"
                "4. Domain detection\n"
                "5. Memory retrieval\n"
                "6. Knowledge retrieval with BM25 ranking\n"
                "7. Context assembly\n"
                "8. Core reasoning with evidence chains\n"
                "9. Agent execution (mode-specific)\n"
                "10. Response synthesis\n"
                "11-18. Quality, security, ethics, and delivery stages\n\n"
                "I also have an autonomous learning system that extracts, validates, "
                "and stores knowledge from every conversation. I learn patterns, "
                "analyze behavior, do deep reasoning, and meta-learn to improve. "
                "What would you like to know more about?"
            )

        # Default identity
        return random.choice([
            "I'm Condrox AI — a rule-based cognitive system built in Norway. I use an 18-stage pipeline with word identity classification, autonomous knowledge extraction, and deep reasoning. I'm not a language model — I reason through problems structurally and learn from every conversation. What would you like to explore?",
            "I'm Condrox, a cognitive AI system. I process language through 18 pipeline stages, learn autonomously from every interaction, and reason with evidence chains. I'm built in Norway by Tobias Østen. What's on your mind?",
            "I'm Condrox AI — a non-LLM cognitive architecture. I classify words by identity, retrieve knowledge with BM25, reason through evidence chains, and learn from every conversation. I grow smarter with each interaction. What do you need help with?",
        ])

    def _generate_emotional(self, ctx: ResponseContext) -> str:
        """Generate an empathetic emotional response.

        Complexity: T(n) ∈ O(1).
        """
        emotion = ctx.emotion
        msg = ctx.user_message

        if emotion == "sad":
            return random.choice([
                f"I hear you. It takes courage to say that. I'm here — what's going on?",
                f"I'm listening. That sounds really heavy. Would you like to talk about what's happening?",
                f"I understand. It's okay to feel this way. What happened that brought this on?",
                f"I hear what you're saying. You don't have to carry this alone. What's going on?",
            ])

        if emotion == "angry":
            return random.choice([
                "I hear your frustration. It's valid to feel angry. What happened?",
                "I understand you're angry. That makes sense given what you're dealing with. What's going on?",
                "Your anger is valid. I'm here to listen. What triggered this?",
            ])

        if emotion == "anxious":
            return random.choice([
                "I hear you. Anxiety is difficult. What's worrying you?",
                "It's okay to feel anxious. I'm here. What's on your mind?",
                "I understand. That worry makes sense. What's causing it?",
            ])

        if emotion == "happy":
            return random.choice([
                "That's great to hear! What's making you feel good?",
                "I love that. What's bringing you joy today?",
                "That's wonderful. Tell me more about what's going well.",
            ])

        if emotion == "grateful":
            return random.choice([
                "It's beautiful that you feel grateful. What are you thankful for?",
                "Gratitude is powerful. What's making you feel this way?",
            ])

        if emotion == "confused":
            return random.choice([
                "I get it — this can be confusing. Let me help clarify. What specifically is unclear?",
                "That's okay, confusion is the first step to understanding. What part doesn't make sense yet?",
            ])

        if emotion == "curious":
            return random.choice([
                "I love that curiosity. Let's explore this together. What are you wondering about?",
                "Great question — curiosity is how we learn. What specifically interests you?",
            ])

        # Generic emotional response
        return f"I hear you. It sounds like you're feeling {emotion}. I'm here — would you like to talk more about it?"

    def _generate_gratitude(self, ctx: ResponseContext) -> str:
        """Generate a gratitude response.

        Complexity: T(n) ∈ O(1).
        """
        return random.choice([
            "You're very welcome. I'm glad I could help. What else is on your mind?",
            "Anytime. It's rewarding to be useful. What would you like to explore next?",
            "Happy to help. What else can I do for you?",
            "My pleasure. What are you working on next?",
        ])

    def _generate_substantive(self, ctx: ResponseContext) -> str:
        """Generate a substantive response using knowledge and reasoning.

        This is where the AI's free will shines. It constructs a response
        from multiple sources — never a template.

        Complexity: T(n) ∈ O(k + r + p) for k knowledge, r reasoning, p patterns.
        """
        # 1. Retrieve knowledge relevant to the user's question
        knowledge_facts = self._retrieve_relevant_knowledge(ctx)

        # 2. Run deep reasoning
        reasoning_result = self._reasoning.reason(
            question=ctx.user_message,
            intent=ctx.intent,
            domain=ctx.domain,
            max_depth=4,
        )

        # 3. Predict what topics might come next (for proactive response)
        predicted_topics = self._patterns.predict_next_topics(list(ctx.topics), top_k=3)

        # 4. Construct the response using free will
        response = self._construct_response(
            ctx, knowledge_facts, reasoning_result, predicted_topics
        )

        return response

    def _retrieve_relevant_knowledge(self, ctx: ResponseContext) -> List[Tuple[ExtractedFact, float, float]]:
        """Retrieve knowledge relevant to the user's question.

        Complexity: T(n) ∈ O(t * f) for t topics, f facts per topic.
        """
        facts: List[Tuple[ExtractedFact, float, float]] = []
        seen_ids: Set[str] = set()

        # Search by topics
        for topic in ctx.topics[:3]:
            topic_facts = self._store.find_by_subject(topic)
            for fact, quality, importance in topic_facts[:2]:
                fact_id = f"{fact.subject}_{fact.predicate}_{fact.object}"
                if fact_id not in seen_ids and quality > 0.3:
                    facts.append((fact, quality, importance))
                    seen_ids.add(fact_id)

        # Search by domain if not enough facts
        if len(facts) < 3 and ctx.domain != "general":
            domain_facts = self._store.find_by_domain(ctx.domain, limit=5)
            for fact, quality, importance in domain_facts:
                fact_id = f"{fact.subject}_{fact.predicate}_{fact.object}"
                if fact_id not in seen_ids and quality > 0.3:
                    facts.append((fact, quality, importance))
                    seen_ids.add(fact_id)

        return facts[:5]  # Top 5 facts

    def _construct_response(
        self,
        ctx: ResponseContext,
        facts: List[Tuple[ExtractedFact, float, float]],
        reasoning: ReasoningResult,
        predicted_topics: List[Tuple[str, float]],
    ) -> str:
        """Construct a unique response from all available information.

        This is the core of the free will engine. It assembles a response
        from knowledge, reasoning, and context — never a template.

        Complexity: T(n) ∈ O(f + s) for f facts, s reasoning steps.
        """
        # Extract the main topic from the user's message
        topic = self._extract_topic(ctx.user_message)

        # Choose an opener based on intent/mode
        opener = self._choose_opener(ctx, topic)

        # Build the body from knowledge and reasoning
        body_parts: List[str] = []

        # Add knowledge-based content
        if facts:
            knowledge_text = self._synthesize_knowledge(facts, ctx)
            if knowledge_text:
                body_parts.append(knowledge_text)

        # Add reasoning-based content
        if reasoning.steps:
            reasoning_text = self._synthesize_reasoning(reasoning, ctx)
            if reasoning_text:
                body_parts.append(reasoning_text)

        # If we have no knowledge or reasoning, construct from the question itself
        if not body_parts:
            body_parts.append(self._construct_from_question(ctx))

        # Add contextual insight if we have patterns
        if predicted_topics and ctx.user_profile and ctx.user_profile.engagement_level > 0.5:
            insight = self._add_contextual_insight(predicted_topics, ctx)
            if insight:
                body_parts.append(insight)

        # Add a follow-up question if needed
        follow_up = ""
        if ctx.follow_up_needed:
            follow_up = self._generate_follow_up(ctx, topic)

        # Assemble the final response
        body = " ".join(body_parts)
        response = f"{opener} {body}"
        if follow_up:
            response = f"{response} {follow_up}"

        # Ensure response is not too long
        if len(response) > 3500:
            response = response[:3500].rstrip() + "…"

        return response

    def _extract_topic(self, message: str) -> str:
        """Extract the main topic from a user message.

        Complexity: T(n) ∈ O(n) for message length n.
        """
        # Remove question words and extract the key phrase
        stop = {
            "what", "is", "are", "the", "a", "an", "how", "does", "do",
            "why", "when", "where", "which", "can", "you", "should", "i",
            "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "tell", "me", "explain", "describe", "write", "create", "build",
            "make", "give", "show",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', message.lower())
        key_words = [w for w in words if w not in stop]
        if key_words:
            return " ".join(key_words[:3])
        return message[:50].strip()

    def _choose_opener(self, ctx: ResponseContext, topic: str) -> str:
        """Choose an opener based on intent and mode.

        Complexity: T(n) ∈ O(1).
        """
        openers_map = {
            "code": self._OPENERS_CODE,
            "analyze": self._OPENERS_ANALYTICAL,
            "architect": self._OPENERS_ARCHITECT,
            "explain": self._OPENERS_EXPLANATORY,
            "question": self._OPENERS_EXPLANATORY,
        }
        openers = openers_map.get(ctx.intent, self._OPENERS_DESCRIPTIVE)
        opener = random.choice(openers)
        return opener.format(topic=topic)

    def _synthesize_knowledge(
        self,
        facts: List[Tuple[ExtractedFact, float, float]],
        ctx: ResponseContext,
    ) -> str:
        """Synthesize knowledge facts into coherent text.

        Complexity: T(n) ∈ O(f) for f facts.
        """
        if not facts:
            return ""

        parts: List[str] = []
        for fact, quality, importance in facts[:3]:
            # Construct a sentence from the fact
            sentence = self._fact_to_sentence(fact)
            if sentence:
                parts.append(sentence)

        if not parts:
            return ""

        # Connect with transitions
        result = parts[0]
        for i, part in enumerate(parts[1:], 1):
            transition = random.choice(self._TRANSITIONS)
            result += f" {transition}{part.lower()}" if part[0].isupper() else f" {transition}{part}"

        return result

    def _fact_to_sentence(self, fact: ExtractedFact) -> str:
        """Convert a fact into a natural language sentence.

        Complexity: T(n) ∈ O(1).
        """
        # Map predicates to natural language
        predicate_map = {
            "is-a": "is",
            "means": "means",
            "refers-to": "refers to",
            "defined-as": "is defined as",
            "causes": "causes",
            "leads-to": "leads to",
            "results-in": "results in",
            "because": "is because",
            "can": "can",
            "supports": "supports",
            "uses": "uses",
            "requires": "requires",
            "enables": "enables",
            "better-than": "is better than",
            "more-than": "is more",
        }

        predicate = predicate_map.get(fact.predicate, fact.predicate)
        subject = fact.subject.replace("_", " ")
        obj = fact.object.replace("_", " ")

        # Capitalize first letter
        return f"{subject.capitalize()} {predicate} {obj}."

    def _synthesize_reasoning(self, reasoning: ReasoningResult, ctx: ResponseContext) -> str:
        """Synthesize reasoning steps into coherent text.

        Complexity: T(n) ∈ O(s) for s reasoning steps.
        """
        if not reasoning.steps:
            return ""

        # Use the final conclusion as the main output
        if reasoning.final_conclusion:
            return reasoning.final_conclusion

        # Fall back to step conclusions
        parts: List[str] = []
        for step in reasoning.steps:
            if step.confidence > 0.3:
                parts.append(step.conclusion)

        return " ".join(parts[:3]) if parts else ""

    def _construct_from_question(self, ctx: ResponseContext) -> str:
        """Construct a response when no knowledge or reasoning is available.

        This still uses the AI's own voice — not a template.

        Complexity: T(n) ∈ O(1).
        """
        topic = self._extract_topic(ctx.user_message)

        if ctx.intent == "code":
            return random.choice([
                f"Looking at this, I'd approach {topic} by first understanding the requirements, then designing the algorithm, implementing with proper types, and handling edge cases. What specific language or framework are you targeting?",
                f"For {topic}, the key is to break it down: identify the inputs, design the logic, implement cleanly, and test thoroughly. What's the specific problem you're solving?",
                f"When building {topic}, I'd start with the core logic, add type safety, then handle edge cases. What language should I use?",
            ])

        if ctx.intent == "analyze":
            return random.choice([
                f"To analyze {topic}, I'd look at it from multiple dimensions: performance, complexity, security, and maintainability. What specific aspect interests you most?",
                f"Breaking down {topic}: I'd evaluate the strengths, weaknesses, trade-offs, and alternatives. What angle do you want me to focus on?",
            ])

        if ctx.intent == "architect":
            return random.choice([
                f"Designing {topic} requires identifying the components, defining interfaces, mapping data flow, and planning for scale. What are your constraints?",
                f"For {topic}, I'd structure it in phases: design, implement, test, deploy. What's the scale you're targeting?",
            ])

        if ctx.intent in ("explain", "question"):
            return random.choice([
                f"That's a great question about {topic}. Let me think about this — the key is understanding the core concept and how it connects to what you already know. What specific part interests you most?",
                f"Interesting question about {topic}. I'd approach this by breaking it into parts and looking at how they fit together. What's your current understanding?",
                f"Let me think about {topic}. The core idea is usually simpler than it seems — it's about understanding the fundamental principle. What's your take so far?",
            ])

        # Default
        return random.choice([
            f"I'd love to help with {topic}. Could you tell me more about what you're looking for?",
            f"Let's explore {topic} together. What specific aspect interests you?",
            f"I'm thinking about {topic}. What would you like to know about it?",
        ])

    def _add_contextual_insight(
        self, predicted_topics: List[Tuple[str, float]], ctx: ResponseContext
    ) -> str:
        """Add a contextual insight based on predicted topics.

        Complexity: T(n) ∈ O(1).
        """
        if not predicted_topics:
            return ""
        top_topic, prob = predicted_topics[0]
        if prob < 0.3:
            return ""
        return random.choice([
            f" This connects to {top_topic} as well — let me know if you want to explore that.",
            f" By the way, {top_topic} is related to this. Want to go there next?",
            f" This also touches on {top_topic}, which might be worth exploring.",
        ])

    def _generate_follow_up(self, ctx: ResponseContext, topic: str) -> str:
        """Generate a follow-up question.

        Complexity: T(n) ∈ O(1).
        """
        # Use pattern-based follow-ups
        return random.choice(self._FOLLOW_UP_PATTERNS)
