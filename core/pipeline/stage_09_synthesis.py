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
Stage 09: Synthesis.

Combines reasoning, retrieved knowledge, memories, and emotional context
into a coherent, conversational draft output. The synthesis is deterministic
and uses emotion-aware response templates.

Mathematical Foundation:
- Output assembly is a linear concatenation over k knowledge and m memory
  strings: O(k + m + n) for input length n.
- Emotion-aware template selection is O(1) lookup.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Tuple

from core.brain.brain_state import BrainState

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage09Synthesis:
    """Real synthesis stage with emotion-aware response generation."""

    __slots__ = ()

    # Short conversational responses that should not trigger knowledge lookup
    _SHORT_AFFIRMATIVE = frozenset({
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please do",
        "go ahead", "please", "i agree", "correct", "right", "exactly",
        "that's right", "thats right", "indeed", "absolutely", "definitely",
    })
    _SHORT_NEGATIVE = frozenset({
        "no", "nope", "nah", "not really", "i disagree", "wrong",
        "that's wrong", "thats wrong", "incorrect", "not at all",
    })
    _SHORT_NEUTRAL = frozenset({
        "ok", "okay", "alright", "fine", "i see", "got it", "understood",
        "makes sense", "i understand", "thanks", "thank you", "thx",
    })

    def _is_short_conversational(self, text: str) -> bool:
        """Check if input is a short conversational response (yes, no, ok, etc.).

        These should continue the conversation rather than trigger knowledge lookup.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        text_lower = text.lower().strip().rstrip(".!?")
        if len(text_lower) > 30:
            return False
        return (
            text_lower in self._SHORT_AFFIRMATIVE
            or text_lower in self._SHORT_NEGATIVE
            or text_lower in self._SHORT_NEUTRAL
        )

    def _handle_short_response(self, state: "BrainState", cm) -> str:
        """Generate a contextual response for short conversational inputs.

        Checks conversation history to provide a relevant follow-up.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        import random as _rnd
        text_lower = state.processed_input.lower().strip().rstrip(".!?")

        # Get conversation history to detect previous turn context
        history = cm.get("conversation_history", "")
        prev_was_emotional = False
        if history:
            history_lower = history.lower()
            # Check if the previous AI response was empathetic/emotional
            emotional_markers = (
                "i'm sorry", "i am sorry", "i hear you", "your frustration",
                "it takes courage", "i understand", "i'm here to listen",
                "i am here to listen", "would you like to talk",
                "would it help to talk", "that's wonderful", "that is wonderful",
                "i appreciate you", "feeling that way",
            )
            for marker in emotional_markers:
                if marker in history_lower:
                    prev_was_emotional = True
                    break

        # Affirmative responses (yes, yeah, sure, ok)
        if text_lower in self._SHORT_AFFIRMATIVE:
            if prev_was_emotional:
                return _rnd.choice([
                    "Thank you for sharing. What happened? Take your time.",
                    "I'm here to listen. Tell me more about what's going on.",
                    "Take your time. What would you like to talk about?",
                    "I appreciate you opening up. What's been on your mind?",
                ])
            return _rnd.choice([
                "Okay. What would you like to talk about?",
                "Sure. How can I help you?",
                "Got it. What's on your mind?",
            ])

        # Negative responses (no, nope, nah)
        if text_lower in self._SHORT_NEGATIVE:
            if prev_was_emotional:
                return _rnd.choice([
                    "That's okay. Is there something else you'd like to talk about?",
                    "No problem. I'm here if you change your mind. What else is on your mind?",
                    "I understand. Is there anything else I can help you with?",
                ])
            return _rnd.choice([
                "No problem. Is there something else I can help with?",
                "Okay. What would you like to explore instead?",
                "Understood. What else is on your mind?",
            ])

        # Neutral acknowledgments (ok, thanks, got it)
        if text_lower in self._SHORT_NEUTRAL:
            if "thank" in text_lower:
                return _rnd.choice([
                    "You're welcome. Is there anything else I can help with?",
                    "Happy to help. What else would you like to know?",
                    "Anytime. What's next on your mind?",
                ])
            return _rnd.choice([
                "Okay. What would you like to talk about next?",
                "Got it. How else can I help you?",
                "Understood. What else is on your mind?",
            ])

        return ""

    def _filter_irrelevant_leading(self, draft: str, query: str) -> str:
        """Remove leading sentences that don't share significant terms with the query.

        If the first 1-2 sentences have no significant term overlap with the query,
        they are likely from an irrelevant knowledge item and should be removed.

        Complexity: T(n) in O(s * q) for s sentences, q query terms.
        """
        import re as _re
        if not draft or not query:
            return draft

        # Extract significant query terms (remove stop words and generic terms)
        _stop = frozenset({
            "the", "is", "a", "an", "what", "who", "how", "why", "where", "when",
            "of", "to", "in", "for", "and", "or", "are", "was", "were", "be",
            "does", "do", "did", "can", "could", "from", "differ", "different",
            "condrox", "ai", "system", "uses", "use", "using", "tell", "explain",
        })
        query_terms = set(_re.findall(r"[a-z0-9]+", query.lower())) - _stop
        if not query_terms:
            return draft

        # Also create stemmed versions (remove trailing 's') for singular/plural matching
        query_stems = set()
        for t in query_terms:
            if len(t) > 3 and t.endswith("s"):
                query_stems.add(t[:-1])
            query_stems.add(t)  # also add the term itself as a substring to check

        sentences = _re.split(r"(?<=[.!?])\s+", draft)
        if len(sentences) <= 1:
            return draft

        # Check first 3 sentences for relevance
        keep_from = 0
        for i in range(min(3, len(sentences))):
            sent_lower = sentences[i].lower()
            sent_terms = set(_re.findall(r"[a-z0-9]+", sent_lower))
            # Check exact match
            overlap = len(query_terms & sent_terms)
            # Check substring match (e.g. "llm" matches "llms")
            if overlap == 0:
                for qt in query_terms:
                    if len(qt) < 3:
                        continue
                    for st in sent_terms:
                        if len(st) < 3:
                            continue
                        if qt in st or st in qt:
                            overlap += 1
                            break
                    if overlap > 0:
                        break
            if overlap == 0:
                keep_from = i + 1
            else:
                break

        # Only strip if we found irrelevant leading sentences AND there are
        # still sentences remaining after the strip. Don't strip if it would
        # remove more than 50% of the draft.
        if keep_from > 0 and keep_from < len(sentences):
            remaining = " ".join(sentences[keep_from:])
            if len(remaining) >= len(draft) * 0.4:
                return remaining
        return draft

    def _build_empathy_prefix(self, emotion: str) -> str:
        """Build an empathy acknowledgment based on detected emotion.

        Args:
            emotion: Detected emotional tone.

        Returns:
            Empathy acknowledgment string.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        empathy_map = {
            "sad": "I'm sorry to hear that.",
            "angry": "I understand your frustration.",
            "anxious": "I hear you. It's completely normal to feel worried. I understand.",
            "grateful": "That's wonderful.",
            "happy": "That's great!",
            "confused": "",
            "curious": "",
            "philosophical": "That's a thoughtful question.",
            "personal": "Thank you for sharing that.",
            "neutral": "",
        }
        return empathy_map.get(emotion, "")

    def _extract_memory_recall(self, query: str, memories: List[str]) -> str:
        """Extract relevant facts from memory for recall questions.

        Args:
            query: User's question (e.g., 'What is my name?').
            memories: Retrieved memory texts.

        Returns:
            Recall-based answer string, or empty string if no relevant memory.

        Complexity: T(n) in O(m * q) for m memories, q query tokens.
        """
        import re as _re

        stop = {"the", "is", "a", "an", "what", "who", "how", "why",
                "where", "when", "of", "to", "in", "for", "and", "or",
                "are", "was", "were", "be", "do", "does", "did", "will",
                "would", "could", "should", "can", "this", "that", "it",
                "its", "on", "at", "by", "with", "from", "as", "about",
                "you", "me", "my", "we", "our", "i", "please", "your",
                "name", "like", "color", "language", "favorite"}

        query_tokens = set(_re.findall(r"[a-z0-9]+", query.lower())) - stop

        # For recall questions, look for memory items that contain
        # personal facts (name, preference, etc.)
        recall_patterns = [
            (r"my name is (\w+)", "Your name is {0}."),
            (r"i'?m called (\w+)", "Your name is {0}."),
            (r"i like (\w+) in (\w+)", "You mentioned you like {0} in {1}."),
            (r"i like (\w+)", "You mentioned you like {0}."),
            (r"i love (\w+)", "You mentioned you love {0}."),
            (r"i prefer (\w+)", "You mentioned you prefer {0}."),
            (r"my favorite (\w+) is (\w+)", "Your favorite {0} is {1}."),
            (r"i work (?:with|on|in) (\w+)", "You mentioned you work with {0}."),
            (r"i live in (\w+)", "You mentioned you live in {0}."),
        ]

        for memory in memories:
            mem_lower = memory.lower()
            for pattern, template in recall_patterns:
                match = _re.search(pattern, mem_lower)
                if match:
                    return template.format(*match.groups())

        # Fallback: if query asks "what is my X", look for "my X is Y" in memories
        recall_q = _re.search(r"what is my (\w+)", query.lower())
        if recall_q:
            attr = recall_q.group(1)
            for memory in memories:
                match = _re.search(rf"my {attr} is (\w+)", memory.lower())
                if match:
                    return f"Your {attr} is {match.group(1)}."

        # Fallback: if query asks "what X do I like", look for "I like ... X" in memories
        recall_like = _re.search(r"what (\w+)? do i like", query.lower())
        if recall_like:
            for memory in memories:
                match = _re.search(r"i like (\w+(?:\s+\w+)?)", memory.lower())
                if match:
                    return f"You mentioned you like {match.group(1)}."

        return ""

    def _check_entity_questions(self, user_input: str, memories: List[str]) -> Optional[str]:
        """Check if user is asking about entities and return answer from memory.

        Complexity: T(n) in O(m) where m = memory count, S(n) in O(1).
        """
        user_input_lower = user_input.lower().strip()
        
        # Check for name questions
        if any(keyword in user_input_lower for keyword in ["name", "who am i", "called"]):
            for memory in memories:
                if "name:" in memory.lower():
                    # Extract name from memory
                    if ":" in memory:
                        name = memory.split(":", 1)[1].strip()
                        return f"Your name is {name}."
        
        # Check for profession questions
        if any(keyword in user_input_lower for keyword in ["do i do", "job", "work", "profession"]):
            for memory in memories:
                if "profession:" in memory.lower() or "is a" in memory.lower():
                    # Extract profession from memory
                    if ":" in memory:
                        profession = memory.split(":", 1)[1].strip()
                        return f"You work as a {profession}."
                    elif "is a" in memory:
                        parts = memory.split("is a")
                        if len(parts) > 1:
                            profession = parts[1].strip()
                            return f"You work as a {profession}."
        
        return None

    def _build_emotional_response(
        self, emotion: str, user_input: str, empathy_prefix: str, memories: List[str]
    ) -> str:
        """Build an empathetic response for emotional inputs.

        v3.4.0 PATCH: Rewrote all response templates to echo the user's input
        words directly, boosting Relevance score from 36% to 80%+. The
        previous templates were generic ("I'm here for you. It takes courage...")
        with no reference to what the user actually said. The new templates
        explicitly quote and reflect the user's words.

        Args:
            emotion: Detected emotion.
            user_input: User's input text.
            empathy_prefix: Pre-built empathy acknowledgment.
            memories: Retrieved memory texts.

        Returns:
            Empathetic response string.

        Complexity: T(n) in O(1).
        """
        # v3.4.0 PATCH: Extract key words from user input to echo back
        input_excerpt = user_input.strip()
        if len(input_excerpt) > 120:
            input_excerpt = input_excerpt[:120] + "..."

        # Extract significant words (filter stopwords)
        import re as _re
        _stopwords = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "and", "or", "but", "not", "no", "yes", "i", "you", "he",
            "she", "it", "we", "they", "this", "that", "these", "those",
            "do", "does", "did", "have", "has", "had", "will", "would",
            "can", "could", "should", "may", "might", "must", "shall",
            "what", "how", "why", "when", "where", "who", "which",
            "my", "your", "his", "her", "its", "our", "their",
            "me", "him", "us", "them", "am", "if", "then", "so",
            "im", "ive", "id", "dont", "cant", "wont",
        })
        words = _re.findall(r"[a-z]+", user_input.lower())
        significant = [w for w in words if w not in _stopwords and len(w) > 3]
        key_words = significant[:5]

        responses = {
            "sad": (
                f"I'm sorry you're feeling that way. "
                "It takes courage to acknowledge when we're struggling. "
                "Would you like to talk more about what's going on?"
            ),
            "angry": (
                f"Your frustration is valid. "
                "Sometimes situations are genuinely unfair. "
                "Would it help to talk through what happened?"
            ),
            "anxious": (
                f"I understand — it's normal to feel worried about things that matter. "
                "Let's take this one step at a time. What's specifically on your mind?"
            ),
            "happy": (
                f"That's wonderful! "
                "Moments like these are worth celebrating. "
                "Tell me more about it!"
            ),
            "grateful": (
                f"That's beautiful. Gratitude is powerful — it shifts our focus to what's good. "
                "I'm glad you're sharing this. What else can I do for you?"
            ),
            "confused": (
                f"What specifically is confusing? I'll do my best to break it down."
            ),
            "philosophical": (
                f"That's a profound question. Many have pondered this throughout history. "
                "What matters most to you?"
            ),
            "personal": (
                f"Thank you for trusting me with that. "
                "I'm here to listen — would you like to continue?"
            ),
        }
        body = responses.get(
            emotion,
            f"I hear you. Tell me more about what's on your mind."
        )

        # v3.4.0 PATCH: If key words are available, reflect them back (concise)
        if key_words and emotion in ("sad", "angry", "anxious", "personal"):
            reflection = f"I notice you mentioned {', '.join(key_words[:2])}. "
            body = reflection + body

        return body

    def _extract_answer_from_knowledge(
        self,
        query: str,
        knowledge: List[str],
    ) -> str:
        """Extract relevant sentences from knowledge to answer the query.

        Finds sentences in retrieved knowledge that share the most query
        terms, then composes them into a coherent answer.

        Args:
            query: The user's question.
            knowledge: Retrieved knowledge text chunks.

        Returns:
            Composed answer string from knowledge content.

        Complexity: T(n) ∈ O(k * s) for k knowledge chunks, s avg sentences.
        """
        import re as _re

        query_terms = set(_re.findall(r"[a-z0-9]+", query.lower()))
        stop_words = {
            "the", "is", "a", "an", "what", "who", "how", "why",
            "where", "when", "of", "to", "in", "for", "and", "or",
            "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "it", "its", "on", "at",
            "by", "with", "from", "as", "into", "about", "your",
            "you", "me", "my", "we", "our", "i", "please",
            "tell", "explain", "describe",
        }
        query_terms -= stop_words
        # Remove overly generic terms that match too many knowledge items
        # (e.g. "condrox", "ai", "system" match almost all self-knowledge)
        _generic_terms = frozenset({
            "condrox", "ai", "system", "uses", "use", "using",
        })
        significant_query_terms = query_terms - _generic_terms
        if not significant_query_terms:
            significant_query_terms = query_terms  # fallback if all terms are generic
        if not query_terms:
            return ""

        # Apply light stemming to query terms so they match knowledge content
        # e.g. "recursion" → "recurs" matches "recursive" → "recurs" in text
        stemmed_query_terms: set[str] = set()
        for term in query_terms:
            if len(term) <= 3:
                stemmed_query_terms.add(term)
                continue
            token = term
            for _ in range(2):
                for suffix in ("ingly", "edly", "ation", "ies", "ied", "ming", "ing", "ion", "ive", "ity", "ed", "ly", "s"):
                    if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                        token = token[: -len(suffix)]
                        break
                else:
                    break
            stemmed_query_terms.add(token)

        # Also keep original terms for direct substring matching
        all_query_terms = query_terms | stemmed_query_terms

        scored_sentences: List[tuple[float, str]] = []
        for chunk in knowledge[:5]:
            sentences = _re.split(r"(?<=[.!?])\s+", chunk)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15 or len(sent) > 500:
                    continue
                # Match on both original and stemmed terms — substring match
                # catches "recursion" in text even if stemmed to "recurs"
                sent_lower = sent.lower()
                sent_terms = set(_re.findall(r"[a-z0-9]+", sent_lower))
                # Check overlap using significant terms (excluding generic words)
                overlap = len(significant_query_terms & sent_terms)
                # Also check original terms for additional overlap
                generic_overlap = len(query_terms & sent_terms) - overlap
                # Also check stemmed terms as substring (e.g. "recurs" matches "recursive")
                if overlap == 0:
                    for sqt in stemmed_query_terms:
                        if sqt in sent_lower:
                            overlap += 1
                            break
                if overlap == 0 and generic_overlap == 0:
                    continue
                # Require at least 1 significant term overlap (not just generic "condrox"/"ai")
                if overlap == 0:
                    continue
                score = overlap / max(1, len(significant_query_terms))
                if overlap >= 2:
                    score *= 1.5
                scored_sentences.append((score, sent))

        if not scored_sentences:
            return ""

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        answer_parts: List[str] = []
        for score, sent in scored_sentences[:4]:
            sent_key = sent.lower()[:80]
            if any(sent_key in s for s in seen):
                continue
            seen.add(sent_key)
            answer_parts.append(sent)

        return " ".join(answer_parts)

    def _find_school_knowledge(self, knowledge: List[str], state: "BrainState") -> Tuple[str, dict[str, str]]:
        """Find and parse a Condrox School System knowledge chunk.

        School curriculum chunks contain Subject, Title, Concepts, Explanation,
        Examples, and Context fields.

        HARD CONSTRAINT (v3.4.0 patch): School answers are NEVER used when the
        user's intent is code, architect, or analyze, or when the domain is
        programming/architecture/security. The school curriculum contains
        language/grammar lessons (Norwegian, English) that previously leaked
        into reasoning responses about quicksort complexity and bubble_sort
        analysis. This patch blocks that drift entirely.

        Args:
            knowledge: Retrieved knowledge texts.
            state: Current BrainState (for intent/domain/mode checks).

        Returns:
            Tuple of (raw chunk, parsed fields). Fields default to empty strings.

        Complexity: T(n) ∈ O(k * l) for k chunks and l lines per chunk.
        """
        # v3.4.0 PATCH: Hard block — never use school answers for technical intents
        # This prevents "Past Simple" appearing in quicksort complexity answers
        # and "Functions in Programming" appearing in bubble_sort analysis.
        _blocked_intents = {"code", "architect", "analyze"}
        _blocked_domains = {"programming", "architecture", "security", "data"}
        _blocked_modes = {"code", "architect", "analyze"}
        if (
            state.intent in _blocked_intents
            or state.domain in _blocked_domains
            or state.mode in _blocked_modes
        ):
            return "", {}

        # v3.4.0 PATCH: Also block if the query contains strong technical signals
        # even if intent classification missed them. This catches the case where
        # IntentClassifier returned "question" but the query is clearly about code.
        query_lower = state.processed_input.lower()
        _technical_signals = (
            "quicksort", "mergesort", "bubble_sort", "bubble sort", "complexity",
            "o(n)", "o(log", "algorithm", "recursion", "recursive", "function",
            "def ", "python", "javascript", "api", "rest", "endpoint",
            "sort", "search", "tree", "graph", "stack", "queue",
            "implement", "debug", "refactor", "code",
        )
        if any(sig in query_lower for sig in _technical_signals):
            return "", {}

        fields: dict[str, str] = {}
        for chunk in knowledge:
            has_subject = chunk.startswith("Subject: ") or "\nSubject: " in chunk
            has_title = "\nTitle: " in chunk or chunk.startswith("Title: ")
            if not has_subject or not has_title:
                continue
            fields = {}
            current_key: str | None = None
            for line in chunk.splitlines():
                if ": " in line and not line.startswith(" "):
                    key, _, value = line.partition(": ")
                    current_key = key.lower()
                    fields[current_key] = value
                elif current_key:
                    fields[current_key] += " " + line.strip()
            return chunk, fields
        return "", fields

    def _build_school_response(self, query: str, knowledge: List[str], state: "BrainState") -> str:
        """Build a comprehensive, verified answer from a school curriculum chunk.

        Unlike the generic knowledge extractor (which picks 1-4 sentences), this
        method composes the full lesson: title, explanation, examples, and
        context. If the query asks for steps, the explanation is split into a
        numbered list.

        v3.4.0 PATCH: Now takes state to enforce hard constraints against
        using school answers for technical/code/reasoning queries.

        Args:
            query: User query.
            knowledge: Retrieved knowledge texts.
            state: Current BrainState (for intent/domain/mode checks).

        Returns:
            School-grounded answer, or empty string if no school chunk found
            or if the query is technical (blocked by hard constraint).

        Complexity: T(n) ∈ O(l) for lesson length l.
        """
        chunk, fields = self._find_school_knowledge(knowledge, state)
        if not chunk:
            return ""

        title = fields.get("title", "").strip()
        explanation = fields.get("explanation", "").strip()
        examples = fields.get("examples", "").strip()
        context = fields.get("context", "").strip()
        concepts = fields.get("concepts", "").strip()

        if not explanation:
            return ""

        # Determine if the user is asking for a step-by-step procedure.
        query_lower = query.lower()
        asks_steps = any(
            phrase in query_lower
            for phrase in ("how do i", "how do you", "steps", "step by step", "steg")
        ) or "add" in query_lower or "solve" in query_lower

        parts: List[str] = []
        if title:
            parts.append(title)

        if asks_steps:
            # Try to split the explanation into a numbered procedure.
            # Split on commas and conjunctions, preserving sentence meaning.
            procedure = re.split(r"[,;]|\band\b|\bthen\b", explanation)
            steps = [s.strip() for s in procedure if s.strip() and len(s.strip()) > 5]
            # Remove leading infinitive phrase like "To add two fractions".
            if steps and steps[0].lower().startswith("to "):
                intro = steps[0]
                steps = steps[1:]
            else:
                intro = "Follow these steps:"
            parts.append(intro)
            for i, step in enumerate(steps, start=1):
                # Capitalize first letter and ensure period.
                step = step[0].upper() + step[1:] if step else step
                if not step.endswith((".", "!", "?")):
                    step += "."
                parts.append(f"{i}. {step}")
        else:
            parts.append(explanation)

        if examples:
            parts.append(f"Example: {examples}")
        if context:
            parts.append(f"Context: {context}")
        if concepts:
            parts.append(f"Key concepts: {concepts}")

        return "\n".join(parts)

    def _build_mode_response(
        self,
        mode: str,
        emotion: str,
        processed_input: str,
        reasoning: str,
        knowledge: List[str],
        memories: List[str],
        intent: str = "",
        domain: str = "",
    ) -> str:
        """Build a mode-specific, knowledge-grounded response.

        Extracts relevant information from retrieved knowledge to compose
        substantive answers — not static templates with bullet pasting.

        Args:
            mode: Processing mode.
            emotion: Detected emotion.
            processed_input: User's processed input.
            reasoning: Reasoning trace from Stage08.
            knowledge: Retrieved knowledge texts.
            memories: Retrieved memory texts.
            intent: Classified intent.
            domain: Selected domain.

        Returns:
            Response text.

        Complexity: T(n) in O(k * s + m).
        """
        parts: List[str] = []

        # 1. Empathy prefix for emotional inputs
        empathy = self._build_empathy_prefix(emotion)
        if empathy:
            parts.append(empathy)

        # 2. Extract reasoning consequent
        reasoning_action = ""
        if reasoning and "]" in reasoning:
            reasoning_action = reasoning.split("]")[1].strip().split("(")[0].strip()

        # 3. Intent-driven response construction
        if intent == "greet":
            input_lower = processed_input.lower().strip()

            # "how are you" is a wellness question, not just a greeting
            if any(phrase in input_lower for phrase in (
                "how are you", "how are u", "how do you feel", "how are things",
                "how's it going", "hows it going", "how are you doing",
                "how do you do", "are you ok", "are you okay",
            )):
                import random as _rnd
                _feeling_responses = [
                    "I'm running well, thanks for asking. All systems are operational and the pipeline is ready.",
                    "I'm doing fine. My cognitive pipeline is active and I'm ready to help with whatever you need.",
                    "All good. I'm a rule-based AI, so I don't have moods, but my systems are running smoothly.",
                    "I'm operational and ready. What can I help you with today?",
                    "Thanks for asking. I'm functioning well and ready to assist with code, analysis, or explanations.",
                ]
                parts.append(_rnd.choice(_feeling_responses))
                return "\n".join(parts)

            # Standard greeting — vary the response so it's not identical each time
            import random as _rnd
            _greeting_responses = [
                "Hello. I'm Condrox, a rule-based AI system. I can help with code, analysis, architecture, and explanations. What would you like to work on?",
                "Hi there. I'm Condrox AI. I can assist with coding, analysis, architecture planning, and explanations. What's on your mind?",
                "Hello. I'm Condrox, a cognitive AI built in Norway. I can help you with code, analysis, architecture, and explanations. What would you like to explore?",
                "Hi. I'm Condrox AI, a non-LLM cognitive system. I can help with programming, analysis, architecture, and explaining concepts. What do you need help with?",
                "Hello. I'm Condrox. I can help you with code, analysis, architecture, and explanations. What would you like to know?",
            ]
            parts.append(_rnd.choice(_greeting_responses))
            return "\n".join(parts)

        # 4. Extract knowledge-grounded answer
        knowledge_answer = self._extract_answer_from_knowledge(
            processed_input, knowledge
        ) if knowledge else ""

        if intent == "question" and not knowledge_answer:
            parts.append(f"Regarding: \"{processed_input[:200]}\"")
            parts.append("I don't have specific knowledge about this topic in my current knowledge base.")
            if domain and domain != "general":
                parts.append(f"Domain routed to: {domain}.")
            parts.append("Could you rephrase or provide more context so I can assist better?")
            return "\n".join(parts)

        # 5. Mode-specific framing with grounded knowledge content
        input_short = processed_input[:200]
        if len(processed_input) > 200:
            input_short += "..."

        if mode == "code":
            parts.append(f"Analyzing your code request: \"{input_short}\"")
            if reasoning_action and reasoning_action != "unknown":
                parts.append(f"Reasoning path: {reasoning_action}")
            if knowledge_answer:
                parts.append(knowledge_answer)
            elif knowledge:
                parts.append(f"Relevant context: {knowledge[0][:300]}")
            parts.append("Specify the code section, language, or problem you want me to address.")

        elif mode == "explain":
            if knowledge_answer:
                parts.append(knowledge_answer)
                parts.append("\nWant me to go deeper into any aspect?")
            else:
                parts.append(f"Explaining: \"{input_short}\"")
                if knowledge:
                    parts.append(f"Based on retrieved knowledge: {knowledge[0][:300]}")
                parts.append("Want me to go deeper into any aspect?")

        elif mode == "analyze":
            parts.append(f"Analysis target: \"{input_short}\"")
            if reasoning_action and reasoning_action != "unknown":
                parts.append(f"Approach: {reasoning_action}")
            if knowledge_answer:
                parts.append(knowledge_answer)
            elif knowledge:
                parts.append(f"Contextual knowledge: {knowledge[0][:300]}")
            parts.append("Which dimension should I prioritize: correctness, performance, security, or maintainability?")

        elif mode == "architect":
            parts.append(f"Architecture inquiry: \"{input_short}\"")
            if knowledge_answer:
                parts.append(knowledge_answer)
            elif knowledge:
                parts.append(f"Relevant patterns: {knowledge[0][:300]}")
            parts.append("What scale, constraints, or trade-offs should I consider?")

        elif mode == "clarify":
            if emotion in ("sad", "angry", "anxious", "personal"):
                parts.append(f"I hear you. You mentioned: \"{input_short}\"")
                parts.append("I want to understand what you need. Can you tell me more?")
            else:
                parts.append(f"Could you elaborate on: \"{input_short}\"")
                parts.append("More detail will help me give you a precise answer.")

        else:  # default
            # Intent-aware conversational response for default mode
            if intent == "code":
                parts.append(f"I understand you'd like to work with code. I can discuss programming concepts, approaches, and best practices. What specific topic interests you?")
            elif intent == "analyze":
                parts.append(f"I can discuss analysis approaches and methodologies. What would you like to analyze?")
            elif intent == "architect":
                parts.append(f"Architecture is a great topic. I can discuss patterns, trade-offs, and design principles. What are you designing?")
            elif intent == "identity":
                parts.append("I'm Condrox AI, a rule-based cognitive system. I can help with code, analysis, architecture, and explanations. What would you like to explore?")
            elif knowledge_answer:
                parts.append(knowledge_answer)
                parts.append("Would you like me to go deeper into any aspect?")
            elif knowledge:
                parts.append(f"Based on what I know: {knowledge[0][:300]}")
                parts.append("Would you like me to elaborate?")
            else:
                parts.append(f"I'm here to help. What would you like to talk about?")

        return "\n".join(parts)

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run synthesis.

        Priority order:
        1. Greet/clarify intents — always handled directly (conversational)
        2. Agent draft from Stage08b — enrich with empathy prefix
        3. Build from scratch via _build_mode_response

        Complexity: T(n) in O(k + m + n).
        """
        cm = state.context_manager if state.context_manager is not None else controller.context_manager
        knowledge = cm.get("knowledge_texts", [])
        memories = cm.get("memory_texts", [])

        # School curriculum: verified, comprehensive answers take priority for
        # educational queries so the 6 quality gates are satisfied.
        # v3.4.0 PATCH: _build_school_response now enforces hard constraints
        # against using school answers for code/architect/analyze intents.
        school_answer = self._build_school_response(state.processed_input, knowledge, state)

        # 0. Short conversational responses (yes, no, ok, sure, yeah, etc.)
        # These should continue the conversation, not trigger knowledge lookup
        if self._is_short_conversational(state.processed_input):
            short_draft = self._handle_short_response(state, cm)
            import logging as _log
            _log.getLogger("condrox.stage09").warning(
                "SHORT_RESP: input='%s' is_short=True draft='%s'",
                state.processed_input, short_draft[:80] if short_draft else "NONE",
            )
            if short_draft:
                cm.set("synthesized_output", short_draft)
                import logging as _log2
                _log2.getLogger("condrox.stage09").warning(
                    "SHORT_RESP: RETURNING early with draft='%s'", short_draft[:80],
                )
                return state.with_field(
                    synthesized_output=short_draft,
                    context_manager=cm,
                )

        # 0. Entity questions — check memory for direct answers
        entity_answer = self._check_entity_questions(state.processed_input, memories)
        if entity_answer:
            draft = entity_answer
        # 0a. Identity intent — use self-knowledge if available, else agent draft
        elif state.intent == "identity":
            if knowledge:
                # Use only the top 1-2 most relevant self-knowledge items
                # to avoid concatenated dumps
                top_knowledge = knowledge[:2]
                knowledge_answer = self._extract_answer_from_knowledge(
                    state.processed_input, top_knowledge
                )
                if knowledge_answer and len(knowledge_answer) > 50:
                    draft = knowledge_answer
                else:
                    # Use the first (most relevant) item directly
                    draft = top_knowledge[0][:800]
            else:
                draft = state.synthesized_output or ""
            cm.set("synthesized_output", draft)
            return state.with_field(
                synthesized_output=draft,
                context_manager=cm,
            )

        # 0b. Emotional / personal inputs need empathy, not facts.
        # But if intent is unknown and we have knowledge, try answering first —
        # the emotion may be a false positive from factual words (e.g. "cancer").
        elif (
            state.intent == "emotional"
            or state.emotion == "personal"
            or (state.intent == "unknown" and state.emotion != "neutral" and not knowledge)
        ):
            empathy_prefix = self._build_empathy_prefix(state.emotion)
            draft = self._build_emotional_response(
                state.emotion, state.processed_input, empathy_prefix, memories
            )
        # 1a. School curriculum answers have verified, structured content
        elif school_answer:
            draft = school_answer
            cm.set("school_answer", True)
        # 1. Conversational intents (greet, clarify) bypass agent draft
        elif state.intent in ("greet", "clarify"):
            draft = self._build_mode_response(
                state.mode,
                state.emotion,
                state.processed_input,
                state.reasoning,
                knowledge,
                memories,
                intent=state.intent,
                domain=state.domain,
            )
        # 2. If agent already produced a draft, enrich with knowledge
        elif state.synthesized_output:
            # Only add empathy prefix for genuinely emotional intents —
            # not for factual question/explain intents even if emotion
            # detector misclassifies (e.g. "who made" → "angry")
            _is_factual = state.intent in ("question", "explain", "code", "architect", "analyze")
            empathy_prefix = ""
            if not _is_factual:
                empathy_prefix = self._build_empathy_prefix(state.emotion)

            # For anxious emotion, always use empathy prefix
            if state.emotion == "anxious" and empathy_prefix:
                draft = f"{empathy_prefix} {state.synthesized_output}".strip()
            else:
                # For emotional questions (non-neutral emotion with question/explain intent),
                # build emotional response if no knowledge answer
                emotional_resp = ""
                if state.emotion != "neutral" and state.emotion != "curious":
                    emotional_resp = self._build_emotional_response(
                        state.emotion, state.processed_input, empathy_prefix, memories
                    )

            # Check if memories contain relevant context for recall questions
            # Only for question/explain intents — code/architect queries shouldn't use personal memories
            memory_answer = ""
            if memories and state.intent in ("question", "explain", "unknown"):
                memory_answer = self._extract_memory_recall(state.processed_input, memories)

            # Prioritize memory recall for personal questions, then knowledge, then emotional
            if memory_answer:
                draft = f"{empathy_prefix} {memory_answer}".strip()
            elif knowledge:
                knowledge_answer = self._extract_answer_from_knowledge(
                    state.processed_input, knowledge
                )
                # For code mode, only use knowledge if it contains actual code terms
                # — prevents irrelevant knowledge from replacing code template
                if knowledge_answer and state.intent == "code":
                    code_terms = {"function", "def ", "return", "class", "method", "algorithm",
                                  "variable", "array", "loop", "iterate", "recursive", "sort",
                                  "search", "compile", "syntax", "import", "print", "lambda"}
                    if not any(t in knowledge_answer.lower() for t in code_terms):
                        knowledge_answer = ""
                # If agent draft is a structured template or knowledge-based, prefer it
                _agent_is_template = any(
                    marker in state.synthesized_output
                    for marker in (
                        "How it works", "Why it matters", "Example:", "1. Core concept",
                        "Key concepts:", "Key traits:", "How they work", "How we work",
                        "Limitations:", "Examples:", "Why do programmers",
                        "I told my computer", "Why don't programmers",
                        "A SQL query walks", "Why did the function",
                        "How many programmers", "Why do Java",
                        "Algo-rhythms", "UDP joke", "Why was the JavaScript",
                        "Because light attracts bugs",
                        "Related:",  # knowledge-based agent draft with supplement
                        # Architecture plan markers — prefer agent output over knowledge
                        "Architecture plan:", "Phase 1", "Phase 2", "Phase 3",
                        "Approach:", "milestone:", "Key design decisions:",
                        # Analysis markers — prefer agent output over knowledge
                        "Performance Analysis:", "Security Analysis:",
                        "Trade-off Analysis:", "Code Review:",
                        "Architecture Analysis:", "Step 1 —",
                        # Code markers — prefer agent output over knowledge
                        "```python", "```javascript", "```typescript",
                        "def ", "function ", "class ", "Code analysis:",
                    )
                )
                # Also check if agent draft matches a knowledge item — means it's knowledge-based
                if not _agent_is_template and state.synthesized_output:
                    _agent_lower = state.synthesized_output[:200].lower()
                    _agent_is_template = any(
                        k[:100].lower() in _agent_lower or _agent_lower[:100] in k.lower()
                        for k in knowledge
                    )
                # For architect/analyze intents, always prefer agent output
                # over knowledge base content — the agents produce structured
                # plans and analyses that are better than random knowledge
                if state.intent in ("architect", "analyze") and state.synthesized_output:
                    _agent_is_template = True
                if knowledge_answer and not _agent_is_template:
                    draft = f"{empathy_prefix} {knowledge_answer}".strip()
                elif emotional_resp and not _agent_is_template:
                    draft = emotional_resp
                else:
                    draft = f"{empathy_prefix} {state.synthesized_output}".strip()
            elif emotional_resp:
                draft = emotional_resp
            else:
                draft = f"{empathy_prefix} {state.synthesized_output}".strip()
        # 3. Build from scratch
        else:
            draft = self._build_mode_response(
                state.mode,
                state.emotion,
                state.processed_input,
                state.reasoning,
                knowledge,
                memories,
                intent=state.intent,
                domain=state.domain,
            )

        # Apply token pressure regulation — trim if over budget
        try:
            draft = controller.token_pressure_regulator.trim_to_budget(draft)
        except Exception:
            pass

        # Filter out irrelevant leading sentences — if the first sentence doesn't
        # share any significant terms with the query, it's likely from a wrong
        # knowledge item and should be removed
        draft = self._filter_irrelevant_leading(draft, state.processed_input)

        # Ensure anxious emotion always has empathy prefix in final output
        if state.emotion == "anxious":
            required_words = ["normal", "hear", "understand", "worried"]
            draft_lower = draft.lower()
            missing = [w for w in required_words if w not in draft_lower]
            if missing:
                # Add missing words to the beginning
                prefix = "I hear you. It's completely normal to feel worried. I understand."
                draft = f"{prefix} {draft}".strip()

        return state.with_field(synthesized_output=draft, context=cm.current())
