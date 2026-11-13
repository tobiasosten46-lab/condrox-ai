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
Metacognition Engine — self-reflection, self-judgment, self-revision.

Gives the AI the ability to:
1. REFLECT on its own response — did it actually answer the question?
2. JUDGE its response quality across multiple cognitive dimensions.
3. REVISE if the judgment is poor — re-ground in knowledge, fix gaps.

This is the "inner critic" that makes responses more realistic and
self-aware. Instead of blindly outputting whatever the synthesis stage
produced, the AI evaluates its own thinking and corrects itself.

Cognitive dimensions evaluated:
- Relevance: Does the response address the user's actual question?
- Grounding: Is the response grounded in retrieved knowledge/memory?
- Coherence: Does the response follow a logical structure?
- Completeness: Does the response cover the key aspects of the topic?
- Specificity: Is the response specific or generic/templated?
- Self-awareness: Does the response show understanding of its own nature?

Complexity:
    evaluate: T(n) ∈ O(n + k) for response length n, knowledge items k.
    revise: T(n) ∈ O(n + k) for response length n, knowledge items k.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.brain.brain_state import BrainState


@dataclass(frozen=True, slots=True)
class CognitiveAssessment:
    """
    Immutable self-assessment of a response.

    Attributes:
        relevance: How well the response addresses the user's question [0, 1].
        grounding: How well grounded in knowledge/memory [0, 1].
        coherence: Logical structure and flow [0, 1].
        completeness: Coverage of key aspects [0, 1].
        specificity: Specificity vs generic/templated [0, 1].
        self_awareness: Understanding of own nature [0, 1].
        overall: Weighted overall score [0, 1].
        issues: List of identified issues.
        should_revise: Whether the response needs revision.

    Complexity: S(n) ∈ O(1) + O(i) for i issues.
    """

    relevance: float
    grounding: float
    coherence: float
    completeness: float
    specificity: float
    self_awareness: float
    overall: float
    issues: Tuple[str, ...] = field(default_factory=tuple)
    should_revise: bool = False


class MetacognitionEngine:
    """
    Self-reflection and self-judgment engine.

    Evaluates the AI's own response across cognitive dimensions and
    decides whether to revise. This is the metacognitive layer that
    makes the AI "think about its thinking."

    Complexity:
        evaluate: T(n) ∈ O(n + k), S(n) ∈ O(1).
        revise: T(n) ∈ O(n + k), S(n) ∈ O(n).
    """

    # Templates/patterns that indicate generic, non-specific responses
    _GENERIC_PATTERNS: Tuple[str, ...] = (
        "the concept of",
        "is important because it appears in many contexts",
        "let's break it down",
        "consider the definition",
        "look at how it's applied in practice",
        "theory and practice reinforce each other",
        "consider a real-world scenario",
        "examples make abstract concepts concrete",
        "always look for examples when learning",
        "if you understand the core concept, the rest follows",
        "since practice reinforces theory",
        "hence, you now have a solid foundation",
        "what specifically is confusing",
        "i don't have specific knowledge",
        "could you rephrase or provide more context",
        "specify the code section",
        "which dimension should i prioritize",
        "what scale, constraints, or trade-offs",
        "how can i assist you further",
        "want me to go deeper",
        "processing:",
        "analyzing your code request:",
        "explaining:",
        "analysis target:",
        "architecture inquiry:",
    )

    # Self-awareness indicators — phrases that show the AI knows what it is
    _SELF_AWARE_PATTERNS: Tuple[str, ...] = (
        "i am condrox",
        "i'm condrox",
        "i am a rule-based",
        "i'm a rule-based",
        "i am a cognitive",
        "i am a hybrid",
        "i don't use llm",
        "i use a pipeline",
        "my pipeline",
        "my knowledge",
        "my architecture",
        "i was built",
        "i was created",
        "i was developed",
        "tobias østen",
        "norway",
        "i process input through",
        "my reasoning",
        "i can help you with",
    )

    # Coherence markers — structural elements that indicate logical flow
    _COHERENCE_MARKERS: Tuple[str, ...] = (
        "first", "second", "third", "finally",
        "because", "therefore", "hence", "thus",
        "however", "but", "although", "while",
        "for example", "for instance", "in practice",
        "in summary", "in conclusion", "to summarize",
        "step 1", "step 2", "step 3",
        "1.", "2.", "3.", "4.",
    )

    __slots__ = ("_revision_count",)

    def __init__(self) -> None:
        """Initialize the metacognition engine.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._revision_count: int = 0

    def evaluate(
        self,
        state: BrainState,
        knowledge_texts: List[str],
        memory_texts: List[str],
    ) -> CognitiveAssessment:
        """Evaluate the AI's own response across cognitive dimensions.

        Args:
            state: Current BrainState with synthesized_output.
            knowledge_texts: Retrieved knowledge texts.
            memory_texts: Retrieved memory texts.

        Returns:
            CognitiveAssessment with dimension scores and issues.

        Complexity: T(n) ∈ O(n + k), S(n) ∈ O(1).
        """
        response = state.synthesized_output or ""
        query = state.processed_input or ""
        issues: List[str] = []

        if not response or len(response.strip()) < 10:
            return CognitiveAssessment(
                relevance=0.0, grounding=0.0, coherence=0.0,
                completeness=0.0, specificity=0.0, self_awareness=0.0,
                overall=0.0,
                issues=("Empty or trivial response",),
                should_revise=True,
            )

        response_lower = response.lower()
        query_lower = query.lower()

        # ─── 1. Relevance — does the response address the query? ───────
        query_terms = set(re.findall(r"[a-z0-9]+", query_lower))
        stop = {"the", "is", "a", "an", "what", "who", "how", "why",
                "where", "when", "of", "to", "in", "for", "and", "or",
                "are", "was", "were", "be", "do", "does", "did", "you",
                "your", "me", "my", "i", "please", "tell"}
        query_terms -= stop
        response_terms = set(re.findall(r"[a-z0-9]+", response_lower))

        if query_terms:
            overlap = len(query_terms & response_terms) / len(query_terms)
        else:
            overlap = 0.3  # neutral if no signal terms

        relevance = min(1.0, overlap * 1.2)
        if overlap < 0.15:
            issues.append("Low query term overlap — response may not address the question")
            relevance = max(0.0, overlap)

        # ─── 2. Grounding — is the response grounded in knowledge? ─────
        grounding = 0.0
        if knowledge_texts:
            # Check if response shares content with knowledge
            knowledge_combined = " ".join(knowledge_texts).lower()
            knowledge_terms = set(re.findall(r"[a-z0-9]+", knowledge_combined))
            if knowledge_terms:
                grounded_terms = response_terms & knowledge_terms
                grounding = min(1.0, len(grounded_terms) / max(10, len(query_terms) * 3))
        if not knowledge_texts and state.intent not in ("greet", "clarify", "emotional"):
            issues.append("No knowledge grounding — response is ungrounded")
        elif grounding < 0.1 and knowledge_texts:
            issues.append("Response does not use retrieved knowledge effectively")

        # ─── 3. Coherence — logical structure ──────────────────────────
        coherence_markers = sum(1 for m in self._COHERENCE_MARKERS if m in response_lower)
        sentence_count = max(1, response.count(".") + response.count("!") + response.count("?"))
        coherence = min(1.0, coherence_markers / max(3, sentence_count * 0.3))
        if sentence_count < 2:
            coherence = min(coherence, 0.4)
            issues.append("Response too short for coherent structure")

        # ─── 4. Completeness — coverage of key aspects ─────────────────
        completeness = 0.5  # baseline
        if len(response) > 200:
            completeness += 0.2
        if len(response) > 500:
            completeness += 0.15
        if coherence_markers >= 3:
            completeness += 0.15
        if state.intent == "code" and "```" in response:
            completeness += 0.2
        if state.intent == "identity" and any(p in response_lower for p in self._SELF_AWARE_PATTERNS):
            completeness += 0.2
        completeness = min(1.0, completeness)
        if len(response) < 50 and state.intent not in ("greet", "clarify"):
            issues.append("Response too short — likely incomplete")
            completeness = min(completeness, 0.3)

        # ─── 5. Specificity — specific vs generic/templated ────────────
        generic_hits = sum(1 for p in self._GENERIC_PATTERNS if p in response_lower)
        if generic_hits >= 3:
            specificity = max(0.0, 1.0 - generic_hits * 0.25)
            issues.append(f"Response appears generic/templated ({generic_hits} template markers)")
        elif generic_hits >= 1:
            specificity = max(0.3, 1.0 - generic_hits * 0.2)
        else:
            specificity = 0.9

        # ─── 6. Self-awareness — for identity/self-referential queries ──
        self_awareness = 0.5  # neutral baseline
        if state.intent == "identity" or "condrox" in query_lower:
            aware_hits = sum(1 for p in self._SELF_AWARE_PATTERNS if p in response_lower)
            self_awareness = min(1.0, aware_hits / 3.0)
            if aware_hits == 0:
                issues.append("Identity query but response lacks self-awareness")
                self_awareness = 0.1
        elif state.intent in ("greet", "clarify"):
            self_awareness = 0.7  # conversational responses don't need deep self-awareness

        # ─── Overall score — weighted combination ──────────────────────
        weights = {
            "relevance": 0.25,
            "grounding": 0.20,
            "coherence": 0.15,
            "completeness": 0.15,
            "specificity": 0.20,
            "self_awareness": 0.05,
        }
        # Boost self_awareness weight for identity queries
        if state.intent == "identity":
            weights = {
                "relevance": 0.15,
                "grounding": 0.15,
                "coherence": 0.10,
                "completeness": 0.15,
                "specificity": 0.15,
                "self_awareness": 0.30,
            }

        scores = {
            "relevance": relevance,
            "grounding": grounding,
            "coherence": coherence,
            "completeness": completeness,
            "specificity": specificity,
            "self_awareness": self_awareness,
        }
        overall = sum(scores[k] * weights[k] for k in weights)

        # ─── Revision decision ─────────────────────────────────────────
        should_revise = (
            overall < 0.4
            or specificity < 0.3
            or (state.intent == "identity" and self_awareness < 0.2)
            or (relevance < 0.15 and state.intent not in ("greet", "emotional"))
        )

        return CognitiveAssessment(
            relevance=relevance,
            grounding=grounding,
            coherence=coherence,
            completeness=completeness,
            specificity=specificity,
            self_awareness=self_awareness,
            overall=overall,
            issues=tuple(issues),
            should_revise=should_revise,
        )

    def revise(
        self,
        state: BrainState,
        assessment: CognitiveAssessment,
        knowledge_texts: List[str],
        memory_texts: List[str],
    ) -> str:
        """Revise the response based on the cognitive assessment.

        The revision strategy depends on what issues were identified:
        - Generic/templated → replace with knowledge-grounded content
        - Low relevance → re-extract from knowledge focusing on query terms
        - Low self-awareness (identity) → use self-knowledge directly
        - Low grounding → inject knowledge content

        Args:
            state: Current BrainState.
            assessment: The cognitive assessment.
            knowledge_texts: Retrieved knowledge texts.
            memory_texts: Retrieved memory texts.

        Returns:
            Revised response text.

        Complexity: T(n) ∈ O(n + k), S(n) ∈ O(n).
        """
        original = state.synthesized_output or ""
        query = state.processed_input or ""
        self._revision_count += 1

        # ─── Strategy 1: Identity queries — use self-knowledge directly ─
        if state.intent == "identity" and knowledge_texts:
            # Find the most relevant self-knowledge item
            best = self._find_most_relevant(query, knowledge_texts)
            if best:
                # Add a reflective prefix for self-awareness
                prefix = self._build_reflective_prefix(state, assessment)
                return f"{prefix}{best}" if prefix else best

        # ─── Strategy 2: Generic/templated response → replace with knowledge
        if assessment.specificity < 0.3 and knowledge_texts:
            best = self._find_most_relevant(query, knowledge_texts)
            if best:
                prefix = self._build_reflective_prefix(state, assessment)
                return f"{prefix}{best}" if prefix else best

        # ─── Strategy 3: Low relevance → re-extract from knowledge ──────
        if assessment.relevance < 0.15 and knowledge_texts:
            extracted = self._extract_relevant_sentences(query, knowledge_texts)
            if extracted and len(extracted) > 50:
                return extracted

        # ─── Strategy 4: Low grounding → inject knowledge ───────────────
        if assessment.grounding < 0.1 and knowledge_texts:
            best = self._find_most_relevant(query, knowledge_texts)
            if best:
                # Keep original structure but add knowledge
                return f"{original}\n\n{best[:500]}"

        # ─── Strategy 5: Low completeness → expand with knowledge ───────
        if assessment.completeness < 0.3 and knowledge_texts:
            best = self._find_most_relevant(query, knowledge_texts)
            if best:
                return f"{original}\n\nAdditional context: {best[:400]}"

        # ─── Strategy 6: Generic template with no knowledge → honest admission
        if assessment.specificity < 0.3 and not knowledge_texts:
            topic = state.processed_input.strip()
            if len(topic) > 100:
                topic = topic[:100] + "..."
            return (
                f"I don't have specific knowledge about \"{topic}\" in my current knowledge base. "
                f"I'm a non-LLM AI, so I can only answer from what I've been taught through my knowledge backend. "
                f"Could you rephrase, or ask me about something I might know about?"
            )

        # ─── Fallback: no specific fix available, keep original ────────
        return original

    def _find_most_relevant(self, query: str, texts: List[str]) -> Optional[str]:
        """Find the most relevant text item for the query using term overlap.

        Complexity: T(n) ∈ O(k * n) for k texts, n avg terms.
        """
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        stop = {"the", "is", "a", "an", "what", "who", "how", "why",
                "where", "when", "of", "to", "in", "for", "and", "or",
                "are", "was", "were", "be", "do", "does", "did", "you",
                "your", "me", "my", "i", "please", "tell"}
        query_terms -= stop
        if not query_terms:
            return texts[0] if texts else None

        best_score = 0
        best_text = None
        for text in texts:
            text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
            overlap = len(query_terms & text_terms)
            # Also check substring matches for partial words
            if overlap == 0:
                for qt in query_terms:
                    if any(qt in t for t in text_terms if len(t) > 3):
                        overlap += 1
                        break
            if overlap > best_score:
                best_score = overlap
                best_text = text

        return best_text

    def _extract_relevant_sentences(self, query: str, texts: List[str]) -> str:
        """Extract sentences from knowledge that are relevant to the query.

        Complexity: T(n) ∈ O(k * s) for k texts, s sentences.
        """
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        stop = {"the", "is", "a", "an", "what", "who", "how", "why",
                "where", "when", "of", "to", "in", "for", "and", "or",
                "are", "was", "were", "be", "do", "does", "did", "you",
                "your", "me", "my", "i", "please", "tell"}
        query_terms -= stop
        if not query_terms:
            return ""

        scored: List[Tuple[float, str]] = []
        for text in texts[:5]:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15 or len(sent) > 500:
                    continue
                sent_terms = set(re.findall(r"[a-z0-9]+", sent.lower()))
                overlap = len(query_terms & sent_terms)
                if overlap == 0:
                    for qt in query_terms:
                        if any(qt in t for t in sent_terms if len(t) > 3):
                            overlap += 1
                            break
                if overlap > 0:
                    score = overlap / max(1, len(query_terms))
                    if overlap >= 2:
                        score *= 1.5
                    scored.append((score, sent))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        return " ".join(s for _, s in scored[:4])

    def _build_reflective_prefix(
        self, state: BrainState, assessment: CognitiveAssessment
    ) -> str:
        """Build a self-reflective prefix for the response.

        Shows the AI is aware of its own thinking process.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if state.intent == "identity":
            return ""
        if assessment.specificity < 0.3:
            return ""
        return ""

    @property
    def revision_count(self) -> int:
        """Total number of revisions performed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._revision_count
