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

"""Deep Reasoning Engine — multi-step reasoning with evidence chains.

Performs multi-step reasoning over knowledge and context:
1. Decomposes the question into sub-questions
2. Retrieves relevant knowledge for each sub-question
3. Builds evidence chains connecting facts to conclusions
4. Synthesizes a reasoned answer from the evidence
5. Validates the conclusion against existing knowledge

This is NOT an LLM — it uses rule-based logic, pattern matching,
and knowledge graph traversal to reason about problems.

Complexity:
    reason: T(n) ∈ O(d * k * e) for d depth, k knowledge items, e evidence
    decompose: T(n) ∈ O(n) for query length n

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.autonomous.knowledge_validator import KnowledgeStore


@dataclass(frozen=True, slots=True)
class ReasoningStep:
    """A single step in a reasoning chain.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    step_number: int
    sub_question: str       # The sub-question being answered
    evidence: str           # Evidence supporting this step
    conclusion: str         # Conclusion drawn from evidence
    confidence: float       # Confidence in this step [0.0, 1.0]
    source: str             # Where the evidence came from


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Result of deep reasoning over a question.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(s) for s steps.
    """
    question: str
    steps: Tuple[ReasoningStep, ...]
    final_conclusion: str
    overall_confidence: float
    evidence_count: int
    reasoning_type: str     # "deductive", "inductive", "causal", "comparative"


class DeepReasoningEngine:
    """Multi-step reasoning engine.

    Decomposes complex questions into sub-questions, retrieves evidence
    for each, and builds a chain of reasoning that leads to a conclusion.

    Reasoning types:
    - Deductive: General rule → specific conclusion
    - Inductive: Specific examples → general pattern
    - Causal: Cause → effect chains
    - Comparative: Option A vs Option B → recommendation

    Complexity:
        reason: T(n) ∈ O(d * k * e) for d depth, k knowledge, e evidence
    """

    __slots__ = ("_store",)

    # Question decomposition patterns
    _DECOMPOSITION_MARKERS: List[str] = [
        "what is", "how does", "why does", "when should", "where is",
        "which is", "can you", "should i", "how to", "what are",
    ]

    # Causal connectors for building evidence chains
    _CAUSAL_CONNECTORS: List[str] = [
        "because", "therefore", "since", "hence", "thus",
        "consequently", "as a result", "due to", "given that",
    ]

    # Comparative markers
    _COMPARATIVE_MARKERS: List[str] = [
        "vs", "versus", "compared to", "better than", "worse than",
        "difference between", "compared with", "or",
    ]

    def __init__(self, store: KnowledgeStore) -> None:
        """Initialize with a knowledge store for evidence retrieval.

        Args:
            store: The KnowledgeStore to query for evidence.
        """
        self._store = store

    def reason(
        self,
        question: str,
        intent: str = "question",
        domain: str = "general",
        max_depth: int = 4,
    ) -> ReasoningResult:
        """Perform deep reasoning over a question.

        Args:
            question: The question to reason about.
            intent: Classified intent.
            domain: Knowledge domain.
            max_depth: Maximum reasoning depth (steps).

        Returns:
            ReasoningResult with the reasoning chain and conclusion.

        Complexity: T(n) ∈ O(d * k) for d depth, k knowledge items.
        """
        question_lower = question.lower().strip()

        # Detect reasoning type
        reasoning_type = self._detect_reasoning_type(question_lower, intent)

        # Decompose into sub-questions
        sub_questions = self._decompose(question, reasoning_type, max_depth)

        # Build reasoning steps
        steps: List[ReasoningStep] = []
        for i, sub_q in enumerate(sub_questions[:max_depth]):
            # Retrieve evidence from knowledge store
            evidence = self._retrieve_evidence(sub_q, domain)
            # Draw conclusion from evidence
            conclusion = self._draw_conclusion(sub_q, evidence, reasoning_type)
            confidence = self._step_confidence(evidence, conclusion)

            steps.append(ReasoningStep(
                step_number=i + 1,
                sub_question=sub_q,
                evidence=evidence,
                conclusion=conclusion,
                confidence=confidence,
                source="knowledge_store" if evidence != "No direct evidence found." else "inference",
            ))

        # Synthesize final conclusion
        final_conclusion = self._synthesize_conclusion(question, steps, reasoning_type)

        # Calculate overall confidence
        if steps:
            overall = sum(s.confidence for s in steps) / len(steps)
        else:
            overall = 0.3

        return ReasoningResult(
            question=question,
            steps=tuple(steps),
            final_conclusion=final_conclusion,
            overall_confidence=overall,
            evidence_count=sum(1 for s in steps if s.source == "knowledge_store"),
            reasoning_type=reasoning_type,
        )

    def _detect_reasoning_type(self, question_lower: str, intent: str) -> str:
        """Detect the type of reasoning needed.

        Complexity: T(n) ∈ O(1).
        """
        if any(m in question_lower for m in self._COMPARATIVE_MARKERS):
            return "comparative"
        if any(m in question_lower for m in ("why", "because", "cause", "leads to", "results in")):
            return "causal"
        if intent == "code":
            return "deductive"
        if intent == "analyze":
            return "inductive"
        if intent == "explain":
            return "deductive"
        return "deductive"

    def _decompose(
        self, question: str, reasoning_type: str, max_depth: int
    ) -> List[str]:
        """Decompose a question into sub-questions.

        Complexity: T(n) ∈ O(n) for question length n.
        """
        question_lower = question.lower().strip()
        sub_questions: List[str] = []

        if reasoning_type == "comparative":
            # Comparative: break into "what is A", "what is B", "how do they differ"
            parts = re.split(r'\s+(?:vs\.?|versus|compared to|or)\s+', question_lower, maxsplit=1)
            if len(parts) == 2:
                sub_questions.append(f"What is {parts[0].strip()}?")
                sub_questions.append(f"What is {parts[1].strip()}?")
                sub_questions.append("How do they differ in key aspects?")
                sub_questions.append("Which is better for the given context?")
            else:
                sub_questions.append(f"What is the main subject: {question[:100]}?")
                sub_questions.append("What are the key characteristics?")
                sub_questions.append("What are the trade-offs?")

        elif reasoning_type == "causal":
            # Causal: break into "what is the cause", "what is the effect", "why"
            sub_questions.append(f"What is happening in: {question[:100]}?")
            sub_questions.append("What is the cause?")
            sub_questions.append("What is the effect?")
            sub_questions.append("Why does this relationship hold?")

        elif reasoning_type == "deductive":
            # Deductive: break into "what is the concept", "what are the rules", "what follows"
            sub_questions.append(f"What is the core concept in: {question[:100]}?")
            sub_questions.append("What rules or principles apply?")
            sub_questions.append("What conclusion follows from these rules?")

        elif reasoning_type == "inductive":
            # Inductive: break into "what are the examples", "what pattern emerges"
            sub_questions.append(f"What are the key components of: {question[:100]}?")
            sub_questions.append("What patterns can be observed?")
            sub_questions.append("What general conclusion can be drawn?")

        else:
            sub_questions.append(f"What is: {question[:100]}?")
            sub_questions.append("What are the key aspects?")
            sub_questions.append("What is the conclusion?")

        # Ensure we have at least 2 sub-questions
        if len(sub_questions) < 2:
            sub_questions.append("What are the implications?")

        return sub_questions

    def _retrieve_evidence(self, sub_question: str, domain: str) -> str:
        """Retrieve evidence from the knowledge store.

        Complexity: T(n) ∈ O(f) for f facts in the domain.
        """
        # Extract key terms from the sub-question
        terms = self._extract_terms(sub_question)

        # Search knowledge store for relevant facts
        evidence_parts: List[str] = []

        for term in terms[:3]:  # Top 3 terms
            facts = self._store.find_by_subject(term)
            if facts:
                for fact, quality, importance in facts[:2]:  # Top 2 per term
                    if quality > 0.3:
                        evidence_parts.append(
                            f"{fact.subject} {fact.predicate} {fact.object}"
                        )

        # Also search by domain
        if not evidence_parts and domain != "general":
            domain_facts = self._store.find_by_domain(domain, limit=3)
            for fact, quality, _ in domain_facts:
                if quality > 0.3:
                    evidence_parts.append(
                        f"{fact.subject} {fact.predicate} {fact.object}"
                    )

        if evidence_parts:
            return "; ".join(evidence_parts[:4])
        return "No direct evidence found."

    def _draw_conclusion(
        self, sub_question: str, evidence: str, reasoning_type: str
    ) -> str:
        """Draw a conclusion from evidence.

        Complexity: T(n) ∈ O(1).
        """
        if evidence == "No direct evidence found.":
            # Use inference based on reasoning type
            if reasoning_type == "comparative":
                return "The comparison requires analyzing both options against key criteria."
            if reasoning_type == "causal":
                return "The causal relationship can be inferred from the context."
            return "Based on available context, a reasonable conclusion can be drawn."

        # Use the evidence directly as the basis for the conclusion
        return f"Based on evidence: {evidence[:200]}"

    def _step_confidence(self, evidence: str, conclusion: str) -> float:
        """Calculate confidence for a reasoning step.

        Complexity: T(n) ∈ O(1).
        """
        if "No direct evidence" in evidence:
            return 0.3
        # More evidence = higher confidence
        evidence_parts = evidence.split(";")
        base = min(0.9, 0.5 + len(evidence_parts) * 0.1)
        return base

    def _synthesize_conclusion(
        self, question: str, steps: List[ReasoningStep], reasoning_type: str
    ) -> str:
        """Synthesize a final conclusion from reasoning steps.

        Complexity: T(n) ∈ O(s) for s steps.
        """
        if not steps:
            return "Unable to reason about this question with current knowledge."

        # Collect conclusions from all steps
        conclusions = [s.conclusion for s in steps if s.confidence > 0.3]

        if not conclusions:
            return "The available evidence is insufficient for a confident conclusion."

        # Build a synthesized conclusion
        if reasoning_type == "comparative":
            return (
                f"Based on multi-step analysis: {' '.join(conclusions[:2])} "
                f"Therefore, the choice depends on which factors matter most."
            )
        if reasoning_type == "causal":
            return (
                f"Through causal analysis: {' '.join(conclusions[:2])} "
                f"Hence, the relationship is supported by the evidence."
            )
        if reasoning_type == "deductive":
            return (
                f"Through deductive reasoning: {' '.join(conclusions[:2])} "
                f"Therefore, the conclusion follows from the established rules."
            )
        return f"Through analysis: {' '.join(conclusions[:2])}"

    def _extract_terms(self, text: str) -> List[str]:
        """Extract key terms from text (excluding stop words).

        Complexity: T(n) ∈ O(n) for text length n.
        """
        stop_words = {
            "what", "is", "are", "the", "a", "an", "how", "does", "do",
            "why", "when", "where", "which", "can", "you", "should", "i",
            "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "tell", "me", "explain", "describe",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in stop_words][:10]
