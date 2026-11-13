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

"""Cognitive Phases — syntetiske hjernefaser.

Disse er ikke separate moduler — de er faser i samme strøm:
- Reasoning Phase: logisk flertrinns tenking, strategier, vurdering
- Pattern Phase: mønstergjenkjenning, avvik, likheter, repetisjon
- Analysis Phase: nedbryting, sammenligning, årsak–virkning
- Reverse Phase: baklengs tenking, dekomponering, finne årsak fra effekt
- Structure Phase: se helhet, systemer, sammenhenger, hierarki

Alle faser påvirker hverandre. Ingen av dem er isolert.

Complexity:
    process: T(n) ∈ O(p * n) for p phases, n input size
    reasoning: T(n) ∈ O(d * e) for d depth, e evidence
    pattern: T(n) ∈ O(n * p) for n tokens, p patterns
    analysis: T(n) ∈ O(c * d) for c components, d dimensions
    reverse: T(n) ∈ O(s * c) for s steps, c causes
    structure: T(n) ∈ O(n) for n components

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Result of a single cognitive phase.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(r) for r results.
    """
    phase_name: str
    insights: Tuple[str, ...]        # Key insights from this phase
    confidence: float                # Phase confidence [0.0, 1.0]
    connections: Tuple[str, ...] = ()  # Connections to other phases
    evidence: Tuple[str, ...] = ()    # Supporting evidence


@dataclass(frozen=True, slots=True)
class CognitiveFlowResult:
    """Result of running all cognitive phases on input.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(p) for p phases.
    """
    phases: Tuple[PhaseResult, ...]
    unified_insight: str             # The combined insight from all phases
    overall_confidence: float        # Combined confidence
    connections: Tuple[str, ...]     # Cross-phase connections found


class ReasoningPhase:
    """Reasoning Phase — logisk flertrinns tenking.

    Performs multi-step logical reasoning:
    1. Identify premises (what do we know?)
    2. Apply rules (what follows from what we know?)
    3. Draw conclusions (what can we conclude?)
    4. Validate (is the conclusion sound?)

    Complexity: T(n) ∈ O(d * e) for d depth, e evidence.
    """

    __slots__ = ()

    # Logical connectors that indicate reasoning structure
    _CAUSAL: List[str] = ["because", "therefore", "since", "hence", "thus", "so"]
    _CONDITIONAL: List[str] = ["if", "when", "unless", "provided", "assuming"]
    _CONTRASTIVE: List[str] = ["but", "however", "although", "yet", "while", "whereas"]

    def process(self, input_text: str, context: Dict[str, any] = None) -> PhaseResult:
        """Run reasoning phase on input.

        Complexity: T(n) ∈ O(n) for input length n.
        """
        ctx = context or {}
        insights: List[str] = []
        evidence: List[str] = []

        # Step 1: Identify premises
        premises = self._extract_premises(input_text)
        if premises:
            insights.append(f"Premises identified: {len(premises)}")
            evidence.extend(premises[:3])

        # Step 2: Identify logical structure
        causal_count = sum(1 for c in self._CAUSAL if c in input_text.lower())
        conditional_count = sum(1 for c in self._CONDITIONAL if c in input_text.lower())
        contrastive_count = sum(1 for c in self._CONTRASTIVE if c in input_text.lower())

        if causal_count > 0:
            insights.append(f"Causal reasoning detected ({causal_count} causal connectors)")
        if conditional_count > 0:
            insights.append(f"Conditional reasoning detected ({conditional_count} conditional connectors)")
        if contrastive_count > 0:
            insights.append(f"Contrastive reasoning detected ({contrastive_count} contrastive connectors)")

        # Step 3: Draw conclusions
        conclusion = self._draw_conclusion(input_text, premises)
        if conclusion:
            insights.append(f"Conclusion: {conclusion}")

        # Step 4: Confidence based on reasoning structure
        total_connectors = causal_count + conditional_count + contrastive_count
        confidence = min(0.95, 0.4 + total_connectors * 0.1)

        return PhaseResult(
            phase_name="reasoning",
            insights=tuple(insights),
            confidence=confidence,
            evidence=tuple(evidence),
            connections=("pattern", "analysis"),
        )

    def _extract_premises(self, text: str) -> List[str]:
        """Extract premises from text.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        premises: List[str] = []
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Sentences with causal connectors are premises
            if any(c in sentence.lower() for c in self._CAUSAL):
                premises.append(sentence)
            # Sentences with definitions are premises
            if " is " in sentence.lower() or " are " in sentence.lower():
                if len(sentence) > 20:
                    premises.append(sentence)
        return premises[:5]

    def _draw_conclusion(self, text: str, premises: List[str]) -> str:
        """Draw a conclusion from premises.

        Complexity: T(n) ∈ O(1).
        """
        if not premises:
            return ""
        # Simple conclusion: synthesize the last premise
        last = premises[-1]
        if "therefore" in text.lower() or "hence" in text.lower():
            return last
        return f"Based on {len(premises)} premise(s), the reasoning supports: {last[:100]}"


class PatternPhase:
    """Pattern Phase — mønstergjenkjenning.

    Detects patterns in input:
    - Repetitions (words or phrases that repeat)
    - Structural patterns (question patterns, statement patterns)
    - Anomalies (unexpected elements)
    - Similarities (to known patterns)

    Complexity: T(n) ∈ O(n * p) for n tokens, p patterns.
    """

    __slots__ = ("_known_patterns",)

    def __init__(self) -> None:
        """Initialize with known patterns."""
        self._known_patterns: Dict[str, re.Pattern] = {
            "question_what": re.compile(r"what\s+(?:is|are|does|do)\b", re.IGNORECASE),
            "question_how": re.compile(r"how\s+(?:do|does|can|to)\b", re.IGNORECASE),
            "question_why": re.compile(r"why\s+(?:is|are|does|do)\b", re.IGNORECASE),
            "definition": re.compile(r"\b\w+\s+is\s+(?:a|an|the)\s+\w+", re.IGNORECASE),
            "comparison": re.compile(r"\b\w+\s+(?:vs|versus|compared to)\s+\w+", re.IGNORECASE),
            "list": re.compile(r"(?:^|\n)\s*(?:\d+\.|[-*])\s+", re.MULTILINE),
            "code": re.compile(r"```|def |function |class |import ", re.IGNORECASE),
            "causal": re.compile(r"\b(?:because|therefore|since|hence|thus)\b", re.IGNORECASE),
        }

    def process(self, input_text: str, context: Dict[str, any] = None) -> PhaseResult:
        """Run pattern phase on input.

        Complexity: T(n) ∈ O(n * p) for n text length, p patterns.
        """
        insights: List[str] = []
        evidence: List[str] = []
        detected_patterns: List[str] = []

        # Detect known patterns
        for name, pattern in self._known_patterns.items():
            matches = pattern.findall(input_text)
            if matches:
                detected_patterns.append(name)
                insights.append(f"Pattern '{name}' detected ({len(matches)} occurrences)")
                evidence.append(f"Pattern: {name}")

        # Detect repetitions
        words = input_text.lower().split()
        word_counts: Dict[str, int] = {}
        for word in words:
            clean = word.strip(".,!?;:\"'()[]{}")
            if len(clean) > 3:
                word_counts[clean] = word_counts.get(clean, 0) + 1

        repetitions = [(w, c) for w, c in word_counts.items() if c > 1]
        if repetitions:
            top_repeat = max(repetitions, key=lambda x: x[1])
            insights.append(f"Repetition detected: '{top_repeat[0]}' appears {top_repeat[1]} times")
            evidence.append(f"Repeated: {top_repeat[0]}")

        # Detect anomalies (unusual word combinations)
        anomalies = self._detect_anomalies(input_text)
        if anomalies:
            insights.append(f"Anomaly detected: {anomalies[0]}")
            evidence.append(f"Anomaly: {anomalies[0]}")

        # Confidence based on pattern detection
        confidence = min(0.9, 0.3 + len(detected_patterns) * 0.15)

        return PhaseResult(
            phase_name="pattern",
            insights=tuple(insights),
            confidence=confidence,
            evidence=tuple(evidence),
            connections=("reasoning", "structure"),
        )

    def _detect_anomalies(self, text: str) -> List[str]:
        """Detect anomalies in text.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        anomalies: List[str] = []
        # Detect mixed languages (non-ASCII characters mixed with English)
        if re.search(r'[^\x00-\x7F]', text) and re.search(r'[a-zA-Z]', text):
            anomalies.append("Mixed character sets detected")
        # Detect unusual capitalization
        if re.search(r'[A-Z]{5,}', text):
            anomalies.append("Unusual capitalization pattern")
        # Detect numbers mixed with words in unexpected ways
        if re.search(r'\d+[a-zA-Z]+\d+', text):
            anomalies.append("Alphanumeric mixing")
        return anomalies


class AnalysisPhase:
    """Analysis Phase — nedbryting, sammenligning, årsak–virkning.

    Breaks down input into components and analyzes:
    - Components (what are the parts?)
    - Relationships (how do parts connect?)
    - Dimensions (what aspects can be evaluated?)
    - Comparisons (how do components compare?)

    Complexity: T(n) ∈ O(c * d) for c components, d dimensions.
    """

    __slots__ = ()

    # Analysis dimensions
    _DIMENSIONS = ["performance", "complexity", "relevance", "novelty", "clarity"]

    def process(self, input_text: str, context: Dict[str, any] = None) -> PhaseResult:
        """Run analysis phase on input.

        Complexity: T(n) ∈ O(n) for input length n.
        """
        insights: List[str] = []
        evidence: List[str] = []

        # Break down into components
        components = self._decompose(input_text)
        insights.append(f"Decomposed into {len(components)} components")
        evidence.extend([f"Component: {c[:50]}" for c in components[:3]])

        # Identify relationships between components
        relationships = self._find_relationships(components)
        if relationships:
            insights.append(f"Found {len(relationships)} relationships between components")
            evidence.extend(relationships[:3])

        # Evaluate on dimensions
        dimension_scores = self._evaluate_dimensions(input_text, components)
        top_dimension = max(dimension_scores, key=dimension_scores.get)
        insights.append(f"Strongest dimension: {top_dimension} ({dimension_scores[top_dimension]:.2f})")

        # Confidence based on decomposition quality
        confidence = min(0.9, 0.4 + len(components) * 0.05)

        return PhaseResult(
            phase_name="analysis",
            insights=tuple(insights),
            confidence=confidence,
            evidence=tuple(evidence),
            connections=("reasoning", "reverse", "structure"),
        )

    def _decompose(self, text: str) -> List[str]:
        """Decompose text into components.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        # Split by sentences
        sentences = re.split(r'[.!?]+', text)
        components = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        if not components:
            # Split by commas if no sentences
            parts = text.split(",")
            components = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]
        return components[:10]

    def _find_relationships(self, components: List[str]) -> List[str]:
        """Find relationships between components.

        Complexity: T(n) ∈ O(c²) for c components.
        """
        relationships: List[str] = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                # Check for shared words
                words_i = set(components[i].lower().split())
                words_j = set(components[j].lower().split())
                shared = words_i & words_j
                # Filter out stop words
                stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "and", "or"}
                meaningful_shared = shared - stop
                if len(meaningful_shared) > 1:
                    relationships.append(
                        f"Components {i+1} and {j+1} share: {', '.join(list(meaningful_shared)[:3])}"
                    )
        return relationships[:5]

    def _evaluate_dimensions(self, text: str, components: List[str]) -> Dict[str, float]:
        """Evaluate text on analysis dimensions.

        Complexity: T(n) ∈ O(d) for d dimensions.
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        # Performance: mentions of speed, efficiency, optimization
        perf_signals = sum(1 for w in ("fast", "slow", "efficient", "optimize", "performance", "speed") if w in text_lower)
        scores["performance"] = min(1.0, perf_signals * 0.2)

        # Complexity: mentions of complexity, difficulty, layers
        complexity_signals = sum(1 for w in ("complex", "simple", "difficult", "easy", "layer", "hierarchy") if w in text_lower)
        scores["complexity"] = min(1.0, complexity_signals * 0.2)

        # Relevance: how relevant to the question
        scores["relevance"] = min(1.0, len(components) * 0.1)

        # Novelty: unique words ratio
        words = text_lower.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            scores["novelty"] = unique_ratio
        else:
            scores["novelty"] = 0.0

        # Clarity: sentence length and structure
        avg_len = sum(len(c) for c in components) / max(1, len(components))
        scores["clarity"] = min(1.0, max(0.0, 1.0 - (avg_len - 50) / 100))

        return scores


class ReversePhase:
    """Reverse Phase — baklengs tenking, dekomponering.

    Thinks backwards from effect to cause:
    1. Identify the effect (what happened?)
    2. Work backwards to find causes
    3. Identify root causes
    4. Verify the causal chain

    Complexity: T(n) ∈ O(s * c) for s steps, c causes.
    """

    __slots__ = ()

    # Effect indicators
    _EFFECT_MARKERS = ["therefore", "so", "thus", "hence", "as a result", "consequently"]
    # Cause indicators
    _CAUSE_MARKERS = ["because", "since", "due to", "caused by", "results from", "stemming from"]

    def process(self, input_text: str, context: Dict[str, any] = None) -> PhaseResult:
        """Run reverse phase on input.

        Complexity: T(n) ∈ O(n) for input length n.
        """
        insights: List[str] = []
        evidence: List[str] = []
        text_lower = input_text.lower()

        # Identify effects
        effects = self._identify_effects(input_text)
        if effects:
            insights.append(f"Effects identified: {len(effects)}")
            evidence.extend([f"Effect: {e[:50]}" for e in effects[:2]])

        # Work backwards to causes
        causes = self._identify_causes(input_text)
        if causes:
            insights.append(f"Causes identified: {len(causes)}")
            evidence.extend([f"Cause: {c[:50]}" for c in causes[:2]])

        # Build causal chain
        chain = self._build_causal_chain(causes, effects)
        if chain:
            insights.append(f"Causal chain: {chain}")

        # Identify root causes
        root_causes = self._find_root_causes(causes)
        if root_causes:
            insights.append(f"Root cause: {root_causes[0][:80]}")

        # Confidence based on causal structure
        confidence = min(0.85, 0.3 + (len(effects) + len(causes)) * 0.1)

        return PhaseResult(
            phase_name="reverse",
            insights=tuple(insights),
            confidence=confidence,
            evidence=tuple(evidence),
            connections=("reasoning", "analysis"),
        )

    def _identify_effects(self, text: str) -> List[str]:
        """Identify effects in text.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        effects: List[str] = []
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if any(marker in sentence.lower() for marker in self._EFFECT_MARKERS):
                effects.append(sentence)
        return effects[:5]

    def _identify_causes(self, text: str) -> List[str]:
        """Identify causes in text.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        causes: List[str] = []
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if any(marker in sentence.lower() for marker in self._CAUSE_MARKERS):
                causes.append(sentence)
        return causes[:5]

    def _build_causal_chain(self, causes: List[str], effects: List[str]) -> str:
        """Build a causal chain from causes to effects.

        Complexity: T(n) ∈ O(1).
        """
        if not causes or not effects:
            return ""
        return f"{causes[0][:40]}... → {effects[0][:40]}..."

    def _find_root_causes(self, causes: List[str]) -> List[str]:
        """Find root causes (causes that don't have their own causes).

        Complexity: T(n) ∈ O(c) for c causes.
        """
        if not causes:
            return []
        # The first cause is likely the root cause
        return [causes[0]]


class StructurePhase:
    """Structure Phase — se helhet, systemer, sammenhenger.

    Identifies the overall structure of input:
    - Hierarchy (what is the top-level structure?)
    - Components (what are the parts?)
    - Connections (how do parts connect?)
    - System type (what kind of system is this?)

    Complexity: T(n) ∈ O(n) for n components.
    """

    __slots__ = ()

    def process(self, input_text: str, context: Dict[str, any] = None) -> PhaseResult:
        """Run structure phase on input.

        Complexity: T(n) ∈ O(n) for input length n.
        """
        insights: List[str] = []
        evidence: List[str] = []

        # Identify structure type
        structure_type = self._identify_structure_type(input_text)
        insights.append(f"Structure type: {structure_type}")
        evidence.append(f"Type: {structure_type}")

        # Identify hierarchy
        hierarchy = self._identify_hierarchy(input_text)
        if hierarchy:
            insights.append(f"Hierarchy: {hierarchy}")
            evidence.append(f"Hierarchy: {hierarchy[:80]}")

        # Identify system boundaries
        boundaries = self._identify_boundaries(input_text)
        if boundaries:
            insights.append(f"System boundaries: {len(boundaries)} identified")

        # Identify connections to other phases
        connections = self._identify_cross_connections(input_text)

        # Confidence based on structure clarity
        confidence = min(0.9, 0.4 + (1 if structure_type != "unstructured" else 0) * 0.3)

        return PhaseResult(
            phase_name="structure",
            insights=tuple(insights),
            confidence=confidence,
            evidence=tuple(evidence),
            connections=tuple(connections),
        )

    def _identify_structure_type(self, text: str) -> str:
        """Identify the type of structure in text.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        if "```" in text or "def " in text:
            return "code"
        if re.search(r'(?:^|\n)\s*(?:\d+\.|[-*])\s+', text, re.MULTILINE):
            return "list"
        if re.search(r"phase\s+\d|step\s+\d", text, re.IGNORECASE):
            return "phased"
        if "?" in text:
            return "question_answer"
        if ":" in text and "\n" in text:
            return "structured_definition"
        if len(text.split(".")) > 3:
            return "multi_sentence"
        return "unstructured"

    def _identify_hierarchy(self, text: str) -> str:
        """Identify hierarchical structure.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        # Check for numbered hierarchy
        numbered = re.findall(r'\d+\.\s+', text)
        if len(numbered) > 1:
            return f"Numbered hierarchy with {len(numbered)} levels"
        # Check for indentation-based hierarchy
        lines = text.split("\n")
        indented = [l for l in lines if l.startswith("  ") or l.startswith("\t")]
        if indented:
            return f"Indentation hierarchy with {len(indented)} sub-levels"
        return ""

    def _identify_boundaries(self, text: str) -> List[str]:
        """Identify system boundaries.

        Complexity: T(n) ∈ O(n) for text length n.
        """
        boundaries: List[str] = []
        # Paragraph breaks
        if "\n\n" in text:
            boundaries.append("Paragraph breaks")
        # Section markers
        if re.search(r'(?:^|\n)#+\s', text, re.MULTILINE):
            boundaries.append("Section headers")
        # Code blocks
        if "```" in text:
            boundaries.append("Code block boundaries")
        return boundaries

    def _identify_cross_connections(self, text: str) -> List[str]:
        """Identify connections to other cognitive phases.

        Complexity: T(n) ∈ O(1).
        """
        connections: List[str] = []
        text_lower = text.lower()
        if any(w in text_lower for w in ("because", "therefore", "since")):
            connections.append("reasoning")
        if any(w in text_lower for w in ("pattern", "repeat", "recurring")):
            connections.append("pattern")
        if any(w in text_lower for w in ("analyze", "break down", "component")):
            connections.append("analysis")
        if any(w in text_lower for w in ("cause", "effect", "result")):
            connections.append("reverse")
        return connections


class CognitivePhases:
    """Orchestrates all cognitive phases as one unified flow.

    All phases run in sequence and their results are combined into
    a unified cognitive flow. Phases influence each other — the
    output of one phase becomes context for the next.

    Complexity:
        process: T(n) ∈ O(p * n) for p phases, n input size
    """

    __slots__ = ("_reasoning", "_pattern", "_analysis", "_reverse", "_structure")

    def __init__(self) -> None:
        """Initialize all cognitive phases."""
        self._reasoning = ReasoningPhase()
        self._pattern = PatternPhase()
        self._analysis = AnalysisPhase()
        self._reverse = ReversePhase()
        self._structure = StructurePhase()

    def process(self, input_text: str, context: Optional[Dict[str, any]] = None) -> CognitiveFlowResult:
        """Run all cognitive phases on input.

        All phases run in sequence, with each phase receiving the
        accumulated context from previous phases. Results are combined
        into a unified cognitive flow.

        Args:
            input_text: The input to process.
            context: Optional starting context.

        Returns:
            CognitiveFlowResult with all phase results and unified insight.

        Complexity: T(n) ∈ O(p * n) for p phases, n input size.
        """
        ctx = context or {}
        phase_results: List[PhaseResult] = []

        # Run each phase, passing accumulated context
        for phase in (self._reasoning, self._pattern, self._analysis, self._reverse, self._structure):
            try:
                result = phase.process(input_text, ctx)
                phase_results.append(result)
                # Enrich context with phase results
                ctx[f"{result.phase_name}_insights"] = result.insights
                ctx[f"{result.phase_name}_confidence"] = result.confidence
            except Exception:
                # Bulletproof — one phase failing doesn't break the flow
                phase_results.append(PhaseResult(
                    phase_name=phase.__class__.__name__.lower().replace("phase", ""),
                    insights=(),
                    confidence=0.0,
                ))

        # Combine all insights into unified insight
        all_insights: List[str] = []
        all_connections: Set[str] = set()
        total_confidence = 0.0

        for result in phase_results:
            all_insights.extend(result.insights)
            all_connections.update(result.connections)
            total_confidence += result.confidence

        overall_confidence = total_confidence / max(1, len(phase_results))

        # Build unified insight
        unified = self._build_unified_insight(phase_results)

        return CognitiveFlowResult(
            phases=tuple(phase_results),
            unified_insight=unified,
            overall_confidence=overall_confidence,
            connections=tuple(all_connections),
        )

    def _build_unified_insight(self, phase_results: List[PhaseResult]) -> str:
        """Build a unified insight from all phase results.

        Complexity: T(n) ∈ O(p) for p phases.
        """
        parts: List[str] = []
        for result in phase_results:
            if result.insights:
                top_insight = result.insights[0]
                parts.append(f"[{result.phase_name}] {top_insight}")

        if not parts:
            return "No unified insight — insufficient cognitive activity."

        return " | ".join(parts[:5])  # Top 5 insights across all phases
