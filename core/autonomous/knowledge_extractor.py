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

"""Knowledge Extractor — extracts structured facts from conversations.

Analyzes every user message and AI response to extract:
- Facts (statements that assert something about the world)
- Concepts (topics and ideas being discussed)
- Relationships (how concepts connect to each other)
- User knowledge (what the user knows or wants to know)
- Contextual metadata (domain, intent, emotion, confidence)

The extractor uses linguistic patterns, not LLMs, to identify factual
statements and structure them for storage.

Complexity:
    extract: T(n) ∈ O(n) for message length n, S(n) ∈ O(f) for f facts
    extract_relationships: T(n) ∈ O(c²) for c concepts, S(n) ∈ O(r) for r relationships

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """A single fact extracted from a conversation.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    subject: str          # What the fact is about (e.g., "Python", "recursion")
    predicate: str        # What it says (e.g., "is", "supports", "uses")
    object: str           # The value (e.g., "a programming language", "functions")
    source: str           # Where it came from (user message or AI response)
    confidence: float     # How confident we are [0.0, 1.0]
    domain: str           # Knowledge domain (programming, science, etc.)
    context: str          # Surrounding context text
    timestamp: float      # When it was extracted


@dataclass(frozen=True, slots=True)
class ExtractedConcept:
    """A concept extracted from a conversation.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    name: str             # Concept name (e.g., "machine learning")
    domain: str           # Knowledge domain
    frequency: int        # How often it appeared
    first_seen: float     # When first encountered
    context_snippets: Tuple[str, ...] = ()  # Context where it appeared


@dataclass(frozen=True, slots=True)
class ExtractedRelationship:
    """A relationship between two concepts.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    subject: str          # Source concept
    relation: str         # Relationship type (e.g., "is-a", "part-of", "used-by")
    object: str           # Target concept
    confidence: float     # How confident we are [0.0, 1.0]


@dataclass(slots=True)
class ExtractionResult:
    """Result of extracting knowledge from a conversation turn.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(f + c + r).
    """
    facts: List[ExtractedFact] = field(default_factory=list)
    concepts: List[ExtractedConcept] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    user_knowledge_level: float = 0.0  # [0.0, 1.0] — how much the user seems to know
    topics: List[str] = field(default_factory=list)
    intent: str = "unknown"
    emotion: str = "neutral"
    timestamp: float = 0.0


class KnowledgeExtractor:
    """Extracts structured knowledge from conversation messages.

    Uses linguistic pattern matching to identify:
    - Definitional facts ("X is Y", "X means Y", "X refers to Y")
    - Causal facts ("X causes Y", "X leads to Y", "X results in Y")
    - Capability facts ("X can do Y", "X supports Y", "X uses Y")
    - Comparative facts ("X is better than Y", "X is faster than Y")
    - Procedural facts ("To do X, first Y, then Z")

    Complexity:
        extract: T(n) ∈ O(n * p) for n tokens, p patterns
        extract_concepts: T(n) ∈ O(n * d) for n tokens, d domain keywords
    """

    __slots__ = (
        "_fact_patterns",
        "_concept_domains",
        "_stop_words",
        "_tech_keywords",
    )

    # Definitional patterns — "X is Y", "X means Y", etc.
    _DEFINITIONAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("is-a", re.compile(
            r'(\b[A-Z][\w\s]+?)\s+is\s+(?:a|an)\s+([\w\s\-]+?)(?:[.,;]|\s+that|\s+which|$)',
            re.IGNORECASE
        )),
        ("means", re.compile(
            r'(\b[\w\s]+?)\s+means\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("refers-to", re.compile(
            r'(\b[\w\s]+?)\s+refers to\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("defined-as", re.compile(
            r'(\b[\w\s]+?)\s+(?:is|are)\s+defined as\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
    ]

    # Causal patterns — "X causes Y", "X leads to Y"
    _CAUSAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("causes", re.compile(
            r'(\b[\w\s]+?)\s+causes?\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("leads-to", re.compile(
            r'(\b[\w\s]+?)\s+leads to\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("results-in", re.compile(
            r'(\b[\w\s]+?)\s+results in\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("because", re.compile(
            r'(\b[\w\s]+?)\s+because\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
    ]

    # Capability patterns — "X can do Y", "X supports Y"
    _CAPABILITY_PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("can", re.compile(
            r'(\b[\w\s]+?)\s+can\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("supports", re.compile(
            r'(\b[\w\s]+?)\s+supports?\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("uses", re.compile(
            r'(\b[\w\s]+?)\s+uses?\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("requires", re.compile(
            r'(\b[\w\s]+?)\s+requires?\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("enables", re.compile(
            r'(\b[\w\s]+?)\s+enables?\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
    ]

    # Comparative patterns — "X is better than Y"
    _COMPARATIVE_PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("better-than", re.compile(
            r'(\b[\w\s]+?)\s+is\s+(?:better|faster|safer|stronger)\s+than\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
        ("more-than", re.compile(
            r'(\b[\w\s]+?)\s+is\s+more\s+([\w\s\-]+?)\s+than\s+([\w\s\-]+?)(?:[.,;]|$)',
            re.IGNORECASE
        )),
    ]

    # Domain keywords — maps concepts to knowledge domains
    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "programming": [
            "python", "javascript", "java", "c++", "rust", "go", "typescript",
            "function", "variable", "class", "object", "array", "dict", "list",
            "loop", "recursion", "algorithm", "data structure", "compiler",
            "debugging", "refactoring", "api", "endpoint", "framework",
            "library", "module", "import", "async", "await", "thread",
            "concurrency", "mutex", "lock", "queue", "stack", "heap",
            "pointer", "reference", "closure", "decorator", "generator",
            "iterator", "callback", "promise", "future", "observable",
        ],
        "ai_ml": [
            "machine learning", "deep learning", "neural network", "transformer",
            "embedding", "vector", "tensor", "gradient", "backpropagation",
            "training", "inference", "model", "dataset", "feature", "label",
            "classification", "regression", "clustering", "reinforcement",
            "supervised", "unsupervised", "attention", "tokenizer", "softmax",
            "loss function", "optimizer", "epoch", "batch", "learning rate",
        ],
        "science": [
            "energy", "force", "mass", "velocity", "acceleration", "momentum",
            "gravity", "friction", "thermodynamics", "entropy", "quantum",
            "relativity", "electromagnetic", "wavelength", "frequency",
            "atom", "molecule", "electron", "proton", "neutron", "photon",
            "dna", "rna", "protein", "cell", "membrane", "enzyme",
            "evolution", "genetics", "ecosystem", "photosynthesis",
        ],
        "technology": [
            "database", "server", "client", "network", "protocol", "tcp",
            "http", "https", "dns", "cdn", "cloud", "docker", "kubernetes",
            "microservice", "rest", "graphql", "websocket", "redis", "postgres",
            "mongodb", "elasticsearch", "kafka", "rabbitmq", "nginx",
            "load balancer", "firewall", "vpn", "proxy", "gateway",
        ],
        "finance": [
            "investment", "stock", "bond", "portfolio", "dividend", "interest",
            "inflation", "gdp", "recession", "bull", "bear", "volatility",
            "asset", "liability", "equity", "revenue", "profit", "loss",
            "roi", "risk", "hedge", "derivative", "option", "future",
        ],
        "general": [
            "history", "culture", "language", "art", "music", "philosophy",
            "psychology", "sociology", "politics", "law", "education",
            "health", "medicine", "nutrition", "exercise", "sleep",
        ],
    }

    _STOP_WORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "shall", "to", "of", "in",
        "on", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "from", "up", "down", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "but", "and", "or", "if", "because", "as",
        "until", "while", "this", "that", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "what", "which", "who", "whom",
        "my", "your", "his", "her", "its", "our", "their",
    }

    def __init__(self) -> None:
        """Initialize the knowledge extractor with compiled patterns."""
        self._fact_patterns: List[Tuple[str, re.Pattern, str]] = (
            [(name, pat, "definitional") for name, pat in self._DEFINITIONAL_PATTERNS]
            + [(name, pat, "causal") for name, pat in self._CAUSAL_PATTERNS]
            + [(name, pat, "capability") for name, pat in self._CAPABILITY_PATTERNS]
            + [(name, pat, "comparative") for name, pat in self._COMPARATIVE_PATTERNS]
        )
        self._concept_domains: Dict[str, str] = {}
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            for kw in keywords:
                self._concept_domains[kw.lower()] = domain
        self._stop_words = self._STOP_WORDS
        self._tech_keywords = set(self._concept_domains.keys())

    def extract(
        self,
        user_message: str,
        ai_response: str,
        intent: str = "unknown",
        emotion: str = "neutral",
        timestamp: float = 0.0,
    ) -> ExtractionResult:
        """Extract knowledge from a single conversation turn.

        Args:
            user_message: What the user said.
            ai_response: What the AI responded.
            intent: Classified intent.
            emotion: Detected emotion.
            timestamp: When the turn occurred.

        Returns:
            ExtractionResult with facts, concepts, and relationships.

        Complexity: T(n) ∈ O(n * p) for n tokens, p patterns.
        """
        import time as _time
        ts = timestamp or _time.time()

        # Extract facts from both user message and AI response
        facts: List[ExtractedFact] = []
        facts.extend(self._extract_facts(user_message, "user", ts))
        facts.extend(self._extract_facts(ai_response, "assistant", ts))

        # Extract concepts
        concepts = self._extract_concepts(user_message + " " + ai_response, ts)

        # Extract relationships between concepts
        relationships = self._extract_relationships(concepts, user_message + " " + ai_response)

        # Estimate user knowledge level from their message
        user_knowledge = self._estimate_knowledge_level(user_message, intent)

        # Extract topics (top concepts by frequency)
        topics = [c.name for c in sorted(concepts, key=lambda c: c.frequency, reverse=True)[:5]]

        return ExtractionResult(
            facts=facts,
            concepts=concepts,
            relationships=relationships,
            user_knowledge_level=user_knowledge,
            topics=topics,
            intent=intent,
            emotion=emotion,
            timestamp=ts,
        )

    def _extract_facts(self, text: str, source: str, timestamp: float) -> List[ExtractedFact]:
        """Extract facts using pattern matching.

        Complexity: T(n) ∈ O(n * p) for n text length, p patterns.
        """
        facts: List[ExtractedFact] = []
        text_clean = text.strip()
        if not text_clean or len(text_clean) < 10:
            return facts

        for pattern_name, pattern, fact_type in self._fact_patterns:
            for match in pattern.finditer(text_clean):
                subject = match.group(1).strip().lower()
                if len(subject) < 2 or subject in self._stop_words:
                    continue
                if len(match.groups()) >= 2:
                    obj = match.group(2).strip().lower()
                else:
                    continue

                if len(obj) < 2 or obj in self._stop_words:
                    continue

                # Determine domain from subject and object
                domain = self._detect_domain(subject + " " + obj)

                # Confidence based on fact type and text quality
                confidence = self._fact_confidence(fact_type, subject, obj, text_clean)

                # Extract context (surrounding text)
                start = max(0, match.start() - 30)
                end = min(len(text_clean), match.end() + 30)
                context = text_clean[start:end].strip()

                facts.append(ExtractedFact(
                    subject=subject[:100],
                    predicate=pattern_name,
                    object=obj[:200],
                    source=source,
                    confidence=confidence,
                    domain=domain,
                    context=context[:300],
                    timestamp=timestamp,
                ))

        return facts

    def _extract_concepts(self, text: str, timestamp: float) -> List[ExtractedConcept]:
        """Extract concepts by matching domain keywords.

        Complexity: T(n) ∈ O(n * d) for n tokens, d domain keywords.
        """
        text_lower = text.lower()
        concept_counts: Dict[str, int] = {}
        concept_domains_map: Dict[str, str] = {}
        concept_contexts: Dict[str, List[str]] = {}

        for keyword, domain in self._concept_domains.items():
            # Use word boundary matching for single words
            if " " in keyword:
                if keyword in text_lower:
                    count = text_lower.count(keyword)
                    concept_counts[keyword] = concept_counts.get(keyword, 0) + count
                    concept_domains_map[keyword] = domain
                    # Extract context
                    idx = text_lower.find(keyword)
                    if idx >= 0:
                        start = max(0, idx - 30)
                        end = min(len(text), idx + len(keyword) + 30)
                        concept_contexts.setdefault(keyword, []).append(
                            text[start:end].strip()
                        )
            else:
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                matches = pattern.findall(text)
                if matches:
                    concept_counts[keyword] = concept_counts.get(keyword, 0) + len(matches)
                    concept_domains_map[keyword] = domain
                    for m in pattern.finditer(text):
                        start = max(0, m.start() - 30)
                        end = min(len(text), m.end() + 30)
                        concept_contexts.setdefault(keyword, []).append(
                            text[start:end].strip()
                        )

        # Also extract multi-word noun phrases (simple heuristic)
        words = text_lower.split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if (words[i] not in self._stop_words and words[i + 1] not in self._stop_words
                    and len(words[i]) > 2 and len(words[i + 1]) > 2
                    and bigram not in concept_counts):
                # Check if this bigram appears multiple times (indicates a concept)
                if text_lower.count(bigram) >= 2:
                    concept_counts[bigram] = text_lower.count(bigram)
                    concept_domains_map[bigram] = self._detect_domain(bigram)
                    idx = text_lower.find(bigram)
                    start = max(0, idx - 30)
                    end = min(len(text), idx + len(bigram) + 30)
                    concept_contexts.setdefault(bigram, []).append(
                        text[start:end].strip()
                    )

        concepts: List[ExtractedConcept] = []
        for name, freq in concept_counts.items():
            # Limit context snippets to 3
            contexts = tuple(concept_contexts.get(name, [])[:3])
            concepts.append(ExtractedConcept(
                name=name,
                domain=concept_domains_map.get(name, "general"),
                frequency=freq,
                first_seen=timestamp,
                context_snippets=contexts,
            ))

        return concepts

    def _extract_relationships(
        self, concepts: List[ExtractedConcept], text: str
    ) -> List[ExtractedRelationship]:
        """Extract relationships between co-occurring concepts.

        Complexity: T(n) ∈ O(c²) for c concepts.
        """
        relationships: List[ExtractedRelationship] = []
        text_lower = text.lower()

        # If two concepts appear within 100 chars of each other, they're related
        concept_positions: Dict[str, List[int]] = {}
        for concept in concepts:
            positions = []
            start = 0
            while True:
                idx = text_lower.find(concept.name, start)
                if idx == -1:
                    break
                positions.append(idx)
                start = idx + 1
            concept_positions[concept.name] = positions

        # Find co-occurring concepts
        concept_list = list(concept_positions.keys())
        for i in range(len(concept_list)):
            for j in range(i + 1, len(concept_list)):
                c1, c2 = concept_list[i], concept_list[j]
                # Check if they appear within 100 characters
                min_dist = float('inf')
                for p1 in concept_positions[c1]:
                    for p2 in concept_positions[c2]:
                        dist = abs(p1 - p2)
                        if dist < min_dist:
                            min_dist = dist

                if min_dist < 100:
                    # Determine relationship type from text between them
                    relation = self._infer_relation_type(c1, c2, text_lower)
                    confidence = max(0.3, 1.0 - (min_dist / 100.0))
                    relationships.append(ExtractedRelationship(
                        subject=c1,
                        relation=relation,
                        object=c2,
                        confidence=confidence,
                    ))

        return relationships[:20]  # Limit to top 20 relationships

    def _infer_relation_type(self, c1: str, c2: str, text: str) -> str:
        """Infer the type of relationship between two concepts.

        Complexity: T(n) ∈ O(1).
        """
        # Find the text between the two concepts
        idx1 = text.find(c1)
        idx2 = text.find(c2)
        if idx1 == -1 or idx2 == -1:
            return "related-to"

        between = text[min(idx1, idx2) + len(c1 if idx1 < idx2 else c2):
                      max(idx1, idx2)].strip()

        if any(w in between for w in (" is ", " are ", " means ")):
            return "is-a"
        if any(w in between for w in (" uses ", " use ", " using ")):
            return "used-by"
        if any(w in between for w in (" has ", " have ", " contains ")):
            return "has"
        if any(w in between for w in (" causes ", " leads to ", " results in ")):
            return "causes"
        if any(w in between for w in (" part of ", " component of ")):
            return "part-of"
        if any(w in between for w in (" better than ", " faster than ")):
            return "better-than"
        return "related-to"

    def _detect_domain(self, text: str) -> str:
        """Detect the knowledge domain from text.

        Complexity: T(n) ∈ O(d * k) for d domains, k keywords.
        """
        text_lower = text.lower()
        domain_scores: Dict[str, int] = {}
        for keyword, domain in self._concept_domains.items():
            if keyword in text_lower:
                domain_scores[domain] = domain_scores.get(domain, 0) + 1
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return "general"

    def _fact_confidence(
        self, fact_type: str, subject: str, obj: str, text: str
    ) -> float:
        """Calculate confidence score for an extracted fact.

        Complexity: T(n) ∈ O(1).
        """
        base = {
            "definitional": 0.85,
            "causal": 0.75,
            "capability": 0.80,
            "comparative": 0.70,
        }.get(fact_type, 0.60)

        # Boost if subject is a known tech keyword
        if subject in self._tech_keywords:
            base += 0.05

        # Boost if the fact appears in a longer, more detailed text
        if len(text) > 200:
            base += 0.05

        # Penalty for very short subject or object
        if len(subject) < 3 or len(obj) < 3:
            base -= 0.10

        return min(1.0, max(0.1, base))

    def _estimate_knowledge_level(self, user_message: str, intent: str) -> float:
        """Estimate how knowledgeable the user is about the topic.

        Uses signals like: technical vocabulary, question complexity,
        code snippets, and specificity of the question.

        Complexity: T(n) ∈ O(n) for message length n.
        """
        if not user_message:
            return 0.0

        msg_lower = user_message.lower()
        score = 0.3  # Base score

        # Technical vocabulary usage
        tech_count = sum(1 for kw in self._tech_keywords if kw in msg_lower)
        score += min(0.3, tech_count * 0.03)

        # Code snippets (indicates hands-on experience)
        if "```" in user_message or "def " in user_message or "function " in user_message:
            score += 0.15

        # Specific terminology (not just generic questions)
        if any(w in msg_lower for w in ("specifically", "actually", "exactly", "precisely")):
            score += 0.05

        # Complex question structure
        if len(user_message) > 100:
            score += 0.05

        # Intent-based adjustment
        if intent == "code":
            score += 0.1
        elif intent == "architect":
            score += 0.1
        elif intent == "analyze":
            score += 0.08
        elif intent in ("greet", "identity"):
            score -= 0.1

        return min(1.0, max(0.0, score))
