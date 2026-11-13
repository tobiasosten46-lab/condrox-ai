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

"""Knowledge Validator — validates extracted knowledge for relevance and accuracy.

Every piece of knowledge extracted from conversations goes through validation
before being stored. The validator checks:
- Relevance: Is this knowledge relevant to the user's question?
- Accuracy: Does this contradict existing knowledge?
- Importance: How valuable is this knowledge?
- Novelty: Is this new information or a duplicate?
- Quality: Is the source reliable (AI response vs user claim)?

The validator does NOT discard knowledge — it flags it with a quality score.
Even irrelevant data is stored with a low score, because the AI remembers
everything but only uses high-quality knowledge for factual answers.

Complexity:
    validate: T(n) ∈ O(f * e) for f facts, e existing items
    check_contradiction: T(n) ∈ O(1) per fact pair

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.autonomous.knowledge_extractor import ExtractedFact, ExtractedConcept, ExtractedRelationship


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating a single fact.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    is_relevant: bool          # Relevant to the current conversation?
    is_accurate: bool          # Consistent with existing knowledge?
    is_novel: bool             # New information (not a duplicate)?
    importance: float          # How important [0.0, 1.0]
    quality: float             # Overall quality score [0.0, 1.0]
    contradictions: Tuple[str, ...] = ()  # IDs of contradicting facts
    duplicates: Tuple[str, ...] = ()      # IDs of duplicate facts
    reason: str = ""           # Human-readable validation summary


@dataclass(slots=True)
class KnowledgeStore:
    """In-memory knowledge store for the autonomous learning system.

    Stores validated facts, concepts, and relationships with quality scores.
    Thread-unsafe — the AutonomousKnowledgeEngine handles synchronization.

    Complexity: T(n) ∈ O(1) for add, O(n) for search.
    """

    _facts: Dict[str, ExtractedFact] = field(default_factory=dict)
    _fact_quality: Dict[str, float] = field(default_factory=dict)
    _fact_importance: Dict[str, float] = field(default_factory=dict)
    _concepts: Dict[str, ExtractedConcept] = field(default_factory=dict)
    _concept_frequency: Dict[str, int] = field(default_factory=dict)
    _relationships: List[ExtractedRelationship] = field(default_factory=list)
    _subject_index: Dict[str, Set[str]] = field(default_factory=dict)  # subject → fact IDs
    _domain_index: Dict[str, Set[str]] = field(default_factory=dict)  # domain → fact IDs
    _total_items: int = 0
    _last_updated: float = field(default_factory=time.time)

    def add_fact(self, fact_id: str, fact: ExtractedFact, quality: float, importance: float) -> None:
        """Add a validated fact to the store.

        Complexity: T(n) ∈ O(1).
        """
        self._facts[fact_id] = fact
        self._fact_quality[fact_id] = quality
        self._fact_importance[fact_id] = importance
        self._subject_index.setdefault(fact.subject, set()).add(fact_id)
        self._domain_index.setdefault(fact.domain, set()).add(fact_id)
        self._total_items += 1
        self._last_updated = time.time()

    def add_concept(self, concept: ExtractedConcept) -> None:
        """Add or update a concept.

        Complexity: T(n) ∈ O(1).
        """
        if concept.name in self._concepts:
            # Update frequency — keep the higher one
            existing = self._concepts[concept.name]
            new_freq = existing.frequency + concept.frequency
            self._concepts[concept.name] = ExtractedConcept(
                name=concept.name,
                domain=concept.domain,
                frequency=new_freq,
                first_seen=existing.first_seen,
                context_snippets=existing.context_snippets + concept.context_snippets,
            )
            self._concept_frequency[concept.name] = new_freq
        else:
            self._concepts[concept.name] = concept
            self._concept_frequency[concept.name] = concept.frequency
        self._last_updated = time.time()

    def add_relationship(self, rel: ExtractedRelationship) -> None:
        """Add a relationship if it doesn't already exist.

        Complexity: T(n) ∈ O(r) for r existing relationships.
        """
        for existing in self._relationships:
            if (existing.subject == rel.subject and
                existing.relation == rel.relation and
                existing.object == rel.object):
                # Already exists — skip
                return
        self._relationships.append(rel)
        self._last_updated = time.time()

    def find_by_subject(self, subject: str) -> List[Tuple[ExtractedFact, float, float]]:
        """Find all facts about a subject.

        Returns list of (fact, quality, importance) tuples.
        Sorted by importance descending.

        Complexity: T(n) ∈ O(f log f) for f matching facts.
        """
        fact_ids = self._subject_index.get(subject.lower(), set())
        results = []
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact:
                results.append((fact, self._fact_quality.get(fid, 0.0), self._fact_importance.get(fid, 0.0)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def find_by_domain(self, domain: str, limit: int = 20) -> List[Tuple[ExtractedFact, float, float]]:
        """Find facts in a knowledge domain.

        Complexity: T(n) ∈ O(f log f) for f matching facts.
        """
        fact_ids = self._domain_index.get(domain, set())
        results = []
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact:
                results.append((fact, self._fact_quality.get(fid, 0.0), self._fact_importance.get(fid, 0.0)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    def find_duplicates(self, fact: ExtractedFact) -> List[str]:
        """Find duplicate facts in the store.

        Complexity: T(n) ∈ O(f) for f facts with same subject.

        A duplicate has the same subject, predicate, and object.
        """
        fact_ids = self._subject_index.get(fact.subject, set())
        duplicates = []
        for fid in fact_ids:
            existing = self._facts.get(fid)
            if existing and existing.predicate == fact.predicate and existing.object == fact.object:
                duplicates.append(fid)
        return duplicates

    def find_contradictions(self, fact: ExtractedFact) -> List[str]:
        """Find facts that contradict the given fact.

        Complexity: T(n) ∈ O(f) for f facts with same subject.

        A contradiction has the same subject and predicate but different object.
        """
        fact_ids = self._subject_index.get(fact.subject, set())
        contradictions = []
        for fid in fact_ids:
            existing = self._facts.get(fid)
            if (existing and existing.predicate == fact.predicate
                    and existing.object != fact.object):
                contradictions.append(fid)
        return contradictions

    def get_top_concepts(self, n: int = 10) -> List[ExtractedConcept]:
        """Get the top-N most frequently mentioned concepts.

        Complexity: T(n) ∈ O(c log c) for c concepts.
        """
        return sorted(self._concepts.values(), key=lambda c: c.frequency, reverse=True)[:n]

    def get_relationships_for(self, concept: str) -> List[ExtractedRelationship]:
        """Get all relationships involving a concept.

        Complexity: T(n) ∈ O(r) for r relationships.
        """
        concept_lower = concept.lower()
        return [
            r for r in self._relationships
            if r.subject == concept_lower or r.object == concept_lower
        ]

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    @property
    def concept_count(self) -> int:
        return len(self._concepts)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)

    @property
    def total_items(self) -> int:
        return self._total_items

    @property
    def last_updated(self) -> float:
        return self._last_updated


class KnowledgeValidator:
    """Validates extracted knowledge for storage.

    Every fact goes through a multi-dimensional validation:
    1. Relevance — is it related to what the user asked?
    2. Accuracy — does it contradict existing knowledge?
    3. Novelty — is it new information?
    4. Importance — how valuable is this knowledge?
    5. Quality — overall quality score

    The validator NEVER discards knowledge. Even low-quality facts are stored
    with a low score. The AI remembers everything but only uses high-quality
    knowledge for factual answers.

    Complexity:
        validate_fact: T(n) ∈ O(f) for f existing facts with same subject
        validate_batch: T(n) ∈ O(n * f) for n facts
    """

    __slots__ = ("_store",)

    # Source reliability weights
    _SOURCE_RELIABILITY: Dict[str, float] = {
        "assistant": 0.85,  # AI-generated content (validated by pipeline)
        "user": 0.50,       # User claims (unverified)
        "system": 0.95,     # System knowledge (pre-loaded)
    }

    # Domain importance weights — some domains are more valuable
    _DOMAIN_IMPORTANCE: Dict[str, float] = {
        "programming": 0.90,
        "ai_ml": 0.90,
        "security": 0.85,
        "science": 0.80,
        "technology": 0.80,
        "finance": 0.70,
        "general": 0.50,
    }

    def __init__(self, store: KnowledgeStore) -> None:
        """Initialize the validator with a knowledge store.

        Args:
            store: The KnowledgeStore to check against and add to.
        """
        self._store = store

    def validate_fact(
        self,
        fact: ExtractedFact,
        conversation_intent: str = "unknown",
        conversation_topics: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate a single extracted fact.

        Args:
            fact: The fact to validate.
            conversation_intent: The intent of the conversation turn.
            conversation_topics: Topics discussed in this turn.

        Returns:
            ValidationResult with all validation dimensions.

        Complexity: T(n) ∈ O(f) for f existing facts with same subject.
        """
        topics = conversation_topics or []

        # 1. Check for duplicates
        duplicates = self._store.find_duplicates(fact)
        is_novel = len(duplicates) == 0

        # 2. Check for contradictions
        contradictions = self._store.find_contradictions(fact)
        is_accurate = len(contradictions) == 0

        # 3. Check relevance to conversation
        is_relevant = self._check_relevance(fact, conversation_intent, topics)

        # 4. Calculate importance
        importance = self._calculate_importance(fact, is_novel, is_relevant)

        # 5. Calculate quality
        quality = self._calculate_quality(fact, is_accurate, is_novel, is_relevant)

        # 6. Generate reason
        reason = self._generate_reason(is_relevant, is_accurate, is_novel, importance, quality)

        return ValidationResult(
            is_relevant=is_relevant,
            is_accurate=is_accurate,
            is_novel=is_novel,
            importance=importance,
            quality=quality,
            contradictions=tuple(contradictions),
            duplicates=tuple(duplicates),
            reason=reason,
        )

    def validate_and_store(
        self,
        fact: ExtractedFact,
        conversation_intent: str = "unknown",
        conversation_topics: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate a fact and store it if it passes quality threshold.

        Even low-quality facts are stored — the quality score determines
        whether they're used for factual answers, not whether they're stored.

        Complexity: T(n) ∈ O(f) for f existing facts.
        """
        result = self.validate_fact(fact, conversation_intent, conversation_topics)

        # Generate a unique fact ID
        fact_id = f"fact_{abs(hash((fact.subject, fact.predicate, fact.object, fact.timestamp)))}"

        # Always store — even duplicates update frequency
        if result.duplicates:
            # Update the existing fact's importance (reinforcement)
            for dup_id in result.duplicates:
                existing_imp = self._store._fact_importance.get(dup_id, 0.0)
                # Reinforcement increases importance slightly
                self._store._fact_importance[dup_id] = min(1.0, existing_imp + 0.05)
        else:
            # Store new fact
            self._store.add_fact(fact_id, fact, result.quality, result.importance)

        return result

    def _check_relevance(
        self, fact: ExtractedFact, intent: str, topics: List[str]
    ) -> bool:
        """Check if a fact is relevant to the conversation.

        Complexity: T(n) ∈ O(t) for t topics.
        """
        # Facts about topics being discussed are relevant
        for topic in topics:
            if topic in fact.subject or topic in fact.object:
                return True
            if fact.subject in topic or fact.object in topic:
                return True

        # Facts from code/analyze/architect intents are always relevant
        if intent in ("code", "analyze", "architect", "explain"):
            return True

        # Facts from emotional/greet intents are less relevant (but still stored)
        if intent in ("greet", "emotional"):
            return False

        # Default: relevant if it has a known domain
        return fact.domain != "general"

    def _calculate_importance(
        self, fact: ExtractedFact, is_novel: bool, is_relevant: bool
    ) -> float:
        """Calculate the importance score for a fact.

        Complexity: T(n) ∈ O(1).
        """
        # Base importance from domain
        base = self._DOMAIN_IMPORTANCE.get(fact.domain, 0.50)

        # Novelty boost — new knowledge is more important
        if is_novel:
            base += 0.15

        # Relevance boost — relevant knowledge is more important
        if is_relevant:
            base += 0.10

        # Confidence from extraction
        base *= (0.5 + fact.confidence * 0.5)

        # Source reliability
        source_weight = self._SOURCE_RELIABILITY.get(fact.source, 0.50)
        base *= (0.6 + source_weight * 0.4)

        return min(1.0, max(0.05, base))

    def _calculate_quality(
        self, fact: ExtractedFact, is_accurate: bool, is_novel: bool, is_relevant: bool
    ) -> float:
        """Calculate the overall quality score.

        Complexity: T(n) ∈ O(1).
        """
        quality = 0.5  # Base

        # Accuracy is critical
        if is_accurate:
            quality += 0.25
        else:
            quality -= 0.20  # Contradictions lower quality

        # Novelty adds value
        if is_novel:
            quality += 0.10

        # Relevance adds value
        if is_relevant:
            quality += 0.10

        # Source reliability
        source_weight = self._SOURCE_RELIABILITY.get(fact.source, 0.50)
        quality = quality * (0.7 + source_weight * 0.3)

        # Extraction confidence
        quality = quality * (0.6 + fact.confidence * 0.4)

        return min(1.0, max(0.01, quality))

    def _generate_reason(
        self, is_relevant: bool, is_accurate: bool, is_novel: bool,
        importance: float, quality: float,
    ) -> str:
        """Generate a human-readable validation reason.

        Complexity: T(n) ∈ O(1).
        """
        parts = []
        if is_novel:
            parts.append("novel")
        else:
            parts.append("duplicate")
        if is_accurate:
            parts.append("consistent")
        else:
            parts.append("contradicts existing")
        if is_relevant:
            parts.append("relevant")
        else:
            parts.append("tangential")
        parts.append(f"importance={importance:.2f}")
        parts.append(f"quality={quality:.2f}")
        return ", ".join(parts)
