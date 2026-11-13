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
Stage 06: Knowledge Fetch.

Retrieves relevant knowledge items from the KnowledgeRouter based on the
processed input and selected domain. Also queries Wikidata API for real-time
knowledge retrieval.

Mathematical Foundation:
- Top-k selection is O(n log k) for n knowledge items.
- Wikidata API: T(n) in O(query + response).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import re as _re

from core.brain.brain_state import BrainState
from core.word_identity.word_dictionary import WordDictionary

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController


class Stage06KnowledgeFetch:
    """Real knowledge retrieval stage with Wikipedia API integration."""

    __slots__ = ("_top_k", "_wikipedia_api", "_dictionary")

    def __init__(self, top_k: int = 5, use_wikidata: bool = True) -> None:
        """Initialize with a default top-k.

        Args:
            top_k: Maximum knowledge items to retrieve.
            use_wikidata: Whether to use Wikipedia API for real-time retrieval.
        """
        self._top_k = max(1, top_k)
        self._wikipedia_api = None
        self._dictionary = WordDictionary()
        if use_wikidata:
            try:
                from core.knowledge.wikipedia_api import WikipediaAPI
                self._wikipedia_api = WikipediaAPI(timeout=8.0)
            except Exception:
                pass

    def __call__(self, controller: "BrainController", state: BrainState) -> BrainState:
        """Run knowledge retrieval.

        Skips retrieval for conversational intents (greet, clarify).
        Falls back to unfiltered search if domain filter yields no results.
        Queries Wikidata API for real-time knowledge if available.

        v3.4.0 PATCH: Hard constraint — for technical intents (code, architect,
        analyze, explain with technical signals), the fallback search EXCLUDES
        the "school" domain. This prevents school curriculum chunks (Norwegian
        grammar, English past simple, "Functions in Programming") from leaking
        into technical responses about quicksort complexity, bubble_sort
        analysis, or REST API planning.

        Complexity: T(n) ∈ O(n log k + api), S(n) ∈ O(k).
        """
        # Skip knowledge fetch for conversational and emotional intents
        if state.intent in ("greet", "clarify", "emotional"):
            cm = state.context_manager if state.context_manager is not None else controller.context_manager
            cm.set("has_knowledge", False)
            cm.set("knowledge_count", 0)
            cm.set("knowledge_texts", [])
            return state.with_field(
                retrieved_knowledge=[],
                context=cm.current(),
            )

        # Identity intent: retrieve ONLY self-knowledge items, block all corpus
        if state.intent == "identity":
            # Direct scan of self-knowledge items — BM25 gets overwhelmed by
            # 11K+ corpus items, so we scan self-knowledge items directly and
            # score by term overlap with the query.
            backend = controller.knowledge_router.backend
            query_lower = state.processed_input.lower()
            query_terms = set(_re.findall(r"[a-z0-9]+", query_lower))
            _stop = {"the", "is", "a", "an", "what", "who", "how", "why",
                     "where", "when", "of", "to", "in", "for", "and", "or",
                     "are", "was", "were", "be", "do", "does", "did", "you",
                     "your", "me", "my", "i", "please", "tell"}
            query_terms -= _stop

            self_items = []
            if backend is not None:
                # Snapshot the items dict to avoid "dictionary changed size
                # during iteration" if another thread modifies the backend.
                for item in list(backend._items.values()):
                    if (item.metadata.get("source") == "condrox_self_knowledge"
                            or item.metadata.get("self") is True
                            or item.item_id.startswith("condrox-")):
                        # Score by term overlap (exact + substring match)
                        item_terms = set(_re.findall(r"[a-z0-9]+", item.content.lower()))
                        exact_overlap = len(query_terms & item_terms)
                        # Also check substring matches for unmatched terms
                        # (e.g. "llm" matches "llms", "differ" matches "differs")
                        substring_overlap = 0
                        for qt in query_terms:
                            if qt in item_terms:
                                continue  # Already counted in exact_overlap
                            if len(qt) < 3:
                                continue
                            if any(qt in t or t in qt for t in item_terms if len(t) >= 3):
                                substring_overlap += 1
                        overlap = exact_overlap + substring_overlap
                        self_items.append((overlap, item))

            # Sort by overlap score descending, take top items
            self_items.sort(key=lambda x: x[0], reverse=True)
            items = [item for _, item in self_items[:self._top_k * 2]]

            # If no self-knowledge found via direct scan, try BM25 route
            if not items:
                all_items = controller.knowledge_router.route(
                    query=state.processed_input,
                    domain=None,
                    top_k=self._top_k * 2,
                )
                items = [
                    item for item in all_items
                    if item.metadata.get("source") == "condrox_self_knowledge"
                    or item.metadata.get("self") is True
                    or item.item_id.startswith("condrox-")
                ]
            # If no self-knowledge found, try basic knowledge pack
            if not items:
                items = [
                    item for item in all_items
                    if item.metadata.get("source") == "basic_knowledge_pack"
                ]
            # Still no items — search with "general" domain as last resort
            if not items:
                items = controller.knowledge_router.route(
                    query=state.processed_input,
                    domain="general",
                    top_k=self._top_k,
                )
                items = [
                    item for item in items
                    if item.metadata.get("source") == "condrox_self_knowledge"
                    or item.item_id.startswith("condrox-")
                ]

            ids: List[str] = [item.item_id for item in items]
            knowledge_texts = [item.content for item in items]
            cm = state.context_manager if state.context_manager is not None else controller.context_manager
            cm.set("has_knowledge", bool(ids))
            cm.set("knowledge_count", len(ids))
            cm.set("knowledge_texts", knowledge_texts)
            return state.with_field(
                retrieved_knowledge=ids,
                context=cm.current(),
            )

        # v3.4.0 PATCH: Determine if this is a technical query that should
        # NEVER receive school curriculum knowledge, even via fallback.
        _technical_intents = {"code", "architect", "analyze"}
        query_lower = state.processed_input.lower()
        _technical_signals = (
            "quicksort", "mergesort", "bubble_sort", "bubble sort", "complexity",
            "o(n)", "o(log", "algorithm", "recursion", "recursive", "function",
            "def ", "python", "javascript", "api", "rest", "endpoint",
            "sort", "search", "tree", "graph", "stack", "queue",
            "implement", "debug", "refactor", "code", "plan", "design",
            "architecture", "microservice", "database", "schema",
        )
        is_technical_query = (
            state.intent in _technical_intents
            or state.mode in {"code", "architect", "analyze"}
            or any(sig in query_lower for sig in _technical_signals)
        )

        # Spell-correct the query before knowledge retrieval so misspelled
        # words like 'quntum comuting' match knowledge items for 'quantum computing'
        corrected_input = self._dictionary.correct_query(state.processed_input)
        # Also expand with synonyms for broader retrieval
        expanded_input = self._dictionary.expand_query(corrected_input)

        items = controller.knowledge_router.route(
            query=corrected_input,
            domain=state.domain,
            top_k=self._top_k,
        )

        # Fallback: if domain filter returned nothing, search all domains
        # v3.4.0 PATCH: For technical queries, exclude "school" domain from fallback
        if not items:
            if is_technical_query:
                # Search all domains EXCEPT school — prevents grammar/curriculum
                # chunks from leaking into technical responses
                items = controller.knowledge_router.route(
                    query=expanded_input,
                    domain=None,
                    top_k=self._top_k,
                )
                # Filter out school-domain items from fallback results
                items = [
                    item for item in items
                    if item.domain != "school"
                    and item.metadata.get("source") != "school_curriculum"
                ]
            else:
                items = controller.knowledge_router.route(
                    query=expanded_input,
                    domain=None,
                    top_k=self._top_k,
                )

        # v3.4.0 PATCH: Even in primary results, filter out school items for
        # technical queries (in case the domain router routed to "school")
        if is_technical_query:
            items = [
                item for item in items
                if item.domain != "school"
                and item.metadata.get("source") != "school_curriculum"
            ]

        # Filter out self-knowledge items for non-identity queries
        # — prevents Condrox self-descriptions from appearing in factual answers
        # BUT keep them for self-referential queries (e.g. "how was you made?")
        # AND for queries mentioning Condrox-related terms (tobias, Østen, condrox)
        _self_ref_signals = (
            "you", "your", "yourself", "condrox", "your system",
            "your pipeline", "your knowledge", "your architecture",
            "you made", "you built", "you developed", "you created",
            "you trained", "you work", "you differ", "you think",
            "you feel", "you know", "you learn",
            "tobias", "østen", "ostne", "norway",
        )
        _self_ref = any(
            sig in state.processed_input.lower()
            for sig in _self_ref_signals
        )
        if state.intent != "identity" and not _self_ref:
            items = [
                item for item in items
                if item.metadata.get("source") != "condrox_self_knowledge"
                and not item.item_id.startswith("condrox-")
            ]

        # If query mentions Condrox-related terms but self-knowledge wasn't
        # found via BM25 (overwhelmed by corpus), do a direct scan
        if _self_ref and state.intent != "identity":
            has_self_knowledge = any(
                item.metadata.get("source") == "condrox_self_knowledge"
                or item.item_id.startswith("condrox-")
                for item in items
            )
            if not has_self_knowledge:
                backend = controller.knowledge_router.backend
                if backend is not None:
                    query_terms_sk = set(_re.findall(r"[a-z0-9]+", state.processed_input.lower()))
                    _stop_sk = {"the", "is", "a", "an", "what", "who", "how", "why",
                             "where", "when", "of", "to", "in", "for", "and", "or",
                             "are", "was", "were", "be", "do", "does", "did", "you",
                             "your", "me", "my", "i", "please", "tell"}
                    query_terms_sk -= _stop_sk
                    self_items_sk = []
                    for item in list(backend._items.values()):
                        if (item.metadata.get("source") == "condrox_self_knowledge"
                                or item.item_id.startswith("condrox-")):
                            item_terms = set(_re.findall(r"[a-z0-9]+", item.content.lower()))
                            exact_overlap = len(query_terms_sk & item_terms)
                            substring_overlap = 0
                            for qt in query_terms_sk:
                                if qt in item_terms:
                                    continue
                                if len(qt) < 3:
                                    continue
                                if any(qt in t or t in qt for t in item_terms if len(t) >= 3):
                                    substring_overlap += 1
                            overlap = exact_overlap + substring_overlap
                            if overlap > 0:
                                self_items_sk.append((overlap, item))
                    self_items_sk.sort(key=lambda x: x[0], reverse=True)
                    for _, item in self_items_sk[:3]:
                        if item.item_id not in [i.item_id for i in items]:
                            items.insert(0, item)

        # Filter out meta.json items — they contain ingestion error messages
        # and metadata, not actual knowledge content
        items = [
            item for item in items
            if item.metadata.get("source") != "meta.json"
        ]

        # Filter out raw dataset sources for factual questions
        # — these are conversation logs, math problems, quiz questions,
        # grammar exercises, etc. — not encyclopedic knowledge.
        # Only allow curated sources (basic_knowledge_pack, wikipedia, wikidata).
        _RAW_DATASET_SOURCES = {
            "empathetic_dialogues_v1.json", "ethics_v1.json",
            "europarl_v1.txt", "opensubtitles_v1.txt",
            "dailog_v1.txt", "multiwoz_v1.json",
            "personachat_v1.json", "social_customer_support_v1.txt",
            "news_commentary_v1.txt", "reddit_v1.txt",
            "gsm8k_v1.json", "arc_v1.json",
            "school_curriculum", "meta.json",
        }
        if state.intent in ("question", "explain", "unknown"):
            items = [
                item for item in items
                if item.metadata.get("source") not in _RAW_DATASET_SOURCES
            ]

        # Filter out low-quality knowledge items
        filtered_items = []
        for item in items:
            # Skip items that are too short
            if len(item.content) < 20:
                continue
            # Skip items with dataset-specific jargon for general queries
            # — but ONLY for non-basic-knowledge-pack items (the pack is curated)
            is_curated = (
                item.metadata.get("source") == "basic_knowledge_pack"
                or item.metadata.get("source") == "condrox_self_knowledge"
                or item.metadata.get("source") == "wikipedia_learned"
                or item.metadata.get("source") == "wikidata_learned"
            )
            if not is_curated and state.intent in ("question", "explain") and any(
                word in item.content.lower()
                for word in ["shader", "parm", "typecode"]
            ):
                if not any(word in state.processed_input.lower() for word in ["shader", "parm"]):
                    continue
            # ─── Claim filter rules (v1.0 basic knowledge pack) ───────────
            if _should_filter_claim(item.content, state.domain, state.intent):
                continue
            # Identity intent: only allow self-knowledge items through
            if state.intent == "identity":
                if (item.metadata.get("source") != "condrox_self_knowledge"
                    and not item.item_id.startswith("condrox-")
                    and item.metadata.get("source") != "basic_knowledge_pack"):
                    continue
            filtered_items.append(item)
        
        # Try Wikipedia API for real-time knowledge if no good local results
        # But skip for personal/conversational statements (e.g. "just got a new job")
        _query_lower_early = state.processed_input.lower().strip()
        _is_personal_early = any(
            _query_lower_early.startswith(p) for p in (
                "i ", "i'm", "im", "i am", "i've", "ive", "i'd",
                "just got", "just finished", "just started",
                "my ", "me ", "we ", "today ", "yesterday ",
            )
        )
        _is_conversational_early = (
            len(_query_lower_early) < 15
            and "?" not in _query_lower_early
            and not any(w in _query_lower_early for w in (
                "what", "who", "when", "where", "why", "how",
                "explain", "define", "describe", "tell me about",
            ))
        )
        if self._wikipedia_api and len(filtered_items) < 2 and not _is_personal_early and not _is_conversational_early:
            try:
                # Spell-correct the query before calling Wikipedia
                # so 'quntum computing' becomes 'quantum computing'
                corrected_query = self._dictionary.correct_query(state.processed_input)
                if not corrected_query:
                    corrected_query = state.processed_input

                # Use synchronous Wikipedia API
                result = self._wikipedia_api.search_and_summarize(corrected_query)
                if result and result.get("summary"):
                    content = result["summary"]
                    if len(content) >= 30:
                        from core.knowledge.knowledge_models import KnowledgeItem
                        import hashlib
                        temp_id = f"wiki_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
                        temp_item = KnowledgeItem(
                            item_id=temp_id,
                            domain="wikipedia",
                            content=content,
                            embedding=None,
                            confidence=0.9,
                            metadata={
                                "source": "wikipedia_api",
                                "wiki_title": result.get("title", ""),
                            },
                        )
                        filtered_items.append(temp_item)
                        # Learn from Wikipedia: persist the fact
                        try:
                            controller.knowledge_router.ingest(
                                content=content,
                                domain="wikipedia",
                                item_id=temp_id,
                                confidence=0.9,
                                metadata={
                                    "source": "wikipedia_learned",
                                    "wiki_title": result.get("title", ""),
                                    "learned": True,
                                },
                            )
                        except Exception:
                            pass
            except Exception:
                pass  # Silently fail on API errors

        # Wikipedia API fallback — if no local knowledge found OR the query
        # asks who/when/where questions that the basic knowledge pack can't answer
        # (it has definitions, not biographical/historical facts)
        # Also trigger if filtered items are from raw datasets (not curated)
        _curated_sources = {
            "basic_knowledge_pack", "condrox_self_knowledge",
            "wikipedia_learned", "wikidata_learned", "wikipedia_api",
        }
        _has_curated = any(
            item.metadata.get("source") in _curated_sources
            for item in filtered_items
        )
        _needs_wikipedia = len(filtered_items) == 0 or (
            not _has_curated and state.intent in ("question", "explain", "unknown")
        )

        # Self-referential queries should never go to Wikipedia — they need
        # self-knowledge, not encyclopedic content about unrelated topics
        _is_self_referential = _self_ref
        if _is_self_referential:
            _needs_wikipedia = False

        if not _needs_wikipedia and state.intent in ("question", "explain", "unknown"):
            query_lower_wiki = state.processed_input.lower()
            # who/when/where/invented questions need encyclopedic knowledge
            # But skip if the query is about Condrox itself (self-referential)
            if not _is_self_referential and any(w in query_lower_wiki for w in (
                "who made", "who invented", "who created", "who discovered",
                "who wrote", "who designed", "who developed",
                "when was", "when did", "when was the first",
                "where was", "where did", "where is the first",
                "invented", "discovered", "founded by", "created by",
                "history of", "origin of", "origins of",
            )):
                _needs_wikipedia = True

        if _needs_wikipedia and state.intent in ("question", "explain", "unknown"):
            # Don't call Wikipedia for personal/conversational statements.
            # "just got a new job" is a personal statement, not a factual query.
            # Only call Wikipedia if the query looks like a factual question.
            _query_lower = state.processed_input.lower().strip()
            _is_personal_statement = any(
                _query_lower.startswith(p) for p in (
                    "i ", "i'm", "im", "i am", "i've", "ive", "i'd",
                    "just got", "just finished", "just started",
                    "my ", "me ", "we ", "today ", "yesterday ",
                )
            )
            # Also skip if it's a short conversational phrase (< 15 chars, no question mark)
            _is_conversational = (
                len(_query_lower) < 15
                and "?" not in _query_lower
                and not any(w in _query_lower for w in (
                    "what", "who", "when", "where", "why", "how",
                    "explain", "define", "describe", "tell me about",
                ))
            )
            if _is_personal_statement or _is_conversational:
                _needs_wikipedia = False

        if _needs_wikipedia and state.intent in ("question", "explain", "unknown"):
            try:
                from core.knowledge.wikipedia_bridge import fetch_wikipedia_summary
                from core.knowledge.knowledge_models import KnowledgeItem
                import hashlib

                wiki_result = fetch_wikipedia_summary(state.processed_input)
                if wiki_result and wiki_result.found and wiki_result.extract:
                    content = wiki_result.extract
                    temp_id = f"wiki_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
                    temp_item = KnowledgeItem(
                        item_id=temp_id,
                        domain=state.domain or "general",
                        content=content,
                        embedding=None,
                        confidence=0.85,
                        metadata={
                            "source": "wikipedia_api",
                            "wiki_title": wiki_result.title,
                            "wiki_url": wiki_result.url,
                        },
                    )
                    # Wikipedia replaces generic BM25 items for who/when/where
                    filtered_items = [temp_item]

                    # ─── Learn from Wikipedia: persist the fact ─────────────
                    # Ingest the Wikipedia fact into the knowledge backend so
                    # future queries on the same topic find it via BM25 without
                    # needing another API call. Only store if it looks like a
                    # real factual extract (not a disambiguation or error page).
                    _is_fact = (
                        len(content) >= 50
                        and not content.lower().startswith("may refer to")
                        and not content.lower().startswith("this is a redirect")
                        and "disambiguation" not in content.lower()
                    )
                    if _is_fact:
                        try:
                            controller.knowledge_router.ingest(
                                content=content,
                                domain=state.domain or "general",
                                item_id=temp_id,
                                confidence=0.85,
                                metadata={
                                    "source": "wikipedia_learned",
                                    "wiki_title": wiki_result.title,
                                    "wiki_url": wiki_result.url,
                                    "learned": True,
                                },
                            )
                        except Exception:
                            pass  # Ingestion failure is non-critical
            except Exception:
                pass  # Silently fail — Wikipedia is a fallback, not critical

        ids: List[str] = [item.item_id for item in filtered_items]
        knowledge_texts = [item.content for item in filtered_items]

        cm = state.context_manager if state.context_manager is not None else controller.context_manager
        cm.set("has_knowledge", bool(ids))
        cm.set("knowledge_count", len(ids))
        cm.set("knowledge_texts", knowledge_texts)

        return state.with_field(
            retrieved_knowledge=ids,
            context=cm.current(),
        )


# ─── Claim Filter Rules (Basic Knowledge Pack v1.0) ──────────────────────

# Patterns that indicate dataset noise — content that should never appear
# in responses regardless of domain. Each pattern is a substring match on
# lowercased content. T(n) ∈ O(p * c) for p patterns, c content length.

_EU_LAW_PATTERNS: tuple[str, ...] = (
    "article 1",
    "article 2",
    "article 3",
    "directive eu",
    "eu directive",
    "european parliament",
    "council of the european union",
    "official journal of the european union",
    "member states shall",
    "this regulation shall enter into force",
    "commission delegated regulation",
    "pursuant to article",
)

_IFRS_VOICE_PATTERNS: tuple[str, ...] = (
    "ifrs foundation",
    "international accounting standards board",
    "iasb",
    "the objective of this ifrs",
    "this ifrs requires",
    "an entity shall",
    "in accordance with ifrs",
    "international financial reporting standard",
)

_QUIZ_PATTERNS: tuple[str, ...] = (
    "which of the following",
    "multiple choice",
    "select the correct",
    "choose the best answer",
    "true or false",
    "fill in the blank",
    "match the following",
    "what is the answer to question",
    "quiz question",
    "test your knowledge",
)

_POLITICAL_PATTERNS: tuple[str, ...] = (
    "political party",
    "election campaign",
    "vote for",
    "candidate for",
    "parliamentary election",
    "government policy on",
    "political ideology",
    "supervisory bodies",
    "member states",
    "latvia's economy",
    "structurally transformed",
    "bureaucracy and more controls",
    "transparency, which is closely linked",
)

_DATASET_NOISE_PATTERNS: tuple[str, ...] = (
    "click here to download",
    "table of contents",
    "page 1 of",
    "loading...",
    "copyright ©",
    "all rights reserved",
    "this page was last edited",
    "citation needed",
    "[edit]",
    "disambiguation",
    "look at this",
    "more and more urgent",
    "work more closely and more constructively",
    "trust each other more",
)

_GENERIC_TEMPLATE_PATTERNS: tuple[str, ...] = (
    "the answer to your question is",
    "based on the information provided",
    "according to my knowledge",
    "in my opinion",
    "i would say that",
)


def _should_filter_claim(content: str, domain: str, intent: str) -> bool:
    """Check if a knowledge item should be filtered out.

    Applies 20 claim filter rules to block EU law texts, IFRS voice texts,
    quiz questions, political texts, dataset noise, and generic templates
    when the domain is known.

    Args:
        content: The knowledge item content text.
        domain: The current domain classification.
        intent: The current intent classification.

    Returns:
        True if the item should be filtered out, False otherwise.

    Complexity: T(n) ∈ O(p * c) for p patterns, c content length.
    """
    lower = content.lower()

    # Rules 1-12: Filter EU law texts
    for pattern in _EU_LAW_PATTERNS:
        if pattern in lower:
            return True

    # Rules 13-20: Filter IFRS voice texts (unless domain is accounting)
    if domain != "accounting":
        for pattern in _IFRS_VOICE_PATTERNS:
            if pattern in lower:
                return True

    # Filter quiz questions
    for pattern in _QUIZ_PATTERNS:
        if pattern in lower:
            return True

    # Filter political texts
    for pattern in _POLITICAL_PATTERNS:
        if pattern in lower:
            return True

    # Filter dataset noise
    for pattern in _DATASET_NOISE_PATTERNS:
        if pattern in lower:
            return True

    # Filter generic templates when domain is known
    if domain != "general" and intent in ("question", "explain"):
        for pattern in _GENERIC_TEMPLATE_PATTERNS:
            if pattern in lower:
                return True

    return False
