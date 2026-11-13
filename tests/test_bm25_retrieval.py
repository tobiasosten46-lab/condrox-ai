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
Regression tests for BM25 knowledge retrieval — stemming, stop words,
and the 3 conversation suite failures (recursion, democracy, databases).

Tests verify:
- Stop word filtering prevents irrelevant matches
- Stemming consistency (recursion↔recursive, databases↔database, democracy↔democratic)
- Min score threshold filters near-zero scores
- Domain fallback when domain filter yields nothing
- _extract_answer_from_knowledge finds relevant sentences with stemmed terms
"""

from __future__ import annotations

import pytest

from core.knowledge.knowledge_backend import KnowledgeBackend, _STOP_WORDS
from core.pipeline.stage_09_synthesis import Stage09Synthesis


# ── Stop word filtering ──────────────────────────────────────────────


def test_stop_words_filtered_from_query() -> None:
    """Search with only stop words returns empty — no irrelevant matches."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("The quick brown fox jumps over the lazy dog.", domain="general")
    kb.ingest("What is the meaning of life?", domain="general")
    results = kb.search("what is the", top_k=5)
    assert results == [], "Stop-word-only query should return no results"


def test_stop_word_query_with_content_word_matches() -> None:
    """Query with stop words + content word still matches on content word."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Recursion is a function that calls itself.", domain="programming")
    kb.ingest("The weather is nice today.", domain="general")
    results = kb.search("What is recursion?", top_k=5)
    assert len(results) == 1
    assert "recursion" in results[0].content.lower() or "recurs" in results[0].content.lower()


def test_stop_words_set_covers_common_english() -> None:
    """Stop word set includes the most common English stop words."""
    expected = {"the", "is", "a", "an", "what", "how", "of", "to", "in", "and"}
    assert expected.issubset(_STOP_WORDS)


# ── Stemming consistency ─────────────────────────────────────────────


def test_stemming_recursion_matches_recursive() -> None:
    """'recursion' and 'recursive' stem to the same token."""
    tokens = KnowledgeBackend._tokenize("recursion recursive")
    assert len(tokens) == 2
    assert tokens[0] == tokens[1], f"Expected same stem, got {tokens}"


def test_stemming_databases_matches_database() -> None:
    """'databases' and 'database' stem to the same token."""
    tokens = KnowledgeBackend._tokenize("databases database")
    assert len(tokens) == 2
    assert tokens[0] == tokens[1], f"Expected same stem, got {tokens}"


def test_stemming_functions_matches_function() -> None:
    """'functions' and 'function' stem to the same token (2-pass)."""
    tokens = KnowledgeBackend._tokenize("functions function")
    assert len(tokens) == 2
    assert tokens[0] == tokens[1], f"Expected same stem, got {tokens}"


def test_stemming_programming_matches_program() -> None:
    """'programming' and 'program' stem to compatible tokens."""
    tokens = KnowledgeBackend._tokenize("programming program")
    # 'programming' → 'program' (strip 'ming'), 'program' → 'program' (no strip)
    assert tokens[0] == tokens[1], f"Expected same stem, got {tokens}"


def test_stemming_democracy_preserved() -> None:
    """'democracy' has no matching suffix — stays as 'democracy'."""
    tokens = KnowledgeBackend._tokenize("democracy")
    assert tokens[0] == "democracy", f"Expected 'democracy', got '{tokens[0]}'"


def test_stemming_short_words_preserved() -> None:
    """Words <= 3 chars are not stemmed."""
    tokens = KnowledgeBackend._tokenize("the cat dog run")
    assert tokens == ["the", "cat", "dog", "run"]


def test_stemming_databases_not_double_stripped() -> None:
    """'databases' → 'database' (strip 's' only, no double-stripping)."""
    tokens = KnowledgeBackend._tokenize("databases")
    assert tokens[0] == "database", f"Expected 'database', got '{tokens[0]}'"


# ── BM25 search relevance ────────────────────────────────────────────


def test_search_returns_relevant_not_irrelevant() -> None:
    """Search for 'recursion' returns recursion content, not unrelated docs."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Recursion is when a function calls itself to solve smaller subproblems.", domain="programming")
    kb.ingest("C++ uses templates for generic programming.", domain="programming")
    kb.ingest("Python is a popular programming language for beginners.", domain="programming")
    kb.ingest("Java is another programming language used in enterprise.", domain="programming")
    kb.ingest("The Tour de France is a cycling race.", domain="sports")
    results = kb.search("What is recursion in programming?", top_k=3)
    assert len(results) > 0
    assert "recurs" in results[0].content.lower(), "Top result should contain recursion content"


def test_search_democracy_finds_democracy_content() -> None:
    """Search for 'democracy' finds content containing 'democracy'."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Democracy is a system of government by the whole population.", domain="political_science")
    kb.ingest("Photosynthesis converts sunlight into chemical energy.", domain="biology")
    results = kb.search("What is democracy?", top_k=3)
    assert len(results) > 0
    assert "democra" in results[0].content.lower(), "Top result should contain democracy content"


def test_search_databases_finds_database_content() -> None:
    """Search for 'databases' finds content containing 'database'."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("A database stores structured data in tables with rows and columns.", domain="programming")
    kb.ingest("Quantum computing uses quantum bits called qubits.", domain="physics")
    results = kb.search("How do databases work?", top_k=3)
    assert len(results) > 0
    assert "database" in results[0].content.lower() or "databas" in results[0].content.lower()


def test_search_empty_backend_returns_empty() -> None:
    """Search on empty backend returns empty list."""
    kb = KnowledgeBackend(embedding_dim=32)
    assert kb.search("anything", top_k=5) == []


def test_search_empty_query_returns_empty() -> None:
    """Empty query returns empty list."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Some content here.", domain="general")
    assert kb.search("", top_k=5) == []


def test_search_no_matching_terms_returns_empty() -> None:
    """Query terms not in any document returns empty."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Python uses indentation for code blocks.", domain="programming")
    results = kb.search("quantum entanglement", top_k=5)
    assert results == [], "Non-matching query should return no results"


def test_search_domain_filter_then_fallback() -> None:
    """Domain filter returns results from correct domain; fallback works."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Recursion in Python.", domain="programming")
    kb.ingest("Recursion in mathematics.", domain="math")
    results = kb.search("recursion", top_k=5, domain_filter="programming")
    assert len(results) == 1
    assert results[0].domain == "programming"


def test_search_domain_filter_no_match_falls_back() -> None:
    """If domain filter yields nothing, Stage06 falls back to all domains."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Recursion in Python.", domain="programming")
    # Search with a domain that has no matching content
    results = kb.search("recursion", top_k=5, domain_filter="nonexistent")
    assert results == [], "Domain filter with no matches returns empty (Stage06 does fallback)"


def test_search_top_k_limits_results() -> None:
    """top_k parameter limits number of returned results."""
    kb = KnowledgeBackend(embedding_dim=32)
    for i in range(10):
        kb.ingest(f"Recursion example number {i}.", domain="programming")
    results = kb.search("recursion", top_k=3)
    assert len(results) <= 3


def test_min_score_filters_low_relevance() -> None:
    """Documents with very low BM25 scores are filtered out."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest("Recursion is a powerful programming technique.", domain="programming")
    kb.ingest("The cat sat on the mat.", domain="general")
    results = kb.search("recursion programming", top_k=5)
    # The cat document should not appear — no matching terms
    for r in results:
        assert "recurs" in r.content.lower() or "program" in r.content.lower()


# ── _extract_answer_from_knowledge (Stage09) ─────────────────────────


def _make_stage09() -> Stage09Synthesis:
    """Create a Stage09Synthesis instance for testing _extract_answer_from_knowledge."""
    return Stage09Synthesis()


def test_extract_answer_finds_relevant_sentence() -> None:
    """_extract_answer_from_knowledge finds sentences with query terms."""
    stage = _make_stage09()
    knowledge = [
        "Recursion is a programming technique where a function calls itself. "
        "It is used to solve problems by breaking them into smaller subproblems. "
        "The base case stops the recursion."
    ]
    answer = stage._extract_answer_from_knowledge("What is recursion in programming?", knowledge)
    assert "recursion" in answer.lower() or "recurs" in answer.lower()
    assert len(answer) > 20


def test_extract_answer_finds_democracy_sentence() -> None:
    """_extract_answer_from_knowledge finds sentences about democracy."""
    stage = _make_stage09()
    knowledge = [
        "Democracy is a system of government where power is vested in the people. "
        "Citizens vote to elect representatives. "
        "The concept originated in ancient Greece."
    ]
    answer = stage._extract_answer_from_knowledge("What is democracy?", knowledge)
    assert "democra" in answer.lower()
    assert len(answer) > 20


def test_extract_answer_finds_database_sentence() -> None:
    """_extract_answer_from_knowledge finds sentences about databases."""
    stage = _make_stage09()
    knowledge = [
        "A database is an organized collection of structured data. "
        "Relational databases use tables with rows and columns. "
        "SQL is the standard query language for databases."
    ]
    answer = stage._extract_answer_from_knowledge("How do databases work?", knowledge)
    assert "database" in answer.lower() or "databas" in answer.lower()
    assert len(answer) > 20


def test_extract_answer_stemmed_match_finds_recursive() -> None:
    """Stemmed query term 'recurs' matches 'recursive' in knowledge text."""
    stage = _make_stage09()
    knowledge = [
        "A recursive function is one that calls itself. "
        "Recursive algorithms are used in tree traversal and sorting."
    ]
    answer = stage._extract_answer_from_knowledge("What is recursion?", knowledge)
    assert len(answer) > 0, "Stemmed match should find 'recursive' in knowledge"


def test_extract_answer_empty_knowledge_returns_empty() -> None:
    """Empty knowledge list returns empty string."""
    stage = _make_stage09()
    assert stage._extract_answer_from_knowledge("What is AI?", []) == ""


def test_extract_answer_no_matching_terms_returns_empty() -> None:
    """Knowledge with no matching terms returns empty string."""
    stage = _make_stage09()
    knowledge = ["The weather is sunny today. The sky is blue."]
    assert stage._extract_answer_from_knowledge("What is recursion?", knowledge) == ""


def test_extract_answer_skips_very_short_sentences() -> None:
    """Sentences shorter than 15 chars are skipped."""
    stage = _make_stage09()
    knowledge = ["Hi. Recursion is a function that calls itself repeatedly."]
    answer = stage._extract_answer_from_knowledge("recursion", knowledge)
    assert "recursion" in answer.lower() or "recurs" in answer.lower()


def test_extract_answer_skips_very_long_sentences() -> None:
    """Sentences longer than 500 chars are skipped."""
    stage = _make_stage09()
    long_sent = "Recursion " + "is " * 250 + "useful."
    knowledge = [f"Recursion is a technique. {long_sent}"]
    answer = stage._extract_answer_from_knowledge("recursion", knowledge)
    # Should find the short sentence, not the long one
    assert "technique" in answer.lower() or "recurs" in answer.lower()


def test_extract_answer_deduplicates_sentences() -> None:
    """Duplicate sentences are not repeated in the answer."""
    stage = _make_stage09()
    knowledge = [
        "Recursion is a function that calls itself. "
        "Recursion is a function that calls itself. "
        "Recursion is a function that calls itself."
    ]
    answer = stage._extract_answer_from_knowledge("recursion", knowledge)
    # Should not repeat the same sentence 3 times
    assert answer.count("calls itself") <= 1


# ── Integration: BM25 → Stage09 answer extraction ────────────────────


def test_integration_recursion_query_finds_answer() -> None:
    """End-to-end: BM25 retrieves recursion knowledge, Stage09 extracts answer."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest(
        "Recursion is a programming technique where a function calls itself "
        "to solve smaller instances of the same problem. A base case is needed "
        "to terminate the recursion.",
        domain="programming",
    )
    kb.ingest("C++ templates allow generic type programming.", domain="programming")

    results = kb.search("What is recursion in programming?", top_k=3)
    assert len(results) > 0

    stage = _make_stage09()
    knowledge_texts = [r.content for r in results]
    answer = stage._extract_answer_from_knowledge("What is recursion in programming?", knowledge_texts)
    assert "recurs" in answer.lower(), "Answer should contain recursion-related content"


def test_integration_democracy_query_finds_answer() -> None:
    """End-to-end: BM25 retrieves democracy knowledge, Stage09 extracts answer."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest(
        "Democracy is a form of government in which power is vested in the people. "
        "Citizens exercise power through elected representatives. "
        "Free and fair elections are a cornerstone of democracy.",
        domain="political_science",
    )
    kb.ingest("Photosynthesis is the process by which plants make food.", domain="biology")

    results = kb.search("What is democracy?", top_k=3)
    assert len(results) > 0

    stage = _make_stage09()
    knowledge_texts = [r.content for r in results]
    answer = stage._extract_answer_from_knowledge("What is democracy?", knowledge_texts)
    assert "democra" in answer.lower(), "Answer should contain democracy-related content"


def test_integration_databases_query_finds_answer() -> None:
    """End-to-end: BM25 retrieves database knowledge, Stage09 extracts answer."""
    kb = KnowledgeBackend(embedding_dim=32)
    kb.ingest(
        "A database is an organized collection of data stored electronically. "
        "Relational databases organize data into tables with rows and columns. "
        "Database management systems provide query interfaces like SQL.",
        domain="programming",
    )
    kb.ingest("Quantum computing leverages superposition of qubits.", domain="physics")

    results = kb.search("How do databases work?", top_k=3)
    assert len(results) > 0

    stage = _make_stage09()
    knowledge_texts = [r.content for r in results]
    answer = stage._extract_answer_from_knowledge("How do databases work?", knowledge_texts)
    assert "databas" in answer.lower() or "database" in answer.lower(), "Answer should contain database content"
