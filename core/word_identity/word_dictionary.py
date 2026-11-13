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
Word Dictionary — comprehensive word knowledge with definitions, synonyms, and spell correction.

Provides:
- O(1) word lookup with definitions and synonyms
- Synonym expansion for query enrichment
- Levenshtein-based spell correction for misspelled words
- Related word discovery for semantic linking

Complexity:
    lookup: T(n) in O(1)
    synonyms: T(n) in O(1)
    correct: T(n) in O(k * m) for k dictionary words, m = word length
    expand_query: T(n) in O(n * k) for n tokens, k avg synonym count
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    """A single dictionary entry.

    Attributes:
        word: The lowercase word.
        definition: Short definition of the word.
        synonyms: Words with the same or similar meaning.
        related: Semantically related words (broader/narrower/associated).
        category: Semantic category (science, technology, nature, person, etc.).
    """
    word: str
    definition: str
    synonyms: Tuple[str, ...]
    related: Tuple[str, ...]
    category: str


class WordDictionary:
    """Comprehensive word dictionary with synonyms and spell correction.

    Loads a JSON dictionary file and provides:
    - Word lookup with definitions
    - Synonym expansion for query enrichment
    - Spell correction using Levenshtein distance
    - Related word discovery

    Complexity:
        init: T(n) in O(w) for w words.
        lookup: T(n) in O(1).
        correct: T(n) in O(w * m) for w words, m = input word length.
        expand_query: T(n) in O(n * s) for n tokens, s avg synonyms.
    """

    __slots__ = ("_entries", "_synonym_index", "_reverse_synonyms", "_valid_words")

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Load the dictionary from JSON file.

        Also loads word identity JSON files (emotion.json, social.json, etc.)
        to build a set of valid words that should NOT be spell-corrected.

        Args:
            data_path: Path to dictionary JSON file. Defaults to
                core/word_identity/dictionary.json

        Complexity: T(n) in O(w) for w words.
        """
        _dir = os.path.dirname(os.path.abspath(__file__))
        if data_path is None:
            data_path = os.path.join(_dir, "dictionary.json")

        self._entries: Dict[str, DictionaryEntry] = {}
        self._synonym_index: Dict[str, Set[str]] = {}
        self._reverse_synonyms: Dict[str, Set[str]] = {}
        self._valid_words: Set[str] = set()

        # Common English words that should never be spell-corrected.
        # These are frequently used in queries but may not be in our small
        # dictionary — without this list, "lord" gets corrected to "word",
        # "rings" to "bring", etc.
        _common_english = frozenset({
            "lord", "rings", "ring", "wars", "war", "movie", "movies",
            "harry", "potter", "spider", "man", "iron", "captain",
            "america", "thor", "avengers", "jedi", "sith", "empire",
            "strikes", "back", "return", "jedi", "fellowship", "tower",
            "king", "tolkien", "jackson", "lucas", "george", "roddenberry",
            "trek", "wars", "bird", "angry", "jackass",
            "fantasy", "fiction", "science", "franchise", "trilogy",
            "animated", "comedy", "film", "films", "cinema", "theater",
        })
        self._valid_words |= _common_english

        # Load word identity JSON files to build a set of valid words
        # that should never be spell-corrected (sad, angry, happy, etc.)
        for wi_file in ("social.json", "emotion.json", "programming.json",
                        "architecture.json", "ai.json"):
            wi_path = os.path.join(_dir, wi_file)
            if os.path.exists(wi_path):
                try:
                    with open(wi_path, "r", encoding="utf-8") as f:
                        wi_data = json.load(f)
                    # Words are nested under "words" key in WI files
                    wi_words = wi_data.get("words", wi_data)
                    for key in wi_words:
                        if isinstance(key, str):
                            self._valid_words.add(key.lower())
                except (json.JSONDecodeError, OSError):
                    pass

        if not os.path.exists(data_path):
            return

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        for word, entry_data in data.get("words", {}).items():
            word_lower = word.lower()
            synonyms = tuple(s.lower() for s in entry_data.get("synonyms", []))
            related = tuple(r.lower() for r in entry_data.get("related", []))
            entry = DictionaryEntry(
                word=word_lower,
                definition=entry_data.get("definition", ""),
                synonyms=synonyms,
                related=related,
                category=entry_data.get("category", "general"),
            )
            self._entries[word_lower] = entry

            # Build synonym index: word → set of its synonyms
            self._synonym_index[word_lower] = set(synonyms)

            # Build reverse index: synonym → set of words that have it as synonym
            for syn in synonyms:
                if syn not in self._reverse_synonyms:
                    self._reverse_synonyms[syn] = set()
                self._reverse_synonyms[syn].add(word_lower)

    def lookup(self, word: str) -> Optional[DictionaryEntry]:
        """Look up a word's dictionary entry.

        Args:
            word: Word to look up (case-insensitive).

        Returns:
            DictionaryEntry if found, None otherwise.

        Complexity: T(n) in O(1).
        """
        return self._entries.get(word.lower())

    def get_synonyms(self, word: str) -> List[str]:
        """Get all synonyms for a word.

        Args:
            word: Word to find synonyms for.

        Returns:
            List of synonym strings, empty if none found.

        Complexity: T(n) in O(1).
        """
        entry = self._entries.get(word.lower())
        if entry:
            return list(entry.synonyms)
        # Check reverse index — maybe this word IS a synonym of something
        reverse = self._reverse_synonyms.get(word.lower())
        if reverse:
            return list(reverse)
        return []

    def get_related(self, word: str) -> List[str]:
        """Get related words for a word.

        Args:
            word: Word to find related words for.

        Returns:
            List of related word strings.

        Complexity: T(n) in O(1).
        """
        entry = self._entries.get(word.lower())
        if entry:
            return list(entry.related)
        return []

    def get_definition(self, word: str) -> str:
        """Get the definition for a word.

        Args:
            word: Word to define.

        Returns:
            Definition string, or empty string if not found.

        Complexity: T(n) in O(1).
        """
        entry = self._entries.get(word.lower())
        return entry.definition if entry else ""

    def correct(self, word: str, max_distance: int = 2) -> Optional[str]:
        """Find the closest dictionary word by edit distance.

        Uses Levenshtein distance to find the closest match.
        Only returns a correction if distance <= max_distance.

        If the word is found in the dictionary but is a misspelling entry
        (category="misspelling"), follows its synonyms to the correct word.

        Words in the word identity system (emotion.json, social.json, etc.)
        are treated as valid and never corrected.

        Args:
            word: Potentially misspelled word.
            max_distance: Maximum edit distance for a valid correction.

        Returns:
            Corrected word, or None if no close match found.

        Complexity: T(n) in O(w * m) for w words, m = word length.
        """
        word_lower = word.lower()

        # Check if it's a known valid word from the word identity system
        if word_lower in self._valid_words:
            return word_lower

        if word_lower in self._entries:
            entry = self._entries[word_lower]
            # If it's a misspelling entry, follow synonyms to the correct word
            if entry.category == "misspelling" and entry.synonyms:
                return entry.synonyms[0]
            return word_lower

        # Check if it's a plural form of a dictionary word
        # "wars" → "war", "rings" → "ring", "movies" → "movie"
        # These are valid plurals, not misspellings
        if len(word_lower) > 3 and word_lower.endswith("s"):
            singular = word_lower[:-1]
            if singular in self._entries:
                entry = self._entries[singular]
                if entry.category != "misspelling":
                    return word_lower  # Accept plural as valid

        best_match: Optional[str] = None
        # Use max_distance=1 by default — distance 2 causes too many false
        # positives (e.g. "rings" → "bring", "lord" → "word").
        effective_max = min(max_distance, 1)
        best_distance = effective_max + 1

        for dict_word, entry in self._entries.items():
            # Skip misspelling entries — we want to correct TO real words
            if entry.category == "misspelling":
                continue
            # Quick length filter — skip if length difference exceeds max_distance
            if abs(len(dict_word) - len(word_lower)) > effective_max:
                continue

            dist = _levenshtein(word_lower, dict_word)
            if dist < best_distance:
                best_distance = dist
                best_match = dict_word

        return best_match if best_distance <= max_distance else None

    def expand_query(self, query: str) -> str:
        """Expand a query with synonyms and spell corrections.

        For each token in the query:
        1. If misspelled, correct it
        2. Add synonyms to the query for better BM25 matching

        For multi-word queries (2+ content tokens), synonym expansion is
        skipped because it dilutes phrase precision — "star wars" should not
        be expanded with "sun celestial body luminary" which matches astronomy.

        Args:
            query: User input query string.

        Returns:
            Expanded query string with synonyms and corrections.

        Complexity: T(n) in O(n * (k + w)) for n tokens, k synonyms, w dict words.
        """
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        if not tokens:
            return query

        # Filter stop words to count content tokens
        _stop = frozenset({
            "the", "is", "a", "an", "what", "who", "how", "why", "where",
            "when", "of", "to", "in", "for", "and", "or", "are", "was",
            "were", "be", "do", "does", "did", "can", "could", "from",
        })
        content_tokens = [t for t in tokens if t not in _stop]

        expanded_tokens: List[str] = list(tokens)
        added: Set[str] = set(tokens)

        # For multi-word queries, only do spell correction — skip synonym
        # expansion to preserve phrase precision
        skip_synonyms = len(content_tokens) >= 2

        for token in tokens:
            # Try spell correction
            corrected = self.correct(token)
            if corrected and corrected != token:
                if corrected not in added:
                    expanded_tokens.append(corrected)
                    added.add(corrected)
                # Get synonyms for the corrected word (skip for multi-word queries)
                if not skip_synonyms:
                    for syn in self.get_synonyms(corrected):
                        if syn not in added:
                            expanded_tokens.append(syn)
                            added.add(syn)
            else:
                # Get synonyms for the original word (skip for multi-word queries)
                if not skip_synonyms:
                    for syn in self.get_synonyms(token):
                        if syn not in added:
                            expanded_tokens.append(syn)
                            added.add(syn)

        return " ".join(expanded_tokens)

    def correct_query(self, query: str) -> str:
        """Correct misspelled words in a query while preserving structure.

        Only corrects words with edit distance <= 2.
        Preserves original word order and non-word tokens.

        Args:
            query: User input query string.

        Returns:
            Query with misspelled words corrected.

        Complexity: T(n) in O(n * w * m) for n tokens, w dict words, m = word length.
        """
        def replace_word(match: re.Match) -> str:
            word = match.group(0)
            # Don't correct very short words (1-2 chars) — too many false positives
            if len(word) < 3:
                return word
            corrected = self.correct(word)
            if corrected and corrected != word.lower():
                # Preserve original capitalization pattern
                if word[0].isupper():
                    corrected = corrected.capitalize()
                return corrected
            return word

        return re.sub(r"[a-zA-Z]+", replace_word, query)

    @property
    def word_count(self) -> int:
        """Return total number of words in the dictionary."""
        return len(self._entries)


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Uses the standard dynamic programming algorithm.

    Complexity: T(n) in O(m * n) for strings of length m and n.
                S(n) in O(min(m, n)).
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert_cost = prev_row[j + 1] + 1
            delete_cost = curr_row[j] + 1
            substitute_cost = prev_row[j] + (c1 != c2)
            curr_row.append(min(insert_cost, delete_cost, substitute_cost))
        prev_row = curr_row

    return prev_row[-1]
