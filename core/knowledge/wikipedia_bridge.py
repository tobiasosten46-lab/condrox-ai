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

"""Wikipedia Knowledge Bridge — deterministic fallback knowledge retrieval.

When BM25 search returns no relevant items, this module queries the
Wikipedia REST API to fetch a concise summary. This provides global
knowledge coverage without neural networks or LLMs.

Complexity:
    fetch: T(n) in O(http) for a single HTTP request, S(n) in O(k) for
           the returned summary text (typically 1-5 KB).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"
_REQUEST_TIMEOUT = 5  # seconds


@dataclass(frozen=True, slots=True)
class WikipediaResult:
    """A single Wikipedia summary result.

    Attributes:
        title: The Wikipedia article title.
        extract: The concise summary text (2-4 sentences).
        url: The canonical Wikipedia URL.
        found: Whether a real article was found (vs. disambiguation/missing).
    """
    title: str
    extract: str
    url: str
    found: bool


def _normalize_query(query: str) -> str:
    """Normalize a user query into a Wikipedia-friendly topic.

    Strips question words, punctuation, and common prefixes.
    Removes pronoun references (it, that, this) that have no antecedent
    in a stateless API call.

    Complexity: T(n) in O(n), S(n) in O(n).
    """
    # Remove common question prefixes
    query = re.sub(
        r"^(?:what\s+(?:is|are)\s+(?:an?\s+|the\s+)?)"
        r"|(?:explain\s+)"
        r"|(?:tell\s+me\s+about\s+)"
        r"|(?:how\s+(?:does|do|is|are)\s+)"
        r"|(?:why\s+(?:is|are|does|do)\s+)"
        r"|(?:define\s+)"
        r"|(?:who\s+(?:made|invented|created|discovered|wrote|designed|developed)\s+)"
        r"|(?:when\s+(?:was|did)\s+)"
        r"|(?:where\s+(?:was|did|is)\s+)",
        "", query, flags=re.IGNORECASE,
    )
    # Remove trailing punctuation
    query = re.sub(r"[?.!,;:]+$", "", query).strip()
    # Remove standalone pronouns that have no antecedent
    query = re.sub(r"\b(?:it|that|this|the first one|the first)\b", "", query, flags=re.IGNORECASE).strip()
    # Collapse whitespace
    query = re.sub(r"\s+", " ", query)
    return query


def _search_wikipedia(query: str) -> Optional[str]:
    """Search Wikipedia for the best matching article title.

    Uses the MediaWiki action API to find the closest article.

    Complexity: T(n) in O(http), S(n) in O(1).
    """
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": "1",
        "format": "json",
    })
    url = f"{_WIKIPEDIA_SEARCH}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CondroxAI/1.0"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])
            if results:
                return results[0].get("title")
    except Exception as exc:
        logger.debug("Wikipedia search failed for '%s': %s", query, exc)
    return None


def fetch_wikipedia_summary(query: str) -> Optional[WikipediaResult]:
    """Fetch a Wikipedia summary for a user query.

    Normalizes the query, searches for the best article, then fetches
    the summary extract. Returns None if no article is found.

    Args:
        query: The user's input query string.

    Returns:
        WikipediaResult with title, extract, and url, or None if not found.

    Complexity: T(n) in O(http), S(n) in O(k) for summary text.
    """
    topic = _normalize_query(query)
    if not topic or len(topic) < 3:
        return None

    # First, search for the best matching article
    title = _search_wikipedia(topic)
    if not title:
        # Try direct page lookup with the normalized topic
        title = topic.replace(" ", "_")

    # Fetch the summary
    encoded = urllib.parse.quote(title, safe="_")
    url = f"{_WIKIPEDIA_API}{encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CondroxAI/1.0"})
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Wikipedia summary fetch failed for '%s': %s", title, exc)
        return None

    extract = data.get("extract", "")
    if not extract or len(extract) < 20:
        return None

    page_type = data.get("type", "standard")
    if page_type == "disambiguation":
        # Try the first disambiguation link
        return None

    content_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    display_title = data.get("title", title)

    return WikipediaResult(
        title=display_title,
        extract=extract,
        url=content_url or f"https://en.wikipedia.org/wiki/{encoded}",
        found=True,
    )
