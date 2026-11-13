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
Wikipedia API client — synchronous, real-time knowledge retrieval.

Uses the Wikipedia REST API to fetch clean article summaries.
No async needed — uses urllib directly.

Complexity: T(n) in O(query + response), S(n) in O(response).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class WikipediaAPI:
    """
    Synchronous Wikipedia API client.

    Uses the Wikipedia REST API and Action API to search and fetch
    article summaries for real-time knowledge retrieval.

    Complexity:
        search: T(n) in O(query + response)
        get_summary: T(n) in O(query + response)
    """

    __slots__ = ("_timeout", "_headers", "_search_url", "_summary_url")

    def __init__(self, timeout: float = 10.0) -> None:
        """Initialize the Wikipedia API client.

        Args:
            timeout: Request timeout in seconds.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        self._timeout = timeout
        self._headers = {
            "User-Agent": "CondroxAI/1.0 (https://condrox.ai; contact@condrox.ai)"
        }
        self._search_url = "https://en.wikipedia.org/w/api.php"
        self._summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    def _clean_text(self, text: str) -> str:
        """Remove HTML, reference numbers, and noise from text.

        Complexity: T(n) in O(m) where m = text length.
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove reference numbers like [1], [2,3]
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[\d+,\s*\d+\]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def search(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """Search Wikipedia for articles matching the query.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of matching articles with title and summary.

        Complexity: T(n) in O(query + response).
        """
        # Clean the query for search
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_query or len(clean_query) < 2:
            return []

        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "srlimit": str(limit),
            "format": "json",
            "srprop": "snippet",
        })

        url = f"{self._search_url}?{params}"

        try:
            req = urllib.request.Request(url, headers=self._headers)
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                results = data.get("query", {}).get("search", [])

                articles = []
                for result in results:
                    title = result.get("title", "")
                    snippet = self._clean_text(result.get("snippet", ""))
                    articles.append({
                        "title": title,
                        "snippet": snippet,
                    })
                return articles
        except Exception:
            return []

    def get_summary(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch a clean article summary from Wikipedia REST API.

        Args:
            title: Article title (will be URL-encoded).

        Returns:
            Dictionary with title, summary, and description, or None.

        Complexity: T(n) in O(query + response).
        """
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{self._summary_url}{encoded_title}"

        try:
            req = urllib.request.Request(url, headers=self._headers)
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

                summary = self._clean_text(data.get("extract", ""))
                if not summary or len(summary) < 20:
                    return None

                return {
                    "title": data.get("title", title),
                    "summary": summary,
                    "description": data.get("description", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
        except Exception:
            return None

    def search_and_summarize(self, query: str) -> Optional[Dict[str, Any]]:
        """Search Wikipedia and return the best matching article summary.

        Combines search + get_summary in one call.

        Args:
            query: Search query.

        Returns:
            Dictionary with title, summary, and description, or None.

        Complexity: T(n) in O(query + response).
        """
        results = self.search(query, limit=1)
        if not results:
            return None

        title = results[0].get("title", "")
        if not title:
            return None

        summary = self.get_summary(title)
        if summary:
            return summary

        # Fallback: use the search snippet if full summary not available
        snippet = results[0].get("snippet", "")
        if snippet and len(snippet) > 20:
            return {
                "title": title,
                "summary": snippet,
                "description": "",
                "url": "",
            }

        return None
