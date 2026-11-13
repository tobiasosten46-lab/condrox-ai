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
Wikidata API integration for real-time knowledge retrieval.

Uses Wikidata's SPARQL endpoint to query structured knowledge without
downloading the full dataset. Returns clean, reference-free information.

Complexity: T(n) in O(query + response), S(n) in O(response).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import aiohttp


class WikidataAPI:
    """
    Real-time Wikidata API client for knowledge retrieval.

    Queries Wikidata's SPARQL endpoint to get clean, structured data
    without reference numbers or citations.

    Complexity:
        query: T(n) in O(n) where n = query complexity
        clean_response: T(n) in O(m) where m = response length
    """

    __slots__ = ("_endpoint", "_timeout", "_headers")

    def __init__(self, timeout: float = 10.0) -> None:
        """Initialize the Wikidata API client.

        Args:
            timeout: Request timeout in seconds.
        """
        self._endpoint = "https://query.wikidata.org/sparql"
        self._timeout = timeout
        self._headers = {
            "User-Agent": "CondroxAI/1.0 (https://condrox.ai; condrox@ai.com)"
        }

    def _clean_response(self, text: str) -> str:
        """Remove reference numbers, citations, and noise from response.

        Complexity: T(n) in O(m) where m = text length.
        """
        if not text:
            return ""
        
        # Remove reference numbers like [1], [2,3], [1-5]
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[\d+,\s*\d+\]", "", text)
        text = re.sub(r"\[\d+–\d+\]", "", text)
        text = re.sub(r"\[\d+-\d+\]", "", text)
        
        # Remove citation templates
        text = re.sub(r"\{\{cite.*?\}\}", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\{\{ref.*?\}\}", "", text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return text

    async def query_entity(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Query Wikidata for an entity by name.

        Args:
            entity_name: Name of the entity to search for.

        Returns:
            Dictionary with entity information or None if not found.

        Complexity: T(n) in O(query + response).
        """
        sparql = f"""
        SELECT ?item ?itemLabel ?itemDescription WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch".
            bd:serviceParam wikibase:endpoint "https://www.wikidata.org/w/api.php".
            bd:serviceParam mwapi:search "{entity_name}".
            bd:serviceParam mwapi:language "en".
            bd:serviceParam wikibase:limit 1.
            ?item wikibase:apiOutputItem ?itemResult.
          }}
          OPTIONAL {{ ?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "en") }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._endpoint,
                    params={"query": sparql, "format": "json"},
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    results = data.get("results", {}).get("bindings", [])
                    
                    if not results:
                        return None
                    
                    result = results[0]
                    return {
                        "id": result.get("item", {}).get("value", ""),
                        "label": result.get("itemLabel", {}).get("value", ""),
                        "description": self._clean_response(
                            result.get("itemDescription", {}).get("value", "")
                        ),
                    }
        except Exception:
            return None

    async def query_property(self, entity_id: str, property_id: str) -> Optional[str]:
        """Query a specific property of an entity.

        Args:
            entity_id: Wikidata entity ID (e.g., "Q42").
            property_id: Wikidata property ID (e.g., "P31").

        Returns:
            Property value or None if not found.

        Complexity: T(n) in O(query + response).
        """
        sparql = f"""
        SELECT ?value ?valueLabel WHERE {{
          wd:{entity_id} wdt:{property_id} ?value.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._endpoint,
                    params={"query": sparql, "format": "json"},
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    results = data.get("results", {}).get("bindings", [])
                    
                    if not results:
                        return None
                    
                    result = results[0]
                    value = result.get("valueLabel", {}).get("value", "")
                    return self._clean_response(value)
        except Exception:
            return None

    async def search(self, query: str) -> List[Dict[str, str]]:
        """Search Wikidata for entities matching the query.

        Args:
            query: Search query.

        Returns:
            List of matching entities with labels and descriptions.

        Complexity: T(n) in O(query + response).
        """
        sparql = f"""
        SELECT ?item ?itemLabel ?itemDescription WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch".
            bd:serviceParam wikibase:endpoint "https://www.wikidata.org/w/api.php".
            bd:serviceParam mwapi:search "{query}".
            bd:serviceParam mwapi:language "en".
            bd:serviceParam wikibase:limit 5.
            ?item wikibase:apiOutputItem ?itemResult.
          }}
          OPTIONAL {{ ?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "en") }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 5
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._endpoint,
                    params={"query": sparql, "format": "json"},
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    results = data.get("results", {}).get("bindings", [])
                    
                    entities = []
                    for result in results:
                        entities.append({
                            "id": result.get("item", {}).get("value", ""),
                            "label": result.get("itemLabel", {}).get("value", ""),
                            "description": self._clean_response(
                                result.get("itemDescription", {}).get("value", "")
                            ),
                        })
                    
                    return entities
        except Exception:
            return []
