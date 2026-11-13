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
KnowledgeRouter v2 — multi-domain semantic routing with fallback support.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """
    Immutable routing decision.

    Attributes:
        domain: Selected knowledge domain.
        confidence: Confidence in [0.0, 1.0].
        fallback_domain: Secondary domain if primary fails.
        matched_keywords: Keywords that triggered the routing.
        source: How the decision was made (keyword, semantic, fallback).

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    domain: str
    confidence: float
    fallback_domain: str
    matched_keywords: tuple[str, ...]
    source: str


@dataclass
class DomainConfig:
    """
    Mutable domain configuration.

    Attributes:
        domain: Domain identifier.
        keywords: Keywords that map to this domain.
        priority: Priority (higher = preferred for ambiguous matches).
        enabled: Whether this domain is active.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    domain: str
    keywords: List[str]
    priority: int
    enabled: bool = True


class KnowledgeRouterV2:
    """
    Multi-domain knowledge router with semantic and fallback routing.

    Complexity:
        route: T(n) ∈ O(d * k) for d domains, k keywords, S(n) ∈ O(d).
        register_domain: T(n) ∈ O(1), S(n) ∈ O(1).
    """

    __slots__ = ("_domains", "_compiled_patterns", "_fallback_domain", "_routing_log")

    def __init__(self, fallback_domain: str = "general") -> None:
        """Initialize the router.

        Args:
            fallback_domain: Default domain when no match is found.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._domains: Dict[str, DomainConfig] = {}
        self._compiled_patterns: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {}
        self._fallback_domain = fallback_domain
        self._routing_log: List[RoutingDecision] = []

    def register_domain(
        self,
        domain: str,
        keywords: List[str],
        priority: int = 0,
        enabled: bool = True,
    ) -> None:
        """Register a knowledge domain.

        Args:
            domain: Domain identifier.
            keywords: Keywords that map to this domain.
            priority: Priority for ambiguous matches.
            enabled: Whether this domain is active.

        Complexity: T(n) ∈ O(k) for k keywords, S(n) ∈ O(k).
        """
        self._domains[domain] = DomainConfig(
            domain=domain,
            keywords=keywords,
            priority=priority,
            enabled=enabled,
        )
        # Pre-compile patterns
        self._compiled_patterns[domain] = [
            (kw, re.compile(re.escape(kw), re.IGNORECASE))
            for kw in keywords
        ]

    def unregister_domain(self, domain: str) -> bool:
        """Remove a domain.

        Returns:
            True if a domain was removed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        removed = self._domains.pop(domain, None) is not None
        self._compiled_patterns.pop(domain, None)
        return removed

    def route(self, query: str) -> RoutingDecision:
        """Route a query to the best matching domain.

        Args:
            query: The query text to route.

        Returns:
            RoutingDecision with selected domain and confidence.

        Complexity: T(n) ∈ O(d * k * n) for d domains, k keywords, n query length.
        """
        if not query or not query.strip():
            return RoutingDecision(
                domain=self._fallback_domain,
                confidence=0.0,
                fallback_domain="",
                matched_keywords=(),
                source="empty_query",
            )

        # Score each domain
        scores: List[Tuple[float, int, str, List[str]]] = []

        for domain_name, config in self._domains.items():
            if not config.enabled:
                continue

            matched: List[str] = []
            for keyword, pattern in self._compiled_patterns.get(domain_name, []):
                if pattern.search(query):
                    matched.append(keyword)

            if matched:
                # Confidence: ratio of matched keywords to total keywords
                confidence = len(matched) / max(1, len(config.keywords))
                # Boost by priority
                score = confidence + config.priority * 0.1
                scores.append((score, config.priority, domain_name, matched))

        if not scores:
            decision = RoutingDecision(
                domain=self._fallback_domain,
                confidence=0.0,
                fallback_domain="",
                matched_keywords=(),
                source="fallback",
            )
            self._routing_log.append(decision)
            return decision

        # Sort by score descending, then priority
        scores.sort(key=lambda x: (x[0], x[1]), reverse=True)

        best_score, _, best_domain, best_matched = scores[0]
        fallback = scores[1][2] if len(scores) > 1 else ""

        # Normalize confidence to [0, 1]
        confidence = min(1.0, best_score)

        decision = RoutingDecision(
            domain=best_domain,
            confidence=confidence,
            fallback_domain=fallback,
            matched_keywords=tuple(best_matched),
            source="keyword",
        )
        self._routing_log.append(decision)
        return decision

    def route_multi(self, query: str, top_k: int = 3) -> List[RoutingDecision]:
        """Route a query to multiple domains, ranked.

        Args:
            query: The query text to route.
            top_k: Maximum number of domains to return.

        Returns:
            List of RoutingDecision sorted by confidence.

        Complexity: T(n) ∈ O(d * k * n), S(n) ∈ O(d).
        """
        if not query or not query.strip():
            return []

        results: List[RoutingDecision] = []
        for domain_name, config in self._domains.items():
            if not config.enabled:
                continue

            matched: List[str] = []
            for keyword, pattern in self._compiled_patterns.get(domain_name, []):
                if pattern.search(query):
                    matched.append(keyword)

            if matched:
                confidence = len(matched) / max(1, len(config.keywords))
                score = confidence + config.priority * 0.1
                results.append(RoutingDecision(
                    domain=domain_name,
                    confidence=min(1.0, score),
                    fallback_domain="",
                    matched_keywords=tuple(matched),
                    source="keyword",
                ))

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_k]

    def get_routing_log(self, limit: int = 100) -> List[RoutingDecision]:
        """Return recent routing decisions.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(n).
        """
        return list(self._routing_log[-limit:])

    def clear_routing_log(self) -> None:
        """Clear the routing log.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._routing_log.clear()

    def domain_count(self) -> int:
        """Return the number of registered domains.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._domains)

    def get_domains(self) -> List[str]:
        """Return all domain names.

        Complexity: T(n) ∈ O(d), S(n) ∈ O(d).
        """
        return list(self._domains.keys())

    def set_fallback_domain(self, domain: str) -> None:
        """Set the fallback domain.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._fallback_domain = domain
