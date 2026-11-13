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

"""Tests for KnowledgeRouterV2 — Phase 2-D component."""

from __future__ import annotations

from core.conversational.knowledge_router_v2 import (
    DomainConfig,
    KnowledgeRouterV2,
    RoutingDecision,
)


class TestKnowledgeRouterV2:
    """Unit tests for KnowledgeRouterV2."""

    def test_register_and_route(self) -> None:
        """Domain is registered and routing works."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python", "django", "flask"], priority=5)
        decision = router.route("How do I use Python decorators?")
        assert decision.domain == "python"
        assert decision.confidence > 0

    def test_no_match_uses_fallback(self) -> None:
        """No match falls back to default domain."""
        router = KnowledgeRouterV2(fallback_domain="general")
        decision = router.route("What is the meaning of life?")
        assert decision.domain == "general"
        assert decision.source == "fallback"

    def test_empty_query(self) -> None:
        """Empty query uses fallback."""
        router = KnowledgeRouterV2()
        decision = router.route("")
        assert decision.domain == "general"
        assert decision.source == "empty_query"

    def test_priority_breaks_ties(self) -> None:
        """Higher priority domain wins on tie."""
        router = KnowledgeRouterV2()
        router.register_domain("low", ["code", "function"], priority=1)
        router.register_domain("high", ["code", "function"], priority=9)
        decision = router.route("Write a code function")
        assert decision.domain == "high"

    def test_multiple_matches_routed_correctly(self) -> None:
        """Multiple domain matches are ranked."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python", "function"])
        router.register_domain("javascript", ["javascript", "function"])
        decision = router.route("Python function example")
        assert decision.domain == "python"

    def test_route_multi(self) -> None:
        """route_multi returns multiple domains."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python", "function"])
        router.register_domain("javascript", ["javascript", "function"])
        results = router.route_multi("function example", top_k=2)
        assert len(results) == 2

    def test_fallback_domain_in_result(self) -> None:
        """Fallback domain is included when multiple matches."""
        router = KnowledgeRouterV2()
        router.register_domain("a", ["test"], priority=1)
        router.register_domain("b", ["test"], priority=2)
        decision = router.route("test query")
        assert decision.fallback_domain != ""

    def test_unregister_domain(self) -> None:
        """Unregister removes a domain."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python"])
        assert router.unregister_domain("python") is True
        assert router.domain_count() == 0

    def test_routing_log(self) -> None:
        """Routing log records decisions."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python"])
        router.route("Python is great")
        log = router.get_routing_log()
        assert len(log) > 0

    def test_clear_routing_log(self) -> None:
        """clear_routing_log empties the log."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python"])
        router.route("Python")
        router.clear_routing_log()
        assert len(router.get_routing_log()) == 0

    def test_disabled_domain_not_routed(self) -> None:
        """Disabled domains are not used for routing."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python"], enabled=False)
        decision = router.route("Python is great")
        assert decision.domain == "general"

    def test_get_domains(self) -> None:
        """get_domains returns all domain names."""
        router = KnowledgeRouterV2()
        router.register_domain("a", ["a"])
        router.register_domain("b", ["b"])
        domains = router.get_domains()
        assert set(domains) == {"a", "b"}

    def test_set_fallback_domain(self) -> None:
        """set_fallback_domain changes the fallback."""
        router = KnowledgeRouterV2()
        router.set_fallback_domain("custom_fallback")
        decision = router.route("unknown query")
        assert decision.domain == "custom_fallback"

    def test_routing_decision_is_frozen(self) -> None:
        """RoutingDecision is immutable."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python"])
        decision = router.route("Python")
        try:
            decision.domain = "hacked"  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass

    def test_matched_keywords(self) -> None:
        """Matched keywords are returned in the decision."""
        router = KnowledgeRouterV2()
        router.register_domain("python", ["python", "django"])
        decision = router.route("I love Python and Django")
        assert "python" in decision.matched_keywords
        assert "django" in decision.matched_keywords
