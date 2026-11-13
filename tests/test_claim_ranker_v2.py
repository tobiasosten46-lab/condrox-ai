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

"""Tests for ClaimRankerV2 — Phase 2-B component."""

from __future__ import annotations

from core.conversational.claim_ranker_v2 import (
    Claim,
    ClaimRankerV2,
    RankedClaim,
)


class TestClaimRankerV2:
    """Unit tests for ClaimRankerV2."""

    def test_score_claim_basic(self) -> None:
        """Basic claim scoring works."""
        ranker = ClaimRankerV2()
        claim = ranker.score_claim(
            text="Python is a programming language.",
            source="docs",
            query="what is python",
            evidence_count=3,
        )
        assert claim.evidence_score > 0
        assert claim.relevance_score > 0
        assert claim.confidence_score > 0

    def test_score_claim_hedge_words_lower_confidence(self) -> None:
        """Hedge words lower confidence score."""
        ranker = ClaimRankerV2()
        confident = ranker.score_claim(
            text="The answer is 42.",
            source="test",
            query="answer",
        )
        hedged = ranker.score_claim(
            text="Maybe the answer is possibly 42.",
            source="test",
            query="answer",
        )
        assert hedged.confidence_score < confident.confidence_score

    def test_composite_score(self) -> None:
        """Composite score is weighted average."""
        ranker = ClaimRankerV2()
        claim = Claim(
            text="test",
            source="test",
            evidence_score=1.0,
            recency_score=0.0,
            relevance_score=0.0,
            confidence_score=0.0,
        )
        expected = 0.35 * 1.0
        assert abs(claim.composite_score - expected) < 0.001

    def test_rank_orders_by_score(self) -> None:
        """Rank orders claims by descending score."""
        ranker = ClaimRankerV2()
        claim_low = Claim(
            text="low",
            source="test",
            evidence_score=0.1,
            recency_score=0.1,
            relevance_score=0.1,
            confidence_score=0.1,
        )
        claim_high = Claim(
            text="high",
            source="test",
            evidence_score=0.9,
            recency_score=0.9,
            relevance_score=0.9,
            confidence_score=0.9,
        )
        ranked = ranker.rank([claim_low, claim_high])
        assert ranked[0].rank == 1
        assert ranked[0].claim.text == "high"
        assert ranked[1].rank == 2
        assert ranked[1].claim.text == "low"

    def test_rank_empty_list(self) -> None:
        """Empty list returns empty."""
        ranker = ClaimRankerV2()
        assert ranker.rank([]) == []

    def test_detect_contradictions(self) -> None:
        """Contradictions are detected."""
        ranker = ClaimRankerV2()
        claims = [
            Claim(
                text="Python is the best language.",
                source="a",
                evidence_score=0.5,
                recency_score=0.5,
                relevance_score=0.5,
                confidence_score=0.5,
            ),
            Claim(
                text="However, Python is not the best language for performance.",
                source="b",
                evidence_score=0.5,
                recency_score=0.5,
                relevance_score=0.5,
                confidence_score=0.5,
            ),
        ]
        contradictions = ranker.detect_contradictions(claims)
        assert len(contradictions) > 0

    def test_no_contradiction_for_unrelated_claims(self) -> None:
        """Unrelated claims have no contradictions."""
        ranker = ClaimRankerV2()
        claims = [
            Claim(
                text="The sky is blue.",
                source="a",
                evidence_score=0.5,
                recency_score=0.5,
                relevance_score=0.5,
                confidence_score=0.5,
            ),
            Claim(
                text="Water boils at 100 degrees.",
                source="b",
                evidence_score=0.5,
                recency_score=0.5,
                relevance_score=0.5,
                confidence_score=0.5,
            ),
        ]
        contradictions = ranker.detect_contradictions(claims)
        assert contradictions == {}

    def test_word_overlap(self) -> None:
        """Word overlap computes Jaccard similarity."""
        ranker = ClaimRankerV2()
        overlap = ranker._word_overlap("python programming language", "python is a language")
        assert overlap > 0.0
        assert overlap <= 1.0

    def test_word_overlap_no_common_words(self) -> None:
        """No common words gives zero overlap."""
        ranker = ClaimRankerV2()
        overlap = ranker._word_overlap("apple banana", "cherry date")
        assert overlap == 0.0

    def test_ranked_claim_is_frozen(self) -> None:
        """RankedClaim is immutable."""
        ranker = ClaimRankerV2()
        claim = Claim(
            text="test",
            source="test",
            evidence_score=0.5,
            recency_score=0.5,
            relevance_score=0.5,
            confidence_score=0.5,
        )
        ranked = ranker.rank([claim])[0]
        try:
            ranked.rank = 99  # type: ignore[misc]
            assert False, "Should have raised"
        except Exception:
            pass  # expected

    def test_evidence_score_logarithmic(self) -> None:
        """Evidence score scales logarithmically."""
        ranker = ClaimRankerV2()
        low = ranker.score_claim("test", "s", "test", evidence_count=1)
        high = ranker.score_claim("test", "s", "test", evidence_count=10)
        assert high.evidence_score > low.evidence_score
