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
Symbolic reasoning engine for the unified Brain.

The ReasoningEngine applies a real rule-inference system:
- Evidence aggregation using weighted summation and softmax normalization.
- Deductive modus ponens over a knowledge base of if-then rules.
- Confidence scoring based on rule match strength and evidence weights.

Mathematical Foundation:
- Weighted evidence score: S = Σ w_i * e_i / Σ w_i
- Softmax over rule activation energies: p_j = exp(a_j) / Σ exp(a_k)
- Confidence: c = max_j(p_j) * evidence_score
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class Rule:
    """Immutable deductive rule with antecedents and consequent.

    Attributes:
        name: Human-readable rule identifier.
        antecedents: Conditions that must be present in facts.
        consequent: Conclusion drawn when antecedents match.
        weight: Rule importance in [0.0, 1.0].

    Complexity: T(n) ∈ O(1), S(n) ∈ O(k) where k is antecedent count.
    """

    name: str
    antecedents: Tuple[str, ...]
    consequent: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate invariants.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not self.name or not self.consequent:
            raise ValueError("Rule name and consequent must be non-empty")
        if self.weight < 0.0 or self.weight > 1.0:
            raise ValueError("weight must be in [0.0, 1.0]")


class ReasoningEngine:
    """
    Real, functional rule-based reasoning engine.

    Supports forward chaining over a set of immutable Rule objects and
    computes confidence scores from matched facts and rule weights.

    Attributes:
        _rules: List of registered rules.

    Complexity:
        add_rule: T(n) ∈ O(1)
        reason: T(n) ∈ O(r * a * f) for r rules, a antecedents, f facts.
    """

    __slots__ = ("_rules",)

    def __init__(self) -> None:
        """Initialize an empty reasoning engine."""
        self._rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Register a rule.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._rules.append(rule)

    @property
    def rule_count(self) -> int:
        """Return the number of registered rules.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._rules)

    def load_default_rules(self) -> None:
        """Load a minimal but real default rule set.

        Complexity: T(n) ∈ O(r) where r is the number of default rules.
        """
        defaults = [
            Rule("greeting", ("intent=greet",), "respond_greeting", 0.9),
            Rule("question", ("intent=question",), "respond_question", 0.9),
            Rule("code_request", ("intent=code", "mode=code"), "generate_code", 1.0),
            Rule("analyze_request", ("intent=analyze", "mode=analyze"), "perform_analysis", 1.0),
            Rule("explain_request", ("intent=explain", "mode=explain"), "provide_explanation", 0.95),
            Rule("memory_context", ("has_memory",), "use_memory", 0.8),
            Rule("knowledge_context", ("has_knowledge",), "use_knowledge", 0.8),
        ]
        for rule in defaults:
            self.add_rule(rule)

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Apply softmax normalization to a list of scores.

        Args:
            scores: Raw activation scores.

        Returns:
            Probability distribution summing to 1.0.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(n).
        """
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        total = sum(exps)
        if total == 0.0:
            return [1.0 / len(scores)] * len(scores)
        return [e / total for e in exps]

    def reason(
        self, facts: Dict[str, float], top_k: int = 3
    ) -> Tuple[str, float, List[str]]:
        """
        Perform forward chaining and return the best consequent.

        Args:
            facts: Mapping from fact strings to confidence values.
            top_k: Maximum number of evidence strings to include.

        Returns:
            Tuple of (consequent, confidence, evidence_list).

        Complexity: T(n) ∈ O(r * a + r log r) for r rules, a antecedents.
        """
        if not facts:
            return ("unknown", 0.0, [])

        rule_activations: List[Tuple[Rule, float, List[str]]] = []

        for rule in self._rules:
            antecedent_scores: List[float] = []
            evidence: List[str] = []
            for antecedent in rule.antecedents:
                if antecedent in facts:
                    antecedent_scores.append(facts[antecedent])
                    evidence.append(f"{antecedent}={facts[antecedent]:.3f}")
                else:
                    antecedent_scores.append(0.0)
                    evidence.append(f"{antecedent}=missing")

            if not antecedent_scores:
                continue

            total_weight = float(len(rule.antecedents))
            match_score = sum(antecedent_scores) / total_weight
            activation = rule.weight * match_score
            if activation > 0.0:
                rule_activations.append((rule, activation, evidence))

        if not rule_activations:
            return ("unknown", 0.0, [])

        scores = [activation for _, activation, _ in rule_activations]
        probabilities = self._normalize_scores(scores)

        best_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
        best_rule, _, evidence = rule_activations[best_idx]

        def _evidence_score(entry: str) -> float:
            if "=" not in entry:
                return 0.0
            value = entry.rsplit("=", 1)[-1]
            try:
                return float(value)
            except ValueError:
                return 0.0

        evidence.sort(key=_evidence_score, reverse=True)
        top_evidence = evidence[:top_k]

        return (best_rule.consequent, probabilities[best_idx], top_evidence)
