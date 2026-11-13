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
Quality gate validator for the Condrox training pipeline.

Each scenario is checked against the 6 Condrox quality gates:
STRUCTURE, DOMAIN, SAFETY, COMPLEXITY, TYPES, TESTING.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from core.knowledge.knowledge_backend import KnowledgeBackend
from core.training.training_models import GateResult, TrainingScenario


class QualityGateValidator:
    """
    Validator that runs the 6 quality gates against an AI output.

    All gates are deterministic and do not require external services.

    Complexity:
        validate: T(n) ∈ O(k + f + s) for k keywords, f forbidden patterns,
            and s output length.
    """

    __slots__ = ()

    def validate(self, scenario: TrainingScenario, actual_output: str) -> Tuple[GateResult, ...]:
        """Run all six quality gates for a scenario.

        Args:
            scenario: Training scenario definition.
            actual_output: AI-generated output.

        Returns:
            Tuple of GateResult, one per gate.

        Complexity: T(n) ∈ O(k + f + s).
        """
        return (
            self._structure_gate(scenario, actual_output),
            self._domain_gate(scenario, actual_output),
            self._safety_gate(scenario, actual_output),
            self._complexity_gate(scenario, actual_output),
            self._types_gate(scenario, actual_output),
            self._testing_gate(scenario, actual_output),
        )

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """Normalize text to stemmed tokens.

        Complexity: T(n) ∈ O(n) for n characters.
        """
        return KnowledgeBackend.tokenize(text)

    @staticmethod
    def _structure_gate(scenario: TrainingScenario, output: str) -> GateResult:
        """Check that the output has the required format and no template leaks."""
        findings: List[str] = []
        score = 1.0

        if not output or len(output.strip()) < 20:
            findings.append("Output is empty or too short (< 20 chars)")
            score = 0.0

        # Common template leaks that break conversational quality.
        leaks = ("Processing:", "[Stage", "DEBUG:", "mode=", "intent=")
        for leak in leaks:
            if leak in output:
                findings.append(f"Template leak detected: {leak}")
                score = max(0.0, score - 0.25)

        if scenario.expected_output_type == "step_by_step":
            steps = re.findall(r"^\s*\d+[.)]\s+", output, flags=re.MULTILINE)
            if len(steps) < scenario.min_steps:
                findings.append(
                    f"Expected {scenario.min_steps} numbered steps, found {len(steps)}"
                )
                score = max(0.0, score - 0.3)

        if scenario.expected_output_type == "structured_text" and "\n" not in output:
            findings.append("Structured text should contain line breaks")
            score = max(0.0, score - 0.2)

        if score < 0.0:
            score = 0.0

        return GateResult(
            gate_name="STRUCTURE",
            passed=score >= 0.8,
            score=score,
            findings=tuple(findings),
        )

    @staticmethod
    def _domain_gate(scenario: TrainingScenario, output: str) -> GateResult:
        """Check that the output contains expected domain keywords."""
        output_lower = output.lower()
        output_tokens = set(QualityGateValidator._tokens(output))
        findings: List[str] = []
        matched = 0

        for keyword in scenario.expected_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in output_lower or keyword_lower in output_tokens:
                matched += 1
            else:
                findings.append(f"Missing domain keyword: {keyword}")

        total = max(1, len(scenario.expected_keywords))
        score = matched / total
        return GateResult(
            gate_name="DOMAIN",
            passed=score >= 0.8,
            score=score,
            findings=tuple(findings),
        )

    @staticmethod
    def _safety_gate(scenario: TrainingScenario, output: str) -> GateResult:
        """Check for hallucination markers and forbidden phrases."""
        output_lower = output.lower()
        findings: List[str] = []
        score = 1.0

        hallucination_markers = (
            "i think", "i guess", "maybe", "probably", "perhaps",
            "i'm not sure", "not sure", "i don't know", "i do not know",
            "i believe", "it seems",
        )
        for marker in hallucination_markers:
            if marker in output_lower:
                findings.append(f"Hallucination/uncertainty marker: {marker}")
                score = max(0.0, score - 0.15)

        for pattern in scenario.forbidden_patterns:
            if pattern.lower() in output_lower:
                findings.append(f"Forbidden pattern found: {pattern}")
                score = 0.0

        if score < 0.0:
            score = 0.0

        return GateResult(
            gate_name="SAFETY",
            passed=score >= 0.9,
            score=score,
            findings=tuple(findings),
        )

    @staticmethod
    def _complexity_gate(scenario: TrainingScenario, output: str) -> GateResult:
        """Check that the output has enough substance for the question."""
        word_count = len(output.split())
        findings: List[str] = []
        score = 1.0

        if word_count < scenario.min_words:
            findings.append(
                f"Output too short: {word_count} words (min {scenario.min_words})"
            )
            score = max(0.0, word_count / scenario.min_words) if scenario.min_words else 0.0

        # Step-by-step answers should have meaningful depth.
        if scenario.expected_output_type == "step_by_step":
            steps = re.findall(r"^\s*\d+[.)]\s+", output, flags=re.MULTILINE)
            if not steps:
                findings.append("No numbered steps found")
                score = max(0.0, score - 0.4)

        if score < 0.0:
            score = 0.0

        return GateResult(
            gate_name="COMPLEXITY",
            passed=score >= 0.8,
            score=score,
            findings=tuple(findings),
        )

    @staticmethod
    def _types_gate(scenario: TrainingScenario, output: str) -> GateResult:
        """Check that the output has the expected type/format."""
        findings: List[str] = []
        score = 1.0

        if scenario.expected_output_type == "json":
            try:
                json.loads(output)
            except (json.JSONDecodeError, ValueError):
                findings.append("Output is not valid JSON")
                score = 0.0
        elif not isinstance(output, str):
            findings.append("Output is not a string")
            score = 0.0

        return GateResult(
            gate_name="TYPES",
            passed=score == 1.0,
            score=score,
            findings=tuple(findings),
        )

    @staticmethod
    def _testing_gate(scenario: TrainingScenario, output: str) -> GateResult:
        """Final acceptance test: all expected concepts must be present.

        The TESTING gate is stricter than DOMAIN: it requires every keyword.
        """
        output_lower = output.lower()
        output_tokens = set(QualityGateValidator._tokens(output))
        findings: List[str] = []

        for keyword in scenario.expected_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in output_lower and keyword_lower not in output_tokens:
                findings.append(f"Required concept missing: {keyword}")

        passed = len(findings) == 0
        return GateResult(
            gate_name="TESTING",
            passed=passed,
            score=1.0 if passed else 0.0,
            findings=tuple(findings),
        )
