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
Training models for the Condrox AI pre-launch validation pipeline.

Defines immutable data structures for scenarios, gate results, and reports.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True, slots=True)
class TrainingScenario:
    """
    Immutable scenario definition for a single training/validation case.

    Attributes:
        scenario_id: Unique identifier.
        domain: Domain label (school, business, public, health).
        role: User role (student, teacher, clerk, accountant, etc.).
        input_text: Input given to the AI.
        expected_output_type: Expected response shape
            (explanation, step_by_step, structured_text, json).
        expected_keywords: Keywords/concepts the answer must contain.
        forbidden_patterns: Phrases that indicate hallucination or unsafe output.
        min_steps: Minimum numbered steps for step_by_step outputs.
        min_words: Minimum word count for the answer.
        requires_citations: Whether the answer should reference sources/rules.

    Complexity: S(n) ∈ O(k + f) for k keywords and f forbidden patterns.
    """

    scenario_id: str
    domain: str
    role: str
    input_text: str
    expected_output_type: str
    expected_keywords: tuple[str, ...]
    forbidden_patterns: tuple[str, ...] = ()
    min_steps: int = 0
    min_words: int = 20
    requires_citations: bool = False


@dataclass(frozen=True, slots=True)
class GateResult:
    """
    Result for one quality gate.

    Attributes:
        gate_name: Name of the gate (STRUCTURE, DOMAIN, SAFETY, COMPLEXITY,
            TYPES, TESTING).
        passed: Whether the gate passed.
        score: Normalized score in [0.0, 1.0].
        findings: Human-readable findings.

    Complexity: S(n) ∈ O(f) for f findings.
    """

    gate_name: str
    passed: bool
    score: float
    findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """
    Result of running one scenario.

    Attributes:
        scenario_id: Identifier of the scenario.
        actual_output: Raw output from the AI.
        gate_results: Tuple of GateResult.
        overall_pass: True if all gates passed.

    Complexity: S(n) ∈ O(g) for g gate results.
    """

    scenario_id: str
    actual_output: str
    gate_results: tuple[GateResult, ...]
    overall_pass: bool


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """
    Aggregated report for a domain or full run.

    Attributes:
        domain: Domain label or "all".
        total: Number of scenarios run.
        passed: Number that passed all gates.
        failed: Number that failed at least one gate.
        results: Tuple of TrainingResult.

    Complexity: S(n) ∈ O(r) for r results.
    """

    domain: str
    total: int
    passed: int
    failed: int
    results: tuple[TrainingResult, ...]


# Type alias for a runner that turns input text into AI output.
TrainingRunner = Callable[[str], str]
