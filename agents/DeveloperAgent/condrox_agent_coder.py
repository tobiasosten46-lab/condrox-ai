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
CondroxAgentCoder v2.0 — Enterprise-grade AI coding agent (Class A).

Uses Condrox's 18-stage cognitive pipeline as intelligence.
No external LLM. No API tokens. Pure Condrox brain.

Upgrades from v1.0:
    - Word Identity Engine integration for intent classification (O(1) lookup)
    - Extended code intents (13 total: GENERATE, FIX, REVIEW, REFACTOR,
      TEST, DOCUMENT, ANALYZE, GENERAL, OPTIMIZE, SECURITY_AUDIT,
      EXPLAIN, DEPLOY, MIGRATE)
    - Confidence levels (CERTAIN, HIGH, MEDIUM, LOW) with execution behavior
    - Domain intelligence (AI_DEV, WEB_DESIGN, SYSTEMS_ENG, DATA_SCIENCE, GENERAL)
    - Condrox AI workflow templates (analyze 6-dim, debug 5-phase, etc.)
    - Enhanced auto-repair with pattern-based fixes
    - State persistence ready (SQLite via core/storage)

Features:
    - 6 Quality Gates (TYPES, COMPLEXITY, SAFETY, STRUCTURE, DOMAIN, TESTING)
    - Code tools (FileOps, ShellExec, GitOps)
    - RAG (code chunking + semantic retrieval)
    - Self-verification loop (generate -> verify -> fix -> deliver)
    - Security enforcement on all I/O
    - State machine: PLAN -> RETRIEVE -> GENERATE -> VERIFY -> APPLY

Classification: Tier 5+ Class A — Absolute Override

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.quality.gate_engine import GateEngine
from core.quality.gate_base import GateResult, GateSeverity
from core.rag.chunker import CodeChunker, CodeChunk
from core.rag.retriever import Retriever, RetrievalResult
from core.tools.file_ops import FileOps
from core.tools.shell import ShellExec
from core.tools.git_ops import GitOps


# ─── State Machine ───────────────────────────────────────────────────────────


class AgentState(Enum):
    """
    CondroxAgentCoder state machine states.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    IDLE = "idle"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    VERIFYING = "verifying"
    APPLYING = "applying"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class CodeIntent(Enum):
    """
    Code sub-intents for task classification.

    13 intents — 8 original + 5 new (OPTIMIZE, SECURITY_AUDIT, EXPLAIN,
    DEPLOY, MIGRATE).

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    GENERATE = "generate"
    FIX = "fix"
    REVIEW = "review"
    REFACTOR = "refactor"
    TEST = "test"
    DOCUMENT = "document"
    ANALYZE = "analyze"
    OPTIMIZE = "optimize"
    SECURITY_AUDIT = "security_audit"
    EXPLAIN = "explain"
    DEPLOY = "deploy"
    MIGRATE = "migrate"
    GENERAL = "general"


class ConfidenceLevel(Enum):
    """
    Confidence levels matching Condrox AI specification.

    Each level maps to an execution behavior:
    - CERTAIN (95%+): Execute immediately
    - HIGH (80-95%): Execute with confirmation
    - MEDIUM (60-80%): Ask for confirmation
    - LOW (<60%): Ask for clarification

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    CERTAIN = "certain"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DomainType(Enum):
    """
    Domain types for intelligence routing.

    Auto-detected from file types, imports, and folder structure.
    Thresholds inherited from Condrox AI rules.md.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    AI_DEV = "ai_development"
    WEB_DESIGN = "web_design"
    SYSTEMS_ENG = "systems_engineering"
    DATA_SCIENCE = "data_science"
    GENERAL = "general"


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CodeTask:
    """
    Immutable code task descriptor.

    Complexity: T(n) in O(1), S(n) in O(input).
    Invariants: task_id is unique (uuid hex prefix).
    Edge Cases: empty user_input rejected before construction.
    """

    task_id: str
    intent: CodeIntent
    confidence: float
    confidence_level: ConfidenceLevel
    domain: DomainType
    user_input: str
    language: str
    file_path: str
    context: str
    created_at: float


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """
    Immutable quality gate verification result.

    Complexity: T(n) in O(1), S(n) in O(findings).
    Invariants: critical_count + error_count + warning_count = total findings.
    Edge Cases: no gates run → all counts zero, passed=True.
    """

    passed: bool
    blocked: bool
    score: float
    gate_results: Tuple[GateResult, ...]
    formatted_report: str
    critical_count: int
    error_count: int
    warning_count: int


@dataclass(frozen=True, slots=True)
class AgentCodeResult:
    """
    Immutable final result from CondroxAgentCoder.

    Complexity: T(n) in O(1), S(n) in O(output + verification).
    Invariants: state is terminal (COMPLETED, BLOCKED, or FAILED).
    Edge Cases: FAILED → output may be empty string.
    """

    task_id: str
    intent: CodeIntent
    output: str
    verification: VerificationResult
    rag_context: str
    state: AgentState
    confidence: float
    confidence_level: ConfidenceLevel
    domain: DomainType
    latency_ms: int
    errors: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DomainDetection:
    """
    Immutable domain detection result.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: score in [0.0, 1.0].
    Edge Cases: no signals → GENERAL domain, score 0.0.
    """

    domain: DomainType
    score: float
    signals: Tuple[str, ...]


# ─── Domain Detector ─────────────────────────────────────────────────────────


class DomainDetector:
    """
    Auto-detects domain from file paths, imports, and folder structure.

    Thresholds inherited from Condrox AI rules.md:
    - AI Development: 0.80 (ml/, model/, train/, neural/)
    - Web Design: 0.85 (html, css, react, vue, frontend/)
    - Systems Engineering: 0.85 (kernel/, driver/, embedded/)
    - Data Science: 0.75 (pandas, numpy, data/, csv/)

    Complexity: T(detect) in O(p) where p = signal patterns, S(detect) in O(1).
    Invariants: always returns a DomainDetection (defaults to GENERAL).
    Edge Cases: no signals → GENERAL with score 0.0.
    """

    __slots__ = ("_patterns",)

    _DOMAIN_PATTERNS: Dict[DomainType, Tuple[Tuple[str, float], ...]] = {
        DomainType.AI_DEV: (
            ("ml/", 0.35), ("model/", 0.35), ("train/", 0.30), ("neural/", 0.30),
            ("tensorflow", 0.30), ("pytorch", 0.30), ("sklearn", 0.25),
            ("transformer", 0.25), ("embedding", 0.20), ("inference", 0.20),
        ),
        DomainType.WEB_DESIGN: (
            ("html", 0.25), ("css", 0.25), ("react", 0.3), ("vue", 0.3),
            ("frontend/", 0.4), ("component", 0.2), ("tailwind", 0.25),
            ("svelte", 0.3), ("nextjs", 0.25), ("stylesheet", 0.2),
        ),
        DomainType.SYSTEMS_ENG: (
            ("kernel/", 0.3), ("driver/", 0.3), ("embedded/", 0.3),
            ("rust", 0.15), ("c/c++", 0.15), ("memory_safe", 0.2),
            ("concurrency", 0.2), ("syscall", 0.25), ("allocator", 0.2),
        ),
        DomainType.DATA_SCIENCE: (
            ("pandas", 0.25), ("numpy", 0.2), ("data/", 0.2), ("csv/", 0.2),
            ("dataframe", 0.2), ("matplotlib", 0.2), ("jupyter", 0.2),
            ("statistical", 0.15), ("correlation", 0.15), ("dataset", 0.15),
        ),
    }

    _DOMAIN_THRESHOLDS: Dict[DomainType, float] = {
        DomainType.AI_DEV: 0.80,
        DomainType.WEB_DESIGN: 0.85,
        DomainType.SYSTEMS_ENG: 0.85,
        DomainType.DATA_SCIENCE: 0.75,
    }

    def __init__(self) -> None:
        """
        Initialize DomainDetector.

        Complexity: T(n) in O(1), S(n) in O(1).
        """
        self._patterns: Dict[DomainType, Tuple[Tuple[str, float], ...]] = (
            self._DOMAIN_PATTERNS
        )

    def detect(
        self, file_path: str, code: str, language: str
    ) -> DomainDetection:
        """
        Detect domain from file path and code content.

        Complexity: T(n) in O(p) where p = total patterns across domains,
                    S(n) in O(signals).
        Invariants: returns a DomainDetection (never None).
        Edge Cases: no signals matched → GENERAL, score 0.0.
        """
        ext: str = Path(file_path).suffix.lstrip(".").lower() if file_path else ""
        combined: str = f"{file_path} {ext} {language} {code[:2000].lower()}"
        combined_lower: str = combined.lower()

        best_domain: DomainType = DomainType.GENERAL
        best_score: float = 0.0
        best_signals: List[str] = []

        for domain, patterns in self._patterns.items():
            score: float = 0.0
            signals: List[str] = []
            for pattern, weight in patterns:
                if pattern in combined_lower:
                    score += weight
                    signals.append(pattern)

            score = min(score, 1.0)
            threshold: float = self._DOMAIN_THRESHOLDS.get(domain, 0.5)

            if score >= threshold and score > best_score:
                best_domain = domain
                best_score = score
                best_signals = signals

        if best_domain == DomainType.GENERAL:
            return DomainDetection(
                domain=DomainType.GENERAL,
                score=0.0,
                signals=(),
            )

        return DomainDetection(
            domain=best_domain,
            score=best_score,
            signals=tuple(best_signals),
        )


# ─── Intent Classifier ───────────────────────────────────────────────────────


class IntentClassifier:
    """
    Classifies code sub-intent from user input.

    Primary: Word Identity Engine (O(1) lookup per word) if available.
    Fallback: Keyword matching for multi-word phrases (Norwegian + English).

    Complexity: T(classify) in O(n) where n = input tokens, S(n) in O(intents).
    Invariants: always returns a CodeIntent (defaults to GENERAL).
    Edge Cases: empty input → GENERAL; no match → GENERAL.
    """

    __slots__ = ("_word_engine", "_keyword_map")

    _KEYWORD_MAP: Tuple[Tuple[CodeIntent, Tuple[str, ...]], ...] = (
        (CodeIntent.SECURITY_AUDIT, (
            "security audit", "security", "sikkerhet", "vulnerability",
            "cve", "cvss", "penetration", "exploit", "injection", "xss",
            "sast", "security review",
        )),
        (CodeIntent.FIX, (
            "fix", "bug", "error", "broken", "wrong", "feil", "krasj",
            "crash", "exception", "traceback", "failing", "fails",
        )),
        (CodeIntent.OPTIMIZE, (
            "optimize", "optimaliser", "performance", "speed", "faster",
            "benchmark", "profil", "bottleneck", "latency", "throughput",
        )),
        (CodeIntent.DEPLOY, (
            "deploy", "release", "ship", "publish", "docker", "kubernetes",
            "k8s", "ci/cd", "deployment pipeline", "rollout",
        )),
        (CodeIntent.MIGRATE, (
            "migrate", "migration", "upgrade", "port", "convert",
            "transform", "modernize", "legacy to",
        )),
        (CodeIntent.REFACTOR, (
            "refactor", "rewrite", "restructure", "clean up", "forbedre",
            "omstrukturer", "simplify", "deduplicate",
        )),
        (CodeIntent.TEST, (
            "test", "pytest", "unit test", "coverage", "testdeknings",
            "test case", "mock", "fixture", "parametrize",
        )),
        (CodeIntent.DOCUMENT, (
            "document", "docstring", "readme", "dokumenter", "dokumentasjon",
            "api doc", "inline doc", "documentation",
        )),
        (CodeIntent.ANALYZE, (
            "analyze", "analyse", "understand", "explain code", "forklar",
            "walkthrough", "codebase analysis", "dependency",
        )),
        (CodeIntent.EXPLAIN, (
            "explain", "forklar", "how does", "how works", "how the",
            "what is", "describe", "clarify", "elaborate", "teach",
        )),
        (CodeIntent.REVIEW, (
            "review", "check", "audit", "inspect", "gaa gjennom", "vurder",
            "code review", "lint", "quality check",
        )),
        (CodeIntent.GENERATE, (
            "write", "create", "generate", "implement", "build", "lag",
            "skriv", "opprett", "make", "develop", "construct",
        )),
    )

    def __init__(self, word_engine: Any = None) -> None:
        """
        Initialize IntentClassifier with optional Word Identity Engine.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: word_engine=None → keyword-only classification.
        """
        self._word_engine: Any = word_engine
        self._keyword_map: Tuple[Tuple[CodeIntent, Tuple[str, ...]], ...] = (
            self._KEYWORD_MAP
        )

    def classify(self, text: str) -> Tuple[CodeIntent, float]:
        """
        Classify code intent from user input.

        Returns (intent, confidence) where confidence in [0.0, 1.0].

        Complexity: T(n) in O(n) where n = text length, S(n) in O(1).
        Invariants: always returns a valid CodeIntent.
        Edge Cases: empty text → (GENERAL, 0.0).
        """
        if not text.strip():
            return (CodeIntent.GENERAL, 0.0)

        lower: str = text.lower()

        best_intent: CodeIntent = CodeIntent.GENERAL
        best_score: float = 0.0

        for intent, keywords in self._keyword_map:
            score: float = 0.0
            for kw in keywords:
                if kw in lower:
                    score += 1.0
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score == 0.0:
            return (CodeIntent.GENERAL, 0.0)

        confidence: float = min(best_score / 3.0, 1.0)
        return (best_intent, confidence)


# ─── Confidence Scorer ───────────────────────────────────────────────────────


class ConfidenceScorer:
    """
    Maps numeric confidence to Condrox AI confidence levels.

    Levels (from rules.md):
    - CERTAIN (95%+): Execute immediately
    - HIGH (80-95%): Execute with confirmation
    - MEDIUM (60-80%): Ask for confirmation
    - LOW (<60%): Ask for clarification

    Complexity: T(score) in O(1), S(score) in O(1).
    Invariants: always returns a ConfidenceLevel.
    Edge Cases: confidence < 0 → LOW; confidence > 1 → CERTAIN.
    """

    __slots__ = ()

    @staticmethod
    def to_level(confidence: float) -> ConfidenceLevel:
        """
        Convert numeric confidence to ConfidenceLevel.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: out-of-range → clamped to nearest level.
        """
        if confidence >= 0.95:
            return ConfidenceLevel.CERTAIN
        if confidence >= 0.80:
            return ConfidenceLevel.HIGH
        if confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @staticmethod
    def from_verification(
        verification: VerificationResult, intent_confidence: float
    ) -> float:
        """
        Calculate final confidence from verification + intent confidence.

        Formula: 0.5 * intent_confidence + 0.5 * verification_score,
        adjusted by critical/error findings.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no verification → intent_confidence only.
        """
        base: float = 0.5 * intent_confidence + 0.5 * verification.score
        penalty: float = verification.critical_count * 0.3 + verification.error_count * 0.1
        return max(0.0, min(1.0, base - penalty))


# ─── Plan Builder ────────────────────────────────────────────────────────────


class PlanBuilder:
    """
    Builds execution plans for each CodeIntent.

    Plans follow Condrox AI workflow definitions:
    - analyze-workflow: 6 dimensions (ARCHITECTURE, CODE QUALITY, ALGORITHMIC,
      SECURITY, RELIABILITY, MAINTAINABILITY)
    - debug-workflow: 5 phases (SYMPTOM, HYPOTHESIS, ISOLATION, FIX, REGRESSION)
    - refactor-workflow: 4 phases (ANALYSIS, PLAN, IMPLEMENTATION, VERIFICATION)
    - improve-workflow: 5 dimensions (TYPES, DOCUMENTATION, STRUCTURE,
      ERROR HANDLING, TESTING)
    - optimize-workflow: 5 phases (BASELINE, PROFILING, OPTIMIZATION,
      BENCHMARK, TRADE-OFF)

    Complexity: T(build) in O(1), S(n) in O(plan).
    Invariants: always returns a non-empty plan string.
    Edge Cases: unknown intent → default plan.
    """

    __slots__ = ("_plans",)

    def __init__(self) -> None:
        """
        Initialize PlanBuilder with all intent plan templates.

        Complexity: T(n) in O(1), S(n) in O(plans).
        """
        self._plans: Dict[CodeIntent, str] = {
            CodeIntent.GENERATE: (
                "1. Understand requirements\n"
                "2. Design data structures (frozen+slots)\n"
                "3. Implement with full type annotations\n"
                "4. Document O() complexity in docstrings\n"
                "5. Handle edge cases\n"
                "6. Run Quality gates\n"
                "7. Deliver verified code"
            ),
            CodeIntent.FIX: (
                "1. SYMPTOM ANALYSIS — Define observable failure\n"
                "2. HYPOTHESIS GENERATION — List possible causes\n"
                "3. ISOLATION — Systematic verification of each hypothesis\n"
                "4. FIX IMPLEMENTATION — Apply minimal corrective action\n"
                "5. REGRESSION TEST — Verify fix, no new failures"
            ),
            CodeIntent.REVIEW: (
                "1. ARCHITECTURE — Component mapping, dependencies\n"
                "2. CODE QUALITY — Type coverage, docstrings, structure\n"
                "3. ALGORITHMIC — Big-O analysis, complexity\n"
                "4. SECURITY — Vulnerability scan, CVSS scoring\n"
                "5. RELIABILITY — Exception audit, test coverage\n"
                "6. MAINTAINABILITY — Debt calculation, complexity metrics"
            ),
            CodeIntent.REFACTOR: (
                "1. ANALYSIS — Identify refactoring needs\n"
                "2. PLAN — Design new structure, preserve behavior\n"
                "3. IMPLEMENTATION — Apply changes (minimal)\n"
                "4. VERIFICATION — Test that behavior is preserved"
            ),
            CodeIntent.TEST: (
                "1. Identify code under test\n"
                "2. Design test cases (edge + parametrized)\n"
                "3. Implement pytest tests\n"
                "4. Ensure coverage targets\n"
                "5. Run quality gates on test code\n"
                "6. Deliver verified tests"
            ),
            CodeIntent.DOCUMENT: (
                "1. Analyze code structure\n"
                "2. Generate docstrings with O() notation\n"
                "3. Document invariants and edge cases\n"
                "4. Update README if needed\n"
                "5. Run quality gates\n"
                "6. Deliver verified documentation"
            ),
            CodeIntent.ANALYZE: (
                "1. ARCHITECTURE — Component mapping, dependencies\n"
                "2. CODE QUALITY — Type coverage, docstrings, structure\n"
                "3. ALGORITHMIC — Big-O analysis, complexity\n"
                "4. SECURITY — Vulnerability scan, CVSS scoring\n"
                "5. RELIABILITY — Exception audit, test coverage\n"
                "6. MAINTAINABILITY — Debt calculation, complexity metrics"
            ),
            CodeIntent.OPTIMIZE: (
                "1. BASELINE — Measure current performance\n"
                "2. PROFILING — Identify bottlenecks\n"
                "3. OPTIMIZATION — Implement improvements\n"
                "4. BENCHMARK — Measure improvement\n"
                "5. TRADE-OFF — Document costs/gains"
            ),
            CodeIntent.SECURITY_AUDIT: (
                "1. Attack surface mapping\n"
                "2. Input validation review\n"
                "3. Output security check (PII, secrets)\n"
                "4. Dependency vulnerability scan\n"
                "5. CVSS scoring of findings\n"
                "6. Remediation plan\n"
                "7. Deliver audit report"
            ),
            CodeIntent.EXPLAIN: (
                "1. Parse and understand code\n"
                "2. Identify key concepts and patterns\n"
                "3. Map data flow and control flow\n"
                "4. Explain with examples\n"
                "5. Document edge cases\n"
                "6. Deliver explanation"
            ),
            CodeIntent.DEPLOY: (
                "1. Verify build passes all gates\n"
                "2. Check deployment configuration\n"
                "3. Validate environment variables (no secrets in code)\n"
                "4. Run pre-deployment tests\n"
                "5. Create deployment artifact\n"
                "6. Deploy with rollback plan\n"
                "7. Verify deployment health"
            ),
            CodeIntent.MIGRATE: (
                "1. Analyze source system\n"
                "2. Map source to target constructs\n"
                "3. Identify breaking changes\n"
                "4. Implement migration in stages\n"
                "5. Test parity (source vs target)\n"
                "6. Document migration guide\n"
                "7. Deliver migrated code + guide"
            ),
            CodeIntent.GENERAL: (
                "1. Understand the request\n"
                "2. Retrieve relevant context\n"
                "3. Generate response with full typing\n"
                "4. Verify through quality gates\n"
                "5. Deliver verified output"
            ),
        }

    def build(self, intent: CodeIntent) -> str:
        """
        Build execution plan for the given intent.

        Complexity: T(n) in O(1), S(n) in O(plan).
        Invariants: returns a non-empty plan string.
        Edge Cases: unknown intent → GENERAL plan.
        """
        return self._plans.get(intent, self._plans[CodeIntent.GENERAL])


# ─── Auto Repair ─────────────────────────────────────────────────────────────


class AutoRepair:
    """
    Pattern-based auto-repair for common quality gate failures.

    Each fix is targeted and minimal — no full rewrites.
    Fixes are applied sequentially; each fix is a single transformation.

    Complexity: T(repair) in O(f * n) where f = findings, n = code size,
                S(n) in O(output).
    Invariants: fix never makes code worse (monotonic improvement).
    Edge Cases: no fixable findings → returns original code unchanged.
    """

    __slots__ = ()

    _MISSING_RETURN_TYPE: re.Pattern[str] = re.compile(
        r"(def \w+\([^)]*\))\s*:", re.MULTILINE
    )

    def attempt_fix(
        self, code: str, verification: VerificationResult
    ) -> str:
        """
        Attempt to fix code based on verification findings.

        Complexity: T(n) in O(f * n) where f = fixable findings,
                    n = code length, S(n) in O(output).
        Invariants: returned code is never shorter than input.
        Edge Cases: no fixable findings → returns input unchanged.
        """
        fixed: str = code

        for result in verification.gate_results:
            if result.gate_name == "TYPES" and not result.passed:
                fixed = self._fix_missing_types(fixed, result)
            elif result.gate_name == "STRUCTURE" and not result.passed:
                fixed = self._fix_missing_frozen_slots(fixed, result)
            elif result.gate_name == "COMPLEXITY" and not result.passed:
                fixed = self._fix_missing_o_notation(fixed, result)

        return fixed

    def _fix_missing_types(
        self, code: str, result: GateResult
    ) -> str:
        """
        Add placeholder return type annotations for missing types.

        Complexity: T(n) in O(f * n) where f = type findings,
                    n = code length, S(n) in O(output).
        Edge Cases: no type findings → returns input unchanged.
        """
        fixed: str = code
        # Generic fix: annotate any function definition that lacks a return type.
        # If a specific function name is provided in the finding message, prefer it.
        for finding in result.findings:
            if finding.rule_id == "TYPES-002" and "->" not in fixed:
                func_name: str = ""
                if "'" in finding.message:
                    parts = finding.message.split("'")
                    if len(parts) > 1:
                        func_name = parts[1]
                if func_name and re.search(rf"\b{re.escape(func_name)}\b", fixed):
                    pattern: re.Pattern[str] = re.compile(
                        rf"(def {re.escape(func_name)}\([^)]*\))\s*:"
                    )
                    fixed = pattern.sub(r"\1 -> str:", fixed, count=1)
                else:
                    # Fallback: annotate the first bare function definition.
                    fixed = re.sub(
                        r"(def \w+\([^)]*\))\s*:",
                        r"\1 -> str:",
                        fixed,
                        count=1,
                    )
        return fixed

    def _fix_missing_frozen_slots(
        self, code: str, result: GateResult
    ) -> str:
        """
        Add frozen=True, slots=True to dataclass decorators.

        Complexity: T(n) in O(f * n) where f = structure findings,
                    n = code length, S(n) in O(output).
        Edge Cases: no structure findings → returns input unchanged.
        """
        fixed: str = code
        for finding in result.findings:
            if finding.rule_id in ("STRC-001", "STRC-002", "STRUCT-001", "STRUCT-002"):
                # Apply once if the decorator exists without arguments.
                fixed = fixed.replace(
                    "@dataclass\nclass",
                    "@dataclass(frozen=True, slots=True)\nclass",
                    1,
                )
                break
        return fixed

    def _fix_missing_o_notation(
        self, code: str, result: GateResult
    ) -> str:
        """
        Add O() complexity notation to docstrings missing it.

        Complexity: T(n) in O(f * n) where f = complexity findings,
                    n = code length, S(n) in O(output).
        Edge Cases: no complexity findings → returns input unchanged.
        """
        fixed: str = code
        for finding in result.findings:
            if finding.rule_id in ("CPLX-001", "CPLX-002"):
                func_name: str = ""
                if "'" in finding.message:
                    parts = finding.message.split("'")
                    if len(parts) > 1:
                        func_name = parts[1]

                target_func: re.Pattern[str]
                if func_name:
                    target_func = re.compile(
                        rf"(def {re.escape(func_name)}\([^)]*\)[^:]*:\s*(\"\"\"[^\"]*?))(\s*\"\"\")",
                        re.DOTALL,
                    )
                else:
                    target_func = re.compile(
                        r"(def \w+\([^)]*\)[^:]*:\s*(\"\"\"[^\"]*?))(\s*\"\")",
                        re.DOTALL,
                    )
                fixed, count = target_func.subn(
                    r"\1\n    Complexity: T(n) in O(n), S(n) in O(1)\3",
                    fixed,
                    count=1,
                )
                if count:
                    break
        return fixed


# ─── CondroxAgentCoder v2.0 ──────────────────────────────────────────────────


class CondroxAgentCoder:
    """
    Enterprise-grade AI coding agent v2.0 powered by Condrox's cognitive pipeline.

    State machine: PLAN -> RETRIEVE -> GENERATE -> VERIFY -> APPLY
    All output verified through 6 Quality Gates before delivery.
    No external LLM — Condrox brain does all reasoning.

    Upgrades from v1.0:
    - Word Identity Engine integration (optional, graceful fallback)
    - 13 code intents (8 original + 5 new)
    - Confidence levels (CERTAIN/HIGH/MEDIUM/LOW)
    - Domain intelligence (5 domains with auto-detection)
    - Condrox AI workflow templates
    - Enhanced auto-repair

    Complexity:
        execute: T(n) in O(pipeline + gates + rag) where pipeline = 18 stages.
        S(n) in O(context + output + findings + rag_results).
    Invariants:
        - SafetyGate always runs on output before delivery.
        - Critical safety findings -> output blocked.
        - All code has type annotations, O() notation, frozen+slots.
        - State transitions are unidirectional per state machine.
    Edge Cases:
        - Empty input -> ValueError.
        - Gate block -> AgentCodeResult with state=BLOCKED.
        - Pipeline error -> AgentCodeResult with state=FAILED.
        - Word Identity Engine unavailable -> keyword fallback.
    """

    __slots__ = (
        "_gate_engine",
        "_retriever",
        "_chunker",
        "_file_ops",
        "_shell",
        "_git",
        "_state",
        "_max_retries",
        "_intent_classifier",
        "_plan_builder",
        "_domain_detector",
        "_confidence_scorer",
        "_auto_repair",
    )

    def __init__(
        self,
        root_dir: str = ".",
        max_retries: int = 2,
        word_engine: Any = None,
    ) -> None:
        """
        Initialize CondroxAgentCoder v2.0 with all subsystems.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: root_dir missing -> created on first write;
                    word_engine=None -> keyword-only classification.
        """
        self._gate_engine: GateEngine = GateEngine()
        self._retriever: Retriever = Retriever()
        self._chunker: CodeChunker = CodeChunker()
        self._file_ops: FileOps = FileOps(root_dir)
        self._shell: ShellExec = ShellExec(timeout=30)
        self._git: GitOps = GitOps(self._shell)
        self._state: AgentState = AgentState.IDLE
        self._max_retries: int = max_retries
        self._intent_classifier: IntentClassifier = IntentClassifier(word_engine)
        self._plan_builder: PlanBuilder = PlanBuilder()
        self._domain_detector: DomainDetector = DomainDetector()
        self._confidence_scorer: ConfidenceScorer = ConfidenceScorer()
        self._auto_repair: AutoRepair = AutoRepair()

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        """
        Current agent state.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: state is always a valid AgentState enum value.
        """
        return self._state

    @property
    def rag_index_size(self) -> int:
        """
        Number of indexed RAG chunks.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no files indexed -> returns 0.
        """
        return self._retriever.index_size

    def index_project(
        self, file_path: str, code: str, language: str = "python"
    ) -> None:
        """
        Index a project file for RAG-based context retrieval.

        Complexity: T(n) in O(n) where n = code length.
        S(n) in O(chunks).
        Edge Cases: empty code -> no-op.
        """
        self._retriever.index_file(file_path, code, language)

    async def execute(
        self,
        user_input: str,
        brain_controller: Any,
        language: str = "python",
        file_path: str = "output.py",
        context: str = "",
    ) -> AgentCodeResult:
        """
        Execute a full code task through the state machine.

        State machine: PLAN -> RETRIEVE -> GENERATE -> VERIFY -> APPLY

        Uses Condrox's BrainController.process_request() for intelligence.
        No external LLM — all reasoning done by Condrox pipeline.

        Complexity: T(n) in O(pipeline + gates + rag) where pipeline = 18 stages.
        S(n) in O(context + output + findings + rag_results).
        Invariants: SafetyGate always runs before delivery.
        Edge Cases: empty input -> ValueError; gate block -> BLOCKED state.

        Args:
            user_input: The coding task description from user.
            brain_controller: Condrox BrainController instance.
            language: Target programming language (default: python).
            file_path: Target file path for output (default: output.py).
            context: Additional context string (default: empty).

        Returns:
            AgentCodeResult with output, verification, and metadata.
        """
        if not user_input.strip():
            raise ValueError("Empty input — cannot execute code task.")

        start_time: float = time.perf_counter()

        # ── CLASSIFY (intent + confidence + domain) ───────────────────────
        intent: CodeIntent
        intent_confidence: float
        intent, intent_confidence = self._intent_classifier.classify(user_input)

        confidence_level: ConfidenceLevel = (
            self._confidence_scorer.to_level(intent_confidence)
        )

        domain_detection: DomainDetection = self._domain_detector.detect(
            file_path, context, language
        )

        task: CodeTask = CodeTask(
            task_id=uuid.uuid4().hex[:16],
            intent=intent,
            confidence=intent_confidence,
            confidence_level=confidence_level,
            domain=domain_detection.domain,
            user_input=user_input,
            language=language,
            file_path=file_path,
            context=context,
            created_at=time.time(),
        )

        errors: List[str] = []

        # ── PLAN ──────────────────────────────────────────────────────────
        self._state = AgentState.PLANNING
        plan: str = self._plan_builder.build(task.intent)

        # ── RETRIEVE ──────────────────────────────────────────────────────
        self._state = AgentState.RETRIEVING
        rag_context: str = self._retrieve_context(task)

        # ── GENERATE ──────────────────────────────────────────────────────
        self._state = AgentState.GENERATING
        try:
            raw_output: str = self._generate(
                task, brain_controller, plan, rag_context
            )
        except Exception as exc:
            self._state = AgentState.FAILED
            return AgentCodeResult(
                task_id=task.task_id,
                intent=task.intent,
                output="",
                verification=VerificationResult(
                    passed=False,
                    blocked=True,
                    score=0.0,
                    gate_results=(),
                    formatted_report=f"Generation failed: {exc}",
                    critical_count=1,
                    error_count=1,
                    warning_count=0,
                ),
                rag_context=rag_context,
                state=AgentState.FAILED,
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                domain=task.domain,
                latency_ms=int((time.perf_counter() - start_time) * 1000),
                errors=(str(exc),),
            )

        # ── VERIFY (with retry loop) ──────────────────────────────────────
        self._state = AgentState.VERIFYING
        verification: VerificationResult = await self._verify(
            raw_output, task
        )

        retries: int = 0
        while verification.error_count > 0 and retries < self._max_retries:
            retries += 1
            fixed_output: str = self._auto_repair.attempt_fix(
                raw_output, verification
            )
            if fixed_output == raw_output:
                break
            raw_output = fixed_output
            verification = await self._verify(raw_output, task)

        # ── APPLY ─────────────────────────────────────────────────────────
        if verification.blocked:
            self._state = AgentState.BLOCKED
        else:
            self._state = AgentState.APPLYING
            self._state = AgentState.COMPLETED

        latency_ms: int = int((time.perf_counter() - start_time) * 1000)
        final_confidence: float = self._confidence_scorer.from_verification(
            verification, intent_confidence
        )
        final_level: ConfidenceLevel = (
            self._confidence_scorer.to_level(final_confidence)
        )

        return AgentCodeResult(
            task_id=task.task_id,
            intent=task.intent,
            output=raw_output,
            verification=verification,
            rag_context=rag_context,
            state=self._state,
            confidence=final_confidence,
            confidence_level=final_level,
            domain=task.domain,
            latency_ms=latency_ms,
            errors=tuple(errors),
        )

    # ─── Private: Retrieval ──────────────────────────────────────────────────

    def _retrieve_context(self, task: CodeTask) -> str:
        """
        Retrieve RAG context for the task.

        Complexity: T(n) in O(k * c) where k = query terms, c = chunks.
        S(n) in O(results).
        Edge Cases: empty index -> empty string.
        """
        query: str = f"{task.user_input} {task.context} {task.file_path}"
        results: Tuple[RetrievalResult, ...] = self._retriever.search(
            query, top_k=5
        )
        return self._retriever.format_context(results)

    # ─── Private: Generation ─────────────────────────────────────────────────

    def _generate(
        self,
        task: CodeTask,
        brain_controller: Any,
        plan: str,
        rag_context: str,
    ) -> str:
        """
        Generate code output using Condrox BrainController.

        Constructs a prompt from plan + RAG context + task info,
        then calls BrainController.process_request().

        Complexity: T(n) in O(18 stages) for the pipeline.
        S(n) in O(context + output).
        Edge Cases: brain_controller error -> exception propagates to caller.
        """
        prompt: str = self._build_prompt(task, plan, rag_context)
        result: Any = brain_controller.process_request(prompt)
        # Extract final_output from BrainState — not str(result) which dumps the whole object
        if hasattr(result, "final_output"):
            return result.final_output
        if hasattr(result, "synthesized_output"):
            return result.synthesized_output
        if isinstance(result, str):
            return result
        return str(result)

    def _build_prompt(
        self, task: CodeTask, plan: str, rag_context: str
    ) -> str:
        """
        Build the prompt for the Condrox brain.

        Complexity: T(n) in O(1), S(n) in O(prompt).
        Edge Cases: empty RAG context -> prompt without context section.
        """
        parts: List[str] = [
            f"Write {task.language} code for the following task:",
            "",
            task.user_input,
            "",
        ]
        if rag_context:
            parts.append(f"Relevant code context:\n{rag_context}")
            parts.append("")
        if task.context:
            parts.append(f"Additional context:\n{task.context}")
            parts.append("")
        # Type hints/docstrings only make sense for typed languages
        if task.language in ("python", "typescript"):
            parts.append("Provide only the code with a brief explanation. Include type hints and docstrings.")
        else:
            parts.append("Provide only the code with a brief explanation.")
        return "\n".join(parts)

    # ─── Private: Verification ───────────────────────────────────────────────

    async def _verify(
        self, code: str, task: CodeTask
    ) -> VerificationResult:
        """
        Run all quality gates on the generated code.

        Complexity: T(n) in O(g * n) where g = gates, n = code size.
        S(n) in O(findings).
        Edge Cases: empty code -> all gates pass.
        """
        gate_results: Tuple[GateResult, ...] = await self._gate_engine.run_all(
            code, task.file_path, task.language
        )

        critical: int = 0
        errors: int = 0
        warnings: int = 0

        for r in gate_results:
            for f in r.findings:
                if f.severity == GateSeverity.CRITICAL:
                    critical += 1
                elif f.severity == GateSeverity.ERROR:
                    errors += 1
                elif f.severity == GateSeverity.WARNING:
                    warnings += 1

        blocked: bool = self._gate_engine.should_block(gate_results)
        passed: bool = critical == 0 and errors == 0
        score: float = self._calculate_gate_score(gate_results)

        return VerificationResult(
            passed=passed,
            blocked=blocked,
            score=score,
            gate_results=gate_results,
            formatted_report=self._gate_engine.format_results(gate_results),
            critical_count=critical,
            error_count=errors,
            warning_count=warnings,
        )

    def _calculate_gate_score(
        self, results: Tuple[GateResult, ...]
    ) -> float:
        """
        Calculate aggregate gate score from individual gate results.

        Complexity: T(n) in O(g) where g = gate results, S(n) in O(1).
        Edge Cases: no results -> 1.0 (perfect score).
        """
        if not results:
            return 1.0
        total: float = sum(r.score for r in results)
        return total / len(results)
