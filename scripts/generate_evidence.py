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
generate_evidence.py — Condrox Evidence Engine CLI

Phase 1 entry point:
  - Collects system and scenario evidence
  - Generates JSON and HTML reports
  - Persists evidence to Reports/evidence/

Usage:
    python scripts/generate_evidence.py
    python scripts/generate_evidence.py --title "My Evidence" --output-dir Reports/evidence

Complexity: T(n) ∈ O(p + s) for p probes and s scenarios.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dataclasses

from core.evidence import (
    AcademicReportConfig,
    AcademicReportGenerator,
    EvidenceCollector,
    EvidencePortalGenerator,
    EvidenceStore,
    ReportGenerator,
)
from core.verification import ScenarioEngine


DEFAULT_SCENARIOS = [
    {
        "name": "personal_empathy",
        "category": "conversation",
        "input_text": "I'm thinking about quitting my job. I don't know what to do.",
        "expected_intent": "question",
        "expected_mode": "explain",
    },
    {
        "name": "explain_algorithm",
        "category": "intelligence",
        "input_text": "Explain how quicksort works step by step.",
        "expected_intent": "explain",
        "expected_mode": "explain",
    },
    {
        "name": "code_agent_trigger",
        "category": "agents",
        "input_text": "Write a Python function to sort a list.",
        "expected_intent": "code",
        "expected_mode": "code",
    },
]

DEFAULT_CONVERSATIONS = [
    {
        "name": "multi_turn_stability",
        "turns": [
            {"input": "Hi, I'm feeling stressed about work."},
            {"input": "My boss keeps giving me impossible deadlines."},
            {"input": "What should I do?"},
            {"input": "That makes sense. Can you help me write a plan?"},
            {"input": "Thanks, this really helps."},
        ],
    },
    {
        "name": "technical_explanation",
        "turns": [
            {"input": "Explain recursion in simple terms."},
        ],
    },
]


def _normalize(raw: dict) -> dict:
    """Normalize legacy scenario dicts to use input_text."""
    if "input" in raw and "input_text" not in raw:
        raw["input_text"] = raw.pop("input")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Condrox Evidence Engine")
    parser.add_argument("--title", default="Condrox AI Evidence Report", help="Report title")
    parser.add_argument("--output-dir", default=str(_PROJECT_ROOT / "Reports" / "evidence"), help="Output directory")
    parser.add_argument("--api-host", default="localhost", help="API host")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    os.environ.setdefault("CONDROX_SKIP_KNOWLEDGE_LOAD", "1")

    # Load scenarios from verification/scenarios if available, otherwise use defaults
    engine = ScenarioEngine()
    if engine.count():
        loaded = [_normalize(dataclasses.asdict(s)) for s in engine.list_scenarios()]
        excluded = ("conversation", "reasoning", "agent_pipeline", "full_system")
        scenarios = [s for s in loaded if s.get("scenario_type", "scenario") not in excluded]
        conversations = []
        reasoning = []
        agent_pipeline = []
        full_system = []
        for s in loaded:
            if s.get("scenario_type", "scenario") == "conversation":
                s["turns"] = s.get("metadata", {}).get("turns", [])
                conversations.append(s)
            elif s.get("scenario_type", "scenario") == "reasoning":
                s["tasks"] = s.get("metadata", {}).get("tasks", [])
                reasoning.append(s)
            elif s.get("scenario_type", "scenario") == "agent_pipeline":
                s["requests"] = s.get("metadata", {}).get("requests", [])
                agent_pipeline.append(s)
            elif s.get("scenario_type", "scenario") == "full_system":
                s["steps"] = s.get("metadata", {}).get("steps", [])
                full_system.append(s)
    else:
        scenarios = DEFAULT_SCENARIOS
        conversations = DEFAULT_CONVERSATIONS
        reasoning = []
        agent_pipeline = []
        full_system = []

    collector = EvidenceCollector(api_host=args.api_host, api_port=args.api_port)
    try:
        report = collector.collect_all(
            title=args.title,
            scenarios=scenarios,
            conversations=conversations,
            reasoning=reasoning,
            agent_pipeline=agent_pipeline,
            full_system=full_system,
        )
    finally:
        collector.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = ReportGenerator()
    json_path = output_dir / f"{report.generated_at.replace(':', '-').replace('.', '_')}.json"
    html_path = output_dir / f"{report.generated_at.replace(':', '-').replace('.', '_')}.html"

    generator.generate_json(report, path=json_path)
    generator.generate_html(report, path=html_path)

    store = EvidenceStore(base_dir=output_dir)
    store.save(report)

    # Academic / lab-grade editions
    academic = AcademicReportGenerator()
    editions = [
        (
            "Condrox AI System Verification — Lab Edition",
            "Automated system-level verification and health attestation.",
            "Lab Edition",
        ),
        (
            "Condrox AI Reasoning Evidence — Academic Edition",
            "Evidence of step-by-step reasoning, explanation quality, and correctness signals.",
            "Academic Edition",
        ),
        (
            "Condrox AI Conversational Ability — Peer Review Edition",
            "Evidence of empathy, relevance, consistency, multi-turn stability, and context preservation.",
            "Peer Review Edition",
        ),
    ]
    academic_paths: List[str] = []
    for title, subtitle, edition in editions:
        config = AcademicReportConfig(
            title=title,
            subtitle=subtitle,
            edition=edition,
            authors=["Condrox Evidence Engine"],
            institution="Condrox AI",
            reproducibility_url=str(json_path),
        )
        academic_path = output_dir / f"{report.generated_at.replace(':', '-').replace('.', '_')}_{edition.lower().replace(' ', '_')}.academic.html"
        academic.generate(report, config, path=academic_path)
        academic_paths.append(str(academic_path))

    print(f"Evidence report generated:")
    print(f"  Overall: {report.overall}")
    print(f"  Score:   {report.score:.1f}%")
    print(f"  Runs:    {len(report.runs)}")
    print(f"  Conversations: {len(report.conversation_sessions)}")
    print(f"  Reasoning:     {len(report.reasoning_sessions)}")
    print(f"  Agent/Pipeline: {len(report.agent_pipeline_sessions)}")
    print(f"  Full System:   {len(report.full_system_sessions)}")
    print(f"  JSON:    {json_path}")
    # Evidence portal
    portal_generator = EvidencePortalGenerator()
    portal_path = portal_generator.generate(report_dir=output_dir)
    print(f"  Portal:  {portal_path}")

    print(f"  HTML:    {html_path}")
    for p in academic_paths:
        print(f"  Academic HTML: {p}")
    return 0 if report.overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
