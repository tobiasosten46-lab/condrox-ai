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

"""Academic and lab-grade report generator for Condrox Evidence Engine.

Produces PDF-compatible HTML reports with methodology, measurement,
references, signature, and reproducibility sections suitable for
research, peer-review, and laboratory environments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.evidence.evidence_report import EvidenceReport


@dataclass(frozen=True, slots=True)
class AcademicReportConfig:
    """Configuration for an academic report edition."""

    title: str
    subtitle: str
    edition: str  # Lab Edition, Academic Edition, Peer Review Edition
    authors: List[str] = field(default_factory=lambda: ["Condrox Evidence Engine"])
    institution: str = "Condrox AI"
    methodology: str = "Automated scenario-based verification with deterministic reproducibility checks."
    references: List[str] = field(default_factory=list)
    signature: str = "Condrox Evidence Engine v1.0"
    reproducibility_url: str = ""


class AcademicReportGenerator:
    """Generate academic/lab-grade HTML reports from EvidenceReport data.

    Complexity: T(generate) ∈ O(r + s) for r runs and s sessions.
    """

    __slots__ = ()

    def generate(
        self,
        report: EvidenceReport,
        config: AcademicReportConfig,
        path: Optional[Path] = None,
    ) -> str:
        """Generate a PDF-compatible HTML report and optionally write it to disk."""
        html = self._render(report, config)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
        return html

    def _render(self, report: EvidenceReport, config: AcademicReportConfig) -> str:
        """Render the full academic HTML document."""
        sections = self._render_sections(report)
        references_html = self._render_references(config.references)
        methodology_html = self._render_methodology(config.methodology)
        signature_html = self._render_signature(config.signature, report)
        reproducibility_html = self._render_reproducibility(config.reproducibility_url, report)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{self._escape(config.title)}</title>
  <style>
    @page {{ size: A4; margin: 2.5cm; }}
    body {{
      font-family: "Times New Roman", Georgia, serif;
      font-size: 11pt;
      line-height: 1.6;
      color: #000;
      max-width: 800px;
      margin: 40px auto;
      padding: 0 20px;
    }}
    header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
    h1 {{ font-size: 18pt; margin-bottom: 6px; }}
    h2 {{ font-size: 14pt; border-bottom: 1px solid #999; padding-bottom: 4px; margin-top: 28px; }}
    h3 {{ font-size: 12pt; margin-top: 20px; }}
    .subtitle {{ font-size: 13pt; color: #333; margin-top: 4px; }}
    .meta {{ font-size: 10pt; color: #333; margin-top: 8px; }}
    .authors {{ font-size: 11pt; margin-top: 12px; }}
    .abstract {{
      background: #f9f9f9;
      border-left: 4px solid #333;
      padding: 14px;
      margin: 20px 0;
      font-size: 10.5pt;
    }}
    .status {{
      display: inline-block;
      padding: 2px 10px;
      border: 1px solid #333;
      font-weight: bold;
      font-size: 10pt;
    }}
    .pass {{ background: #e8f5e9; }}
    .warn {{ background: #fff8e1; }}
    .fail {{ background: #ffebee; }}
    .ok {{ background: #e8f5e9; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      font-size: 10pt;
    }}
    th, td {{ border: 1px solid #999; padding: 6px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    pre {{
      background: #f5f5f5;
      padding: 10px;
      overflow-x: auto;
      font-size: 9pt;
      border: 1px solid #ccc;
    }}
    .section {{ margin-bottom: 24px; }}
    .signature {{ margin-top: 40px; border-top: 1px solid #333; padding-top: 10px; font-size: 10pt; }}
    footer {{ margin-top: 40px; font-size: 9pt; color: #555; border-top: 1px solid #ccc; padding-top: 10px; }}
    .page-break {{ page-break-before: always; }}
  </style>
</head>
<body>
  <header>
    <h1>{self._escape(config.title)}</h1>
    <p class="subtitle">{self._escape(config.subtitle)} — {self._escape(config.edition)}</p>
    <p class="authors">{self._escape(", ".join(config.authors))}<br>{self._escape(config.institution)}</p>
    <p class="meta">Generated: {self._escape(report.generated_at)} · Version: {self._escape(report.version)}</p>
  </header>

  <section class="abstract">
    <strong>Abstract.</strong>
    This report presents an automated verification of Condrox AI ({self._escape(report.version)}).
    Overall status: <span class="status {report.overall.lower()}">{report.overall}</span>.
    Aggregate score: {report.score:.1f}%.
    The evaluation covered {len(report.runs)} scenario runs, {len(report.conversation_sessions)} conversation
    sessions, {len(report.reasoning_sessions)} reasoning sessions, {len(report.agent_pipeline_sessions)} agent/pipeline
    sessions, and {len(report.full_system_sessions)} full-system integration sessions.
  </section>

  {methodology_html}

  <h2>Executive Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Overall Status</td><td><span class="status {report.overall.lower()}">{report.overall}</span></td></tr>
    <tr><td>Aggregate Score</td><td>{report.score:.1f}%</td></tr>
    <tr><td>Scenario Runs</td><td>{len(report.runs)}</td></tr>
    <tr><td>Conversation Sessions</td><td>{len(report.conversation_sessions)}</td></tr>
    <tr><td>Reasoning Sessions</td><td>{len(report.reasoning_sessions)}</td></tr>
    <tr><td>Agent/Pipeline Sessions</td><td>{len(report.agent_pipeline_sessions)}</td></tr>
    <tr><td>Full-System Sessions</td><td>{len(report.full_system_sessions)}</td></tr>
    <tr><td>Deviations</td><td>{len(report.deviations)}</td></tr>
  </table>

  <h2>System Signature &amp; Environment</h2>
  <pre>{self._escape(json.dumps(report.environment, indent=2))}</pre>

  {sections}

  {references_html}

  {reproducibility_html}

  {signature_html}

  <footer>
    Condrox Evidence Engine — {self._escape(config.edition)}. This report was generated automatically and is
    intended for academic, laboratory, and technical evaluation purposes.
  </footer>
</body>
</html>"""

    def _render_sections(self, report: EvidenceReport) -> str:
        """Render evidence sections as academic subsections."""
        parts: List[str] = []

        if report.runs:
            parts.append("<h2>Scenario Runs</h2>")
            for run in report.runs:
                parts.append(f"""
                <div class="section">
                  <h3>{self._escape(run.name)} <span class="status {run.status.lower()}">{run.status}</span></h3>
                  <p>Category: {self._escape(run.category)} · Score: {run.score:.1f}%</p>
                  <pre>{self._escape(json.dumps(run.output_data, indent=2, ensure_ascii=False))}</pre>
                </div>
                """)

        for session in report.conversation_sessions:
            parts.append(f"<h2 class=\"page-break\">Conversation Session: {self._escape(session.name)}</h2>")
            parts.append(self._summary_table(session.summary))
            for turn in session.turns:
                parts.append(f"""
                <div class="section">
                  <h3>Turn</h3>
                  <p><strong>User:</strong> {self._escape(turn.get("input", ""))}</p>
                  <p><strong>Assistant:</strong> {self._escape(turn.get("output", ""))}</p>
                  <table>
                    <tr><th>Metric</th><th>Status</th><th>Score</th></tr>
                    {self._metric_rows(turn.get("metrics", []))}
                  </table>
                </div>
                """)

        for session in report.reasoning_sessions:
            parts.append(f"<h2 class=\"page-break\">Reasoning Session: {self._escape(session.name)}</h2>")
            parts.append(self._summary_table(session.summary))
            for task in session.tasks:
                parts.append(f"""
                <div class="section">
                  <h3>Task</h3>
                  <p><strong>Input:</strong> {self._escape(task.get("input", ""))}</p>
                  <p><strong>Output:</strong> {self._escape(task.get("output", ""))}</p>
                  <table>
                    <tr><th>Metric</th><th>Status</th><th>Score</th></tr>
                    {self._metric_rows(task.get("metrics", []))}
                  </table>
                </div>
                """)

        for session in report.full_system_sessions:
            parts.append(f"<h2 class=\"page-break\">Full-System Integration: {self._escape(session.name)}</h2>")
            parts.append(self._summary_table(session.summary))
            parts.append("<h3>Integration Checks</h3>")
            parts.append("<table><tr><th>Check</th><th>Status</th><th>Score</th><th>Details</th></tr>")
            for check in session.checks:
                details = self._escape(json.dumps(check.get("details", {}), ensure_ascii=False))
                parts.append(
                    f"<tr><td>{self._escape(check.get('name', ''))}</td>"
                    f"<td><span class=\"status {check.get('status', 'FAIL').lower()}\">{check.get('status', 'FAIL')}</span></td>"
                    f"<td>{check.get('score', 0.0):.1f}%</td>"
                    f"<td><pre>{details}</pre></td></tr>"
                )
            parts.append("</table>")

        return "\n".join(parts)

    def _summary_table(self, summary: Dict[str, Any]) -> str:
        """Render a summary dictionary as a compact table."""
        rows = "\n".join(
            f"<tr><td>{self._escape(str(k))}</td><td>{self._escape(json.dumps(v, ensure_ascii=False))}</td></tr>"
            for k, v in summary.items()
        )
        return f"""
        <table>
          <tr><th>Summary Field</th><th>Value</th></tr>
          {rows}
        </table>
        """

    def _metric_rows(self, metrics: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"<tr><td>{self._escape(m.get('name', ''))}</td>"
            f"<td><span class=\"status {m.get('status', 'FAIL').lower()}\">{m.get('status', 'FAIL')}</span></td>"
            f"<td>{m.get('score', 0.0):.1f}%</td></tr>"
            for m in metrics
        )

    @staticmethod
    def _render_methodology(methodology: str) -> str:
        return f"""
        <h2>Methodology</h2>
        <p>{methodology}</p>
        <p>Measurements are deterministic and based on surface-level output analysis. Each metric uses
        explicit, reproducible rules applied uniformly across scenarios. Scores are normalized to a 0–100
        scale and aggregated by arithmetic mean unless otherwise noted.</p>
        """

    @staticmethod
    def _render_references(references: List[str]) -> str:
        if not references:
            references = [
                "Condrox AI System Verification Roadmap",
                "Condrox Evidence Engine Specification",
                "Project source: core/evidence/ module",
            ]
        items = "\n".join(f"<li>{ref}</li>" for ref in references)
        return f"""
        <h2 class="page-break">References</h2>
        <ol>
          {items}
        </ol>
        """

    def _render_signature(self, signature: str, report: EvidenceReport) -> str:
        return f"""
        <div class="signature">
          <h2>Signature and Attestation</h2>
          <p><strong>Signed by:</strong> {self._escape(signature)}</p>
          <p><strong>Report hash identifier:</strong> {self._escape(report.generated_at)}</p>
          <p><strong>Version attested:</strong> {self._escape(report.version)}</p>
          <p>This report was generated automatically by the Condrox Evidence Engine. The contents represent
          the state of the Condrox AI system at the time of generation.</p>
        </div>
        """

    def _render_reproducibility(self, url: str, report: EvidenceReport) -> str:
        repro = url or "Run `python scripts/generate_evidence.py` with the same environment and scenario files."
        return f"""
        <h2>Reproducibility</h2>
        <p>To reproduce this report:</p>
        <pre>{self._escape(repro)}</pre>
        <p>Environment snapshot:</p>
        <pre>{self._escape(json.dumps(report.environment, indent=2))}</pre>
        """

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
