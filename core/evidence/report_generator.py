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

"""Academic-grade report generation for Condrox Evidence Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core.evidence.evidence_report import EvidenceReport


class ReportGenerator:
    """Generate JSON and HTML evidence reports.

    Complexity: T(generate) ∈ O(r) for r evidence runs.
    """

    __slots__ = ()

    def generate_json(self, report: EvidenceReport, path: Optional[Path] = None) -> str:
        """Generate a JSON report and optionally write it to disk."""
        text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def generate_html(self, report: EvidenceReport, path: Optional[Path] = None) -> str:
        """Generate an academic-style HTML report and optionally write it."""
        html = self._render_html(report)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
        return html

    def _render_html(self, report: EvidenceReport) -> str:
        """Render EvidenceReport as a clean academic HTML document."""
        runs_html = ""
        for run in report.runs:
            runs_html += f"""
            <section class="run">
              <h3>{self._escape(run.name)} <span class="status {run.status.lower()}">{run.status}</span></h3>
              <p><strong>Category:</strong> {self._escape(run.category)}</p>
              <p><strong>Score:</strong> {run.score:.1f}%</p>
              <p><strong>Timestamp:</strong> {self._escape(run.timestamp)}</p>
              <h4>Input</h4>
              <pre>{self._escape(json.dumps(run.input_data, indent=2))}</pre>
              <h4>Output</h4>
              <pre>{self._escape(json.dumps(run.output_data, indent=2))}</pre>
              <h4>Metrics</h4>
              <pre>{self._escape(json.dumps(run.metrics, indent=2))}</pre>
              {"<h4>Deviations</h4><ul>" + "".join(f"<li>{self._escape(d)}</li>" for d in run.deviations) + "</ul>" if run.deviations else ""}
            </section>
            """

        deviations_html = ""
        if report.deviations:
            deviations_html = "<h2>Deviations</h2><ul>" + "".join(
                f"<li>{self._escape(d)}</li>" for d in report.deviations
            ) + "</ul>"

        conversations_html = ""
        for session in report.conversation_sessions:
            turns_html = ""
            for turn in session.turns:
                metrics_html = "<ul>" + "".join(
                    f'<li>{self._escape(m["name"])}: <span class="status {m["status"].lower()}">{m["status"]}</span> · {m["score"]:.1f}%</li>'
                    for m in turn.get("metrics", [])
                ) + "</ul>"
                turns_html += f"""
                <div class="turn">
                  <p><strong>User:</strong> {self._escape(turn.get("input", ""))}</p>
                  <p><strong>Assistant:</strong> {self._escape(turn.get("output", ""))}</p>
                  <p><strong>Emotion:</strong> {self._escape(turn.get("emotion", ""))} · <strong>Mode:</strong> {self._escape(turn.get("mode", ""))}</p>
                  <h5>Metrics</h5>
                  {metrics_html}
                </div>
                """
            summary = session.summary
            summary_html = f"""
            <div class="summary">
              <p><strong>Summary:</strong> Score {summary.get("score", 0.0):.1f}% · Status <span class="status {summary.get('status', 'FAIL').lower()}">{summary.get('status', 'FAIL')}</span></p>
              <pre>{self._escape(json.dumps(summary.get("metric_averages", {}), indent=2))}</pre>
            </div>
            """
            conversations_html += f"""
            <section class="conversation-session">
              <h3>{self._escape(session.name)}</h3>
              {summary_html}
              {turns_html}
            </section>
            """

        reasoning_html = ""
        for session in report.reasoning_sessions:
            tasks_html = ""
            for task in session.tasks:
                metrics_html = "<ul>" + "".join(
                    f'<li>{self._escape(m["name"])}: <span class="status {m["status"].lower()}">{m["status"]}</span> · {m["score"]:.1f}%</li>'
                    for m in task.get("metrics", [])
                ) + "</ul>"
                tasks_html += f"""
                <div class="task">
                  <p><strong>Input:</strong> {self._escape(task.get("input", ""))}</p>
                  <p><strong>Output:</strong> {self._escape(task.get("output", ""))}</p>
                  <p><strong>Intent/Mode/Domain:</strong> {self._escape(task.get("intent", ""))} / {self._escape(task.get("mode", ""))} / {self._escape(task.get("domain", ""))}</p>
                  <h5>Metrics</h5>
                  {metrics_html}
                </div>
                """
            summary = session.summary
            summary_html = f"""
            <div class="summary">
              <p><strong>Summary:</strong> Score {summary.get("score", 0.0):.1f}% · Status <span class="status {summary.get('status', 'FAIL').lower()}">{summary.get('status', 'FAIL')}</span></p>
              <pre>{self._escape(json.dumps(summary.get("metric_averages", {}), indent=2))}</pre>
            </div>
            """
            reasoning_html += f"""
            <section class="reasoning-session">
              <h3>{self._escape(session.name)}</h3>
              {summary_html}
              {tasks_html}
            </section>
            """

        agent_pipeline_html = ""
        for session in report.agent_pipeline_sessions:
            requests_html = ""
            for req in session.requests:
                metrics_html = "<ul>" + "".join(
                    f'<li>{self._escape(m["name"])}: <span class="status {m["status"].lower()}">{m["status"]}</span> · {m["score"]:.1f}%</li>'
                    for m in req.get("metrics", [])
                ) + "</ul>"
                requests_html += f"""
                <div class="request">
                  <p><strong>Input:</strong> {self._escape(req.get("input", ""))}</p>
                  <p><strong>Agent:</strong> {self._escape(req.get("agent_id", ""))} ({self._escape(req.get("agent_mode", ""))}) · <strong>Mode:</strong> {self._escape(req.get("mode", ""))} · <strong>Domain:</strong> {self._escape(req.get("domain", ""))}</p>
                  <h5>Metrics</h5>
                  {metrics_html}
                </div>
                """
            traces_html = "<ul>" + "".join(
                f'<li>{self._escape(t.get("input", ""))[:60]}: {t.get("stages", 0)} stages · {t.get("duration_ms", 0.0):.1f} ms</li>'
                for t in session.pipeline_traces
            ) + "</ul>"
            summary = session.summary
            summary_html = f"""
            <div class="summary">
              <p><strong>Summary:</strong> Score {summary.get("score", 0.0):.1f}% · Status <span class="status {summary.get('status', 'FAIL').lower()}">{summary.get('status', 'FAIL')}</span></p>
              <pre>{self._escape(json.dumps(summary.get("metric_averages", {}), indent=2))}</pre>
            </div>
            """
            agent_pipeline_html += f"""
            <section class="agent-pipeline-session">
              <h3>{self._escape(session.name)}</h3>
              {summary_html}
              {requests_html}
              <h4>Pipeline Traces</h4>
              {traces_html}
            </section>
            """

        full_system_html = ""
        for session in report.full_system_sessions:
            checks_html = ""
            for check in session.checks:
                checks_html += f"""
                <div class="check">
                  <p><strong>{self._escape(check.get("name", ""))}</strong> · <span class="status {check.get('status', 'FAIL').lower()}">{check.get('status', 'FAIL')}</span> · {check.get('score', 0.0):.1f}%</p>
                  <pre>{self._escape(json.dumps(check.get("details", {}), indent=2))}</pre>
                </div>
                """
            summary = session.summary
            summary_html = f"""
            <div class="summary">
              <p><strong>Summary:</strong> Score {summary.get("score", 0.0):.1f}% · Status <span class="status {summary.get('status', 'FAIL').lower()}">{summary.get('status', 'FAIL')}</span></p>
              <pre>{self._escape(json.dumps(summary, indent=2))}</pre>
            </div>
            """
            full_system_html += f"""
            <section class="full-system-session">
              <h3>{self._escape(session.name)}</h3>
              {summary_html}
              <h4>Integration Checks ({len(session.checks)})</h4>
              {checks_html}
            </section>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{self._escape(report.title)}</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto; line-height: 1.6; color: #222; }}
    header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
    h1 {{ font-size: 1.8em; margin-bottom: 0.2em; }}
    h2 {{ font-size: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    h3 {{ font-size: 1.2em; margin-top: 1.5em; }}
    h4 {{ font-size: 1.05em; }}
    .meta {{ color: #555; font-size: 0.95em; }}
    .status {{ font-weight: bold; padding: 2px 8px; border-radius: 4px; }}
    .pass {{ background: #d4edda; color: #155724; }}
    .warn {{ background: #fff3cd; color: #856404; }}
    .fail {{ background: #f8d7da; color: #721c24; }}
    .ok {{ background: #d4edda; color: #155724; }}
    pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; border-left: 3px solid #333; }}
    .run {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 6px; }}
    .conversation-session {{ margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; border-radius: 6px; background: #fafafa; }}
    .reasoning-session {{ margin-bottom: 30px; border: 1px solid #bcd; padding: 15px; border-radius: 6px; background: #f8fbff; }}
    .agent-pipeline-session {{ margin-bottom: 30px; border: 1px solid #cbd; padding: 15px; border-radius: 6px; background: #f7fff7; }}
    .full-system-session {{ margin-bottom: 30px; border: 2px solid #4a4; padding: 15px; border-radius: 6px; background: #f9fff9; }}
    .turn {{ margin: 10px 0; padding: 10px; border-left: 3px solid #666; background: #fff; }}
    .task {{ margin: 10px 0; padding: 10px; border-left: 3px solid #48c; background: #fff; }}
    .request {{ margin: 10px 0; padding: 10px; border-left: 3px solid #282; background: #fff; }}
    .check {{ margin: 10px 0; padding: 10px; border-left: 3px solid #844; background: #fff; }}
    .summary {{ margin-bottom: 15px; }}
    footer {{ margin-top: 40px; border-top: 1px solid #ccc; padding-top: 10px; font-size: 0.85em; color: #666; }}
  </style>
</head>
<body>
  <header>
    <h1>{self._escape(report.title)}</h1>
    <p class="meta">Overall: <span class="status {report.overall.lower()}">{report.overall}</span> · Score: {report.score:.1f}% · Version: {self._escape(report.version)}</p>
    <p class="meta">Generated: {self._escape(report.generated_at)}</p>
  </header>

  <h2>System Signature</h2>
  <pre>{self._escape(json.dumps(report.environment, indent=2))}</pre>

  <h2>Metadata</h2>
  <pre>{self._escape(json.dumps(report.metadata, indent=2))}</pre>

  {deviations_html}

  <h2>Evidence Runs ({len(report.runs)})</h2>
  {runs_html}

  <h2>Conversation Sessions ({len(report.conversation_sessions)})</h2>
  {conversations_html}

  <h2>Reasoning Sessions ({len(report.reasoning_sessions)})</h2>
  {reasoning_html}

  <h2>Agent & Pipeline Sessions ({len(report.agent_pipeline_sessions)})</h2>
  {agent_pipeline_html}

  <h2>Full System Integration Sessions ({len(report.full_system_sessions)})</h2>
  {full_system_html}

  <footer>
    Condrox Evidence Engine · Academic-grade system verification report.
  </footer>
</body>
</html>"""

    @staticmethod
    def _escape(text: str) -> str:
        """Escape HTML entities in a string."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
