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

"""Static Evidence Portal generator.

Reads all JSON evidence reports from a directory and produces a self-contained
HTML portal with report listing, score history, version history, measurements,
and system signature — no external dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class EvidencePortalReport:
    """Minimal summary of one evidence report for the portal."""

    filename: str
    generated_at: str
    overall: str
    score: float
    version: str
    environment: Dict[str, Any] = field(default_factory=dict)


class EvidencePortalGenerator:
    """Generate a static HTML portal from stored evidence JSON reports.

    Complexity: T(generate) ∈ O(r) for r reports, S(n) ∈ O(r).
    """

    __slots__ = ()

    def generate(
        self,
        report_dir: Path,
        output_path: Optional[Path] = None,
        title: str = "Condrox Evidence Portal",
    ) -> Path:
        """Read reports from report_dir and write a static HTML portal."""
        output_path = output_path or (report_dir / "index.html")
        reports = self._load_reports(report_dir)
        html = self._render(reports, title, report_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _load_reports(self, report_dir: Path) -> List[EvidencePortalReport]:
        """Load summaries from all evidence JSON files in the directory."""
        reports: List[EvidencePortalReport] = []
        for path in sorted(report_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                reports.append(
                    EvidencePortalReport(
                        filename=path.name,
                        generated_at=str(data.get("generated_at", "")),
                        overall=str(data.get("overall", "")),
                        score=float(data.get("score", 0.0)),
                        version=str(data.get("version", "")),
                        environment=dict(data.get("environment", {})),
                    )
                )
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        return reports

    def _render(
        self,
        reports: List[EvidencePortalReport],
        title: str,
        report_dir: Path,
    ) -> str:
        """Render the full portal HTML page."""
        scores = [(r.generated_at, r.score) for r in reports if r.generated_at]
        versions = sorted({r.version for r in reports if r.version})
        signatures = sorted(
            {json.dumps(r.environment, sort_keys=True, ensure_ascii=False) for r in reports if r.environment}
        )
        latest = reports[-1] if reports else None

        chart_svg = self._score_history_svg(scores)

        rows = "\n".join(
            f"""
            <tr>
              <td><a href=\"{r.filename}\">{self._escape(r.filename)}</a></td>
              <td>{self._escape(r.generated_at)}</td>
              <td><span class=\"badge {r.overall.lower()}\">{self._escape(r.overall)}</span></td>
              <td>{r.score:.1f}%</td>
              <td>{self._escape(r.version)}</td>
            </tr>
            """
            for r in reversed(reports)
        )

        version_items = "\n".join(f"<li>{self._escape(v)}</li>" for v in versions)
        signature_blocks = "\n".join(
            f"<pre>{self._escape(sig)}</pre>"
            for sig in signatures[:5]
        )

        latest_summary = ""
        if latest:
            latest_summary = f"""
            <section>
              <h2>Latest Report</h2>
              <p><strong>File:</strong> <a href="{latest.filename}">{self._escape(latest.filename)}</a></p>
              <p><strong>Overall:</strong> <span class="badge {latest.overall.lower()}">{self._escape(latest.overall)}</span></p>
              <p><strong>Score:</strong> {latest.score:.1f}%</p>
              <p><strong>Version:</strong> {self._escape(latest.version)}</p>
              <p><strong>Generated:</strong> {self._escape(latest.generated_at)}</p>
            </section>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self._escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
      background: #f4f6f8;
      color: #222;
    }}
    header {{ text-align: center; padding: 30px 0; }}
    h1 {{ margin-bottom: 0.2em; }}
    .subtitle {{ color: #666; }}
    .card {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      padding: 20px;
      margin-bottom: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .metric {{
      background: #fff;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      padding: 16px;
      text-align: center;
    }}
    .metric .value {{ font-size: 2em; font-weight: bold; }}
    .metric .label {{ color: #666; font-size: 0.9em; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      font-weight: bold;
      font-size: 0.85em;
    }}
    .pass {{ background: #d4edda; color: #155724; }}
    .warn {{ background: #fff3cd; color: #856404; }}
    .fail {{ background: #f8d7da; color: #721c24; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 0.95em;
    }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #f0f2f5; }}
    a {{ color: #0366d6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    pre {{
      background: #f5f5f5;
      padding: 12px;
      overflow-x: auto;
      border: 1px solid #ddd;
      font-size: 0.9em;
    }}
    svg {{ max-width: 100%; height: auto; }}
    footer {{ margin-top: 40px; text-align: center; color: #666; font-size: 0.85em; }}
  </style>
</head>
<body>
  <header>
    <h1>{self._escape(title)}</h1>
    <p class="subtitle">Evidence library, score history, and system signatures</p>
  </header>

  <div class="grid">
    <div class="metric">
      <div class="value">{len(reports)}</div>
      <div class="label">Reports</div>
    </div>
    <div class="metric">
      <div class="value">{latest.score:.1f}%</div>
      <div class="label">Latest Score</div>
    </div>
    <div class="metric">
      <div class="value">{len(versions)}</div>
      <div class="label">Versions</div>
    </div>
    <div class="metric">
      <div class="value">{len(signatures)}</div>
      <div class="label">System Signatures</div>
    </div>
  </div>

  {latest_summary}

  <section class="card">
    <h2>Score History</h2>
    {chart_svg}
  </section>

  <section class="card">
    <h2>Version History</h2>
    <ul>{version_items}</ul>
  </section>

  <section class="card">
    <h2>System Signatures</h2>
    {signature_blocks}
  </section>

  <section class="card">
    <h2>Evidence Reports</h2>
    <table>
      <tr>
        <th>Filename</th>
        <th>Generated</th>
        <th>Overall</th>
        <th>Score</th>
        <th>Version</th>
      </tr>
      {rows}
    </table>
  </section>

  <footer>
    Condrox Evidence Engine · Generated automatically from {self._escape(str(report_dir))}
  </footer>
</body>
</html>"""

    def _score_history_svg(self, scores: List[Tuple[str, float]]) -> str:
        """Render a simple SVG line chart of score history."""
        if not scores:
            return "<p>No score history available.</p>"

        width = 800
        height = 240
        padding = 40
        n = len(scores)
        if n == 1:
            xs = [width / 2]
        else:
            usable = width - 2 * padding
            xs = [padding + i * (usable / (n - 1)) for i in range(n)]
        ys = [
            height - padding - (s / 100.0) * (height - 2 * padding)
            for _, s in scores
        ]

        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        polyline = f'<polyline points="{points}" fill="none" stroke="#0366d6" stroke-width="2" />'
        circles = "\n".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#0366d6" />'
            for x, y in zip(xs, ys)
        )

        # Axis labels
        y_labels = "\n".join(
            f'<text x="{padding - 8}" y="{height - padding - (p / 100.0) * (height - 2 * padding) + 4}" '
            f'text-anchor="end" font-size="10" fill="#666">{p}%</text>'
            for p in (0, 25, 50, 75, 100)
        )
        x_labels = ""
        if n <= 8:
            x_labels = "\n".join(
                f'<text x="{x:.1f}" y="{height - padding + 18}" text-anchor="middle" font-size="9" fill="#666">{self._escape(scores[i][0][:10])}</text>'
                for i, x in enumerate(xs)
            )
        else:
            step = max(1, n // 6)
            x_labels = "\n".join(
                f'<text x="{xs[i]:.1f}" y="{height - padding + 18}" text-anchor="middle" font-size="9" fill="#666">{self._escape(scores[i][0][:10])}</text>'
                for i in range(0, n, step)
            )

        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
          <rect x="{padding}" y="{padding}" width="{width - 2 * padding}" height="{height - 2 * padding}" fill="#fafafa" stroke="#ddd" />
          {y_labels}
          {polyline}
          {circles}
          {x_labels}
        </svg>
        """

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
