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
run_quality_check.py — Condrox Code Quality System CLI

Entry point for Phase 1:
  - Scans the project
  - Loads the ruleset
  - Prints a simple PASS / WARN / FAIL report per file

Usage:
    python scripts/code_quality/run_quality_check.py
    python scripts/code_quality/run_quality_check.py --root .
    python scripts/code_quality/run_quality_check.py --format json

Complexity: T(n) ∈ O(n * m), n = files, m = average file length.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output even on Windows terminals / pipes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make the project root importable as a namespace package root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.code_quality.scanner import scan_project


def load_ruleset(root: Path) -> dict:
    """Load the JSON ruleset."""
    ruleset_path = root / "scripts" / "code_quality" / "ruleset.json"
    try:
        with ruleset_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load ruleset: {exc}", file=sys.stderr)
        return {}


def grade_file(file_info, ruleset: dict) -> str:
    """Return a simple PASS/WARN/FAIL grade for a file."""
    thresholds = ruleset.get("thresholds", {})
    max_lines = thresholds.get("max_file_lines", 1000)

    if file_info.lines > max_lines:
        return "FAIL"
    if file_info.size_bytes == 0:
        return "WARN"
    return "PASS"


def run_check(root: Path, output_format: str = "text") -> dict:
    """Run the Phase 1 quality check and return summary statistics."""
    ruleset = load_ruleset(root)
    files = scan_project(root)
    results = []
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for file_info in files:
        grade = grade_file(file_info, ruleset)
        counts[grade] = counts.get(grade, 0) + 1
        results.append({
            "file": file_info.relative,
            "extension": file_info.extension,
            "lines": file_info.lines,
            "size_bytes": file_info.size_bytes,
            "grade": grade,
        })

    summary = {
        "project_root": str(root),
        "ruleset": ruleset.get("name", "unknown"),
        "total_files": len(files),
        "grades": counts,
        "files": results,
    }

    if output_format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(f"Condrox Code Quality System — {ruleset.get('name', 'unknown')}")
        print(f"Project root: {root}")
        print(f"Total files scanned: {len(files)}")
        print(f"PASS: {counts['PASS']}  WARN: {counts['WARN']}  FAIL: {counts['FAIL']}")
        print("-" * 60)
        for item in results:
            status = item["grade"].ljust(5)
            print(f"[{status}] {item['file']} ({item['lines']} lines)")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Condrox Code Quality System")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root directory to scan",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: root path does not exist: {root}", file=sys.stderr)
        return 1

    run_check(root, args.format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
