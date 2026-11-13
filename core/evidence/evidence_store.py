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

"""Persistent store for evidence reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.evidence.evidence_report import EvidenceReport


class EvidenceStore:
    """File-based evidence store with JSON persistence.

    Complexity: T(save) ∈ O(r) for r runs, T(load) ∈ O(r).
    """

    __slots__ = ("_base_dir",)

    def __init__(self, base_dir: Optional[str | Path] = None) -> None:
        default = Path(__file__).resolve().parents[2] / "Reports" / "evidence"
        self._base_dir: Path = Path(base_dir) if base_dir else default
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: EvidenceReport, filename: Optional[str] = None) -> Path:
        """Save a report as JSON and return the file path."""
        if filename is None:
            filename = f"evidence_{report.generated_at.replace(':', '-').replace('.', '_')}.json"
        path = self._base_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        return path

    def list_reports(self) -> List[Path]:
        """Return all stored evidence report paths sorted by name."""
        return sorted(self._base_dir.glob("evidence_*.json"))

    def load(self, filename: str) -> Dict[str, Any]:
        """Load a stored report as a dictionary."""
        path = self._base_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
