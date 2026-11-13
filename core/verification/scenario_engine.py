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

"""Scenario engine for Condrox AI verification.

Phase 1 skeleton: loads JSON scenario files and provides a registry.
Future phases will execute scenarios and score responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class Scenario:
    """A single verification scenario."""

    name: str
    category: str
    scenario_type: str = "scenario"  # scenario | conversation
    input_text: str = ""
    expected_intent: str = ""
    expected_mode: str = ""
    rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScenarioEngine:
    """Loads and manages verification scenarios.

    Complexity: T(load) ∈ O(n) for n scenario files, S(n) ∈ O(scenarios).
    """

    __slots__ = ("_scenarios", "_scenario_dir")

    def __init__(self, scenario_dir: Optional[str | Path] = None) -> None:
        self._scenario_dir = Path(scenario_dir) if scenario_dir else self._default_dir()
        self._scenarios: Dict[str, Scenario] = {}
        self._load_all()

    @staticmethod
    def _default_dir() -> Path:
        """Return the default scenario directory."""
        return Path(__file__).resolve().parents[2] / "verification" / "scenarios"

    def _load_all(self) -> None:
        """Load all JSON scenario files from the scenario directory."""
        if not self._scenario_dir.exists():
            return
        for path in self._scenario_dir.rglob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    self._register(data)
                elif isinstance(data, list):
                    for item in data:
                        self._register(item)
            except (OSError, json.JSONDecodeError):
                continue

    def _register(self, data: Dict[str, Any]) -> None:
        """Register a scenario from a dict."""
        name = str(data.get("name", ""))
        if not name:
            return
        scenario_type = str(data.get("type", data.get("scenario_type", "scenario")))
        metadata = dict(data.get("metadata", {}))
        # Preserve type-specific payload (conversation turns, reasoning tasks,
        # agent/pipeline requests, full-system steps) in metadata.
        if "turns" in data:
            metadata["turns"] = data["turns"]
        if "tasks" in data:
            metadata["tasks"] = data["tasks"]
        if "requests" in data:
            metadata["requests"] = data["requests"]
        if "steps" in data:
            metadata["steps"] = data["steps"]
        self._scenarios[name] = Scenario(
            name=name,
            category=str(data.get("category", "general")),
            scenario_type=scenario_type,
            input_text=str(data.get("input_text", data.get("input", ""))),
            expected_intent=str(data.get("expected_intent", "")),
            expected_mode=str(data.get("expected_mode", "")),
            rules=list(data.get("rules", [])),
            metadata=metadata,
        )

    def list_scenarios(self) -> List[Scenario]:
        """Return all loaded scenarios."""
        return list(self._scenarios.values())

    def list_by_type(self, scenario_type: str) -> List[Scenario]:
        """Return scenarios of a given type."""
        return [s for s in self._scenarios.values() if s.scenario_type == scenario_type]

    def count(self) -> int:
        """Return the number of loaded scenarios."""
        return len(self._scenarios)
