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

"""Condrox core agents — mode-specific reasoning modules."""

from __future__ import annotations

from core.agents.base import AgentResult, BaseAgent
from core.agents.code_agent import CodeAgent
from core.agents.explain_agent import ExplainAgent
from core.agents.analyze_agent import AnalyzeAgent
from core.agents.architect_agent import ArchitectAgent
from core.agents.default_agent import DefaultAgent
from core.agents.registry import AgentRegistry

__all__ = [
    "AgentResult",
    "BaseAgent",
    "CodeAgent",
    "ExplainAgent",
    "AnalyzeAgent",
    "ArchitectAgent",
    "DefaultAgent",
    "AgentRegistry",
]
