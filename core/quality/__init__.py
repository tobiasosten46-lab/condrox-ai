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
Condrox Quality Gates — integrated into Condrox AI.

Provides 4 active gates: TYPES, COMPLEXITY, SAFETY, STRUCTURE.
Used by CondroxCodeAgent to verify all generated code.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from core.quality.gate_base import QualityGate, GateResult, GateSeverity
from core.quality.types_gate import TypesGate
from core.quality.complexity_gate import ComplexityGate
from core.quality.safety_gate import SafetyGate
from core.quality.structure_gate import StructureGate
from core.quality.gate_engine import GateEngine

__all__ = [
    "QualityGate",
    "GateResult",
    "GateSeverity",
    "TypesGate",
    "ComplexityGate",
    "SafetyGate",
    "StructureGate",
    "GateEngine",
]
