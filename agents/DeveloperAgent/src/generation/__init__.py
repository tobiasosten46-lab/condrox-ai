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
DeveloperAgent generation subpackage — code generation and transformation.

Test generator, refactoring engine, diff engine, code pattern library.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from agents.DeveloperAgent.src.generation.test_generator import (
    GeneratedTest,
    TestCase,
    TestGenerator,
)
from agents.DeveloperAgent.src.generation.diff_engine import (
    DiffEngine,
    DiffHunk,
    DiffResult,
)
from agents.DeveloperAgent.src.generation.code_pattern_library import (
    CodePattern,
    CodePatternLibrary,
    PatternCategory,
)

__all__ = [
    "CodePattern",
    "CodePatternLibrary",
    "DiffEngine",
    "DiffHunk",
    "DiffResult",
    "GeneratedTest",
    "TestCase",
    "TestGenerator",
]
