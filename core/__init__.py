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
Condrox AI — Core Intelligence Layer.

This package contains the unified Brain orchestrator, the 18-stage
cognitive pipeline, and all core intelligence modules (memory, knowledge,
cognition, engines, personality, security, ethics, boot).

All public components expose typed interfaces and are wired into the
BrainController for end-to-end request processing.

Version: 3.4.0
"""

__version__ = "3.4.0"

from core.brain.brain_config import BrainConfig
from core.brain.brain_state import BrainState
from core.brain.brain_controller import BrainController

__all__ = ["BrainConfig", "BrainState", "BrainController", "__version__"]
