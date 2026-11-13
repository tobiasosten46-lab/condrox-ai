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
Shared pytest configuration — sets environment variables to prevent
slow disk I/O (knowledge loading from disk datasets)
during test execution.

All tests that call boot_system() will skip knowledge loading and
AI-OS hot-reloader setup, reducing boot time from minutes to milliseconds.
"""

from __future__ import annotations

import os

# Prevent knowledge loading from disk (120+ GB) — tests use in-memory backends
os.environ["CONDROX_SKIP_KNOWLEDGE_LOAD"] = "1"

# Use temp path for AI-OS state DB — avoid polluting project directory
import tempfile
import uuid
_state_db = os.path.join(tempfile.gettempdir(), f"condrox_os_test_{uuid.uuid4().hex[:8]}.db")
os.environ.setdefault("CONDROX_OS_STATE_DB", _state_db)
