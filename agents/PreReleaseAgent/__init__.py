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

"""Condrox AI Pre-Release Agent — public, isolated, no-auth chat demo.

This agent exposes the full Condrox AI cognitive pipeline to the public
website chat page with strict isolation guarantees:

- Dedicated BrainController instance (no shared memory with main API)
- store_memory=False on every request (zero persistent memory writes)
- Per-session ephemeral conversation context (in-memory, TTL'd 30 min)
- Conversation history capped at last 5 turns
- Input capped at 500 chars, output capped at 1000 chars
- Rate limited per IP (10 req/min)

No accounts, no sign-up, no cross-user contamination. Pure evidence that
the AI actually works.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from agents.PreReleaseAgent.condrox_pre_release_agent import (
    PreReleaseAgent,
    PreReleaseSession,
    PreReleaseResult,
)

__all__ = ["PreReleaseAgent", "PreReleaseSession", "PreReleaseResult"]
__version__ = "1.0.0"
