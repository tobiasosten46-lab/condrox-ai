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

"""Smoke test — boots the BrainController, ingests knowledge, processes one request."""

import sys
from core import BrainController


def main() -> None:
    bc = BrainController()
    print("boot:", bc.boot_system().success)
    bc.ingest_knowledge("Python is a programming language.", domain="programming")
    bc.add_memory("User prefers concise responses.", tags=["preference"])
    state = bc.process_request("Explain Python.")
    print("---")
    print(state.final_output[:500])
    print("---")
    print("mode:", state.mode, "domain:", state.domain, "intent:", state.intent)


if __name__ == "__main__":
    main()
