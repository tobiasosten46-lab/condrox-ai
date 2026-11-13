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

"""Demo script for the unified Brain system."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import BrainController


def main() -> None:
    """Run a small end-to-end demo of the Brain system."""
    brain = BrainController()
    boot = brain.boot_system()
    print(f"Boot success: {boot.success}")
    print(f"Health score: {brain.health().score}")

    brain.ingest_knowledge(
        "Python functions are defined with the def keyword and can return values.",
        domain="programming",
    )
    brain.add_memory("User is learning Python basics.", tags=["learning"], weight=0.9)

    queries = [
        "Hello!",
        "Explain Python functions.",
        "How do I write a Python function?",
    ]

    out_path = Path(__file__).resolve().parent / "demo_output.txt"
    with out_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write("\n---\n")
            handle.write(f"User: {query}\n")
            state = brain.process_request(query)
            handle.write(f"Mode: {state.mode} | Domain: {state.domain} | Intent: {state.intent}\n")
            handle.write("Final output:\n")
            handle.write(state.final_output)
            handle.write("\n")

    print(f"Demo complete. Output written to: {out_path}")


if __name__ == "__main__":
    main()
