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
Condrox AI Terminal — simple chat client for the local API.

Usage:
    python scripts/ai_terminal.py
    python scripts/ai_terminal.py --url http://localhost:8000 --api-key <key>
"""

from __future__ import annotations

import argparse
import os
import sys
from urllib.parse import urljoin

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Condrox AI chat terminal")
    parser.add_argument("--url", default=os.environ.get("CONDROX_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("CONDROX_API_KEY", ""))
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print("Condrox AI Terminal")
    print(f"Connected to {args.url}")
    print("Type 'exit' to quit, 'clear' to reset context.\n")

    context: list[dict[str, str]] = []
    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        text = user_input.strip()
        if not text:
            continue
        if text.lower() == "exit":
            print("Goodbye.")
            break
        if text.lower() == "clear":
            context.clear()
            print("Context cleared.")
            continue

        payload = {
            "input": text,
            "mode": "default",
            "context": context,
        }
        try:
            response = requests.post(
                urljoin(args.url, "/process"),
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            output = data.get("output", data.get("final_output", str(data)))
            print(f"\nAI: {output}\n")
            context.append({"role": "user", "content": text})
            context.append({"role": "assistant", "content": output})
            # Keep context from growing unbounded.
            if len(context) > 20:
                context = context[-20:]
        except requests.exceptions.ConnectionError:
            print("Error: Cannot connect to the API. Is 'condrox start' running?\n")
        except requests.exceptions.HTTPError as exc:
            print(f"API error: {exc}\n")
        except Exception as exc:
            print(f"Unexpected error: {exc}\n")


if __name__ == "__main__":
    main()
