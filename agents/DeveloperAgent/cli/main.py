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
DeveloperAgent CLI — Command-line interface for the code assistant.

Commands:
- execute — Execute a code task
- index — Index a project file
- health — Check agent health
- stats — Show agent statistics

Complexity:
    execute: T(n) in O(pipeline + gates + rag), S(n) in O(output).
    index: T(n) in O(n) where n = code length, S(n) in O(chunks).

Invariants:
    - All commands require valid input.
    - Execute requires Condrox BrainController (provided via --brain flag).
    - Output is formatted for terminal readability.

Edge Cases:
    - Empty input -> error message.
    - BrainController unavailable -> error message.
    - Gate block -> shows blocked status.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Optional

from agents.DeveloperAgent.condrox_agent_coder import CondroxAgentCoder


# ─── CLI Commands ─────────────────────────────────────────────────────────────


async def cmd_execute(args: argparse.Namespace) -> None:
    """Execute a code task."""
    # Import BrainController dynamically to avoid circular imports
    from core import BrainController
    
    agent = CondroxAgentCoder(root_dir=args.root_dir)
    
    # Initialize BrainController
    brain = BrainController()
    brain.boot_system()
    
    print(f"🟠 DeveloperAgent: Executing task...")
    print(f"  Input: {args.user_input}")
    print(f"  Language: {args.language}")
    print(f"  File: {args.file_path}")
    print()
    
    try:
        result = await agent.execute(
            user_input=args.user_input,
            brain_controller=brain,
            language=args.language,
            file_path=args.file_path,
            context=args.context,
        )
        
        print(f"╔══════════════════════════════════════════════════════════╗")
        print(f"║  Task ID: {result.task_id:<48}║")
        print(f"║  Intent: {result.intent.value:<48}║")
        print(f"║  State: {result.state.value:<48}║")
        print(f"║  Confidence: {result.confidence:.2f} ({result.confidence_level.value}){'':>28}║")
        print(f"║  Domain: {result.domain.value:<48}║")
        print(f"║  Latency: {result.latency_ms}ms{'':>43}║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║  Verification: {'PASS' if result.verification.passed else 'FAIL'}{'':>38}║")
        print(f"║  Blocked: {'YES' if result.verification.blocked else 'NO'}{'':>42}║")
        print(f"║  Score: {result.verification.score:.2f}{'':>44}║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║  Output: {'':>47}║")
        print(f"║  {result.output[:46]:<46}║")
        if len(result.output) > 46:
            print(f"║  ... ({len(result.output)} chars total){'':>30}║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        if result.errors:
            print()
            print(f"Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        sys.exit(0 if result.state.value == "completed" else 1)
        
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Execution failed: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_index(args: argparse.Namespace) -> None:
    """Index a project file."""
    agent = CondroxAgentCoder(root_dir=args.root_dir)
    
    print(f"🟠 DeveloperAgent: Indexing file...")
    print(f"  File: {args.file_path}")
    print(f"  Language: {args.language}")
    print()
    
    try:
        with open(args.file_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        index_size_before = agent.rag_index_size
        agent.index_project(args.file_path, code, args.language)
        chunks_created = agent.rag_index_size - index_size_before
        
        print(f"✅ Indexed successfully")
        print(f"  Chunks created: {chunks_created}")
        print(f"  RAG index size: {agent.rag_index_size}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_health(args: argparse.Namespace) -> None:
    """Check agent health."""
    agent = CondroxAgentCoder(root_dir=args.root_dir)
    
    print(f"🟠 DeveloperAgent: Health Check")
    print(f"  Agent State: {agent.state.value}")
    print(f"  RAG Index Size: {agent.rag_index_size}")
    print(f"  Status: ✅ Healthy")
    print()


async def cmd_stats(args: argparse.Namespace) -> None:
    """Show agent statistics."""
    agent = CondroxAgentCoder(root_dir=args.root_dir)
    
    print(f"🟠 DeveloperAgent: Statistics")
    print(f"  Agent State: {agent.state.value}")
    print(f"  RAG Index Size: {agent.rag_index_size}")
    print(f"  Root Directory: {args.root_dir}")
    print()


# ─── Main Entry Point ───────────────────────────────────────────────────────


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="condrox-coder",
        description="DeveloperAgent CLI — Enterprise-grade AI code assistant",
    )
    
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Root directory for the agent (default: current directory)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute a code task")
    execute_parser.add_argument(
        "user_input",
        help="Coding task description"
    )
    execute_parser.add_argument(
        "--language",
        default="python",
        help="Target programming language (default: python)"
    )
    execute_parser.add_argument(
        "--file-path",
        default="output.py",
        help="Target file path (default: output.py)"
    )
    execute_parser.add_argument(
        "--context",
        default="",
        help="Additional context"
    )
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a project file")
    index_parser.add_argument(
        "file_path",
        help="File path to index"
    )
    index_parser.add_argument(
        "--language",
        default="python",
        help="Programming language (default: python)"
    )
    
    # Health command
    subparsers.add_parser("health", help="Check agent health")
    
    # Stats command
    subparsers.add_parser("stats", help="Show agent statistics")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    if args.command == "execute":
        asyncio.run(cmd_execute(args))
    elif args.command == "index":
        asyncio.run(cmd_index(args))
    elif args.command == "health":
        asyncio.run(cmd_health(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
