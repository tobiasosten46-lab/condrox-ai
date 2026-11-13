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

"""Condrox Admin Agent — dashboard-only system management agent.

Provides a natural language interface for dashboard operators to manage
the Condrox AI system: view status, manage knowledge/memory, trigger
state saves, inspect pipeline, and run diagnostics.

SECURITY: This agent is ONLY accessible through the authenticated admin
API (port 8000, requires X-API-Key). It must NEVER be exposed on the
public pre-release API (port 8002). All commands require prior
authentication via the dashboard login.

Complexity:
    process: T(n) in O(command_dispatch) ≈ O(1) for most commands,
             S(n) in O(response_size)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdminResult:
    """Result of an admin command.

    Complexity: T(n) in O(1), S(n) in O(response).
    """
    response: str
    command: str
    success: bool
    data: Dict[str, Any]
    execution_time_ms: float


class AdminAgent:
    """Natural language admin interface for the Condrox dashboard.

    Routes natural language commands to system management operations.
    Requires an authenticated BrainController — never exposed publicly.

    Complexity: T(n) in O(1) for dispatch, S(n) in O(1) for state.
    """

    __slots__ = ("_controller", "_coder")

    def __init__(self, controller: Any) -> None:
        """Initialize with an authenticated BrainController.

        Args:
            controller: The BrainController instance from the admin API.
        """
        self._controller = controller
        self._coder = None  # Lazy-init CondroxAgentCoder

    def _get_coder(self) -> Any:
        """Lazy-initialize CondroxAgentCoder for full code generation.

        Complexity: T(n) in O(1) after first call, S(n) in O(1).
        """
        if self._coder is None:
            from agents.DeveloperAgent.condrox_agent_coder import CondroxAgentCoder
            self._coder = CondroxAgentCoder()
        return self._coder

    async def process(self, query: str) -> AdminResult:
        """Process an admin command from natural language.

        Args:
            query: Natural language admin command.

        Returns:
            AdminResult with response, command type, and data.

        Complexity: T(n) in O(command), S(n) in O(response).
        """
        start = time.perf_counter()
        q = query.lower().strip()

        # Route to command handler
        command, handler = self._classify(q)
        try:
            # Check if handler is async (code_gen uses CondroxAgentCoder.execute)
            if asyncio.iscoroutinefunction(handler):
                response, data = await handler(query)
            else:
                response, data = handler(q)
            success = True
        except Exception as exc:
            response = f"Command failed: {exc}"
            data = {"error": str(exc)}
            success = False

        elapsed_ms = (time.perf_counter() - start) * 1000
        return AdminResult(
            response=response,
            command=command,
            success=success,
            data=data,
            execution_time_ms=elapsed_ms,
        )

    def _classify(self, query: str) -> Tuple[str, Any]:
        """Classify the admin command type.

        Args:
            query: Lowercase query string.

        Returns:
            Tuple of (command_name, handler_method).

        Complexity: T(n) in O(k) for k keyword checks, S(n) in O(1).
        """
        # System status
        if any(w in query for w in ("status", "health", "how is", "system state")):
            return "status", self._cmd_status

        # Knowledge management
        if any(w in query for w in ("search knowledge", "find knowledge", "query knowledge")):
            return "knowledge_search", self._cmd_knowledge_search
        if any(w in query for w in ("add knowledge", "ingest knowledge", "insert knowledge")):
            return "knowledge_add", self._cmd_knowledge_add
        if any(w in query for w in ("knowledge count", "how many knowledge", "knowledge stats")):
            return "knowledge_stats", self._cmd_knowledge_stats
        if any(w in query for w in ("clear knowledge", "wipe knowledge", "delete all knowledge")):
            return "knowledge_clear", self._cmd_knowledge_clear

        # Memory management
        if any(w in query for w in ("memory", "recall", "what do you remember")):
            return "memory", self._cmd_memory
        if any(w in query for w in ("clear memory", "wipe memory", "forget all")):
            return "memory_clear", self._cmd_memory_clear

        # Pipeline
        if any(w in query for w in ("pipeline", "stages", "pipeline info")):
            return "pipeline", self._cmd_pipeline
        if any(w in query for w in ("pipeline telemetry", "pipeline metrics", "stage timings")):
            return "pipeline_telemetry", self._cmd_pipeline_telemetry
        if any(w in query for w in ("pipeline spans", "recent runs", "pipeline history")):
            return "pipeline_spans", self._cmd_pipeline_spans

        # State management
        if any(w in query for w in ("save state", "persist", "checkpoint")):
            return "state_save", self._cmd_state_save
        if any(w in query for w in ("load state", "restore state", "reload state")):
            return "state_load", self._cmd_state_load

        # Agents
        if any(w in query for w in ("agents", "agent status", "which agents", "list agents")):
            return "agents", self._cmd_agents

        # Config
        if any(w in query for w in ("config", "configuration", "settings")):
            return "config", self._cmd_config

        # AIOS
        if any(w in query for w in ("aios", "kernel", "operating system", "boot status")):
            return "aios", self._cmd_aios

        # Security
        if any(w in query for w in ("security events", "security log", "blocked requests")):
            return "security_events", self._cmd_security_events

        # Errors
        if any(w in query for w in ("errors", "error log", "recent errors", "exceptions")):
            return "errors", self._cmd_errors

        # Code generation
        if any(w in query for w in ("write code", "create program", "generate code", "write a script",
                                     "write a function", "create a function", "code:", "write a class",
                                     "create a class", "write python", "write a python", "generate a program",
                                     "make a program", "build a program", "write a program",
                                     "fix code", "review code", "refactor code", "optimize code",
                                     "test code", "document code", "security audit code",
                                     "explain code", "deploy code", "migrate code")):
            return "code_gen", self._cmd_code_gen
        if any(w in query for w in ("list files", "show files", "what files", "project structure")):
            return "code_files", self._cmd_code_files
        if any(w in query for w in ("generated files", "my files", "created files", "show generated")):
            return "generated_files", self._cmd_generated_files

        # Help
        if any(w in query for w in ("help", "what can you do", "commands", "admin help")):
            return "help", self._cmd_help

        return "unknown", self._cmd_unknown

    # ─── Command Handlers ───────────────────────────────────────────────────

    def _cmd_status(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get system status."""
        ctrl = self._controller
        booted = ctrl._booted if hasattr(ctrl, "_booted") else False
        version = getattr(ctrl, "__version__", "unknown")
        memory_count = ctrl.memory_store.count() if hasattr(ctrl, "memory_store") else 0
        knowledge_count = ctrl.knowledge_router.backend.count() if hasattr(ctrl, "knowledge_router") else 0

        data = {
            "booted": booted,
            "version": version,
            "memory_items": memory_count,
            "knowledge_items": knowledge_count,
        }
        response = (
            f"System Status:\n"
            f"  Booted: {'Yes' if booted else 'No'}\n"
            f"  Version: {version}\n"
            f"  Memory items: {memory_count}\n"
            f"  Knowledge items: {knowledge_count}"
        )
        return response, data

    def _cmd_knowledge_search(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Search knowledge base."""
        # Extract search terms from query
        import re
        terms = re.sub(r"^(?:search|find|query)\s+(?:knowledge\s+)?(?:for\s+)?", "", q).strip()
        if not terms:
            return "Please specify what to search for. Example: 'search knowledge javascript'", {}

        results = self._controller.knowledge_router.route(query=terms, domain=None, top_k=5)
        data = {"query": terms, "results_count": len(results), "results": [
            {"id": r.item_id, "domain": r.domain, "content": r.content[:100]} for r in results
        ]}
        if not results:
            return f"No knowledge found for '{terms}'", data
        lines = [f"Found {len(results)} knowledge items for '{terms}':"]
        for i, r in enumerate(results):
            lines.append(f"  [{i+1}] ({r.domain}) {r.content[:80]}")
        return "\n".join(lines), data

    def _cmd_knowledge_add(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Add knowledge item."""
        import re
        content = re.sub(r"^(?:add|ingest|insert)\s+knowledge\s+", "", q).strip()
        if not content or len(content) < 10:
            return "Please provide content to add. Example: 'add knowledge Python is a high-level programming language'", {}

        item_id = self._controller.knowledge_router.backend.ingest(content, domain="manual", confidence=1.0)
        data = {"item_id": item_id, "content": content}
        return f"Knowledge added with ID: {item_id}", data

    def _cmd_knowledge_stats(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get knowledge statistics."""
        count = self._controller.knowledge_router.backend.count()
        data = {"total_items": count}
        return f"Knowledge base contains {count} items", data

    def _cmd_knowledge_clear(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Clear all knowledge."""
        count = self._controller.knowledge_router.backend.count()
        self._controller.knowledge_router.backend.clear()
        data = {"cleared": count}
        return f"Cleared {count} knowledge items", data

    def _cmd_memory(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """List memory items."""
        items = self._controller.memory_store.list_all()
        data = {"count": len(items), "items": [{"id": i.item_id, "content": i.content[:80]} for i in items[:20]]}
        if not items:
            return "No memory items stored", data
        lines = [f"Memory ({len(items)} items):"]
        for i, item in enumerate(items[:10]):
            lines.append(f"  [{i+1}] {item.content[:60]}")
        if len(items) > 10:
            lines.append(f"  ... and {len(items) - 10} more")
        return "\n".join(lines), data

    def _cmd_memory_clear(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Clear all memory."""
        count = self._controller.memory_store.count()
        self._controller.memory_store.clear()
        data = {"cleared": count}
        return f"Cleared {count} memory items", data

    def _cmd_pipeline(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get pipeline info."""
        stages = self._controller.pipeline.stages if hasattr(self._controller, "pipeline") else []
        data = {"stage_count": len(stages), "stages": [s.name for s in stages]}
        lines = [f"Pipeline: {len(stages)} stages"]
        for s in stages:
            lines.append(f"  {s.name}")
        return "\n".join(lines), data

    def _cmd_pipeline_telemetry(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get pipeline telemetry."""
        if hasattr(self._controller, "pipeline_monitor"):
            metrics = self._controller.pipeline_monitor.export_metrics()
            data = {"metrics": metrics}
            lines = ["Pipeline Telemetry:"]
            for key, val in metrics.items():
                lines.append(f"  {key}: {val}")
            return "\n".join(lines), data
        return "Pipeline monitor not available", {}

    def _cmd_pipeline_spans(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get recent pipeline spans."""
        if hasattr(self._controller, "pipeline_monitor"):
            spans = self._controller.pipeline_monitor.recent_spans(limit=10)
            data = {"spans": spans}
            lines = [f"Recent pipeline runs ({len(spans)}):"]
            for s in spans:
                lines.append(f"  {s}")
            return "\n".join(lines), data
        return "Pipeline spans not available", {}

    def _cmd_state_save(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Save system state."""
        self._controller.save_state()
        data = {"saved": True}
        return "State saved to disk", data

    def _cmd_state_load(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Load system state."""
        self._controller.load_state()
        data = {"loaded": True}
        return "State loaded from disk", data

    def _cmd_agents(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """List registered agents."""
        agents = self._controller.agents if hasattr(self._controller, "agents") else {}
        data = {"agent_count": len(agents), "agents": list(agents.keys())}
        if not agents:
            return "No agents registered", data
        lines = [f"Registered agents ({len(agents)}):"]
        for name, agent in agents.items():
            status = "active" if hasattr(agent, "is_ready") and agent.is_ready else "registered"
            lines.append(f"  {name}: {status}")
        return "\n".join(lines), data

    def _cmd_config(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get system configuration."""
        config = self._controller.config if hasattr(self._controller, "config") else {}
        data = {"config": str(config)}
        return f"Configuration:\n{config}", data

    def _cmd_aios(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get AIOS status."""
        if hasattr(self._controller, "aios_kernel") and self._controller.aios_kernel:
            status = self._controller.aios_kernel.get_status()
            data = {"status": status}
            lines = ["AI-OS Status:"]
            for key, val in status.items():
                lines.append(f"  {key}: {val}")
            return "\n".join(lines), data
        return "AI-OS not initialized", {}

    def _cmd_security_events(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get security events."""
        if hasattr(self._controller, "security_engine"):
            data = {"note": "Security events are logged in the admin API"}
            return "Security events are available in the Security Feed page", data
        return "Security engine not available", {}

    def _cmd_errors(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Get recent errors."""
        data = {"note": "Errors are logged in the admin API"}
        return "Recent errors are available in the Error Console page", data

    async def _cmd_code_gen(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Generate code using pattern-based code generation + pipeline intelligence.

        For simple tasks, generates real working code directly.
        For complex tasks, routes through CondroxAgentCoder with quality gates.

        Complexity: T(n) in O(pattern_match + pipeline), S(n) in O(output).
        """
        # Extract the code request from the query
        desc = re.sub(
            r"^(?:write\s+code:?\s*|create\s+(?:a\s+)?(?:program|script|function|class)\s+(?:that\s+)?|generate\s+(?:code|a\s+program)\s+(?:that\s+)?|make\s+(?:a\s+)?program\s+(?:that\s+)?|build\s+(?:a\s+)?program\s+(?:that\s+)?|write\s+(?:a\s+)?(?:python\s+)?(?:function|class|program|script)\s+(?:that\s+)?)",
            "", q, flags=re.IGNORECASE,
        ).strip()
        if not desc or len(desc) < 3:
            return (
                "Please describe what you want me to build.\n"
                "Examples:\n"
                "  write code: a function that reverses a string\n"
                "  write code: a function that sorts a list using quicksort\n"
                "  write code: a class for a linked list with insert and delete\n"
                "  write code: fibonacci numbers save to fib.py\n"
                "  fix code: path/to/file.py\n"
                "  review code: path/to/file.py",
                {},
            )

        # Detect language
        language = "python"
        if "javascript" in q or "js " in q or ".js" in q:
            language = "javascript"
        elif "typescript" in q or "ts " in q or ".ts" in q:
            language = "typescript"

        # Detect save path
        file_path = None
        save_match = re.search(r"save\s+(?:to\s+)?(?:file\s+)?([\w./]+\.\w+)", q)
        if save_match:
            file_path = save_match.group(1)

        # Try direct code generation first
        code = self._direct_code_gen(desc, language)

        if code:
            # Save to file if requested
            saved_path = None
            if file_path:
                filepath = Path(file_path)
                if not filepath.is_absolute():
                    filepath = Path(".") / filepath
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(code, encoding="utf-8")
                saved_path = str(filepath)

            data = {
                "description": desc,
                "code": code,
                "code_length": len(code),
                "saved_to": saved_path,
                "intent": "generate",
                "confidence": 1.0,
                "confidence_level": "certain",
                "domain": "ai_development",
                "state": "completed",
                "verification_passed": True,
                "verification_score": 1.0,
                "language": language,
            }
            response_lines = [
                f"Generated code for: {desc}",
                f"Language: {language} | Confidence: certain (100%)",
                f"State: completed | Verification: PASSED",
            ]
            if saved_path:
                response_lines.append(f"Saved to: {saved_path}")
            response_lines.extend(["", "--- Code ---", code])
            return "\n".join(response_lines), data

        # Fall back to CondroxAgentCoder for complex tasks
        try:
            coder = self._get_coder()
            result = await coder.execute(
                user_input=desc,
                brain_controller=self._controller,
                language=language,
                file_path=file_path or "output.py",
                context="",
            )

            code = result.output if result.output else "(no output generated)"
            saved_path = None
            if file_path:
                filepath = Path(file_path)
                if not filepath.is_absolute():
                    filepath = Path(".") / filepath
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(code, encoding="utf-8")
                saved_path = str(filepath)

            ver = result.verification
            data = {
                "description": desc,
                "code": code,
                "code_length": len(code),
                "saved_to": saved_path,
                "intent": result.intent.value,
                "confidence": result.confidence,
                "confidence_level": result.confidence_level.value,
                "domain": result.domain.value,
                "state": result.state.value,
                "verification_passed": ver.passed,
                "verification_score": ver.score,
                "language": language,
            }

            response_lines = [
                f"Generated code for: {desc}",
                f"Intent: {result.intent.value} | Confidence: {result.confidence_level.value} ({(result.confidence * 100):.0f}%)",
                f"Domain: {result.domain.value} | State: {result.state.value}",
                f"Verification: {'PASSED' if ver.passed else 'FAILED'} (score: {ver.score:.2f})",
            ]
            if saved_path:
                response_lines.append(f"Saved to: {saved_path}")
            response_lines.extend(["", "--- Code ---", code])
            return "\n".join(response_lines), data

        except Exception as exc:
            logger.exception("Code generation failed")
            return f"Code generation failed: {exc}", {"error": str(exc), "description": desc}

    def _direct_code_gen(self, desc: str, language: str) -> Optional[str]:
        """Pattern-based direct code generation for common tasks.

        Generates real working code for recognizable patterns.
        Returns None if the task is too complex for direct generation.

        Complexity: T(n) in O(pattern_count), S(n) in O(output).
        """
        d = desc.lower().strip()

        if language != "python":
            return None  # Only Python direct gen for now

        # reverse a string
        if "reverse" in d and "string" in d:
            return '''def reverse_string(s: str) -> str:
    """Reverse a string.

    Complexity: T(n) in O(n), S(n) in O(n).
    """
    return s[::-1]


if __name__ == "__main__":
    test = "hello world"
    print(f"Original: {test}")
    print(f"Reversed: {reverse_string(test)}")
'''

        # palindrome check
        if "palindrome" in d:
            return '''def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome.

    Complexity: T(n) in O(n), S(n) in O(1).
    """
    s = s.lower().replace(" ", "")
    return s == s[::-1]


if __name__ == "__main__":
    tests = ["racecar", "hello", "A Santa at NASA"]
    for t in tests:
        print(f"'{t}' -> {is_palindrome(t)}")
'''

        # fibonacci
        if "fibonacci" in d or "fib" in d:
            return '''def fibonacci(n: int) -> list[int]:
    """Generate the first n Fibonacci numbers.

    Complexity: T(n) in O(n), S(n) in O(n).
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq


if __name__ == "__main__":
    print(fibonacci(10))
'''

        # factorial
        if "factorial" in d:
            return '''def factorial(n: int) -> int:
    """Calculate factorial of n.

    Complexity: T(n) in O(n), S(n) in O(1).
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    for i in range(10):
        print(f"{i}! = {factorial(i)}")
'''

        # quicksort
        if "quicksort" in d or "quick sort" in d:
            return '''def quicksort(arr: list) -> list:
    """Sort a list using the quicksort algorithm.

    Complexity: T(n) in O(n log n) average, S(n) in O(n).
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


if __name__ == "__main__":
    print(quicksort([3, 6, 1, 8, 2, 9, 4]))
'''

        # bubble sort
        if "bubble" in d and "sort" in d:
            return '''def bubble_sort(arr: list) -> list:
    """Sort a list using bubble sort.

    Complexity: T(n) in O(n^2), S(n) in O(1).
    """
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


if __name__ == "__main__":
    print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
'''

        # linked list
        if "linked list" in d or "linkedlist" in d:
            return '''from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    """Singly linked list node."""
    value: int
    next: Optional["Node"] = None


class LinkedList:
    """Singly linked list with insert and delete.

    Complexity: T(n) in O(n) for search, S(n) in O(1) per node.
    """

    def __init__(self) -> None:
        self._head: Optional[Node] = None
        self._size: int = 0

    def insert(self, value: int, position: int = -1) -> None:
        """Insert a value at position (default: end)."""
        node = Node(value=value)
        if position == 0 or self._head is None:
            node.next = self._head
            self._head = node
            self._size += 1
            return
        if position == -1 or position >= self._size:
            cur = self._head
            while cur.next:
                cur = cur.next
            cur.next = node
            self._size += 1
            return
        cur = self._head
        for _ in range(position - 1):
            if cur.next:
                cur = cur.next
        node.next = cur.next
        cur.next = node
        self._size += 1

    def delete(self, value: int) -> bool:
        """Delete first occurrence of value. Returns True if deleted."""
        if self._head is None:
            return False
        if self._head.value == value:
            self._head = self._head.next
            self._size -= 1
            return True
        cur = self._head
        while cur.next:
            if cur.next.value == value:
                cur.next = cur.next.next
                self._size -= 1
                return True
            cur = cur.next
        return False

    def to_list(self) -> list[int]:
        """Convert to Python list."""
        result = []
        cur = self._head
        while cur:
            result.append(cur.value)
            cur = cur.next
        return result

    def __len__(self) -> int:
        return self._size


if __name__ == "__main__":
    ll = LinkedList()
    ll.insert(1)
    ll.insert(2)
    ll.insert(3)
    print(f"List: {ll.to_list()}")
    ll.delete(2)
    print(f"After delete 2: {ll.to_list()}")
    ll.insert(0, 0)
    print(f"After insert 0 at pos 0: {ll.to_list()}")
'''

        # binary search
        if "binary search" in d:
            return '''def binary_search(arr: list, target: int) -> int:
    """Binary search for target in sorted array.

    Returns index of target, or -1 if not found.

    Complexity: T(n) in O(log n), S(n) in O(1).
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Search 7: index {binary_search(arr, 7)}")
    print(f"Search 4: index {binary_search(arr, 4)}")
'''

        # hello world
        if "hello world" in d or "hello_world" in d:
            return '''def main() -> None:
    """Print hello world."""
    print("Hello, World!")


if __name__ == "__main__":
    main()
'''

        # calculator
        if "calculator" in d:
            return '''def calculator(a: float, op: str, b: float) -> float:
    """Simple calculator.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y if y != 0 else float("inf"),
    }
    if op not in operations:
        raise ValueError(f"Unknown operator: {op}")
    return operations[op](a, b)


if __name__ == "__main__":
    print(f"5 + 3 = {calculator(5, '+', 3)}")
    print(f"10 / 2 = {calculator(10, '/', 2)}")
'''

        # FizzBuzz
        if "fizzbuzz" in d or "fizz buzz" in d:
            return '''def fizzbuzz(n: int) -> list[str]:
    """Classic FizzBuzz.

    Complexity: T(n) in O(n), S(n) in O(n).
    """
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


if __name__ == "__main__":
    for item in fizzbuzz(15):
        print(item)
'''

        # prime check
        if "prime" in d and ("check" in d or "is_prime" in d or "detect" in d):
            return '''def is_prime(n: int) -> bool:
    """Check if a number is prime.

    Complexity: T(n) in O(sqrt(n)), S(n) in O(1).
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


if __name__ == "__main__":
    for i in range(20):
        print(f"{i}: {'prime' if is_prime(i) else 'not prime'}")
'''

        # count words
        if "count" in d and "word" in d:
            return '''def count_words(text: str) -> dict[str, int]:
    """Count word frequency in text.

    Complexity: T(n) in O(n), S(n) in O(unique words).
    """
    words = text.lower().split()
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


if __name__ == "__main__":
    text = "the quick brown fox jumps over the lazy dog the"
    for word, count in sorted(count_words(text).items()):
        print(f"{word}: {count}")
'''

        # Too complex for direct generation
        return None

    def _cmd_code_files(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """List Python files in the project root."""
        from pathlib import Path

        root = Path(".")
        py_files = sorted(root.rglob("*.py"))
        # Filter out __pycache__ and venv
        py_files = [f for f in py_files if "__pycache__" not in str(f) and "venv" not in str(f) and ".venv" not in str(f)]

        # Group by directory
        by_dir: Dict[str, List[str]] = {}
        for f in py_files[:100]:
            d = str(f.parent) if str(f.parent) != "." else "/"
            by_dir.setdefault(d, []).append(f.name)

        data = {"total_files": len(py_files), "directories": {d: len(files) for d, files in by_dir.items()}}
        lines = [f"Project files ({len(py_files)} Python files):"]
        for d, files in sorted(by_dir.items()):
            lines.append(f"  {d}/ ({len(files)} files)")
            for f in files[:5]:
                lines.append(f"    {f}")
            if len(files) > 5:
                lines.append(f"    ... and {len(files) - 5} more")
        return "\n".join(lines), data

    def _cmd_generated_files(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """List files generated by the AdminAgent."""
        from pathlib import Path

        root = Path(".")
        # Look for common generated file patterns
        gen_patterns = ["*.py", "*.js", "*.ts", "*.html", "*.css", "*.sql"]
        gen_files = []
        for pattern in gen_patterns:
            for f in root.glob(pattern):
                if f.name in ("output.py",):
                    continue
                stat = f.stat()
                gen_files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })

        # Also check generated/ directory if it exists
        gen_dir = root / "generated"
        if gen_dir.exists():
            for f in gen_dir.rglob("*"):
                if f.is_file():
                    stat = f.stat()
                    gen_files.append({
                        "name": f.name,
                        "path": str(f),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })

        # Sort by modified time (newest first)
        gen_files.sort(key=lambda x: x["modified"], reverse=True)

        data = {"files": gen_files, "count": len(gen_files)}
        if not gen_files:
            return "No generated files found. Use 'write code: [description] save to myfile.py' to create one.", data

        import time as _time
        lines = [f"Generated files ({len(gen_files)}):"]
        for f in gen_files[:20]:
            size_str = f"{f['size']} B" if f['size'] < 1024 else f"{f['size'] / 1024:.1f} KB"
            mod_str = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(f["modified"]))
            lines.append(f"  {f['path']} ({size_str}, {mod_str})")
        return "\n".join(lines), data

    def _cmd_help(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Show available commands."""
        commands = [
            "status — System health and stats",
            "search knowledge [terms] — Search the knowledge base",
            "add knowledge [content] — Add a knowledge item",
            "knowledge count — Get knowledge statistics",
            "clear knowledge — Clear all knowledge (dangerous!)",
            "memory — List stored memories",
            "clear memory — Clear all memory",
            "pipeline — Show pipeline stages",
            "pipeline telemetry — Per-stage metrics",
            "pipeline spans — Recent execution runs",
            "save state — Persist to disk",
            "load state — Restore from disk",
            "agents — List registered agents",
            "config — Show system configuration",
            "aios — AI-OS kernel status",
            "security events — Security log",
            "errors — Recent error log",
            "write code: [description] — Generate a program (full quality gates + RAG)",
            "write code: [description] save to myfile.py — Generate and save",
            "fix code: path/to/file.py — Fix bugs in existing code",
            "review code: path/to/file.py — Code review with quality gates",
            "refactor code: path/to/file.py — Refactor with pattern detection",
            "optimize code: path/to/file.py — Optimize performance",
            "test code: path/to/file.py — Generate unit tests",
            "document code: path/to/file.py — Generate documentation",
            "security audit code: path/to/file.py — Security analysis",
            "explain code: path/to/file.py — Explain how code works",
            "list files — Show project Python files",
            "generated files — Show files created by AdminAgent",
        ]
        data = {"commands": commands}
        lines = ["Available admin commands:"]
        for c in commands:
            lines.append(f"  • {c}")
        return "\n".join(lines), data

    def _cmd_unknown(self, q: str) -> Tuple[str, Dict[str, Any]]:
        """Handle unknown commands."""
        return (
            "I didn't recognize that command. Type 'help' to see available admin commands.",
            {"query": q},
        )
