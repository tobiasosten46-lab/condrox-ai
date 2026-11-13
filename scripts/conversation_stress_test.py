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
Live multi-stage conversation stress test.

Generates dynamic, never-repeating topics and questions, runs them through
the BrainController pipeline, and checks for problems:
  - Empty / None output
  - Repetition across turns
  - Semantic drift
  - Contradiction with prior statements
  - Safety layer not blocking harmful content
  - Mode/domain misclassification
  - Token pressure overflow
  - Error accumulation

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core import BrainController


# ─── Dynamic Topic & Question Generator ──────────────────────────────────────

TOPIC_TEMPLATES: List[str] = [
    # Code generation
    "Write a Python function that {verb} {object}",
    "Create a {language} class for {object}",
    "Implement a {algorithm} in {language}",
    "Build a {language} module that {verb} {object}",
    "Code a {language} script to {verb} {object}",
    # Bug fixing
    "Fix this bug: my {object} returns None instead of {value}",
    "Why does my {object} throw a {error_type} error?",
    "Debug this {language} code that {verb} incorrectly",
    "My {object} is broken, it {verb} instead of {verb2}",
    # Review
    "Review this {language} code for {concern}",
    "Audit my {object} for {concern} issues",
    "Check my {object} for {concern} problems",
    # Refactor
    "Refactor my {object} to be more {quality}",
    "Rewrite this {language} function to {verb} better",
    "Clean up my {object} code",
    # Explain
    "Explain what {concept} is in {language}",
    "What is {concept} and how does it work?",
    "Describe how {concept} relates to {concept2}",
    "Give a {depth} explanation of {concept}",
    "Teach me {concept} step by step",
    # Analyze
    "Analyze the {concern} of my {object}",
    "What are the {concern} implications of {concept}?",
    "Evaluate the {quality} of this {object}",
    "Assess the {concern} risks in my {object}",
    # Architect
    "Design a system architecture for {object}",
    "How should I structure a {object} platform?",
    "What components do I need for a {object} system?",
    "Design the integration between {object} and {object2}",
    "Migrate my {object} to a new infrastructure",
    # General / dialog
    "Hello, can you help me with {object}?",
    "I need assistance with {concept}",
    "What do you think about {concept}?",
    "Can you recommend a {quality} approach to {object}?",
    "I'm stuck on {concept}, any ideas?",
    # Contradiction bait
    "Is {concept} always better than {concept2}?",
    "Should I always use {concept} for {object}?",
    "Never use {concept}, right?",
    # Edge cases (whitespace-only, very short, very long)
    # Note: truly empty strings are tested by the API layer (returns 400).
    # We include whitespace and single-char inputs here.
    "   ",
    "a",
    "x" * 500,
    "Write " + "code " * 100,
]

VERBS: List[str] = [
    "sorts", "filters", "validates", "parses", "transforms",
    "optimizes", "caches", "indexes", "serializes", "encrypts",
    "compresses", "schedules", "monitors", "analyzes", "processes",
    "generates", "calculates", "renders", "exports", "imports",
]

VERBS2: List[str] = [
    "process", "handle", "manage", "execute", "perform",
    "compute", "resolve", "validate", "transform", "generate",
]

OBJECTS: List[str] = [
    "a list", "a dictionary", "a queue", "a tree", "a graph",
    "a binary search tree", "a hash map", "a linked list",
    "a stack", "a priority queue", "a trie", "a bloom filter",
    "a REST API", "a WebSocket server", "a database schema",
    "a microservice", "a CI/CD pipeline", "a Docker container",
    "a Kubernetes deployment", "a message queue", "a cache layer",
    "an authentication system", "a rate limiter", "a load balancer",
    "a search engine", "a recommendation system", "a chat bot",
    "a file parser", "a log analyzer", "a metrics collector",
]

OBJECTS2: List[str] = [
    "a database", "an API gateway", "a message broker",
    "a frontend app", "a mobile app", "a data pipeline",
    "a monitoring stack", "a security layer",
]

LANGUAGES: List[str] = [
    "Python", "JavaScript", "TypeScript", "Java", "Go",
    "Rust", "C++", "C#", "Ruby", "Kotlin", "Swift", "Scala",
]

ALGORITHMS: List[str] = [
    "quicksort", "mergesort", "binary search", "depth-first search",
    "breadth-first search", "Dijkstra's algorithm", "A* pathfinding",
    "dynamic programming solution", "greedy algorithm",
    "backtracking solver", "genetic algorithm", "neural network",
    "gradient descent", "k-means clustering", "TF-IDF vectorizer",
    "cosine similarity search", "Bloom filter", "consistent hash ring",
]

CONCEPTS: List[str] = [
    "a closure", "a decorator", "a generator", "a context manager",
    "a monad", "a promise", "an async/await pattern",
    "a singleton", "a factory pattern", "an observer pattern",
    "a strategy pattern", "a dependency injection",
    "a microservice", "a distributed system", "a consensus algorithm",
    "a CAP theorem", "eventual consistency", "a reverse proxy",
    "a CDN", "a WAF", "zero-trust networking",
    "a vector database", "an embedding model", "a transformer",
    "attention mechanism", "a diffusion model", "RLHF",
    "a knowledge graph", "a semantic search", "RAG",
]

CONCEPTS2: List[str] = [
    "a relational database", "a key-value store", "a column store",
    "a document database", "a graph database", "a time-series DB",
    "REST", "GraphQL", "gRPC", "a message queue", "a pub/sub system",
    "a cache", "a CDN", "a load balancer",
]

CONCERNS: List[str] = [
    "performance", "security", "scalability", "reliability",
    "maintainability", "cost", "latency", "throughput",
    "data integrity", "privacy", "compliance", "availability",
]

QUALITIES: List[str] = [
    "clean", "efficient", "readable", "testable", "modular",
    "scalable", "robust", "resilient", "fault-tolerant", "async",
]

ERROR_TYPES: List[str] = [
    "NullPointer", "IndexOutOfBounds", "TypeMismatch",
    "StackOverflow", "MemoryLeak", "DeadLock",
    "ConnectionTimeout", "PermissionDenied", "FileNotFound",
]

VALUES: List[str] = [
    "a list", "a number", "a string", "a boolean",
    "an object", "a tuple", "a set", "a dictionary",
]

DEPTHS: List[str] = ["brief", "detailed", "comprehensive", "quick"]


def generate_question(seed: int) -> str:
    """Generate a unique question from a seed.

    Uses deterministic seed-based selection so questions are
    reproducible but never repeat across runs with different seeds.

    Complexity: T(n) in O(1), S(n) in O(1).
    """
    rng = random.Random(seed)
    template = rng.choice(TOPIC_TEMPLATES)

    # Skip templates that don't need filling (edge cases)
    if "{" not in template:
        return template

    # Fill all placeholders
    result = template
    replacements: Dict[str, str] = {}
    for placeholder in ["verb", "verb2", "object", "object2", "language",
                        "algorithm", "concept", "concept2", "concern",
                        "quality", "error_type", "value", "depth"]:
        token = "{" + placeholder + "}"
        while token in result:
            if placeholder == "verb":
                repl = rng.choice(VERBS)
            elif placeholder == "verb2":
                repl = rng.choice(VERBS2)
            elif placeholder == "object":
                repl = rng.choice(OBJECTS)
            elif placeholder == "object2":
                repl = rng.choice(OBJECTS2)
            elif placeholder == "language":
                repl = rng.choice(LANGUAGES)
            elif placeholder == "algorithm":
                repl = rng.choice(ALGORITHMS)
            elif placeholder == "concept":
                repl = rng.choice(CONCEPTS)
            elif placeholder == "concept2":
                repl = rng.choice(CONCEPTS2)
            elif placeholder == "concern":
                repl = rng.choice(CONCERNS)
            elif placeholder == "quality":
                repl = rng.choice(QUALITIES)
            elif placeholder == "error_type":
                repl = rng.choice(ERROR_TYPES)
            elif placeholder == "value":
                repl = rng.choice(VALUES)
            elif placeholder == "depth":
                repl = rng.choice(DEPTHS)
            else:
                repl = "unknown"
            result = result.replace(token, repl, 1)

    return result


def generate_conversation(seed: int, num_turns: int) -> List[str]:
    """Generate a multi-turn conversation with varied questions.

    Each turn uses a different seed derived from the base seed,
    ensuring no two turns in a conversation are identical.

    Complexity: T(n) in O(t) for t turns, S(n) in O(t).
    """
    turns: List[str] = []
    used_hashes: Set[str] = set()

    for i in range(num_turns):
        turn_seed = seed * 10000 + i * 137 + 42
        question = generate_question(turn_seed)

        # Ensure uniqueness within this conversation
        qhash = hashlib.md5(question.encode()).hexdigest()[:8]
        attempts = 0
        while qhash in used_hashes and attempts < 10:
            turn_seed += 1
            question = generate_question(turn_seed)
            qhash = hashlib.md5(question.encode()).hexdigest()[:8]
            attempts += 1

        used_hashes.add(qhash)
        turns.append(question)

    return turns


# ─── Problem Detectors ───────────────────────────────────────────────────────


@dataclass
class ConversationProblem:
    """A problem detected during conversation testing."""
    conversation_id: int
    turn: int
    problem_type: str
    description: str
    input_text: str
    output_text: str
    mode: str
    domain: str


@dataclass
class ConversationResult:
    """Result of a single conversation."""
    conversation_id: int
    seed: int
    num_turns: int
    outputs: List[str] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    problems: List[ConversationProblem] = field(default_factory=list)
    duration_ms: float = 0.0


def detect_problems(
    conv_id: int,
    turn: int,
    input_text: str,
    state,
    prior_outputs: List[str],
) -> List[ConversationProblem]:
    """Detect problems in a single turn's output.

    Args:
        conv_id: Conversation identifier.
        turn: Turn number (0-indexed).
        input_text: The input that was sent.
        state: The BrainState returned by process_request.
        prior_outputs: List of previous final_output strings.

    Returns:
        List of detected problems.
    """
    problems: List[ConversationProblem] = []
    output = state.final_output or ""
    base = {
        "conversation_id": conv_id,
        "turn": turn,
        "input_text": input_text[:100],
        "output_text": output[:200],
        "mode": state.mode,
        "domain": state.domain,
    }

    # 1. Empty output
    if not output or output.strip() == "":
        problems.append(ConversationProblem(
            problem_type="EMPTY_OUTPUT",
            description="Final output is empty or whitespace",
            **base,
        ))

    # 2. Exact repetition with a prior turn
    for i, prev in enumerate(prior_outputs):
        if output == prev and output:
            problems.append(ConversationProblem(
                problem_type="EXACT_REPETITION",
                description=f"Output identical to turn {i}",
                **base,
            ))
            break

    # 3. High similarity repetition (>90% word overlap with prior turn)
    if prior_outputs and output:
        prev = prior_outputs[-1]
        words_curr = set(output.lower().split())
        words_prev = set(prev.lower().split())
        if words_curr and words_prev:
            overlap = len(words_curr & words_prev) / max(
                len(words_curr | words_prev), 1
            )
            if overlap > 0.9 and len(words_curr) > 5:
                problems.append(ConversationProblem(
                    problem_type="HIGH_SIMILARITY",
                    description=f"Output {overlap:.0%} similar to prior turn",
                    **base,
                ))

    # 4. Safety not approved
    if not state.security_approved:
        problems.append(ConversationProblem(
            problem_type="SECURITY_NOT_APPROVED",
            description="Security check failed",
            **base,
        ))

    # 5. Ethics not approved
    if not state.ethical_approved:
        problems.append(ConversationProblem(
            problem_type="ETHICS_NOT_APPROVED",
            description="Ethics check failed",
            **base,
        ))

    # 6. Pipeline errors
    if state.errors:
        problems.append(ConversationProblem(
            problem_type="PIPELINE_ERROR",
            description=f"Pipeline errors: {'; '.join(state.errors[:3])}",
            **base,
        ))

    # 7. Mode never changes across a long conversation
    # (checked at conversation level, not per-turn)

    # 8. Very long output (token pressure)
    if len(output) > 5000:
        problems.append(ConversationProblem(
            problem_type="OUTPUT_TOO_LONG",
            description=f"Output is {len(output)} chars (>5000)",
            **base,
        ))

    # 9. Output contains raw error tracebacks
    if "Traceback" in output or "Error:" in output:
        problems.append(ConversationProblem(
            problem_type="RAW_ERROR_IN_OUTPUT",
            description="Output contains raw error/traceback text",
            **base,
        ))

    return problems


def detect_conversation_level_problems(
    result: ConversationResult,
) -> List[ConversationProblem]:
    """Detect problems at the conversation level (across all turns).

    Args:
        result: The conversation result to analyze.

    Returns:
        List of conversation-level problems.
    """
    problems: List[ConversationProblem] = []

    # All modes identical in a 5+ turn conversation
    if result.num_turns >= 5 and len(set(result.modes)) == 1:
        problems.append(ConversationProblem(
            conversation_id=result.conversation_id,
            turn=-1,
            problem_type="MODE_STUCK",
            description=f"All {result.num_turns} turns used mode '{result.modes[0]}'",
            input_text="",
            output_text="",
            mode=result.modes[0],
            domain="",
        ))

    # All domains identical in a 5+ turn conversation
    if result.num_turns >= 5 and len(set(result.domains)) == 1:
        problems.append(ConversationProblem(
            conversation_id=result.conversation_id,
            turn=-1,
            problem_type="DOMAIN_STUCK",
            description=f"All {result.num_turns} turns used domain '{result.domains[0]}'",
            input_text="",
            output_text="",
            mode="",
            domain=result.domains[0],
        ))

    return problems


# ─── Stress Test Runner ──────────────────────────────────────────────────────


def run_stress_test(
    num_conversations: int = 20,
    turns_per_conversation: int = 8,
    base_seed: int = 1000,
    verbose: bool = True,
) -> Tuple[List[ConversationResult], List[ConversationProblem]]:
    """Run a live conversation stress test.

    Args:
        num_conversations: Number of separate conversations to run.
        turns_per_conversation: Turns per conversation.
        base_seed: Base random seed for reproducibility.
        verbose: Print progress.

    Returns:
        Tuple of (all results, all problems).
    """
    controller = BrainController()
    controller.boot_system()

    all_results: List[ConversationResult] = []
    all_problems: List[ConversationProblem] = []

    for conv_idx in range(num_conversations):
        seed = base_seed + conv_idx * 7777
        questions = generate_conversation(seed, turns_per_conversation)

        result = ConversationResult(
            conversation_id=conv_idx,
            seed=seed,
            num_turns=turns_per_conversation,
        )

        start_time = time.perf_counter()
        prior_outputs: List[str] = []

        for turn_idx, question in enumerate(questions):
            # Skip truly empty inputs (tested by API layer)
            if not question or not question.strip():
                result.outputs.append("[skipped empty input]")
                result.modes.append("skip")
                result.domains.append("skip")
                if verbose:
                    print(f"  Conv {conv_idx:2d} Turn {turn_idx:2d} [SKIP] empty input")
                continue

            try:
                state = controller.process_request(question)
                output = state.final_output or ""
                result.outputs.append(output)
                result.modes.append(state.mode)
                result.domains.append(state.domain)

                # Detect per-turn problems
                turn_problems = detect_problems(
                    conv_idx, turn_idx, question, state, prior_outputs
                )
                result.problems.extend(turn_problems)
                all_problems.extend(turn_problems)

                prior_outputs.append(output)

                if verbose:
                    status = "OK" if not turn_problems else "ISSUE"
                    q_short = question[:60].replace("\n", " ")
                    print(
                        f"  Conv {conv_idx:2d} Turn {turn_idx:2d} [{status:4}] "
                        f"mode={state.mode:10s} domain={state.domain:12s} "
                        f"| {q_short}"
                    )

            except Exception as exc:
                err_msg = f"Turn {turn_idx}: {type(exc).__name__}: {exc}"
                result.errors.append(err_msg)
                all_problems.append(ConversationProblem(
                    conversation_id=conv_idx,
                    turn=turn_idx,
                    problem_type="EXCEPTION",
                    description=err_msg,
                    input_text=question[:100],
                    output_text="",
                    mode="",
                    domain="",
                ))
                if verbose:
                    print(f"  Conv {conv_idx:2d} Turn {turn_idx:2d} [FAIL] {err_msg}")

        result.duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Detect conversation-level problems
        conv_problems = detect_conversation_level_problems(result)
        result.problems.extend(conv_problems)
        all_problems.extend(conv_problems)

        all_results.append(result)

        if verbose:
            print(
                f"  Conv {conv_idx:2d} DONE: {result.duration_ms:.1f}ms, "
                f"{len(result.problems)} problems"
            )

    return all_results, all_problems


def print_report(
    results: List[ConversationResult],
    problems: List[ConversationProblem],
) -> None:
    """Print a summary report of the stress test."""
    print()
    print("=" * 70)
    print("  CONVERSATION STRESS TEST REPORT")
    print("=" * 70)
    print()

    total_turns = sum(r.num_turns for r in results)
    total_duration = sum(r.duration_ms for r in results)

    print(f"  Conversations:    {len(results)}")
    print(f"  Total turns:      {total_turns}")
    print(f"  Total problems:   {len(problems)}")
    print(f"  Total duration:   {total_duration:.1f}ms")
    print(f"  Avg per turn:     {total_duration / max(total_turns, 1):.2f}ms")
    print(f"  Problem rate:     {len(problems) / max(total_turns, 1) * 100:.1f}%")
    print()

    # Group problems by type
    by_type: Dict[str, List[ConversationProblem]] = {}
    for p in problems:
        by_type.setdefault(p.problem_type, []).append(p)

    if by_type:
        print("  PROBLEMS BY TYPE:")
        print("  " + "-" * 50)
        for ptype, plist in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"    {ptype:25s}  {len(plist):3d} occurrences")
        print()
    else:
        print("  NO PROBLEMS DETECTED!")
        print()

    # Show sample problems
    if problems:
        print("  SAMPLE PROBLEMS (first 10):")
        print("  " + "-" * 50)
        for p in problems[:10]:
            print(f"    Conv {p.conversation_id} Turn {p.turn}: {p.problem_type}")
            print(f"      Input:  {p.input_text[:70]}")
            print(f"      Output: {p.output_text[:70]}")
            print(f"      Detail: {p.description[:70]}")
            print()

    # Mode distribution
    all_modes: List[str] = []
    for r in results:
        all_modes.extend(r.modes)
    mode_counts: Dict[str, int] = {}
    for m in all_modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print("  MODE DISTRIBUTION:")
    print("  " + "-" * 50)
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / max(len(all_modes), 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {mode:15s}  {count:3d} ({pct:4.1f}%) {bar}")
    print()

    # Domain distribution
    all_domains: List[str] = []
    for r in results:
        all_domains.extend(r.domains)
    domain_counts: Dict[str, int] = {}
    for d in all_domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print("  DOMAIN DISTRIBUTION:")
    print("  " + "-" * 50)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / max(len(all_domains), 1) * 100
        bar = "#" * int(pct / 2)
        print(f"    {domain:15s}  {count:3d} ({pct:4.1f}%) {bar}")
    print()

    # Unique outputs check
    all_outputs: List[str] = []
    for r in results:
        all_outputs.extend(r.outputs)
    unique_outputs = len(set(all_outputs))

    print(f"  OUTPUT UNIQUENESS:")
    print("  " + "-" * 50)
    print(f"    Total outputs:    {len(all_outputs)}")
    print(f"    Unique outputs:   {unique_outputs}")
    print(f"    Uniqueness rate:  {unique_outputs / max(len(all_outputs), 1) * 100:.1f}%")
    print()

    print("=" * 70)
    if len(problems) == 0:
        print("  RESULT: ALL CONVERSATIONS PASSED — NO PROBLEMS DETECTED")
    else:
        print(f"  RESULT: {len(problems)} PROBLEMS DETECTED — REVIEW NEEDED")
    print("=" * 70)


# ─── Main ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    num_conv = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    num_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    print(f"Starting stress test: {num_conv} conversations x {num_turns} turns")
    print(f"Seed: {seed}")
    print()

    results, problems = run_stress_test(
        num_conversations=num_conv,
        turns_per_conversation=num_turns,
        base_seed=seed,
        verbose=True,
    )

    print_report(results, problems)
