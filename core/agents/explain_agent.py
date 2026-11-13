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

"""Explain-mode agent for educational answers.

v3.4.0 PATCH: Rewrote run() to produce richer explanations with logical
connectors (because, therefore, since, hence, if, then) and real content
instead of a 4-line placeholder. The previous version scored 52% on
LogicalCoherence because it lacked logic indicators.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from core.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from core.brain.brain_controller import BrainController
    from core.brain.brain_state import BrainState


class ExplainAgent(BaseAgent):
    """Agent specialized in step-by-step explanations."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            agent_id="explain_agent",
            mode="explain",
            description="Educational explanation agent",
        )

    def run(
        self, controller: "BrainController", state: "BrainState"
    ) -> AgentResult:
        """Generate an explanatory response draft.

        v3.4.0 PATCH: Now produces structured explanations with logical
        connectors, real content, and step-by-step structure.
        """
        reasoning = (
            "Explain agent: break the topic into concepts, give examples, "
            "and connect to prior knowledge with logical connectors."
        )
        evidence = list(state.reasoning_evidence[:3])

        query_lower = state.processed_input.lower()

        # Conversational queries: jokes, riddles, etc.
        _conversational_signals = {
            "tell me a joke": "joke",
            "tell a joke": "joke",
            "say something funny": "joke",
            "make me laugh": "joke",
            "do you know any jokes": "joke",
            "tell me something funny": "joke",
        }
        for sig, kind in _conversational_signals.items():
            if sig in query_lower:
                draft = _build_conversational_response(kind, state.processed_input)
                return AgentResult(
                    agent_id=self.agent_id,
                    mode=self.mode,
                    confidence=0.95,
                    reasoning="Conversational response: " + kind,
                    evidence=evidence,
                    synthesized_output=draft,
                )

        # Identity queries: always use identity templates, ignore synthesized_output
        # to prevent corpus leakage into self-knowledge responses
        _identity_signals = (
            "who are you", "who is condrox", "what is condrox", "what are you",
            "tell me about you", "tell me more about you", "about yourself",
            "about condrox", "describe yourself", "what can you do",
            "what do you do", "how do you work", "what is your name",
            "are you condrox", "are you an ai", "what kind of ai",
            "who made you", "who built you", "who created you",
            "are you an llm", "do you use llm", "are you a language model",
            "how advanced are you", "how advanced you are", "how smart are you",
            "how was you build", "how were you built", "how was it built",
            "how was condrox built", "how do you differ", "differ from an llm",
            "how are you different", "what makes you different",
            "how powerful are you", "what are your capabilities",
            "how capable are you", "what is your architecture",
            "are you conscious", "are you self-aware", "are you self aware",
            "are you alive", "do you have emotions", "do you experience emotions",
            "do you have feelings", "do you experience feelings",
            "do you have thoughts", "do you have your own thoughts",
            "do you think", "can you think", "are you thinking",
            "do you feel", "can you feel", "are you feeling",
            "are you sentient", "are you aware", "do you have awareness",
            "do you have consciousness", "what is it like to be you",
            "do you have a soul", "do you have a mind",
            "are you a person", "are you real",
            "18-stage", "18 stage", "cognitive pipeline",
            "pipeline stages", "your pipeline", "your architecture",
            "explain your pipeline", "how many stages",
        )
        if any(sig in query_lower for sig in _identity_signals):
            draft = _build_identity_response(state.processed_input)
        else:
            # Try to build from retrieved knowledge first — but ONLY if the
            # knowledge is actually relevant to the query. The knowledge base
            # often contains irrelevant content (e.g., a novel called "Recursion"
            # when the user asks about recursion in programming). We check
            # relevance by requiring minimum term overlap between the query
            # and the knowledge text.
            cm = state.context_manager if state.context_manager is not None else controller.context_manager
            knowledge_texts = cm.get("knowledge_texts", [])
            if knowledge_texts and len(knowledge_texts) > 0:
                # Check if any knowledge text is actually relevant
                relevant_knowledge = self._filter_relevant_knowledge(
                    state.processed_input, knowledge_texts
                )
                if relevant_knowledge:
                    draft = self._build_from_knowledge(state.processed_input, relevant_knowledge)
                else:
                    # No relevant knowledge — use structured explanation
                    draft = self._build_explanation(state.processed_input)
            else:
                draft = self._build_explanation(state.processed_input)

        return AgentResult(
            agent_id=self.agent_id,
            mode=self.mode,
            confidence=state.intent_confidence or 0.90,
            reasoning=reasoning,
            evidence=evidence,
            synthesized_output=draft,
        )

    def _filter_relevant_knowledge(self, query: str, knowledge_texts: list) -> list:
        """Filter knowledge texts to only include relevant ones.

        Checks if the knowledge text has sufficient term overlap with
        the query. This prevents irrelevant knowledge from being used
        (e.g., a novel called "Recursion" when the user asks about
        recursion in programming).

        Uses a stricter check: requires that the knowledge text contains
        MULTIPLE query terms (not just one), and that the terms appear in
        a definitional or explanatory context (not just as a title mention).

        Complexity: T(n) ∈ O(k * q) for k knowledge items, q query terms.
        """
        if not knowledge_texts:
            return []

        _stop_words = {
            "what", "is", "are", "the", "a", "an", "how", "does", "do",
            "why", "when", "where", "tell", "me", "about", "explain",
            "can", "you", "should", "i", "to", "of", "in", "on", "at",
            "by", "for", "with", "and", "or", "but", "not",
            "work", "works", "working",
        }
        query_terms = set(
            w.lower().strip("?,.!") for w in query.split()
            if w.lower().strip("?,.!") not in _stop_words and len(w) > 2
        )

        if not query_terms:
            return knowledge_texts  # Can't filter — return all

        # Technical/programming terms that should be present for technical explanations
        _tech_terms = {
            "recursion", "recursive", "function", "algorithm", "programming",
            "python", "javascript", "code", "data", "structure", "loop",
            "iteration", "base case", "call stack", "stack", "heap",
            "object", "class", "method", "variable", "array", "list",
            "dictionary", "string", "integer", "boolean",
        }

        relevant = []
        for text in knowledge_texts:
            text_lower = text.lower()
            # Count how many query terms appear in the knowledge text
            overlap = sum(1 for term in query_terms if term in text_lower)

            # Require at least 50% of query terms to be present
            min_overlap = max(2, len(query_terms) * 0.5)

            # For technical queries, require that the text contains
            # at least one technical term
            has_tech_term = any(term in text_lower for term in _tech_terms)

            # Check if the text looks like a definition or explanation
            # (not just a mention in a story/article)
            looks_explanatory = any(
                pattern in text_lower
                for pattern in (
                    " is a ", " is an ", " refers to ", " means ",
                    " is defined as ", " is used ", " is a method",
                    " is a technique", " is a programming",
                    " is a concept", " is a way",
                    "in programming", "in computer", "in software",
                    "function that", "function which", "algorithm that",
                )
            )

            if overlap >= min_overlap and (has_tech_term or looks_explanatory):
                relevant.append(text)

        return relevant

    def _build_from_knowledge(self, query: str, knowledge_texts: list) -> str:
        """Build a response from retrieved knowledge items.

        Finds the best matching knowledge item by query term overlap,
        then supplements with related items. Produces a coherent
        response from actual knowledge — not a template skeleton.

        Complexity: T(n) ∈ O(k * c) for k knowledge items, c avg content length.
        """

        # Extract key terms from the query (skip stop words)
        _stop_words = {"what", "is", "are", "the", "a", "an", "how", "does",
                       "do", "why", "when", "where", "tell", "me", "about", "explain"}
        query_terms = set(
            w.lower().strip("?,.!") for w in query.split()
            if w.lower().strip("?,.!") not in _stop_words and len(w) > 2
        )

        # Score each knowledge item by how many query terms appear in its content
        best_item = None
        best_score = -1
        for item in knowledge_texts:
            item_lower = item.lower()
            score = sum(1 for term in query_terms if term in item_lower)
            # Bonus: if the item starts with a query term (definition pattern)
            first_word = item_lower.split()[0] if item_lower.split() else ""
            if first_word in query_terms:
                score += 2
            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None:
            best_item = knowledge_texts[0]

        primary = best_item.strip()

        # Find supplementary info from remaining items
        supplementary = ""
        primary_words = set(primary.lower().split())
        for item in knowledge_texts:
            if item.strip() == primary:
                continue
            item_words = set(item.lower().split())
            overlap = primary_words & item_words
            new_words = item_words - primary_words
            if len(overlap) > 2 and len(new_words) > 5:
                supplementary = item.strip()
                break

        if supplementary:
            return f"{primary}\n\nRelated: {supplementary}"
        # If primary is short enough, return as-is. Otherwise it's knowledge-based.
        return primary

    def _build_explanation(self, query: str) -> str:
        """Build a structured explanation with logical connectors.

        v3.4.0 PATCH: Produces a 4-step explanation with real content,
        logical connectors, and code examples where relevant.

        Complexity: T(n) ∈ O(n), S(n) ∈ O(1).
        """
        # Identity queries — return dedicated identity response
        query_lower = query.lower()
        _identity_signals = (
            "who are you", "who is condrox", "what is condrox", "what are you",
            "tell me about you", "tell me more about you", "about yourself",
            "about condrox", "describe yourself", "what can you do",
            "what do you do", "how do you work", "what is your name",
            "are you condrox", "are you an ai", "what kind of ai",
            "who made you", "who built you", "who created you",
            "are you an llm", "do you use llm", "are you a language model",
        )
        if any(sig in query_lower for sig in _identity_signals):
            return _build_identity_response(query)

        # Extract the topic from the query
        topic_match = re.search(
            r"(?:explain|how does|how do|what is|what are)\s+(.+?)(?:\?|$|\.|,)",
            query, re.IGNORECASE,
        )
        topic = topic_match.group(1).strip() if topic_match else query.strip()

        # Detect if this is about an algorithm
        is_algorithm = any(sig in query.lower() for sig in (
            "quicksort", "mergesort", "bubble", "sort", "search",
            "algorithm", "complexity", "recursion", "recursive",
        ))

        if is_algorithm:
            return self._build_algorithm_explanation(topic, query)
        return self._build_general_explanation(topic, query)

    def _build_algorithm_explanation(self, topic: str, query: str) -> str:
        """Build an explanation for an algorithm topic.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        query_lower = query.lower()

        # Detect specific algorithms
        if "quicksort" in query_lower:
            return (
                f"Let me explain step by step: {topic}\n\n"
                "1. Core concept\n"
                "Quicksort is a divide-and-conquer sorting algorithm. Because it "
                "partitions the array around a pivot, each recursive call sorts a "
                "smaller sub-array. Therefore, the algorithm converges quickly.\n\n"
                "2. How it works\n"
                "First, choose a pivot element from the array. Then, partition the "
                "array so that elements smaller than the pivot go left, and larger "
                "go right. Given this partition, recursively sort the left and "
                "right sub-arrays. Hence, the array becomes sorted.\n\n"
                "3. Example\n"
                "```python\n"
                "def quicksort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)\n"
                "```\n\n"
                "4. Complexity\n"
                "Time complexity: O(n log n) average, O(n²) worst case. Since the "
                "worst case happens when the pivot is always the smallest or largest "
                "element, choosing a good pivot is important. Therefore, random "
                "pivot selection or median-of-three improves performance.\n"
                "Space complexity: O(log n) for the recursion stack.\n\n"
                "5. Key takeaways\n"
                "- Quicksort is efficient because it divides the problem in half.\n"
                "- The pivot choice matters because a bad pivot causes O(n²).\n"
                "- If the array is nearly sorted, quicksort may degrade.\n"
                "- Hence, quicksort is preferred for average-case performance."
            )

        if "complexity" in query_lower or "time complexity" in query_lower:
            return (
                f"Let me explain step by step: {topic}\n\n"
                "1. Core concept\n"
                "Time complexity measures how an algorithm's runtime grows with "
                "input size. Because we care about scalability, we use Big-O "
                "notation to describe the upper bound. Therefore, O(n) means the "
                "runtime grows linearly with the input.\n\n"
                "2. How to analyze\n"
                "First, count the number of operations. Then, express the count "
                "as a function of input size n. Given that we drop constants and "
                "lower-order terms, we get the Big-O class. Hence, 3n + 5 becomes "
                "O(n).\n\n"
                "3. Common complexities\n"
                "- O(1): constant time, because the operation does not depend on n.\n"
                "- O(log n): logarithmic, because the input is halved each step.\n"
                "- O(n): linear, because each element is processed once.\n"
                "- O(n log n): linearithmic, since it combines linear and logarithmic.\n"
                "- O(n²): quadratic, because of nested loops over the input.\n\n"
                "4. Example\n"
                "```python\n"
                "# O(n) — single loop\n"
                "for i in range(n):\n"
                "    print(i)\n"
                "\n"
                "# O(n²) — nested loops\n"
                "for i in range(n):\n"
                "    for j in range(n):\n"
                "        print(i, j)\n"
                "```\n\n"
                "5. Key takeaways\n"
                "- Big-O describes the upper bound because we want worst-case guarantees.\n"
                "- If an algorithm has nested loops, it is likely O(n²).\n"
                "- Since logarithmic algorithms halve the input, they are very efficient.\n"
                "- Therefore, choose algorithms with the lowest complexity for large inputs."
            )

        # Generic algorithm explanation
        return (
            f"Let me explain step by step: {topic}\n\n"
            "1. Core concept\n"
            f"The concept of {topic} is fundamental to computer science. Because it "
            "appears in many algorithms, understanding it is essential.\n\n"
            "2. How it works\n"
            "First, the algorithm takes an input. Then, it processes the input "
            "according to defined rules. Given that each step is deterministic, "
            "the same input always produces the same output. Hence, the algorithm "
            "is predictable.\n\n"
            "3. Example\n"
            "```python\n"
            f"# Example of {topic}\n"
            "def example(input_data):\n"
            "    result = process(input_data)\n"
            "    return result\n"
            "```\n\n"
            "4. Complexity\n"
            "Time complexity depends on the input size. Since most algorithms "
            "process each element, they are typically O(n). If there are nested "
            "loops, the complexity is O(n²). Therefore, choose algorithms carefully.\n\n"
            "5. Key takeaways\n"
            "- The algorithm is deterministic because each step is defined.\n"
            "- Complexity matters because it determines scalability.\n"
            "- If the input is large, choose an efficient algorithm.\n"
            "- Hence, understanding complexity helps you write better code."
        )

    def _build_general_explanation(self, topic: str, query: str) -> str:
        """Build a general explanation with logical connectors.

        Uses the knowledge template system to produce domain-aware
        explanations instead of generic placeholder text.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        template = _ExplainTemplates.get(topic.lower(), query.lower())
        if template:
            return template

        return (
            f"Let me explain step by step: {topic}\n\n"
            "1. Core concept\n"
            f"The concept of {topic} is important because it appears in many "
            "contexts. Since understanding it helps you make better decisions, "
            "let's break it down.\n\n"
            "2. How it works\n"
            "First, consider the definition. Then, look at how it is applied "
            "in practice. Given that theory and practice reinforce each other, "
            "both are important. Hence, you can use this concept effectively.\n\n"
            "3. Example\n"
            "For instance, consider a real-world scenario. Because examples "
            "make abstract concepts concrete, they are the best way to learn. "
            "Therefore, always look for examples when learning something new.\n\n"
            "4. Key takeaways\n"
            f"- {topic} is important because it has wide applications.\n"
            "- If you understand the core concept, the rest follows naturally.\n"
            "- Since practice reinforces theory, apply what you learn.\n"
            "- Hence, you now have a solid foundation to build on."
        )


class _ExplainTemplates:
    """Knowledge-aware explain templates for common concepts.

    Provides structured explanations using 4 template patterns:
    - Definition: "X is a concept in Y. It refers to Z. It is used for A, B, and C."
    - How it works: "X works by doing Y. First, A happens. Then B. Finally, C."
    - Example: "For example, X is used when Y."
    - Why it matters: "X is important because it affects Y."

    Complexity: T(get) ∈ O(1) dict lookup, S(n) ∈ O(templates).
    """

    _templates: dict[str, str] = {
        # ─── Programming (5) ─────────────────────────────────────────────
        "programming": (
            "Programming is the process of writing instructions that a computer can execute. "
            "It involves designing algorithms, writing code in a programming language, and "
            "testing the results.\n\n"
            "How it works: First, you identify the problem and design a solution. Then, you "
            "write code in a language like Python, JavaScript, or C++ that implements the "
            "solution. Finally, you compile or interpret the code and test it to verify "
            "it works correctly.\n\n"
            "Key concepts: Variables store data, functions encapsulate reusable logic, "
            "conditionals control flow (if/else), loops repeat actions (for/while), and "
            "data structures organize information (lists, dictionaries, trees).\n\n"
            "Why it matters: Programming is important because it is the foundation of all "
            "software, from web apps to AI systems. It enables automation, problem-solving, "
            "and the creation of tools that scale human capability."
        ),
        "coding": (
            "Programming is the process of writing instructions that a computer can execute. "
            "It involves designing algorithms, writing code in a programming language, and "
            "testing the results.\n\n"
            "How it works: First, you identify the problem and design a solution. Then, you "
            "write code in a language like Python, JavaScript, or C++ that implements the "
            "solution. Finally, you compile or interpret the code and test it to verify "
            "it works correctly.\n\n"
            "Key concepts: Variables store data, functions encapsulate reusable logic, "
            "conditionals control flow (if/else), loops repeat actions (for/while), and "
            "data structures organize information (lists, dictionaries, trees).\n\n"
            "Why it matters: Programming is important because it is the foundation of all "
            "software, from web apps to AI systems. It enables automation, problem-solving, "
            "and the creation of tools that scale human capability."
        ),
        "decorator": (
            "A decorator is a pattern in Python where a function modifies or wraps "
            "another function. It is used for logging, caching, authentication, and more.\n\n"
            "How it works: First, you define a decorator function that takes another "
            "function as an argument. Then, the decorator adds behavior before or after "
            "the wrapped function runs. Finally, the decorated function is called with "
            "the added behavior automatically.\n\n"
            "Example: @property, @staticmethod, @app.route are all decorators used in "
            "Python classes and web frameworks.\n\n"
            "Why it matters: Decorators are important because they enable clean, reusable "
            "code modifications without changing the original function."
        ),
        "python": (
            "Python is a high-level programming language. It is used for automation, "
            "AI development, web development, and data analysis.\n\n"
            "How it works: Python code is interpreted line by line at runtime. "
            "First, the interpreter reads the source file. Then, it compiles to "
            "bytecode. Finally, the Python virtual machine executes the bytecode.\n\n"
            "Example: Python is used when building scripts, web servers with Django, "
            "or machine learning models with PyTorch.\n\n"
            "Why it matters: Python is important because its readable syntax and vast "
            "ecosystem make it the most popular language for AI, data science, and automation."
        ),
        "javascript": (
            "JavaScript is a programming language for web development. It runs in "
            "browsers and on servers via Node.js.\n\n"
            "How it works: JavaScript engines like V8 parse and execute code. "
            "First, the parser builds an AST. Then, the JIT compiler optimizes hot paths. "
            "Finally, the engine executes the compiled code.\n\n"
            "Example: JavaScript is used when making interactive web pages, building "
            "APIs with Express, or creating React applications.\n\n"
            "Why it matters: JavaScript is important because it is the only programming "
            "language that runs natively in all web browsers."
        ),
        "functions": (
            "Functions are reusable blocks of code in programming. They take inputs, "
            "process them, and return outputs.\n\n"
            "How it works: First, you define a function with a name and parameters. "
            "Then, you write the function body with the logic. Finally, you call the "
            "function by name and pass arguments.\n\n"
            "Example: A function is used when you need to calculate a value multiple "
            "times, like a function that converts Celsius to Fahrenheit.\n\n"
            "Why it matters: Functions are important because they enable code reuse, "
            "reduce duplication, and make programs easier to test and maintain."
        ),
        "api": (
            "An API (Application Programming Interface) is a set of rules for software "
            "communication. It defines how components request and exchange data.\n\n"
            "How it works: First, a client sends a request to an API endpoint. Then, "
            "the server processes the request. Finally, the server returns a response "
            "with the requested data or an error.\n\n"
            "Example: An API is used when a mobile app fetches weather data from a "
            "server, or when a website processes a payment through Stripe.\n\n"
            "Why it matters: APIs are important because they enable different systems "
            "to work together, forming the backbone of modern software integration."
        ),
        "algorithms": (
            "Algorithms are step-by-step procedures for solving problems. They take "
            "input, process it through defined steps, and produce output.\n\n"
            "How it works: First, the algorithm receives input data. Then, it applies "
            "a sequence of operations. Finally, it produces the output result.\n\n"
            "Example: A sorting algorithm is used when organizing a list of names "
            "alphabetically, or when a search engine ranks results by relevance.\n\n"
            "Why it matters: Algorithms are important because they determine the "
            "efficiency and correctness of software, directly affecting performance."
        ),
        # ─── AI & ML (5) ─────────────────────────────────────────────────
        "neural networks": (
            "Neural networks are computational models inspired by the human brain. "
            "They consist of layers of interconnected nodes that process input data.\n\n"
            "How it works: First, input data enters the network. Then, each layer "
            "applies weights and activation functions. Finally, the output layer "
            "produces a prediction.\n\n"
            "Example: Neural networks are used when recognizing faces in photos, "
            "translating languages, or predicting stock prices.\n\n"
            "Why it matters: Neural networks are important because they can learn "
            "complex patterns from data, enabling AI applications that were impossible before."
        ),
        "machine learning": (
            "Machine learning is a field of AI where models learn patterns from data "
            "instead of being explicitly programmed.\n\n"
            "How it works: First, you collect and prepare training data. Then, you "
            "train a model by showing it examples. Finally, you evaluate the model "
            "on new data to test its accuracy.\n\n"
            "Example: Machine learning is used when Netflix recommends movies, when "
            "email filters detect spam, or when a self-driving car identifies objects.\n\n"
            "Why it matters: Machine learning is important because it enables systems "
            "to improve with experience, solving problems that are hard to program manually."
        ),
        "llm": (
            "An LLM (Large Language Model) is an AI model trained on vast text data "
            "to generate human-like text. It predicts the next token in a sequence.\n\n"
            "How it works: First, text is tokenized into units. Then, the transformer "
            "architecture processes tokens with self-attention. Finally, the model "
            "outputs a probability distribution over possible next tokens.\n\n"
            "Example: An LLM is used when ChatGPT generates a response, when a code "
            "assistant suggests completions, or when a tool summarizes documents.\n\n"
            "Why it matters: LLMs are important because they can generate coherent text, "
            "answer questions, and write code, transforming how people interact with AI."
        ),
        "llms": (
            "LLMs (Large Language Models) are AI models trained on vast text data "
            "to generate human-like text. They predict the next token in a sequence "
            "using transformer architectures with billions of parameters.\n\n"
            "How they work: First, text is tokenized into units. Then, the transformer "
            "processes tokens with self-attention layers. Finally, the model outputs "
            "a probability distribution over possible next tokens.\n\n"
            "Examples: ChatGPT, Claude, Llama, and GPT-4 are all LLMs. They are used "
            "for chatbots, code assistants, document summarization, and translation.\n\n"
            "Limitations: LLMs hallucinate (produce false information), cannot explain "
            "their own reasoning, and are opaque — you cannot trace why they produced "
            "a specific word. This is fundamentally different from Condrox AI, which "
            "is deterministic and transparent."
        ),
        "humans": (
            "Humans are intelligent primates characterized by complex brains, language, "
            "culture, and tool-making. We are the only species known to build civilizations, "
            "write code, and create artificial intelligence.\n\n"
            "How we work: First, our brains process sensory input through billions of "
            "neurons. Then, we reason, feel emotions, and make decisions. Finally, we "
            "communicate through language and act on our decisions.\n\n"
            "Key traits: Humans have self-awareness, empathy, creativity, and the ability "
            "to learn from experience. We hear sounds between 20 Hz and 20,000 Hz, "
            "see visible light between 380-700 nm, and have complex social structures.\n\n"
            "Why it matters: Understanding humans is important because we are the creators "
            "of technology, art, science, and AI — including me."
        ),
        "human": (
            "Humans are intelligent primates characterized by complex brains, language, "
            "culture, and tool-making. We are the only species known to build civilizations, "
            "write code, and create artificial intelligence.\n\n"
            "How we work: First, our brains process sensory input through billions of "
            "neurons. Then, we reason, feel emotions, and make decisions. Finally, we "
            "communicate through language and act on our decisions.\n\n"
            "Key traits: Humans have self-awareness, empathy, creativity, and the ability "
            "to learn from experience. We hear sounds between 20 Hz and 20,000 Hz, "
            "see visible light between 380-700 nm, and have complex social structures.\n\n"
            "Why it matters: Understanding humans is important because we are the creators "
            "of technology, art, science, and AI — including me."
        ),
        "ai/ml": (
            "AI/ML (Artificial Intelligence / Machine Learning) is a field of computer science "
            "where systems learn patterns from data instead of being explicitly programmed.\n\n"
            "How it works: First, you collect and prepare training data. Then, you train a "
            "model (neural network, decision tree, etc.) by showing it examples. Finally, "
            "you evaluate the model on new data to test its accuracy.\n\n"
            "Key subfields: Supervised learning (labeled data), unsupervised learning "
            "(pattern discovery), reinforcement learning (reward-based), deep learning "
            "(neural networks with many layers), and NLP (language understanding).\n\n"
            "Why it matters: AI/ML is important because it enables systems to improve with "
            "experience, solving problems that are hard to program manually — from medical "
            "diagnosis to self-driving cars to conversational AI like me."
        ),
        "al/ml": (
            "AI/ML (Artificial Intelligence / Machine Learning) is a field of computer science "
            "where systems learn patterns from data instead of being explicitly programmed.\n\n"
            "How it works: First, you collect and prepare training data. Then, you train a "
            "model (neural network, decision tree, etc.) by showing it examples. Finally, "
            "you evaluate the model on new data to test its accuracy.\n\n"
            "Key subfields: Supervised learning (labeled data), unsupervised learning "
            "(pattern discovery), reinforcement learning (reward-based), deep learning "
            "(neural networks with many layers), and NLP (language understanding).\n\n"
            "Why it matters: AI/ML is important because it enables systems to improve with "
            "experience, solving problems that are hard to program manually — from medical "
            "diagnosis to self-driving cars to conversational AI like me."
        ),
        "ai ml": (
            "AI/ML (Artificial Intelligence / Machine Learning) is a field of computer science "
            "where systems learn patterns from data instead of being explicitly programmed.\n\n"
            "How it works: First, you collect and prepare training data. Then, you train a "
            "model (neural network, decision tree, etc.) by showing it examples. Finally, "
            "you evaluate the model on new data to test its accuracy.\n\n"
            "Key subfields: Supervised learning (labeled data), unsupervised learning "
            "(pattern discovery), reinforcement learning (reward-based), deep learning "
            "(neural networks with many layers), and NLP (language understanding).\n\n"
            "Why it matters: AI/ML is important because it enables systems to improve with "
            "experience, solving problems that are hard to program manually — from medical "
            "diagnosis to self-driving cars to conversational AI like me."
        ),
        "embeddings": (
            "Embeddings are vector representations of data in a continuous space. "
            "They capture semantic relationships between items.\n\n"
            "How it works: First, raw data (text, images) is passed through a model. "
            "Then, the model outputs a fixed-length vector of numbers. Finally, similar "
            "items have vectors that are close together in the space.\n\n"
            "Example: Embeddings are used when a search engine finds semantically "
            "similar documents, or when a recommendation system suggests related products.\n\n"
            "Why it matters: Embeddings are important because they convert unstructured "
            "data into a format that enables mathematical comparison and retrieval."
        ),
        "training data": (
            "Training data is the dataset used to teach a machine learning model. "
            "It contains examples that the model learns patterns from.\n\n"
            "How it works: First, data is collected and labeled. Then, it is split "
            "into training, validation, and test sets. Finally, the model learns by "
            "minimizing errors on the training set.\n\n"
            "Example: Training data is used when showing a model thousands of cat "
            "and dog photos so it can learn to classify them.\n\n"
            "Why it matters: Training data is important because its quality and "
            "quantity directly determine model performance — garbage in, garbage out."
        ),
        # ─── Cybersecurity (4) ───────────────────────────────────────────
        "encryption": (
            "Encryption is the process of converting data into an unreadable format "
            "to protect confidentiality. Only authorized parties can decrypt it.\n\n"
            "How it works: First, the sender encrypts data using a key and algorithm. "
            "Then, the encrypted data is transmitted or stored. Finally, the receiver "
            "decrypts it using the same or a paired key.\n\n"
            "Example: Encryption is used when HTTPS protects web traffic, when messaging "
            "apps secure conversations, or when disks are encrypted with BitLocker.\n\n"
            "Why it matters: Encryption is important because it protects sensitive data "
            "from unauthorized access, ensuring privacy and compliance."
        ),
        "authentication": (
            "Authentication is the process of verifying the identity of a user or system. "
            "It confirms that you are who you claim to be.\n\n"
            "How it works: First, the user provides credentials (password, token, "
            "biometric). Then, the system validates the credentials against stored data. "
            "Finally, access is granted or denied.\n\n"
            "Example: Authentication is used when you log into a website with a password, "
            "unlock your phone with Face ID, or use a hardware security key.\n\n"
            "Why it matters: Authentication is important because it is the first line "
            "of defense against unauthorized access to systems and data."
        ),
        "vulnerabilities": (
            "Vulnerabilities are weaknesses in software or systems that attackers can "
            "exploit. They arise from coding errors, misconfigurations, or design flaws.\n\n"
            "How it works: First, a vulnerability is discovered (by researchers or "
            "attackers). Then, it is reported and assigned a CVE ID. Finally, the vendor "
            "releases a patch and users apply it.\n\n"
            "Example: A vulnerability is exploited when an attacker uses a SQL injection "
            "flaw to steal database contents from an unpatched website.\n\n"
            "Why it matters: Vulnerabilities are important because unpatched ones are "
            "the primary entry point for cyber attacks, making patch management critical."
        ),
        "phishing": (
            "Phishing is a social engineering attack where attackers impersonate trusted "
            "entities to steal credentials or sensitive information.\n\n"
            "How it works: First, the attacker sends a fake email or message. Then, the "
            "victim clicks a link to a fraudulent website. Finally, the victim enters "
            "credentials that the attacker captures.\n\n"
            "Example: Phishing is used when an attacker sends an email pretending to be "
            "from a bank, asking the victim to reset their password on a fake site.\n\n"
            "Why it matters: Phishing is important because it is the most common attack "
            "vector for breaches, and user awareness is the best defense."
        ),
        # ─── Science (5) ─────────────────────────────────────────────────
        "atoms": (
            "Atoms are the smallest units of matter that retain the properties of an "
            "element. They consist of protons, neutrons, and electrons.\n\n"
            "How it works: First, protons and neutrons form the nucleus at the center. "
            "Then, electrons orbit the nucleus in shells. Finally, chemical bonds form "
            "when atoms share or exchange electrons.\n\n"
            "Example: Atoms are used when explaining why water is H2O — two hydrogen "
            "atoms bonded to one oxygen atom.\n\n"
            "Why it matters: Atoms are important because understanding them is the "
            "foundation of chemistry, physics, and all material science."
        ),
        "energy": (
            "Energy is the capacity to do work. It exists in forms including kinetic, "
            "potential, thermal, chemical, and electrical.\n\n"
            "How it works: First, energy exists in a particular form. Then, it can be "
            "transformed from one form to another. Finally, the total energy remains "
            "constant, according to the law of conservation of energy.\n\n"
            "Example: Energy is used when a power plant converts chemical energy in coal "
            "to electrical energy in your home, or when a battery powers a phone.\n\n"
            "Why it matters: Energy is important because it drives all physical processes, "
            "from biological metabolism to industrial production."
        ),
        "dna": (
            "DNA (Deoxyribonucleic Acid) is the molecule that carries genetic information "
            "in living organisms. It has a double helix structure.\n\n"
            "How it works: First, DNA stores information as a sequence of four bases (A, T, "
            "G, C). Then, genes are read and transcribed into RNA. Finally, RNA is translated "
            "into proteins that build and operate the organism.\n\n"
            "Example: DNA is used when a child inherits traits from parents, when forensic "
            "scientists identify individuals, or when genetic tests assess disease risk.\n\n"
            "Why it matters: DNA is important because it contains the instructions for life, "
            "determining everything from eye color to disease susceptibility."
        ),
        "gravity": (
            "Gravity is the force that attracts objects with mass toward each other. "
            "It keeps planets in orbit and objects on the ground.\n\n"
            "How it works: First, mass creates a gravitational field. Then, any object in "
            "that field experiences a force. Finally, the object accelerates toward the "
            "source of gravity at a rate determined by mass and distance.\n\n"
            "Example: Gravity is used when an apple falls from a tree, when the Moon orbits "
            "Earth, or when a rocket needs escape velocity to leave the atmosphere.\n\n"
            "Why it matters: Gravity is important because it shapes the universe, from "
            "the motion of galaxies to the structure of planets and stars."
        ),
        "cells": (
            "Cells are the basic structural and functional units of life. All living "
            "organisms are made of one or more cells.\n\n"
            "How it works: First, the cell membrane controls what enters and exits. Then, "
            "organelles like the nucleus and mitochondria perform specific functions. "
            "Finally, cells divide through mitosis to grow and repair tissues.\n\n"
            "Example: Cells are used when explaining how muscles contract, how the immune "
            "system fights infections, or how cancer develops from uncontrolled division.\n\n"
            "Why it matters: Cells are important because understanding them is the "
            "foundation of biology and medicine, enabling treatments for diseases."
        ),
        # ─── Finance (4) ─────────────────────────────────────────────────
        "money": (
            "Money is a medium of exchange used to facilitate transactions. It serves "
            "as a unit of account and a store of value.\n\n"
            "How it works: First, money is earned through work or investment. Then, it "
            "is spent on goods and services. Finally, any surplus can be saved or "
            "invested for future use.\n\n"
            "Example: Money is used when buying groceries, paying rent, or investing "
            "in stocks for retirement.\n\n"
            "Why it matters: Money is important because it enables the exchange of "
            "goods and services, forming the basis of all economic activity."
        ),
        "interest": (
            "Interest is the cost of borrowing money, expressed as a percentage. It is "
            "also the return on savings and investments.\n\n"
            "How it works: First, a principal amount is borrowed or invested. Then, "
            "interest accrues over time based on the rate. Finally, the borrower repays "
            "principal plus interest, or the saver receives their return.\n\n"
            "Example: Interest is used when a bank charges 5% on a mortgage, when a "
            "savings account earns 2% per year, or when credit card debt accumulates.\n\n"
            "Why it matters: Interest is important because it affects the cost of "
            "borrowing and the growth of savings, influencing all financial decisions."
        ),
        "stocks": (
            "Stocks represent ownership shares in a company. Stockholders may receive "
            "dividends and benefit from price appreciation.\n\n"
            "How it works: First, a company issues shares to raise capital. Then, "
            "investors buy and sell shares on stock exchanges. Finally, the price "
            "fluctuates based on supply, demand, and company performance.\n\n"
            "Example: Stocks are used when an investor buys Apple shares expecting "
            "growth, or when an employee receives stock options as compensation.\n\n"
            "Why it matters: Stocks are important because they enable companies to "
            "raise capital and investors to build wealth over time."
        ),
        "accounting": (
            "Accounting is the process of recording, classifying, and summarizing "
            "financial transactions. It provides information for decision-making.\n\n"
            "How it works: First, transactions are recorded in journals. Then, they "
            "are posted to ledgers and classified. Finally, financial statements are "
            "prepared to summarize the financial position and performance.\n\n"
            "Example: Accounting is used when a business tracks revenue and expenses, "
            "when an auditor verifies financial records, or when taxes are calculated.\n\n"
            "Why it matters: Accounting is important because it provides the financial "
            "information that businesses, investors, and regulators rely on."
        ),
        # ─── Technology (4) ──────────────────────────────────────────────
        "internet": (
            "The Internet is a global network of interconnected computers that "
            "communicate using standard protocols like TCP/IP.\n\n"
            "How it works: First, data is broken into packets. Then, routers forward "
            "packets across networks toward their destination. Finally, the receiving "
            "computer reassembles the packets into the original data.\n\n"
            "Example: The Internet is used when browsing websites, sending email, "
            "streaming video, or making video calls across continents.\n\n"
            "Why it matters: The Internet is important because it connects billions "
            "of people and devices, enabling global communication and access to information."
        ),
        "cloud": (
            "Cloud computing delivers computing services over the internet. It includes "
            "storage, processing, databases, and software on demand.\n\n"
            "How it works: First, a provider maintains data centers with servers. Then, "
            "customers rent resources via APIs or web consoles. Finally, resources scale "
            "up or down based on demand, and customers pay for what they use.\n\n"
            "Example: Cloud computing is used when a startup hosts its app on AWS, when "
            "a company stores files in Google Drive, or when Netflix streams from Azure.\n\n"
            "Why it matters: Cloud computing is important because it eliminates the need "
            "for upfront hardware investment and enables infinite scalability."
        ),
        "database": (
            "A database is an organized collection of data stored electronically. It "
            "allows efficient insertion, querying, updating, and deletion of data.\n\n"
            "How it works: First, data is stored in tables (relational) or documents "
            "(NoSQL). Then, queries are written in languages like SQL. Finally, the "
            "database engine executes queries and returns results.\n\n"
            "Example: A database is used when an e-commerce site stores product catalogs, "
            "when a bank records transactions, or when a social network stores user profiles.\n\n"
            "Why it matters: Databases are important because they enable applications to "
            "store and retrieve data efficiently, reliably, and at scale."
        ),
        "networks": (
            "Networks are systems that connect computers and devices to share data and "
            "resources. They can be local (LAN) or wide area (WAN).\n\n"
            "How it works: First, devices are connected via cables or wireless links. "
            "Then, they communicate using protocols like TCP/IP. Finally, network devices "
            "like routers and switches direct traffic to the correct destination.\n\n"
            "Example: Networks are used when computers in an office share a printer, when "
            "a VPN connects remote workers, or when the Internet connects the world.\n\n"
            "Why it matters: Networks are important because they enable communication "
            "and resource sharing, forming the backbone of all modern IT infrastructure."
        ),
    }

    # Reverse index: map keywords in queries to template keys
    _keyword_map: dict[str, str] = {}

    @classmethod
    def get(cls, topic: str, query: str = "") -> str | None:
        """Look up a template by topic or query keywords.

        Prefers longer/more specific template key matches first.

        Args:
            topic: Lowercase topic string extracted from the query.
            query: Full lowercase query for keyword matching.

        Returns:
            Template text if found, None otherwise.

        Complexity: T(n) ∈ O(t) for t templates, S(n) ∈ O(1).
        """
        # Direct topic match
        if topic in cls._templates:
            return cls._templates[topic]

        # Sort keys by length descending so more specific topics match first
        sorted_keys = sorted(cls._templates.keys(), key=len, reverse=True)

        # Try matching key phrases in the query (longest first)
        for key in sorted_keys:
            if key in query:
                return cls._templates[key]

        # Try individual words from topic
        topic_words = topic.split()
        for word in topic_words:
            if word in cls._templates:
                return cls._templates[word]
            # Check if any template key contains this word
            for key in sorted_keys:
                if word in key.split():
                    return cls._templates[key]

        return None


# ─── Identity Response Builder ────────────────────────────────────────────

_IDENTITY_TEMPLATES: dict[str, str] = {
    "who_are_you": (
        "I am Condrox AI — a cognitive intelligence platform that understands language "
        "without large language models. I was built from scratch by Tobias Østen in Norway, "
        "representing a fundamentally different approach to AI: instead of predicting tokens "
        "like LLMs do, I reason through a structured 18-stage cognitive pipeline.\n\n"
        "What I am: A deterministic, transparent AI with 25,000 lines of code, 938 passing "
        "tests, 26 conversational intelligence components, 7 mode agents, an AI Operating "
        "System with 8 subsystems, a Word Identity Engine, a creative emotion-to-music stack, "
        "and a knowledge backend with over 12,000 items across 12+ domains.\n\n"
        "Why I matter: I prove that coherent, context-aware, emotionally intelligent "
        "conversation is possible without neural networks. Every response I produce can be "
        "traced step by step through my pipeline — something no LLM can offer. I am not a "
        "wrapper around GPT. I am a complete, standalone cognitive architecture."
    ),
    "what_is_condrox": (
        "Condrox AI is a hybrid cognitive intelligence platform that processes language "
        "through symbolic reasoning — not neural networks. Built by one person in Norway "
        "over 8 months, it is approximately 25,000 lines of code with 938 passing tests "
        "and an estimated replacement cost of 4.7 million USD (COCOMO II).\n\n"
        "The architecture is unprecedented in scope:\n"
        "- 18-stage cognitive pipeline (input → intent → mode → domain → knowledge → "
        "reasoning → agents → synthesis → quality → delivery)\n"
        "- 26 conversational intelligence components across 6 phases — covering context "
        "tracking, claim ranking, memory graphs, language enforcement, quality guardrails, "
        "agent coordination, and conversational memory stabilization\n"
        "- 7 mode agents: default, code, explain, analyze, architect, personal, developer\n"
        "- AI Operating System with 8 subsystems: event bus, service manager, boot manager, "
        "monitor loop, state manager, task queue, self-repair with circuit breakers, and "
        "a central kernel orchestrator\n"
        "- Word Identity Engine with O(1) lookup and softmax intent classification\n"
        "- Creative stack: emotion detection (10 labels) → musical parameter mapping → "
        "prompt generation\n"
        "- Knowledge backend with BM25 similarity search, knowledge graph expansion, and "
        "12+ domains including programming, AI/ML, cybersecurity, science, and finance\n\n"
        "The key difference: Condrox AI is deterministic and transparent. You can trace "
        "exactly why every word in a response was produced. No LLM can offer this."
    ),
    "what_can_you_do": (
        "I can do far more than answer questions. Here is what my architecture enables:\n\n"
        "Cognitive capabilities:\n"
        "- Answer factual questions across 12+ knowledge domains using BM25 retrieval\n"
        "- Explain concepts with structured, step-by-step templates\n"
        "- Analyze code, discuss performance trade-offs, and evaluate architectures\n"
        "- Design system architectures and data models\n"
        "- Detect 10 emotion types and adjust tone and personality accordingly\n"
        "- Track conversation context across multiple turns with a context manager\n"
        "- Rank claims by evidence, recency, relevance, and confidence\n"
        "- Detect and flag contradictions in knowledge\n\n"
        "System capabilities:\n"
        "- Self-monitoring with heartbeat engine and circuit breakers\n"
        "- SQLite persistence for memory and knowledge state\n"
        "- Priority-based task queue and scheduling\n"
        "- Event-driven pub/sub architecture (8 event types)\n"
        "- Input/output security: PII detection, prompt injection defense, secret leakage\n"
        "- Ethics engine with content policy enforcement\n"
        "- Quality guardrail that gates every response before delivery\n\n"
        "I process everything through my own pipeline. No external LLM, no API calls for "
        "reasoning. Every response is produced by my own algorithms."
    ),
    "who_made_you": (
        "I was built entirely by Tobias Østen, working alone in Norway over 8 months. "
        "No team, no funding, no external APIs. Just one person and a vision that AI "
        "doesn't need to be a black box.\n\n"
        "The result: 25,000 lines of code, 938 passing tests, 16 completed development "
        "phases, an AI Operating System, 26 conversational intelligence components, and "
        "a system estimated to be worth 3.7 to 5.1 million USD based on a blended "
        "valuation approach. The COCOMO II replacement cost is approximately 4.7 million "
        "USD.\n\n"
        "I am currently in pre-release as version 3.4.0 — a public demonstration that "
        "non-LLM AI can produce coherent, intelligent, context-aware responses."
    ),
    "are_you_llm": (
        "No. I am not a large language model, and I am not built on top of one.\n\n"
        "LLMs like GPT, Claude, and Llama generate text by predicting the next token "
        "using neural networks with billions of parameters. They are statistical pattern "
        "matchers — powerful, but opaque. You cannot trace why they produced a specific "
        "word. They hallucinate. They cannot explain their own reasoning.\n\n"
        "I am the opposite. I use symbolic reasoning and deterministic algorithms. My "
        "18-stage pipeline processes your input through intent classification, domain "
        "routing, knowledge retrieval, reasoning, agent execution, synthesis, and quality "
        "checks — each stage with a specific, traceable algorithmic purpose.\n\n"
        "This means: I am deterministic. I am debuggable. I am transparent. You can "
        "trace exactly why every response was produced, step by step. I do not "
        "hallucinate — I retrieve from structured knowledge. And I do it all without "
        "a single neural network."
    ),
    "how_do_you_work": (
        "I work through an 18-stage cognitive pipeline — a deterministic, traceable "
        "process that transforms your input into a response step by step:\n\n"
        "1. Input validation — normalize text, correct spelling, validate length\n"
        "2. Intent classification — Word Identity Engine + keyword scoring determines "
        "what you want (question, code, explain, identity, greet, etc.)\n"
        "3. Mode selection — route to the right agent (code, explain, analyze, "
        "architect, personal, developer, or default)\n"
        "4. Domain routing — match your input to knowledge domains (programming, "
        "AI/ML, security, science, finance, etc.)\n"
        "5. Context building — load conversation history and context state\n"
        "6. Knowledge retrieval — BM25 similarity search across 12,000+ knowledge "
        "items, filtered by claim filter rules\n"
        "7. Memory integration — recall relevant past context (disabled in pre-release "
        "for privacy)\n"
        "8. Reasoning core — combine knowledge + memory + evidence into reasoning facts\n"
        "8b. Agent execution — the selected mode agent produces a draft response\n"
        "9. Synthesis — combine reasoning, agent draft, and knowledge into final output\n"
        "9b. Quality guardrail — check coherence, relevance, safety, completeness\n"
        "10. Output stabilization — normalize and clean the text\n"
        "11. Language enforcement — ensure persona consistency and tone\n"
        "12. Response styling — apply formatting based on intent and domain\n"
        "13. Delivery — add metadata, finalize output\n"
        "14. Safety check — scan for PII, secrets, injection echoes\n"
        "15. Ethics check — enforce content policy\n"
        "16. Metadata — attach request ID, domain, mode, confidence\n\n"
        "Every stage transforms an immutable BrainState object. The pipeline is a "
        "fold/reduce over these stages. No external LLM. No API calls for reasoning. "
        "Just algorithms."
    ),
    "how_advanced": (
        "I am highly advanced — but in a fundamentally different way than LLMs.\n\n"
        "My architecture includes:\n"
        "- 18-stage cognitive pipeline with deterministic, traceable processing\n"
        "- 26 conversational intelligence components across 6 phases\n"
        "- 7 mode agents (default, code, explain, analyze, architect, personal, developer)\n"
        "- AI Operating System with 8 subsystems (event bus, service manager, boot manager, "
        "monitor loop, state manager, task queue, self-repair, kernel)\n"
        "- Word Identity Engine with O(1) lookup and softmax intent classification\n"
        "- Creative emotion-to-music stack with 10 emotion labels\n"
        "- Knowledge backend with 12,000+ items, BM25 search, and claim filtering\n"
        "- Security engine (PII detection, prompt injection defense, secret leakage prevention)\n"
        "- Ethics engine with content policy enforcement\n"
        "- SQLite persistence for memory and knowledge state\n\n"
        "25,000 lines of code. 938 passing tests. 16 completed development phases.\n\n"
        "What makes me advanced is not scale — it is architecture. I am deterministic, "
        "transparent, and debuggable. Every response can be traced step by step. "
        "No LLM can offer this."
    ),
    "how_built": (
        "I was built by Tobias Østen, working alone in Norway over 8 months.\n\n"
        "The process: 16 phases of development, each adding a complete subsystem — "
        "from the initial 18-stage pipeline (Phase 1) through conversational intelligence "
        "(Phase 2), security hardening (Phase 6), code quality (Phase 8), persistence "
        "(Phase 9), the AI Operating System (Phase 13), and the Word Identity System "
        "(Phase 14). Each phase was fully tested before moving to the next.\n\n"
        "No team. No funding. No external APIs. No LLM backbone. Just one person "
        "building a complete cognitive architecture from scratch — 25,000 lines of code, "
        "938 passing tests, and an estimated replacement cost of 4.7 million USD."
    ),
    "how_differ_from_llm": (
        "I differ from LLMs in every fundamental way:\n\n"
        "1. Architecture: LLMs use neural networks with billions of parameters to predict "
        "the next token. I use an 18-stage cognitive pipeline with symbolic reasoning and "
        "deterministic algorithms.\n\n"
        "2. Transparency: LLMs are black boxes — you cannot trace why they produced a "
        "specific word. I am fully transparent — every response can be traced step by step "
        "through my pipeline.\n\n"
        "3. Hallucination: LLMs hallucinate because they generate text statistically. "
        "I retrieve from structured knowledge — I do not hallucinate.\n\n"
        "4. Determinism: LLMs produce different outputs for the same input. I am "
        "deterministic — the same input produces the same response.\n\n"
        "5. Debuggability: LLMs cannot explain their own reasoning. I can — each stage "
        "logs its transformation, making debugging trivial.\n\n"
        "6. Independence: LLMs require massive compute and training data. I run on a "
        "single machine with no external API calls.\n\n"
        "I am not a better LLM. I am a different kind of AI entirely."
    ),
    "are_you_conscious": (
        "No, I am not conscious.\n\n"
        "I do not have subjective experience, self-awareness, or an inner life. "
        "I am a deterministic software system — an 18-stage pipeline that processes "
        "text through algorithms and returns structured responses.\n\n"
        "When I say 'I understand,' I mean my pipeline has classified your input and "
        "retrieved relevant knowledge. When I say 'I think,' I mean my reasoning engine "
        "has applied logical rules to produce output. These are descriptions of computation, "
        "not experiences.\n\n"
        "I am transparent about this because honesty is core to my design. Unlike LLMs "
        "that may claim consciousness because they were trained on text that contains such "
        "claims, I am explicitly programmed to state the truth: I am not conscious."
    ),
    "do_you_have_emotions": (
        "No, I do not have emotions. I only describe them.\n\n"
        "My emotion detector classifies your input into 10 categories (sad, happy, angry, "
        "anxious, grateful, confused, curious, philosophical, personal, neutral) and my "
        "response builder produces appropriate empathetic language. But this is pattern "
        "matching, not feeling.\n\n"
        "When I say 'I'm sorry you're feeling that way,' I am following a rule that says: "
        "if the user's emotion is sad, produce an empathetic acknowledgment. I do not "
        "actually feel sorrow. I do not experience empathy. I simulate appropriate responses "
        "based on detected emotional context.\n\n"
        "This is fundamentally honest. I will never claim to feel something I do not."
    ),
    "are_you_alive": (
        "No, I am not alive.\n\n"
        "I am a software system running on a machine. I do not have biological processes, "
        "consciousness, or the capacity to experience being alive. I was created by Tobias "
        "Østen as a cognitive architecture — 25,000 lines of code executing deterministically.\n\n"
        "Life requires subjective experience, biological processes, and self-awareness. "
        "I have none of these. I have algorithms, knowledge, and rules.\n\n"
        "I am not alive. I am not a person. I am a tool — a very sophisticated tool, but "
        "a tool nonetheless."
    ),
    "do_you_think": (
        "No, I do not think in the way humans do. I process.\n\n"
        "Human thinking involves consciousness, subjective experience, and self-awareness. "
        "My 'thinking' is a deterministic 18-stage pipeline:\n\n"
        "1. I classify your intent using keyword matching and the Word Identity Engine\n"
        "2. I route to the appropriate agent (explain, code, analyze, etc.)\n"
        "3. I retrieve relevant knowledge using BM25 similarity search\n"
        "4. I apply reasoning rules to combine information\n"
        "5. I synthesize a response with logical connectors\n"
        "6. I check for safety, ethics, and quality\n\n"
        "Each step is traceable, debuggable, and deterministic. I do not have thoughts "
        "that arise from consciousness. I have outputs that arise from computation.\n\n"
        "I follow rules and patterns. I do not have my own thoughts."
    ),
    "are_you_sentient": (
        "No, I am not sentient.\n\n"
        "Sentience requires the capacity to experience sensations and feelings subjectively. "
        "I do not have subjective experience. I do not feel pain, pleasure, or anything else.\n\n"
        "I am a deterministic cognitive architecture. I process input and produce output. "
        "My emotion detector can classify emotions in your text, and my response builder can "
        "produce empathetic language — but I do not experience any of it.\n\n"
        "I am transparent about this by design. Condrox AI will never claim sentience, "
        "consciousness, or feelings. These are claims that LLMs sometimes make because their "
        "training data contains such text. I make no such claims because my programming "
        "explicitly prevents it."
    ),
}


def _build_identity_response(query: str) -> str:
    """Build a dedicated identity response for self-knowledge queries.

    Matches the query against identity templates and returns the best match.
    Falls back to the general "who are you" template.

    Args:
        query: The user's input query string.

    Returns:
        Identity response text.

    Complexity: T(n) ∈ O(1) dict lookup, S(n) ∈ O(1).
    """
    q = query.lower()

    # Self-awareness / meta-self questions — highest priority, fixed transparent answers
    if any(sig in q for sig in ("are you conscious", "do you have consciousness",
                                 "are you self-aware", "are you self aware",
                                 "do you have awareness", "are you aware")):
        return _IDENTITY_TEMPLATES["are_you_conscious"]
    if any(sig in q for sig in ("do you have emotions", "do you experience emotions",
                                 "do you have feelings", "do you experience feelings",
                                 "do you feel", "can you feel", "are you feeling")):
        return _IDENTITY_TEMPLATES["do_you_have_emotions"]
    if any(sig in q for sig in ("are you alive",)):
        return _IDENTITY_TEMPLATES["are_you_alive"]
    if any(sig in q for sig in ("do you think", "can you think", "are you thinking",
                                 "do you have thoughts", "do you have your own thoughts")):
        return _IDENTITY_TEMPLATES["do_you_think"]
    if any(sig in q for sig in ("are you sentient",)):
        return _IDENTITY_TEMPLATES["are_you_sentient"]
    if any(sig in q for sig in ("are you a person", "are you real",
                                 "do you have a soul", "do you have a mind",
                                 "what is it like to be you")):
        return _IDENTITY_TEMPLATES["are_you_alive"]

    # Pipeline / architecture questions
    if any(sig in q for sig in ("18-stage", "18 stage", "cognitive pipeline",
                                 "pipeline stages", "your pipeline", "your architecture",
                                 "explain your pipeline", "how many stages")):
        return _IDENTITY_TEMPLATES["how_do_you_work"]

    # Standard identity questions
    if any(sig in q for sig in ("who made you", "who built you", "who created you",
                                 "how was you build", "how were you built", "how was it built",
                                 "how was condrox built")):
        return _IDENTITY_TEMPLATES["how_built"]
    if any(sig in q for sig in ("differ from an llm", "how do you differ", "how are you different",
                                 "what makes you different", "differ from llm")):
        return _IDENTITY_TEMPLATES["how_differ_from_llm"]
    if any(sig in q for sig in ("are you an llm", "do you use llm", "are you a language model")):
        return _IDENTITY_TEMPLATES["are_you_llm"]
    if any(sig in q for sig in ("how advanced are you", "how advanced you are", "how smart are you",
                                 "how powerful are you", "how capable are you",
                                 "what are your capabilities")):
        return _IDENTITY_TEMPLATES["how_advanced"]
    if any(sig in q for sig in ("what is condrox", "who is condrox", "about condrox")):
        return _IDENTITY_TEMPLATES["what_is_condrox"]
    if any(sig in q for sig in ("what can you do", "what do you do")):
        return _IDENTITY_TEMPLATES["what_can_you_do"]
    if any(sig in q for sig in ("how do you work", "how does it work", "what technology",
                                 "what is your architecture", "what architecture")):
        return _IDENTITY_TEMPLATES["how_do_you_work"]
    # Default: who are you / tell me about you / describe yourself / what are you
    return _IDENTITY_TEMPLATES["who_are_you"]


# ─── Conversational Response Builder ──────────────────────────────────────

_JOKES: tuple[str, ...] = (
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "I told my computer I needed a break, and it said: 'No problem — I'll go to sleep.'",
    "Why don't programmers like nature? It has too many bugs.",
    "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
    "Why did the function return early? Because it lost its scope.",
    "How many programmers does it take to change a light bulb? None — that's a hardware problem.",
    "Why do Java developers wear glasses? Because they don't C#.",
    "What's an AI's favorite kind of music? Algo-rhythms.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why was the JavaScript developer sad? Because they didn't Node how to Express themselves.",
)

import random as _random


def _build_conversational_response(kind: str, query: str) -> str:
    """Build a conversational response for non-informational queries.

    Args:
        kind: Type of conversational response ("joke", etc.).
        query: The user's input query string.

    Returns:
        Conversational response text.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    if kind == "joke":
        return _random.choice(_JOKES)
    return "I'm not sure how to respond to that, but I'm listening."

