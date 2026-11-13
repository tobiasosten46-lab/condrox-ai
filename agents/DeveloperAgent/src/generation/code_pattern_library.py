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
Code Pattern Library — reusable code patterns and templates.

Provides a library of proven code patterns that DeveloperAgent can use
when generating new code. Each pattern includes:
- Template code with placeholders
- Description and use cases
- Required imports
- Complexity characteristics

Complexity:
    get_pattern: T(n) in O(1), S(n) in O(1).
    search: T(n) in O(p) where p = patterns, S(n) in O(results).

Invariants:
    - All patterns are valid Python templates.
    - Patterns are immutable.
    - Search returns patterns sorted by relevance.

Edge Cases:
    - No matching patterns -> empty tuple.
    - Empty query -> all patterns returned.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


# ─── Enums ───────────────────────────────────────────────────────────────────


class PatternCategory(Enum):
    """
    Category of code pattern.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    CREATIONAL = "creational"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CONCURRENCY = "concurrency"
    ERROR_HANDLING = "error_handling"
    DATA_STRUCTURE = "data_structure"
    ALGORITHM = "algorithm"
    API_DESIGN = "api_design"
    TESTING = "testing"
    SECURITY = "security"


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CodePattern:
    """
    Immutable code pattern with template.

    Complexity: T(n) in O(1), S(n) in O(template).
    Invariants: template is valid Python with placeholders.
    Edge Cases: required_imports may be empty.
    """

    pattern_id: str
    name: str
    category: PatternCategory
    description: str
    template: str
    required_imports: Tuple[str, ...]
    use_cases: Tuple[str, ...]
    complexity: str
    tags: Tuple[str, ...]


# ─── Code Pattern Library ────────────────────────────────────────────────────


class CodePatternLibrary:
    """
    Library of reusable code patterns for DeveloperAgent v3.0.

    Contains proven patterns for common programming tasks. Patterns
    can be searched by category, tags, or keywords.

    Complexity:
        get_pattern: T(n) in O(1), S(n) in O(1).
        search: T(n) in O(p) where p = patterns, S(n) in O(results).
        get_by_category: T(n) in O(p), S(n) in O(results).
    Invariants:
        - All patterns are immutable.
        - Pattern IDs are unique.
    Edge Cases:
        - No matching patterns -> empty tuple.
    """

    __slots__ = ("_patterns",)

    _PATTERNS: Tuple[CodePattern, ...] = (
        CodePattern(
            pattern_id="PAT-001",
            name="Frozen Dataclass",
            category=PatternCategory.DATA_STRUCTURE,
            description="Immutable data container with slots for memory efficiency",
            template=(
                "@dataclass(frozen=True, slots=True)\n"
                "class {ClassName}:\n"
                '    """\n'
                "    {description}\n"
                "\n"
                "    Complexity: T(n) in O(1), S(n) in O(1).\n"
                "    Edge Cases: {edge_cases}\n"
                '    """\n'
                "\n"
                "    {field1}: {type1}\n"
                "    {field2}: {type2}\n"
            ),
            required_imports=("from dataclasses import dataclass",),
            use_cases=("Value objects", "Configuration", "DTOs", "Events"),
            complexity="T(n) in O(1), S(n) in O(fields)",
            tags=("dataclass", "frozen", "immutable", "slots", "value-object"),
        ),
        CodePattern(
            pattern_id="PAT-002",
            name="Strategy Pattern",
            category=PatternCategory.BEHAVIORAL,
            description="Interchangeable algorithms behind a common interface",
            template=(
                "from abc import ABC, abstractmethod\n"
                "\n"
                "class {StrategyName}(ABC):\n"
                '    """Strategy interface."""\n'
                "\n"
                "    @abstractmethod\n"
                "    def execute(self, data: {DataType}) -> {ResultType}:\n"
                '        """Execute strategy."""\n'
                "        ...\n"
                "\n"
                "class {ConcreteName}({StrategyName}):\n"
                '    """Concrete strategy implementation."""\n'
                "\n"
                "    def execute(self, data: {DataType}) -> {ResultType}:\n"
                "        return {implementation}\n"
            ),
            required_imports=("from abc import ABC, abstractmethod",),
            use_cases=("Algorithm selection", "Payment methods", "Sorting strategies"),
            complexity="T(n) in O(strategy), S(n) in O(1)",
            tags=("strategy", "behavioral", "interface", "polymorphism"),
        ),
        CodePattern(
            pattern_id="PAT-003",
            name="Singleton with __slots__",
            category=PatternCategory.CREATIONAL,
            description="Memory-efficient singleton pattern",
            template=(
                "class {ClassName}:\n"
                '    """Singleton with __slots__."""\n'
                "\n"
                "    __slots__ = (\"_instance\",)\n"
                "    _instance: \"{ClassName}\" = None\n"
                "\n"
                "    def __new__(cls) -> \"{ClassName}\":\n"
                "        if cls._instance is None:\n"
                "            cls._instance = super().__new__(cls)\n"
                "        return cls._instance\n"
            ),
            required_imports=(),
            use_cases=("Database connections", "Loggers", "Configuration managers"),
            complexity="T(n) in O(1), S(n) in O(1)",
            tags=("singleton", "creational", "slots", "instance"),
        ),
        CodePattern(
            pattern_id="PAT-004",
            name="Retry with Exponential Backoff",
            category=PatternCategory.ERROR_HANDLING,
            description="Retry failed operations with increasing delay",
            template=(
                "import asyncio\n"
                "import time\n"
                "\n"
                "async def {func_name}(\n"
                "    operation: Callable,\n"
                "    max_retries: int = 3,\n"
                "    base_delay: float = 1.0,\n"
                ") -> Any:\n"
                '    """Retry with exponential backoff."""\n'
                "    for attempt in range(max_retries):\n"
                "        try:\n"
                "            return await operation()\n"
                "        except Exception:\n"
                "            if attempt == max_retries - 1:\n"
                "                raise\n"
                "            delay = base_delay * (2 ** attempt)\n"
                "            await asyncio.sleep(delay)\n"
            ),
            required_imports=("import asyncio", "from typing import Any, Callable"),
            use_cases=("API calls", "Network operations", "Database queries"),
            complexity="T(n) in O(retries * delay), S(n) in O(1)",
            tags=("retry", "backoff", "error-handling", "async", "resilience"),
        ),
        CodePattern(
            pattern_id="PAT-005",
            name="Circuit Breaker",
            category=PatternCategory.ERROR_HANDLING,
            description="Prevent cascading failures with circuit breaker pattern",
            template=(
                "class {CircuitName}:\n"
                '    """Circuit breaker for fault tolerance."""\n'
                "\n"
                "    __slots__ = (\"_state\", \"_failures\", \"_threshold\", \"_timeout\")\n"
                "\n"
                "    def __init__(self, threshold: int = 5, timeout: float = 60.0) -> None:\n"
                "        self._state = \"closed\"\n"
                "        self._failures = 0\n"
                "        self._threshold = threshold\n"
                "        self._timeout = timeout\n"
                "\n"
                "    def record_failure(self) -> None:\n"
                "        self._failures += 1\n"
                "        if self._failures >= self._threshold:\n"
                "            self._state = \"open\"\n"
                "\n"
                "    def record_success(self) -> None:\n"
                "        self._failures = 0\n"
                "        self._state = \"closed\"\n"
                "\n"
                "    @property\n"
                "    def is_open(self) -> bool:\n"
                "        return self._state == \"open\"\n"
            ),
            required_imports=(),
            use_cases=("Microservices", "External API calls", "Database connections"),
            complexity="T(n) in O(1), S(n) in O(1)",
            tags=("circuit-breaker", "fault-tolerance", "resilience", "error-handling"),
        ),
        CodePattern(
            pattern_id="PAT-006",
            name="Observer Pattern",
            category=PatternCategory.BEHAVIORAL,
            description="Publish-subscribe pattern for event notification",
            template=(
                "from typing import Callable, Tuple\n"
                "\n"
                "class {SubjectName}:\n"
                '    """Observable subject."""\n'
                "\n"
                "    __slots__ = (\"_observers\",)\n"
                "\n"
                "    def __init__(self) -> None:\n"
                "        self._observers: list[Callable] = []\n"
                "\n"
                "    def subscribe(self, observer: Callable) -> None:\n"
                "        self._observers.append(observer)\n"
                "\n"
                "    def unsubscribe(self, observer: Callable) -> None:\n"
                "        self._observers.remove(observer)\n"
                "\n"
                "    def notify(self, event: Any) -> None:\n"
                "        for observer in self._observers:\n"
                "            observer(event)\n"
            ),
            required_imports=("from typing import Any, Callable",),
            use_cases=("Event systems", "GUI updates", "Message buses"),
            complexity="T(n) in O(o) where o = observers, S(n) in O(o)",
            tags=("observer", "pubsub", "events", "behavioral"),
        ),
        CodePattern(
            pattern_id="PAT-007",
            name="Factory Method",
            category=PatternCategory.CREATIONAL,
            description="Create objects without specifying exact class",
            template=(
                "from typing import Dict, Type\n"
                "\n"
                "class {FactoryName}:\n"
                '    """Factory for creating {ProductType} instances."""\n'
                "\n"
                "    __slots__ = (\"_registry\",)\n"
                "\n"
                "    def __init__(self) -> None:\n"
                "        self._registry: Dict[str, Type[{ProductType}]] = {}\n"
                "\n"
                "    def register(self, key: str, cls: Type[{ProductType}]) -> None:\n"
                "        self._registry[key] = cls\n"
                "\n"
                "    def create(self, key: str, **kwargs: Any) -> {ProductType}:\n"
                "        cls = self._registry.get(key)\n"
                "        if cls is None:\n"
                "            raise ValueError(f\"Unknown type: {key}\")\n"
                "        return cls(**kwargs)\n"
            ),
            required_imports=("from typing import Any, Dict, Type",),
            use_cases=("Plugin systems", "Document parsers", "Protocol handlers"),
            complexity="T(n) in O(1), S(n) in O(registry)",
            tags=("factory", "creational", "registry", "polymorphism"),
        ),
        CodePattern(
            pattern_id="PAT-008",
            name="Async Context Manager",
            category=PatternCategory.ERROR_HANDLING,
            description="Resource management with async context manager",
            template=(
                "from contextlib import asynccontextmanager\n"
                "\n"
                "@asynccontextmanager\n"
                "async def {resource_name}(config: {ConfigType}):\n"
                '    """Async context manager for {resource_name}."""\n'
                "    resource = await {acquire}\n"
                "    try:\n"
                "        yield resource\n"
                "    finally:\n"
                "        await {release}(resource)\n"
            ),
            required_imports=("from contextlib import asynccontextmanager",),
            use_cases=("Database connections", "File handles", "Network sessions"),
            complexity="T(n) in O(acquire + release), S(n) in O(resource)",
            tags=("async", "context-manager", "resource-management", "cleanup"),
        ),
        CodePattern(
            pattern_id="PAT-009",
            name="LRU Cache",
            category=PatternCategory.DATA_STRUCTURE,
            description="Least Recently Used cache with OrderedDict",
            template=(
                "from collections import OrderedDict\n"
                "\n"
                "class {CacheName}:\n"
                '    """LRU cache with configurable size."""\n'
                "\n"
                "    __slots__ = (\"_cache\", \"_max_size\")\n"
                "\n"
                "    def __init__(self, max_size: int = 128) -> None:\n"
                "        self._cache: OrderedDict = OrderedDict()\n"
                "        self._max_size = max_size\n"
                "\n"
                "    def get(self, key: {KeyType}) -> Any:\n"
                "        if key in self._cache:\n"
                "            self._cache.move_to_end(key)\n"
                "            return self._cache[key]\n"
                "        return None\n"
                "\n"
                "    def put(self, key: {KeyType}, value: Any) -> None:\n"
                "        self._cache[key] = value\n"
                "        self._cache.move_to_end(key)\n"
                "        while len(self._cache) > self._max_size:\n"
                "            self._cache.popitem(last=False)\n"
            ),
            required_imports=("from collections import OrderedDict", "from typing import Any"),
            use_cases=("Memoization", "HTTP response caching", "Session storage"),
            complexity="T(n) in O(1), S(n) in O(max_size)",
            tags=("cache", "lru", "ordereddict", "eviction"),
        ),
        CodePattern(
            pattern_id="PAT-010",
            name="Type-Safe Builder",
            category=PatternCategory.CREATIONAL,
            description="Fluent builder with type safety for complex objects",
            template=(
                "@dataclass(frozen=True, slots=True)\n"
                "class {ProductName}:\n"
                '    """Immutable product."""\n'
                "    {fields}\n"
                "\n"
                "class {BuilderName}:\n"
                '    """Type-safe builder for {ProductName}."""\n'
                "\n"
                "    __slots__ = (\"_data\",)\n"
                "\n"
                "    def __init__(self) -> None:\n"
                "        self._data: dict = {}\n"
                "\n"
                "    def with_{field1}(self, value: {type1}) -> \"{BuilderName}\":\n"
                "        self._data[\"{field1}\"] = value\n"
                "        return self\n"
                "\n"
                "    def build(self) -> {ProductName}:\n"
                "        return {ProductName}(**self._data)\n"
            ),
            required_imports=("from dataclasses import dataclass",),
            use_cases=("Complex object construction", "Configuration builders", "Query builders"),
            complexity="T(n) in O(1), S(n) in O(fields)",
            tags=("builder", "fluent", "creational", "immutable", "type-safe"),
        ),
    )

    def __init__(self) -> None:
        """
        Initialize CodePatternLibrary.

        Complexity: T(n) in O(1), S(n) in O(patterns).
        Edge Cases: none.
        """
        self._patterns: Tuple[CodePattern, ...] = self._PATTERNS

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def pattern_count(self) -> int:
        """
        Number of patterns in the library.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty library -> 0.
        """
        return len(self._patterns)

    def get_pattern(self, pattern_id: str) -> CodePattern:
        """
        Get a pattern by its ID.

        Complexity: T(n) in O(p) where p = patterns, S(n) in O(1).
        Edge Cases: not found -> None.
        """
        for p in self._patterns:
            if p.pattern_id == pattern_id:
                return p
        return None

    def get_by_category(
        self, category: PatternCategory
    ) -> Tuple[CodePattern, ...]:
        """
        Get all patterns in a category.

        Complexity: T(n) in O(p) where p = patterns, S(n) in O(results).
        Edge Cases: no patterns in category -> empty tuple.
        """
        return tuple(p for p in self._patterns if p.category == category)

    def search(self, query: str) -> Tuple[CodePattern, ...]:
        """
        Search patterns by keyword (matches name, description, tags).

        Complexity: T(n) in O(p * q) where p = patterns, q = query terms,
                    S(n) in O(results).
        Edge Cases: empty query -> all patterns.
        """
        if not query.strip():
            return self._patterns

        query_lower: str = query.lower()
        scored: List[Tuple[float, CodePattern]] = []

        for pattern in self._patterns:
            score: float = 0.0
            if query_lower in pattern.name.lower():
                score += 3.0
            if query_lower in pattern.description.lower():
                score += 2.0
            for tag in pattern.tags:
                if query_lower in tag.lower():
                    score += 1.5
            for use_case in pattern.use_cases:
                if query_lower in use_case.lower():
                    score += 1.0
            if score > 0:
                scored.append((score, pattern))

        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(p for _, p in scored)

    def get_all_categories(self) -> Tuple[PatternCategory, ...]:
        """
        Get all categories that have patterns.

        Complexity: T(n) in O(p) where p = patterns, S(n) in O(categories).
        Edge Cases: empty library -> empty tuple.
        """
        seen: set = set()
        for p in self._patterns:
            seen.add(p.category)
        return tuple(sorted(seen, key=lambda c: c.value))

    def get_all_tags(self) -> Tuple[str, ...]:
        """
        Get all unique tags across all patterns.

        Complexity: T(n) in O(p * t) where p = patterns, t = tags,
                    S(n) in O(tags).
        Edge Cases: empty library -> empty tuple.
        """
        tags: set = set()
        for p in self._patterns:
            for tag in p.tags:
                tags.add(tag)
        return tuple(sorted(tags))
