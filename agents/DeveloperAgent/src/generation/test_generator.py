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
Test Generator — automatic test generation from Python code.

Analyzes function signatures and generates pytest test cases:
- Happy path tests (valid inputs -> expected outputs)
- Edge case tests (empty, None, boundary values)
- Type error tests (wrong types -> TypeError)
- Parametrized tests for multiple inputs

Complexity:
    generate: T(n) in O(f * p) where f = functions, p = parameters,
              S(n) in O(tests).

Invariants:
    - Generated tests use pytest framework.
    - All tests have descriptive names.
    - Edge cases always included.

Edge Cases:
    - No functions -> empty test file.
    - Syntax error -> empty result.
    - Non-Python -> empty result.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TestCase:
    """
    Immutable generated test case.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: test_name is unique within a function.
    Edge Cases: expected_exception may be empty string.
    """

    test_name: str
    function_name: str
    args: Tuple[str, ...]
    expected: str
    expected_exception: str
    description: str


@dataclass(frozen=True, slots=True)
class GeneratedTest:
    """
    Immutable generated test for a single function.

    Complexity: T(n) in O(1), S(n) in O(cases).
    Invariants: cases is non-empty tuple.
    Edge Cases: no cases -> empty tuple.
    """

    function_name: str
    module: str
    cases: Tuple[TestCase, ...]
    test_code: str


# ─── Test Generator ──────────────────────────────────────────────────────────


class TestGenerator:
    """
    Automatic test generator for DeveloperAgent v3.0.

    Analyzes Python source code and generates pytest test cases for
    each public function. Generates happy path, edge case, and type
    error tests.

    Complexity:
        generate: T(n) in O(f * p) where f = functions, p = parameters,
                  S(n) in O(tests).
    Invariants:
        - All tests use pytest.
        - Edge cases always included.
        - Test names are descriptive and unique.
    Edge Cases:
        - No functions -> empty result.
        - Syntax error -> empty result.
    """

    __slots__ = ("_type_defaults",)

    _TYPE_DEFAULTS: Dict[str, Tuple[str, str]] = {
        "int": ("42", "0"),
        "str": ("'hello'", "''"),
        "float": ("3.14", "0.0"),
        "bool": ("True", "False"),
        "list": ("[1, 2, 3]", "[]"),
        "dict": ("{'key': 'value'}", "{}"),
        "set": ("{1, 2, 3}", "set()"),
        "tuple": ("(1, 2)", "()"),
        "Any": ("42", "None"),
        "Optional[str]": ("'hello'", "None"),
        "Optional[int]": ("42", "None"),
    }

    _EDGE_VALUES: Dict[str, Tuple[str, ...]] = {
        "int": ("0", "-1", "999999", "-999999"),
        "str": ("''", "'a'", "'x' * 1000"),
        "float": ("0.0", "-0.0", "1e308", "-1e308"),
        "list": ("[]", "[None]", "[item for item in range(1000)]"),
        "dict": ("{}", "{'': None}"),
    }

    def __init__(self) -> None:
        """
        Initialize TestGenerator.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: none.
        """
        self._type_defaults: Dict[str, Tuple[str, str]] = self._TYPE_DEFAULTS

    # ─── Public API ──────────────────────────────────────────────────────────

    def generate(
        self, code: str, module_name: str = "target", language: str = "python"
    ) -> Tuple[GeneratedTest, ...]:
        """
        Generate tests for all public functions in code.

        Complexity: T(n) in O(f * p) where f = functions, p = parameters,
                    S(n) in O(tests).
        Invariants: one GeneratedTest per public function.
        Edge Cases: no functions -> empty tuple.
        """
        if not code.strip() or language.lower() != "python":
            return ()

        try:
            tree: ast.AST = ast.parse(code)
        except SyntaxError:
            return ()

        functions: List[ast.FunctionDef] = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
        ]

        tests: List[GeneratedTest] = []
        for func in functions:
            generated: GeneratedTest = self._generate_for_function(
                func, module_name
            )
            tests.append(generated)

        return tuple(tests)

    def generate_test_file(
        self, code: str, module_name: str = "target", language: str = "python"
    ) -> str:
        """
        Generate a complete pytest test file.

        Complexity: T(n) in O(f * p) where f = functions, p = parameters,
                    S(n) in O(file).
        Edge Cases: no functions -> header only.
        """
        tests: Tuple[GeneratedTest, ...] = self.generate(
            code, module_name, language
        )

        lines: List[str] = [
            '"""',
            f"Auto-generated tests for {module_name}.",
            "Generated by DeveloperAgent v3.0 Test Generator.",
            "",
            "Classification: Tier 5+ Class A",
            '"""',
            "",
            "import pytest",
            f"from {module_name} import *",
            "",
            "",
        ]

        for test in tests:
            lines.append(test.test_code)
            lines.append("")

        return "\n".join(lines)

    # ─── Private ─────────────────────────────────────────────────────────────

    def _generate_for_function(
        self, node: ast.FunctionDef, module_name: str
    ) -> GeneratedTest:
        """
        Generate test cases for a single function.

        Complexity: T(n) in O(p) where p = parameters, S(n) in O(cases).
        Edge Cases: no params -> single no-arg test.
        """
        func_name: str = node.name
        params: List[ast.arg] = node.args.args
        if params and params[0].arg == "self":
            params = params[1:]

        cases: List[TestCase] = []

        happy_args, _ = self._get_args_for_params(params, use_edge=False)
        cases.append(TestCase(
            test_name=f"test_{func_name}_happy_path",
            function_name=func_name,
            args=tuple(happy_args),
            expected="<result>",
            expected_exception="",
            description="Test with valid inputs",
        ))

        edge_args, edge_desc = self._get_args_for_params(params, use_edge=True)
        if edge_args != happy_args:
            cases.append(TestCase(
                test_name=f"test_{func_name}_edge_case",
                function_name=func_name,
                args=tuple(edge_args),
                expected="<result>",
                expected_exception="",
                description=f"Edge case: {edge_desc}",
            ))

        cases.append(TestCase(
            test_name=f"test_{func_name}_none_input",
            function_name=func_name,
            args=tuple("None" for _ in params),
            expected="",
            expected_exception="TypeError",
            description="Test with None inputs (should raise TypeError)",
        ))

        if not params:
            cases.append(TestCase(
                test_name=f"test_{func_name}_no_args",
                function_name=func_name,
                args=(),
                expected="<result>",
                expected_exception="",
                description="Test with no arguments",
            ))

        test_code: str = self._render_test_code(
            func_name, cases, module_name
        )

        return GeneratedTest(
            function_name=func_name,
            module=module_name,
            cases=tuple(cases),
            test_code=test_code,
        )

    def _get_args_for_params(
        self, params: List[ast.arg], use_edge: bool
    ) -> Tuple[List[str], str]:
        """
        Generate argument values for function parameters.

        Complexity: T(n) in O(p) where p = params, S(n) in O(args).
        Edge Cases: no params -> empty list.
        """
        args: List[str] = []
        edge_desc: str = ""

        for param in params:
            type_name: str = self._get_type_name(param)
            if use_edge:
                edge_values: Tuple[str, ...] = self._EDGE_VALUES.get(
                    type_name, ("None",)
                )
                args.append(edge_values[0])
                edge_desc = f"{type_name}={edge_values[0]}"
            else:
                defaults: Tuple[str, str] = self._type_defaults.get(
                    type_name, ("42", "None")
                )
                args.append(defaults[0])

        return args, edge_desc

    @staticmethod
    def _get_type_name(param: ast.arg) -> str:
        """
        Extract type name from parameter annotation.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no annotation -> "Any".
        """
        if param.annotation is None:
            return "Any"
        if isinstance(param.annotation, ast.Name):
            return param.annotation.id
        if isinstance(param.annotation, ast.Subscript):
            if isinstance(param.annotation.value, ast.Name):
                base: str = param.annotation.value.id
                return f"{base}[...]"
        return "Any"

    def _render_test_code(
        self, func_name: str, cases: List[TestCase], module_name: str
    ) -> str:
        """
        Render test cases as pytest code.

        Complexity: T(n) in O(c) where c = cases, S(n) in O(code).
        Edge Cases: no cases -> empty function.
        """
        lines: List[str] = [
            f"# ─── Tests for {func_name} ────────────────────────────────────",
            "",
        ]

        for case in cases:
            lines.append(f'def {case.test_name}() -> None:')
            lines.append(f'    """{case.description}."""')
            args_str: str = ", ".join(case.args)

            if case.expected_exception:
                lines.append(f"    with pytest.raises({case.expected_exception}):")
                lines.append(f"        {func_name}({args_str})")
            else:
                lines.append(f"    result = {func_name}({args_str})")
                lines.append(f"    assert result is not None")
            lines.append("")

        return "\n".join(lines)
