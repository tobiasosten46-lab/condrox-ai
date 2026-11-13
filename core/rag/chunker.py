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
AST-aware code chunker for Condrox AI RAG system.

Splits code files into semantic units (functions, classes, modules)
for embedding and retrieval.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """
    A single code chunk for RAG indexing.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(content)
    """

    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    content: str
    language: str


class CodeChunker:
    """
    AST-aware code chunker.

    Complexity: T(chunk) ∈ O(n) where n = AST nodes, S(chunk) ∈ O(chunks)
    Invariants: chunks cover entire file
    Edge Cases: syntax error → fallback to line-based chunking
    """

    __slots__ = ()

    def chunk_file(
        self, file_path: str, code: str, language: str
    ) -> Tuple[CodeChunk, ...]:
        """
        Chunk a code file into semantic units.

        Complexity: T(n) ∈ O(n) where n = code length, S(n) ∈ O(chunks)
        Edge Cases: empty file → empty tuple; syntax error → generic chunks
        """
        if not code.strip():
            return ()

        if language == "python":
            return self._chunk_python(file_path, code)
        return self._chunk_generic(file_path, code, language)

    def _chunk_python(
        self, file_path: str, code: str
    ) -> Tuple[CodeChunk, ...]:
        """
        Chunk Python code using AST.

        Complexity: T(n) ∈ O(n) where n = AST nodes, S(n) ∈ O(chunks)
        Edge Cases: syntax error → fallback to generic
        """
        try:
            tree: ast.Module = ast.parse(code)
        except SyntaxError:
            return self._chunk_generic(file_path, code, "python")

        chunks: list[CodeChunk] = []
        lines: list[str] = code.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._make_chunk(
                    file_path, node, lines, "function", node.name, "python"
                ))
            elif isinstance(node, ast.ClassDef):
                chunks.append(self._make_chunk(
                    file_path, node, lines, "class", node.name, "python"
                ))

        if not chunks:
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                chunk_type="module",
                name=file_path,
                content=code,
                language="python",
            ))

        return tuple(chunks)

    def _make_chunk(
        self,
        file_path: str,
        node: ast.AST,
        lines: list[str],
        chunk_type: str,
        name: str,
        language: str,
    ) -> CodeChunk:
        """
        Create a CodeChunk from an AST node.

        Complexity: T(n) ∈ O(n) where n = node lines, S(n) ∈ O(content)
        Edge Cases: node spans entire file → full content
        """
        start: int = getattr(node, "lineno", 1)
        end: int = getattr(node, "end_lineno", len(lines))
        content: str = "\n".join(lines[start - 1:end])

        return CodeChunk(
            file_path=file_path,
            start_line=start,
            end_line=end,
            chunk_type=chunk_type,
            name=name,
            content=content,
            language=language,
        )

    def _chunk_generic(
        self, file_path: str, code: str, language: str
    ) -> Tuple[CodeChunk, ...]:
        """
        Generic line-based chunking for non-Python files.

        Complexity: T(n) ∈ O(n) where n = lines, S(n) ∈ O(chunks)
        Edge Cases: very short file → single chunk
        """
        lines: list[str] = code.splitlines()
        chunk_size: int = 50
        chunks: list[CodeChunk] = []

        for i in range(0, len(lines), chunk_size):
            end: int = min(i + chunk_size, len(lines))
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=i + 1,
                end_line=end,
                chunk_type="block",
                name=f"{file_path}:{i + 1}-{end}",
                content="\n".join(lines[i:end]),
                language=language,
            ))

        return tuple(chunks) if chunks else ()
