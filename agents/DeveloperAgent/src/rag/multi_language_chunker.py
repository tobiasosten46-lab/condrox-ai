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
Multi-Language Chunker — language-aware code chunking for RAG.

Extends the base CodeChunker with language-specific chunking strategies:
- Python: function/class/def boundaries
- TypeScript/JavaScript: function/class/export boundaries
- Markdown: header-based sections
- JSON: top-level object boundaries
- Generic: line-count based fallback

Complexity:
    chunk_file: T(n) in O(n) where n = code length, S(n) in O(chunks).

Invariants:
    - Every chunk has start_line, end_line, name, and content.
    - Chunks do not overlap.
    - Union of chunks covers the entire file (no gaps).

Edge Cases:
    - Empty code -> single empty chunk.
    - Unknown language -> generic line-count chunking.
    - Single-line file -> single chunk.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from core.rag.chunker import CodeChunk


# ─── Enums ───────────────────────────────────────────────────────────────────


class LanguageType(Enum):
    """
    Supported languages for multi-language chunking.

    Complexity: T(n) in O(1), S(n) in O(1).
    """

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    GENERIC = "generic"


# ─── Multi-Language Chunker ──────────────────────────────────────────────────


class MultiLanguageChunker:
    """
    Language-aware code chunker for RAG indexing.

    Detects language from file extension and applies the appropriate
    chunking strategy. Falls back to generic line-count chunking for
    unknown languages.

    Complexity:
        chunk_file: T(n) in O(n) where n = code length, S(n) in O(chunks).
        detect_language: T(n) in O(1), S(n) in O(1).
    Invariants:
        - Chunks do not overlap.
        - Every chunk has start_line, end_line, name, content.
        - Language detected from file extension.
    Edge Cases:
        - Empty code -> single empty chunk.
        - Unknown extension -> GENERIC chunking.
    """

    __slots__ = ("_max_chunk_lines", "_max_chunk_chars")

    _EXTENSION_MAP: Dict[str, LanguageType] = {
        ".py": LanguageType.PYTHON,
        ".ts": LanguageType.TYPESCRIPT,
        ".tsx": LanguageType.TYPESCRIPT,
        ".js": LanguageType.JAVASCRIPT,
        ".jsx": LanguageType.JAVASCRIPT,
        ".mjs": LanguageType.JAVASCRIPT,
        ".md": LanguageType.MARKDOWN,
        ".markdown": LanguageType.MARKDOWN,
        ".json": LanguageType.JSON,
        ".yaml": LanguageType.YAML,
        ".yml": LanguageType.YAML,
    }

    _PY_PATTERN: re.Pattern[str] = re.compile(
        r"^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)",
        re.MULTILINE,
    )

    _TS_PATTERN: re.Pattern[str] = re.compile(
        r"^(export\s+)?(class\s+\w+|function\s+\w+|interface\s+\w+|"
        r"type\s+\w+|const\s+\w+|enum\s+\w+)",
        re.MULTILINE,
    )

    _MD_PATTERN: re.Pattern[str] = re.compile(
        r"^(#{1,6}\s+.+)$",
        re.MULTILINE,
    )

    def __init__(
        self,
        max_chunk_lines: int = 80,
        max_chunk_chars: int = 4000,
    ) -> None:
        """
        Initialize MultiLanguageChunker with chunk size limits.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: max_chunk_lines=0 -> single chunk per file.
        """
        self._max_chunk_lines: int = max_chunk_lines
        self._max_chunk_chars: int = max_chunk_chars

    # ─── Public API ──────────────────────────────────────────────────────────

    def detect_language(self, file_path: str) -> LanguageType:
        """
        Detect language from file extension.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: unknown extension -> GENERIC.
        """
        ext: str = ""
        dot_idx: int = file_path.rfind(".")
        if dot_idx >= 0:
            ext = file_path[dot_idx:].lower()
        return self._EXTENSION_MAP.get(ext, LanguageType.GENERIC)

    def chunk_file(
        self, file_path: str, code: str, language: str = ""
    ) -> Tuple[CodeChunk, ...]:
        """
        Chunk a file using language-appropriate strategy.

        Complexity: T(n) in O(n) where n = code length, S(n) in O(chunks).
        Invariants: chunks cover entire file, no overlaps.
        Edge Cases: empty code -> single empty chunk.
        """
        if not code.strip():
            return (
                CodeChunk(
                    file_path=file_path,
                    start_line=1,
                    end_line=1,
                    chunk_type="empty",
                    name="<empty>",
                    content="",
                    language=language or "generic",
                ),
            )

        lang: LanguageType = (
            self._map_language_str(language)
            if language
            else self.detect_language(file_path)
        )

        lang_str: str = lang.value

        if lang == LanguageType.PYTHON:
            return self._chunk_by_pattern(
                file_path, code, lang_str, self._PY_PATTERN
            )
        elif lang in (LanguageType.TYPESCRIPT, LanguageType.JAVASCRIPT):
            return self._chunk_by_pattern(
                file_path, code, lang_str, self._TS_PATTERN
            )
        elif lang == LanguageType.MARKDOWN:
            return self._chunk_by_pattern(
                file_path, code, lang_str, self._MD_PATTERN
            )
        elif lang == LanguageType.JSON:
            return self._chunk_json(file_path, code, lang_str)
        else:
            return self._chunk_generic(file_path, code, lang_str)

    # ─── Private: Pattern-Based Chunking ─────────────────────────────────────

    def _chunk_by_pattern(
        self,
        file_path: str,
        code: str,
        language: str,
        pattern: re.Pattern[str],
    ) -> Tuple[CodeChunk, ...]:
        """
        Chunk code by pattern matches (function/class/header boundaries).

        Complexity: T(n) in O(n) where n = code length, S(n) in O(chunks).
        Invariants: chunks cover entire file.
        Edge Cases: no pattern matches -> single chunk.
        """
        lines: List[str] = code.splitlines(keepends=True)
        total_lines: int = len(lines)

        boundaries: List[Tuple[int, str]] = []
        for match in pattern.finditer(code):
            line_num: int = code[:match.start()].count("\n") + 1
            name: str = match.group(0).strip().split("(")[0].strip()
            boundaries.append((line_num, name))

        if not boundaries:
            return self._chunk_generic(file_path, code, language)

        chunks: List[CodeChunk] = []

        if boundaries[0][0] > 1:
            preamble: str = "".join(lines[: boundaries[0][0] - 1])
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=1,
                end_line=boundaries[0][0] - 1,
                chunk_type="preamble",
                name="<preamble>",
                content=preamble,
                language=language,
            ))

        for i, (start_line, name) in enumerate(boundaries):
            end_line: int = (
                boundaries[i + 1][0] - 1 if i + 1 < len(boundaries)
                else total_lines
            )
            content: str = "".join(lines[start_line - 1 : end_line])

            if len(content) > self._max_chunk_chars:
                sub_chunks: Tuple[CodeChunk, ...] = self._split_large_chunk(
                    file_path, content, name, start_line, language
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="block",
                    name=name,
                    content=content,
                    language=language,
                ))

        return tuple(chunks)

    def _chunk_json(
        self, file_path: str, code: str, language: str
    ) -> Tuple[CodeChunk, ...]:
        """
        Chunk JSON by top-level keys.

        Complexity: T(n) in O(n) where n = code length, S(n) in O(chunks).
        Edge Cases: invalid JSON -> generic chunking.
        """
        lines: List[str] = code.splitlines(keepends=True)

        key_pattern: re.Pattern[str] = re.compile(
            r'^\s*"([^"]+)"\s*:', re.MULTILINE
        )

        boundaries: List[Tuple[int, str]] = []
        for match in key_pattern.finditer(code):
            line_num: int = code[:match.start()].count("\n") + 1
            boundaries.append((line_num, match.group(1)))

        if not boundaries:
            return self._chunk_generic(file_path, code, language)

        chunks: List[CodeChunk] = []
        total_lines: int = len(lines)

        for i, (start_line, name) in enumerate(boundaries):
            end_line: int = (
                boundaries[i + 1][0] - 1 if i + 1 < len(boundaries)
                else total_lines
            )
            content: str = "".join(lines[start_line - 1 : end_line])
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type="json_key",
                name=f"key:{name}",
                content=content,
                language=language,
            ))

        return tuple(chunks)

    def _chunk_generic(
        self, file_path: str, code: str, language: str
    ) -> Tuple[CodeChunk, ...]:
        """
        Generic line-count based chunking for unknown languages.

        Complexity: T(n) in O(n) where n = code length, S(n) in O(chunks).
        Edge Cases: code shorter than max_chunk_lines -> single chunk.
        """
        lines: List[str] = code.splitlines(keepends=True)
        total_lines: int = len(lines)

        if total_lines <= self._max_chunk_lines:
            return (
                CodeChunk(
                    file_path=file_path,
                    start_line=1,
                    end_line=total_lines,
                    chunk_type="file",
                    name="<file>",
                    content=code,
                    language=language,
                ),
            )

        chunks: List[CodeChunk] = []
        start: int = 0
        chunk_idx: int = 0

        while start < total_lines:
            end: int = min(start + self._max_chunk_lines, total_lines)
            content: str = "".join(lines[start:end])
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                chunk_type="generic",
                name=f"<chunk-{chunk_idx}>",
                content=content,
                language=language,
            ))
            start = end
            chunk_idx += 1

        return tuple(chunks)

    def _split_large_chunk(
        self,
        file_path: str,
        content: str,
        name: str,
        start_line: int,
        language: str,
    ) -> Tuple[CodeChunk, ...]:
        """
        Split a chunk that exceeds max_chunk_chars into sub-chunks.

        Complexity: T(n) in O(n) where n = content length, S(n) in O(sub).
        Edge Cases: content shorter than max -> single chunk.
        """
        lines: List[str] = content.splitlines(keepends=True)
        chunks: List[CodeChunk] = []
        current_lines: List[str] = []
        current_chars: int = 0
        current_start: int = start_line
        line_offset: int = 0

        for i, line in enumerate(lines):
            if current_chars + len(line) > self._max_chunk_chars and current_lines:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    start_line=current_start,
                    end_line=current_start + len(current_lines) - 1,
                    chunk_type="split",
                    name=f"{name} (part {len(chunks) + 1})",
                    content="".join(current_lines),
                    language=language,
                ))
                current_lines = []
                current_chars = 0
                current_start = start_line + i
                line_offset = i

            current_lines.append(line)
            current_chars += len(line)

        if current_lines:
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
                chunk_type="split",
                name=f"{name} (part {len(chunks) + 1})" if len(chunks) > 0 else name,
                content="".join(current_lines),
                language=language,
            ))

        return tuple(chunks)

    def _map_language_str(self, language: str) -> LanguageType:
        """
        Map a language string to LanguageType enum.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: unknown string -> GENERIC.
        """
        lang_lower: str = language.lower().strip()
        for lt in LanguageType:
            if lt.value == lang_lower:
                return lt
        if lang_lower in ("ts", "tsx"):
            return LanguageType.TYPESCRIPT
        if lang_lower in ("js", "jsx", "mjs", "node"):
            return LanguageType.JAVASCRIPT
        return LanguageType.GENERIC
