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
Condrox Knowledge Ingestion Engine.

Downloads, cleans, deduplicates, and chunks intelligence datasets into a
structured knowledge store. The engine is designed to stream from Hugging
Face `datasets` so it can handle multi-hundred-gigabyte corpora without
loading the entire dataset into memory.

Configuration: config/knowledge_sources.json
Output: <knowledge_root>/<category>/<name>_<source>_v1.<ext>

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Immutable chunk with content and provenance."""
    content: str
    source: str
    category: str
    chunk_id: str
    hash: str


@dataclass
class IngestionStats:
    """Mutable statistics for an ingestion run."""
    examples_read: int = 0
    chunks_written: int = 0
    bytes_written: int = 0
    duplicates_removed: int = 0
    errors: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "examples_read": self.examples_read,
            "chunks_written": self.chunks_written,
            "bytes_written": self.bytes_written,
            "duplicates_removed": self.duplicates_removed,
            "errors": self.errors[-20:],  # keep last 20
        }


class TextCleaner:
    """Real text normalization and cleaning."""

    __slots__ = ("_html_tag", "_control")

    def __init__(self) -> None:
        self._html_tag = re.compile(r"<[^>]+>")
        self._control = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")

    def clean(self, text: str) -> str:
        """Normalize to UTF-8, remove HTML, scripts, control chars, and noise."""
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._html_tag.sub("", text)
        text = self._control.sub("", text)
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n +", "\n", text)
        text = re.sub(r" +\n", "\n", text)
        return text.strip()


class Chunker:
    """Streaming chunker that produces 4–8 KB UTF-8 blocks."""

    __slots__ = ("min_bytes", "max_bytes", "buffer", "buffer_len", "index", "source", "category")

    def __init__(self, min_bytes: int = 4096, max_bytes: int = 8192, source: str = "", category: str = "") -> None:
        if not (0 < min_bytes < max_bytes):
            raise ValueError("min_bytes must be positive and less than max_bytes")
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.buffer: List[str] = []
        self.buffer_len = 0
        self.index = 0
        self.source = source
        self.category = category

    def feed(self, text: str) -> List[ChunkRecord]:
        """Feed text into the chunker and emit any completed chunks."""
        if not text:
            return []

        emitted: List[ChunkRecord] = []
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_bytes = len(paragraph.encode("utf-8"))

            if self.buffer_len + para_bytes >= self.max_bytes:
                # Emit current buffer before adding the new paragraph
                if self.buffer:
                    content = "\n".join(self.buffer)
                    if self._size(content) >= self.min_bytes:
                        emitted.append(self._record(content))
                        self.buffer = []
                        self.buffer_len = 0

            self.buffer.append(paragraph)
            self.buffer_len += para_bytes + 1

            # If we hit the upper bound after adding, emit now
            if self.buffer_len >= self.max_bytes:
                content = "\n".join(self.buffer)
                if self._size(content) >= self.min_bytes:
                    emitted.append(self._record(content))
                    self.buffer = []
                    self.buffer_len = 0

        return emitted

    def flush(self) -> List[ChunkRecord]:
        """Emit any remaining buffered text as the final chunk(s)."""
        if not self.buffer:
            return []
        content = "\n".join(self.buffer)
        self.buffer = []
        self.buffer_len = 0
        if content.strip():
            return [self._record(content)]
        return []

    def _size(self, content: str) -> int:
        return len(content.encode("utf-8"))

    def _record(self, content: str) -> ChunkRecord:
        chunk_id = f"{self.source}_{self.index:08d}"
        self.index += 1
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return ChunkRecord(content=content, source=self.source, category=self.category, chunk_id=chunk_id, hash=digest)


class KnowledgeIngestionEngine:
    """
    Real knowledge ingestion engine for Condrox.

    Streams datasets from Hugging Face, cleans, chunks, deduplicates, and
    writes to a structured folder tree.
    """

    __slots__ = (
        "config",
        "cleaner",
        "knowledge_root",
        "max_samples",
        "seen_hashes",
    )

    def __init__(self, config_path: Path, knowledge_root: Optional[Path] = None) -> None:
        self.config = self._load_config(config_path)
        self.knowledge_root = knowledge_root or Path(self.config["knowledge_root"])
        self.max_samples = int(self.config.get("max_samples", 0))
        self.cleaner = TextCleaner()
        self.seen_hashes: Set[str] = set()

    def _load_config(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _extract_text(self, example: Dict[str, Any], source_cfg: Dict[str, Any]) -> str:
        """Extract text from an example using configured fields."""
        parts: List[str] = []
        for field in source_cfg.get("text_fields", ["text"]):
            value = example.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    parts.append("\n".join(value))
                elif value and isinstance(value[0], dict):
                    # e.g. ARC choices
                    for choice in value:
                        for sub in ("text", "label", "value"):
                            if sub in choice and isinstance(choice[sub], str):
                                parts.append(choice[sub])
            elif isinstance(value, dict):
                key = source_cfg.get("translation_key", "en")
                if key in value and isinstance(value[key], str):
                    parts.append(value[key])
        return "\n\n".join(p for p in parts if p.strip())

    def _load_dataset(self, source_cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Load a dataset from Hugging Face with streaming."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets library not installed: pip install datasets") from exc

        hf_name = source_cfg["hf_name"]
        config_name = source_cfg.get("subset") or source_cfg.get("version")
        split = source_cfg.get("split", "train")

        kwargs: Dict[str, Any] = {"streaming": True, "trust_remote_code": True}
        if config_name:
            kwargs["name"] = config_name

        try:
            ds = load_dataset(hf_name, split=split, **kwargs)
        except Exception as exc:
            error_text = str(exc).lower()
            # If config is missing, try to auto-detect an English config
            if "config name" in error_text or "builderconfig" in error_text:
                try:
                    from datasets import get_dataset_config_names
                    configs = get_dataset_config_names(hf_name, trust_remote_code=True)
                    english = [c for c in configs if "en" in c.lower()]
                    chosen = english[0] if english else configs[0]
                    kwargs["name"] = chosen
                    ds = load_dataset(hf_name, split=split, **kwargs)
                except Exception as inner:
                    raise RuntimeError(f"Failed to load {hf_name}: {inner}") from inner
            else:
                raise RuntimeError(f"Failed to load {hf_name}: {exc}") from exc

        for i, example in enumerate(ds):
            if self.max_samples > 0 and i >= self.max_samples:
                break
            yield example

    def _write_chunks(self, chunks: List[ChunkRecord], out_dir: Path, source_name: str, ext: str) -> List[Path]:
        """Write chunks to a single file and return paths."""
        if not chunks:
            return []

        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{source_name}_v1.{ext}"
        out_path = out_dir / filename

        written_paths: List[Path] = []
        if ext == "json":
            with out_path.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps({
                        "id": chunk.chunk_id,
                        "hash": chunk.hash,
                        "source": chunk.source,
                        "category": chunk.category,
                        "text": chunk.content,
                    }, ensure_ascii=False) + "\n")
            written_paths.append(out_path)
        else:
            with out_path.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(f"--- {chunk.chunk_id} ---\n")
                    handle.write(chunk.content)
                    handle.write("\n\n")
            written_paths.append(out_path)

        return written_paths

    def _generate_meta(self, category_dir: Path, files: List[Path], stats: IngestionStats) -> None:
        """Generate meta.json for a category."""
        meta = {
            "category": category_dir.name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": [f.name for f in files],
            "stats": stats.snapshot(),
        }
        meta_path = category_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

    def _generate_readme(self, category_dir: Path, description: str, files: List[Path]) -> None:
        """Generate README.md for a category."""
        readme = category_dir / "README.md"
        file_list = "\n".join(f"- `{f.name}`" for f in files) if files else "- No files generated yet."
        with readme.open("w", encoding="utf-8") as handle:
            handle.write(f"# {category_dir.name}\n\n")
            handle.write(f"{description}\n\n")
            handle.write(f"## Files\n{file_list}\n")

    def _dedupe(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        """Remove duplicate chunks by hash."""
        unique: List[ChunkRecord] = []
        for chunk in chunks:
            if chunk.hash in self.seen_hashes:
                continue
            self.seen_hashes.add(chunk.hash)
            unique.append(chunk)
        return unique

    def ingest_source(self, category: str, source_cfg: Dict[str, Any], description: str) -> Tuple[Path, IngestionStats]:
        """Ingest a single source using a streaming chunker."""
        out_dir = self.knowledge_root / category
        out_dir.mkdir(parents=True, exist_ok=True)

        source_name = source_cfg["name"]
        ext = source_cfg.get("format", "txt")
        stats = IngestionStats()
        all_chunks: List[ChunkRecord] = []
        files: List[Path] = []

        chunk_cfg = self.config.get("chunk", {})
        chunker = Chunker(
            min_bytes=int(chunk_cfg.get("min_bytes", 4096)),
            max_bytes=int(chunk_cfg.get("max_bytes", 8192)),
            source=source_name,
            category=category,
        )

        try:
            for example in self._load_dataset(source_cfg):
                text = self._extract_text(example, source_cfg)
                text = self.cleaner.clean(text)
                if not text:
                    continue
                emitted = chunker.feed(text)
                all_chunks.extend(emitted)
                stats.examples_read += 1

            # Emit final tail
            all_chunks.extend(chunker.flush())

            deduped = self._dedupe(all_chunks)
            stats.duplicates_removed = len(all_chunks) - len(deduped)

            files = self._write_chunks(deduped, out_dir, source_name, ext)
            if files:
                stats.chunks_written = len(deduped)
                stats.bytes_written = sum(f.stat().st_size for f in files)

        except Exception as exc:
            stats.errors.append(f"{source_name}: {exc}")

        self._generate_meta(out_dir, files, stats)
        self._generate_readme(out_dir, description, files)
        return out_dir, stats

    def run(self, category_filter: Optional[str] = None, source_filter: Optional[str] = None) -> Dict[str, IngestionStats]:
        """Run ingestion for all or selected categories/sources."""
        results: Dict[str, IngestionStats] = {}
        categories = self.config.get("categories", {})

        for category, cfg in categories.items():
            if category_filter and category != category_filter:
                continue
            description = cfg.get("description", "")
            for source_cfg in cfg.get("sources", []):
                if source_filter and source_cfg["name"] != source_filter:
                    continue
                print(f"[ingestion] starting {category}/{source_cfg['name']}", flush=True)
                _, stats = self.ingest_source(category, source_cfg, description)
                print(f"[ingestion] finished {category}/{source_cfg['name']}: read={stats.examples_read}, chunks={stats.chunks_written}, bytes={stats.bytes_written}, errors={len(stats.errors)}", flush=True)
                for err in stats.errors:
                    print(f"  ERROR: {err}", flush=True)
                results[f"{category}/{source_cfg['name']}"] = stats

        return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Condrox Knowledge Ingestion Engine")
    parser.add_argument("--config", default="config/knowledge_sources.json", help="Path to JSON config")
    parser.add_argument("--root", default=None, help="Override knowledge root")
    parser.add_argument("--category", default=None, help="Ingest a single category")
    parser.add_argument("--source", default=None, help="Ingest a single source")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per source (0 = unlimited)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path

    root = Path(args.root) if args.root else None
    engine = KnowledgeIngestionEngine(config_path, root)
    if args.max_samples:
        engine.max_samples = args.max_samples

    results = engine.run(category_filter=args.category, source_filter=args.source)

    print("\n=== Ingestion Summary ===")
    for key, stats in results.items():
        print(f"{key}: read={stats.examples_read}, chunks={stats.chunks_written}, bytes={stats.bytes_written}, errors={len(stats.errors)}")
        for err in stats.errors:
            print(f"  ERROR: {err}")


if __name__ == "__main__":
    main()
