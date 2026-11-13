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
Condrox Official Dataset Downloader & Processor.

Downloads, extracts, cleans, chunks, deduplicates, and stores all configured
datasets from their official sources (no Hugging Face dependency). Runs
downloads in parallel via asyncio/aiohttp and processes each file as it
arrives.

Config: config/official_dataset_sources.json
Output: <knowledge_root>/<category>/<filename>_NNNN.<ext>

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import bz2
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import aiofiles
import aiohttp
import zstandard as zstd

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 encoding crashes
# when printing text containing non-ASCII characters (e.g. C4 web text).
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Text cleaning and chunking
# ---------------------------------------------------------------------------

class TextCleaner:
    """Real text normalization and cleaning."""

    __slots__ = ("_html_tag", "_control")

    def __init__(self) -> None:
        self._html_tag = re.compile(r"<[^>]+>")
        self._control = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")

    def clean(self, text: str) -> str:
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


@dataclass
class ChunkRecord:
    content: str
    source: str
    chunk_id: str
    hash: str


class StreamingChunker:
    """Streaming chunker producing 4–8 KB UTF-8 blocks."""

    __slots__ = ("min_bytes", "max_bytes", "buffer", "buffer_len", "index", "source")

    def __init__(self, min_bytes: int, max_bytes: int, source: str) -> None:
        self.min_bytes = min_bytes
        self.max_bytes = max_bytes
        self.buffer: List[str] = []
        self.buffer_len = 0
        self.index = 0
        self.source = source

    def feed(self, text: str) -> List[ChunkRecord]:
        if not text:
            return []
        emitted: List[ChunkRecord] = []
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            para_bytes = len(paragraph.encode("utf-8"))
            if self.buffer_len + para_bytes >= self.max_bytes:
                if self.buffer:
                    content = "\n".join(self.buffer)
                    if self._size(content) >= self.min_bytes:
                        emitted.append(self._record(content))
                    self.buffer = []
                    self.buffer_len = 0
            self.buffer.append(paragraph)
            self.buffer_len += para_bytes + 1
            if self.buffer_len >= self.max_bytes:
                content = "\n".join(self.buffer)
                if self._size(content) >= self.min_bytes:
                    emitted.append(self._record(content))
                    self.buffer = []
                    self.buffer_len = 0
        return emitted

    def flush(self) -> List[ChunkRecord]:
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
        return ChunkRecord(content=content, source=self.source, chunk_id=chunk_id, hash=digest)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

class ChunkWriter:
    """Writes chunks to rotating files (max_file_size_mb)."""

    __slots__ = ("out_dir", "base_name", "ext", "max_file_bytes", "current_file", "current_path", "current_size", "file_index", "files", "chunks_written", "bytes_written")

    def __init__(self, out_dir: Path, base_name: str, ext: str, max_file_size_mb: int) -> None:
        self.out_dir = out_dir
        self.base_name = base_name
        self.ext = ext
        self.max_file_bytes = max_file_size_mb * 1024 * 1024
        self.current_file = None
        self.current_path: Optional[Path] = None
        self.current_size = 0
        self.file_index = 0
        self.files: List[Path] = []
        self.chunks_written = 0
        self.bytes_written = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _open_new(self) -> None:
        if self.current_file:
            self.current_file.close()
        suffix = f"_{self.file_index:04d}" if self.max_file_bytes > 0 else ""
        filename = f"{self.base_name}{suffix}.{self.ext}"
        self.current_path = self.out_dir / filename
        self.current_file = self.current_path.open("w", encoding="utf-8")
        self.files.append(self.current_path)
        self.current_size = 0
        self.file_index += 1

    def write(self, chunk: ChunkRecord) -> None:
        if self.current_file is None or self.current_size >= self.max_file_bytes:
            self._open_new()
        if self.ext == "json":
            line = json.dumps({
                "id": chunk.chunk_id,
                "hash": chunk.hash,
                "source": chunk.source,
                "text": chunk.content,
            }, ensure_ascii=False) + "\n"
        else:
            line = f"--- {chunk.chunk_id} ---\n{chunk.content}\n\n"
        data = line.encode("utf-8")
        self.current_file.write(line)
        self.current_size += len(data)
        self.chunks_written += 1
        self.bytes_written += len(data)

    def close(self) -> None:
        if self.current_file:
            self.current_file.close()
            self.current_file = None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

async def download_file(session: aiohttp.ClientSession, url: str, dest: Path, semaphore: asyncio.Semaphore) -> bool:
    """Download a single file with streaming. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    async with semaphore:
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=300)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    print(f"  [download] HTTP {resp.status} for {url}", flush=True)
                    return False
                dest.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        await f.write(chunk)
            return True
        except Exception as exc:
            print(f"  [download] error {url}: {exc}", flush=True)
            if dest.exists():
                try:
                    dest.unlink()
                except Exception:
                    pass
            return False


def generate_urls(source: Dict[str, Any]) -> List[str]:
    """Generate URLs from pattern or return explicit list."""
    if "urls" in source:
        urls = source["urls"]
        if urls == "generated":
            pattern = source["url_pattern"]
            count = int(source.get("count", 0))
            return [pattern.format(index=i) for i in range(count)]
        if isinstance(urls, str):
            return [urls]
        if isinstance(urls, list):
            return urls
    if "url" in source:
        return [source["url"]]
    return []


# ---------------------------------------------------------------------------
# Text extraction from various formats
# ---------------------------------------------------------------------------

def _extract_jsonl(handle: Iterable[str], text_fields: List[str]) -> Iterable[str]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        parts = []
        for f in text_fields:
            v = obj.get(f)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        for sub in ("text", "label", "value"):
                            if sub in item and isinstance(item[sub], str):
                                parts.append(item[sub])
        if parts:
            yield "\n\n".join(parts)


def _extract_json(handle: Iterable[str], text_fields: List[str]) -> Iterable[str]:
    data = handle.read() if hasattr(handle, "read") else "".join(handle)
    try:
        obj = json.loads(data)
    except Exception:
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                parts = [str(item.get(f, "")) for f in text_fields if item.get(f)]
                if parts:
                    yield "\n\n".join(parts)
    elif isinstance(obj, dict):
        parts = [str(obj.get(f, "")) for f in text_fields if obj.get(f)]
        if parts:
            yield "\n\n".join(parts)


def _extract_csv(handle: Iterable[str], text_fields: List[str]) -> Iterable[str]:
    reader = csv.reader(handle)
    for row in reader:
        if not row:
            continue
        # Use all columns if no specific field index; ConceptNet rows are wide
        yield " | ".join(row)


def _extract_txt(handle: Iterable[str], text_fields: List[str]) -> Iterable[str]:
    for line in handle:
        line = line.rstrip("\n")
        if line.strip():
            yield line


def _extract_xml_bz2(path: Path, text_fields: List[str]) -> Iterable[str]:
    """Extract text from MediaWiki XML dump (bz2)."""
    ns = "{http://www.mediawiki.org/xml/export-0.10/}"
    with bz2.open(path, "rb") as raw:
        for event, elem in ET.iterparse(raw, events=("end",)):
            if elem.tag == f"{ns}text":
                if elem.text:
                    yield elem.text
                elem.clear()


def _open_decompressed(path: Path, fmt: str) -> Tuple[Any, Iterable[str]]:
    """Open a file possibly compressed, return (handle, iterable of str lines)."""
    if fmt.endswith("_gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace"), None
    if fmt.endswith("_bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace"), None
    if fmt.endswith("_zst"):
        dctx = zstd.ZstdDecompressor()
        fh = path.open("rb")
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), None
    return path.open("rt", encoding="utf-8", errors="replace"), None


def extract_text_from_file(path: Path, fmt: str, text_fields: List[str]) -> Iterable[str]:
    """Yield cleaned text strings from a downloaded file."""
    if fmt == "skip":
        return
    if fmt == "nltk":
        # Handled separately
        return
    if fmt == "tar_gz":
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if not f:
                    continue
                text = f.read().decode("utf-8", errors="replace")
                if text.strip():
                    yield text
        return
    if fmt == "zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                try:
                    with zf.open(name) as f:
                        text = f.read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                if text.strip():
                    yield text
        return
    if fmt == "xml_bz2":
        yield from _extract_xml_bz2(path, text_fields)
        return

    base_fmt = fmt.rsplit("_", 1)[0] if "_" in fmt else fmt
    handle, _ = _open_decompressed(path, fmt)
    try:
        if base_fmt == "jsonl":
            yield from _extract_jsonl(handle, text_fields)
        elif base_fmt == "json":
            yield from _extract_json(handle, text_fields)
        elif base_fmt == "csv":
            yield from _extract_csv(handle, text_fields)
        else:
            yield from _extract_txt(handle, text_fields)
    finally:
        try:
            handle.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-source processing
# ---------------------------------------------------------------------------

@dataclass
class SourceStats:
    name: str
    category: str
    files_downloaded: int = 0
    examples_read: int = 0
    chunks_written: int = 0
    bytes_written: int = 0
    duplicates_removed: int = 0
    errors: List[str] = field(default_factory=list)
    skipped: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "files_downloaded": self.files_downloaded,
            "examples_read": self.examples_read,
            "chunks_written": self.chunks_written,
            "bytes_written": self.bytes_written,
            "duplicates_removed": self.duplicates_removed,
            "errors": self.errors[:20],
            "skipped": self.skipped,
        }


async def process_source(
    source: Dict[str, Any],
    knowledge_root: Path,
    downloads_dir: Path,
    chunk_cfg: Dict[str, Any],
    max_parallel: int,
    session: aiohttp.ClientSession,
    seen_hashes: Set[str],
) -> SourceStats:
    name = source["name"]
    category = source["category"]
    fmt = source.get("format", "txt")
    text_fields = source.get("text_fields", ["text"])
    base_name = source.get("filename", f"{category}_{name}_v1")
    ext = "json" if source.get("format", "").startswith("json") else "txt"
    max_file_mb = int(source.get("max_file_size_mb", 1024))
    stats = SourceStats(name=name, category=category)

    if fmt == "skip":
        stats.skipped = True
        print(f"[{name}] skipped (format=skip)", flush=True)
        return stats

    if source.get("local_path") == "" and "url" not in source and "urls" not in source:
        stats.skipped = True
        print(f"[{name}] skipped (no URL/local path)", flush=True)
        return stats

    out_dir = knowledge_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    src_download_dir = downloads_dir / name
    src_download_dir.mkdir(parents=True, exist_ok=True)

    # NLTK special case
    if fmt == "nltk":
        try:
            import nltk
            nltk.download(source["nltk"], download_dir=str(downloads_dir / "nltk_data"))
            from nltk.corpus import wordnet as wn
            writer = ChunkWriter(out_dir, base_name, ext, max_file_mb)
            chunker = StreamingChunker(
                int(chunk_cfg.get("min_bytes", 4096)),
                int(chunk_cfg.get("max_bytes", 8192)),
                name,
            )
            cleaner = TextCleaner()
            for synset in wn.all_synsets():
                text = f"{synset.name()}: {synset.definition()}"
                for lemma in synset.lemmas():
                    text += f"\n{lemma.name()}"
                text = cleaner.clean(text)
                for chunk in chunker.feed(text):
                    if chunk.hash not in seen_hashes:
                        seen_hashes.add(chunk.hash)
                        writer.write(chunk)
                        stats.chunks_written += 1
                        stats.bytes_written += len(chunk.content.encode("utf-8"))
                    else:
                        stats.duplicates_removed += 1
                stats.examples_read += 1
            for chunk in chunker.flush():
                if chunk.hash not in seen_hashes:
                    seen_hashes.add(chunk.hash)
                    writer.write(chunk)
                    stats.chunks_written += 1
                    stats.bytes_written += len(chunk.content.encode("utf-8"))
            writer.close()
            _write_meta(out_dir, base_name, writer.files, stats)
            return stats
        except Exception as exc:
            stats.errors.append(str(exc))
            return stats

    # Download phase — batched to avoid filling disk on large sources (e.g. C4 1024 shards)
    urls = generate_urls(source)
    if not urls:
        stats.errors.append("no URLs generated")
        return stats

    # Resume support: load progress file to skip already-processed URLs
    progress_path = out_dir / f"{base_name}_progress.json"
    processed_urls: Set[str] = set()
    if progress_path.exists():
        try:
            with progress_path.open("r", encoding="utf-8") as pf:
                processed_urls = set(json.load(pf))
            if processed_urls:
                print(f"  [{name}] resuming — {len(processed_urls)} URLs already processed", flush=True)
        except Exception:
            pass

    semaphore = asyncio.Semaphore(max_parallel)
    download_tasks: List[Tuple[str, Path]] = []
    for i, url in enumerate(urls):
        # Skip already-processed URLs (resume support)
        if url in processed_urls:
            continue
        # Derive filename from URL or index
        url_path = url.split("/")[-1].split("?")[0]
        if not url_path:
            url_path = f"{name}_{i:05d}"
        dest = src_download_dir / url_path
        download_tasks.append((url, dest))

    if not download_tasks:
        print(f"  [{name}] all URLs already processed — skipping", flush=True)
        # Still write meta/readme
        writer = ChunkWriter(out_dir, base_name, ext, max_file_mb)
        writer.close()
        _write_meta(out_dir, base_name, writer.files, stats)
        _write_readme(out_dir, source.get("description", ""), writer.files)
        return stats

    # For sources with many files (e.g. C4 with 1024 shards), download and
    # process in batches to avoid filling the disk with raw downloads.
    # Each batch: download N files → process them → delete raw downloads → next batch.
    batch_size = max_parallel * 2 if len(download_tasks) > max_parallel * 4 else len(download_tasks)

    writer = ChunkWriter(out_dir, base_name, ext, max_file_mb)
    chunker = StreamingChunker(
        int(chunk_cfg.get("min_bytes", 4096)),
        int(chunk_cfg.get("max_bytes", 8192)),
        name,
    )
    cleaner = TextCleaner()

    for batch_start in range(0, len(download_tasks), batch_size):
        batch = download_tasks[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(download_tasks) + batch_size - 1) // batch_size
        print(f"  [{name}] batch {batch_num}/{total_batches} ({len(batch)} files)", flush=True)

        # Disk space check before downloading
        try:
            usage = shutil.disk_usage(src_download_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < 2.0:
                stats.errors.append(f"disk space low: {free_gb:.1f} GB free on {src_download_dir.anchor}")
                print(f"  [{name}] WARNING: only {free_gb:.1f} GB free — stopping", flush=True)
                break
        except Exception:
            pass  # Non-critical

        # Download this batch in parallel
        tasks = [download_file(session, url, dest, semaphore) for url, dest in batch]
        results = await asyncio.gather(*tasks)
        stats.files_downloaded += sum(1 for r in results if r)
        failed = [url for (url, _), r in zip(batch, results) if not r]
        if failed:
            stats.errors.append(f"failed downloads in batch {batch_num}: {len(failed)}")

        # Process this batch immediately
        for url, dest in batch:
            if not dest.exists():
                continue
            try:
                for raw_text in extract_text_from_file(dest, fmt, text_fields):
                    text = cleaner.clean(raw_text)
                    if not text:
                        continue
                    for chunk in chunker.feed(text):
                        if chunk.hash not in seen_hashes:
                            seen_hashes.add(chunk.hash)
                            writer.write(chunk)
                            stats.chunks_written += 1
                            stats.bytes_written += len(chunk.content.encode("utf-8"))
                        else:
                            stats.duplicates_removed += 1
                    stats.examples_read += 1
            except Exception as exc:
                stats.errors.append(f"{dest.name}: {exc}")

        # Delete raw downloads for this batch to free disk space
        for url, dest in batch:
            try:
                if dest.exists():
                    dest.unlink()
            except Exception:
                pass

        # Save progress for resume support
        for url, _ in batch:
            processed_urls.add(url)
        try:
            with progress_path.open("w", encoding="utf-8") as pf:
                json.dump(sorted(processed_urls), pf)
        except Exception:
            pass

        print(f"  [{name}] batch {batch_num} done: chunks={stats.chunks_written}, examples={stats.examples_read}", flush=True)

    for chunk in chunker.flush():
        if chunk.hash not in seen_hashes:
            seen_hashes.add(chunk.hash)
            writer.write(chunk)
            stats.chunks_written += 1
            stats.bytes_written += len(chunk.content.encode("utf-8"))

    writer.close()
    _write_meta(out_dir, base_name, writer.files, stats)
    _write_readme(out_dir, source.get("description", ""), writer.files)
    return stats


def _write_meta(out_dir: Path, base_name: str, files: List[Path], stats: SourceStats) -> None:
    meta = {
        "category": out_dir.name,
        "base_name": base_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [f.name for f in files],
        "stats": stats.snapshot(),
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def _write_readme(out_dir: Path, description: str, files: List[Path]) -> None:
    readme = out_dir / "README.md"
    file_list = "\n".join(f"- `{f.name}`" for f in files) if files else "- No files generated."
    with readme.open("w", encoding="utf-8") as handle:
        handle.write(f"# {out_dir.name}\n\n{description}\n\n## Files\n{file_list}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(config: Dict[str, Any], source_filter: Optional[str] = None, max_files: int = 0) -> Dict[str, SourceStats]:
    knowledge_root = Path(config["knowledge_root"])
    downloads_dir = Path(config["downloads_dir"])
    chunk_cfg = config.get("chunk", {})
    max_parallel = int(config.get("max_parallel_downloads", 8))
    sources = config.get("sources", [])

    knowledge_root.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    results: Dict[str, SourceStats] = {}

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=600)
    connector = aiohttp.TCPConnector(limit=max_parallel, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for source in sources:
            if source_filter and source["name"] != source_filter:
                continue
            if max_files and source.get("count", 0) > max_files:
                # For testing: limit generated URLs
                source = dict(source)
                source["count"] = max_files
            print(f"[ingestion] starting {source['name']} ({source['category']})", flush=True)
            stats = await process_source(
                source, knowledge_root, downloads_dir, chunk_cfg, max_parallel, session, seen_hashes
            )
            print(
                f"[ingestion] finished {source['name']}: files={stats.files_downloaded}, "
                f"examples={stats.examples_read}, chunks={stats.chunks_written}, "
                f"bytes={stats.bytes_written}, errors={len(stats.errors)}",
                flush=True,
            )
            for err in stats.errors:
                print(f"  ERROR: {err}", flush=True)
            results[source["name"]] = stats

    # Write global meta
    global_meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "knowledge_root": str(knowledge_root),
        "sources": {name: s.snapshot() for name, s in results.items()},
    }
    meta_path = knowledge_root / "meta" / "global_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(global_meta, handle, indent=2, ensure_ascii=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Condrox Official Dataset Downloader")
    parser.add_argument("--config", default="config/official_dataset_sources.json")
    parser.add_argument("--source", default=None, help="Process a single source")
    parser.add_argument("--max-files", type=int, default=0, help="Limit generated URL count (testing)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if args.max_files:
        config["max_parallel_downloads"] = min(config.get("max_parallel_downloads", 8), 4)

    results = asyncio.run(run_all(config, source_filter=args.source, max_files=args.max_files))

    print("\n=== Ingestion Summary ===")
    total_bytes = 0
    for name, stats in results.items():
        print(f"{name}: files={stats.files_downloaded}, chunks={stats.chunks_written}, bytes={stats.bytes_written}, errors={len(stats.errors)}")
        total_bytes += stats.bytes_written
    print(f"\nTotal bytes written: {total_bytes / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
