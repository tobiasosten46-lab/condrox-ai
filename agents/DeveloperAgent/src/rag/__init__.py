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
DeveloperAgent RAG subpackage — upgraded retrieval pipeline.

Hybrid retrieval (BM25 + keyword), LRU caching, context builder with
token budget, file watcher, multi-language chunker.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from agents.DeveloperAgent.src.rag.hybrid_retriever import (
    HybridRetriever,
    HybridResult,
)
from agents.DeveloperAgent.src.rag.rag_cache import RAGCache, CacheEntry
from agents.DeveloperAgent.src.rag.context_builder import (
    ContextBuilder,
    ContextResult,
)
from agents.DeveloperAgent.src.rag.file_watcher import FileWatcher, WatchEvent
from agents.DeveloperAgent.src.rag.multi_language_chunker import (
    MultiLanguageChunker,
    LanguageType,
)

__all__ = [
    "CacheEntry",
    "ContextBuilder",
    "ContextResult",
    "FileWatcher",
    "HybridRetriever",
    "HybridResult",
    "LanguageType",
    "MultiLanguageChunker",
    "RAGCache",
    "WatchEvent",
]
