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
Knowledge Provenance — tracks the source, ingestion time, and checksum for
every knowledge chunk.

Provides an audit trail from a `KnowledgeItem` back to the original file on
disk and a content hash for integrity checks.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Union


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """
    Immutable provenance metadata for a single knowledge chunk.

    Attributes:
        chunk_id: The KnowledgeItem.item_id this record describes.
        source_path: Absolute path to the source file on disk.
        source_name: File name component of source_path.
        ingested_at: ISO-8601 UTC timestamp of ingestion.
        checksum: SHA-256 hex digest of the raw source chunk.

    Complexity: S(n) ∈ O(1).
    """

    chunk_id: str
    source_path: Path
    source_name: str
    ingested_at: str
    checksum: str


class KnowledgeProvenance:
    """
    In-memory provenance registry for the knowledge backend.

    Records are keyed by chunk_id and can be queried directly or iterated.
    The registry is intentionally independent of the knowledge content so it
    can be loaded/saved separately.

    Attributes:
        _records: Mapping from chunk_id to ProvenanceRecord.

    Complexity:
        register: T(n) ∈ O(1), S(n) ∈ O(1).
        get: T(n) ∈ O(1), S(n) ∈ O(1).
        records: T(n) ∈ O(r), S(n) ∈ O(1) for the iterator.
    """

    __slots__ = ("_records",)

    def __init__(self) -> None:
        """Initialize an empty provenance registry.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._records: Dict[str, ProvenanceRecord] = {}

    @staticmethod
    def _now() -> str:
        """Return the current UTC timestamp as an ISO-8601 string.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return datetime.now(timezone.utc).isoformat()

    def register(
        self,
        chunk_id: str,
        source_path: Union[str, Path],
        checksum: str,
        ingested_at: Optional[str] = None,
    ) -> ProvenanceRecord:
        """Register or overwrite provenance for a chunk.

        Args:
            chunk_id: The knowledge item identifier.
            source_path: Path to the source file.
            checksum: SHA-256 hex digest of the raw chunk.
            ingested_at: Optional ISO-8601 timestamp; defaults to now.

        Returns:
            The created ProvenanceRecord.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        path = Path(source_path)
        record = ProvenanceRecord(
            chunk_id=chunk_id,
            source_path=path,
            source_name=path.name,
            ingested_at=ingested_at or self._now(),
            checksum=checksum,
        )
        self._records[chunk_id] = record
        return record

    def get(self, chunk_id: str) -> Optional[ProvenanceRecord]:
        """Retrieve the provenance record for a chunk.

        Args:
            chunk_id: Knowledge item identifier.

        Returns:
            The ProvenanceRecord or None if not registered.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return self._records.get(chunk_id)

    def records(self) -> Iterable[ProvenanceRecord]:
        """Iterate over all registered provenance records.

        Returns:
            Iterable of ProvenanceRecord instances.

        Complexity: T(n) ∈ O(1) for the iterator, O(r) to traverse r records.
        """
        return self._records.values()

    def count(self) -> int:
        """Return the number of registered provenance records.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        return len(self._records)
