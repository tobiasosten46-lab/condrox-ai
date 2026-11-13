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
Repair Learner — learns from past repairs and successes.

Tracks repair attempts (fix type, success/failure, before/after code)
and provides statistics and pattern recommendations for future repairs.

Complexity:
    record: T(n) in O(1), S(n) in O(1).
    get_stats: T(n) in O(1), S(n) in O(1).
    get_success_rate: T(n) in O(1), S(n) in O(1).

Invariants:
    - All entries are immutable.
    - Success rate is [0.0, 1.0].
    - Entries sorted by timestamp (newest first).

Edge Cases:
    - No entries -> success_rate=0.0, empty stats.
    - All failures -> success_rate=0.0.
    - All successes -> success_rate=1.0.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ─── Immutable Data Containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LearningEntry:
    """
    Immutable learning entry from a repair attempt.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: entry_id is unique; timestamp is monotonic.
    Edge Cases: error_message may be empty on success.
    """

    entry_id: str
    fix_type: str
    rule_id: str
    success: bool
    timestamp: float
    code_hash_before: str
    code_hash_after: str
    error_message: str
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class LearningStats:
    """
    Immutable aggregate learning statistics.

    Complexity: T(n) in O(1), S(n) in O(1).
    Invariants: success_rate in [0.0, 1.0].
    Edge Cases: no entries -> all zero.
    """

    total_entries: int
    successful: int
    failed: int
    success_rate: float
    avg_elapsed_ms: float
    most_common_fix: str
    fix_type_counts: Dict[str, int]


# ─── Repair Learner ──────────────────────────────────────────────────────────


class RepairLearner:
    """
    Learns from past repair attempts to improve future repairs.

    Tracks every repair attempt with its outcome and provides:
    - Success rate per fix type
    - Most common successful fixes
    - Average repair time
    - Pattern recommendations

    Complexity:
        record: T(n) in O(1), S(n) in O(1).
        get_stats: T(n) in O(e) where e = entries, S(n) in O(1).
        get_success_rate: T(n) in O(1), S(n) in O(1).
    Invariants:
        - All entries immutable.
        - Bounded history (LRU eviction at max_entries).
    Edge Cases:
        - No entries -> empty stats, 0.0 success rate.
    """

    __slots__ = ("_entries", "_max_entries", "_fix_counts", "_success_counts")

    def __init__(self, max_entries: int = 1000) -> None:
        """
        Initialize RepairLearner with bounded history.

        Complexity: T(n) in O(1), S(n) in O(max_entries).
        Edge Cases: max_entries=0 -> no learning.
        """
        self._entries: OrderedDict[str, LearningEntry] = OrderedDict()
        self._max_entries: int = max_entries
        self._fix_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        """
        Number of recorded entries.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty -> 0.
        """
        return len(self._entries)

    def record(
        self,
        fix_type: str,
        rule_id: str,
        success: bool,
        code_before: str,
        code_after: str,
        error_message: str = "",
        elapsed_ms: int = 0,
    ) -> str:
        """
        Record a repair attempt.

        Complexity: T(n) in O(1), S(n) in O(1).
        Invariants: returns unique entry_id.
        Edge Cases: max_entries=0 -> no recording, returns empty string.
        """
        if self._max_entries == 0:
            return ""

        entry_id: str = hashlib.sha256(
            f"{fix_type}:{rule_id}:{time.time()}".encode()
        ).hexdigest()[:16]

        hash_before: str = hashlib.sha256(
            code_before.encode()
        ).hexdigest()[:16]
        hash_after: str = hashlib.sha256(
            code_after.encode()
        ).hexdigest()[:16]

        entry: LearningEntry = LearningEntry(
            entry_id=entry_id,
            fix_type=fix_type,
            rule_id=rule_id,
            success=success,
            timestamp=time.time(),
            code_hash_before=hash_before,
            code_hash_after=hash_after,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
        )

        self._entries[entry_id] = entry
        self._entries.move_to_end(entry_id)

        self._fix_counts[fix_type] = self._fix_counts.get(fix_type, 0) + 1
        if success:
            self._success_counts[fix_type] = (
                self._success_counts.get(fix_type, 0) + 1
            )

        while len(self._entries) > self._max_entries:
            oldest_id, oldest = self._entries.popitem(last=False)
            self._fix_counts[oldest.fix_type] = max(
                0, self._fix_counts.get(oldest.fix_type, 0) - 1
            )
            if oldest.success:
                self._success_counts[oldest.fix_type] = max(
                    0, self._success_counts.get(oldest.fix_type, 0) - 1
                )

        return entry_id

    def get_success_rate(self, fix_type: str = "") -> float:
        """
        Get success rate for a fix type (or overall if empty).

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: no entries -> 0.0.
        """
        if fix_type:
            total: int = self._fix_counts.get(fix_type, 0)
            successes: int = self._success_counts.get(fix_type, 0)
            return successes / total if total > 0 else 0.0

        total_all: int = len(self._entries)
        if total_all == 0:
            return 0.0
        successes_all: int = sum(1 for e in self._entries.values() if e.success)
        return successes_all / total_all

    def get_stats(self) -> LearningStats:
        """
        Get aggregate learning statistics.

        Complexity: T(n) in O(e) where e = entries, S(n) in O(1).
        Edge Cases: no entries -> all zero.
        """
        total: int = len(self._entries)
        if total == 0:
            return LearningStats(
                total_entries=0,
                successful=0,
                failed=0,
                success_rate=0.0,
                avg_elapsed_ms=0.0,
                most_common_fix="",
                fix_type_counts={},
            )

        successful: int = sum(1 for e in self._entries.values() if e.success)
        failed: int = total - successful
        success_rate: float = successful / total
        avg_elapsed: float = sum(e.elapsed_ms for e in self._entries.values()) / total

        most_common: str = max(
            self._fix_counts.items(), key=lambda x: x[1]
        )[0] if self._fix_counts else ""

        return LearningStats(
            total_entries=total,
            successful=successful,
            failed=failed,
            success_rate=success_rate,
            avg_elapsed_ms=avg_elapsed,
            most_common_fix=most_common,
            fix_type_counts=dict(self._fix_counts),
        )

    def get_best_fix_for(self, rule_id: str) -> Optional[str]:
        """
        Get the most successful fix type for a given rule.

        Complexity: T(n) in O(e) where e = entries, S(n) in O(1).
        Edge Cases: no entries for rule -> None.
        """
        fix_success: Dict[str, int] = {}
        fix_total: Dict[str, int] = {}

        for entry in self._entries.values():
            if entry.rule_id != rule_id:
                continue
            fix_total[entry.fix_type] = fix_total.get(entry.fix_type, 0) + 1
            if entry.success:
                fix_success[entry.fix_type] = fix_success.get(entry.fix_type, 0) + 1

        if not fix_total:
            return None

        best_fix: str = ""
        best_rate: float = -1.0
        for fix_type, total in fix_total.items():
            rate: float = fix_success.get(fix_type, 0) / total
            if rate > best_rate:
                best_rate = rate
                best_fix = fix_type

        return best_fix if best_rate > 0 else None

    def get_recent_entries(
        self, limit: int = 10
    ) -> Tuple[LearningEntry, ...]:
        """
        Get most recent learning entries.

        Complexity: T(n) in O(limit), S(n) in O(limit).
        Edge Cases: fewer entries than limit -> all returned.
        """
        entries: List[LearningEntry] = list(reversed(self._entries.values()))
        return tuple(entries[:limit])

    def clear(self) -> int:
        """
        Clear all learning entries.

        Complexity: T(n) in O(1), S(n) in O(1).
        Edge Cases: empty -> returns 0.
        """
        count: int = len(self._entries)
        self._entries.clear()
        self._fix_counts.clear()
        self._success_counts.clear()
        return count
