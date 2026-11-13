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

"""Monitoring tracker for live traffic and aggregate stats.

Tracks request metadata (intent, domain, mode, emotion, response time,
status, errors) in a ring buffer for the dashboard's Pre-Release Monitor
and Live Traffic pages.

Complexity:
    record: T(n) ∈ O(1), S(n) ∈ O(k) for k = max_entries.
    get_live: T(n) ∈ O(k), S(n) ∈ O(k).
    get_stats: T(n) ∈ O(k), S(n) ∈ O(e) for e = distinct distributions.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class RequestRecord:
    """A single request record for monitoring.

    Attributes:
        ts: Unix timestamp (seconds).
        ts_iso: ISO 8601 timestamp string.
        session: Session identifier (or 'anonymous').
        msg_preview: First 80 chars of user input.
        msg_len: Full input length.
        intent: Classified intent.
        domain: Knowledge domain.
        mode: Processing mode.
        emotion: Detected emotion.
        confidence: Intent confidence (0.0–1.0).
        resp_len: Response length.
        resp_preview: First 120 chars of response.
        truncated: Whether response preview was truncated.
        errors: List of error strings.
        rt_ms: Response time in milliseconds.
        status: 'ok', 'error', or 'rejected'.
        ip_hash: Hashed IP (first 8 chars of SHA-256).
        country: Geo country (placeholder).
        city: Geo city (placeholder).
    """
    ts: float
    ts_iso: str
    session: str
    msg_preview: str
    msg_len: int
    intent: str
    domain: str
    mode: str
    emotion: str
    confidence: float
    resp_len: int
    resp_preview: str
    truncated: bool
    errors: List[str]
    rt_ms: float
    status: str
    ip_hash: str
    country: str
    city: str


class MonitoringTracker:
    """Thread-safe request tracker with ring buffer storage.

    Complexity:
        record: T(n) ∈ O(1), S(n) ∈ O(k).
        get_live: T(n) ∈ O(k), S(n) ∈ O(k).
        get_stats: T(n) ∈ O(k), S(n) ∈ O(e).
    """

    __slots__ = ("_lock", "_records", "_max_entries", "_total_requests",
                 "_total_errors", "_total_rejected", "_unique_sessions",
                 "_rt_history")

    def __init__(self, max_entries: int = 500) -> None:
        """Initialize the tracker.

        Args:
            max_entries: Maximum number of records to keep in the ring buffer.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(max_entries).
        """
        self._lock = threading.Lock()
        self._records: deque[RequestRecord] = deque(maxlen=max_entries)
        self._max_entries = max_entries
        self._total_requests = 0
        self._total_errors = 0
        self._total_rejected = 0
        self._unique_sessions: set[str] = set()
        self._rt_history: deque[float] = deque(maxlen=200)

    def record(
        self,
        session: str,
        msg: str,
        intent: str,
        domain: str,
        mode: str,
        emotion: str,
        confidence: float,
        resp: str,
        errors: List[str],
        rt_ms: float,
        status: str = "ok",
        ip_hash: str = "unknown",
        country: str = "—",
        city: str = "—",
    ) -> None:
        """Record a request.

        Args:
            session: Session identifier.
            msg: User input text.
            intent: Classified intent.
            domain: Knowledge domain.
            mode: Processing mode.
            emotion: Detected emotion.
            confidence: Intent confidence.
            resp: Response text.
            errors: List of errors.
            rt_ms: Response time in milliseconds.
            status: 'ok', 'error', or 'rejected'.
            ip_hash: Hashed IP.
            country: Geo country.
            city: Geo city.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1) amortized.
        """
        now = time.time()
        ts_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        msg_preview = msg[:80] if len(msg) > 80 else msg
        resp_preview = resp[:120] if len(resp) > 120 else resp

        record = RequestRecord(
            ts=now,
            ts_iso=ts_iso,
            session=session or "anonymous",
            msg_preview=msg_preview,
            msg_len=len(msg),
            intent=intent or "unknown",
            domain=domain or "general",
            mode=mode or "default",
            emotion=emotion or "neutral",
            confidence=confidence,
            resp_len=len(resp),
            resp_preview=resp_preview,
            truncated=len(resp) > 120,
            errors=errors or [],
            rt_ms=rt_ms,
            status=status,
            ip_hash=ip_hash,
            country=country,
            city=city,
        )

        with self._lock:
            self._records.append(record)
            self._total_requests += 1
            if status == "error":
                self._total_errors += 1
            elif status == "rejected":
                self._total_rejected += 1
            self._unique_sessions.add(session or "anonymous")
            self._rt_history.append(rt_ms)

    def get_live(self) -> Dict[str, Any]:
        """Get live traffic data (recent feed + summary).

        Returns:
            Dict with total_requests, unique_visitors, active_sessions,
            feed (list of recent entries), country_distribution, server_time.

        Complexity: T(n) ∈ O(k) for k records, S(n) ∈ O(k).
        """
        with self._lock:
            records = list(self._records)
            total = self._total_requests
            unique = len(self._unique_sessions)
            # Active sessions = sessions with activity in last 60s
            now = time.time()
            active = set()
            for r in records:
                if now - r.ts < 60:
                    active.add(r.session)
            country_dist: Counter = Counter()
            for r in records[-50:]:
                country_dist[r.country] += 1

            feed = [
                {
                    "ts": r.ts,
                    "ts_iso": r.ts_iso,
                    "session": r.session,
                    "msg_preview": r.msg_preview,
                    "intent": r.intent,
                    "domain": r.domain,
                    "mode": r.mode,
                    "emotion": r.emotion,
                    "status": r.status,
                    "rt_ms": round(r.rt_ms, 1),
                    "ip_hash": r.ip_hash,
                    "country": r.country,
                    "city": r.city,
                }
                for r in records[-30:]
            ]

        return {
            "total_requests": total,
            "unique_visitors": unique,
            "active_sessions": len(active),
            "feed": feed,
            "country_distribution": dict(country_dist),
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate monitoring stats.

        Returns:
            Dict with total counts, error rate, distributions, response
            time stats, and recent messages.

        Complexity: T(n) ∈ O(k) for k records, S(n) ∈ O(e).
        """
        with self._lock:
            records = list(self._records)
            total = self._total_requests
            errors = self._total_errors
            rejected = self._total_rejected
            rt_list = list(self._rt_history)

        intent_dist: Counter = Counter(r.intent for r in records)
        domain_dist: Counter = Counter(r.domain for r in records)
        mode_dist: Counter = Counter(r.mode for r in records)
        emotion_dist: Counter = Counter(r.emotion for r in records)
        status_dist: Counter = Counter(r.status for r in records)

        if rt_list:
            rt_sorted = sorted(rt_list)
            n = len(rt_sorted)
            avg = sum(rt_sorted) / n
            p95_idx = min(int(n * 0.95), n - 1)
            rt_stats = {
                "avg": round(avg, 1),
                "min": round(rt_sorted[0], 1),
                "max": round(rt_sorted[-1], 1),
                "p95": round(rt_sorted[p95_idx], 1),
            }
        else:
            rt_stats = {"avg": 0, "min": 0, "max": 0, "p95": 0}

        recent = [
            {
                "ts": r.ts,
                "ts_iso": r.ts_iso,
                "session": r.session,
                "msg": r.msg_preview,
                "msg_len": r.msg_len,
                "intent": r.intent,
                "domain": r.domain,
                "mode": r.mode,
                "emotion": r.emotion,
                "confidence": r.confidence,
                "resp_len": r.resp_len,
                "resp_preview": r.resp_preview,
                "truncated": r.truncated,
                "errors": r.errors,
                "rt_ms": round(r.rt_ms, 1),
                "status": r.status,
            }
            for r in records[-20:]
        ]

        error_rate = round(errors / total, 4) if total > 0 else 0.0

        return {
            "total_requests": total,
            "total_errors": errors,
            "total_rejected": rejected,
            "error_rate": error_rate,
            "intent_distribution": dict(intent_dist),
            "domain_distribution": dict(domain_dist),
            "mode_distribution": dict(mode_dist),
            "emotion_distribution": dict(emotion_dist),
            "status_distribution": dict(status_dist),
            "response_time_ms": rt_stats,
            "recent_messages": recent,
        }


# Global singleton
_tracker: Optional[MonitoringTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> MonitoringTracker:
    """Get the global MonitoringTracker singleton.

    Returns:
        The shared MonitoringTracker instance.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = MonitoringTracker(max_entries=500)
    return _tracker
