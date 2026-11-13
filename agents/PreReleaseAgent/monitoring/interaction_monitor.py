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

"""Monitoring system for Condrox AI Pre-Release.

Logs every user interaction to a JSONL file for later analysis:
- Query text, intent, domain, mode, emotion, confidence
- Response preview, truncation status, errors
- Timestamp, session ID hash, response time
- Security events (injection attempts, rate limit hits, bans)

Also provides aggregate statistics via the stats() method.

Complexity:
    log_interaction: T(n) ∈ O(1) amortized, S(n) ∈ O(1)
    stats: T(n) ∈ O(n) for n logged entries, S(n) ∈ O(domains + intents)

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True, slots=True)
class InteractionLog:
    """Single interaction log entry.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    timestamp: float
    session_hash: str
    message: str
    message_length: int
    intent: str
    domain: str
    mode: str
    emotion: str
    confidence: float
    response_length: int
    response_preview: str
    truncated: bool
    errors: List[str]
    response_time_ms: float
    status: str  # "ok", "error", "rejected"


class InteractionMonitor:
    """Thread-safe interaction monitor that logs to JSONL and tracks stats.

    Complexity:
        log_interaction: T(n) ∈ O(1) amortized, S(n) ∈ O(1)
        stats: T(n) ∈ O(n) for n entries, S(n) ∈ O(d + i) for d domains, i intents
    """

    __slots__ = (
        "_log_path", "_lock", "_total", "_errors", "_rejected",
        "_intent_counts", "_domain_counts", "_mode_counts",
        "_emotion_counts", "_status_counts", "_response_times",
        "_recent_messages", "_max_recent",
        "_live_feed", "_max_live", "_unique_ips",
    )

    def __init__(self, log_dir: str = "monitoring_logs") -> None:
        """Initialize monitor with log directory.

        Args:
            log_dir: Directory for JSONL log files. Created if not exists.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._log_path = Path(log_dir)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._total = 0
        self._errors = 0
        self._rejected = 0
        self._intent_counts: Counter = Counter()
        self._domain_counts: Counter = Counter()
        self._mode_counts: Counter = Counter()
        self._emotion_counts: Counter = Counter()
        self._status_counts: Counter = Counter()
        self._response_times: List[float] = []
        self._recent_messages: List[Dict[str, Any]] = []
        self._max_recent = 100
        self._live_feed: List[Dict[str, Any]] = []
        self._max_live = 50
        self._unique_ips: set = set()

    def _hash_session(self, session_id: Optional[str]) -> str:
        """Hash session ID for privacy — no raw session IDs in logs.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not session_id:
            return "anonymous"
        return hashlib.sha256(session_id.encode()).hexdigest()[:12]

    def log_interaction(
        self,
        message: str,
        session_id: Optional[str],
        intent: str,
        domain: str,
        mode: str,
        emotion: str,
        confidence: float,
        response: str,
        truncated: bool,
        errors: List[str],
        response_time_ms: float,
        status: str = "ok",
        client_ip: str = "",
        client_country: str = "",
        client_city: str = "",
    ) -> None:
        """Log a single interaction to JSONL file and update stats.

        Args:
            message: User's input message.
            session_id: Session ID (will be hashed).
            intent: Classified intent.
            domain: Routed domain.
            mode: Selected mode.
            emotion: Detected emotion.
            confidence: Intent confidence.
            response: AI response text.
            truncated: Whether response was truncated.
            errors: List of non-fatal errors.
            response_time_ms: Response time in milliseconds.
            status: "ok", "error", or "rejected".
            client_ip: Client IP address (hashed in logs).
            client_country: Client country (from Cloudflare header).
            client_city: Client city (from Cloudflare header).

        Complexity: T(n) ∈ O(1) amortized, S(n) ∈ O(1).
        """
        ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:12] if client_ip else ""
        entry = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "session": self._hash_session(session_id),
            "msg": message[:200],
            "msg_len": len(message),
            "intent": intent,
            "domain": domain,
            "mode": mode,
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "resp_len": len(response),
            "resp_preview": response[:150],
            "truncated": truncated,
            "errors": errors[:3],
            "rt_ms": round(response_time_ms, 1),
            "status": status,
            "ip_hash": ip_hash,
            "country": client_country,
            "city": client_city,
        }

        with self._lock:
            self._total += 1
            self._status_counts[status] += 1
            self._intent_counts[intent] += 1
            self._domain_counts[domain] += 1
            self._mode_counts[mode] += 1
            self._emotion_counts[emotion] += 1
            self._response_times.append(response_time_ms)
            if len(self._response_times) > 1000:
                self._response_times = self._response_times[-500:]
            if status == "error":
                self._errors += 1
            elif status == "rejected":
                self._rejected += 1

            self._recent_messages.append(entry)
            if len(self._recent_messages) > self._max_recent:
                self._recent_messages = self._recent_messages[-self._max_recent:]

            # Live feed entry (includes location for dashboard)
            live_entry = {
                "ts": entry["ts"],
                "ts_iso": entry["ts_iso"],
                "session": entry["session"],
                "msg_preview": message[:80],
                "intent": intent,
                "domain": domain,
                "mode": mode,
                "emotion": emotion,
                "status": status,
                "rt_ms": entry["rt_ms"],
                "ip_hash": ip_hash,
                "country": client_country,
                "city": client_city,
            }
            self._live_feed.append(live_entry)
            if len(self._live_feed) > self._max_live:
                self._live_feed = self._live_feed[-self._max_live:]
            if client_ip:
                self._unique_ips.add(ip_hash)

            # Append to JSONL file (one file per day)
            date_str = time.strftime("%Y-%m-%d", time.gmtime())
            log_file = self._log_path / f"interactions_{date_str}.jsonl"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass  # Don't let logging break the request

    def log_security_event(
        self,
        event_type: str,
        client_ip: str,
        detail: str,
    ) -> None:
        """Log a security event (injection attempt, rate limit, ban).

        Args:
            event_type: "injection", "rate_limit", "ban", "oversized".
            client_ip: Client IP address.
            detail: Event detail string.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        entry = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "type": "security",
            "event": event_type,
            "ip_hash": hashlib.sha256(client_ip.encode()).hexdigest()[:12],
            "detail": detail[:200],
        }

        with self._lock:
            date_str = time.strftime("%Y-%m-%d", time.gmtime())
            log_file = self._log_path / f"security_{date_str}.jsonl"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        """Get aggregate statistics.

        Returns:
            Dict with total, errors, intent/domain/mode/emotion distributions,
            response time stats, and recent messages.

        Complexity: T(n) ∈ O(n) for n response times, S(n) ∈ O(d + i).
        """
        with self._lock:
            rts = self._response_times
            avg_rt = sum(rts) / len(rts) if rts else 0.0
            max_rt = max(rts) if rts else 0.0
            min_rt = min(rts) if rts else 0.0
            p95_idx = int(len(rts) * 0.95) if rts else 0
            sorted_rts = sorted(rts) if rts else []
            p95_rt = sorted_rts[p95_idx - 1] if p95_idx > 0 and sorted_rts else 0.0

            return {
                "total_requests": self._total,
                "total_errors": self._errors,
                "total_rejected": self._rejected,
                "error_rate": round(self._errors / max(self._total, 1), 4),
                "intent_distribution": dict(self._intent_counts.most_common()),
                "domain_distribution": dict(self._domain_counts.most_common()),
                "mode_distribution": dict(self._mode_counts.most_common()),
                "emotion_distribution": dict(self._emotion_counts.most_common()),
                "status_distribution": dict(self._status_counts.most_common()),
                "response_time_ms": {
                    "avg": round(avg_rt, 1),
                    "min": round(min_rt, 1),
                    "max": round(max_rt, 1),
                    "p95": round(p95_rt, 1),
                },
                "recent_messages": list(self._recent_messages[-20:]),
                "unique_visitors": len(self._unique_ips),
            }

    def live_feed(self) -> Dict[str, Any]:
        """Get live traffic feed for dashboard.

        Returns recent interactions with location data, unique visitor count,
        and country distribution.

        Complexity: T(n) in O(n) for n live entries, S(n) in O(n).
        """
        with self._lock:
            feed = list(reversed(self._live_feed[-30:]))
            country_counts: Counter = Counter()
            for entry in self._live_feed:
                if entry.get("country"):
                    country_counts[entry["country"]] += 1

            return {
                "total_requests": self._total,
                "unique_visitors": len(self._unique_ips),
                "active_sessions": len({e["session"] for e in self._live_feed if e["session"] != "anonymous"}),
                "feed": feed,
                "country_distribution": dict(country_counts.most_common(10)),
                "server_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            }
