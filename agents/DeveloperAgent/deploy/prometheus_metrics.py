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
Prometheus Metrics for DeveloperAgent — monitoring and observability.

Metrics exposed:
- condrox_executions_total — Total executions by intent and state
- condrox_execution_duration_seconds — Execution latency
- condrox_rag_index_size — Current RAG index size
- condrox_gate_results — Gate results by gate name and status
- condrox_cache_hits_total — Cache hit/miss counts
- condrox_agent_state — Current agent state

Complexity:
    record_execution: T(n) in O(1), S(n) in O(1).
    record_gate: T(n) in O(1), S(n) in O(1).

Invariants:
    - All metrics are thread-safe.
    - Metrics are exposed at /metrics endpoint.

Edge Cases:
    - No executions -> counters at zero.
    - Missing labels -> default values used.

Classification: Tier 5+ Class A

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ─── Metrics Registry ───────────────────────────────────────────────────────────


if PROMETHEUS_AVAILABLE:
    _REGISTRY = CollectorRegistry()
    
    # Execution metrics
    _executions_total = Counter(
        "condrox_executions_total",
        "Total code task executions",
        ["intent", "state", "domain"],
        registry=_REGISTRY,
    )
    
    _execution_duration = Histogram(
        "condrox_execution_duration_seconds",
        "Execution latency in seconds",
        ["intent", "domain"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
        registry=_REGISTRY,
    )
    
    # RAG metrics
    _rag_index_size = Gauge(
        "condrox_rag_index_size",
        "Current RAG index size (number of chunks)",
        registry=_REGISTRY,
    )
    
    _rag_cache_hits = Counter(
        "condrox_cache_hits_total",
        "RAG cache hit/miss counts",
        ["type"],  # hit or miss
        registry=_REGISTRY,
    )
    
    # Gate metrics
    _gate_results = Counter(
        "condrox_gate_results_total",
        "Quality gate results",
        ["gate_name", "status"],  # status: pass, fail, blocked
        registry=_REGISTRY,
    )
    
    # Agent state
    _agent_state = Gauge(
        "condrox_agent_state",
        "Current agent state (0=IDLE, 1=PLANNING, 2=RETRIEVING, 3=GENERATING, 4=VERIFYING, 5=APPLYING, 6=COMPLETED, 7=BLOCKED, 8=FAILED)",
        registry=_REGISTRY,
    )
    
    # Build info
    _build_info = Info(
        "condrox_build_info",
        "DeveloperAgent build information",
        registry=_REGISTRY,
    )
    _build_info.info({
        "version": "2.0.0",
        "tier": "Class A",
    })


# ─── Metrics Recorder ─────────────────────────────────────────────────────────


class MetricsRecorder:
    """
    Records Prometheus metrics for DeveloperAgent operations.
    
    Complexity:
        record_execution: T(n) in O(1), S(n) in O(1).
        record_gate: T(n) in O(1), S(n) in O(1).
    Invariants:
        - Thread-safe metric updates.
        - No-op if prometheus_client not available.
    Edge Cases:
        - prometheus_client not available -> no-ops.
    """
    
    __slots__ = ()
    
    @staticmethod
    def record_execution(
        intent: str,
        state: str,
        domain: str,
        duration_seconds: float,
    ) -> None:
        """Record an execution with latency."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        _executions_total.labels(
            intent=intent,
            state=state,
            domain=domain,
        ).inc()
        
        _execution_duration.labels(
            intent=intent,
            domain=domain,
        ).observe(duration_seconds)
    
    @staticmethod
    def update_rag_index_size(size: int) -> None:
        """Update RAG index size gauge."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        _rag_index_size.set(size)
    
    @staticmethod
    def record_cache_hit(hit: bool) -> None:
        """Record a cache hit or miss."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        cache_type = "hit" if hit else "miss"
        _rag_cache_hits.labels(type=cache_type).inc()
    
    @staticmethod
    def record_gate_result(gate_name: str, passed: bool, blocked: bool) -> None:
        """Record a gate result."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if blocked:
            status = "blocked"
        elif passed:
            status = "pass"
        else:
            status = "fail"
        
        _gate_results.labels(
            gate_name=gate_name,
            status=status,
        ).inc()
    
    @staticmethod
    def update_agent_state(state: int) -> None:
        """Update agent state gauge."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        _agent_state.set(state)
    
    @staticmethod
    def generate_metrics() -> str:
        """Generate Prometheus metrics text format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus metrics not available\n"
        
        from prometheus_client import generate_latest
        return generate_latest(_REGISTRY).decode("utf-8")


# ─── FastAPI Integration ───────────────────────────────────────────────────────


def add_metrics_endpoint(app) -> None:
    """Add /metrics endpoint to FastAPI app."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    from fastapi import Response
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=MetricsRecorder.generate_metrics(),
            media_type="text/plain",
        )
